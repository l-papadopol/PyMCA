#!/usr/bin/env python3
# ----------------------------------------
# basic MCA made from a PicoScope 2204A
# Papadopol Lucian-Ioan l.i.papadopol@gmail.com
# All rights reserved (C) 2025
#-----------------------------------------


#-----------------------------------------
# Imports and Constants
# ----------------------------------------
import ctypes
import time
import sys

import numpy as np
from picosdk.ps2000 import ps2000 as ps
from picosdk.functions import assert_pico2000_ok, adc2mV
from picosdk.PicoDeviceEnums import picoEnum
from picosdk.errors import PicoSDKCtypesError

from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg


def format_time_div_label(t_div):
    """
    Convert time-per-division in seconds to human-readable string in s, ms, µs, or ns.
    """
    if t_div >= 1.0:
        return f"{t_div:.2f} s/div"
    elif t_div >= 1e-3:
        return f"{t_div * 1e3:.2f} ms/div"
    elif t_div >= 1e-6:
        return f"{t_div * 1e6:.2f} µs/div"
    else:
        return f"{t_div * 1e9:.2f} ns/div"

# ----------------------------------------
# PicoScope Wrapper Class
# ----------------------------------------
class SimplePicoScope2000:
    """Minimal PS2000 wrapper for single-channel A with hardware trigger."""
    AVAILABLE_RANGES = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    MAX_ADC = 32767

    def __init__(self):
        status = ps.ps2000_open_unit()
        assert_pico2000_ok(status)
        self.handle = ctypes.c_int16(status)
        self.oversample = 1
        self._range_enum = ps.PS2000_VOLTAGE_RANGE['PS2000_1V']
        self._range_v = 1.0
        self._time_interval = None
        self._time_units = None
        self.set_channel('A', True, 1.0, 'DC')

    def close(self):
        ps.ps2000_close_unit(self.handle)

    def set_channel(self, channel, enable, range_v, coupling):
        ch = picoEnum.PICO_CHANNEL[f'PICO_CHANNEL_{channel}']
        coup = picoEnum.PICO_COUPLING[f'PICO_{coupling}']
        rng_map = {
            0.05: 'PS2000_50MV', 0.1: 'PS2000_100MV', 0.2: 'PS2000_200MV',
            0.5: 'PS2000_500MV', 1.0: 'PS2000_1V',   2.0: 'PS2000_2V',
            5.0: 'PS2000_5V',   10.0:'PS2000_10V',  20.0:'PS2000_20V'
        }
        rng = ps.PS2000_VOLTAGE_RANGE[rng_map[range_v]]
        status = ps.ps2000_set_channel(self.handle, ch, int(enable), coup, rng)
        assert_pico2000_ok(status)
        self._range_enum = rng
        self._range_v = range_v

    def set_trigger(self, threshold, direction, delay=0, auto_ms=0):
        ch = picoEnum.PICO_CHANNEL['PICO_CHANNEL_A']
        cnt = int((threshold / self._range_v) * self.MAX_ADC)
        dir_map = {'RISING': 0, 'FALLING': 1}
        status = ps.ps2000_set_trigger(self.handle, ch, cnt, dir_map[direction], delay, auto_ms)
        assert_pico2000_ok(status)

    def _set_timebase(self, tb, n):
        ti, tu, ms = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
        status = ps.ps2000_get_timebase(self.handle, tb, n,
                                         ctypes.byref(ti), ctypes.byref(tu),
                                         self.oversample, ctypes.byref(ms))
        assert_pico2000_ok(status)
        self._time_interval = ti.value
        self._time_units = tu.value

    def run_block(self, n, tb):
        self._set_timebase(tb, n)
        td = ctypes.c_int32()
        status = ps.ps2000_run_block(self.handle, n, tb, self.oversample, ctypes.byref(td))
        assert_pico2000_ok(status)
        while ps.ps2000_ready(self.handle) == 0:
            time.sleep(0.001)

        buf = (ctypes.c_int16 * n)()
        ov = ctypes.c_int16()
        status = ps.ps2000_get_times_and_values(self.handle,
                                                 None, ctypes.byref(buf),
                                                 None, None, None,
                                                 ctypes.byref(ov),
                                                 self._time_units,
                                                 ctypes.c_int32(n))
        assert_pico2000_ok(status)

        t = np.linspace(0, (n - 1) * self._time_interval, n) * 1e-9
        mv = adc2mV(buf, self._range_enum, ctypes.c_int16(self.MAX_ADC))
        v = np.array(mv, dtype=float) * 1e-3
        return t, v

# ----------------------------------------
# Acquisition Thread for Oscilloscope
# ----------------------------------------
class AcqThread(QtCore.QThread):
    data_ready = QtCore.pyqtSignal(object, object)  # t, v

    def __init__(self, scope, parent=None):
        super().__init__(parent)
        self.scope = scope
        self._run = True

    def run(self):
        while self._run:
            win = self.parent()
            rng = win.range_cb.currentData()
            coup = win.coupling_cb.currentText()
            ns = win.ns_sb.value()
            tb = win.tb_cb.currentData()

            self.scope.set_channel('A', True, rng, coup)
            if win.trig_enable_cb.isChecked() and win.trig_level_sb.value() > 0:
                self.scope.set_trigger(win.trig_level_sb.value(), win.trig_dir_cb.currentText(), delay=0, auto_ms=0)
            else:
                self.scope.set_trigger(0.0, 'RISING', delay=0, auto_ms=1)

            t, v = self.scope.run_block(ns, tb)
            self.data_ready.emit(t, v)
            time.sleep(0.02)

    def stop(self):
        self._run = False
        self.wait()

# ----------------------------------------
# MCA Thread for Spectrum Analysis
# ----------------------------------------
class McaThread(QtCore.QThread):
    bin_ready = QtCore.pyqtSignal(int)
    waveform_ready = QtCore.pyqtSignal(object, object)
    rate_update = QtCore.pyqtSignal(int, float)

    def __init__(self, scope, parent=None):
        super().__init__(parent)
        self.scope = scope
        self._run = True

    def run(self):
        self.start_time = time.time()
        self.total_count = 0
        while self._run:
            win = self.parent()
            rng = win.mca_range_cb.currentData()
            lld = win.mca_lld_sb.value()
            pol = win.mca_pol_cb.currentText()
            direc = 'RISING' if pol == 'Positive' else 'FALLING'
            tb = win.mca_tb_cb.currentData()
            ns = win.mca_ns_sb.value()

            self.scope.oversample = 1
            self.scope.set_channel('A', True, rng, 'AC')
            self.scope.set_trigger(lld, direc, delay=0, auto_ms=0)

            t, v = self.scope.run_block(ns, tb)
            self.waveform_ready.emit(t, v)

            peak = v.max() if pol == 'Positive' else -v.min()
            idx = int((peak / rng) * 255)
            idx = min(max(idx, 0), 255)

            self.total_count += 1
            elapsed = time.time() - self.start_time
            rate = self.total_count / elapsed if elapsed > 0 else 0.0

            self.bin_ready.emit(idx)
            self.rate_update.emit(self.total_count, rate)

    def stop(self):
        self._run = False
        self.wait()

# ----------------------------------------
# Main Window and GUI Layout
# ----------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.scope = SimplePicoScope2000()

        self.setWindowTitle("PicoScope 2000 GUI")
        self.resize(1100, 600)

        # Menu setup
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        fm.addAction("&Open...", self.on_open)
        fm.addAction("&Save...", self.on_save)
        fm.addSeparator()
        fm.addAction("E&xit", self.close)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # Oscilloscope Tab
        osc = QtWidgets.QWidget()
        tabs.addTab(osc, "Oscilloscope")
        lo = QtWidgets.QHBoxLayout(osc)
        form = QtWidgets.QFormLayout()

        # Range selection
        self.range_cb = QtWidgets.QComboBox()
        for r in SimplePicoScope2000.AVAILABLE_RANGES:
            lbl = f"±{int(r*1000)} mV" if r < 1 else f"±{int(r)} V"
            self.range_cb.addItem(lbl, r)
        self.range_cb.setCurrentIndex(4)
        form.addRow("Full-Scale Range:", self.range_cb)

        # Coupling selection
        self.coupling_cb = QtWidgets.QComboBox()
        self.coupling_cb.addItems(["DC", "AC"])
        form.addRow("Coupling:", self.coupling_cb)

        # Time/div combo
        self.tb_cb = QtWidgets.QComboBox()
        sample_test = 1000
        for tb_idx in range(26):
            try:
                ti, tu, ms = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
                status = ps.ps2000_get_timebase(
                    self.scope.handle, tb_idx, sample_test,
                    ctypes.byref(ti), ctypes.byref(tu),
                    self.scope.oversample, ctypes.byref(ms)
                )
                assert_pico2000_ok(status)
                dt = ti.value * 1e-9
                t_tot = dt * (sample_test - 1)
                t_div = t_tot / 10
                label = format_time_div_label(t_div)
                self.tb_cb.addItem(label, tb_idx)
            except (AssertionError, PicoSDKCtypesError):
                pass
        form.addRow("Time/div:", self.tb_cb)

        # Samples selection
        self.ns_sb = QtWidgets.QSpinBox()
        self.ns_sb.setRange(100, 1_000_000)
        self.ns_sb.setValue(1000)
        form.addRow("Samples N:", self.ns_sb)

        # Trigger Controls
        self.trig_enable_cb = QtWidgets.QCheckBox("Enable HW Trigger")
        form.addRow(self.trig_enable_cb)
        self.trig_level_sb = QtWidgets.QDoubleSpinBox()
        self.trig_level_sb.setRange(-20, 20); self.trig_level_sb.setDecimals(3)
        form.addRow("Threshold (V):", self.trig_level_sb)
        self.trig_dir_cb = QtWidgets.QComboBox()
        self.trig_dir_cb.addItems(["RISING", "FALLING"])
        form.addRow("Direction:", self.trig_dir_cb)

        # Noise measurement
        self.noise_btn = QtWidgets.QPushButton("Measure Noise")
        self.noise_label = QtWidgets.QLabel("Noise σ: — mV")
        nh = QtWidgets.QHBoxLayout()
        nh.addWidget(self.noise_btn); nh.addWidget(self.noise_label)
        form.addRow(nh)
        self.noise_btn.clicked.connect(self.measure_noise)

        # Start/Stop Controls
        hb = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        hb.addWidget(self.start_btn); hb.addWidget(self.stop_btn)
        form.addRow(hb)

        lo.addLayout(form)
        # Oscilloscope Plot Area
        self.plot = pg.PlotWidget(title="Channel A")
        self.plot.setLabel('bottom','Time',units='s')
        self.plot.setLabel('left','Voltage',units='V')
        self.curve = self.plot.plot(pen='y')
        self.plot.enableAutoRange(False)
        lo.addWidget(self.plot,1)

        # MCA Tab
        mca = QtWidgets.QWidget()
        tabs.addTab(mca, "MCA")
        vlay = QtWidgets.QVBoxLayout(mca)

        top_h = QtWidgets.QHBoxLayout()
        mca_form = QtWidgets.QFormLayout()

        # MCA Controls
        self.mca_range_cb = QtWidgets.QComboBox()
        for r in SimplePicoScope2000.AVAILABLE_RANGES:
            lbl = f"±{int(r*1000)} mV" if r<1 else f"±{int(r)} V"
            self.mca_range_cb.addItem(lbl, r)
        mca_form.addRow("Full-Scale Range:", self.mca_range_cb)

        self.mca_coupling_cb = QtWidgets.QComboBox()
        self.mca_coupling_cb.addItems(["DC","AC"])
        mca_form.addRow("Coupling:", self.mca_coupling_cb)

        self.mca_lld_sb = QtWidgets.QDoubleSpinBox()
        self.mca_lld_sb.setRange(-20,20); self.mca_lld_sb.setDecimals(3)
        mca_form.addRow("LLD (V):", self.mca_lld_sb)

        self.mca_pol_cb = QtWidgets.QComboBox()
        self.mca_pol_cb.addItems(["Positive","Negative"])
        mca_form.addRow("Polarity:", self.mca_pol_cb)

        # MCA Time/div
        self.mca_tb_cb = QtWidgets.QComboBox()
        for i in range(self.tb_cb.count()):
            label = self.tb_cb.itemText(i)
            self.mca_tb_cb.addItem(label, self.tb_cb.itemData(i))
        mca_form.addRow("Time/div:", self.mca_tb_cb)

        self.mca_ns_sb = QtWidgets.QSpinBox()
        self.mca_ns_sb.setRange(100,1_000_000); self.mca_ns_sb.setValue(1000)
        mca_form.addRow("Samples N:", self.mca_ns_sb)

        self.display_mode_cb = QtWidgets.QComboBox()
        self.display_mode_cb.addItems(["Bar","Line"])
        mca_form.addRow("Display Mode:", self.display_mode_cb)

        hb2 = QtWidgets.QHBoxLayout()
        self.start_mca_btn = QtWidgets.QPushButton("Start MCA")
        self.pause_mca_btn = QtWidgets.QPushButton("Pause MCA"); self.pause_mca_btn.setEnabled(False)
        self.stop_mca_btn = QtWidgets.QPushButton("Stop MCA"); self.stop_mca_btn.setEnabled(False)
        hb2.addWidget(self.start_mca_btn); hb2.addWidget(self.pause_mca_btn); hb2.addWidget(self.stop_mca_btn)
        mca_form.addRow(hb2)

        top_h.addLayout(mca_form)

        # Pulse Plot
        self.pulse_plot_small = pg.PlotWidget(title="Last Pulse")
        self.pulse_plot_small.setFixedSize(400,400)
        self.pulse_plot_small.setLabel('bottom','Time',units='s')
        self.pulse_plot_small.setLabel('left','Voltage',units='V')
        self.pulse_curve_small = self.pulse_plot_small.plot(pen='c')
        top_h.addWidget(self.pulse_plot_small)
        vlay.addLayout(top_h)

        # Histogram & Counts
        self.hist_plot = pg.PlotWidget(title="MCA Spectrum")
        self.hist_plot.setLabel('bottom','Channel (bin)')
        self.hist_plot.setLabel('left','Counts')
        self.hist = np.zeros(256, dtype=np.uint32)
        self.bins = np.arange(256)
        self.hist_item_bar = pg.BarGraphItem(x=self.bins, height=self.hist, width=0.8)
        self.hist_plot.addItem(self.hist_item_bar)
        self.hist_curve_line = self.hist_plot.plot(self.bins, self.hist, pen='b', connect='all')
        self.hist_curve_line.setVisible(False)
        vlay.addWidget(self.hist_plot,1)

        cnt_h = QtWidgets.QHBoxLayout()
        self.total_label = QtWidgets.QLabel("Total events: 0")
        self.rate_label = QtWidgets.QLabel("Rate: 0.0 evt/s")
        self.elapsed_label = QtWidgets.QLabel("Elapsed: 0.0 s")
        cnt_h.addWidget(self.total_label); cnt_h.addWidget(self.rate_label); cnt_h.addWidget(self.elapsed_label); cnt_h.addStretch()
        vlay.addLayout(cnt_h)

        # Signal Connections
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.noise_btn.clicked.connect(self.measure_noise)
        self.start_mca_btn.clicked.connect(self.start_mca)
        self.pause_mca_btn.clicked.connect(self.pause_mca)
        self.stop_mca_btn.clicked.connect(self.stop_mca)
        self.display_mode_cb.currentIndexChanged.connect(self.toggle_display_mode)

        # Threads and State
        self.acq_thread = None
        self.mca_thread = None
        self.mca_paused = False
        self.mca_start_time = 0.0
        self.mca_elapsed_at_pause = 0.0
        self.mca_timer = QtCore.QTimer(self)
        self.mca_timer.setInterval(200)
        self.mca_timer.timeout.connect(self.update_elapsed)
        self.grid_items = []

    def on_open(self):
        QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "", "*.*")

    def on_save(self):
        QtWidgets.QFileDialog.getSaveFileName(self, "Save File", "", "*.*")

    def measure_noise(self):
        self.noise_btn.setEnabled(False)
        try:
            rng = self.range_cb.currentData()
            coup = self.coupling_cb.currentText()
            ns = self.ns_sb.value()
            tb = self.tb_cb.currentData()

            self.scope.set_channel('A', True, rng, coup)
            self.scope.set_trigger(0.0, 'RISING', delay=0, auto_ms=1)
            _, v = self.scope.run_block(ns, tb)
            sigma = np.std(v) * 1e3
            self.noise_label.setText(f"Noise σ: {sigma:.2f} mV")
        finally:
            self.noise_btn.setEnabled(True)

    def start(self):
        self.acq_thread = AcqThread(self.scope, parent=self)
        self.acq_thread.data_ready.connect(self.update_osc)
        self.acq_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop(self):
        if self.acq_thread:
            self.acq_thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def start_mca(self):
        self.hist[:] = 0
        self.hist_item_bar.setOpts(height=self.hist)
        self.hist_curve_line.setData(self.bins, self.hist, connect='all')
        self.total_label.setText("Total events: 0")
        self.rate_label.setText("Rate: 0.0 evt/s")
        self.mca_elapsed_at_pause = 0.0
        self.mca_start_time = time.time()

        self.mca_thread = McaThread(self.scope, parent=self)
        self.mca_thread.waveform_ready.connect(self.update_pulse_small)
        self.mca_thread.bin_ready.connect(self.update_mca_bin)
        self.mca_thread.rate_update.connect(self.update_mca_rate)
        self.mca_thread.start()
        self.mca_timer.start()
        self.mca_paused = False
        self.start_mca_btn.setEnabled(False)
        self.pause_mca_btn.setEnabled(True)
        self.stop_mca_btn.setEnabled(True)

    def pause_mca(self):
        if self.mca_thread:
            self.mca_thread.stop()
        self.mca_elapsed_at_pause = time.time() - self.mca_start_time
        self.mca_timer.stop()
        self.mca_paused = True
        self.start_mca_btn.setEnabled(True)
        self.pause_mca_btn.setEnabled(False)

    def stop_mca(self):
        if self.mca_thread:
            self.mca_thread.stop()
        self.mca_timer.stop()
        self.mca_paused = False
        self.mca_elapsed_at_pause = 0.0
        self.update_elapsed()
        self.start_mca_btn.setEnabled(True)
        self.pause_mca_btn.setEnabled(False)
        self.stop_mca_btn.setEnabled(False)

    def toggle_display_mode(self):
        mode = self.display_mode_cb.currentText()
        self.hist_item_bar.setVisible(mode == "Bar")
        self.hist_curve_line.setVisible(mode == "Line")

    def update_elapsed(self):
        elapsed = (time.time() - self.mca_start_time) if not self.mca_paused else self.mca_elapsed_at_pause
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f} s")

    def draw_grid(self, t):
        for ln in self.grid_items:
            self.plot.removeItem(ln)
        self.grid_items.clear()
        tmax = t[-1]
        for i in range(11):
            x = i * tmax / 10
            ln = pg.InfiniteLine(x, angle=90, pen=pg.mkPen((150,150,150), style=QtCore.Qt.DotLine))
            ln.setZValue(-10); self.plot.addItem(ln); self.grid_items.append(ln)
        fs = self.range_cb.currentData()
        for i in range(11):
            y = -fs + i * (2 * fs / 10)
            ln = pg.InfiniteLine(y, angle=0, pen=pg.mkPen((150,150,150), style=QtCore.Qt.DotLine))
            ln.setZValue(-10); self.plot.addItem(ln); self.grid_items.append(ln)

    @QtCore.pyqtSlot(object, object)
    def update_osc(self, t, v):
        self.curve.setData(t, v)
        self.plot.setXRange(0, t[-1])
        fs = self.range_cb.currentData()
        self.plot.setYRange(-fs, fs)
        self.draw_grid(t)

    @QtCore.pyqtSlot(object, object)
    def update_pulse_small(self, t, v):
        self.pulse_curve_small.setData(t, v)

    @QtCore.pyqtSlot(int)
    def update_mca_bin(self, idx):
        self.hist[idx] += 1
        if self.display_mode_cb.currentText() == "Bar":
            self.hist_item_bar.setOpts(height=self.hist)
        else:
            self.hist_curve_line.setData(self.bins, self.hist, connect='all')
        ymax = self.hist.max()
        self.hist_plot.setYRange(0, ymax * 1.1 if ymax > 0 else 1)

    @QtCore.pyqtSlot(int, float)
    def update_mca_rate(self, total, rate):
        self.total_label.setText(f"Total events: {total}")
        self.rate_label.setText(f"Rate: {rate:.1f} evt/s")

    def closeEvent(self, ev):
        self.stop()
        self.pause_mca()
        self.scope.close()
        super().closeEvent(ev)

# ----------------------------------------
# Application Entry Point
# ----------------------------------------
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())

