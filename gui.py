#!/usr/bin/env python3
# -------------------------------------------------
# gui.py – PicoScope 2204A (Oscilloscopio + MCA)
# Oversample hardware fisso a 2
# -------------------------------------------------

import sys, time, numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from picosdk.errors import PicoSDKCtypesError

from scope   import SimplePicoScope2000
from threads import AcqThread, McaThread


# ---------- funzione di utilità per label Time/div -------------------
def fmt_time_div(t):
    if t >= 1:     return f"{t:.2f} s/div"
    if t >= 1e-3:  return f"{t*1e3:.2f} ms/div"
    if t >= 1e-6:  return f"{t*1e6:.2f} µs/div"
    return              f"{t*1e9:.2f} ns/div"


# =====================================================================
#                         Main Window
# =====================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # ----- apertura PicoScope -----
        try:
            self.scope = SimplePicoScope2000()
        except PicoSDKCtypesError as e:
            QtWidgets.QMessageBox.critical(self, "PicoScope error", str(e))
            sys.exit(1)

        self.setWindowTitle("PicoScope 2204A – Oscilloscope + MCA")
        self.resize(1100, 600)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # ---------------- Tab Oscilloscopio ----------------
        osc = QtWidgets.QWidget(); tabs.addTab(osc, "Oscilloscope")
        self.build_osc_tab(osc)

        # ---------------- Tab MCA ----------------
        mca = QtWidgets.QWidget(); tabs.addTab(mca, "MCA")
        self.build_mca_tab(mca)

        # ----- thread / stato MCA -----
        self.acq_thread = None
        self.mca_thread = None
        self.mca_paused = False
        self.mca_t0     = 0.0
        self.mca_pause_acc = 0.0
        self.mca_timer = QtCore.QTimer(self)
        self.mca_timer.setInterval(200)
        self.mca_timer.timeout.connect(self.update_elapsed)

    # ------------------------------------------------------------------
    #                 COSTRUISCE TAB OSCILLOSCOPIO
    # ------------------------------------------------------------------
    def build_osc_tab(self, parent):
        layout = QtWidgets.QHBoxLayout(parent)
        form   = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignRight)

        # ----- Range full-scale -----
        self.range_cb = QtWidgets.QComboBox()
        for r in SimplePicoScope2000.AVAILABLE_RANGES:
            self.range_cb.addItem(f"±{int(r*1000) if r<1 else int(r)} "
                                  f"{'mV' if r<1 else 'V'}", r)
        self.range_cb.setCurrentIndex(4)
        form.addRow("Full-Scale:", self.range_cb)

        # ----- Coupling -----
        self.coupling_cb = QtWidgets.QComboBox()
        self.coupling_cb.addItems(["DC", "AC"])
        form.addRow("Coupling:", self.coupling_cb)

        # ----- Time/div -----
        self.tb_cb = QtWidgets.QComboBox()
        sample = 1000
        for tb in range(26):
            try:
                ticks, unit = self.scope._timebase_info(tb, sample)
                dt = ticks * SimplePicoScope2000._UNIT_TO_SEC.get(unit, 1e-9)
                self.tb_cb.addItem(fmt_time_div(dt*(sample-1)/10), tb)
            except PicoSDKCtypesError:
                pass
        form.addRow("Time/div:", self.tb_cb)

        # ----- Samples -----
        self.ns_sb = QtWidgets.QSpinBox()
        self.ns_sb.setRange(100, 1_000_000)
        self.ns_sb.setValue(1000)
        form.addRow("Samples N:", self.ns_sb)

        # ----- Trigger -----
        self.trig_enable_cb = QtWidgets.QCheckBox("Enable HW Trigger")
        form.addRow(self.trig_enable_cb)
        self.trig_level_sb = QtWidgets.QDoubleSpinBox()
        self.trig_level_sb.setRange(-20, 20)
        self.trig_level_sb.setDecimals(3)
        form.addRow("Threshold (V):", self.trig_level_sb)
        self.trig_dir_cb = QtWidgets.QComboBox()
        self.trig_dir_cb.addItems(["RISING", "FALLING"])
        form.addRow("Direction:", self.trig_dir_cb)

        # ----- Noise button -----
        self.noise_btn   = QtWidgets.QPushButton("Measure Noise")
        self.noise_label = QtWidgets.QLabel("Noise σ: — mV")
        h_noise = QtWidgets.QHBoxLayout()
        h_noise.addWidget(self.noise_btn); h_noise.addWidget(self.noise_label)
        form.addRow(h_noise)

        # ----- Start / Stop -----
        h_ctrl = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn  = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        h_ctrl.addWidget(self.start_btn); h_ctrl.addWidget(self.stop_btn)
        form.addRow(h_ctrl)

        layout.addLayout(form)

        # ----- Plot Oscilloscopio -----
        self.plot = pg.PlotWidget(title="Channel A")
        self.plot.setLabel('bottom', 'Time', 's')
        self.plot.setLabel('left',   'Voltage', 'V')
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot(pen='y')
        self.plot.enableAutoRange(False)
        layout.addWidget(self.plot, 1)

        # --- connessioni Oscilloscopio ---
        self.noise_btn.clicked.connect(self.measure_noise)
        self.start_btn.clicked.connect(self.start_osc)
        self.stop_btn.clicked.connect(self.stop_osc)

    # ------------------------------------------------------------------
    #                    COSTRUISCE TAB MCA
    # ------------------------------------------------------------------
    def build_mca_tab(self, parent):
        outer = QtWidgets.QVBoxLayout(parent)

        # ---------- controlli in due colonne ----------
        top_ctrl = QtWidgets.QHBoxLayout()
        outer.addLayout(top_ctrl)

        def narrow(widget, w=150):
            widget.setMaximumWidth(w)
            return widget

        left  = QtWidgets.QFormLayout(); left.setLabelAlignment(QtCore.Qt.AlignRight)
        right = QtWidgets.QFormLayout(); right.setLabelAlignment(QtCore.Qt.AlignRight)

        # ---- Widgets sinistra ----
        self.mca_range_cb = narrow(QtWidgets.QComboBox())
        for r in SimplePicoScope2000.AVAILABLE_RANGES:
            self.mca_range_cb.addItem(f"±{int(r*1000) if r<1 else int(r)} "
                                      f"{'mV' if r<1 else 'V'}", r)
        self.mca_lld_sb = narrow(QtWidgets.QDoubleSpinBox()); self.mca_lld_sb.setRange(-20, 20); self.mca_lld_sb.setDecimals(3)
        self.mca_tb_cb  = narrow(QtWidgets.QComboBox())
        for i in range(self.tb_cb.count()):
            self.mca_tb_cb.addItem(self.tb_cb.itemText(i), self.tb_cb.itemData(i))
        self.gain_sb = narrow(QtWidgets.QDoubleSpinBox()); self.gain_sb.setRange(0.001, 1e4); self.gain_sb.setDecimals(3); self.gain_sb.setValue(1); self.gain_sb.setSuffix(" keV/ch")
        self.display_mode_cb = narrow(QtWidgets.QComboBox()); self.display_mode_cb.addItems(["Bar", "Line"])

        # ---- Widgets destra ----
        self.mca_coupling_cb = narrow(QtWidgets.QComboBox()); self.mca_coupling_cb.addItems(["DC", "AC"])
        self.mca_pol_cb      = narrow(QtWidgets.QComboBox()); self.mca_pol_cb.addItems(["Positive", "Negative"])
        self.mca_ns_sb       = narrow(QtWidgets.QSpinBox());  self.mca_ns_sb.setRange(100, 1_000_000); self.mca_ns_sb.setValue(1000)
        self.offset_sb       = narrow(QtWidgets.QDoubleSpinBox()); self.offset_sb.setRange(-1e4, 1e4); self.offset_sb.setDecimals(2); self.offset_sb.setSuffix(" keV")
        self.metric_cb       = narrow(QtWidgets.QComboBox()); self.metric_cb.addItems(["Peak", "Area"])
        self.baseline_cb     = QtWidgets.QCheckBox(); self.baseline_cb.setChecked(True)

        # ---- composizione colonne ----
        left .addRow("Full-Scale:", self.mca_range_cb)
        left .addRow("LLD (V):",    self.mca_lld_sb)
        left .addRow("Time/div:",   self.mca_tb_cb)
        left .addRow("Gain:",       self.gain_sb)
        left .addRow("Display:",    self.display_mode_cb)

        right.addRow("Coupling:",   self.mca_coupling_cb)
        right.addRow("Polarity:",   self.mca_pol_cb)
        right.addRow("Samples N:",  self.mca_ns_sb)
        right.addRow("Offset:",     self.offset_sb)
        right.addRow("Metric:",     self.metric_cb)
        right.addRow("Baseline:",   self.baseline_cb)

        top_ctrl.addLayout(left); top_ctrl.addLayout(right)

        # ---------- pulse preview ----------
        self.pulse_plot = pg.PlotWidget(title="Last Pulse")
        self.pulse_plot.setFixedSize(250, 180)
        self.pulse_plot.setLabel('bottom', 'Time', 's')
        self.pulse_plot.setLabel('left',   'Voltage', 'V')
        self.pulse_curve = self.pulse_plot.plot(pen='c')
        top_ctrl.addWidget(self.pulse_plot)

        # ---------- Start / Pause / Stop ----------
        h_btns = QtWidgets.QHBoxLayout()
        self.start_mca_btn = QtWidgets.QPushButton("Start MCA")
        self.pause_mca_btn = QtWidgets.QPushButton("Pause"); self.pause_mca_btn.setEnabled(False)
        self.stop_mca_btn  = QtWidgets.QPushButton("Stop");  self.stop_mca_btn.setEnabled(False)
        h_btns.addWidget(self.start_mca_btn); h_btns.addWidget(self.pause_mca_btn); h_btns.addWidget(self.stop_mca_btn)
        outer.addLayout(h_btns)

        # ---------- Histogram ----------
        self.hist_plot = pg.PlotWidget(title="MCA Spectrum")
        self.hist_plot.setLabel('bottom', 'Energy', 'keV')
        self.hist_plot.setLabel('left',   'Counts')
        self.hist   = np.zeros(256, dtype=np.uint32)
        self.bins   = np.arange(256)
        self.energy = self.bins
        self.bar_item  = pg.BarGraphItem(x=self.energy, height=self.hist, width=0.8)
        self.hist_plot.addItem(self.bar_item)
        self.line_item = self.hist_plot.plot(self.energy, self.hist, pen='b', connect='all')
        self.line_item.setVisible(False)
        outer.addWidget(self.hist_plot, 1)

        # ---------- contatori ----------
        h_cnt = QtWidgets.QHBoxLayout()
        self.total_label   = QtWidgets.QLabel("Total: 0")
        self.rate_label    = QtWidgets.QLabel("Rate: 0 evt/s")
        self.elapsed_label = QtWidgets.QLabel("Elapsed: 0 s")
        h_cnt.addWidget(self.total_label); h_cnt.addWidget(self.rate_label); h_cnt.addWidget(self.elapsed_label); h_cnt.addStretch()
        outer.addLayout(h_cnt)

        # --- connessioni MCA ---
        self.display_mode_cb.currentIndexChanged.connect(self.update_histogram)
        self.gain_sb.valueChanged.connect(self.update_histogram)
        self.offset_sb.valueChanged.connect(self.update_histogram)
        self.start_mca_btn.clicked.connect(self.start_mca)
        self.pause_mca_btn.clicked.connect(self.pause_mca)
        self.stop_mca_btn.clicked.connect(self.stop_mca)

    # ==================================================================
    #               FUNZIONI AUSILIARIE E SLOT
    # ==================================================================
    # ---------- misura rumore ----------
    def measure_noise(self):
        self.noise_btn.setEnabled(False)
        try:
            rng  = self.range_cb.currentData()
            coup = self.coupling_cb.currentText()
            self.scope.set_channel('A', True, rng, coup)
            self.scope.set_trigger(0, 'RISING', auto_ms=1)
            _, v = self.scope.run_block(self.ns_sb.value(), self.tb_cb.currentData())
            self.noise_label.setText(f"Noise σ: {np.std(v)*1e3:.2f} mV")
        finally:
            self.noise_btn.setEnabled(True)

    # ---------- start / stop Oscilloscopio ----------
    def start_osc(self):
        self.acq_thread = AcqThread(self.scope, self)
        self.acq_thread.data_ready.connect(self.update_osc)
        self.acq_thread.error_occurred.connect(self.thread_error)
        self.acq_thread.start()
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def stop_osc(self):
        if self.acq_thread:
            self.acq_thread.stop()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)

    @QtCore.pyqtSlot(object, object)
    def update_osc(self, t, v):
        self.curve.setData(t, v)
        self.plot.setXRange(0, t[-1])
        fs = self.range_cb.currentData()
        self.plot.setYRange(-fs, fs)

    # ---------- start / pause / stop MCA ----------
    def start_mca(self):
        self.hist.fill(0)
        self.update_energy(); self.update_histogram()
        self.total_label.setText("Total: 0"); self.rate_label.setText("Rate: 0 evt/s")
        self.mca_t0 = time.time(); self.mca_pause_acc = 0.0

        self.mca_thread = McaThread(self.scope, self)
        self.mca_thread.waveform_ready.connect(self.update_pulse)
        self.mca_thread.bin_ready.connect(self.add_bin)
        self.mca_thread.rate_update.connect(self.update_rate)
        self.mca_thread.error_occurred.connect(self.thread_error)
        self.mca_thread.start()
        self.mca_timer.start()
        self.mca_paused = False
        self.start_mca_btn.setEnabled(False); self.pause_mca_btn.setEnabled(True); self.stop_mca_btn.setEnabled(True)

    def pause_mca(self):
        if self.mca_thread:
            self.mca_thread.stop()
        self.mca_pause_acc = time.time() - self.mca_t0
        self.mca_paused = True
        self.mca_timer.stop()
        self.start_mca_btn.setEnabled(True); self.pause_mca_btn.setEnabled(False)

    def stop_mca(self):
        if self.mca_thread:
            self.mca_thread.stop()
        self.mca_timer.stop()
        self.mca_paused = False; self.mca_pause_acc = 0.0; self.update_elapsed()
        self.start_mca_btn.setEnabled(True); self.pause_mca_btn.setEnabled(False); self.stop_mca_btn.setEnabled(False)

    # ---------- istogramma ----------
    def update_energy(self):
        self.energy = self.bins * self.gain_sb.value() + self.offset_sb.value()

    def update_histogram(self):
        self.update_energy()
        if self.display_mode_cb.currentText() == "Bar":
            self.bar_item.setOpts(x=self.energy, height=self.hist, width=self.gain_sb.value()*0.8)
            self.bar_item.setVisible(True); self.line_item.setVisible(False)
        else:
            self.line_item.setData(self.energy, self.hist, connect='all')
            self.line_item.setVisible(True); self.bar_item.setVisible(False)
        ymax = self.hist.max()
        self.hist_plot.setYRange(0, ymax*1.1 if ymax else 1)

    # ---------- slot dai thread ----------
    @QtCore.pyqtSlot(object, object)
    def update_pulse(self, t, v):
        self.pulse_curve.setData(t, v)

    @QtCore.pyqtSlot(int)
    def add_bin(self, idx):
        self.hist[idx] += 1
        self.update_histogram()

    @QtCore.pyqtSlot(int, float)
    def update_rate(self, total, rate):
        self.total_label.setText(f"Total: {total}")
        self.rate_label.setText(f"Rate: {rate:.1f} evt/s")

    def update_elapsed(self):
        elapsed = (time.time() - self.mca_t0) if not self.mca_paused else self.mca_pause_acc
        self.elapsed_label.setText(f"Elapsed: {elapsed:.1f} s")

    # ---------- error handling ----------
    @QtCore.pyqtSlot(str)
    def thread_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "PicoScope error", msg)
        self.stop_osc(); self.stop_mca()
        self.start_btn.setEnabled(False); self.start_mca_btn.setEnabled(False)

    # ---------- cleanup ----------
    def closeEvent(self, e):
        self.stop_osc(); self.pause_mca(); self.scope.close()
        super().closeEvent(e)


# =====================================================================
# entry-point
# =====================================================================
if __name__ == "__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    mw  = MainWindow(); mw.show()
    sys.exit(app.exec_())

