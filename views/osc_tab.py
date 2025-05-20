#!/usr/bin/env python3
# -------------------------------------------------------------
# views/osc_tab.py – Tab “Oscilloscopio” (solo View, MVC)
# -------------------------------------------------------------
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from controllers.osc_controller import OscController
from models.scope import SimplePicoScope2000


def fmt_time_div(sec: float) -> str:
    if sec >= 1:       return f"{sec:.2f} s/div"
    if sec >= 1e-3:    return f"{sec*1e3:.2f} ms/div"
    if sec >= 1e-6:    return f"{sec*1e6:.2f} µs/div"
    return                   f"{sec*1e9:.2f} ns/div"


class OscTab(QtWidgets.QWidget):
    """View pura; la logica è in OscController."""
    # ----------------------------------------------------------
    def __init__(self, scope, parent=None):
        super().__init__(parent)

        self.controller = OscController(scope)
        self.scope = scope                           # <──  FIX
        self._build_ui()
        self._connect_signals()

    # ----------------------------------------------------------
    def _build_ui(self):
        scope = self.scope                           # <──  uso locale

        h = QtWidgets.QHBoxLayout(self)
        f = QtWidgets.QFormLayout(); f.setLabelAlignment(QtCore.Qt.AlignRight)

        # range ------------------------------------------------
        self.range_cb = QtWidgets.QComboBox()
        for r in SimplePicoScope2000.AVAILABLE_RANGES:
            self.range_cb.addItem(f"±{int(r*1e3)} mV" if r < 1 else f"±{int(r)} V", r)
        self.range_cb.setCurrentIndex(4)
        f.addRow("Full-Scale:", self.range_cb)

        # coupling ---------------------------------------------
        self.coupling_cb = QtWidgets.QComboBox(); self.coupling_cb.addItems(["DC", "AC"])
        f.addRow("Coupling:", self.coupling_cb)

        # time/div ---------------------------------------------
        self.tb_cb = QtWidgets.QComboBox()
        sample_test = 1000
        for tb in range(26):
            try:
                ticks, u = scope._timebase_info(tb, sample_test)
                dt = ticks * SimplePicoScope2000._UNIT_TO_SEC.get(u, 1e-9)
                self.tb_cb.addItem(fmt_time_div(dt*(sample_test-1)/10), tb)
            except Exception:
                pass
        f.addRow("Time/div:", self.tb_cb)

        # samples ----------------------------------------------
        self.ns_sb = QtWidgets.QSpinBox(); self.ns_sb.setRange(100, 1_000_000); self.ns_sb.setValue(1000)
        f.addRow("Samples N:", self.ns_sb)

        # resolution -------------------------------------------
        self.re_cb = QtWidgets.QComboBox(); self.re_cb.addItems(["8", "8.5", "9", "10"])
        f.addRow("Resolution:", self.re_cb)

        # trigger ----------------------------------------------
        self.trig_enable_cb = QtWidgets.QCheckBox("Enable HW trigger"); f.addRow(self.trig_enable_cb)
        self.trig_level_sb  = QtWidgets.QDoubleSpinBox(); self.trig_level_sb.setRange(-20,20); self.trig_level_sb.setDecimals(3)
        f.addRow("Threshold (V):", self.trig_level_sb)
        self.trig_dir_cb     = QtWidgets.QComboBox(); self.trig_dir_cb.addItems(["RISING","FALLING"])
        f.addRow("Direction:", self.trig_dir_cb)

        # noise -------------------------------------------------
        self.noise_btn   = QtWidgets.QPushButton("Measure noise")
        self.noise_label = QtWidgets.QLabel("Noise σ: — mV")
        hn = QtWidgets.QHBoxLayout(); hn.addWidget(self.noise_btn); hn.addWidget(self.noise_label)
        f.addRow(hn)

        # start/stop -------------------------------------------
        hb = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn  = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        hb.addWidget(self.start_btn); hb.addWidget(self.stop_btn); f.addRow(hb)

        h.addLayout(f)

        # plot --------------------------------------------------
        self.plot = pg.PlotWidget(title="Channel A")
        self.plot.setLabel('bottom', 'Time', 's'); self.plot.setLabel('left', 'Voltage', 'V')
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot(pen='y')
        h.addWidget(self.plot, 1)

    # ----------------------------------------------------------
    def _connect_signals(self):
        # UI → controller
        self.start_btn.clicked.connect(self._start_stream)
        self.stop_btn .clicked.connect(self._stop_stream)
        self.noise_btn.clicked.connect(self._measure_noise)

        # controller → UI
        c = self.controller
        c.waveform_ready.connect(self._update_plot)
        c.noise_measured.connect(lambda s: self.noise_label.setText(f"Noise σ: {s*1e3:.2f} mV"))
        c.error_occurred.connect(self._show_error)

    # ----------------------------------------------------------
    # callback UI
    def _start_stream(self):
        opts = dict(
            rng      = self.range_cb.currentData(),
            coupling = self.coupling_cb.currentText(),
            tb       = self.tb_cb.currentData(),
            ns       = self.ns_sb.value(),
            trig_on  = self.trig_enable_cb.isChecked(),
            trig_lvl = self.trig_level_sb.value(),
            trig_dir = self.trig_dir_cb.currentText(),
            res_bits = self.re_cb.currentText(),
        )
        self.controller.start_stream(**opts)
        self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def _stop_stream(self):
        self.controller.stop_stream()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)

    def _measure_noise(self):
        self.noise_btn.setEnabled(False)
        self.controller.measure_noise(
            rng      = self.range_cb.currentData(),
            coupling = self.coupling_cb.currentText(),
            tb       = self.tb_cb.currentData(),
            ns       = self.ns_sb.value(),
        )
        self.noise_btn.setEnabled(True)

    # ----------------------------------------------------------
    # handler segnali controller
    @QtCore.pyqtSlot(object, object)
    def _update_plot(self, t, v):
        self.curve.setData(t, v)
        self.plot.setXRange(0, t[-1])
        fs = self.range_cb.currentData(); self.plot.setYRange(-fs, fs)

    @QtCore.pyqtSlot(str)
    def _show_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "PicoScope error", msg)
        self._stop_stream()

    # ----------------------------------------------------------
    def clean_up(self):
        self.controller.stop_stream()

