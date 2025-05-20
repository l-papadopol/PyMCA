#!/usr/bin/env python3
# -------------------------------------------------------------
# views/mca_tab.py – Tab “MCA” (Multi-Channel Analyser)
# -------------------------------------------------------------
import time
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

from controllers.mca_controller import McaController
from models.scope import SimplePicoScope2000
from .osc_tab import fmt_time_div   # riusiamo la stessa utility


# ----------------------------------------------------------------------
# widget helpers
# ----------------------------------------------------------------------
def _volt_range_combo(max_w=150):
    cb = QtWidgets.QComboBox()
    cb.setMaximumWidth(max_w)
    for r in SimplePicoScope2000.AVAILABLE_RANGES:
        cb.addItem(f"±{int(r*1e3)} mV" if r < 1 else f"±{int(r)} V", r)
    return cb


def _make_dspin(vmin, vmax, dec, value=None, suffix="", max_w=150):
    sb = QtWidgets.QDoubleSpinBox()
    sb.setRange(vmin, vmax)
    sb.setDecimals(dec)
    if value is not None:
        sb.setValue(value)
    if suffix:
        sb.setSuffix(suffix)
    sb.setMaximumWidth(max_w)
    return sb


# ======================================================================
class McaTab(QtWidgets.QWidget):
    """View pura: visualizza dati e inoltra input al McaController."""
    # ------------------------------------------------------------------
    def __init__(self, scope: SimplePicoScope2000, parent=None):
        super().__init__(parent)

        # ---------------- Controller -----------------------------------
        self.ctrl = McaController(scope)

        # ---------------- Dati run-time --------------------------------
        self.bins = np.arange(256)
        self.energy = self.bins.copy()
        self.hist = np.zeros_like(self.bins, dtype=np.uint32)
        self.mca_t0 = 0.0
        self._paused = False
        self._pause_elapsed = 0.0

        # ---------------- UI -------------------------------------------
        self._build_ui(scope)
        self._wire_signals()
        self._load_prototype_ui_defaults()   # se c’è prototype.npz

        # timer per il label "Elapsed"
        self._timer = QtCore.QTimer(self, interval=200, timeout=self._update_elapsed)

    # ------------------------------------------------------------------
    # UI construction
    def _build_ui(self, scope: SimplePicoScope2000):
        outer = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        outer.addLayout(top)

        # ------------ LEFT controls ------------------------------------
        left = QtWidgets.QFormLayout(); left.setLabelAlignment(QtCore.Qt.AlignRight)

        self.range_cb = _volt_range_combo()
        self.lld_sb   = _make_dspin(-20, 20, 3)
        self.tb_cb    = QtWidgets.QComboBox()
        self.re_cb    = QtWidgets.QComboBox(); self.re_cb.addItems(["8","8.5","9","10"])
        self.gain_sb  = _make_dspin(0.001, 1e4, 3, 1.0, suffix=" keV/ch")
        self.display_cb = QtWidgets.QComboBox(); self.display_cb.addItems(["Bar", "Line"])

        # timebase list (uguale all’osc_tab)
        sample_test = 1000
        for tb in range(26):
            try:
                ticks, u = scope._timebase_info(tb, sample_test)
                dt = ticks * SimplePicoScope2000._UNIT_TO_SEC.get(u, 1e-9)
                self.tb_cb.addItem(fmt_time_div(dt*(sample_test-1)/10), tb)
            except Exception:
                pass

        left.addRow("Full-Scale:", self.range_cb)
        left.addRow("LLD (V):",    self.lld_sb)
        left.addRow("Time/div:",   self.tb_cb)
        left.addRow("Resolution:", self.re_cb)
        left.addRow("Gain:",       self.gain_sb)
        left.addRow("Display:",    self.display_cb)

        # ------------ RIGHT controls -----------------------------------
        right = QtWidgets.QFormLayout(); right.setLabelAlignment(QtCore.Qt.AlignRight)

        self.coupling_cb = QtWidgets.QComboBox(); self.coupling_cb.addItems(["DC","AC"])
        self.pol_cb      = QtWidgets.QComboBox(); self.pol_cb.addItems(["Positive","Negative"])
        self.ns_sb       = QtWidgets.QSpinBox();  self.ns_sb.setRange(100, 1_000_000); self.ns_sb.setValue(1000)

        self.offset_sb   = _make_dspin(-1e4, 1e4, 2, suffix=" keV")
        self.tail_sb     = QtWidgets.QSpinBox(); self.tail_sb.setRange(1, 500); self.tail_sb.setValue(20); self.tail_sb.setSuffix(" samp")
        self.baseline_cb = QtWidgets.QCheckBox()

        self.shape_tol_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.shape_tol_slider.setRange(70, 99); self.shape_tol_slider.setValue(85)
        self.shape_tol_label  = QtWidgets.QLabel("0.85")

        right.addRow("Coupling:",    self.coupling_cb)
        right.addRow("Polarity:",    self.pol_cb)
        right.addRow("Samples N:",   self.ns_sb)
        right.addRow("Offset:",      self.offset_sb)
        right.addRow("Tail length:", self.tail_sb)
        right.addRow("Baseline:",    self.baseline_cb)
        right.addRow("Shape tol ρ:", self.shape_tol_slider)
        right.addRow("",             self.shape_tol_label)

        top.addLayout(left); top.addLayout(right)

        # ------------ Pulse preview ------------------------------------
        self.pulse_plot = pg.PlotWidget(); self.pulse_plot.setFixedSize(260, 180)
        self.pulse_plot.hideAxis('bottom'); self.pulse_plot.hideAxis('left')
        self.pulse_curve  = self.pulse_plot.plot(pen='c')
        self.tail_scatter = self.pulse_plot.plot([], [], pen=None, symbol='d', symbolSize=6,
                                                 symbolBrush='r', symbolPen=None)
        self.peak_scatter = self.pulse_plot.plot([], [], pen=None, symbol='o', symbolSize=8,
                                                 symbolBrush='g', symbolPen=None)
        top.addWidget(self.pulse_plot)

        # ------------ Buttons ------------------------------------------
        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start MCA")
        self.pause_btn = QtWidgets.QPushButton("Pause"); self.pause_btn.setEnabled(False)
        self.stop_btn  = QtWidgets.QPushButton("Stop");  self.stop_btn.setEnabled(False)
        self.shape_btn = QtWidgets.QPushButton("Pulse shape…")
        btn_row.addWidget(self.start_btn); btn_row.addWidget(self.pause_btn)
        btn_row.addWidget(self.stop_btn);  btn_row.addWidget(self.shape_btn)
        outer.addLayout(btn_row)

        # ------------ Histogram ----------------------------------------
        pen_y, brush_y = pg.mkPen('y'), pg.mkBrush('y')
        self.hist_plot = pg.PlotWidget(title="MCA Spectrum")
        self.hist_plot.setLabel('bottom', 'Energy', 'keV'); self.hist_plot.setLabel('left', 'Counts')
        self.bar_item  = pg.BarGraphItem(x=self.energy, height=self.hist,
                                         width=self.gain_sb.value()*0.8, brush=brush_y)
        self.hist_plot.addItem(self.bar_item)
        self.line_item = self.hist_plot.plot(self.energy, self.hist, pen=pen_y, connect='all')
        self.line_item.setVisible(False)

        self.roi = pg.LinearRegionItem(values=(50, 150), brush=(150,150,0,30))
        self.hist_plot.addItem(self.roi); self.roi.sigRegionChanged.connect(self._update_fwhm)
        self.fwhm_enable_cb = QtWidgets.QCheckBox("Enable FWHM"); self.fwhm_enable_cb.setChecked(True)
        self.fwhm_enable_cb.toggled.connect(self.roi.setVisible)

        outer.addWidget(self.hist_plot, 1)

        # ------------ Counters -----------------------------------------
        cnt = QtWidgets.QHBoxLayout()
        self.total_label   = QtWidgets.QLabel("Total: 0")
        self.rate_label    = QtWidgets.QLabel("Rate: 0 evt/s")
        self.elapsed_label = QtWidgets.QLabel("Elapsed: 0 s")
        self.fwhm_label    = QtWidgets.QLabel("FWHM: —")
        cnt.addWidget(self.total_label); cnt.addWidget(self.rate_label)
        cnt.addWidget(self.elapsed_label)
        cnt.addWidget(self.fwhm_enable_cb); cnt.addWidget(self.fwhm_label)
        cnt.addStretch()
        outer.addLayout(cnt)

    # ------------------------------------------------------------------
    # wiring signals controller ↔ view  
    def _wire_signals(self):
        # UI → controller
        self.start_btn.clicked.connect(self._start_mca)
        self.pause_btn.clicked.connect(self._pause_mca)
        self.stop_btn .clicked.connect(self._stop_mca)
        self.shape_btn.clicked.connect(self._open_calib_dialog)

        # live changes that influenzano la resa grafica
        self.display_cb.currentIndexChanged.connect(self._update_histogram)
        self.gain_sb.valueChanged.connect(self._update_histogram)
        self.offset_sb.valueChanged.connect(self._update_histogram)
        self.shape_tol_slider.valueChanged.connect(
            lambda v: self.shape_tol_label.setText(f"{v/100:.2f}")
        )

        # controller → UI
        c = self.ctrl
        c.pulse_ready.connect(self._update_pulse)
        c.bin_ready.connect(self._add_bin)
        c.rate_update.connect(self._update_rate)
        c.error_occurred.connect(self._on_error)

    # ------------------------------------------------------------------
    # ---------------------- Run control -------------------------------
    def _build_settings(self) -> dict:
        """
        Raccoglie le impostazioni correnti e restituisce un dizionario
        **con nomi primitivi**, NON widget.

        Assicurati che `controllers/mca_controller.py` e (se necessario)
        una versione adattata di `McaThread` leggano questi campi.
        """
        return dict(
            rng        = self.range_cb.currentData(),
            lld        = self.lld_sb.value(),
            tb         = self.tb_cb.currentData(),
            ns         = self.ns_sb.value(),
            polarity   = self.pol_cb.currentText(),
            coupling   = self.coupling_cb.currentText(),
            resolution = self.re_cb.currentText(),
            baseline   = self.baseline_cb.isChecked(),
            tail       = self.tail_sb.value(),
            shape_tol  = self.shape_tol_slider.value() / 100,
            gain       = self.gain_sb.value(),
            offset     = self.offset_sb.value(),
        )

    # slot UI
    def _start_mca(self):
        self.hist.fill(0); self._update_histogram()
        self.total_label.setText("Total: 0")
        self.rate_label.setText("Rate: 0 evt/s")
        self.mca_t0 = time.time(); self._pause_elapsed = 0; self._paused = False
        self.ctrl.start_mca(self._build_settings())   # <-- controller
        self._timer.start()

        # pulsanti
        self.start_btn.setEnabled(False); self.pause_btn.setEnabled(True); self.stop_btn.setEnabled(True)

    def _pause_mca(self):
        if not self._paused:
            self.ctrl.pause_mca()
            self._pause_elapsed = time.time() - self.mca_t0
            self._timer.stop(); self._paused = True
            self.start_btn.setEnabled(True); self.pause_btn.setEnabled(False)

    def _stop_mca(self):
        self.ctrl.stop_mca()
        self._timer.stop(); self._paused = False; self._pause_elapsed = 0
        self._update_elapsed()
        self.start_btn.setEnabled(True); self.pause_btn.setEnabled(False); self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # ---------------------- Update handlers ---------------------------
    @QtCore.pyqtSlot(object, object)
    def _update_pulse(self, t, v):
        self.pulse_curve.setData(t, v)
        pk = int(np.argmax(v))
        self.peak_scatter.setData([t[pk]], [v[pk]])
        if self.baseline_cb.isChecked():
            n = self.tail_sb.value()
            if n < len(v):
                self.tail_scatter.setData(t[-n:], v[-n:])
            else:
                self.tail_scatter.clear()
        else:
            self.tail_scatter.clear()

    @QtCore.pyqtSlot(int)
    def _add_bin(self, i: int):
        self.hist[i] += 1
        self._update_histogram()

    @QtCore.pyqtSlot(int, float)
    def _update_rate(self, tot, rate):
        self.total_label.setText(f"Total: {tot}")
        self.rate_label .setText(f"Rate: {rate:.1f} evt/s")

    def _update_elapsed(self):
        el = (time.time() - self.mca_t0) if not self._paused else self._pause_elapsed
        self.elapsed_label.setText(f"Elapsed: {el:.1f} s")

    # ------------------------------------------------------------------
    # histogram / FWHM
    def _update_energy(self):
        self.energy = self.bins * self.gain_sb.value() + self.offset_sb.value()

    def _update_histogram(self):
        self._update_energy()
        if self.display_cb.currentText() == "Bar":
            self.bar_item.setOpts(x=self.energy, height=self.hist,
                                  width=self.gain_sb.value()*0.8)
            self.bar_item.setVisible(True);  self.line_item.setVisible(False)
        else:
            self.line_item.setData(self.energy, self.hist, connect='all')
            self.line_item.setVisible(True); self.bar_item.setVisible(False)

        ymax = self.hist.max(); self.hist_plot.setYRange(0, ymax*1.1 if ymax else 1)
        self._update_fwhm()

    def _update_fwhm(self):
        if not self.fwhm_enable_cb.isChecked():
            self.fwhm_label.setText("FWHM: —"); return
        x1, x2 = self.roi.getRegion()
        m = (self.energy >= x1) & (self.energy <= x2)
        h, e = self.hist[m], self.energy[m]
        if h.size < 3 or h.max() == 0: self.fwhm_label.setText("FWHM: —"); return
        pk = int(np.argmax(h)); half = h[pk]/2

        # interpolazione linear
        def _cross(side):
            rng = range(pk-1, -1, -1) if side == 'L' else range(pk, h.size-1)
            for i in rng:
                j = i+1 if side == 'L' else i+1
                if h[j] <= half <= h[i]:
                    frac = (half - h[j]) / (h[i]-h[j] + 1e-12) if side == 'R' else (half - h[i]) / (h[j]-h[i] + 1e-12)
                    return e[j] + frac*(e[i]-e[j])
            return None

        left  = _cross('L'); right = _cross('R')
        if left is None or right is None:
            self.fwhm_label.setText("FWHM: —")
            return
        fwhm = right - left; pct = fwhm/e[pk]*100 if e[pk] else 0
        self.fwhm_label.setText(f"FWHM: {fwhm:.2f} keV ({pct:.1f} %)")

    # ------------------------------------------------------------------
    # calibrazione
    def _open_calib_dialog(self):
        dlg = CalibDialog(self.ctrl, self)   # dialog usa controller per salvare
        dlg.exec_()

    def _load_prototype_ui_defaults(self):
        """Se esiste prototype.npz, sincronizza alcune combobox/spinbox."""
        if not Path("prototype.npz").exists(): return
        try:
            d = np.load("prototype.npz", allow_pickle=True)
            def set_idx(cb, value):
                i = cb.findData(value)
                if i >= 0: cb.setCurrentIndex(i)

            set_idx(self.range_cb, float(d["rng"]))
            set_idx(self.tb_cb,    int(d["tb"]))
            self.ns_sb.setValue(int(d["ns"]))
            self.lld_sb.setValue(float(d["lld"]))
            self.pol_cb.setCurrentText(str(d["polarity"]))
            if "resolution" in d.files:
                idx = self.re_cb.findText(str(d["resolution"]))
                if idx >= 0: self.re_cb.setCurrentIndex(idx)
        except Exception as e:
            print("prototype load failed:", e)

    # ------------------------------------------------------------------
    # error and cleanup
    @QtCore.pyqtSlot(str)
    def _on_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "PicoScope error", msg)
        self._stop_mca()
        self.start_btn.setEnabled(False)

    def clean_up(self):
        self.ctrl.stop_mca()


# ======================================================================
# Calibrazione prototipo dialog
# ======================================================================
class CalibDialog(QtWidgets.QDialog):
    """Acquisisce N pulse, calcola la media e la salva via controller."""
    def __init__(self, controller: McaController, parent: McaTab):
        super().__init__(parent)
        self.setWindowTitle("Pulse shape calibration")
        self.ctrl = controller
        self.scope = controller.scope
        self.proto = None
        self.settings = None

        v = QtWidgets.QVBoxLayout(self)

        self.plot = pg.PlotWidget(title="Running average (0/—)")
        self.plot.hideAxis('bottom'); self.plot.hideAxis('left')
        self.avg_curve = self.plot.plot(pen='c')
        v.addWidget(self.plot)

        # ---------------- form -----------------------------------------
        g = QtWidgets.QGridLayout(); v.addLayout(g)
        self.range_cb = _volt_range_combo()
        self.tb_cb    = QtWidgets.QComboBox()
        # usa stessa timebase del tab
        for i in range(parent.tb_cb.count()):
            self.tb_cb.addItem(parent.tb_cb.itemText(i), parent.tb_cb.itemData(i))
        self.ns_sb    = QtWidgets.QSpinBox(); self.ns_sb.setRange(100,1_000_000); self.ns_sb.setValue(parent.ns_sb.value())
        self.lld_sb   = _make_dspin(-20, 20, 3, parent.lld_sb.value())
        self.pol_cb   = QtWidgets.QComboBox(); self.pol_cb.addItems(["Positive","Negative"]); self.pol_cb.setCurrentText(parent.pol_cb.currentText())
        self.re_cb    = QtWidgets.QComboBox(); self.re_cb.addItems(["8","8.5","9","10"]); self.re_cb.setCurrentText(parent.re_cb.currentText())
        self.npulse_sb= QtWidgets.QSpinBox(); self.npulse_sb.setRange(10,1000); self.npulse_sb.setValue(100)

        labels = ["Range:", "Time/div:", "Samples N:", "LLD (V):", "Polarity:", "Resolution:", "Pulses:"]
        widgets= [self.range_cb, self.tb_cb, self.ns_sb, self.lld_sb,
                  self.pol_cb, self.re_cb, self.npulse_sb]
        for r,(lab,w) in enumerate(zip(labels, widgets)):
            g.addWidget(QtWidgets.QLabel(lab), r,0); g.addWidget(w, r,1)

        # ---------------- buttons --------------------------------------
        hb = QtWidgets.QHBoxLayout(); v.addLayout(hb)
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_save  = QtWidgets.QPushButton("Save"); self.btn_save.setEnabled(False)
        self.btn_cancel= QtWidgets.QPushButton("Cancel")
        hb.addStretch(); hb.addWidget(self.btn_start); hb.addWidget(self.btn_save); hb.addWidget(self.btn_cancel)

        # ---------------- thread worker --------------------------------
        self.worker = _ProtoWorker(self.scope)
        self.worker.pulse_ready.connect(self._update_average)
        self.worker.finished.connect(self._acq_finished)

        # connect btn
        self.btn_start.clicked.connect(self._start_acq)
        self.btn_save.clicked.connect(self._save_and_close)
        self.btn_cancel.clicked.connect(self.reject)

    # ------------------------------------------------------------------
    def _start_acq(self):
        self.btn_start.setEnabled(False); self.btn_save.setEnabled(False)
        self.btn_start.setText("Running…")
        self.proto = None; self._collected = 0
        n = self.npulse_sb.value()
        self.plot.setTitle(f"Running average (0/{n})")
        self.settings = dict(
            rng      = float(self.range_cb.currentData()),
            tb       = int(self.tb_cb.currentData()),
            ns       = int(self.ns_sb.value()),
            lld      = float(self.lld_sb.value()),
            polarity = self.pol_cb.currentText(),
            re       = self.re_cb.currentText(),
            n_target = n
        )
        self.worker.configure(**self.settings); self.worker.start()

    @QtCore.pyqtSlot()
    def _acq_finished(self):
        self.btn_save.setEnabled(True)
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Try again")

    @QtCore.pyqtSlot(object)
    def _update_average(self, v):
        self._collected += 1
        if self.proto is None: self.proto = v.copy()
        else: self.proto += (v - self.proto) / self._collected
        self.avg_curve.setData(self.proto)
        self.plot.setTitle(f"Running average ({self._collected}/{self.settings['n_target']})")

    def _save_and_close(self):
        if self.proto is not None:
            self.ctrl.save_prototype(np.asarray(self.proto), self.settings)
        self.accept()

    # stop thread if dialog closed
    def reject(self):
        self.worker.stop(); super().reject()

# ----------------------------------------------------------------------
class _ProtoWorker(QtCore.QThread):
    """Thread di acquisizione singoli pulse per il prototipo."""
    pulse_ready = QtCore.pyqtSignal(object)

    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        self.cfg = None

    def configure(self, **kw):
        self.cfg = kw

    def run(self):
        if not self.cfg: return
        import picosdk.ps2000 as ps2k
        ps = ps2k.ps2000
        s = self.cfg; captured = 0
        try:
            while captured < s['n_target'] and not self.isInterruptionRequested():
                self.scope.oversample = 2
                self.scope.set_channel('A', True, s['rng'], 'AC')
                self.scope.set_trigger(
                    s['lld'],
                    'RISING' if s['polarity']=="Positive" else 'FALLING',
                    auto_ms=500
                )
                _, v = self.scope.run_block(s['ns'], s['tb'])
                if s['polarity']=="Negative": v = -v
                if v.max() < 1.2*s['lld']: continue
                self.pulse_ready.emit(v); captured += 1
        finally:
            ps.ps2000_stop(self.scope.handle)

    def stop(self):
        self.requestInterruption(); self.wait()

