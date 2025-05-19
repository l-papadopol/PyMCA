#!/usr/bin/env python3
# ---------------------------------------------------------------------
# gui.py – PicoScope 2204A Oscilloscope + MCA
# ---------------------------------------------------------------------
import os, sys, time, numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from picosdk.errors import PicoSDKCtypesError

from scope   import SimplePicoScope2000
from threads import AcqThread, McaThread


# ---------- helper ---------------------------------------------------
def fmt_time_div(sec: float) -> str:
    if sec >= 1:      return f"{sec:.2f} s/div"
    if sec >= 1e-3:   return f"{sec*1e3:.2f} ms/div"
    if sec >= 1e-6:   return f"{sec*1e6:.2f} µs/div"
    return                   f"{sec*1e9:.2f} ns/div"


# =====================================================================
class MainWindow(QtWidgets.QMainWindow):
    """Main window."""
    # ------------------------------------------------------------------
    def __init__(self):
        super().__init__()
        try:
            self.scope = SimplePicoScope2000()
        except PicoSDKCtypesError as err:
            QtWidgets.QMessageBox.critical(self, "PicoScope error", str(err))
            sys.exit(1)

        self.setWindowTitle("PicoScope 2204A – Oscilloscope + MCA")
        self.resize(1100, 600)

        tabs = QtWidgets.QTabWidget(); self.setCentralWidget(tabs)
        self.build_osc_tab(QtWidgets.QWidget(), tabs)
        self.build_mca_tab(QtWidgets.QWidget(), tabs)

        self.acq_thread = self.mca_thread = None
        self.mca_paused = False; self.mca_t0 = 0; self.mca_pause = 0
        self.mca_timer  = QtCore.QTimer(self, interval=200,
                                        timeout=self.update_elapsed)

        self.load_prototype_settings()

    # =================================================================
    # ---------------------------- OSC TAB ----------------------------
    # =================================================================
    def build_osc_tab(self, w, tabs):
        tabs.addTab(w, "Oscilloscope")
        h = QtWidgets.QHBoxLayout(w)
        f = QtWidgets.QFormLayout(); f.setLabelAlignment(QtCore.Qt.AlignRight)

        # range
        self.range_cb = QtWidgets.QComboBox()
        for r in SimplePicoScope2000.AVAILABLE_RANGES:
            self.range_cb.addItem(f"±{int(r*1e3)} mV" if r < 1 else f"±{int(r)} V", r)
        self.range_cb.setCurrentIndex(4); f.addRow("Full-Scale:", self.range_cb)

        # coupling
        self.coupling_cb = QtWidgets.QComboBox(); self.coupling_cb.addItems(["DC", "AC"])
        f.addRow("Coupling:", self.coupling_cb)

        # time/div
        self.tb_cb = QtWidgets.QComboBox(); sample_test = 1000
        for tb in range(26):
            try:
                ticks, u = self.scope._timebase_info(tb, sample_test)
                dt = ticks*SimplePicoScope2000._UNIT_TO_SEC.get(u, 1e-9)
                self.tb_cb.addItem(fmt_time_div(dt*(sample_test-1)/10), tb)
            except PicoSDKCtypesError:
                pass
        f.addRow("Time/div:", self.tb_cb)

        # samples
        self.ns_sb = QtWidgets.QSpinBox(); self.ns_sb.setRange(100, 1_000_000); self.ns_sb.setValue(1000)
        f.addRow("Samples N:", self.ns_sb)

        # resolution
        self.re_cb = QtWidgets.QComboBox(); self.re_cb.addItems(["8", "8.5", "9", "10"])
        f.addRow("Resolution:", self.re_cb)

        # trigger
        self.trig_enable_cb = QtWidgets.QCheckBox("Enable HW trigger"); f.addRow(self.trig_enable_cb)
        self.trig_level_sb  = QtWidgets.QDoubleSpinBox(); self.trig_level_sb.setRange(-20,20); self.trig_level_sb.setDecimals(3)
        f.addRow("Threshold (V):", self.trig_level_sb)
        self.trig_dir_cb    = QtWidgets.QComboBox(); self.trig_dir_cb.addItems(["RISING","FALLING"])
        f.addRow("Direction:", self.trig_dir_cb)

        # noise
        self.noise_btn = QtWidgets.QPushButton("Measure noise")
        self.noise_label = QtWidgets.QLabel("Noise σ: — mV")
        hn = QtWidgets.QHBoxLayout(); hn.addWidget(self.noise_btn); hn.addWidget(self.noise_label)
        f.addRow(hn)

        # start/stop
        hb = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn  = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        hb.addWidget(self.start_btn); hb.addWidget(self.stop_btn); f.addRow(hb)

        h.addLayout(f)

        # waveform plot
        self.plot = pg.PlotWidget(title="Channel A")
        self.plot.setLabel('bottom', 'Time', 's'); self.plot.setLabel('left', 'Voltage', 'V')
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot(pen='y')
        self.plot.enableAutoRange(False)
        h.addWidget(self.plot, 1)

        self.noise_btn.clicked.connect(self.measure_noise)
        self.start_btn.clicked.connect(self.start_osc); self.stop_btn.clicked.connect(self.stop_osc)

    # =================================================================
    # ----------------------------- MCA TAB ---------------------------
    # =================================================================
    def build_mca_tab(self, w, tabs):
        tabs.addTab(w, "MCA")
        outer = QtWidgets.QVBoxLayout(w)
        top   = QtWidgets.QHBoxLayout(); outer.addLayout(top)
        def narrow(wid,mx=150): wid.setMaximumWidth(mx); return wid

        # left controls
        left = QtWidgets.QFormLayout(); left.setLabelAlignment(QtCore.Qt.AlignRight)
        self.mca_range_cb = narrow(QtWidgets.QComboBox())
        for r in SimplePicoScope2000.AVAILABLE_RANGES:
            self.mca_range_cb.addItem(f"±{int(r*1e3)} mV" if r < 1 else f"±{int(r)} V", r)
        self.mca_lld_sb = narrow(QtWidgets.QDoubleSpinBox()); self.mca_lld_sb.setRange(-20,20); self.mca_lld_sb.setDecimals(3)
        self.mca_tb_cb  = narrow(QtWidgets.QComboBox())
        self.re_mca_cb  = narrow(QtWidgets.QComboBox()); self.re_mca_cb.addItems(["8","8.5","9","10"])
        for i in range(self.tb_cb.count()):
            self.mca_tb_cb.addItem(self.tb_cb.itemText(i), self.tb_cb.itemData(i))
        self.gain_sb = narrow(QtWidgets.QDoubleSpinBox()); self.gain_sb.setRange(0.001,1e4); self.gain_sb.setDecimals(3); self.gain_sb.setValue(1.0); self.gain_sb.setSuffix(" keV/ch")
        self.display_mode_cb = narrow(QtWidgets.QComboBox()); self.display_mode_cb.addItems(["Bar","Line"])

        left.addRow("Full-Scale:", self.mca_range_cb)
        left.addRow("LLD (V):",    self.mca_lld_sb)
        left.addRow("Time/div:",   self.mca_tb_cb)
        left.addRow("Resolution:", self.re_mca_cb)
        left.addRow("Gain:",       self.gain_sb)
        left.addRow("Display:",    self.display_mode_cb)

        # right controls
        right = QtWidgets.QFormLayout(); right.setLabelAlignment(QtCore.Qt.AlignRight)
        self.mca_coupling_cb = narrow(QtWidgets.QComboBox()); self.mca_coupling_cb.addItems(["DC","AC"])
        self.mca_pol_cb      = narrow(QtWidgets.QComboBox()); self.mca_pol_cb.addItems(["Positive","Negative"])
        self.mca_ns_sb       = narrow(QtWidgets.QSpinBox());  self.mca_ns_sb.setRange(100,1_000_000); self.mca_ns_sb.setValue(1000)

        self.offset_sb = narrow(QtWidgets.QDoubleSpinBox()); self.offset_sb.setRange(-1e4,1e4); self.offset_sb.setDecimals(2); self.offset_sb.setSuffix(" keV")
        self.tail_sb   = narrow(QtWidgets.QSpinBox()); self.tail_sb.setRange(1,500); self.tail_sb.setValue(20); self.tail_sb.setSuffix(" samp")
        self.baseline_cb = QtWidgets.QCheckBox()

        self.shape_tol_slider = narrow(QtWidgets.QSlider(QtCore.Qt.Horizontal),180)
        self.shape_tol_slider.setRange(70,99); self.shape_tol_slider.setValue(85)
        self.shape_tol_label  = QtWidgets.QLabel("0.85")

        right.addRow("Coupling:",    self.mca_coupling_cb)
        right.addRow("Polarity:",    self.mca_pol_cb)
        right.addRow("Samples N:",   self.mca_ns_sb)
        right.addRow("Offset:",      self.offset_sb)
        right.addRow("Tail length:", self.tail_sb)
        right.addRow("Baseline:",    self.baseline_cb)
        right.addRow("Shape tol ρ:", self.shape_tol_slider)
        right.addRow("",             self.shape_tol_label)

        top.addLayout(left); top.addLayout(right)

        # pulse preview
        self.pulse_plot = pg.PlotWidget(); self.pulse_plot.setFixedSize(250,180)
        self.pulse_plot.hideAxis('bottom'); self.pulse_plot.hideAxis('left')
        self.pulse_curve   = self.pulse_plot.plot(pen='c')
        self.tail_scatter  = self.pulse_plot.plot([],[],pen=None,symbol='d',symbolSize=6,symbolBrush='r',symbolPen=None)
        self.peak_scatter  = self.pulse_plot.plot([],[],pen=None,symbol='o',symbolSize=8,symbolBrush='g',symbolPen=None)
        top.addWidget(self.pulse_plot)

        # buttons
        hbtn = QtWidgets.QHBoxLayout()
        self.start_mca_btn = QtWidgets.QPushButton("Start MCA")
        self.pause_mca_btn = QtWidgets.QPushButton("Pause"); self.pause_mca_btn.setEnabled(False)
        self.stop_mca_btn  = QtWidgets.QPushButton("Stop");  self.stop_mca_btn.setEnabled(False)
        self.shape_btn     = QtWidgets.QPushButton("Pulse shape…")
        hbtn.addWidget(self.start_mca_btn); hbtn.addWidget(self.pause_mca_btn); hbtn.addWidget(self.stop_mca_btn); hbtn.addWidget(self.shape_btn)
        outer.addLayout(hbtn)

        # histogram
        self.bins   = np.arange(256)
        self.energy = self.bins
        self.hist   = np.zeros_like(self.bins, dtype=np.uint32)

        self.hist_plot = pg.PlotWidget(title="MCA Spectrum")
        self.hist_plot.setLabel('bottom','Energy','keV'); self.hist_plot.setLabel('left','Counts')
        pen_y, brush_y = pg.mkPen('y'), pg.mkBrush('y')
        self.bar_item  = pg.BarGraphItem(x=self.energy, height=self.hist, width=self.gain_sb.value()*0.8, brush=brush_y)
        self.hist_plot.addItem(self.bar_item)
        self.line_item = self.hist_plot.plot(self.energy, self.hist, pen=pen_y, connect='all'); self.line_item.setVisible(False)

        self.roi = pg.LinearRegionItem(values=(50,150), brush=(150,150,0,30))
        self.hist_plot.addItem(self.roi); self.roi.sigRegionChanged.connect(self.update_fwhm)
        self.fwhm_enable_cb = QtWidgets.QCheckBox("Enable FWHM"); self.fwhm_enable_cb.setChecked(True)
        self.fwhm_enable_cb.toggled.connect(self.roi.setVisible)

        outer.addWidget(self.hist_plot, 1)

        # counters
        cnt = QtWidgets.QHBoxLayout()
        self.total_label   = QtWidgets.QLabel("Total: 0")
        self.rate_label    = QtWidgets.QLabel("Rate: 0 evt/s")
        self.elapsed_label = QtWidgets.QLabel("Elapsed: 0 s")
        self.fwhm_label    = QtWidgets.QLabel("FWHM: —")
        cnt.addWidget(self.total_label); cnt.addWidget(self.rate_label); cnt.addWidget(self.elapsed_label)
        cnt.addWidget(self.fwhm_enable_cb); cnt.addWidget(self.fwhm_label); cnt.addStretch()
        outer.addLayout(cnt)

        # connections
        self.display_mode_cb.currentIndexChanged.connect(self.update_histogram)
        self.gain_sb.valueChanged.connect(self.update_histogram); self.offset_sb.valueChanged.connect(self.update_histogram)
        self.shape_tol_slider.valueChanged.connect(lambda v: self.shape_tol_label.setText(f"{v/100:.2f}"))

        self.start_mca_btn.clicked.connect(self.start_mca)
        self.pause_mca_btn.clicked.connect(self.pause_mca)
        self.stop_mca_btn .clicked.connect(self.stop_mca)
        self.shape_btn    .clicked.connect(self.open_calib_dialog)

    # =================================================================
    # -------------------- LOAD PROTOTYPE SETTINGS --------------------
    # =================================================================
    def load_prototype_settings(self):
        if not os.path.exists("prototype.npz"): return
        try:
            d = np.load("prototype.npz", allow_pickle=True)
            def set_idx(cb,val):
                i = cb.findData(val)
                if i >= 0: cb.setCurrentIndex(i)
            set_idx(self.mca_range_cb, float(d["rng"]))
            set_idx(self.mca_tb_cb,    int(d["tb"]))
            self.mca_ns_sb.setValue(int(d["ns"]))
            self.mca_lld_sb.setValue(float(d["lld"]))
            self.mca_pol_cb.setCurrentText(str(d["pol"]))
            # nuova: risoluzione
            if "re" in d.files:
                idx = self.re_mca_cb.findText(str(d["re"]))
                if idx >= 0: self.re_mca_cb.setCurrentIndex(idx)
        except Exception as e:
            print("prototype load failed:", e)

    # =================================================================
    #                            SLOTS
    # =================================================================
    # --- oscilloscope
    def measure_noise(self):
        self.noise_btn.setEnabled(False)
        try:
            self.scope.set_channel('A', True, self.range_cb.currentData(), self.coupling_cb.currentText())
            self.scope.set_trigger(0,'RISING',auto_ms=1)
            _, v = self.scope.run_block(self.ns_sb.value(), self.tb_cb.currentData())
            self.noise_label.setText(f"Noise σ: {np.std(v)*1e3:.2f} mV")
        finally: self.noise_btn.setEnabled(True)

    def start_osc(self):
        self.acq_thread = AcqThread(self.scope, self)
        self.acq_thread.data_ready.connect(self.update_osc); self.acq_thread.error_occurred.connect(self.thread_error)
        self.acq_thread.start(); self.start_btn.setEnabled(False); self.stop_btn.setEnabled(True)

    def stop_osc(self):
        if self.acq_thread: self.acq_thread.stop()
        self.start_btn.setEnabled(True); self.stop_btn.setEnabled(False)

    @QtCore.pyqtSlot(object, object)
    def update_osc(self, t, v):
        self.curve.setData(t, v)
        self.plot.setXRange(0, t[-1]); fs = self.range_cb.currentData(); self.plot.setYRange(-fs, fs)

    # --- MCA control
    def start_mca(self):
        self.hist.fill(0); self.update_histogram()
        self.total_label.setText("Total: 0"); self.rate_label.setText("Rate: 0 evt/s")
        self.mca_t0=time.time(); self.mca_pause=0
        self.mca_thread=McaThread(self.scope, self)
        self.mca_thread.waveform_ready.connect(self.update_pulse)
        self.mca_thread.bin_ready.connect(self.add_bin)
        self.mca_thread.rate_update.connect(self.update_rate)
        self.mca_thread.error_occurred.connect(self.thread_error)
        self.mca_thread.start(); self.mca_timer.start(); self.mca_paused=False
        self.start_mca_btn.setEnabled(False); self.pause_mca_btn.setEnabled(True); self.stop_mca_btn.setEnabled(True)

    def pause_mca(self):
        if self.mca_thread: self.mca_thread.stop()
        self.mca_pause=time.time()-self.mca_t0; self.mca_paused=True; self.mca_timer.stop()
        self.start_mca_btn.setEnabled(True); self.pause_mca_btn.setEnabled(False)

    def stop_mca(self):
        if self.mca_thread: self.mca_thread.stop()
        self.mca_timer.stop(); self.mca_paused=False; self.mca_pause=0; self.update_elapsed()
        self.start_mca_btn.setEnabled(True); self.pause_mca_btn.setEnabled(False); self.stop_mca_btn.setEnabled(False)

    # --- histogram helpers
    def update_energy(self):
        self.energy = self.bins*self.gain_sb.value()+self.offset_sb.value()

    def update_histogram(self):
        self.update_energy()
        if self.display_mode_cb.currentText()=="Bar":
            self.bar_item.setOpts(x=self.energy,height=self.hist,width=self.gain_sb.value()*0.8)
            self.bar_item.setVisible(True); self.line_item.setVisible(False)
        else:
            self.line_item.setData(self.energy,self.hist,connect='all'); self.line_item.setVisible(True); self.bar_item.setVisible(False)
        ymax=self.hist.max(); self.hist_plot.setYRange(0,ymax*1.1 if ymax else 1)
        self.update_fwhm()

    # --- live updates
    @QtCore.pyqtSlot(object, object)
    def update_pulse(self, t, v):
        self.pulse_curve.setData(t, v)
        pk = int(np.argmax(v))
        self.peak_scatter.setData([t[pk]],[v[pk]])
        if self.baseline_cb.isChecked():
            n = self.tail_sb.value()
            if n < len(v):
                self.tail_scatter.setData(t[-n:], v[-n:])
            else:
                self.tail_scatter.clear()
        else:
            self.tail_scatter.clear()

    @QtCore.pyqtSlot(int)   # <-- corretto: su più righe
    def add_bin(self, i):
        self.hist[i] += 1
        self.update_histogram()

    @QtCore.pyqtSlot(int, float)
    def update_rate(self, tot, rate):
        self.total_label.setText(f"Total: {tot}")
        self.rate_label .setText(f"Rate: {rate:.1f} evt/s")

    def update_elapsed(self):
        el = (time.time()-self.mca_t0) if not self.mca_paused else self.mca_pause
        self.elapsed_label.setText(f"Elapsed: {el:.1f} s")

    # --- FWHM (interpolated) -----------------------------------------
    def update_fwhm(self):
        if not self.fwhm_enable_cb.isChecked():
            self.fwhm_label.setText("FWHM: —"); return
        x1,x2 = self.roi.getRegion()
        mask   = (self.energy >= x1) & (self.energy <= x2)
        h, e   = self.hist[mask], self.energy[mask]
        if h.size < 3 or h.max()==0:
            self.fwhm_label.setText("FWHM: —"); return
        pk     = int(np.argmax(h)); hpk = float(h[pk]); half = hpk/2

        # find left crossing
        left_idx = None
        for i in range(pk-1, -1, -1):
            if h[i] <= half:
                # linear interpolation between i,i+1
                frac = (half - h[i]) / (h[i+1]-h[i] + 1e-12)
                left  = e[i] + frac*(e[i+1]-e[i])
                left_idx = left; break
        # find right crossing
        right_idx = None
        for i in range(pk, h.size-1):
            if h[i+1] <= half:
                frac = (half - h[i]) / (h[i+1]-h[i] + 1e-12)
                right = e[i] + frac*(e[i+1]-e[i])
                right_idx = right; break

        if left_idx is None or right_idx is None:
            self.fwhm_label.setText("FWHM: —"); return
        fwhm = right_idx - left_idx
        pct  = fwhm / e[pk] * 100 if e[pk] else 0
        self.fwhm_label.setText(f"FWHM: {fwhm:.2f} keV ({pct:.1f} %)")

    # calibration dialog
    def open_calib_dialog(self):
        dlg = CalibDialog(self.scope, self)
        if dlg.exec_()==QtWidgets.QDialog.Accepted and dlg.prototype is not None:
            np.savez("prototype.npz", proto=dlg.prototype,
                     rng=dlg.settings['rng'], tb=dlg.settings['tb'],
                     ns=dlg.settings['ns'], lld=dlg.settings['lld'],
                     pol=dlg.settings['polarity'], re=dlg.settings['re'])
            QtWidgets.QMessageBox.information(self,"Pulse shape","prototype.npz saved.")
            self.load_prototype_settings()

    # error/cleanup
    @QtCore.pyqtSlot(str)
    def thread_error(self,msg):
        QtWidgets.QMessageBox.critical(self,"PicoScope error",msg)
        self.stop_osc(); self.stop_mca(); self.start_btn.setEnabled(False); self.start_mca_btn.setEnabled(False)
    def closeEvent(self,e):
        self.stop_osc(); self.pause_mca(); self.scope.close(); super().closeEvent(e)

# =====================================================================
# ------------- CALIBRATION DIALOG & WORKER THREAD --------------------
# =====================================================================
class CalibDialog(QtWidgets.QDialog):
    """Acquire pulses and build prototype waveform."""
    def __init__(self, scope, main:'MainWindow'):
        super().__init__(main)
        self.setWindowTitle("Pulse shape calibration")
        self.scope = scope; self.prototype=None; self.settings=None
        v = QtWidgets.QVBoxLayout(self)

        self.plot=pg.PlotWidget(title="Running average (0/—)")
        self.plot.hideAxis('bottom'); self.plot.hideAxis('left')
        self.avg_curve=self.plot.plot(pen='c'); v.addWidget(self.plot)

        g = QtWidgets.QGridLayout(); v.addLayout(g)
        self.range_cb=QtWidgets.QComboBox()
        for i in range(main.mca_range_cb.count()):
            self.range_cb.addItem(main.mca_range_cb.itemText(i),main.mca_range_cb.itemData(i))
        self.range_cb.setCurrentIndex(main.mca_range_cb.currentIndex())
        self.tb_cb=QtWidgets.QComboBox()
        for i in range(main.mca_tb_cb.count()):
            self.tb_cb.addItem(main.mca_tb_cb.itemText(i),main.mca_tb_cb.itemData(i))
        self.tb_cb.setCurrentIndex(main.mca_tb_cb.currentIndex())
        self.ns_sb=QtWidgets.QSpinBox(); self.ns_sb.setRange(100,1_000_000); self.ns_sb.setValue(main.mca_ns_sb.value())
        self.lld_sb=QtWidgets.QDoubleSpinBox(); self.lld_sb.setRange(-20,20); self.lld_sb.setDecimals(3); self.lld_sb.setValue(main.mca_lld_sb.value())
        self.pol_cb=QtWidgets.QComboBox(); self.pol_cb.addItems(["Positive","Negative"]); self.pol_cb.setCurrentText(main.mca_pol_cb.currentText())
        self.re_cb=QtWidgets.QComboBox(); self.re_cb.addItems(["8","8.5","9","10"]); self.re_cb.setCurrentText(main.re_mca_cb.currentText())
        self.npulse_sb=QtWidgets.QSpinBox(); self.npulse_sb.setRange(10,1000); self.npulse_sb.setValue(100)
        labels=["Range:","Time/div:","Samples N:","LLD (V):","Polarity:","Resolution:","Pulses:"]
        widgets=[self.range_cb,self.tb_cb,self.ns_sb,self.lld_sb,self.pol_cb,self.re_cb,self.npulse_sb]
        for r,(lab,w) in enumerate(zip(labels,widgets)):
            g.addWidget(QtWidgets.QLabel(lab),r,0); g.addWidget(w,r,1)

        hb=QtWidgets.QHBoxLayout(); v.addLayout(hb)
        self.btn_start=QtWidgets.QPushButton("Start")
        self.btn_save =QtWidgets.QPushButton("Save"); self.btn_save.setEnabled(False)
        self.btn_cancel=QtWidgets.QPushButton("Cancel")
        hb.addStretch(); hb.addWidget(self.btn_start); hb.addWidget(self.btn_save); hb.addWidget(self.btn_cancel)

        self.thread=_ProtoThread(scope,self)
        self.thread.pulse_received.connect(self.update_average); self.thread.finished.connect(self.finished_acq)
        self.btn_start.clicked.connect(self.start_acq); self.btn_save.clicked.connect(self.accept); self.btn_cancel.clicked.connect(self.reject)

    # slots
    def start_acq(self):
        self.btn_start.setEnabled(False); self.btn_save.setEnabled(False); self.btn_start.setText("Running…")
        self.prototype=None; self.collected=0
        n=self.npulse_sb.value(); self.plot.setTitle(f"Running average (0/{n})")
        self.settings=dict(rng=float(self.range_cb.currentData()),
                           tb=int(self.tb_cb.currentData()),
                           ns=int(self.ns_sb.value()),
                           lld=float(self.lld_sb.value()),
                           polarity=self.pol_cb.currentText(),
                           re=self.re_cb.currentText(),
                           n_target=n)
        self.thread.configure(**self.settings); self.thread.start()

    @QtCore.pyqtSlot()
    def finished_acq(self):
        self.btn_save.setEnabled(True); self.btn_start.setEnabled(True); self.btn_start.setText("Try again")

    @QtCore.pyqtSlot(object)
    def update_average(self,v):
        self.collected+=1
        if self.prototype is None: self.prototype=v.copy()
        else: self.prototype+=(v-self.prototype)/self.collected
        self.avg_curve.setData(self.prototype)
        self.plot.setTitle(f"Running average ({self.collected}/{self.settings['n_target']})")

class _ProtoThread(QtCore.QThread):
    pulse_received=QtCore.pyqtSignal(object)
    def __init__(self,scope,parent): super().__init__(parent); self.scope=scope; self.settings=None
    def configure(self,**kw): self.settings=kw
    def run(self):
        if not self.settings: return
        s=self.settings; cap=0
        try:
            while cap<s['n_target'] and not self.isInterruptionRequested():
                self.scope.oversample=2
                self.scope.set_channel('A',True,s['rng'],'AC')
                self.scope.set_trigger(s['lld'],'RISING' if s['polarity']=="Positive" else 'FALLING',auto_ms=500)
                _,v=self.scope.run_block(s['ns'],s['tb'])
                if s['polarity']=="Negative": v=-v
                if v.max()<1.2*s['lld']: continue
                self.pulse_received.emit(v); cap+=1
        finally:
            ps = __import__("picosdk.ps2000",fromlist=["ps2000"]).ps2000
            ps.ps2000_stop(self.scope.handle)

# =====================================================================
if __name__=="__main__":
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling,True)
    pg.setConfigOptions(antialias=True)
    app=QtWidgets.QApplication(sys.argv); mw=MainWindow(); mw.show()
    sys.exit(app.exec_())

