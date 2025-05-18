#!/usr/bin/env python3
# -------------------------------------------------
# threads.py  – acquisition & MCA worker threads
# oversample fisso = 2
# -------------------------------------------------

import time, numpy as np
from PyQt5 import QtCore
from picosdk.errors import PicoSDKCtypesError
from picosdk.ps2000 import ps2000 as ps      # per ps2000_stop()

TRIG_TIMEOUT_MS = 500   # autotrigger se manca il fronte
OVERSAMPLE_DEFAULT = 2  # << fisso

# ------------------------------------------------- Oscilloscope live
class AcqThread(QtCore.QThread):
    data_ready     = QtCore.pyqtSignal(object, object)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, scope, gui_parent):
        super().__init__(gui_parent)
        self.scope = scope

    def run(self):
        g = self.parent()
        try:
            while not self.isInterruptionRequested():
                try:
                    # oversample fisso
                    self.scope.oversample = OVERSAMPLE_DEFAULT

                    self.scope.set_channel('A', True,
                                            g.range_cb.currentData(),
                                            g.coupling_cb.currentText())

                    if g.trig_enable_cb.isChecked() and g.trig_level_sb.value()!=0:
                        self.scope.set_trigger(g.trig_level_sb.value(),
                                               g.trig_dir_cb.currentText(),
                                               auto_ms=TRIG_TIMEOUT_MS)
                    else:
                        self.scope.set_trigger(0, 'RISING', auto_ms=1)

                    t, v = self.scope.run_block(g.ns_sb.value(), g.tb_cb.currentData())
                    self.data_ready.emit(t, v)
                    time.sleep(0.02)
                except PicoSDKCtypesError as e:
                    self.error_occurred.emit(str(e)); break
        finally:
            ps.ps2000_stop(self.scope.handle)

    def stop(self):
        self.requestInterruption(); self.wait()


# ------------------------------------------------- MCA worker
class McaThread(QtCore.QThread):
    bin_ready      = QtCore.pyqtSignal(int)
    waveform_ready = QtCore.pyqtSignal(object, object)
    rate_update    = QtCore.pyqtSignal(int, float)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, scope, gui_parent):
        super().__init__(gui_parent)
        self.scope = scope

    def run(self):
        g = self.parent(); total=0; t0=time.time()
        try:
            while not self.isInterruptionRequested():
                try:
                    rng = g.mca_range_cb.currentData()
                    pol = g.mca_pol_cb.currentText()
                    direc = 'RISING' if pol=='Positive' else 'FALLING'

                    self.scope.oversample = OVERSAMPLE_DEFAULT

                    self.scope.set_channel('A', True, rng, 'AC')
                    self.scope.set_trigger(g.mca_lld_sb.value(),
                                           direc,
                                           auto_ms=TRIG_TIMEOUT_MS)

                    t, v = self.scope.run_block(g.mca_ns_sb.value(), g.mca_tb_cb.currentData())

                    # baseline (se spuntato)
                    if g.baseline_cb.isChecked():
                        n_bl = max(1, int(0.1*len(v)))
                        v = v - v[:n_bl].mean()

                    self.waveform_ready.emit(t, v)

                    if g.metric_cb.currentText() == "Peak":
                        peak  = v.max() if pol=='Positive' else -v.min()
                        value = peak
                        norm  = rng
                    else:
                        dt    = t[1]-t[0] if len(t)>1 else 1
                        area  = np.trapz(np.clip(v,0,None) if pol=='Positive' else np.clip(-v,0,None), dx=dt)
                        value = area
                        norm  = rng * dt * len(v)

                    idx = int((value / norm) * 255)
                    idx = min(max(idx,0),255)

                    total += 1
                    elapsed = time.time()-t0
                    self.bin_ready.emit(idx)
                    self.rate_update.emit(total, total/elapsed if elapsed else 0.0)
                except PicoSDKCtypesError as e:
                    self.error_occurred.emit(str(e)); break
        finally:
            ps.ps2000_stop(self.scope.handle)

    def stop(self):
        self.requestInterruption(); self.wait()

