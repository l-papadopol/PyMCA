#!/usr/bin/env python3
# ---------------------------------------------------------------------
# threads.py – acquisition workers
# ---------------------------------------------------------------------
import os, time, numpy as np
from PyQt5 import QtCore
from picosdk.errors import PicoSDKCtypesError
from picosdk.ps2000 import ps2000 as ps

TRIG_TIMEOUT_MS    = 500
OVERSAMPLE_DEFAULT = 2

# ---------- Resolution Enhancement -----------------------------------
def res_enh(arr, bits: str):
    n={"8":1,"8.5":2,"9":4,"10":16}.get(bits,1)
    if n==1: return arr
    return np.convolve(arr,np.ones(n)/n,mode="same")

# ---------------------------------------------------------------------
class AcqThread(QtCore.QThread):
    data_ready     = QtCore.pyqtSignal(object,object)
    error_occurred = QtCore.pyqtSignal(str)
    def __init__(self,scope,gui): super().__init__(gui); self.scope=scope
    def run(self):
        g=self.parent()
        try:
            while not self.isInterruptionRequested():
                try:
                    self.scope.oversample=OVERSAMPLE_DEFAULT
                    self.scope.set_channel('A',True,g.range_cb.currentData(),g.coupling_cb.currentText())
                    if g.trig_enable_cb.isChecked() and g.trig_level_sb.value():
                        self.scope.set_trigger(g.trig_level_sb.value(),g.trig_dir_cb.currentText(),auto_ms=TRIG_TIMEOUT_MS)
                    else: self.scope.set_trigger(0,'RISING',auto_ms=1)
                    t,v=self.scope.run_block(g.ns_sb.value(),g.tb_cb.currentData())
                    v=res_enh(v,g.re_cb.currentText())
                    self.data_ready.emit(t,v); time.sleep(0.02)
                except PicoSDKCtypesError as e:
                    self.error_occurred.emit(str(e)); break
        finally: ps.ps2000_stop(self.scope.handle)
    def stop(self): self.requestInterruption(); self.wait()

# ---------------------------------------------------------------------
class McaThread(QtCore.QThread):
    bin_ready      = QtCore.pyqtSignal(int)
    waveform_ready = QtCore.pyqtSignal(object,object)
    rate_update    = QtCore.pyqtSignal(int,float)
    error_occurred = QtCore.pyqtSignal(str)
    def __init__(self,scope,gui):
        super().__init__(gui); self.scope=scope
        if os.path.exists("prototype.npz"):
            d=np.load("prototype.npz"); self.prototype=d["proto"]; self.proto_norm=np.linalg.norm(self.prototype)
        else: self.prototype=None; self.proto_norm=1.0
    def run(self):
        g=self.parent(); total=0; t0=time.time()
        try:
            while not self.isInterruptionRequested():
                try:
                    rng=g.mca_range_cb.currentData(); pol=g.mca_pol_cb.currentText()
                    self.scope.oversample=OVERSAMPLE_DEFAULT
                    self.scope.set_channel('A',True,rng,'AC')
                    self.scope.set_trigger(g.mca_lld_sb.value(),'RISING' if pol=="Positive" else 'FALLING',auto_ms=TRIG_TIMEOUT_MS)
                    t,v_raw=self.scope.run_block(g.mca_ns_sb.value(), g.mca_tb_cb.currentData())
                    v=res_enh(v_raw,g.re_mca_cb.currentText())
                    if pol=="Negative": v=-v
                    # shape filter
                    tol=g.shape_tol_slider.value()/100
                    if self.prototype is not None:
                        score=np.dot(v,self.prototype)/(np.linalg.norm(v)*self.proto_norm+1e-12)
                        if score < tol: continue
                    # baseline
                    if g.baseline_cb.isChecked():
                        n=g.tail_sb.value()
                        if n<len(v): v=v-v[-n:].mean()
                    self.waveform_ready.emit(t,v)
                    idx=int(round((v.max()/rng)*255)); idx=max(0,min(255,idx))
                    total+=1; el=time.time()-t0
                    self.bin_ready.emit(idx); self.rate_update.emit(total,total/el if el else 0.0)
                except PicoSDKCtypesError as e:
                    self.error_occurred.emit(str(e)); break
        finally: ps.ps2000_stop(self.scope.handle)
    def stop(self): self.requestInterruption(); self.wait()

