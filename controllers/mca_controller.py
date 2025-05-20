#!/usr/bin/env python3
# ------------------------------------------------------------------
# controllers/mca_controller.py – Controller per il tab MCA (MVC)
# ------------------------------------------------------------------
from pathlib import Path
import time
import numpy as np
from PyQt5 import QtCore
from picosdk.errors import PicoSDKCtypesError
from picosdk.ps2000 import ps2000 as ps

from threads import OVERSAMPLE_DEFAULT, TRIG_TIMEOUT_MS, res_enh


# ======================================================================
# Thread di acquisizione indipendente dalla GUI
# ======================================================================
class _McaWorker(QtCore.QThread):
    """
    Acquisizione MCA continua: emette pulse, bin e rate.

    Parametri (dict `cfg`):
        rng          : full-scale Volt (float)
        coupling     : "DC" / "AC"
        lld          : livello trigger volt (float)
        tb           : timebase index (int)
        ns           : samples (int)
        polarity     : "Positive"/"Negative"
        resolution   : "8","8.5","9","10"
        baseline     : True/False
        tail         : n campioni per baseline
        shape_tol    : 0.70-0.99  (correlazione minima col prototipo)
    """
    pulse_ready  = QtCore.pyqtSignal(object, object)   # t, v
    bin_ready    = QtCore.pyqtSignal(int)              # indice canale 0-255
    rate_update  = QtCore.pyqtSignal(int, float)       # total, rate (Hz)
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, scope, cfg: dict, prototype: np.ndarray | None):
        super().__init__()
        self.scope = scope
        self.cfg   = cfg
        self.prototype = prototype
        self.proto_norm = np.linalg.norm(prototype) if prototype is not None else 1.0

    # --------------------------------------------------------------
    def run(self):
        c = self.cfg
        total = 0
        t0 = time.time()

        try:
            while not self.isInterruptionRequested():
                # ---------- hardware ------------------------------
                self.scope.oversample = OVERSAMPLE_DEFAULT
                self.scope.set_channel('A', True, c['rng'], c['coupling'])
                trig_dir = 'RISING' if c['polarity'] == 'Positive' else 'FALLING'
                self.scope.set_trigger(c['lld'], trig_dir, auto_ms=TRIG_TIMEOUT_MS)

                # ---------- acquisizione blocco --------------------
                t, v_raw = self.scope.run_block(c['ns'], c['tb'])
                v = res_enh(v_raw, c['resolution'])

                if c['polarity'] == 'Negative':
                    v = -v

                # ---------- shape filter --------------------------
                if self.prototype is not None:
                    score = np.dot(v, self.prototype) / (np.linalg.norm(v)*self.proto_norm + 1e-12)
                    if score < c['shape_tol']:
                        continue

                # ---------- baseline ------------------------------
                if c['baseline']:
                    n = c['tail']
                    if n < len(v):
                        v = v - v[-n:].mean()

                self.pulse_ready.emit(t, v)

                # ---------- bin MCA -------------------------------
                idx = int(round((v.max() / c['rng']) * 255))
                idx = max(0, min(255, idx))
                self.bin_ready.emit(idx)

                # ---------- rate ----------------------------------
                total += 1
                elapsed = time.time() - t0
                self.rate_update.emit(total, total/elapsed if elapsed else 0.0)

        except PicoSDKCtypesError as e:
            self.error_occurred.emit(str(e))

        finally:
            ps.ps2000_stop(self.scope.handle)

    # --------------------------------------------------------------
    def stop(self):
        self.requestInterruption()
        self.wait()


# ======================================================================
# Controller vero e proprio
# ======================================================================
class McaController(QtCore.QObject):
    # segnali verso la View
    pulse_ready   = QtCore.pyqtSignal(object, object)
    bin_ready     = QtCore.pyqtSignal(int)
    rate_update   = QtCore.pyqtSignal(int, float)
    error_occurred= QtCore.pyqtSignal(str)

    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        self._thread: _McaWorker | None = None

        # prototipo (facoltativo)
        self.prototype: np.ndarray | None = None
        self.proto_settings: dict = {}
        self._load_prototype()

    # --------------------------------------------------------------
    # gestione run
    def start_mca(self, settings: dict):
        """Avvia acquisizione MCA con le impostazioni fornite dalla View."""
        self.stop_mca()   # assicuriamoci che non ci sia già un thread

        self._thread = _McaWorker(self.scope, settings, self.prototype)
        self._thread.pulse_ready.connect(self.pulse_ready)
        self._thread.bin_ready.connect(self.bin_ready)
        self._thread.rate_update.connect(self.rate_update)
        self._thread.error_occurred.connect(self._relay_error)
        self._thread.start()

    def pause_mca(self):
        """Ferma il thread (può essere riavviato con start_mca)."""
        self.stop_mca()

    def stop_mca(self):
        if self._thread:
            self._thread.stop()
            self._thread = None

    # --------------------------------------------------------------
    # prototipo
    def save_prototype(self, proto: np.ndarray, settings: dict):
        np.savez("prototype.npz", proto=proto, **settings)
        self.prototype       = proto
        self.proto_settings  = settings

    def _load_prototype(self):
        if Path("prototype.npz").exists():
            d = np.load("prototype.npz", allow_pickle=True)
            self.prototype = d["proto"]
            self.proto_settings = {k: d[k].item() for k in d.files if k != "proto"}

    # --------------------------------------------------------------
    # error handling
    def _relay_error(self, msg: str):
        self.error_occurred.emit(msg)
        self.stop_mca()

