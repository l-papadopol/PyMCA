# controllers/osc_controller.py
from PyQt5 import QtCore
import time
import numpy as np
from picosdk.errors import PicoSDKCtypesError
from picosdk.ps2000 import ps2000 as ps

from threads import res_enh, OVERSAMPLE_DEFAULT, TRIG_TIMEOUT_MS


class _OscAcqThread(QtCore.QThread):
    """Thread di acquisizione che usa SOLO parametri primitivi."""
    data_ready     = QtCore.pyqtSignal(object, object)   # t, v
    error_occurred = QtCore.pyqtSignal(str)

    def __init__(self, scope, cfg: dict):
        super().__init__()
        self.scope = scope
        self.cfg   = cfg

    # ----------------------------------------------------------
    def run(self):
        g = self.cfg
        try:
            while not self.isInterruptionRequested():
                # HW config
                self.scope.oversample = OVERSAMPLE_DEFAULT
                self.scope.set_channel('A', True, g['rng'], g['coupling'])

                if g['trig_on'] and g['trig_lvl']:
                    self.scope.set_trigger(
                        g['trig_lvl'], g['trig_dir'], auto_ms=TRIG_TIMEOUT_MS
                    )
                else:
                    self.scope.set_trigger(0, 'RISING', auto_ms=1)

                # acquisizione
                t, v = self.scope.run_block(g['ns'], g['tb'])
                v = res_enh(v, g['res_bits'])
                self.data_ready.emit(t, v)
                time.sleep(0.02)

        except PicoSDKCtypesError as e:
            self.error_occurred.emit(str(e))

        finally:
            ps.ps2000_stop(self.scope.handle)

    # ----------------------------------------------------------
    def stop(self):
        self.requestInterruption()
        self.wait()


# ======================================================================
class OscController(QtCore.QObject):
    """Gestisce l’acquisizione del tab Oscilloscopio (MVC – Controller)."""

    waveform_ready  = QtCore.pyqtSignal(object, object)   # t, v
    noise_measured  = QtCore.pyqtSignal(float)            # sigma (Volt)
    error_occurred  = QtCore.pyqtSignal(str)

    def __init__(self, scope):
        super().__init__()
        self.scope = scope
        self._thread = None

    # ----------------------------------------------------------
    # STREAM CONTINUO
    def start_stream(self, *, rng, coupling, tb, ns,
                     trig_on, trig_lvl, trig_dir, res_bits):

        # stop eventuale stream precedente
        self.stop_stream()

        # config packed in a dict
        cfg = dict(rng=rng, coupling=coupling, tb=tb, ns=ns,
                   trig_on=trig_on, trig_lvl=trig_lvl,
                   trig_dir=trig_dir, res_bits=res_bits)

        self._thread = _OscAcqThread(self.scope, cfg)
        self._thread.data_ready.connect(self.waveform_ready)
        self._thread.error_occurred.connect(self._relay_error)
        self._thread.start()

    def stop_stream(self):
        if self._thread:
            self._thread.stop()
            self._thread = None

    # ----------------------------------------------------------
    # MISURA RUMORE (bloccante ma veloce)
    def measure_noise(self, *, rng, coupling, tb, ns):
        try:
            self.scope.set_channel('A', True, rng, coupling)
            self.scope.set_trigger(0, 'RISING', auto_ms=1)
            _, v = self.scope.run_block(ns, tb)
            self.noise_measured.emit(float(np.std(v)))
        except PicoSDKCtypesError as e:
            self._relay_error(str(e))

    # ----------------------------------------------------------
    def _relay_error(self, msg):
        self.error_occurred.emit(msg)
        self.stop_stream()

