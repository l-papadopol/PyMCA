#!/usr/bin/env python3
# -------------------------------------------------
# scope.py  –  PicoScope 2204A thin wrapper
# -------------------------------------------------

import ctypes, time
import numpy as np
from picosdk.ps2000 import ps2000 as ps
from picosdk.functions import assert_pico2000_ok, adc2mV
from picosdk.PicoDeviceEnums import picoEnum

class SimplePicoScope2000:
    """Minimal wrapper: single-channel A, run-block only."""
    AVAILABLE_RANGES = [0.05,0.1,0.2,0.5,1,2,5,10,20]  # Volt FS
    MAX_ADC = 32767
    _UNIT_TO_SEC = {0:1e-15,1:1e-12,2:1e-9,3:1e-6,4:1e-3,5:1}

    def __init__(self):
        h = ps.ps2000_open_unit(); assert_pico2000_ok(h)
        self.handle = ctypes.c_int16(h)
        self.oversample = 1
        self._range_enum = ps.PS2000_VOLTAGE_RANGE['PS2000_1V']
        self._range_v    = 1
        self.set_channel('A', True, 1, 'DC')

    # ---------- hardware config ----------
    def close(self): ps.ps2000_close_unit(self.handle)

    def set_channel(self, ch, enable, rng_v, coupling):
        ch_e = picoEnum.PICO_CHANNEL[f'PICO_CHANNEL_{ch}']
        coup = picoEnum.PICO_COUPLING[f'PICO_{coupling}']
        rmap = {0.05:'50MV',0.1:'100MV',0.2:'200MV',0.5:'500MV',
                1:'1V',2:'2V',5:'5V',10:'10V',20:'20V'}
        r_e  = ps.PS2000_VOLTAGE_RANGE[f'PS2000_{rmap[rng_v]}']
        assert_pico2000_ok(ps.ps2000_set_channel(self.handle, ch_e,
                                                 int(enable), coup, r_e))
        self._range_enum, self._range_v = r_e, rng_v

    def set_trigger(self, thr, direction, delay=0, auto_ms=0):
        ch   = picoEnum.PICO_CHANNEL['PICO_CHANNEL_A']
        cnt  = int((thr/self._range_v) * self.MAX_ADC)
        dmap = {'RISING':0,'FALLING':1}
        assert_pico2000_ok(ps.ps2000_set_trigger(self.handle, ch, cnt,
                                                 dmap[direction], delay, auto_ms))

    # ---------- acquisition ----------
    def _timebase_info(self, tb, n):
        ti, tu, ms = ctypes.c_int32(), ctypes.c_int32(), ctypes.c_int32()
        assert_pico2000_ok(ps.ps2000_get_timebase(self.handle, tb, n,
                             ctypes.byref(ti), ctypes.byref(tu),
                             self.oversample, ctypes.byref(ms)))
        return ti.value, tu.value

    def run_block(self, n, tb):
        ticks, unit = self._timebase_info(tb, n)
        td = ctypes.c_int32()
        assert_pico2000_ok(ps.ps2000_run_block(self.handle, n, tb,
                             self.oversample, ctypes.byref(td)))
        while ps.ps2000_ready(self.handle) == 0:
            time.sleep(0.001)

        buf = (ctypes.c_int16*n)(); ov = ctypes.c_int16()
        assert_pico2000_ok(ps.ps2000_get_times_and_values(
            self.handle, None, ctypes.byref(buf),
            None, None, None, ctypes.byref(ov), unit, ctypes.c_int32(n)))

        dt = ticks*self._UNIT_TO_SEC.get(unit,1e-9)
        t  = np.linspace(0, dt*(n-1), n)
        mv = adc2mV(buf, self._range_enum, ctypes.c_int16(self.MAX_ADC))
        v  = np.array(mv, dtype=float)*1e-3
        return t, v

