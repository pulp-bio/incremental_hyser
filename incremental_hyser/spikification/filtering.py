from __future__ import annotations

import numpy as np
import scipy.signal as ssg

from ..hyser import hyser as hy


# FREQ_CUT_LO_HZ = 10.0
# FREQ_CUT_HI_HZ = 500.0
# ORDER = 4


def create_butterworth_filter(
    fs_hz: float,
    lowcut_hz: float,
    highcut_hz: float,
    order: int,
) -> np.ndarray[np.float64]:

    fnyquist_hz = fs_hz / 2.0
    lowcut_norm = lowcut_hz / fnyquist_hz
    highcut_norm = highcut_hz / fnyquist_hz

    assert lowcut_norm >= 0.0
    assert highcut_norm <= 1.0

    if lowcut_norm == 0.0 and highcut_norm == 1.0:

        raise NotImplementedError("The band is all-pass: not a filter!")

    elif lowcut_norm == 0.0 and highcut_norm < 1.0:

        Wn = highcut_norm
        btype = 'lowpass'

    elif lowcut_norm > 0.0 and highcut_norm < 1.0:

        Wn = [lowcut_norm, highcut_norm]
        btype = 'bandpass'

    elif lowcut_norm > 0.0 and highcut_norm == 1.0:

        Wn = lowcut_norm
        btype = 'highpass'

    sos = ssg.butter(N=order, Wn=Wn, btype=btype, output='sos')

    return sos


def main():
    pass


if __name__ == '__main__':
    main()
