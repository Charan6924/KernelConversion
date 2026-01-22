import numpy as np

def radial_average(psd):
    y, x = np.indices(psd.shape)
    center = np.array(psd.shape) / 2.0
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    return radial_profile