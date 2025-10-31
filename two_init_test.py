# two_init_test.py
import time, mvsdk
devs = mvsdk.CameraEnumerateDevice()
assert len(devs)>=2, "need 2 cameras"
def get_name(d): return d.GetFriendlyName()
def init(d):
    for k in range(4):
        try:
            h = mvsdk.CameraInit(d, -1, -1)
            print("INIT OK:", get_name(d))
            return h
        except mvsdk.CameraException as e:
            print("INIT RETRY", k+1, get_name(d), e.error_code, e.message)
            time.sleep(0.2*(k+1))
    raise
h1 = init(devs[0]); time.sleep(0.3); h2 = init(devs[1])
mvsdk.CameraUnInit(h2); mvsdk.CameraUnInit(h1); print("BOTH OK")
