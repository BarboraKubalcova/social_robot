from tuning import Tuning
import usb.core
import usb.util
import time

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
print(dev)
if dev:
    print("Before tuning")
    Mic_tuning = Tuning(dev)
    print("After tuning")
    # print(Mic_tuning.is_voice())
    while True:
        try:
            print(Mic_tuning.is_voice())
            time.sleep(1)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(str(e))
    
else:
    print("Nejdzem")