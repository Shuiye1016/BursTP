import os
import time
startTime = time.time()
os.system("python process_burst.py")
# ---
os.system("python gen_globalg.py")
# ---
os.system("python key_sample.py")
# ---
os.system("python split.py")
# ---
os.system("python data_transforms.py")
# ---
endTime = time.time()
useTime = (endTime - startTime) / 60
print("preprocess %s" % useTime)