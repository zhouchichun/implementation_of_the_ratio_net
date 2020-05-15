import os
import glob

ckpt_path=glob.glob("ckpt_*")
for ckpt in ckpt_path:
    for file in glob.glob(ckpt+"/*"):
        os.remove(file)
        
ckpt_path=glob.glob("*log.txt")
for ckpt in ckpt_path:
    os.remove(ckpt)
ckpt_path=glob.glob("*.png")
for ckpt in ckpt_path:
    os.remove(ckpt)