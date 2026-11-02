from pathlib import Path
import copy
import csv
import json
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
#importLib

torch.manual_seed(42)
torch.set_num_threads(1)
root = Path(__file__).resolve().parent
out = root / "outputs"
out.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
#importLib2

path = sorted((root / "data" / "train" / "horizontal").glob("*.png"))[0]
chw = transforms.ToTensor()(Image.open(path).convert("RGB"))
# BEGIN A01
# TODO A01: 채널 축을 마지막으로 보내 HWC를 만든다.
#torch.permute(inputTensor,[0,1,2]<-chageAxisNum)
hwc = torch.permute(chw,[1,2,0])
# END A01
nchw = hwc.permute(2, 0, 1).unsqueeze(0)
#uncompress hwc to have lists? need to  check

plt.figure(figsize=(4, 4))
plt.imshow(hwc.numpy())
plt.title("HWC view: horizontal stripes")
plt.axis("off")
plt.tight_layout()
plt.savefig(out / "image_axes.png", dpi=120)
plt.close()
#matplotLib stuff to vis

print("CHW / HWC / NCHW:", tuple(chw.shape), tuple(hwc.shape), tuple(nchw.shape))
print("왕복 값 일치:", torch.equal(chw, nchw[0]))
#outputs
