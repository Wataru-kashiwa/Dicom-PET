#%%
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob

files = []
path = "./SamplePETCT/PET_WB_TrueX_Exp(AC)_103/*"
for fname in glob.glob(path,recursive=False):
  files.append(pydicom.dcmread(fname))
  
slices = []
skipcount = 0
for f in files:
    if hasattr(f, 'SliceLocation'):
        slices.append(f)
    else:
        skipcount = skipcount + 1

print("該当ファイル数: {}".format(len(slices)))
print("スライス位置がないファイル: {}".format(skipcount))

# スライスの順番をそろえる
slices = sorted(slices, key=lambda s: s.SliceLocation) # .SliceLocationで場所情報を取得
print(len(slices))

ps = slices[0].PixelSpacing
ss = slices[0].SliceThickness
ax_aspect = ps[0]/ps[1]
sag_aspect = ss/ps[0]
cor_aspect = ss/ps[1]
print(ps,ss,ax_aspect,sag_aspect,cor_aspect)
# %%
# 3Dnumpy array 0配列
img_shape = list(slices[0].pixel_array.shape)
img_shape.insert(0,len(slices))
img_3d = np.zeros(img_shape)
print(img_shape)

# %%
# 3Dのnumpy array
for i, s in enumerate(slices):
  if i == 0:
    print(s)
  img2d = s.pixel_array
  img_3d[i,:,:] = img2d

# プロットする
middle0 = img_shape[0]//2  
middle1 = img_shape[1]//2  #3d array の(y軸)上下軸中央　512/2 = 256
middle2 = img_shape[2]//2  #3d array の(x軸)左右軸中央　512/2 = 256

a1 = plt.subplot(1, 3, 1)
plt.imshow(img_3d[middle0,:, :])
a1.set_aspect(ax_aspect)  # ax_aspect = 1

a2 = plt.subplot(1, 3, 2)
plt.imshow(img_3d[:, middle1, :])
a2.set_aspect(cor_aspect) # cor_aspect = 5.25

a3 = plt.subplot(1, 3, 3)
plt.imshow(img_3d[:, :, middle2]) # 
a3.set_aspect(sag_aspect) # sag_aspect = 5.25

plt.show()
# %%
a1 = plt.subplot(1, 1, 1)
plt.imshow(img_3d[135, :, :], cmap=plt.cm.gray,vmin = 0, vmax =75)
a1.set_aspect(ax_aspect)
plt.show()
# %%
import matplotlib.pyplot as plt
from ipywidgets import interact
import numpy as np

param = len(slices)
    
def f(k):
    a1 = plt.subplot(1, 1, 1)
    plt.imshow(img_3d[k, :, :], cmap=plt.cm.gray)
    a1.set_aspect(ax_aspect)
    plt.show()

interact(f, k=(0,param-1) )
# %%
def f(k):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.imshow(img_3d[:, k, :], cmap=plt.cm.gray)
    ax.set_aspect(sag_aspect)
    plt.show()

interact(f, k=(0,param-1) )

# %%
