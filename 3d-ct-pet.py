#%%
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
from numpy import unravel_index
from ipywidgets import interact
import cv2

path = "./SamplePETCT/FusionCT_Exp_(5mm5mm_inc)_5/*"
# DICOM ファイルを読み込み
def dicom_read(path):
    files = []
    for fname in glob.glob(path, recursive=False):
        files.append(pydicom.dcmread(fname))
    return files

files = dicom_read(path)

# slicesというリストを作成。ファイルに、slicelocationという属性があれば追加していく。
slices = []
skipcount = 0
for f in files:
    if hasattr(f, 'SliceLocation'):
            if f.SeriesNumber==5: #5回CTを撮影し、5回目のデータのみを抽出
                slices.append(f)
    else:
        skipcount = skipcount + 1

print("該当ファイル数: {}".format(len(slices)))
print("スライス位置がないファイル: {}".format(skipcount))

# スライスの順番をそろえる
slices = sorted(slices, key=lambda s: s.SliceLocation, reverse=True) # .SliceLocationで場所情報を取得

# アスペクト比を計算する
ps = slices[0].PixelSpacing    # ps = [0.976, 0.976] 1ピクセルの [y, x] 長さ
ss = slices[0].SliceThickness  # ss = "5.0" スライスの厚み
ax_aspect = ps[0]/ps[1]        # yの長さ/xの長さ =  1
sag_aspect = ss/ps[0]          # スライスの厚み/y軸方向への１ピクセルの長さ = 5/0.976 = 5.12
cor_aspect = ss/ps[1]          # スライスの厚み/x軸方向への１ピクセルの長さ = 3/0.976 = 5.12

# 空の3Dのnumpy arrayを作成する 
img_shape = list(slices[0].pixel_array.shape) # img_shape = [512, 512]
img_shape.insert(0,len(slices)) # img_shape = [225, 512, 512]  
img3d = np.zeros(img_shape) # 空のarrayを作るnp.zeros([225, 512, 512])

# 3Dのnumpy arrayを作る
for i, s in enumerate(slices):
    img2d = s.pixel_array  # img2d.shape = (512, 512) 
    print(img2d)
    img3d[i,:, :] = img2d   # img3d.shape = (225, 512, 512)
    
img3d -= 1024
# プロットする
middle0 = img_shape[0]//2  #3d array の(z軸)頭尾軸中央　270/2 = 135
middle1 = img_shape[1]//2  #3d array の(y軸)上下軸中央　512/2 = 256
middle2 = img_shape[2]//2  #3d array の(x軸)左右軸中央　512/2 = 256

# a1 = plt.subplot(1, 3, 1)
# plt.imshow(img3d[129,:, :],cmap=plt.cm.gray, vmin=-1000, vmax=500)
# a1.set_aspect(ax_aspect)  # ax_aspect = 1
# print(unravel_index(np.argmax(img3d[135,:,:]), img3d[135,:,:].shape))

a2 = plt.subplot(1, 3, 2)
plt.imshow(img3d[:, 224, :],cmap=plt.cm.gray,vmin=-1000,vmax=500)
a2.set_aspect(cor_aspect) # cor_aspect = 5.25

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
plt.imshow(img3d[:, 224, :], cmap=plt.cm.gray, vmin =-200 , vmax = 300)
ax.set_aspect(sag_aspect)

# a3 = plt.subplot(1, 3, 3)
# plt.imshow(img3d[:, :, 188],cmap=plt.cm.gray,vmin=-1000,vmax=500) # 
# a3.set_aspect(sag_aspect) # sag_aspect = 5.25

plt.show()

param = len(slices)
    
def f(k):
    a1 = plt.subplot(1, 1, 1)
    plt.imshow(img3d[k, :, :], cmap=plt.cm.gray, vmin = -200, vmax = 300)
    a1.set_aspect(ax_aspect)
    plt.show()

interact(f, k=(0,param-1) )
#%%
def f(k):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.imshow(img3d[:, k, :], cmap=plt.cm.gray, vmin =-200 , vmax = 300)
    ax.set_aspect(sag_aspect)
    plt.show()

interact(f, k=(0,param-1) )
# %%
def f(k):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.imshow(img3d[:, :, k].T, cmap=plt.cm.gray, vmin = -200, vmax = 300) # .T はtransposeで形を反転
    ax.set_aspect(1/sag_aspect) # aspectを反転させます。
    plt.show()

interact(f, k=(0,param-1) )

# %%
print(slices[0].RescaleSlope)
print(slices[0].RescaleIntercept)
# %%
