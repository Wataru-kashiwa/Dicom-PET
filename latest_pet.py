#%%
from audioop import reverse
import pydicom
import os
import numpy as np
path = 'img2/563'

for cur_dir,dirs,files in os.walk(path):
    cur_dir = cur_dir
    dirs = dirs
    files = files
    
# %%
dcm_files = []
len_file = len(files)

for i in range(len_file):
    dcm_files.append(pydicom.dcmread(cur_dir+'/'+files[i]))

print(len(dcm_files))
# %%

slices_168 = []
slices_512 = []
skipcnt = 0

for f in dcm_files:
    if hasattr(f, 'SliceLocation'):
        if f.pixel_array.shape == (512,512):
            slices_512.append(f)
        
        else:
            slices_168.append(f)
    
    else:
        skipcnt += 1


print("該当ファイル数: {}".format(len(slices_168)))
print("該当ファイル数: {}".format(len(slices_512)))
print("スライス位置がないファイル: {}".format(skipcnt))





# %%

def mk_img3d(slices):
    slices = sorted(slices ,key=lambda s:s.SliceLocation,reverse=False)
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.insert(0,len(slices))
    img3d = np.zeros(img_shape)
    for i, s in enumerate(slices):

        ri = s.RescaleIntercept
        rs = s.RescaleSlope
        wc = s.WindowCenter
        ww = s.WindowWidth
        img2d = s.pixel_array
        img2d = img2d*rs + ri
        maxval = wc + ww//2
        minval = wc - ww//2
        img2d = (img2d - minval)/(maxval - minval)*255
        img3d[i,:,:] = img2d
    return img3d

# %%
import matplotlib.pyplot as plt
from ipywidgets import interact

img3d = mk_img3d(slices_512)
param = len(slices_512)
param_168 = len(slices_512)
img3d_168 = mk_img3d(slices_168)
    
def f(k):
    a1 = plt.subplot(1, 1, 1)
    plt.imshow(img3d[k, :, :],cmap='gray')
    plt.show()
def f(k):
    a1 = plt.subplot(1, 1, 1)
    plt.imshow(img3d_168[k, :, :],cmap='gray')
    plt.show()

interact(f, k=(0,param-1) )

interact(f, k=(0,param_168-1) )
# %%
ps = slices_512[0].PixelSpacing    # ps = [0.571, 0.571] 1ピクセルの [y, x] 長さ
ss = slices_512[0].SliceThickness  # ss = "3.0" スライスの厚み
ax_aspect = ps[0]/ps[1]        # yの長さ/xの長さ =  1
sag_aspect = ss/ps[0]          # スライスの厚み/y軸方向への１ピクセルの長さ = 3/0.571 = 5.25
cor_aspect = ss/ps[1]
def f(k):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.imshow(img3d[:, k, :], cmap=plt.cm.gray, vmin = -500, vmax = 500)
    ax.set_aspect(sag_aspect)
    plt.show()

interact(f, k=(0,param-1) )
# %%
# ps = slices_168[0].PixelSpacing    # ps = [0.571, 0.571] 1ピクセルの [y, x] 長さ
# ss = slices_168[0].SliceThickness  # ss = "3.0" スライスの厚み
# ax_aspect = ps[0]/ps[1]        # yの長さ/xの長さ =  1
# sag_aspect = ss/ps[0]          # スライスの厚み/y軸方向への１ピクセルの長さ = 3/0.571 = 5.25
# cor_aspect = ss/ps[1]

# def f(k):
#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111)
#     plt.imshow(img3d_168[:, k, :], cmap='gray')
#     ax.set_aspect(sag_aspect)
#     plt.show()

# interact(f, k=(0,param_168-1) )
# %%
