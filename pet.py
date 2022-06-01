#%%
from audioop import reverse
import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

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

slices_PET = []
slices_CT = []
skipcnt = 0

for f in dcm_files:
    if hasattr(f, 'SliceLocation'):
        if f.Modality == 'CT':
            slices_CT.append(f)
        
        else:
            slices_PET.append(f)
    
    else:
        skipcnt += 1


print("該当ファイル数: {}".format(len(slices_PET)))
print("該当ファイル数: {}".format(len(slices_CT)))
print("スライス位置がないファイル: {}".format(skipcnt))


def SUV_img(slice):
    ri = slice.RescaleIntercept
    rs = slice.RescaleSlope
    rad_st = slice['0054','0016'].value[0]['0018','1072'].value
    acqu_ti = slice['0008','0032'].value
    TIME = (int(acqu_ti[:2]) - int(rad_st[:2]))*3600 + (int(acqu_ti[2:4]) - int(rad_st[2:4]))*60 +int(acqu_ti[4:6]) - int(rad_st[4:6])
    T = slice['0054','0016'].value[0]['0018','1075'].value
    RTD = slice['0054','0016'].value[0]['0018','1074'].value
    RTD_TIME = RTD*0.5**(TIME/T)
    P_waight = float(slice['0010','1030'].value)*1000
    SUV_PET = slice.pixel_array*rs +ri
    SUV = SUV_PET*int(P_waight)/RTD_TIME
    return SUV




# %%

def mk_img3d_CT(slices):
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
        img3d[i,:,:] = img2d
    return img3d_CT

#%%
def mk_img3d_PET(slices):
    slices = sorted(slices ,key=lambda s:s.SliceLocation,reverse=False)
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.insert(0,len(slices))
    img3d = np.zeros(img_shape)
    img3d_SUV = np.zeros(img_shape)
    for i, s in enumerate(slices):
        SUV = SUV_img(s)
        img3d_SUV[i,:,:] = SUV
    return img3d_SUV

img3d_SUV = mk_img3d_PET(slices_PET)

param_PET = len(slices_PET[0].pixel_array)
print(param_PET)


#%%
def f_SUV(k):
    a1 = plt.subplot(1, 1, 1)
    plt.imshow(img3d_SUV[k, :, :],cmap='gray')
    plt.show()

interact(f_SUV, k=(0,param_PET-1) )
# %%
ps_PET = slices_PET[0].PixelSpacing    # ps = [0.571, 0.571] 1ピクセルの [y, x] 長さ
ss_PET = slices_PET[0].SliceThickness  # ss = "3.0" スライスの厚み
ax_aspect_PET = ps_PET[0]/ps_PET[1]        # yの長さ/xの長さ =  1
sag_aspect_PET = ss_PET/ps_PET[0]          # スライスの厚み/y軸方向への１ピクセルの長さ = 3/0.571 = 5.25
cor_aspect_PET = ss_PET/ps_PET[1]
def f(k):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.imshow(img3d_SUV[:, k, :], cmap='gray')
    ax.set_aspect(sag_aspect_PET)
    plt.colorbar()
    plt.show()

interact(f, k=(0,param_PET-1) )
#%%


#%%



#%%


img3d_CT = mk_img3d_CT(slices_CT)
param_CT = len(slices_CT)
param_PET = len(slices_PET[0].pixel_array)
img3d_PET = mk_img3d_PET(slices_PET)[0]
img3d_PET_SUV = mk_img3d_PET(slices_PET)[1]


# def f(k):
#     a1 = plt.subplot(1, 1, 1)
#     plt.imshow(img3d_CT[k, :, :],cmap='gray')
#     plt.show()
def f(k):
    a1 = plt.subplot(1, 1, 1)
    plt.imshow(img3d_PET[k, :, :],cmap='gray')
    plt.show()


# interact(f, k=(0,param_CT-1) )

interact(f, k=(0,param_PET-1) )



# %%
ps_CT = slices_CT[0].PixelSpacing    # ps = [0.571, 0.571] 1ピクセルの [y, x] 長さ
ss_CT = slices_CT[0].SliceThickness  # ss = "3.0" スライスの厚み
ax_aspect_CT = ps_CT[0]/ps_CT[1]        # yの長さ/xの長さ =  1
sag_aspect_CT = ss_CT/ps_CT[0]          # スライスの厚み/y軸方向への１ピクセルの長さ = 3/0.571 = 5.25
cor_aspect_CT = ss_CT/ps_CT[1]

def f(k):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.imshow(img3d_CT[:, k, :], cmap=plt.cm.gray, vmin = -500, vmax = 500)
    ax.set_aspect(sag_aspect_CT)
    plt.show()

interact(f, k=(0,param_CT-1) )
# %%


def f(k):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    plt.imshow(img3d_PET[:, k, :], cmap='gray')
    ax.set_aspect(sag_aspect_PET)
    plt.colorbar()
    plt.show()

interact(f, k=(0,param_PET-1) )
# %%
print(slices_PET[0])
# %%
# rad_st = slices_PET[0]['0054','0016'].value['0018','1072']

# %%

#%%










from mpl_toolkits.mplot3d import Axes3D
def make_ax(grid=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.grid(grid)
    return ax

filled = np.array([
    [[1, 0, 1], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 1], [1, 0, 0], [1, 0, 1]],
    [[1, 1, 0], [1, 1, 1], [0, 0, 0]]
])

ax = make_ax(True)
ax.voxels(filled, edgecolors='gray', shade=False)
plt.show()

ax = make_ax()
ax.voxels(filled, facecolors='#1f77b430', edgecolors='gray', shade=False)
plt.show()

ax = make_ax()
ax.voxels(np.ones((3, 3, 3)), facecolors='#1f77b430', edgecolors='gray', shade=False)
plt.show()


IMG_DIM = 100

from skimage.transform import resize
from matplotlib import cm

resized = resize(img3d_SUV, (IMG_DIM, IMG_DIM, IMG_DIM), mode='constant')
def explode(data):
    shape_arr = np.array(data.shape)
    size = shape_arr[:3]*2 - 1
    exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
    exploded[::2, ::2, ::2] = data
    return exploded

def expand_coordinates(indices):
    x, y, z = indices
    x[1::2, :, :] += 1
    y[:, 1::2, :] += 1
    z[:, :, 1::2] += 1
    return x, y, z

def normalize(arr):
    arr_min = np.min(arr)
    return (arr-arr_min)/(np.max(arr)-arr_min)

def plot_cube(cube, angle=320):
    cube = normalize(cube)
    
    facecolors = cm.viridis(cube)
    facecolors[:,:,:,-1] = cube
    facecolors = explode(facecolors)
    
    filled = facecolors[:,:,:,-1] != 0
    x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

    fig = plt.figure(figsize=(30/2.54, 30/2.54))
    ax = fig.gca(projection='3d')
    ax.view_init(30, angle)
    ax.set_xlim(right=IMG_DIM*2)
    ax.set_ylim(top=IMG_DIM*2)
    ax.set_zlim(top=IMG_DIM*2)
    
    ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
    plt.show()


plot_cube(resized[:35,::-1,:25])

