#%%
from audioop import reverse
import pydicom
import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
from pathlib import Path
import glob
from PIL import Image
#%%
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

def mk_img3d_CT(slices):
    slices = sorted(slices ,key=lambda s:s.SliceLocation,reverse=True)
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.insert(0,len(slices))
    img3d_CT = np.zeros(img_shape)
    for i, s in enumerate(slices):

        ri = s.RescaleIntercept
        rs = s.RescaleSlope
        wc = s.WindowCenter
        ww = s.WindowWidth
        img2d = s.pixel_array
        img2d = img2d*rs + ri
        img3d_CT[i,:,:] = img2d
    return img3d_CT

def mk_img3d_PET(slices):
    slices = sorted(slices ,key=lambda s:s.SliceLocation,reverse=True)
    img_shape = list(slices[0].pixel_array.shape)
    img_shape.insert(0,len(slices))
    img3d = np.zeros(img_shape)
    img3d_SUV = np.zeros(img_shape)
    for i, s in enumerate(slices):
        SUV = SUV_img(s)
        img3d_SUV[i,:,:] = SUV
    return img3d_SUV

def mk_MIP_SUV(img3d_SUV):
    MIP_SUV = np.nanmax(img3d_SUV,axis=1)
    return MIP_SUV


def f_SUV(k):
    a1 = plt.subplot(1, 1, 1)
    plt.imshow(img3d_SUV[k, :, :],cmap='gray')
    plt.show()
#%%
path = glob.glob('img1/*')
print(path)
dcm_files_list = []
name_lst = []
for p in path:
    files = glob.glob(p+'/*')
    dcm_files = []
    name_lst.append(p)
    len_file = len(files)
    for i in range(len_file):
        dcm_files.append(pydicom.dcmread(files[i]))


    dcm_files_list.append(dcm_files)

# for cur_dir,dirs,files in os.walk(path):
#     cur_dir = cur_dir
#     dirs = dirs
#     files = files


#%%
print('dcm_files_list',len(dcm_files_list)) 
slices_PET_lst = []
slices_CT_lst = []

for dcm_files,n in zip(dcm_files_list,name_lst):
    print('dcm_files',len(dcm_files))
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
    
    img3d_SUV = mk_img3d_PET(slices_PET)
    img3d_CT = mk_img3d_CT(slices_CT)
    param_CT = len(slices_CT)
    param_PET = len(slices_PET[0].pixel_array)
    print(n)
    print('param_PET',param_PET)
    slices_PET_lst.append(slices_PET)
    slices_CT_lst.append(slices_CT)
    print("slices_PET: {}".format(len(slices_PET)))
    print("slices_CT: {}".format(len(slices_CT)))
    print("スライス位置がないファイル: {}".format(skipcnt))
    ps_PET = slices_PET[0].PixelSpacing    # ps = [0.571, 0.571] 1ピクセルの [y, x] 長さ
    ss_PET = slices_PET[0].SliceThickness  # ss = "3.0" スライスの厚み
    ax_aspect_PET = ps_PET[0]/ps_PET[1]        # yの長さ/xの長さ =  1
    sag_aspect_PET = ss_PET/ps_PET[0]          # スライスの厚み/y軸方向への１ピクセルの長さ = 3/0.571 = 5.25
    cor_aspect_PET = ss_PET/ps_PET[1]

    MIP_SUV = mk_MIP_SUV(img3d_SUV)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    plt.imshow(MIP_SUV, cmap='gray')
    ax.set_aspect(sag_aspect_PET)
    plt.axis('off')
    plt.show()
    plt.imsave(r'D:\PET\MIP-img/{}MIP.png'.format(n.split('\\')[1]), MIP_SUV,cmap='gray')



    
    print('----------------------------------------------')

