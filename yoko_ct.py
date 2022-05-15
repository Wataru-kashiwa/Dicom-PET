import pydicom
import matplotlib.pyplot as plt 
import cv2
import os

path = "./SamplePETCT/FusionCT_Exp_(5mm5mm_inc)_5/IM-0001-0100.dcm"
dcm_sample = pydicom.dcmread(path)
dcm_wc  = dcm_sample.WindowCenter[1]
dcm_ww  = dcm_sample.WindowWidth[1]
dcm_img = dcm_sample.pixel_array 
name = os.path.basename(path)

#ウィンドウ処理
window_max = dcm_wc + dcm_ww/2                     
window_min = dcm_wc - dcm_ww/2
dcm_img = 255*(dcm_img - window_min)/(window_max - window_min)    
dcm_img[dcm_img > 255] = 255
dcm_img[dcm_img < 0] = 0 
plt.imshow(dcm_img,cmap=plt.cm.bone)
plt.show()