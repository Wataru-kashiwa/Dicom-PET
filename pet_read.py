#%%
import pydicom
import numpy as np 
import cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut

dcmfnm = './SamplePETCT/PET_WB_TrueX_Exp(AC)_103/IM-0002-0300.dcm'
print(pydicom.read_file(dcmfnm))
#%%
def readimageWindowing8(filepath):
    print("Image Widowing 8bit")
    ds = pydicom.dcmread(filepath)
    #print("ds=\n",ds)
    img = ds.pixel_array
    print("\timg shape=",img.shape, type(img))
    print("\tOrg img max=",np.max(img),"min=",np.min(img))
    
    wc  = ds.WindowCenter
    ww  = ds.WindowWidth
    ri  = ds.RescaleIntercept
    rs  = ds.RescaleSlope
    
    img = img*rs + ri   # Convert pixel value to PET value

    print("\twc=",wc,type(wc), "\n\tww=", ww, type(ww))
    print("\trsIntersept =",ri, "\n\trsSlope     =",rs)
    
    maxval = wc + ww//2
    minval = wc - ww//2
    img8 = (img - minval)/(maxval - minval)*255
    img8 = np.clip(img8, 0, 255).astype(np.uint8)
       
    return img8    


def readimageWindowing16(filepath):
    print("Image Widowing16")
    ds = pydicom.dcmread(filepath)
    #print("ds=\n",ds)
    img = ds.pixel_array
    print("\timg shape=",img.shape, type(img))
    print("\tOrg img max=",np.max(img),"min=",np.min(img))
    
    ri  = ds.RescaleIntercept
    rs  = ds.RescaleSlope
    
    img = img*rs + ri   # Convert pixel value to PET value
    img = img.astype(np.uint16) # Unsigned 16bit

    print("\trsIntersept =",ri, "\n\trsSlope     =",rs)
    
    return img


def readimageModalityLut(filepath):
    print("\nConvert with Modalyty_Lut")
    ds = pydicom.dcmread(filepath)   
    pxarry = ds.pixel_array
 
    img = apply_modality_lut(pxarry, ds)
    img = img.astype(np.uint16)
    
    return img


img8   = readimageWindowing8(dcmfnm)  
img16  = readimageWindowing16(dcmfnm)
imgMod = readimageModalityLut(dcmfnm) 


print("\nConverted Windowing  8bit img: max =", np.max(img8), "min=",np.min(img8), "type=",img8.dtype)
cv2.imwrite("win8.png", img8)

# %%
