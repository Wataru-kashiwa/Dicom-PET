import pydicom
import matplotlib.pyplot as plt

ct_path = "./SamplePETCT/FusionCT_Exp_(5mm5mm_inc)_5/IM-0001-0005.dcm"
pet_path = "./SamplePETCT/PET_WB_TrueX_Exp(AC)_103/IM-0002-0005.dcm"

def read_dicom(path):
  d = pydicom.filereader.dcmread(path)
  d_img = d.pixel_array
  return d_img

def mkfig_imshow(dicom_img,title):
  fig, ax = plt.subplots()
  ax.set_title(title)
  img = ax.imshow(dicom_img, cmap=plt.cm.jet, interpolation='nearest', origin='upper')
  cbar = fig.colorbar(img, ax=ax, aspect=50, pad=0.08, shrink=0.9, orientation='vertical')
  return fig

ct_img = read_dicom(ct_path)
pet_img = read_dicom(pet_path)
mkfig_imshow(ct_img,"CT-image")
mkfig_imshow(pet_img,"PET-image")
plt.show()
