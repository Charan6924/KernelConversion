from SplineEstimator import KernelEstimator
from filterModel import FilterEstimator
from utils import spline_to_kernel, get_torch_spline, generate_images, compute_fft, compute_psd
from PSDDataset import PSDDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

device = 'cuda'
model = FilterEstimator()
checkpoint = torch.load('/home/cxv166/KernelConversionResearch/training_filter_model/checkpoints/epoch_17.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval() 
checkpoint = torch
dataset = PSDDataset(root_dir=r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root")
loader = DataLoader(dataset=dataset,batch_size=32)
I_smooth,I_sharp, _,_ = next(iter(loader))
psd_smooth = compute_psd(I_smooth, device='cuda').to(device, non_blocking=True)
psd_sharp  = compute_psd(I_sharp,  device='cuda').to(device, non_blocking=True)
filters2sh, filtersh2s = model(psd_smooth,psd_sharp)

plt.imshow(filters2sh, cmap='hot')
plt.title('Filter smooth to sharp')
plt.savefig('s2sh')
plt.clf()

plt.imshow(filters2sh, cmap='hot')
plt.title('Filter sharp to smooth')
plt.savefig('sh2s')
plt.clf()

#I_sharp_fft = I_sharp_fft.real.clamp(min=1e-7)
#print(torch.min(I_sharp_fft))
#print(torch.max(I_smooth_fft))
#print(torch.min(torch.log(I_sharp_fft)))
#print(torch.max(torch.log(I_sharp_fft)))


