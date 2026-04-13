from SplineEstimator import KernelEstimator
from utils import spline_to_kernel, get_torch_spline, generate_images, compute_fft, compute_psd
from PSDDataset import PSDDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

device = 'cuda'
model = KernelEstimator()
#checkpoint = torch.load("/home/cxv166/PhantomTesting/Code/training_output_0.5/checkpoints/best_checkpoint.pth", map_location=device)
#model.load_state_dict(checkpoint['model_state_dict'])
#model.to(device)
#model.eval() 
checkpoint = torch
dataset = PSDDataset(root_dir=r"/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root")
loader = DataLoader(dataset=dataset,batch_size=32)
I_smooth,I_sharp, _,_ = next(iter(loader))
psd_smooth = compute_psd(I_smooth, device='cuda').to(device, non_blocking=True)
psd_sharp  = compute_psd(I_sharp,  device='cuda').to(device, non_blocking=True)
I_smooth_fft = compute_fft(I_smooth)
I_sharp_fft = compute_fft(I_sharp)

plt.plot(I_smooth_fft.real[0,:,:].to('cpu'))
plt.savefig("plot2")
plt.clf()
I_smooth_fft = I_smooth_fft.real
print(torch.min(I_smooth_fft))
print(torch.max(I_smooth_fft))
print(torch.min(torch.log(I_smooth_fft) + 1e-7))
print(torch.max(torch.log(I_smooth_fft)))

#I_sharp_fft = I_sharp_fft.real.clamp(min=1e-7)
#print(torch.min(I_sharp_fft))
#print(torch.max(I_smooth_fft))
#print(torch.min(torch.log(I_sharp_fft)))
#print(torch.max(torch.log(I_sharp_fft)))


