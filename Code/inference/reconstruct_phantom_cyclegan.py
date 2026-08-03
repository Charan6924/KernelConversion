import os
import numpy as np
import torch
import pydicom
from pydicom.uid import generate_uid
from models.cycle_gan_model import CycleGANModel


class CycleGANOptions:
    def __init__(self):
        self.dataroot = 'placeholder'
        self.name = 'Cycle-GAN'
        self.easy_label = 'Cycle-GAN'
        self.gpu_ids = [0]
        self.checkpoints_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_cyclegan'
        self.model = 'cycle_gan'
        self.input_nc = 1
        self.output_nc = 1
        self.preprocess = 'none'
        self.ngf = 64
        self.ndf = 64
        self.netG = 'resnet_9blocks'
        self.netD = 'basic'
        self.n_layers_D = 3
        self.normG = 'instance'
        self.normD = 'instance'
        self.init_type = 'normal'
        self.init_gain = 0.02
        self.no_dropout = True
        self.no_antialias = False
        self.no_antialias_up = False
        self.stylegan2_G_num_downsampling = 1
        self.dataset_mode = 'unaligned'
        self.direction = 'AtoB'
        self.serial_batches = False
        self.num_threads = 4
        self.batch_size = 1
        self.max_dataset_size = float('inf')
        self.no_flip = False
        self.display_winsize = 256
        self.random_scale_max = 3.0
        self.epoch = 'latest'
        self.verbose = False
        self.suffix = ''
        self.isTrain = False
        self.phase = 'test'
        self.amp = False
        self.display_freq = 400
        self.display_ncols = 4
        self.display_id = None
        self.display_server = 'http://localhost'
        self.display_env = 'main'
        self.display_port = 8097
        self.update_html_freq = 1000
        self.print_freq = 100
        self.no_html = False
        self.save_latest_freq = 5000
        self.save_epoch_freq = 5
        self.evaluation_freq = 5000
        self.save_by_iter = False
        self.continue_train = False
        self.epoch_count = 1
        self.pretrained_name = None
        self.n_epochs = 100
        self.n_epochs_decay = 100
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lr = 0.0002
        self.lr_policy = 'linear'
        self.lr_decay_iters = 50
        self.gan_mode = 'lsgan'
        self.pool_size = 50
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_identity = 0.5


HU_MIN, HU_MAX = -1000, 3000
SPAN = HU_MAX - HU_MIN
DEVICE = 'cuda'

# ---------------------------------------------------------------------------
# Only these two files (one from S2030, one from S2050) will be processed.
# ---------------------------------------------------------------------------
SHARP_FILENAME = 'I20'   # file inside S2030 (sharp_dir)
SMOOTH_FILENAME = 'I20'  # file inside S2050 (smooth_dir)

VOLUME_PAIRS = [
    {
        'volume_id': 'S65840_S2030',
        'sharp_dir': r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2040',
        'smooth_dir': r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840/S2050',
        'sharp_kernel_name': 'sharp',
        'smooth_kernel_name': 'smooth',
        'sharp_filename': SHARP_FILENAME,
        'smooth_filename': SMOOTH_FILENAME,
    },
]

OUTPUT_DIR = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_cyclegan_dicom'


def get_signed_dtype_and_range(ds):
    bits_stored = int(getattr(ds, 'BitsStored', 16))
    signed = int(getattr(ds, 'PixelRepresentation', 0)) == 1
    if signed:
        dtype = np.int16
        lo, hi = -(2 ** (bits_stored - 1)), (2 ** (bits_stored - 1)) - 1
    else:
        dtype = np.uint16
        lo, hi = 0, (2 ** bits_stored) - 1
    return dtype, lo, hi


def load_slice_hu(path):
    """Read a DICOM file, return (dataset, HU-valued float32 array)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))
    hu = arr * slope + intercept
    return ds, hu


def hu_to_normalized(hu):
    return (np.clip(hu, HU_MIN, HU_MAX) - HU_MIN) / SPAN


def save_reconstructed_slice(reference_ds, hu_array, output_path):
    ds = reference_ds
    slope = float(getattr(ds, 'RescaleSlope', 1))
    intercept = float(getattr(ds, 'RescaleIntercept', 0))

    dtype, lo, hi = get_signed_dtype_and_range(ds)

    raw = (hu_array - intercept) / slope
    raw = np.round(raw)
    raw = np.clip(raw, lo, hi)
    raw = raw.astype(dtype)

    ds.PixelData = raw.tobytes()
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.save_as(output_path)


def reconstruct_slice_dicom(sharp_dir, smooth_dir, sharp_kernel_name, smooth_kernel_name,
                             volume_id, model, device, output_dir,
                             sharp_filename, smooth_filename):
    sharp_path = os.path.join(sharp_dir, sharp_filename)
    smooth_path = os.path.join(smooth_dir, smooth_filename)

    for p in (sharp_path, smooth_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f'Specified file not found: {p}')

    out_smooth_dir = os.path.join(output_dir, f'{volume_id}_{sharp_kernel_name}_to_{smooth_kernel_name}')
    out_sharp_dir = os.path.join(output_dir, f'{volume_id}_{smooth_kernel_name}_to_{sharp_kernel_name}')
    os.makedirs(out_smooth_dir, exist_ok=True)
    os.makedirs(out_sharp_dir, exist_ok=True)

    print(f'sharp={sharp_path}')
    print(f'smooth={smooth_path}')

    ds_sharp, hu_sharp = load_slice_hu(sharp_path)
    ds_smooth, hu_smooth = load_slice_hu(smooth_path)

    norm_sharp = hu_to_normalized(hu_sharp)
    norm_smooth = hu_to_normalized(hu_smooth)

    I_sharp_tensor = torch.from_numpy(norm_sharp).float().unsqueeze(0).unsqueeze(0).to(device)
    I_smooth_tensor = torch.from_numpy(norm_smooth).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # A = sharp (S2030) -> AtoB processes 2030, produces generated smooth (fake_B)
        # B = smooth (S2050) -> BtoA processes 2050, produces generated sharp (fake_A)
        data = {'A': I_sharp_tensor, 'B': I_smooth_tensor, 'A_paths': '', 'B_paths': ''}
        model.set_input(data)
        model.forward()
        I_gen_smooth = model.fake_B
        I_gen_sharp = model.fake_A

    res_sharp = I_gen_sharp.detach().cpu().numpy().squeeze()
    res_smooth = I_gen_smooth.detach().cpu().numpy().squeeze()

    res_sharp = res_sharp.clip(0, 1.0)
    res_smooth = res_smooth.clip(0, 1.0)

    hu_gen_sharp = (res_sharp * SPAN + HU_MIN).clip(HU_MIN, HU_MAX)
    hu_gen_smooth = (res_smooth * SPAN + HU_MIN).clip(HU_MIN, HU_MAX)

    out_smooth_path = os.path.join(out_smooth_dir, '0000.dcm')
    out_sharp_path = os.path.join(out_sharp_dir, '0000.dcm')

    save_reconstructed_slice(ds_sharp, hu_gen_smooth, out_smooth_path)
    save_reconstructed_slice(ds_smooth, hu_gen_sharp, out_sharp_path)

    print(f'[{volume_id}] Saved 1 slice to:\n  {out_smooth_path}\n  {out_sharp_path}')


if __name__ == '__main__':
    opt = CycleGANOptions()
    model = CycleGANModel(opt)
    model.setup(opt)
    model.eval()
    print('Loaded model successfully')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for pair in VOLUME_PAIRS:
        print(f"\nProcessing {pair['volume_id']}")
        reconstruct_slice_dicom(
            sharp_dir=pair['sharp_dir'],
            smooth_dir=pair['smooth_dir'],
            sharp_kernel_name=pair['sharp_kernel_name'],
            smooth_kernel_name=pair['smooth_kernel_name'],
            volume_id=pair['volume_id'],
            model=model,
            device=DEVICE,
            output_dir=OUTPUT_DIR,
            sharp_filename=pair['sharp_filename'],
            smooth_filename=pair['smooth_filename'],
        )

    print(f'\nReconstruction complete! All files saved to: {OUTPUT_DIR}')
