import os
import glob
import numpy as np
import torch
import pydicom
from pydicom.uid import generate_uid
from models.cut_model import CUTModel


class CUTSharp2SmoothOptions:
    def __init__(self):
        self.dataroot = 'mnt'
        self.name = 'CUTModelSharp2Smooth'
        self.easy_label = 'cut_train'
        self.gpu_ids = [0]
        self.checkpoints_dir = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/training_output_cut'
        self.preprocess = 'none'
        self.model = 'cut'
        self.input_nc = 1
        self.output_nc = 1
        self.ngf = 64
        self.ndf = 64
        self.netD = 'basic'
        self.netG = 'resnet_9blocks'
        self.n_layers_D = 3
        self.normG = 'instance'
        self.normD = 'instance'
        self.init_type = 'xavier'
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
        self.n_epochs = 200
        self.n_epochs_decay = 200
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.lr = 0.0002
        self.gan_mode = 'lsgan'
        self.pool_size = 0
        self.lr_policy = 'linear'
        self.lr_decay_iters = 50
        self.CUT_mode = 'CUT'
        self.lambda_GAN = 1.0
        self.lambda_smooth = 0.0
        self.lambda_NCE = 1.0
        self.nce_idt = True
        self.nce_layers = '0,4,8,12,16'
        self.nce_includes_all_negatives_from_minibatch = False
        self.netF = 'mlp_sample'
        self.netF_nc = 256
        self.nce_T = 0.07
        self.num_patches = 256
        self.flip_equivariance = False
        self.device = 'cuda:0'
        self.lambda_spa_unsup_A = 1.0
        self.lambda_kl = 1.0
        self.unsup_idt_spa = True
        self.motion_level = 8.0
        self.shift_level = 10.0
        self.scale_level = 0.0
        self.noise_level = 0.001

class CUTSmooth2SharpOptions(CUTSharp2SmoothOptions):
    def __init__(self):
        super().__init__()
        self.name = 'CUTModel'

HU_MIN, HU_MAX = -1000, 3000
SPAN = HU_MAX - HU_MIN
DEVICE = 'cuda:0'
SHARP_FILENAME = 'I20'  
SMOOTH_FILENAME = 'I20'  

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

OUTPUT_DIR = r'/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/reconstructions_cut_dicom'

_range_checked = {'done': False}

def list_dicom_files(folder):
    paths = [p for p in glob.glob(os.path.join(folder, '*')) if os.path.isfile(p)]
    if not paths:
        raise FileNotFoundError(f'No files found in {folder}')

    def sort_key(p):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            if hasattr(ds, 'InstanceNumber'):
                return (0, int(ds.InstanceNumber))
            if hasattr(ds, 'SliceLocation'):
                return (1, float(ds.SliceLocation))
        except Exception:
            pass
        return (2, p)  # fallback: filename order

    return sorted(paths, key=sort_key)


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


def check_output_range(tensor_out, label):
    if not _range_checked['done']:
        mn, mx = float(tensor_out.min()), float(tensor_out.max())
        print(f'[range check] {label} raw output min={mn:.4f} max={mx:.4f}')
        print('  -> tanh output detected, rescaling to [0,1]' if mn < -0.05
              else '  -> [0,1]-range output, using as-is')


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


def run_model(model, A_tensor, B_tensor):
    with torch.no_grad():
        model.set_input({'A': A_tensor, 'B': B_tensor, 'A_paths': '', 'B_paths': ''})
        model.forward()
        return model.fake_B.detach().cpu().numpy().squeeze()


def reconstruct_volume_dicom(sharp_dir, smooth_dir, sharp_kernel_name, smooth_kernel_name,
                              volume_id, model_s2h, model_h2s, device, output_dir,
                              sharp_filename=None, smooth_filename=None):
    """
    If sharp_filename / smooth_filename are given, ONLY that single file pair
    is processed (folder scanning is skipped entirely). Otherwise falls back
    to processing every file found via list_dicom_files (original behavior).
    """
    if sharp_filename is not None and smooth_filename is not None:
        sharp_paths = [os.path.join(sharp_dir, sharp_filename)]
        smooth_paths = [os.path.join(smooth_dir, smooth_filename)]
        for p in sharp_paths + smooth_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f'Specified file not found: {p}')
    else:
        sharp_paths = list_dicom_files(sharp_dir)
        smooth_paths = list_dicom_files(smooth_dir)

        if len(sharp_paths) != len(smooth_paths):
            raise ValueError(
                f'Slice count mismatch for {volume_id}: '
                f'{len(sharp_paths)} sharp vs {len(smooth_paths)} smooth files. '
                'These two series must correspond slice-for-slice.'
            )

    out_smooth_dir = os.path.join(output_dir, f'{volume_id}_{sharp_kernel_name}_to_{smooth_kernel_name}')
    out_sharp_dir = os.path.join(output_dir, f'{volume_id}_{smooth_kernel_name}_to_{sharp_kernel_name}')
    os.makedirs(out_smooth_dir, exist_ok=True)
    os.makedirs(out_sharp_dir, exist_ok=True)

    for k, (sharp_path, smooth_path) in enumerate(zip(sharp_paths, smooth_paths)):
        print(f'[{k}] sharp={sharp_path}')
        print(f'[{k}] smooth={smooth_path}')

        ds_sharp, hu_sharp = load_slice_hu(sharp_path)
        ds_smooth, hu_smooth = load_slice_hu(smooth_path)

        norm_sharp = hu_to_normalized(hu_sharp)
        norm_smooth = hu_to_normalized(hu_smooth)

        I_sharp = torch.from_numpy(norm_sharp).float().unsqueeze(0).unsqueeze(0).to(device)
        I_smooth = torch.from_numpy(norm_smooth).float().unsqueeze(0).unsqueeze(0).to(device)

        raw_smooth = run_model(model_s2h, I_sharp, I_smooth)
        raw_sharp = run_model(model_h2s, I_smooth, I_sharp)

        res_smooth = raw_smooth.clip(0, 1)
        res_sharp = raw_sharp.clip(0, 1)

        hu_gen_smooth = (res_smooth * SPAN + HU_MIN).clip(HU_MIN, HU_MAX)
        hu_gen_sharp = (res_sharp * SPAN + HU_MIN).clip(HU_MIN, HU_MAX)

        out_smooth_path = os.path.join(out_smooth_dir, f'{k:04d}.dcm')
        out_sharp_path = os.path.join(out_sharp_dir, f'{k:04d}.dcm')

        save_reconstructed_slice(ds_sharp, hu_gen_smooth, out_smooth_path)
        save_reconstructed_slice(ds_smooth, hu_gen_sharp, out_sharp_path)

    print(f'[{volume_id}] Saved {len(sharp_paths)} slice(s) to:\n  {out_smooth_dir}\n  {out_sharp_dir}')


if __name__ == '__main__':
    opt_s2h = CUTSharp2SmoothOptions()
    model_s2h = CUTModel(opt_s2h)
    model_s2h.setup(opt_s2h)
    model_s2h.eval()
    print('Loaded Sharp2Smooth model')

    opt_h2s = CUTSmooth2SharpOptions()
    model_h2s = CUTModel(opt_h2s)
    model_h2s.setup(opt_h2s)
    model_h2s.eval()
    print('Loaded Smooth2Sharp model')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for pair in VOLUME_PAIRS:
        print(f"\nProcessing {pair['volume_id']}")
        reconstruct_volume_dicom(
            sharp_dir=pair['sharp_dir'],
            smooth_dir=pair['smooth_dir'],
            sharp_kernel_name=pair['sharp_kernel_name'],
            smooth_kernel_name=pair['smooth_kernel_name'],
            volume_id=pair['volume_id'],
            model_s2h=model_s2h,
            model_h2s=model_h2s,
            device=DEVICE,
            output_dir=OUTPUT_DIR,
            sharp_filename=pair['sharp_filename'],
            smooth_filename=pair['smooth_filename'],
        )

    print(f'\nDone! All files saved to: {OUTPUT_DIR}')
