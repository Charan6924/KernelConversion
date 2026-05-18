import os
import glob
import torch
import numpy as np
import pydicom
from pathlib import Path
from tqdm import tqdm
import csv
import json
from datetime import datetime

def compute_psd_batch_gpu(image_batch, device):
    image_batch = image_batch.to(device)
    freq_map = torch.fft.fftshift(torch.fft.fft2(image_batch), dim=(-2, -1))
    psd = torch.abs(freq_map) ** 2
    psd = torch.log(psd + 1)
    b, c, h, w = psd.shape
    psd_flat = psd.view(b, -1)
    p_min = psd_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    p_max = psd_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    psd = (psd - p_min) / (p_max - p_min + 1e-8)
    return psd


def is_dicom_file(filepath):
    if os.path.basename(filepath).upper() in ['DICOMDIR', 'DIRFILE']:
        return False
    
    try:
        dcm = pydicom.dcmread(filepath, stop_before_pixels=True)
        # Check if it has image data
        return hasattr(dcm, 'Rows') and hasattr(dcm, 'Columns')
    except:
        return False


def find_dicom_files(root_dir):
    """Recursively find all DICOM files"""
    dicom_files = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if is_dicom_file(filepath):
                dicom_files.append(filepath)
    
    return dicom_files


def extract_kernel_name(dcm):
    """Extract kernel name from DICOM metadata"""
    valid_kernels = ['B', 'C', 'CB', 'D', 'E', 'YA', 'YB']
    
    # Try different fields
    fields_to_check = [
        'ConvolutionKernel',
        'ReconstructionKernel', 
        'SeriesDescription',
        'ProtocolName'
    ]
    
    for field in fields_to_check:
        if hasattr(dcm, field):
            value = str(getattr(dcm, field)).upper()
            
            # Check for exact match
            if value in valid_kernels:
                return value
            
            # Check for YA/YB first
            if 'YA' in value:
                return 'YA'
            if 'YB' in value:
                return 'YB'
            
            # Check if it's a short code (2-3 chars)
            if len(value) >= 2 and len(value) <= 3 and value.isalpha():
                return value
            
            # Check for single letter kernels
            for kernel in valid_kernels:
                if len(kernel) == 1 and kernel in value:
                    return kernel
    
    return 'Unknown'


def read_dicom_image(filepath):
    try:
        dcm = pydicom.dcmread(filepath)
        image = dcm.pixel_array.astype(np.float32)
        
        # Check if image is RGB/color (3 channels) or grayscale
        if len(image.shape) == 3:
            # This is an RGB image (likely a screenshot or summary image)
            raise Exception(f"Skipping RGB/color image with shape {image.shape}")
        
        # Ensure we have a 2D grayscale image
        if len(image.shape) != 2:
            raise Exception(f"Unexpected image shape: {image.shape}")
        
        # Apply rescale if available
        if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
            image = image * dcm.RescaleSlope + dcm.RescaleIntercept
        
        # Extract metadata
        metadata = {
            'kernel': extract_kernel_name(dcm),
            'pixel_spacing': getattr(dcm, 'PixelSpacing', [1.0, 1.0])[0],
            'series_number': getattr(dcm, 'SeriesNumber', None),
            'instance_number': getattr(dcm, 'InstanceNumber', None),
            'rows': dcm.Rows,
            'columns': dcm.Columns
        }
        
        return image, metadata
    except Exception as e:
        raise Exception(f"Error reading DICOM: {str(e)}")


def process_dicom_files(input_dir, output_dir, batch_size=8, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find all DICOM files
    print(f"\nSearching for DICOM files in: {input_dir}")
    dicom_files = find_dicom_files(input_dir)
    print(f"Found {len(dicom_files)} DICOM files\n")
    
    if len(dicom_files) == 0:
        print("No DICOM files found!")
        return
    
    # Statistics
    kernel_counts = {}
    success_count = 0
    error_count = 0
    skipped_rgb = 0
    error_log = []
    
    # Process files
    batch = []
    batch_metadata = []
    batch_filepaths = []
    
    for i, filepath in enumerate(tqdm(dicom_files, desc="Processing DICOMs")):
        try:
            # Read DICOM
            image, metadata = read_dicom_image(filepath)
            
            # Add to batch
            batch.append(image)
            batch_metadata.append(metadata)
            batch_filepaths.append(filepath)
            
            # Process batch when full or last file
            if len(batch) == batch_size or i == len(dicom_files) - 1:
                # Find max dimensions in batch
                max_h = max(img.shape[0] for img in batch)
                max_w = max(img.shape[1] for img in batch)
                
                # Pad images to same size
                padded_batch = []
                for img in batch:
                    h, w = img.shape
                    pad_h = max_h - h
                    pad_w = max_w - w
                    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
                    padded_batch.append(padded)
                
                # Convert to tensor (B, 1, H, W)
                batch_tensor = torch.from_numpy(np.array(padded_batch)).unsqueeze(1)
                
                # Compute PSD
                with torch.no_grad():
                    psd_batch = compute_psd_batch_gpu(batch_tensor, device)
                    psd_batch = psd_batch.cpu().numpy()
                
                # Save individual PSDs
                for j, (psd, meta, fpath) in enumerate(zip(psd_batch, batch_metadata, batch_filepaths)):
                    # Get original size
                    orig_h = meta['rows']
                    orig_w = meta['columns']
                    
                    # Crop back to original size
                    psd = psd[0, :orig_h, :orig_w]  # Remove channel and padding
                    
                    # Create filename
                    kernel_name = meta['kernel']
                    base_name = Path(fpath).stem
                    output_name = f"{base_name}_Kernel_{kernel_name}_PSD"
                    
                    # Save as numpy array
                    np.save(os.path.join(output_dir, f"{output_name}.npy"), psd)
                    
                    # Save metadata as JSON
                    meta_dict = {
                        'source_file': fpath,
                        'kernel': kernel_name,
                        'pixel_spacing': float(meta['pixel_spacing']),
                        'series_number': int(meta['series_number']) if meta['series_number'] is not None else None,
                        'instance_number': int(meta['instance_number']) if meta['instance_number'] is not None else None,
                        'shape': psd.shape,
                        'processing_date': datetime.now().isoformat()
                    }
                    
                    with open(os.path.join(output_dir, f"{output_name}_metadata.json"), 'w') as f:
                        json.dump(meta_dict, f, indent=2)
                    
                    # Update statistics
                    kernel_counts[kernel_name] = kernel_counts.get(kernel_name, 0) + 1
                    success_count += 1
                
                # Clear batch
                batch = []
                batch_metadata = []
                batch_filepaths = []
        
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            
            # Track RGB images separately
            if "RGB/color image" in error_msg:
                skipped_rgb += 1
            else:
                # Log other errors
                error_log.append({
                    'file': filepath,
                    'error': error_msg,
                    'error_type': error_type
                })
            
            error_count += 1
            
            # Clear the failed item from batch
            if batch and batch[-1] is not None:
                batch = batch[:-1]
                batch_metadata = batch_metadata[:-1]
                batch_filepaths = batch_filepaths[:-1]
            continue
    
    # Print summary
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files found: {len(dicom_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (RGB/color images): {skipped_rgb}")
    print(f"Other errors: {len(error_log)}")
    print(f"Total errors: {error_count}")
    print(f"\nKernel Distribution:")
    for kernel in sorted(kernel_counts.keys()):
        print(f"  {kernel}: {kernel_counts[kernel]} files")
    print("="*50)
    
    # Save summary
    summary = {
        'total_files': len(dicom_files),
        'success_count': success_count,
        'skipped_rgb': skipped_rgb,
        'error_count': error_count,
        'other_errors': len(error_log),
        'kernel_distribution': kernel_counts,
        'processing_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'processing_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save error log if there are non-RGB errors
    if error_log:
        with open(os.path.join(output_dir, 'error_log.json'), 'w') as f:
            json.dump(error_log, f, indent=2)
        print(f"\nError log saved to: {os.path.join(output_dir, 'error_log.json')}")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary saved to: {os.path.join(output_dir, 'processing_summary.json')}")


def load_and_visualize_psd(psd_file):
    """
    Load a PSD file and return it for visualization
    
    Args:
        psd_file: Path to .npy PSD file
    
    Returns:
        psd: numpy array
        metadata: dict with metadata
    """
    psd = np.load(psd_file)
    
    # Load metadata if available
    metadata_file = psd_file.replace('.npy', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    return psd, metadata


if __name__ == "__main__":
    # Configuration
    input_dir = r'C:\Users\hxw352\Downloads\kernels-20260122T210314Z-3-001\kernels'
    output_dir = r'D:\Charan work file\PhantomTesting\PSD_Results_Output'
    
    # Process files
    process_dicom_files(
        input_dir=input_dir,
        output_dir=output_dir,
        batch_size=16,  # Process 16 images at once
        device='cuda'   # Use 'cpu' if no GPU available
    )
    
    print("\n✓ PSD computation complete!")
    print("\nTo load and visualize a PSD:")
    print("  import numpy as np")
    print("  import matplotlib.pyplot as plt")
    print("  psd = np.load('your_file_PSD.npy')")
    print("  plt.imshow(psd, cmap='hot')")
    print("  plt.colorbar()")
    print("  plt.show()")
