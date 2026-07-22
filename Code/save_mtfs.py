from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from Code.Dataset import MTFDataset # Assuming your class is in Dataset.py

def save_2d_mtf_images(npz_path, output_root):
    # 1. Load Dataset
    dataset = MTFDataset(npz_path)
    # We use batch_size=1 to process each file individually
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Starting 2D reconstruction for {len(dataset)} samples...")

    for i, (profile, mtf_1d) in enumerate(loader):
        # Convert back to numpy and remove batch dimension
        mtf_1d = mtf_1d.squeeze().numpy()
        
        # Get kernel type for the folder name
        # We access the raw sample for the string label
        kernel_label = str(dataset.samples[i]['kernel']).strip("[]' ")
        
        # Create kernel-specific directory
        folder_path = os.path.join(output_root, kernel_label)
        os.makedirs(folder_path, exist_ok=True)

        # 2. Reconstruct 2D Radial Grid
        # Create a coordinate system
        grid_size = 256
        y, x = np.indices((grid_size, grid_size))
        center = grid_size / 2.0
        r = np.sqrt((x - center)**2 + (y - center)**2)
        
        # Normalize radius to match the length of our 1D MTF (64 bins)
        # This maps the physical distance from center to the index of our MTF array
        r_normalized = (r / r.max()) * (len(mtf_1d) - 1)
        r_indices = r_normalized.astype(np.int32)
        
        # Clip indices to stay within bounds
        r_indices = np.clip(r_indices, 0, len(mtf_1d) - 1)
        
        # Map 1D values to the 2D grid
        mtf_2d = mtf_1d[r_indices]

        # 3. Save the Image
        file_name = f"sample_{i:03d}_mtf2d.png"
        save_path = os.path.join(folder_path, file_name)
        
        plt.imsave(save_path, mtf_2d, cmap='hot')

    print(f"Done! All images saved to: {output_root}")

if __name__ == "__main__":
    npz_path = r"D:\Charan work file\PhantomTesting\training_dataset.npz"
    output_root = r"D:\Charan work file\PhantomTesting\MTF_2D_Visuals"
    
    save_2d_mtf_images(npz_path, output_root)