import sys
import pydicom
from pathlib import Path

def print_file_kernels(root_dir: str) -> None:
    root = Path(root_dir)
    dicom_files = sorted(root.rglob("*"))
    found = 0
    with open("kernels.txt", "w") as out:
        for path in dicom_files:
            if not path.is_file():
                continue
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
                kernel = getattr(ds, "ConvolutionKernel", "N/A")
                out.write(f"{str(path):<80} kernel: {kernel}\n")
                found += 1
            except pydicom.errors.InvalidDicomError:
                pass  # skip non-DICOM files silently
            except Exception as e:
                out.write(f"{str(path):<80} ERROR: {type(e).__name__}: {e}\n")
        if found == 0:
            out.write("No valid DICOM files found.\n")
        else:
            out.write(f"\n{found} DICOM file(s) found.\n")

if __name__ == "__main__":
    print_file_kernels('/mnt/vstor/CSE_BME_DLW/cxv166/Data_Root/kernels/S65840')
