import cv2
import numpy as np

def preprocess_receipt_images(input_image: np.ndarray) -> np.ndarray:
    """
    Preprocess a receipt image with speed-optimized steps.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # 2. Downscaled upscale (2x instead of 4x) → preserves detail with less cost
    upscale = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # 3. Replace fastNlMeansDenoising with faster bilateral filter (or skip if clean)
    denoised = cv2.bilateralFilter(upscale, d=5, sigmaColor=50, sigmaSpace=50)

    # 4. CLAHE contrast enhancement (can tune parameters)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    final = clahe.apply(denoised)
    # final = clahe.apply(upscale)

    return final
