# tesseract_working.py

import cv2
import numpy as np
import time
from tesseract_cli import tesseract_cli_ocr, save_temp_image
from reducing_time_preprocessing import preprocess_receipt_images

def resize_if_needed(image, max_dim=1024):
    h, w = image.shape[:2]
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    return image

def is_gpay_screenshot(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    h, w = image.shape[:2]
    return mean_brightness > 245 and w < 800 and h > 1000

def crop_text_region(image):
    orig = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:
            boxes.append((x, y, x + w, y + h))

    if not boxes:
        return None

    x_coords = np.array([x for x1, y1, x2, y2 in boxes for x in (x1, x2)])
    y_coords = np.array([y for x1, y1, x2, y2 in boxes for y in (y1, y2)])
    min_x, max_x = np.clip([x_coords.min(), x_coords.max()], 0, orig.shape[1])
    min_y, max_y = np.clip([y_coords.min(), y_coords.max()], 0, orig.shape[0])

    cropped = orig[min_y:max_y, min_x:max_x]
    cropped = cv2.copyMakeBorder(cropped, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return cropped

def main_ocr_function(image_np: np.ndarray) -> str:
    t0 = time.time()

    # Resize
    img_resized = resize_if_needed(image_np)

    # Crop or skip
    if is_gpay_screenshot(img_resized):
        cropped = img_resized
    else:
        cropped = crop_text_region(img_resized)
        if cropped is None:
            cropped = img_resized

    t1 = time.time()

    # Preprocessing
    processed = preprocess_receipt_images(cropped)
    t2 = time.time()

    # Save temp image and perform OCR
    save_temp_image(processed, "receipt_input01.png")

    text = tesseract_cli_ocr(
        image_path="receipt_input01.png",
        output_txt_file="receipt.txt",
        psm="6",
        oem="3",
        lang="eng"
    )

    t3 = time.time()
    print(f"⏱ Total Time: {(t3 - t0):.2f} seconds")
    return text
