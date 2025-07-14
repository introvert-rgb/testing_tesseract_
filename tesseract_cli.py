import subprocess
import os
import cv2

def save_temp_image(image, path="temp_input.png"):
    """
    Save OpenCV image to disk as PNG (for Tesseract CLI input).
    """
    cv2.imwrite(path, image)
    return path

def tesseract_cli_ocr(
    image_path,
    output_txt_file="testreceiptscreenshotwithoutpreprocess.txt",
    psm="6",
    oem="3",
    lang="eng",
    whitelist=None
):
    """
    Run Tesseract CLI directly (not pytesseract), with full config support.

    Args:
        image_path (str): Path to the input image
        output_txt_file (str): Desired output .txt filename
        psm (str): Page segmentation mode (e.g., '3', '6', '11', etc.)
        oem (str): OCR Engine mode ('1': Legacy, '3': Default LSTM)
        lang (str): Language code ('eng', 'hin', etc.)
        whitelist (str or None): Characters to whitelist (e.g., '0123456789.')

    Returns:
        str: OCR-extracted text from the output file
    """
    try:
        base_output = os.path.splitext(output_txt_file)[0]  # remove .txt

        # Build command
        cmd = [
            "tesseract", image_path, base_output,
            "--psm", str(psm),
            "--oem", str(oem),
            "-l", lang
        ]

        # Add whitelist config if provided
        if whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]

        subprocess.run(cmd, check=True)

        # Read the output text file
        with open(f"{base_output}.txt", "r", encoding="utf-8") as f:
            return f.read()

    except subprocess.CalledProcessError as e:
        return f"[❌ ERROR] Tesseract failed: {e}"
