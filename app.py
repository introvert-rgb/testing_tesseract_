import gradio as gr
import numpy as np
from tesseract_working import main_ocr_function

def process_image(image: np.ndarray) -> str:
    result = main_ocr_function(image)
    return result

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Textbox(label="OCR Result"),
    title="📄 Tesseract OCR Demo",
    description="Upload a document or receipt image. This demo uses OpenCV + custom preprocessing + Tesseract CLI.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
