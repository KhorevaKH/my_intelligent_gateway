import os
import easyocr
import torch
import pdfplumber

_ocr_reader = None

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        use_gpu = torch.cuda.is_available()
        print(f"[OCR] GPU доступен: {use_gpu}")
        _ocr_reader = easyocr.Reader(['ru', 'en'], gpu=use_gpu)
        if use_gpu:
            print("[OCR] EasyOCR запущен на GPU")
        else:
            print("[OCR] EasyOCR запущен на CPU")
    return _ocr_reader

def extract_text_from_image(image_path):
    reader = get_ocr_reader()
    result = reader.readtext(image_path, paragraph=True)
    text = " ".join([item[1] for item in result])
    return text

def extract_text_from_pdf(pdf_path):
    full_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text.append(page_text)
        if full_text:
            return "\n".join(full_text)
        else:
            return None
    except Exception as e:
        print(f"Ошибка при извлечении текста из PDF: {e}")
        return None

def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['jpg', 'jpeg', 'png']:
        return extract_text_from_image(filepath)
    elif ext == '.pdf':
        text = extract_text_from_pdf(filepath)
        if text:
            return text
        else:
            return "Не удалось извлечь текст из PDF (возможно, сканированный). Используйте JPG/PNG."
    else:
        return "Неподдерживаемый формат (только JPG, PNG, PDF с текстовым слоем)."