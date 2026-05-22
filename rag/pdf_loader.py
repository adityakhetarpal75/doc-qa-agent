from pypdf import PdfReader
import io

def load_pdf(file_path: str) -> list:
    reader = PdfReader(file_path)
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            chunks.append(text)
    return chunks

def load_pdf_bytes(file_bytes: bytes) -> list:
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            chunks.append(text)
    return chunks