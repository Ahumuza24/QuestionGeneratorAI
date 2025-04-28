import os
import PyPDF2
import docx

def process_document(file_path):
    """
    Extract text from PDF or Word document
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        str: Extracted text from the document
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def extract_text_from_pdf(pdf_path):
    """
    Extract text from PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    
    return text

def extract_text_from_docx(docx_path):
    """
    Extract text from Word document
    
    Args:
        docx_path (str): Path to the Word document
        
    Returns:
        str: Extracted text from the Word document
    """
    doc = docx.Document(docx_path)
    text = ""
    
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    
    # Extract text from tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                text += cell.text + " "
            text += "\n"
        text += "\n"
    
    return text 