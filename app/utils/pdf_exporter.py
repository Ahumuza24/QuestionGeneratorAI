from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import string
import re

def export_to_pdf(document, questions, output_path):
    """
    Export questions and answers to PDF with improved handling for large content
    
    Args:
        document (Document): Document object
        questions (list): List of Question objects
        output_path (str): Path to save the PDF
    """
    # Create PDF document with larger margins for better readability
    pdf = SimpleDocTemplate(
        output_path, 
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=20
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=10
    )
    
    normal_style = ParagraphStyle(
        'Normal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6
    )
    
    answer_style = ParagraphStyle(
        'Answer',
        parent=styles['Normal'],
        fontSize=10,
        leftIndent=5,
        rightIndent=5
    )
    
    # Process text to ensure it works well in PDF
    def clean_text_for_pdf(text):
        if not text:
            return ""
        # Replace problematic characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        # Convert multiple newlines to a single newline
        text = re.sub(r'\n\s*\n', '\n', text)
        # Convert newlines to <br/>
        text = text.replace('\n', '<br/>')
        return text
    
    # Create content
    content = []
    
    # Add title
    content.append(Paragraph(f"Questions for: {document.title}", title_style))
    content.append(Spacer(1, 20))
    
    # Add questions section
    content.append(Paragraph("Questions", heading_style))
    content.append(Spacer(1, 10))
    
    # Add questions - each wrapped in KeepTogether to prevent awkward breaks
    for i, question in enumerate(questions):
        question_section = []
        
        # Question number and text
        question_text = clean_text_for_pdf(question.content)
        question_section.append(Paragraph(f"<b>Q{i+1}.</b> {question_text}", normal_style))
        
        # If multiple choice, add options
        if question.question_type == 'multiple_choice' and question.options:
            for j, option in enumerate(question.options):
                option_letter = string.ascii_uppercase[j]
                option_text = clean_text_for_pdf(option)
                question_section.append(Paragraph(f"<b>{option_letter}.</b> {option_text}", normal_style))
        
        question_section.append(Spacer(1, 10))
        
        # Try to keep each question together, but allow page breaks if necessary
        try:
            content.append(KeepTogether(question_section))
        except:
            # If the content is too large, don't use KeepTogether
            content.extend(question_section)
    
    # Add page break before answers
    content.append(PageBreak())
    
    # Add answers section header
    content.append(Paragraph("Answers", heading_style))
    content.append(Spacer(1, 10))
    
    # Handle answers individually to avoid table layout issues
    for i, question in enumerate(questions):
        answer_section = []
        
        # Create a mini-table for each answer
        if question.question_type == 'multiple_choice' and question.options:
            try:
                # Find the index of the correct answer in options
                correct_index = question.options.index(question.answer)
                answer_text = f"<b>{string.ascii_uppercase[correct_index]}.</b> {clean_text_for_pdf(question.answer)}"
            except (ValueError, IndexError):
                # Fallback if the answer is not found in options
                answer_text = clean_text_for_pdf(question.answer)
        else:
            answer_text = clean_text_for_pdf(question.answer)
        
        # Create answer entry
        answer_data = [[f"<b>Q{i+1}</b>", Paragraph(answer_text, answer_style)]]
        
        # Create a table with flexible width
        answer_table = Table(answer_data, colWidths=[40, None])
        
        # Style the mini-table
        answer_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.lightgrey),
            ('ALIGNMENT', (0, 0), (0, 0), 'CENTER'),
            ('VALIGN', (0, 0), (1, 0), 'TOP'),
            ('GRID', (0, 0), (1, 0), 0.5, colors.grey),
            ('LEFTPADDING', (0, 0), (1, 0), 3),
            ('RIGHTPADDING', (0, 0), (1, 0), 3),
            ('BOTTOMPADDING', (0, 0), (1, 0), 5),
            ('TOPPADDING', (0, 0), (1, 0), 5),
        ]))
        
        answer_section.append(answer_table)
        answer_section.append(Spacer(1, 5))
        
        # Add each answer section to the content
        content.extend(answer_section)
    
    # Build the PDF with careful error handling
    try:
        pdf.build(content)
    except Exception as e:
        # If there's still an error, attempt a simpler format
        simple_content = create_simple_format(document, questions, styles)
        pdf.build(simple_content)

def create_simple_format(document, questions, styles):
    """
    Create a simpler format PDF without tables in case the main format fails
    This is a fallback to ensure we can always generate something
    """
    content = []
    
    # Add title
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=14,
        alignment=TA_CENTER
    )
    content.append(Paragraph(f"Questions for: {document.title}", title_style))
    content.append(Spacer(1, 20))
    
    # Add questions
    for i, question in enumerate(questions):
        content.append(Paragraph(f"<b>Question {i+1}:</b>", styles['Heading3']))
        content.append(Paragraph(question.content, styles['Normal']))
        content.append(Spacer(1, 5))
        
        # Add options for multiple choice
        if question.question_type == 'multiple_choice' and question.options:
            for j, option in enumerate(question.options):
                option_letter = string.ascii_uppercase[j]
                content.append(Paragraph(f"{option_letter}. {option}", styles['Normal']))
        
        content.append(Spacer(1, 10))
    
    # Add page break before answers
    content.append(PageBreak())
    
    # Add answers
    content.append(Paragraph("Answers:", styles['Heading2']))
    content.append(Spacer(1, 10))
    
    for i, question in enumerate(questions):
        if question.question_type == 'multiple_choice' and question.options:
            try:
                correct_index = question.options.index(question.answer)
                answer_text = f"{string.ascii_uppercase[correct_index]}. {question.answer}"
            except (ValueError, IndexError):
                answer_text = question.answer
        else:
            answer_text = question.answer
        
        content.append(Paragraph(f"<b>Q{i+1}:</b> {answer_text}", styles['Normal']))
        content.append(Spacer(1, 5))
    
    return content