import os
import uuid
from datetime import datetime
from flask import Blueprint, render_template, url_for, flash, redirect, request, abort, current_app, send_file, after_this_request
from flask_login import login_required, current_user
from app import db
from app.models.course import Course
from app.models.document import Document
from app.models.chat import Chat, Message
from app.forms.course_forms import CourseForm, DocumentUploadForm, BatchUploadForm, DocumentSearchForm
from werkzeug.utils import secure_filename
from app.utils.document_processor import process_document
import tempfile
from bs4 import BeautifulSoup

# Remove pdfkit import since we're using ReportLab instead
# import pdfkit

# Define blueprint
courses = Blueprint('courses', __name__)


@courses.route('/courses')
@login_required
def list_courses():
    """Display all courses for the current user"""
    return render_template('courses/list.html', courses=current_user.courses)


@courses.route('/courses/new', methods=['GET', 'POST'])
@login_required
def create_course():
    """Create a new course"""
    form = CourseForm()
    if form.validate_on_submit():
        course = Course(
            name=form.name.data,
            description=form.description.data,
            user_id=current_user.id
        )
        db.session.add(course)
        db.session.commit()
        flash('Course created successfully!', 'success')
        return redirect(url_for('courses.view_course', course_id=course.id))
    return render_template('courses/create.html', form=form, title='New Course')


@courses.route('/courses/<int:course_id>')
@login_required
def view_course(course_id):
    """View a specific course"""
    course = Course.query.get_or_404(course_id)
    
    # Verify the course belongs to the current user
    if course.user_id != current_user.id:
        flash('Access denied. This course does not belong to you.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    # Initialize forms
    document_form = DocumentUploadForm()
    search_form = DocumentSearchForm()
    
    # If search was performed
    query = request.args.get('query', '')
    doc_type = request.args.get('document_type', 'all')
    
    if query:
        search_form.query.data = query
        search_form.document_type.data = doc_type
        
        # Filter documents based on search
        documents = Document.query.filter(Document.course_id == course_id)
        
        # Apply document type filter if not 'all'
        if doc_type != 'all':
            documents = documents.filter(Document.document_type == doc_type)
        
        # Apply text search filter
        documents = documents.filter(Document.title.contains(query) | Document.content.contains(query))
        
        # Get results
        documents = documents.all()
        
        # Override course.documents for this view
        course.documents = documents
    
    return render_template(
        'courses/view.html', 
        course=course, 
        document_form=document_form, 
        search_form=search_form
    )


@courses.route('/courses/<int:course_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_course(course_id):
    """Edit an existing course"""
    course = Course.query.get_or_404(course_id)
    
    # Verify the course belongs to the current user
    if course.user_id != current_user.id:
        flash('Access denied. This course does not belong to you.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    form = CourseForm()
    
    if form.validate_on_submit():
        course.name = form.name.data
        course.description = form.description.data
        course.updated_at = datetime.utcnow()
        db.session.commit()
        flash('Course updated successfully!', 'success')
        return redirect(url_for('courses.view_course', course_id=course.id))
    
    # Pre-populate the form with existing data
    elif request.method == 'GET':
        form.name.data = course.name
        form.description.data = course.description
    
    return render_template('courses/edit.html', form=form, title="Edit Course", course=course)


@courses.route('/courses/<int:course_id>/delete')
@login_required
def delete_course(course_id):
    """Delete a course"""
    course = Course.query.get_or_404(course_id)
    
    # Verify the course belongs to the current user
    if course.user_id != current_user.id:
        flash('Access denied. This course does not belong to you.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    db.session.delete(course)
    db.session.commit()
    flash('Course deleted successfully!', 'success')
    return redirect(url_for('courses.list_courses'))


@courses.route('/courses/<int:course_id>/upload', methods=['POST'])
@login_required
def upload_document(course_id):
    """Upload a document to a course"""
    course = Course.query.get_or_404(course_id)
    
    # Verify the course belongs to the current user
    if course.user_id != current_user.id:
        flash('Access denied. This course does not belong to you.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    form = DocumentUploadForm()
    if form.validate_on_submit():
        # Save the file
        file = form.document.data
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process document content
        content = process_document(file_path)
        
        # Create document record
        document = Document(
            title=form.title.data,
            filename=filename,
            file_path=file_path,
            content=content,
            document_type=form.document_type.data,
            user_id=current_user.id,
            course_id=course_id
        )
        
        db.session.add(document)
        db.session.commit()
        flash('Document uploaded successfully!', 'success')
    else:
        for field, errors in form.errors.items():
            for error in errors:
                flash(f"{getattr(form, field).label.text}: {error}", 'danger')
    
    # Redirect back to the course view page, showing the documents tab
    return redirect(url_for('courses.view_course', course_id=course_id) + '#documents')


@courses.route('/courses/<int:course_id>/batch-upload', methods=['GET', 'POST'])
@login_required
def batch_upload_documents(course_id):
    """Batch upload multiple documents to a course"""
    course = Course.query.get_or_404(course_id)
    
    # Check if the current user owns this course
    if course.user_id != current_user.id:
        abort(403)
    
    form = BatchUploadForm()
    
    if form.validate_on_submit():
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(current_app.root_path, 'static/uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Process all uploaded files
        documents_uploaded = 0
        
        for document_file in form.documents.data:
            if document_file:
                # Save the uploaded file
                filename = secure_filename(document_file.filename)
                file_path = os.path.join(uploads_dir, filename)
                document_file.save(file_path)
                
                # Process document content
                content = process_document(file_path)
                
                # Extract title from filename (remove extension)
                title = os.path.splitext(filename)[0]
                
                # Create document in database
                document = Document(
                    title=title,
                    filename=filename,
                    file_path=os.path.join('static/uploads', filename),
                    content=content,
                    document_type=form.document_type.data,
                    user_id=current_user.id,
                    course_id=course.id
                )
                
                db.session.add(document)
                documents_uploaded += 1
        
        if documents_uploaded > 0:
            db.session.commit()
            flash(f'{documents_uploaded} documents have been uploaded!', 'success')
        else:
            flash('No documents were uploaded.', 'warning')
            
        return redirect(url_for('courses.view_course', course_id=course.id))
    
    return render_template('courses/batch_upload.html', form=form, course=course)


@courses.route('/courses/<int:course_id>/document/<int:document_id>')
@login_required
def view_document(course_id, document_id):
    """View a specific document within a course"""
    course = Course.query.get_or_404(course_id)
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the course
    if document.course_id != course.id:
        abort(404)
    
    # Check if the current user owns this course
    if course.user_id != current_user.id:
        abort(403)
    
    return render_template('courses/view_document.html', course=course, document=document)


@courses.route('/courses/<int:course_id>/document/<int:document_id>/delete', methods=['POST'])
@login_required
def delete_document(course_id, document_id):
    """Delete a document from a course"""
    course = Course.query.get_or_404(course_id)
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the course
    if document.course_id != course.id:
        abort(404)
    
    # Check if the current user owns this course
    if course.user_id != current_user.id:
        abort(403)
    
    # Delete the document file if it exists
    file_path = os.path.join(current_app.root_path, document.file_path)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Delete the document from the database
    db.session.delete(document)
    db.session.commit()
    
    flash(f'Document "{document.title}" has been deleted!', 'success')
    return redirect(url_for('courses.view_course', course_id=course.id))


@courses.route('/courses/<int:course_id>/chats/new')
@login_required
def create_chat(course_id):
    """Create a new chat for a course"""
    course = Course.query.get_or_404(course_id)
    
    # Verify the course belongs to the current user
    if course.user_id != current_user.id:
        flash('Access denied. This course does not belong to you.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    # Create a new chat
    chat = Chat(
        title=f"Chat {len(course.chats) + 1}",
        user_id=current_user.id,
        course_id=course_id
    )
    
    db.session.add(chat)
    db.session.commit()
    
    # Redirect to the chat view
    return redirect(url_for('courses.view_chat', course_id=course_id, chat_id=chat.id))


@courses.route('/courses/<int:course_id>/chats/<int:chat_id>')
@login_required
def view_chat(course_id, chat_id):
    """View a specific chat"""
    course = Course.query.get_or_404(course_id)
    chat = Chat.query.get_or_404(chat_id)
    
    # Verify the course and chat belong to the current user
    if course.user_id != current_user.id or chat.user_id != current_user.id:
        flash('Access denied. This resource does not belong to you.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    # Verify the chat belongs to the course
    if chat.course_id != course_id:
        flash('This chat does not belong to the selected course.', 'danger')
        return redirect(url_for('courses.view_course', course_id=course_id))
    
    return render_template('chats/chat.html', course=course, chat=chat)


@courses.route('/courses/<int:course_id>/chats/<int:chat_id>/messages', methods=['POST'])
@login_required
def send_message(course_id, chat_id):
    course = Course.query.get_or_404(course_id)
    chat = Chat.query.get_or_404(chat_id)
    
    # Verify ownership
    if course.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    # Get message content from form
    content = request.form.get('message')
    
    if content:
        # Add user message
        user_message = Message(
            content=content,
            role='user',
            chat_id=chat.id
        )
        db.session.add(user_message)
        
        # Generate AI response by integrating with your question generator
        try:
            # Get course documents content to provide context
            course_docs = [doc.content for doc in course.documents]
            context = "\n\n".join(course_docs)
            
            # If context is too large, trim it
            if len(context) > 15000:
                context = context[:15000]
            
            # Import your Gemini integration
            from app.utils.gemini_validator import get_model
            import markdown
            
            model = get_model()
            if model:
                # Prepare the prompt with context and formatting instructions
                prompt = f"""
                You are a helpful teaching assistant for a course on {course.name}.
                
                Course context: {context}
                
                Respond to the following student question:
                {content}
                
                Provide a helpful, educational response. Use markdown formatting for your response:
                - Use **bold** for important concepts
                - Use bullet points for lists
                - Use numbered lists for step-by-step instructions
                - Use headings with # for sections if needed
                - Use code blocks for code examples if applicable
                
                Keep your response well-structured and easy to read.
                """
                
                # Get response from Gemini
                response = model.generate_content(prompt)
                
                # Convert markdown to HTML for better formatting
                ai_content = markdown.markdown(response.text)
            else:
                # Fallback if model is unavailable
                ai_content = "I'm sorry, but I couldn't process your request at this time. Please try again later."
        except Exception as e:
            current_app.logger.error(f"Error generating AI response: {str(e)}")
            ai_content = "I'm having trouble connecting to my knowledge base right now. Please try again shortly."
        
        # Add AI response message
        ai_message = Message(
            content=ai_content,
            role='assistant',
            chat_id=chat.id
        )
        db.session.add(ai_message)
        
        db.session.commit()
    
    return redirect(url_for('courses.view_chat', course_id=course_id, chat_id=chat_id))


@courses.route('/courses/<int:course_id>/chats/<int:chat_id>/rename', methods=['POST'])
@login_required
def rename_chat(course_id, chat_id):
    """Rename a chat"""
    course = Course.query.get_or_404(course_id)
    chat = Chat.query.get_or_404(chat_id)
    
    # Verify ownership
    if course.user_id != current_user.id or chat.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    # Update chat title
    new_title = request.form.get('title')
    if new_title:
        chat.title = new_title
        db.session.commit()
        flash('Chat renamed successfully!', 'success')
    
    return redirect(url_for('courses.view_chat', course_id=course_id, chat_id=chat_id))


@courses.route('/courses/<int:course_id>/chats/<int:chat_id>/delete')
@login_required
def delete_chat(course_id, chat_id):
    """Delete a chat"""
    course = Course.query.get_or_404(course_id)
    chat = Chat.query.get_or_404(chat_id)
    
    # Verify ownership
    if course.user_id != current_user.id or chat.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    # Delete the chat
    db.session.delete(chat)
    db.session.commit()
    flash('Chat deleted successfully!', 'success')
    
    return redirect(url_for('courses.view_course', course_id=course_id))


@courses.route('/courses/<int:course_id>/chats/<int:chat_id>/messages/<int:message_id>/export', methods=['GET'])
@login_required
def export_message_pdf(course_id, chat_id, message_id):
    # Verify course ownership
    course = Course.query.get_or_404(course_id)
    if course.user_id != current_user.id:
        flash('Access denied.', 'danger')
        return redirect(url_for('courses.list_courses'))
    
    # Get the message
    message = Message.query.get_or_404(message_id)
    
    # Verify the message belongs to the chat
    if message.chat_id != chat_id:
        flash('Message not found in this chat.', 'danger')
        return redirect(url_for('courses.view_chat', course_id=course_id, chat_id=chat_id))
    
    # Only allow exporting AI responses
    if message.role != 'assistant':
        flash('Only AI responses can be exported.', 'warning')
        return redirect(url_for('courses.view_chat', course_id=course_id, chat_id=chat_id))
    
    # Get the related user question (if available)
    user_question = Message.query.filter_by(
        chat_id=chat_id,
        role='user'
    ).filter(
        Message.created_at < message.created_at
    ).order_by(Message.created_at.desc()).first()
    
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
        pdf_path = tmp.name
    
    try:
        # Import ReportLab components
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from html import unescape
        import re
        from bs4 import BeautifulSoup
        
        # Function to parse HTML and convert to ReportLab flowables
        def html_to_flowables(html_content, body_style, header_style):
            soup = BeautifulSoup(html_content, 'html.parser')
            flowables = []
            
            # Process all elements
            for element in soup.children:
                if element.name == 'p':
                    flowables.append(Paragraph(element.get_text(), body_style))
                    flowables.append(Spacer(1, 6))
                    
                elif element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    flowables.append(Spacer(1, 12))
                    flowables.append(Paragraph(element.get_text(), header_style))
                    flowables.append(Spacer(1, 6))
                    
                elif element.name == 'ul':
                    items = []
                    for li in element.find_all('li'):
                        bullet_text = li.get_text().strip()
                        items.append(ListItem(Paragraph(bullet_text, body_style)))
                    flowables.append(ListFlowable(items, bulletType='bullet', start='•'))
                    flowables.append(Spacer(1, 6))
                    
                elif element.name == 'ol':
                    items = []
                    for i, li in enumerate(element.find_all('li')):
                        bullet_text = li.get_text().strip()
                        items.append(ListItem(Paragraph(bullet_text, body_style)))
                    flowables.append(ListFlowable(items, bulletType='bullet', start=1))
                    flowables.append(Spacer(1, 6))
                    
                elif element.name == 'pre':
                    code_style = ParagraphStyle(
                        'Code',
                        parent=body_style,
                        fontName='Courier',
                        fontSize=9,
                        leftIndent=20,
                        rightIndent=20,
                        backColor=colors.lightgrey
                    )
                    code_text = element.get_text()
                    flowables.append(Paragraph(code_text, code_style))
                    flowables.append(Spacer(1, 6))
                    
                elif element.name == 'blockquote':
                    quote_style = ParagraphStyle(
                        'Quote',
                        parent=body_style,
                        leftIndent=30,
                        rightIndent=30,
                        italic=True
                    )
                    flowables.append(Paragraph(element.get_text(), quote_style))
                    flowables.append(Spacer(1, 6))
                    
                elif element.string and element.string.strip():
                    # Plain text nodes
                    flowables.append(Paragraph(element.string.strip(), body_style))
                    flowables.append(Spacer(1, 6))
            
            return flowables
        
        # Set up the document
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get the sample stylesheet
        styles = getSampleStyleSheet()
        
        # Create custom styles with unique names
        title_style = ParagraphStyle(
            name='ChatTitle',
            parent=styles['Heading1'],
            alignment=TA_CENTER,
            fontSize=16,
            textColor=colors.darkblue
        )
        
        header_style = ParagraphStyle(
            name='ChatHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue
        )
        
        question_style = ParagraphStyle(
            name='ChatQuestionHeader',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.darkblue
        )
        
        body_style = ParagraphStyle(
            name='ChatBodyText',
            parent=styles['Normal'],
            fontSize=11,
            leading=14
        )
        
        footer_style = ParagraphStyle(
            name='ChatFooter',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey
        )
        
        bold_style = ParagraphStyle(
            name='ChatBoldText',
            parent=body_style,
            fontName='Helvetica-Bold'
        )
        
        # Prepare content
        content = []
        
        # Add title
        content.append(Paragraph(f"{course.name} - Chat Response", title_style))
        content.append(Spacer(1, 12))
        content.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", body_style))
        content.append(Spacer(1, 24))
        
        # Add question if exists
        if user_question:
            content.append(Paragraph("Question:", question_style))
            content.append(Spacer(1, 6))
            content.append(Paragraph(user_question.content, body_style))
            content.append(Spacer(1, 24))
        
        # Add AI response
        content.append(Paragraph("Response:", header_style))
        content.append(Spacer(1, 6))
        
        # Parse the HTML content and convert to ReportLab flowables
        content.extend(html_to_flowables(message.content, body_style, header_style))
        
        content.append(Spacer(1, 36))
        
        # Add footer
        content.append(Paragraph(f"This document was exported from {course.name} course chat.", footer_style))
        content.append(Paragraph(f"© {datetime.now().year} Question Generator AI", footer_style))
        
        # Build the PDF
        doc.build(content)
        
        # Generate a meaningful filename
        filename = f"Chat_Response_{course.name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        # Send the file to the user
        return send_file(
            pdf_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        # Log the error
        current_app.logger.error(f"PDF generation error: {str(e)}")
        flash('Error generating PDF. Please try again.', 'danger')
        return redirect(url_for('courses.view_chat', course_id=course_id, chat_id=chat_id))
    finally:
        # Clean up - delete the temp file after sending
        @after_this_request
        def remove_file(response):
            try:
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
            except Exception as e:
                current_app.logger.error(f"Error deleting temp file: {str(e)}")
            return response