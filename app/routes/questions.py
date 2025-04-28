import os
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, send_file, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models.document import Document
from app.models.question import Question
from app.forms.document_forms import DocumentUploadForm
from app.forms.question_forms import QuestionGeneratorForm
from app.utils.document_processor import process_document
from app.utils.question_generator import generate_questions
from app.utils.pdf_exporter import export_to_pdf
import tempfile

questions = Blueprint('questions', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@questions.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_document():
    """Document upload route"""
    form = DocumentUploadForm()
    
    if form.validate_on_submit():
        file = form.document.data
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.root_path, 'static/uploads', filename)
            
            # Ensure the upload directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            file.save(file_path)
            
            # Extract text from document
            document_text = process_document(file_path)
            
            # Save document to database
            document = Document(
                title=form.title.data,
                filename=filename,
                file_path=file_path,
                content=document_text,
                user_id=current_user.id
            )
            
            db.session.add(document)
            db.session.commit()
            
            flash('Document uploaded successfully!', 'success')
            return redirect(url_for('questions.generate_questions_form', document_id=document.id))
        else:
            flash('Invalid file format. Please upload a PDF or Word document.', 'danger')
    
    return render_template('questions/upload.html', form=form)

@questions.route('/documents')
@login_required
def list_documents():
    """List user's documents"""
    documents = Document.query.filter_by(user_id=current_user.id).all()
    return render_template('questions/documents.html', documents=documents)

@questions.route('/documents/<int:document_id>')
@login_required
def view_document(document_id):
    """View document details"""
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the current user
    if document.user_id != current_user.id:
        flash('You do not have permission to view this document.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    return render_template('questions/view_document.html', document=document)

@questions.route('/document/<int:document_id>/delete', methods=['POST'])
@login_required
def delete_document(document_id):
    document = Document.query.get_or_404(document_id)
    
    # Verify ownership
    if document.user_id != current_user.id:
        flash('You do not have permission to delete this document.', 'error')
        return redirect(url_for('questions.list_documents'))
    
    try:
        # Delete associated questions first
        Question.query.filter_by(document_id=document.id).delete()
        
        # Delete the document
        db.session.delete(document)
        db.session.commit()
        
        flash('Document and associated questions have been deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while deleting the document: {str(e)}', 'error')
    
    return redirect(url_for('questions.list_documents'))

@questions.route('/documents/<int:document_id>/generate', methods=['GET', 'POST'])
@login_required
def generate_questions_form(document_id):
    document = Document.query.get_or_404(document_id)
    
    # Verify the document belongs to the current user
    if document.user_id != current_user.id:
        flash('Access denied. This resource does not belong to you.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    form = QuestionGeneratorForm()
    
    # Check Gemini availability silently in the background
    try:
        from app.utils.gemini_validator import test_gemini_connection
        gemini_available = test_gemini_connection()
    except:
        gemini_available = False
    
    if form.validate_on_submit():
        try:
            print(f"Generating questions for document {document_id} with options: {form.data}")
            
            # Always try to use Gemini first
            use_gemini = True
            
            print(f"Using Gemini: {use_gemini}")
            
            generated_questions = generate_questions(
                document.content,
                num_questions=form.num_questions.data,
                question_types=form.question_type.data,
                difficulty=form.difficulty.data,
                use_gemini=use_gemini
            )
            
            # Debug log the generated questions
            print(f"Generated {len(generated_questions)} questions")
            
            # Check if questions were actually generated
            if not generated_questions or len(generated_questions) == 0:
                flash('No questions could be generated. The document may be too short or in an unsupported format.', 'warning')
                return redirect(url_for('questions.generate_questions_form', document_id=document_id))
            
            # Convert complex objects to simple dictionaries for session storage
            serializable_questions = []
            for q in generated_questions:
                # Create a simplified dict with only the necessary fields
                serializable_q = {
                    'question': q.get('question', ''),
                    'answer': q.get('answer', ''),
                    'type': q.get('type', 'structured'),
                    'difficulty': q.get('difficulty', 'medium'),
                    'context': q.get('context', '')
                }
                
                # Handle options for multiple choice questions
                if 'options' in q:
                    serializable_q['options'] = q['options']
                
                serializable_questions.append(serializable_q)
            
            # Save to session and ensure it's saved immediately
            session['generated_questions'] = serializable_questions
            session.modified = True
            
            print(f"Questions stored in session, redirecting to review page")
            return redirect(url_for('questions.review_questions', document_id=document_id))
            
        except Exception as e:
            print(f"Exception during question generation: {str(e)}")
            flash(f'Error generating questions. Please try again later.', 'danger')
    
    # Don't pass gemini_available to the template
    return render_template('questions/generate.html', form=form, document=document)

@questions.route('/documents/<int:document_id>/questions')
@login_required
def view_questions(document_id):
    """View generated questions"""
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the current user
    if document.user_id != current_user.id:
        flash('You do not have permission to view questions for this document.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    questions = Question.query.filter_by(document_id=document.id).all()
    
    return render_template('questions/view_questions.html', document=document, questions=questions)

@questions.route('/questions/<int:question_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_question(question_id):
    """Edit question"""
    question = Question.query.get_or_404(question_id)
    
    # Check if the question belongs to the current user
    if question.user_id != current_user.id:
        flash('You do not have permission to edit this question.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    if request.method == 'POST':
        question.content = request.form.get('content')
        question.answer = request.form.get('answer')
        
        if question.question_type == 'multiple_choice':
            options = []
            for i in range(4):  # Assuming 4 options for multiple choice
                option = request.form.get(f'option_{i}')
                if option:
                    options.append(option)
            question.options = options
        
        db.session.commit()
        
        flash('Question updated successfully!', 'success')
        return redirect(url_for('questions.view_questions', document_id=question.document_id))
    
    return render_template('questions/edit_question.html', question=question)

@questions.route('/questions/<int:question_id>/delete', methods=['POST'])
@login_required
def delete_question(question_id):
    """Delete question"""
    question = Question.query.get_or_404(question_id)
    
    # Check if the question belongs to the current user
    if question.user_id != current_user.id:
        flash('You do not have permission to delete this question.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    document_id = question.document_id
    
    db.session.delete(question)
    db.session.commit()
    
    flash('Question deleted successfully!', 'success')
    return redirect(url_for('questions.view_questions', document_id=document_id))

@questions.route('/documents/<int:document_id>/export', methods=['GET'])
@login_required
def export_questions(document_id):
    """Export questions to PDF"""
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the current user
    if document.user_id != current_user.id:
        flash('You do not have permission to export questions for this document.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    questions = Question.query.filter_by(document_id=document.id).all()
    
    if not questions:
        flash('No questions to export.', 'warning')
        return redirect(url_for('questions.view_questions', document_id=document.id))
    
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        pdf_path = temp_file.name
    
    # Generate PDF
    export_to_pdf(document, questions, pdf_path)
    
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f"{document.title}_questions.pdf",
        mimetype='application/pdf'
    )

@questions.route('/documents/<int:document_id>/review', methods=['GET', 'POST'])
@login_required
def review_questions(document_id):
    """
    Review generated questions before saving them to the database
    """
    document = Document.query.get_or_404(document_id)
    
    # Check if user has access to this document
    if document.user_id != current_user.id:
        flash('You do not have access to this document', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    # Get generated questions from session
    generated_questions = session.get('generated_questions', [])
    
    print(f"Review page: Found {len(generated_questions)} questions in session")
    
    if not generated_questions:
        flash('No questions to review. Please generate questions first.', 'warning')
        return redirect(url_for('questions.generate_questions_form', document_id=document_id))
    
    if request.method == 'POST':
        # Save the questions to the database
        questions_saved = 0
        
        for q_data in generated_questions:
            try:
                # Map the fields correctly from generation format to database model
                question = Question(
                    # Use 'question' field from generated data and map to 'content' field in model
                    content=q_data.get('question', ''),
                    answer=q_data.get('answer', ''),
                    document_id=document_id,
                    user_id=current_user.id,
                    question_type=q_data.get('type', 'structured'),
                    difficulty=q_data.get('difficulty', 'medium'),
                    options=q_data.get('options', [])
                )
                
                db.session.add(question)
                questions_saved += 1
            except Exception as e:
                print(f"Error processing question: {str(e)}")
                continue
        
        try:
            db.session.commit()
            # Clear the session after saving
            session.pop('generated_questions', None)
            flash(f'Successfully saved {questions_saved} questions.', 'success')
            return redirect(url_for('questions.view_questions', document_id=document_id))
        except Exception as e:
            db.session.rollback()
            print(f"Database error when saving questions: {str(e)}")
            flash(f'Error saving questions: {str(e)}', 'danger')
    
    return render_template(
        'questions/review_questions.html', 
        document=document,
        questions=generated_questions
    )