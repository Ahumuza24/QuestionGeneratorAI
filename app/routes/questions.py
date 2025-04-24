import os
from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, send_file, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import db
from app.models.document import Document
from app.models.question import Question
from app.forms.document_forms import DocumentUploadForm, QuestionGenerationForm, TrainingQuestionForm
from app.utils.document_processor import process_document
from app.utils.question_generator import generate_questions, train_model_on_examples, prepare_examples_from_question_paper, load_question_papers_from_directory, parse_questions_from_pdf, extract_multiple_choice_from_pdf
from app.utils.pdf_exporter import export_to_pdf
import tempfile
import json

questions = Blueprint('questions', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'json'}

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
    """Question generation form"""
    document = Document.query.get_or_404(document_id)
    
    # Check if the document belongs to the current user
    if document.user_id != current_user.id:
        flash('You do not have permission to generate questions for this document.', 'danger')
        return redirect(url_for('questions.list_documents'))
    
    form = QuestionGenerationForm()
    
    if form.validate_on_submit():
        # Generate questions
        generated_questions = generate_questions(
            document.content,
            form.num_questions.data,
            form.question_type.data,
            form.difficulty.data
        )
        
        # Save questions to database
        for q in generated_questions:
            question = Question(
                content=q['question'],
                answer=q['answer'],
                options=q.get('options', None),  # Only for multiple choice
                question_type=form.question_type.data,
                difficulty=form.difficulty.data,
                document_id=document.id,
                user_id=current_user.id
            )
            db.session.add(question)
        
        db.session.commit()
        
        flash(f'{len(generated_questions)} questions generated successfully!', 'success')
        return redirect(url_for('questions.view_questions', document_id=document.id))
    
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

@questions.route('/train-model', methods=['GET', 'POST'])
@login_required
def train_model():
    """Train question generation model on example question papers"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'question_paper' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
            
        file = request.files['question_paper']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if file and file.filename.endswith('.json'):
            filename = secure_filename(file.filename)
            
            # Create directory for training data if it doesn't exist
            training_dir = os.path.join(current_app.root_path, 'static/training_data')
            os.makedirs(training_dir, exist_ok=True)
            
            file_path = os.path.join(training_dir, filename)
            file.save(file_path)
            
            try:
                # Parse the uploaded question paper
                examples = prepare_examples_from_question_paper(file_path)
                
                if len(examples) == 0:
                    flash('No valid examples found in the uploaded file. Please check the format.', 'danger')
                    return redirect(request.url)
                
                # Get training parameters
                epochs = int(request.form.get('epochs', 3))
                batch_size = int(request.form.get('batch_size', 8))
                
                # Start training in a background thread to avoid blocking the request
                # In a production app, this would be better handled with a task queue like Celery
                import threading
                
                def train_in_background():
                    train_model_on_examples(examples, epochs=epochs, batch_size=batch_size)
                    
                training_thread = threading.Thread(target=train_in_background)
                training_thread.daemon = True
                training_thread.start()
                
                flash(f'Training started with {len(examples)} examples. This may take some time.', 'success')
                return redirect(url_for('questions.train_model'))
                
            except Exception as e:
                flash(f'Error processing training data: {str(e)}', 'danger')
                return redirect(request.url)
        else:
            flash('Invalid file format. Please upload a JSON file.', 'danger')
            return redirect(request.url)
            
    # GET request - show the upload form
    # Check if there are existing training files
    training_dir = os.path.join(current_app.root_path, 'static/training_data')
    existing_files = []
    
    if os.path.exists(training_dir):
        existing_files = [f for f in os.listdir(training_dir) if f.endswith('.json')]
        
    return render_template('questions/train_model.html', existing_files=existing_files)

@questions.route('/run-training', methods=['POST'])
@login_required
def run_training():
    """Run training on existing files"""
    training_dir = os.path.join(current_app.root_path, 'static/training_data')
    
    if not os.path.exists(training_dir):
        flash('No training data directory found', 'danger')
        return redirect(url_for('questions.train_model'))
        
    # Load all examples from the training directory
    all_examples = load_question_papers_from_directory(training_dir)
    
    if len(all_examples) == 0:
        flash('No valid examples found in training data', 'danger')
        return redirect(url_for('questions.train_model'))
        
    # Get training parameters
    epochs = int(request.form.get('epochs', 3))
    batch_size = int(request.form.get('batch_size', 8))
    
    # Start training in a background thread
    import threading
    
    def train_in_background():
        train_model_on_examples(all_examples, epochs=epochs, batch_size=batch_size)
        
    training_thread = threading.Thread(target=train_in_background)
    training_thread.daemon = True
    training_thread.start()
    
    flash(f'Training started with {len(all_examples)} examples from {len(os.listdir(training_dir))} files. This may take some time.', 'success')
    return redirect(url_for('questions.train_model'))

@questions.route('/create-example-paper', methods=['GET', 'POST'])
@login_required
def create_example_paper():
    """Convert generated questions into a training example"""
    if request.method == 'POST':
        document_id = request.form.get('document_id')
        
        if not document_id:
            flash('No document selected', 'danger')
            return redirect(url_for('questions.list_documents'))
            
        document = Document.query.get_or_404(document_id)
        
        # Verify ownership
        if document.user_id != current_user.id:
            flash('You do not have permission to access this document', 'danger')
            return redirect(url_for('questions.list_documents'))
            
        # Get all questions for this document
        questions_list = Question.query.filter_by(document_id=document.id).all()
        
        if not questions_list:
            flash('No questions found for this document', 'danger')
            return redirect(url_for('questions.view_document', document_id=document.id))
            
        # Create training examples
        examples = []
        for q in questions_list:
            example = {
                'context': q.context if hasattr(q, 'context') else document.content,
                'question': q.content,
                'answer': q.answer,
                'type': q.question_type
            }
            
            if q.question_type == 'multiple_choice' and q.options:
                example['options'] = q.options
                
            examples.append(example)
            
        # Create directory for training data if it doesn't exist
        training_dir = os.path.join(current_app.root_path, 'static/training_data')
        os.makedirs(training_dir, exist_ok=True)
        
        # Save as JSON file
        file_name = f"{document.title.replace(' ', '_')}_examples.json"
        file_path = os.path.join(training_dir, secure_filename(file_name))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2)
            
        flash(f'Created training example file with {len(examples)} questions', 'success')
        return redirect(url_for('questions.train_model'))
        
    # GET request - show documents to choose from
    documents = Document.query.filter_by(user_id=current_user.id).all()
    return render_template('questions/create_example.html', documents=documents)

@questions.route('/manual-training-examples', methods=['GET', 'POST'])
@login_required
def manual_training_examples():
    """Manually create training examples"""
    form = TrainingQuestionForm()
    
    # Get existing examples from session or initialize empty list
    session_examples = request.args.get('edit_file')
    examples = []
    file_path = None
    
    if session_examples:
        # We're editing an existing file
        training_dir = os.path.join(current_app.root_path, 'static/training_data')
        file_path = os.path.join(training_dir, secure_filename(session_examples))
        
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    examples = json.load(f)
        except Exception as e:
            flash(f'Error loading examples: {str(e)}', 'danger')
            examples = []
    
    if form.validate_on_submit():
        new_example = {
            'context': form.context.data,
            'question': form.question.data,
            'answer': form.answer.data,
            'type': form.question_type.data
        }
        
        examples.append(new_example)
        
        # Create directory for training data if it doesn't exist
        training_dir = os.path.join(current_app.root_path, 'static/training_data')
        os.makedirs(training_dir, exist_ok=True)
        
        # If we're editing an existing file, use that name, otherwise create a new one
        if not file_path:
            file_name = f"manual_examples_{len(os.listdir(training_dir)) + 1}.json"
            file_path = os.path.join(training_dir, file_name)
        
        # Save to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2)
        
        flash('Training example added successfully!', 'success')
        
        # Redirect to the same page with the file parameter to continue adding
        return redirect(url_for('questions.manual_training_examples', edit_file=os.path.basename(file_path)))
    
    # List existing training files for selection
    training_dir = os.path.join(current_app.root_path, 'static/training_data')
    existing_files = []
    
    if os.path.exists(training_dir):
        existing_files = [f for f in os.listdir(training_dir) if f.endswith('.json')]
    
    return render_template('questions/manual_examples.html', 
                          form=form, 
                          examples=examples, 
                          existing_files=existing_files,
                          current_file=os.path.basename(file_path) if file_path else None)

@questions.route('/pdf-to-training', methods=['GET', 'POST'])
@login_required
def pdf_to_training():
    """Extract training examples directly from PDF files"""
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'pdf_file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
            
        file = request.files['pdf_file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
            
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            
            # Create a temporary directory to store the uploaded file
            temp_dir = os.path.join(current_app.root_path, 'static/temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            temp_path = os.path.join(temp_dir, filename)
            file.save(temp_path)
            
            try:
                # Parse questions from the PDF file
                questions_list = parse_questions_from_pdf(temp_path)
                
                # Also try to extract multiple choice questions
                mc_questions = extract_multiple_choice_from_pdf(temp_path)
                
                # Combine the results
                all_questions = questions_list + mc_questions
                
                if len(all_questions) == 0:
                    flash('No questions could be extracted from the PDF. Please check the format or use manual entry.', 'warning')
                    return redirect(request.url)
                
                # Create directory for training data if it doesn't exist
                training_dir = os.path.join(current_app.root_path, 'static/training_data')
                os.makedirs(training_dir, exist_ok=True)
                
                # Save as JSON file
                base_filename = os.path.splitext(filename)[0]
                file_name = f"{base_filename}_pdf_examples.json"
                file_path = os.path.join(training_dir, secure_filename(file_name))
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(all_questions, f, indent=2)
                
                # Clean up the temporary file
                os.remove(temp_path)
                
                # Redirect to edit these examples
                flash(f'Successfully extracted {len(all_questions)} questions from the PDF. You can now review and edit them.', 'success')
                return redirect(url_for('questions.manual_training_examples', edit_file=os.path.basename(file_path)))
                
            except Exception as e:
                flash(f'Error processing PDF: {str(e)}', 'danger')
                return redirect(request.url)
                
            finally:
                # Make sure we clean up the temp file if it exists
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        else:
            flash('Invalid file format. Please upload a PDF file.', 'danger')
            return redirect(request.url)
    
    # GET request - show the upload form
    return render_template('questions/pdf_to_training.html') 