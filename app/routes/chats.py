import os
import json
from datetime import datetime
from flask import Blueprint, render_template, url_for, flash, redirect, request, abort, jsonify, current_app
from flask_login import login_required, current_user
from app import db
from app.models.course import Course
from app.models.document import Document
from app.models.chat import Chat, Message
from app.forms.chat_forms import ChatForm, MessageForm, SaveMessageForm, ExportForm, QuestionGenerationForm
from app.utils.question_generator import generate_questions_with_context, generate_questions_from_context
from app.utils.pdf_exporter import export_chat_to_pdf
from markupsafe import Markup
import markdown

# Define blueprint
chats = Blueprint('chats', __name__)


@chats.route('/courses/<int:course_id>/chats/new', methods=['GET', 'POST'])
@login_required
def new_chat(course_id):
    """Create a new chat for a course"""
    course = Course.query.get_or_404(course_id)
    
    # Ensure the current user owns this course
    if course.user_id != current_user.id:
        abort(403)
    
    form = ChatForm()
    
    if form.validate_on_submit():
        chat = Chat(
            title=form.title.data,
            user_id=current_user.id,
            course_id=course_id
        )
        
        db.session.add(chat)
        db.session.commit()
        
        # Add a welcome message from the assistant
        welcome_message = Message(
            content=f"Hello! I'm your AI assistant for the course '{course.name}'. How can I help you today?",
            role='assistant',
            chat_id=chat.id
        )
        
        db.session.add(welcome_message)
        db.session.commit()
        
        # If chat is created for question generation, redirect to question generation
        if request.args.get('action') == 'generate_questions':
            return redirect(url_for('chats.view_chat', course_id=course_id, chat_id=chat.id, action='generate_questions'))
        
        flash('Chat created successfully!', 'success')
        return redirect(url_for('chats.view_chat', course_id=course_id, chat_id=chat.id))
    
    # Auto-generate title if not specified
    if not form.title.data:
        form.title.data = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    return render_template(
        'chats/new.html', 
        form=form, 
        course=course,
        title=f"New Chat - {course.name}"
    )


@chats.route('/courses/<int:course_id>/chats/<int:chat_id>', methods=['GET'])
@login_required
def view_chat(course_id, chat_id):
    """View a specific chat"""
    course = Course.query.get_or_404(course_id)
    chat = Chat.query.get_or_404(chat_id)
    
    # Ensure the chat belongs to the specified course
    if chat.course_id != course_id:
        abort(404)
    
    # Ensure the current user owns this chat
    if chat.user_id != current_user.id:
        abort(403)
    
    # Get all messages for this chat
    messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at).all()
    
    # Get all chats for the course (for sidebar)
    course_chats = Chat.query.filter_by(course_id=course_id).order_by(Chat.updated_at.desc()).all()
    
    # Get all documents for the course (for sidebar)
    course_documents = Document.query.filter_by(course_id=course_id).all()
    
    # Forms for chat functionality
    message_form = MessageForm()
    save_form = SaveMessageForm()
    export_form = ExportForm()
    question_form = QuestionGenerationForm()
    
    # If action is generate_questions, open the question generation modal
    show_question_modal = True if request.args.get('action') == 'generate_questions' else False
    
    return render_template(
        'chats/view.html', 
        course=course,
        chat=chat,
        messages=messages,
        course_chats=course_chats,
        course_documents=course_documents,
        message_form=message_form,
        save_form=save_form,
        export_form=export_form,
        question_form=question_form,
        show_question_modal=show_question_modal,
        title=f"{chat.title} - {course.name}"
    )


@chats.route('/courses/<int:course_id>/chats/<int:chat_id>/send', methods=['POST'])
@login_required
def send_message(course_id, chat_id):
    """Send a message in a chat"""
    course = Course.query.get_or_404(course_id)
    chat = Chat.query.get_or_404(chat_id)
    
    # Ensure the chat belongs to the specified course and user
    if chat.course_id != course_id or chat.user_id != current_user.id:
        abort(403)
    
    form = MessageForm()
    
    if form.validate_on_submit():
        # Create user message
        user_message = Message(
            content=form.content.data,
            role='user',
            chat_id=chat_id
        )
        db.session.add(user_message)
        db.session.commit()
        
        # Get all course documents for context
        documents = Document.query.filter_by(course_id=course_id).all()
        context = " ".join([doc.content for doc in documents])
        
        # Get chat history for context
        messages = Message.query.filter_by(chat_id=chat_id).order_by(Message.created_at).all()
        chat_history = []
        for msg in messages:
            chat_history.append({
                'role': 'user' if msg.is_user else 'assistant',
                'content': msg.content
            })
        
        # Generate AI response
        response = generate_ai_response(form.content.data, context, chat_history)
        
        # Create assistant message
        ai_message = Message(
            content=response,
            role='assistant',
            chat_id=chat_id
        )
        db.session.add(ai_message)
        
        # Update chat's last activity time
        chat.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        # If it's an AJAX request, return JSON response
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({
                'user_message': {
                    'id': user_message.id,
                    'content': user_message.content,
                    'created_at': user_message.created_at.strftime('%H:%M')
                },
                'ai_message': {
                    'id': ai_message.id,
                    'content': ai_message.content,
                    'created_at': ai_message.created_at.strftime('%H:%M')
                }
            })
        
        return redirect(url_for('chats.view_chat', course_id=course_id, chat_id=chat_id))
    
    # If form validation fails
    return redirect(url_for('chats.view_chat', course_id=course_id, chat_id=chat_id))


@chats.route('/courses/<int:course_id>/chats/<int:chat_id>/save-message', methods=['POST'])
@login_required
def save_message(course_id, chat_id):
    """Save a message with specific tags and type"""
    course = Course.query.get_or_404(course_id)
    chat = Chat.query.get_or_404(chat_id)
    
    # Ensure the chat belongs to the specified course and user
    if chat.course_id != course_id or chat.user_id != current_user.id:
        abort(403)
    
    form = SaveMessageForm()
    
    if form.validate_on_submit():
        message_id = form.message_id.data
        message = Message.query.get_or_404(message_id)
        
        # Make sure the message belongs to the current chat
        if message.chat_id != chat_id:
            abort(403)
        
        # Update message
        message.is_saved = True
        message.save_type = form.save_type.data
        message.tags = form.tags.data
        
        db.session.commit()
        
        flash('Message saved successfully!', 'success')
        
    return redirect(url_for('chats.view_chat', course_id=course_id, chat_id=chat_id))


@chats.route('/courses/<int:course_id>/chats/<int:chat_id>/generate-questions', methods=['POST'])
@login_required
def generate_questions_in_chat(course_id, chat_id):
    """Generate questions within a chat"""
    course = Course.query.get_or_404(course_id)
    chat = Chat.query.get_or_404(chat_id)
    
    # Ensure the chat belongs to the specified course and user
    if chat.course_id != course_id or chat.user_id != current_user.id:
        abort(403)
    
    form = QuestionGenerationForm()
    
    if form.validate_on_submit():
        # Create user message
        user_message = Message(
            content=f"Generate {form.num_questions.data} {form.difficulty.data} {form.question_type.data} questions based on the course materials.",
            role='user',
            chat_id=chat_id
        )
        db.session.add(user_message)
        
        # Get all course documents for context
        documents = Document.query.filter_by(course_id=course_id).all()
        doc_contents = [doc.content for doc in documents]
        
        # Generate questions
        questions = generate_questions_with_context(
            num_questions=int(form.num_questions.data),
            difficulty=form.difficulty.data,
            question_type=form.question_type.data,
            context=doc_contents
        )
        
        # Create AI message with generated questions
        ai_message = Message(
            content=questions,
            role='assistant',
            chat_id=chat_id,
            is_saved=True,
            save_type='question',
            tags=f"generated,{form.difficulty.data},{form.question_type.data}"
        )
        db.session.add(ai_message)
        
        # Update chat's last activity time
        chat.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        flash('Questions generated successfully!', 'success')
        
    return redirect(url_for('chats.view_chat', course_id=course_id, chat_id=chat_id))


@chats.route('/courses/<int:course_id>/chats/<int:chat_id>/export', methods=['POST'])
@login_required
def export_chat(course_id, chat_id):
    """Export chat content"""
    course = Course.query.get_or_404(course_id)
    chat = Chat.query.get_or_404(chat_id)
    
    # Ensure the chat belongs to the specified course and user
    if chat.course_id != course_id or chat.user_id != current_user.id:
        abort(403)
    
    form = ExportForm()
    
    if form.validate_on_submit():
        export_type = form.export_type.data
        content_selection = form.content_selection.data
        tags_filter = form.tags_filter.data
        
        # Get all messages for this chat
        query = Message.query.filter_by(chat_id=chat_id)
        
        # Filter messages based on selected content
        if content_selection == 'saved':
            query = query.filter_by(is_saved=True)
        elif content_selection == 'questions':
            query = query.filter_by(is_saved=True, save_type='question')
        elif content_selection == 'answers':
            query = query.filter_by(is_saved=True, save_type='answer')
        
        # Filter by tags if provided
        if tags_filter:
            tags_list = [tag.strip() for tag in tags_filter.split(',')]
            for tag in tags_list:
                query = query.filter(Message.tags.like(f"%{tag}%"))
        
        messages = query.order_by(Message.created_at).all()
        
        # If no messages match the filter
        if not messages:
            flash('No messages match the selected filters.', 'warning')
            return redirect(url_for('chats.view_chat', course_id=course_id, chat_id=chat_id))
        
        # Generate export
        if export_type == 'pdf':
            pdf_path = export_chat_to_pdf(
                chat=chat, 
                messages=messages, 
                course=course
            )
            return redirect(url_for('static', filename=f"exports/{os.path.basename(pdf_path)}"))
            
        elif export_type == 'docx':
            # Handle docx export (similar to pdf_exporter)
            flash('DOCX export feature coming soon!', 'info')
            return redirect(url_for('chats.view_chat', course_id=course_id, chat_id=chat_id))
            
        elif export_type == 'json':
            # Create JSON export
            export_data = {
                'chat_title': chat.title,
                'course_name': course.name,
                'export_date': datetime.now().isoformat(),
                'messages': []
            }
            
            for msg in messages:
                export_data['messages'].append({
                    'role': 'user' if msg.is_user else 'assistant',
                    'content': msg.content,
                    'timestamp': msg.created_at.isoformat(),
                    'saved': msg.is_saved,
                    'save_type': msg.save_type,
                    'tags': msg.tags
                })
            
            # Return as a downloadable JSON file
            return jsonify(export_data)
    
    # If form validation fails
    return redirect(url_for('chats.view_chat', course_id=course_id, chat_id=chat_id))


@chats.route('/courses/<int:course_id>/chats/<int:chat_id>/delete', methods=['POST'])
@login_required
def delete_chat(course_id, chat_id):
    """Delete a chat"""
    course = Course.query.get_or_404(course_id)
    chat = Chat.query.get_or_404(chat_id)
    
    # Ensure the chat belongs to the specified course
    if chat.course_id != course_id:
        abort(404)
    
    # Ensure the current user owns this chat
    if chat.user_id != current_user.id:
        abort(403)
    
    db.session.delete(chat)
    db.session.commit()
    
    flash('Chat deleted successfully!', 'success')
    return redirect(url_for('courses.view_course', course_id=course_id))


# Helper function to generate AI responses
def generate_ai_response(user_message, context, chat_history):
    """Generate AI response to user message with context"""
    # In a real implementation, this would call an API like OpenAI's GPT
    # For now, we'll use a simple placeholder response
    
    # This is where you would integrate with your AI model
    # Example API call to GPT or similar model
    try:
        from app.utils.question_generator import get_ai_response
        response = get_ai_response(user_message, context, chat_history)
        return response
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return """I'm having trouble connecting to the AI service. 
Please try again later or contact support if the problem persists.

In the meantime, you can try:
1. Refreshing the page
2. Checking your internet connection
3. Making your query more specific"""


# Register Jinja filter
@chats.app_template_filter('markdown')
def render_markdown(content):
    """Convert markdown to HTML"""
    if content:
        # Convert markdown to HTML
        html = markdown.markdown(content, extensions=['fenced_code', 'tables'])
        return Markup(html)
    return ''