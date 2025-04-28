from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, SelectField, RadioField, HiddenField
from wtforms.validators import DataRequired, Length, Optional

class ChatForm(FlaskForm):
    """Form for creating a new chat"""
    title = StringField('Chat Title', validators=[DataRequired(), Length(min=3, max=100)])
    submit = SubmitField('Start Chat')


class MessageForm(FlaskForm):
    """Form for sending messages in a chat"""
    content = TextAreaField('Message', validators=[DataRequired()])
    submit = SubmitField('Send')


class SaveMessageForm(FlaskForm):
    """Form for saving a message with metadata"""
    message_id = HiddenField('Message ID', validators=[DataRequired()])
    save_type = RadioField('Save As', choices=[
        ('question', 'Question'),
        ('answer', 'Answer'),
        ('note', 'Note')
    ], default='note')
    tags = StringField('Tags', validators=[Optional(), Length(max=200)])
    submit = SubmitField('Save')


class ExportForm(FlaskForm):
    """Form for exporting chat content"""
    export_type = SelectField('Export Format', choices=[
        ('pdf', 'PDF Document'),
        ('docx', 'Word Document'),
        ('json', 'JSON Data')
    ], default='pdf')
    
    content_selection = SelectField('Content to Export', choices=[
        ('all', 'All Messages'),
        ('saved', 'Only Saved Messages'),
        ('questions', 'Only Questions'),
        ('answers', 'Only Answers')
    ], default='all')
    
    tags_filter = StringField('Filter by Tags', validators=[Optional(), Length(max=200)])
    submit = SubmitField('Export')


class QuestionGenerationForm(FlaskForm):
    """Form for generating exam questions"""
    num_questions = SelectField('Number of Questions', choices=[
        ('3', '3 Questions'),
        ('5', '5 Questions'),
        ('10', '10 Questions')
    ], default='5')
    
    difficulty = SelectField('Difficulty', choices=[
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard')
    ], default='medium')
    
    question_type = SelectField('Question Type', choices=[
        ('multiple_choice', 'Multiple Choice'),
        ('structured', 'Open-ended'),
        ('both', 'Mixed')
    ], default='both')
    
    submit = SubmitField('Generate Questions')