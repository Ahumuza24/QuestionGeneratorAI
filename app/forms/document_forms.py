from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, IntegerField, SelectField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Length, NumberRange

class DocumentUploadForm(FlaskForm):
    """Document upload form"""
    title = StringField('Document Title', validators=[
        DataRequired(),
        Length(min=3, max=100)
    ])
    document = FileField('Upload Document', validators=[
        FileRequired(),
        FileAllowed(['pdf', 'docx', 'doc'], 'Only PDF and Word documents are allowed!')
    ])
    submit = SubmitField('Upload')

class QuestionGenerationForm(FlaskForm):
    """Question generation form"""
    num_questions = IntegerField('Number of Questions', validators=[
        DataRequired(),
        NumberRange(min=1, max=50, message='Please enter a number between 1 and 50')
    ], default=10)
    
    question_type = SelectField('Question Type', validators=[
        DataRequired()
    ], choices=[
        ('multiple_choice', 'Multiple Choice'),
        ('structured', 'Structured'),
        ('both', 'Both')
    ], default='both')
    
    difficulty = SelectField('Difficulty Level', validators=[
        DataRequired()
    ], choices=[
        ('easy', 'Easy'),
        ('medium', 'Medium'),
        ('hard', 'Hard'),
        ('mixed', 'Mixed')
    ], default='medium')
    
    submit = SubmitField('Generate Questions')

class TrainingQuestionForm(FlaskForm):
    """Form for manually creating training questions"""
    context = TextAreaField('Context', validators=[
        DataRequired(),
        Length(min=10, max=2000, message='Context must be between 10 and 2000 characters')
    ])
    
    question = StringField('Question', validators=[
        DataRequired(),
        Length(min=5, max=200, message='Question must be between 5 and 200 characters')
    ])
    
    answer = TextAreaField('Answer', validators=[
        DataRequired(),
        Length(min=1, max=500, message='Answer must be between 1 and 500 characters')
    ])
    
    question_type = SelectField('Question Type', validators=[
        DataRequired()
    ], choices=[
        ('structured', 'Structured Question'),
        ('multiple_choice', 'Multiple Choice')
    ], default='structured')
    
    submit = SubmitField('Save Question') 