from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, TextAreaField, SelectField, MultipleFileField, SubmitField
from wtforms.validators import DataRequired, Length, Optional


class CourseForm(FlaskForm):
    """Form for creating or updating a course"""
    name = StringField('Course Name', validators=[DataRequired(), Length(min=3, max=100)])
    description = TextAreaField('Description', validators=[Length(max=500)])
    submit = SubmitField('Save Course')


class DocumentUploadForm(FlaskForm):
    """Form for uploading a document to a course"""
    title = StringField('Document Title', validators=[DataRequired(), Length(min=3, max=100)])
    document = FileField('Upload Document', validators=[
        FileRequired(),
        FileAllowed(['pdf', 'docx', 'txt'], 'Only PDF, Word, and text files are allowed.')
    ])
    document_type = SelectField('Document Type', choices=[
        ('note', 'Course Notes'),
        ('paper', 'Past Paper')
    ], default='note')
    submit = SubmitField('Upload Document')


class BatchUploadForm(FlaskForm):
    """Form for batch uploading multiple documents"""
    documents = MultipleFileField('Upload Documents', validators=[
        FileRequired(),
        FileAllowed(['pdf', 'docx', 'txt'], 'Only PDF, Word, and text files are allowed.')
    ])
    document_type = SelectField('Document Type', choices=[
        ('note', 'Course Notes'),
        ('paper', 'Past Paper')
    ], default='note')
    submit = SubmitField('Upload Documents')


class DocumentSearchForm(FlaskForm):
    """Form for searching documents within a course"""
    query = StringField('Search Term', validators=[DataRequired(), Length(min=2, max=100)])
    document_type = SelectField('Document Type', choices=[
        ('all', 'All Documents'),
        ('note', 'Course Notes'),
        ('paper', 'Past Paper')
    ], default='all')
    submit = SubmitField('Search')