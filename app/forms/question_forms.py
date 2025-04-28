from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, FileField, SubmitField, IntegerField, SelectField, BooleanField
from wtforms.validators import DataRequired, NumberRange, Length

class QuestionGeneratorForm(FlaskForm):
    """Form for generating questions from a document"""
    num_questions = IntegerField('Number of Questions', 
                                validators=[DataRequired(), NumberRange(min=1, max=20)],
                                default=5)
    question_type = SelectField('Question Type', 
                               choices=[
                                   ('multiple_choice', 'Multiple Choice'),
                                   ('structured', 'Short Answer'),
                                   ('both', 'Both')
                               ],
                               default='both')
    difficulty = SelectField('Difficulty', 
                            choices=[
                                ('easy', 'Easy'),
                                ('medium', 'Medium'),
                                ('hard', 'Hard')
                            ],
                            default='medium')
    submit = SubmitField('Generate Questions')