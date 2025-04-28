from datetime import datetime
from app import db

class Question(db.Model):
    """Question model for storing generated questions"""
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    options = db.Column(db.JSON, nullable=True)  # For multiple choice questions
    question_type = db.Column(db.String(20), nullable=False)  # 'multiple_choice' or 'structured'
    difficulty = db.Column(db.String(20), nullable=False)  # 'easy', 'medium', 'hard'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    document_id = db.Column(db.Integer, db.ForeignKey('documents.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    def __repr__(self):
        return f"Question('{self.content[:30]}...', '{self.question_type}')"