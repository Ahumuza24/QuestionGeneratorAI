from datetime import datetime
from app import db

class Document(db.Model):
    """Document model for storing uploaded documents"""
    __tablename__ = 'documents'  # Make sure this is plural
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    document_type = db.Column(db.String(20), nullable=False, default='note')  # 'note' or 'paper'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=True)
    
    def __repr__(self):
        return f"Document('{self.title}', '{self.filename}')"
    
    @property
    def is_note(self):
        return self.document_type == 'note'
    
    @property
    def is_paper(self):
        return self.document_type == 'paper'