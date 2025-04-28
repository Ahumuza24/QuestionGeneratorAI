from datetime import datetime
from app import db
from app.models.user import User

class Course(db.Model):
    """Course model for storing course information"""
    __tablename__ = 'courses'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Remove the user relationship from here - it's defined in the User model
    # user = db.relationship('User', backref=db.backref('courses', lazy=True))
    
    documents = db.relationship('Document', backref='course', lazy=True, cascade="all, delete-orphan")
    chats = db.relationship('Chat', backref='course', lazy=True, cascade="all, delete-orphan")
    
    @property
    def document_count(self):
        """Returns the number of documents in the course"""
        return len(self.documents)
    
    @property
    def notes_count(self):
        """Returns the number of course notes in the course"""
        return sum(1 for doc in self.documents if doc.document_type == 'note')
    
    @property
    def papers_count(self):
        """Returns the number of past papers in the course"""
        return sum(1 for doc in self.documents if doc.document_type == 'paper')
    
    @property
    def last_activity_date(self):
        """Returns the date of the last activity in the course"""
        dates = [self.updated_at]
        
        # Add document dates
        if self.documents:
            dates.extend(doc.created_at for doc in self.documents)
        
        # Add chat dates
        if self.chats:
            dates.extend(chat.updated_at for chat in self.chats)
        
        return max(dates)
    
    def __repr__(self):
        return f"Course('{self.name}')"