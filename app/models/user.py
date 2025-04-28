from datetime import datetime
from flask_login import UserMixin
from app import db, login_manager

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    documents = db.relationship('Document', backref='author', lazy=True, 
                               foreign_keys='Document.user_id', cascade="all, delete-orphan")
    questions = db.relationship('Question', backref='author', lazy=True, cascade="all, delete-orphan")
    courses = db.relationship('Course', backref='user', lazy=True, cascade="all, delete-orphan")
    chats = db.relationship('Chat', backref='user', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"User('{self.username}', '{self.email}')"