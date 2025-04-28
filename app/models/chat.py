from datetime import datetime
from app import db

class Chat(db.Model):
    """Chat model for storing conversations"""
    __tablename__ = 'chats'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    course_id = db.Column(db.Integer, db.ForeignKey('courses.id'), nullable=True)
    
    # Relationships
    messages = db.relationship('Message', backref='chat', lazy=True, cascade="all, delete-orphan", 
                             order_by="Message.created_at")
    
    @property
    def message_count(self):
        """Returns the number of messages in the chat"""
        return len(self.messages)
    
    @property
    def last_message(self):
        """Returns the last message in the chat"""
        if self.messages:
            return max(self.messages, key=lambda m: m.created_at)
        return None
    
    def __repr__(self):
        return f"Chat('{self.title}')"


class Message(db.Model):
    """Message model for storing chat messages"""
    __tablename__ = 'messages'
    
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), default='user')  # 'user' or 'assistant'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_saved = db.Column(db.Boolean, default=False)
    save_type = db.Column(db.String(20), nullable=True)  # 'question', 'answer', etc.
    tags = db.Column(db.String(100), nullable=True)
    
    # Foreign keys
    chat_id = db.Column(db.Integer, db.ForeignKey('chats.id'), nullable=False)
    
    @property
    def is_user(self):
        """Property to provide backward compatibility"""
        return self.role == 'user'
    
    def __repr__(self):
        return f"Message('{self.content[:20]}...')"