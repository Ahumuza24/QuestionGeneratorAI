import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from dotenv import load_dotenv
from flask_wtf.csrf import CSRFProtect
from flask_migrate import Migrate

# Load environment variables
load_dotenv()

# Initialize SQLAlchemy
db = SQLAlchemy()

# Initialize LoginManager
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

def create_app():
    # Initialize Flask app
    app = Flask(__name__)
    
    # Configure app
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', 'sqlite:///questions.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
    app.config['UPLOAD_FOLDER'] = os.path.join(app.static_folder, 'uploads')
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    csrf = CSRFProtect(app)
    
    # Register markdown filter
    import markdown as md
    from markupsafe import Markup
    
    @app.template_filter('markdown')
    def markdown_filter(text):
        if text:
            return Markup(md.markdown(text, extensions=['extra', 'nl2br']))
        return ""
    
    # Register custom template filters
    from app.utils.filters import markdown
    app.jinja_env.filters['markdown'] = markdown
    
    # Import models
    with app.app_context():
        from app.models.user import User
        from app.models.course import Course
        from app.models.document import Document
        from app.models.question import Question
        from app.models.chat import Chat, Message
    
    # Register blueprints
    from app.routes.main import main
    from app.routes.auth import auth
    from app.routes.questions import questions
    
    app.register_blueprint(main)
    app.register_blueprint(auth)
    app.register_blueprint(questions)
    
    # Register courses blueprint
    try:
        from app.routes.courses import courses as courses_blueprint
        app.register_blueprint(courses_blueprint)
    except ImportError:
        print("Warning: Could not import courses blueprint. Some features may be unavailable.")
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app