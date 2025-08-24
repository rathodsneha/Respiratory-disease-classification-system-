from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_cors import CORS
import os

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()

def create_app(config_name=None):
    """Application factory pattern"""
    
    app = Flask(__name__)
    
    # Load configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    from config.config import config
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    CORS(app)
    
    # Configure login manager
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    @login_manager.user_loader
    def load_user(user_id):
        from app.auth.models import User
        return User.query.get(int(user_id))
    
    # Register blueprints
    from app.auth import bp as auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')
    
    from app.patient import bp as patient_bp
    app.register_blueprint(patient_bp, url_prefix='/patient')
    
    from app.audio import bp as audio_bp
    app.register_blueprint(audio_bp, url_prefix='/audio')
    
    from app.models import bp as models_bp
    app.register_blueprint(models_bp, url_prefix='/models')
    
    from app.dashboard import bp as dashboard_bp
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
    
    # Main route
    @app.route('/')
    def index():
        return redirect(url_for('auth.login'))
    
    from flask import redirect, url_for
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app