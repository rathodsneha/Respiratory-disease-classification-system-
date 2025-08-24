from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db, login_manager

class User(UserMixin, db.Model):
    """User model for authentication and role management"""
    
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    first_name = db.Column(db.String(64), nullable=False)
    last_name = db.Column(db.String(64), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='technician')
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    last_login = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patients = db.relationship('Patient', backref='created_by', lazy='dynamic')
    audio_recordings = db.relationship('AudioRecording', backref='recorded_by', lazy='dynamic')
    analysis_results = db.relationship('AnalysisResult', backref='analyzed_by', lazy='dynamic')
    
    # Role constants
    ROLES = {
        'admin': 'Administrator',
        'doctor': 'Doctor',
        'technician': 'Technician',
        'researcher': 'Researcher'
    }
    
    def __init__(self, **kwargs):
        super(User, self).__init__(**kwargs)
        if self.role not in self.ROLES:
            self.role = 'technician'
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def has_role(self, role):
        """Check if user has specific role"""
        return self.role == role
    
    def is_admin(self):
        """Check if user is admin"""
        return self.role == 'admin'
    
    def is_doctor(self):
        """Check if user is doctor"""
        return self.role in ['admin', 'doctor']
    
    def can_manage_patients(self):
        """Check if user can manage patients"""
        return self.role in ['admin', 'doctor', 'technician']
    
    def can_view_analytics(self):
        """Check if user can view analytics"""
        return self.role in ['admin', 'doctor', 'researcher']
    
    def get_full_name(self):
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}"
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'role': self.role,
            'role_display': self.ROLES.get(self.role, self.role),
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __repr__(self):
        return f'<User {self.username}>'

@login_manager.user_loader
def load_user(user_id):
    """User loader for Flask-Login"""
    return User.query.get(int(user_id))