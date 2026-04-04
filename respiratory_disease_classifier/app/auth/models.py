from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from app import db

class Role(db.Model):
    """User roles for role-based access control"""
    __tablename__ = 'roles'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True, nullable=False)
    description = db.Column(db.String(255))
    
    # Permissions
    can_view_patients = db.Column(db.Boolean, default=True)
    can_edit_patients = db.Column(db.Boolean, default=False)
    can_delete_patients = db.Column(db.Boolean, default=False)
    can_upload_audio = db.Column(db.Boolean, default=True)
    can_run_models = db.Column(db.Boolean, default=True)
    can_view_reports = db.Column(db.Boolean, default=True)
    can_manage_users = db.Column(db.Boolean, default=False)
    can_train_models = db.Column(db.Boolean, default=False)
    
    # Relationships
    users = db.relationship('User', backref='role', lazy='dynamic')
    
    def __repr__(self):
        return f'<Role {self.name}>'
    
    @staticmethod
    def insert_roles():
        """Insert default roles into the database"""
        roles = {
            'Doctor': {
                'description': 'Medical doctor with full patient access',
                'can_view_patients': True,
                'can_edit_patients': True,
                'can_delete_patients': False,
                'can_upload_audio': True,
                'can_run_models': True,
                'can_view_reports': True,
                'can_manage_users': False,
                'can_train_models': False
            },
            'Technician': {
                'description': 'Medical technician with limited access',
                'can_view_patients': True,
                'can_edit_patients': False,
                'can_delete_patients': False,
                'can_upload_audio': True,
                'can_run_models': True,
                'can_view_reports': True,
                'can_manage_users': False,
                'can_train_models': False
            },
            'Admin': {
                'description': 'System administrator with full access',
                'can_view_patients': True,
                'can_edit_patients': True,
                'can_delete_patients': True,
                'can_upload_audio': True,
                'can_run_models': True,
                'can_view_reports': True,
                'can_manage_users': True,
                'can_train_models': True
            },
            'Researcher': {
                'description': 'Researcher with model training access',
                'can_view_patients': False,
                'can_edit_patients': False,
                'can_delete_patients': False,
                'can_upload_audio': True,
                'can_run_models': True,
                'can_view_reports': True,
                'can_manage_users': False,
                'can_train_models': True
            }
        }
        
        for role_name, permissions in roles.items():
            role = Role.query.filter_by(name=role_name).first()
            if role is None:
                role = Role(name=role_name)
            
            for permission, value in permissions.items():
                setattr(role, permission, value)
            
            db.session.add(role)
        
        db.session.commit()

class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128))
    
    # Personal information
    first_name = db.Column(db.String(64), nullable=False)
    last_name = db.Column(db.String(64), nullable=False)
    phone = db.Column(db.String(20))
    
    # Role and permissions
    role_id = db.Column(db.Integer, db.ForeignKey('roles.id'))
    
    # Account status
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Two-factor authentication
    two_factor_enabled = db.Column(db.Boolean, default=False)
    two_factor_secret = db.Column(db.String(32))
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def get_full_name(self):
        """Get user's full name"""
        return f"{self.first_name} {self.last_name}"
    
    def can(self, permission):
        """Check if user has specific permission"""
        return self.role is not None and getattr(self.role, permission, False)
    
    def is_admin(self):
        """Check if user is admin"""
        return self.role and self.role.name == 'Admin'
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        db.session.commit()

class LoginAttempt(db.Model):
    """Track login attempts for security"""
    __tablename__ = 'login_attempts'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), nullable=False)
    ip_address = db.Column(db.String(45), nullable=False)  # IPv6 compatible
    success = db.Column(db.Boolean, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_agent = db.Column(db.String(255))
    
    def __repr__(self):
        return f'<LoginAttempt {self.username} - {"Success" if self.success else "Failed"}>'