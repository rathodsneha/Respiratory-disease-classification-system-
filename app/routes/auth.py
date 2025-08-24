"""
Authentication routes for user login, registration, and session management
"""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
from app import db
from app.models import User
from app.forms.auth_forms import LoginForm, RegistrationForm, PasswordResetForm
from app.utils.decorators import admin_required

# Create blueprint
auth_bp = Blueprint('auth', __name__, url_prefix='/auth')

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login route"""
    
    # Redirect if user is already logged in
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = LoginForm()
    
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        if user and user.check_password(form.password.data):
            if user.is_active:
                login_user(user, remember=form.remember_me.data)
                user.update_last_login()
                
                # Redirect to next page or dashboard
                next_page = request.args.get('next')
                if not next_page or not next_page.startswith('/'):
                    next_page = url_for('main.dashboard')
                
                flash(f'Welcome back, {user.get_full_name()}!', 'success')
                return redirect(next_page)
            else:
                flash('Your account has been deactivated. Please contact an administrator.', 'error')
        else:
            flash('Invalid username or password. Please try again.', 'error')
    
    return render_template('auth/login.html', form=form, title='Login')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    """User registration route"""
    
    # Redirect if user is already logged in
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    form = RegistrationForm()
    
    if form.validate_on_submit():
        # Check if username or email already exists
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists. Please choose a different one.', 'error')
            return render_template('auth/register.html', form=form, title='Register')
        
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered. Please use a different email or login.', 'error')
            return render_template('auth/register.html', form=form, title='Register')
        
        # Create new user
        user = User(
            username=form.username.data,
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            role='technician'  # Default role for new registrations
        )
        user.set_password(form.password.data)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login with your new account.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html', form=form, title='Register')

@auth_bp.route('/logout')
@login_required
def logout():
    """User logout route"""
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('auth.login'))

@auth_bp.route('/profile')
@login_required
def profile():
    """User profile route"""
    return render_template('auth/profile.html', title='Profile')

@auth_bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change password route"""
    from app.forms.auth_forms import ChangePasswordForm
    
    form = ChangePasswordForm()
    
    if form.validate_on_submit():
        if current_user.check_password(form.current_password.data):
            current_user.set_password(form.new_password.data)
            db.session.commit()
            flash('Password changed successfully!', 'success')
            return redirect(url_for('auth.profile'))
        else:
            flash('Current password is incorrect.', 'error')
    
    return render_template('auth/change_password.html', form=form, title='Change Password')

@auth_bp.route('/password-reset', methods=['GET', 'POST'])
def password_reset():
    """Password reset request route"""
    form = PasswordResetForm()
    
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            # In a real application, send password reset email
            # For now, just show a message
            flash('If an account with that email exists, a password reset link has been sent.', 'info')
        else:
            # Don't reveal if email exists or not
            flash('If an account with that email exists, a password reset link has been sent.', 'info')
        
        return redirect(url_for('auth.login'))
    
    return render_template('auth/password_reset.html', form=form, title='Password Reset')

# API Routes for AJAX requests
@auth_bp.route('/api/login', methods=['POST'])
def api_login():
    """API endpoint for user login"""
    data = request.get_json()
    
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'error': 'Missing username or password'}), 400
    
    user = User.query.filter_by(username=data['username']).first()
    
    if user and user.check_password(data['password']):
        if user.is_active:
            login_user(user)
            user.update_last_login()
            
            return jsonify({
                'success': True,
                'message': 'Login successful',
                'user': user.to_dict(),
                'redirect_url': url_for('main.dashboard')
            })
        else:
            return jsonify({'error': 'Account deactivated'}), 403
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@auth_bp.route('/api/logout', methods=['POST'])
@login_required
def api_logout():
    """API endpoint for user logout"""
    logout_user()
    return jsonify({
        'success': True,
        'message': 'Logout successful',
        'redirect_url': url_for('auth.login')
    })

@auth_bp.route('/api/check-auth')
def api_check_auth():
    """API endpoint to check authentication status"""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': current_user.to_dict()
        })
    else:
        return jsonify({
            'authenticated': False,
            'user': None
        })

# Admin routes
@auth_bp.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Admin route to manage users"""
    users = User.query.all()
    return render_template('admin/users.html', users=users, title='User Management')

@auth_bp.route('/admin/users/<int:user_id>/toggle-status', methods=['POST'])
@login_required
@admin_required
def admin_toggle_user_status(user_id):
    """Admin route to toggle user active status"""
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        return jsonify({'error': 'Cannot deactivate your own account'}), 400
    
    user.is_active = not user.is_active
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'User {user.username} {"activated" if user.is_active else "deactivated"}',
        'user_active': user.is_active
    })

@auth_bp.route('/admin/users/<int:user_id>/change-role', methods=['POST'])
@login_required
@admin_required
def admin_change_user_role(user_id):
    """Admin route to change user role"""
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    if not data or 'role' not in data:
        return jsonify({'error': 'Role is required'}), 400
    
    if data['role'] not in User.ROLES:
        return jsonify({'error': 'Invalid role'}), 400
    
    if user.id == current_user.id and data['role'] != 'admin':
        return jsonify({'error': 'Cannot change your own role'}), 400
    
    user.role = data['role']
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': f'User {user.username} role changed to {data["role"]}',
        'new_role': data['role']
    })