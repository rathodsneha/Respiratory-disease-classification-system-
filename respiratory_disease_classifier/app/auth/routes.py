from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_user, logout_user, current_user, login_required
from datetime import datetime
from app.auth import bp
from app.auth.forms import LoginForm, RegistrationForm, ChangePasswordForm, EditUserForm
from app.auth.models import User, Role, LoginAttempt
from app import db

@bp.route('/login', methods=['GET', 'POST'])
def login():
    """User login route"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        # Log login attempt
        attempt = LoginAttempt(
            username=form.username.data,
            ip_address=request.remote_addr,
            success=False,
            user_agent=request.headers.get('User-Agent', '')
        )
        
        if user is None or not user.check_password(form.password.data):
            db.session.add(attempt)
            db.session.commit()
            flash('Invalid username or password', 'error')
            return redirect(url_for('auth.login'))
        
        if not user.is_active:
            db.session.add(attempt)
            db.session.commit()
            flash('Your account has been deactivated. Please contact an administrator.', 'error')
            return redirect(url_for('auth.login'))
        
        # Successful login
        attempt.success = True
        db.session.add(attempt)
        
        login_user(user, remember=form.remember_me.data)
        user.update_last_login()
        
        db.session.commit()
        
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('dashboard.index')
        
        flash(f'Welcome back, {user.get_full_name()}!', 'success')
        return redirect(next_page)
    
    return render_template('auth/login.html', title='Sign In', form=form)

@bp.route('/logout')
@login_required
def logout():
    """User logout route"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))

@bp.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    """User registration route (admin only)"""
    if not current_user.can('can_manage_users'):
        flash('You do not have permission to register new users.', 'error')
        return redirect(url_for('dashboard.index'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            phone=form.phone.data,
            role_id=form.role.data
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        
        flash(f'User {user.username} has been registered successfully!', 'success')
        return redirect(url_for('auth.users'))
    
    return render_template('auth/register.html', title='Register User', form=form)

@bp.route('/users')
@login_required
def users():
    """List all users (admin only)"""
    if not current_user.can('can_manage_users'):
        flash('You do not have permission to view users.', 'error')
        return redirect(url_for('dashboard.index'))
    
    page = request.args.get('page', 1, type=int)
    users = User.query.paginate(
        page=page, 
        per_page=20, 
        error_out=False
    )
    
    return render_template('auth/users.html', title='User Management', users=users)

@bp.route('/user/<int:id>/edit', methods=['GET', 'POST'])
@login_required
def edit_user(id):
    """Edit user profile"""
    user = User.query.get_or_404(id)
    
    # Check permissions
    if not current_user.can('can_manage_users') and current_user.id != id:
        flash('You do not have permission to edit this user.', 'error')
        return redirect(url_for('dashboard.index'))
    
    form = EditUserForm(user.username, user.email, obj=user)
    
    if form.validate_on_submit():
        user.username = form.username.data
        user.email = form.email.data
        user.first_name = form.first_name.data
        user.last_name = form.last_name.data
        user.phone = form.phone.data
        
        # Only admins can change roles and account status
        if current_user.can('can_manage_users'):
            user.role_id = form.role.data
            user.is_active = form.is_active.data
            user.is_verified = form.is_verified.data
        
        db.session.commit()
        flash('User profile updated successfully!', 'success')
        
        if current_user.can('can_manage_users'):
            return redirect(url_for('auth.users'))
        else:
            return redirect(url_for('dashboard.index'))
    
    return render_template('auth/edit_user.html', title='Edit User', form=form, user=user)

@bp.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password"""
    form = ChangePasswordForm()
    
    if form.validate_on_submit():
        if not current_user.check_password(form.current_password.data):
            flash('Current password is incorrect.', 'error')
            return redirect(url_for('auth.change_password'))
        
        current_user.set_password(form.new_password.data)
        db.session.commit()
        
        flash('Your password has been updated successfully!', 'success')
        return redirect(url_for('dashboard.index'))
    
    return render_template('auth/change_password.html', title='Change Password', form=form)

@bp.route('/user/<int:id>/toggle-status')
@login_required
def toggle_user_status(id):
    """Toggle user active status (admin only)"""
    if not current_user.can('can_manage_users'):
        return jsonify({'error': 'Permission denied'}), 403
    
    user = User.query.get_or_404(id)
    
    # Prevent deactivating self
    if user.id == current_user.id:
        return jsonify({'error': 'Cannot deactivate your own account'}), 400
    
    user.is_active = not user.is_active
    db.session.commit()
    
    status = 'activated' if user.is_active else 'deactivated'
    flash(f'User {user.username} has been {status}.', 'success')
    
    return jsonify({'success': True, 'is_active': user.is_active})

@bp.route('/profile')
@login_required
def profile():
    """View user profile"""
    return render_template('auth/profile.html', title='Profile', user=current_user)

@bp.route('/login-history')
@login_required
def login_history():
    """View login history (admin only)"""
    if not current_user.can('can_manage_users'):
        flash('You do not have permission to view login history.', 'error')
        return redirect(url_for('dashboard.index'))
    
    page = request.args.get('page', 1, type=int)
    attempts = LoginAttempt.query.order_by(LoginAttempt.timestamp.desc()).paginate(
        page=page,
        per_page=50,
        error_out=False
    )
    
    return render_template('auth/login_history.html', title='Login History', attempts=attempts)

# Permission decorator
from functools import wraps

def permission_required(permission):
    """Decorator to check user permissions"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('auth.login'))
            if not current_user.can(permission):
                flash(f'You do not have permission to access this page.', 'error')
                return redirect(url_for('dashboard.index'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator