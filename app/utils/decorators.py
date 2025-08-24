"""
Decorators for role-based access control and other functionality
"""

from functools import wraps
from flask import abort, flash, redirect, url_for, request, jsonify
from flask_login import current_user, login_required

def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        
        if not current_user.is_admin():
            if request.is_xhr:
                return jsonify({'error': 'Admin access required'}), 403
            else:
                flash('Admin access required for this page.', 'error')
                return redirect(url_for('main.dashboard'))
        
        return f(*args, **kwargs)
    return decorated_function

def doctor_required(f):
    """Decorator to require doctor or admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        
        if not current_user.is_doctor():
            if request.is_xhr:
                return jsonify({'error': 'Doctor access required'}), 403
            else:
                flash('Doctor access required for this page.', 'error')
                return redirect(url_for('main.dashboard'))
        
        return f(*args, **kwargs)
    return decorated_function

def role_required(roles):
    """Decorator to require specific role(s)"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('auth.login'))
            
            if isinstance(roles, str):
                required_roles = [roles]
            else:
                required_roles = roles
            
            if current_user.role not in required_roles and not current_user.is_admin():
                if request.is_xhr:
                    return jsonify({'error': f'Access denied. Required roles: {", ".join(required_roles)}'}), 403
                else:
                    flash(f'Access denied. Required roles: {", ".join(required_roles)}', 'error')
                    return redirect(url_for('main.dashboard'))
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def can_manage_patients(f):
    """Decorator to require patient management permission"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        
        if not current_user.can_manage_patients():
            if request.is_xhr:
                return jsonify({'error': 'Patient management access required'}), 403
            else:
                flash('Patient management access required for this page.', 'error')
                return redirect(url_for('main.dashboard'))
        
        return f(*args, **kwargs)
    return decorated_function

def can_view_analytics(f):
    """Decorator to require analytics viewing permission"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.login'))
        
        if not current_user.can_view_analytics():
            if request.is_xhr:
                return jsonify({'error': 'Analytics access required'}), 403
            else:
                flash('Analytics access required for this page.', 'error')
                return redirect(url_for('main.dashboard'))
        
        return f(*args, **kwargs)
    return decorated_function

def ajax_required(f):
    """Decorator to require AJAX requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_xhr:
            abort(400, description="AJAX request required")
        return f(*args, **kwargs)
    return decorated_function

def json_required(f):
    """Decorator to require JSON requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            if request.is_xhr:
                return jsonify({'error': 'JSON request required'}), 400
            else:
                abort(400, description="JSON request required")
        return f(*args, **kwargs)
    return decorated_function

def validate_patient_access(f):
    """Decorator to validate patient access permissions"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from app.models import Patient
        
        patient_id = kwargs.get('patient_id')
        if patient_id:
            patient = Patient.query.get_or_404(patient_id)
            
            # Admin and doctors can access all patients
            if current_user.is_admin() or current_user.is_doctor():
                return f(*args, **kwargs)
            
            # Technicians can only access patients they created
            if current_user.role == 'technician' and patient.created_by_id != current_user.id:
                if request.is_xhr:
                    return jsonify({'error': 'Access denied to this patient'}), 403
                else:
                    flash('Access denied to this patient.', 'error')
                    return redirect(url_for('patients.list_patients'))
        
        return f(*args, **kwargs)
    return decorated_function

def validate_recording_access(f):
    """Decorator to validate audio recording access permissions"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from app.models import AudioRecording
        
        recording_id = kwargs.get('recording_id')
        if recording_id:
            recording = AudioRecording.query.get_or_404(recording_id)
            
            # Admin and doctors can access all recordings
            if current_user.is_admin() or current_user.is_doctor():
                return f(*args, **kwargs)
            
            # Technicians can only access recordings they created
            if current_user.role == 'technician' and recording.recorded_by_id != current_user.id:
                if request.is_xhr:
                    return jsonify({'error': 'Access denied to this recording'}), 403
                else:
                    flash('Access denied to this recording.', 'error')
                    return redirect(url_for('audio.list_recordings'))
        
        return f(*args, **kwargs)
    return decorated_function

def rate_limit_exempt(f):
    """Decorator to exempt route from rate limiting"""
    f.rate_limit_exempt = True
    return f

def cache_control(max_age=300):
    """Decorator to set cache control headers"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = f(*args, **kwargs)
            if hasattr(response, 'headers'):
                response.headers['Cache-Control'] = f'public, max-age={max_age}'
            return response
        return decorated_function
    return decorator

def require_https(f):
    """Decorator to require HTTPS in production"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        from flask import current_app
        
        if not current_app.debug and not request.is_secure:
            if request.is_xhr:
                return jsonify({'error': 'HTTPS required'}), 403
            else:
                abort(403, description="HTTPS required")
        
        return f(*args, **kwargs)
    return decorated_function