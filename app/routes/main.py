"""
Main routes for dashboard and home page
"""

from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user
from app.models import Patient, AudioRecording, AnalysisResult, MedicalRecord
from app.utils.decorators import role_required

# Create blueprint
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Home page route"""
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    
    return render_template('main/index.html', title='Home')

@main_bp.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard route"""
    
    # Get statistics for dashboard
    stats = get_dashboard_stats()
    
    # Get recent activities
    recent_activities = get_recent_activities()
    
    return render_template('main/dashboard.html', 
                         title='Dashboard',
                         stats=stats,
                         recent_activities=recent_activities)

@main_bp.route('/about')
def about():
    """About page route"""
    return render_template('main/about.html', title='About')

@main_bp.route('/contact')
def contact():
    """Contact page route"""
    return render_template('main/contact.html', title='Contact')

@main_bp.route('/help')
def help_page():
    """Help page route"""
    return render_template('main/help.html', title='Help')

@main_bp.route('/api/dashboard-stats')
@login_required
def api_dashboard_stats():
    """API endpoint for dashboard statistics"""
    stats = get_dashboard_stats()
    return stats

def get_dashboard_stats():
    """Get dashboard statistics"""
    
    # Get counts for different entities
    total_patients = Patient.query.filter_by(is_active=True).count()
    total_recordings = AudioRecording.query.count()
    total_analyses = AnalysisResult.query.count()
    total_medical_records = MedicalRecord.query.count()
    
    # Get counts for current user
    if current_user.is_authenticated:
        user_patients = Patient.query.filter_by(created_by_id=current_user.id, is_active=True).count()
        user_recordings = AudioRecording.query.filter_by(recorded_by_id=current_user.id).count()
        user_analyses = AnalysisResult.query.filter_by(analyzed_by_id=current_user.id).count()
    else:
        user_patients = 0
        user_recordings = 0
        user_analyses = 0
    
    # Get recent counts (last 30 days)
    from datetime import datetime, timedelta
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    
    recent_patients = Patient.query.filter(
        Patient.created_at >= thirty_days_ago,
        Patient.is_active == True
    ).count()
    
    recent_recordings = AudioRecording.query.filter(
        AudioRecording.created_at >= thirty_days_ago
    ).count()
    
    recent_analyses = AnalysisResult.query.filter(
        AnalysisResult.created_at >= thirty_days_ago
    ).count()
    
    # Get disease distribution from analysis results
    disease_stats = {}
    if total_analyses > 0:
        from sqlalchemy import func
        results = db.session.query(
            AnalysisResult.hybrid_prediction,
            func.count(AnalysisResult.id)
        ).filter(
            AnalysisResult.hybrid_prediction.isnot(None)
        ).group_by(AnalysisResult.hybrid_prediction).all()
        
        for disease, count in results:
            disease_stats[disease] = count
    
    return {
        'total_patients': total_patients,
        'total_recordings': total_recordings,
        'total_analyses': total_analyses,
        'total_medical_records': total_medical_records,
        'user_patients': user_patients,
        'user_recordings': user_recordings,
        'user_analyses': user_analyses,
        'recent_patients': recent_patients,
        'recent_recordings': recent_recordings,
        'recent_analyses': recent_analyses,
        'disease_stats': disease_stats
    }

def get_recent_activities():
    """Get recent activities for dashboard"""
    
    activities = []
    
    # Get recent patients
    recent_patients = Patient.query.filter_by(is_active=True).order_by(
        Patient.created_at.desc()
    ).limit(5).all()
    
    for patient in recent_patients:
        activities.append({
            'type': 'patient',
            'action': 'created',
            'description': f'New patient {patient.get_full_name()} registered',
            'timestamp': patient.created_at,
            'user': patient.created_by.get_full_name(),
            'link': url_for('patients.view_patient', patient_id=patient.id)
        })
    
    # Get recent audio recordings
    recent_recordings = AudioRecording.query.order_by(
        AudioRecording.created_at.desc()
    ).limit(5).all()
    
    for recording in recent_recordings:
        activities.append({
            'type': 'recording',
            'action': 'uploaded',
            'description': f'Audio recording uploaded for {recording.patient.get_full_name()}',
            'timestamp': recording.created_at,
            'user': recording.recorded_by.get_full_name(),
            'link': url_for('audio.view_recording', recording_id=recording.id)
        })
    
    # Get recent analysis results
    recent_analyses = AnalysisResult.query.order_by(
        AnalysisResult.created_at.desc()
    ).limit(5).all()
    
    for analysis in recent_analyses:
        primary_prediction, primary_confidence = analysis.get_primary_prediction()
        activities.append({
            'type': 'analysis',
            'action': 'completed',
            'description': f'Analysis completed: {primary_prediction} ({primary_confidence:.1%})',
            'timestamp': analysis.created_at,
            'user': analysis.analyzed_by.get_full_name(),
            'link': url_for('audio.view_analysis', analysis_id=analysis.id)
        })
    
    # Sort all activities by timestamp
    activities.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Return top 10 most recent activities
    return activities[:10]