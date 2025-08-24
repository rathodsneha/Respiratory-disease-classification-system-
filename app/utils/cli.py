"""
CLI commands for the Flask application
"""

import click
from flask.cli import with_appcontext
from app import db
from app.models import User, Patient, AudioRecording, AnalysisResult, MedicalRecord

def register_cli_commands(app):
    """Register CLI commands with the Flask app"""
    
    @app.cli.command()
    @with_appcontext
    def create_tables():
        """Create all database tables"""
        try:
            db.create_all()
            click.echo('Database tables created successfully!')
        except Exception as e:
            click.echo(f'Error creating tables: {e}')
    
    @app.cli.command()
    @with_appcontext
    def drop_tables():
        """Drop all database tables"""
        if click.confirm('Are you sure you want to drop all tables? This will delete all data!'):
            try:
                db.drop_all()
                click.echo('All database tables dropped successfully!')
            except Exception as e:
                click.echo(f'Error dropping tables: {e}')
    
    @app.cli.command()
    @with_appcontext
    def reset_db():
        """Reset the database (drop and recreate all tables)"""
        if click.confirm('Are you sure you want to reset the database? This will delete all data!'):
            try:
                db.drop_all()
                db.create_all()
                click.echo('Database reset successfully!')
            except Exception as e:
                click.echo(f'Error resetting database: {e}')
    
    @app.cli.command()
    @with_appcontext
    def list_users():
        """List all users in the system"""
        try:
            users = User.query.all()
            if not users:
                click.echo('No users found in the system.')
                return
            
            click.echo('Users in the system:')
            click.echo('-' * 80)
            for user in users:
                click.echo(f'ID: {user.id} | Username: {user.username} | Role: {user.role} | Active: {user.is_active}')
        except Exception as e:
            click.echo(f'Error listing users: {e}')
    
    @app.cli.command()
    @with_appcontext
    def list_patients():
        """List all patients in the system"""
        try:
            patients = Patient.query.all()
            if not patients:
                click.echo('No patients found in the system.')
                return
            
            click.echo('Patients in the system:')
            click.echo('-' * 80)
            for patient in patients:
                click.echo(f'ID: {patient.id} | Patient ID: {patient.patient_id} | Name: {patient.get_full_name()} | Age: {patient.age}')
        except Exception as e:
            click.echo(f'Error listing patients: {e}')
    
    @app.cli.command()
    @with_appcontext
    def list_audio_recordings():
        """List all audio recordings in the system"""
        try:
            recordings = AudioRecording.query.all()
            if not recordings:
                click.echo('No audio recordings found in the system.')
                return
            
            click.echo('Audio recordings in the system:')
            click.echo('-' * 80)
            for recording in recordings:
                click.echo(f'ID: {recording.id} | File: {recording.filename} | Duration: {recording.get_duration_display()} | Patient: {recording.patient.get_full_name()}')
        except Exception as e:
            click.echo(f'Error listing audio recordings: {e}')
    
    @app.cli.command()
    @with_appcontext
    def list_analysis_results():
        """List all analysis results in the system"""
        try:
            results = AnalysisResult.query.all()
            if not results:
                click.echo('No analysis results found in the system.')
                return
            
            click.echo('Analysis results in the system:')
            click.echo('-' * 80)
            for result in results:
                primary_prediction, primary_confidence = result.get_primary_prediction()
                click.echo(f'ID: {result.id} | Primary: {primary_prediction} | Confidence: {primary_confidence:.2f} | Patient: {result.patient.get_full_name()}')
        except Exception as e:
            click.echo(f'Error listing analysis results: {e}')
    
    @app.cli.command()
    @with_appcontext
    def system_status():
        """Display system status and statistics"""
        try:
            click.echo('System Status Report')
            click.echo('=' * 50)
            
            # User statistics
            total_users = User.query.count()
            active_users = User.query.filter_by(is_active=True).count()
            click.echo(f'Users: {total_users} total, {active_users} active')
            
            # Patient statistics
            total_patients = Patient.query.count()
            active_patients = Patient.query.filter_by(is_active=True).count()
            click.echo(f'Patients: {total_patients} total, {active_patients} active')
            
            # Audio recording statistics
            total_recordings = AudioRecording.query.count()
            processed_recordings = AudioRecording.query.filter_by(is_processed=True).count()
            click.echo(f'Audio Recordings: {total_recordings} total, {processed_recordings} processed')
            
            # Analysis result statistics
            total_analyses = AnalysisResult.query.count()
            click.echo(f'Analysis Results: {total_analyses} total')
            
            # Medical record statistics
            total_records = MedicalRecord.query.count()
            click.echo(f'Medical Records: {total_records} total')
            
            click.echo('=' * 50)
            click.echo('System status check completed successfully!')
            
        except Exception as e:
            click.echo(f'Error checking system status: {e}')
    
    @app.cli.command()
    @with_appcontext
    def cleanup_orphaned_files():
        """Clean up orphaned audio files that don't have database records"""
        try:
            import os
            from app import current_app
            
            upload_folder = current_app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_folder):
                click.echo('Upload folder does not exist.')
                return
            
            # Get all files in upload folder
            all_files = set()
            for root, dirs, files in os.walk(upload_folder):
                for file in files:
                    all_files.add(os.path.join(root, file))
            
            # Get all file paths from database
            db_files = set()
            recordings = AudioRecording.query.all()
            for recording in recordings:
                if recording.file_path and os.path.exists(recording.file_path):
                    db_files.add(recording.file_path)
            
            # Find orphaned files
            orphaned_files = all_files - db_files
            
            if not orphaned_files:
                click.echo('No orphaned files found.')
                return
            
            click.echo(f'Found {len(orphaned_files)} orphaned files:')
            for file_path in orphaned_files:
                click.echo(f'  {file_path}')
            
            if click.confirm('Do you want to delete these orphaned files?'):
                deleted_count = 0
                for file_path in orphaned_files:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except OSError as e:
                        click.echo(f'Error deleting {file_path}: {e}')
                
                click.echo(f'Deleted {deleted_count} orphaned files.')
            else:
                click.echo('No files were deleted.')
                
        except Exception as e:
            click.echo(f'Error cleaning up orphaned files: {e}')
    
    @app.cli.command()
    @with_appcontext
    def backup_database():
        """Create a backup of the database"""
        try:
            import sqlite3
            import shutil
            from datetime import datetime
            
            # Get database path
            db_path = db.engine.url.database
            if db_path == ':memory:':
                click.echo('Cannot backup in-memory database.')
                return
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f'{db_path}.backup_{timestamp}'
            
            # Copy database file
            shutil.copy2(db_path, backup_path)
            click.echo(f'Database backed up to: {backup_path}')
            
        except Exception as e:
            click.echo(f'Error backing up database: {e}')
    
    @app.cli.command()
    @with_appcontext
    def validate_data():
        """Validate data integrity in the database"""
        try:
            click.echo('Validating data integrity...')
            issues = []
            
            # Check for orphaned audio recordings
            orphaned_recordings = AudioRecording.query.filter(
                ~AudioRecording.patient_id.in_([p.id for p in Patient.query.all()])
            ).all()
            
            if orphaned_recordings:
                issues.append(f'Found {len(orphaned_recordings)} audio recordings with invalid patient references')
            
            # Check for orphaned analysis results
            orphaned_analyses = AnalysisResult.query.filter(
                ~AnalysisResult.audio_recording_id.in_([r.id for r in AudioRecording.query.all()])
            ).all()
            
            if orphaned_analyses:
                issues.append(f'Found {len(orphaned_analyses)} analysis results with invalid audio recording references')
            
            # Check for orphaned medical records
            orphaned_records = MedicalRecord.query.filter(
                ~MedicalRecord.patient_id.in_([p.id for p in Patient.query.all()])
            ).all()
            
            if orphaned_records:
                issues.append(f'Found {len(orphaned_records)} medical records with invalid patient references')
            
            if not issues:
                click.echo('Data validation completed successfully. No issues found.')
            else:
                click.echo('Data validation completed. Issues found:')
                for issue in issues:
                    click.echo(f'  - {issue}')
                    
        except Exception as e:
            click.echo(f'Error validating data: {e}')