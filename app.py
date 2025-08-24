#!/usr/bin/env python3
"""
Respiratory Disease Classification System
Main application entry point
"""

import os
import sys
from app import create_app, db
from app.models import User, Patient, AudioRecording, AnalysisResult, MedicalRecord

# Create Flask application instance
app = create_app()

@app.shell_context_processor
def make_shell_context():
    """Make database models available in Flask shell"""
    return {
        'db': db,
        'User': User,
        'Patient': Patient,
        'AudioRecording': AudioRecording,
        'AnalysisResult': AnalysisResult,
        'MedicalRecord': MedicalRecord
    }

@app.cli.command()
def init_db():
    """Initialize the database with sample data"""
    from app.utils.seed_data import seed_database
    
    print("Initializing database...")
    seed_database()
    print("Database initialization complete!")

@app.cli.command()
def create_admin():
    """Create an admin user"""
    from app.utils.seed_data import create_admin_user
    
    print("Creating admin user...")
    create_admin_user()
    print("Admin user created successfully!")

@app.cli.command()
def test_models():
    """Test all models"""
    from app.utils.test_models import test_all_models
    
    print("Testing models...")
    test_all_models()
    print("Model testing complete!")

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    # Get host from environment or use default
    host = os.environ.get('HOST', '0.0.0.0')
    
    # Get debug mode from environment
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting Respiratory Disease Classification System...")
    print(f"Environment: {os.environ.get('FLASK_ENV', 'development')}")
    print(f"Debug mode: {debug}")
    print(f"Server will be available at: http://{host}:{port}")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)