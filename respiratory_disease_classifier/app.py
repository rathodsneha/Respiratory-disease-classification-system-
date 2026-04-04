#!/usr/bin/env python3
"""
Respiratory Disease Classification System
Main application entry point
"""

import os
import sys
from flask import Flask, render_template, redirect, url_for

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app import create_app, db
from app.auth.models import Role, User

def main():
    """Main application entry point"""
    
    # Create Flask application
    app = create_app()
    
    with app.app_context():
        # Create database tables
        db.create_all()
        
        # Insert default roles
        Role.insert_roles()
        
        # Create default admin user if it doesn't exist
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            admin_role = Role.query.filter_by(name='Admin').first()
            admin_user = User(
                username='admin',
                email='admin@respiratory-classifier.com',
                first_name='System',
                last_name='Administrator',
                role_id=admin_role.id if admin_role else 1,
                is_active=True,
                is_verified=True
            )
            admin_user.set_password('admin123')  # Change in production
            db.session.add(admin_user)
            db.session.commit()
            print("Default admin user created: admin/admin123")
    
    # Add main route
    @app.route('/')
    def index():
        """Main application route"""
        return redirect(url_for('auth.login'))
    
    @app.route('/health')
    def health_check():
        """Health check endpoint"""
        return {'status': 'healthy', 'service': 'respiratory-disease-classifier'}, 200
    
    return app

if __name__ == '__main__':
    app = main()
    
    # Development server configuration
    debug_mode = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    
    print(f"Starting Respiratory Disease Classification System...")
    print(f"Server: http://{host}:{port}")
    print(f"Debug mode: {debug_mode}")
    
    app.run(
        host=host,
        port=port,
        debug=debug_mode,
        threaded=True
    )