#!/usr/bin/env python3
"""
Setup script for Respiratory Disease Classification System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'static/uploads',
        'static/css',
        'static/js',
        'ml/models',
        'ml/data',
        'ml/preprocessing',
        'ml/training',
        'ml/evaluation',
        'tests',
        'docs'
    ]
    
    print("\n📁 Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def create_env_file():
    """Create .env file from template"""
    if not os.path.exists('.env'):
        print("\n🔧 Creating .env file...")
        shutil.copy('.env.example', '.env')
        print("✅ .env file created from template")
        print("⚠️  Please edit .env file with your configuration")
    else:
        print("✅ .env file already exists")

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    # Check if virtual environment exists
    if not os.path.exists('venv'):
        print("🔧 Creating virtual environment...")
        if not run_command('python3 -m venv venv', 'Creating virtual environment'):
            return False
    
    # Activate virtual environment and install dependencies
    if sys.platform == 'win32':
        activate_cmd = 'venv\\Scripts\\activate'
        pip_cmd = 'venv\\Scripts\\pip'
    else:
        activate_cmd = 'source venv/bin/activate'
        pip_cmd = 'venv/bin/pip'
    
    # Install requirements
    if not run_command(f'{pip_cmd} install -r requirements.txt', 'Installing dependencies'):
        return False
    
    return True

def setup_database():
    """Setup database and create initial data"""
    print("\n🗄️  Setting up database...")
    
    # Create database tables
    if not run_command('python app.py init-db', 'Creating database tables'):
        return False
    
    # Create admin user
    if not run_command('python app.py create-admin', 'Creating admin user'):
        return False
    
    # Seed database with sample data
    if not run_command('python app.py init-db', 'Seeding database with sample data'):
        return False
    
    return True

def run_tests():
    """Run basic tests"""
    print("\n🧪 Running basic tests...")
    
    if not run_command('python app.py test-models', 'Testing database models'):
        return False
    
    return True

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup completed successfully!")
    print("="*60)
    print("\n📋 Next steps:")
    print("1. Edit .env file with your configuration")
    print("2. Activate virtual environment:")
    if sys.platform == 'win32':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("3. Start the application:")
    print("   python app.py")
    print("4. Open your browser and go to: http://localhost:5000")
    print("5. Login with demo account: admin / admin123")
    print("\n📚 Documentation:")
    print("- README.md: Project overview and setup instructions")
    print("- docs/: Detailed documentation")
    print("\n🐛 Troubleshooting:")
    print("- Check logs/ directory for error logs")
    print("- Ensure all dependencies are installed")
    print("- Verify database configuration in .env")
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("🏥 Respiratory Disease Classification System Setup")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup database
    if not setup_database():
        print("❌ Failed to setup database")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("❌ Some tests failed")
        print("⚠️  Continuing with setup...")
    
    # Print next steps
    print_next_steps()

if __name__ == '__main__':
    main()