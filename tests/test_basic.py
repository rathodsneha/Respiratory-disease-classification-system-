"""
Basic tests for Respiratory Disease Classification System
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestBasicFunctionality(unittest.TestCase):
    """Test basic system functionality"""
    
    def test_imports(self):
        """Test that all required modules can be imported"""
        try:
            from app import create_app, db
            from app.models import User, Patient, AudioRecording, AnalysisResult, MedicalRecord
            from config import get_config
            print("✅ All modules imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")
    
    def test_config_loading(self):
        """Test configuration loading"""
        try:
            from config import get_config
            config = get_config()
            self.assertIsNotNone(config)
            self.assertTrue(hasattr(config, 'SECRET_KEY'))
            self.assertTrue(hasattr(config, 'SQLALCHEMY_DATABASE_URI'))
            print("✅ Configuration loaded successfully")
        except Exception as e:
            self.fail(f"Failed to load configuration: {e}")
    
    def test_app_creation(self):
        """Test Flask app creation"""
        try:
            from app import create_app
            app = create_app('testing')
            self.assertIsNotNone(app)
            self.assertTrue(hasattr(app, 'config'))
            print("✅ Flask app created successfully")
        except Exception as e:
            self.fail(f"Failed to create Flask app: {e}")
    
    def test_database_models(self):
        """Test database model definitions"""
        try:
            from app.models import User, Patient, AudioRecording, AnalysisResult, MedicalRecord
            
            # Test User model
            user = User(
                username='testuser',
                email='test@example.com',
                first_name='Test',
                last_name='User',
                role='technician'
            )
            user.set_password('testpass123')
            self.assertTrue(user.check_password('testpass123'))
            self.assertFalse(user.check_password('wrongpass'))
            self.assertEqual(user.get_full_name(), 'Test User')
            self.assertTrue(user.can_manage_patients())
            print("✅ User model working correctly")
            
            # Test Patient model
            from datetime import date
            patient = Patient(
                first_name='John',
                last_name='Doe',
                date_of_birth=date(1990, 1, 1),
                gender='Male',
                created_by_id=1
            )
            self.assertIsNotNone(patient.patient_id)
            self.assertEqual(patient.get_full_name(), 'John Doe')
            print("✅ Patient model working correctly")
            
        except Exception as e:
            self.fail(f"Failed to test database models: {e}")
    
    def test_utility_functions(self):
        """Test utility functions"""
        try:
            from app.utils.seed_data import create_admin_user
            from app.utils.test_models import test_all_models
            
            # Test that utility functions exist
            self.assertTrue(callable(create_admin_user))
            self.assertTrue(callable(test_all_models))
            print("✅ Utility functions accessible")
            
        except Exception as e:
            self.fail(f"Failed to test utility functions: {e}")
    
    def test_forms(self):
        """Test form definitions"""
        try:
            from app.forms.auth_forms import LoginForm, RegistrationForm
            
            # Test form creation
            login_form = LoginForm()
            self.assertIsNotNone(login_form.username)
            self.assertIsNotNone(login_form.password)
            
            reg_form = RegistrationForm()
            self.assertIsNotNone(reg_form.username)
            self.assertIsNotNone(reg_form.email)
            self.assertIsNotNone(reg_form.password)
            print("✅ Forms created successfully")
            
        except Exception as e:
            self.fail(f"Failed to test forms: {e}")
    
    def test_routes(self):
        """Test route definitions"""
        try:
            from app.routes.auth import auth_bp
            from app.routes.main import main_bp
            
            # Test that blueprints exist
            self.assertIsNotNone(auth_bp)
            self.assertIsNotNone(main_bp)
            self.assertEqual(auth_bp.name, 'auth')
            self.assertEqual(main_bp.name, 'main')
            print("✅ Routes defined correctly")
            
        except Exception as e:
            self.fail(f"Failed to test routes: {e}")
    
    def test_decorators(self):
        """Test decorator functions"""
        try:
            from app.utils.decorators import admin_required, doctor_required
            
            # Test that decorators exist
            self.assertTrue(callable(admin_required))
            self.assertTrue(callable(doctor_required))
            print("✅ Decorators accessible")
            
        except Exception as e:
            self.fail(f"Failed to test decorators: {e}")

class TestConfiguration(unittest.TestCase):
    """Test configuration settings"""
    
    def test_environment_variables(self):
        """Test environment variable handling"""
        try:
            from config import get_config
            config = get_config()
            
            # Test required configuration
            self.assertIsNotNone(config.SECRET_KEY)
            self.assertIsNotNone(config.SQLALCHEMY_DATABASE_URI)
            self.assertIsNotNone(config.UPLOAD_FOLDER)
            self.assertIsNotNone(config.MODEL_PATH)
            
            print("✅ Configuration values set correctly")
            
        except Exception as e:
            self.fail(f"Failed to test configuration: {e}")
    
    def test_disease_classes(self):
        """Test disease class configuration"""
        try:
            from config import get_config
            config = get_config()
            
            # Test disease classes
            self.assertIsInstance(config.DISEASE_CLASSES, list)
            self.assertGreater(len(config.DISEASE_CLASSES), 0)
            self.assertIn('healthy', config.DISEASE_CLASSES)
            self.assertIn('asthma', config.DISEASE_CLASSES)
            
            print("✅ Disease classes configured correctly")
            
        except Exception as e:
            self.fail(f"Failed to test disease classes: {e}")

class TestDatabaseConnection(unittest.TestCase):
    """Test database connectivity"""
    
    def test_database_connection(self):
        """Test database connection"""
        try:
            from app import create_app, db
            app = create_app('testing')
            
            with app.app_context():
                # Test database connection
                db.engine.execute('SELECT 1')
                print("✅ Database connection successful")
                
        except Exception as e:
            self.fail(f"Failed to connect to database: {e}")

def run_tests():
    """Run all tests"""
    print("🧪 Running Respiratory Disease Classification System Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBasicFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestConfiguration))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseConnection))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 All tests passed successfully!")
        return True
    else:
        print(f"❌ {len(result.failures)} tests failed")
        print(f"❌ {len(result.errors)} tests had errors")
        return False

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)