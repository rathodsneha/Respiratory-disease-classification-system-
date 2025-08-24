"""
Seed data utility for initializing the database with sample data
"""

from datetime import datetime, date, timedelta
from app import db
from app.models import User, Patient, AudioRecording, AnalysisResult, MedicalRecord
from werkzeug.security import generate_password_hash

def create_admin_user():
    """Create an admin user if it doesn't exist"""
    
    # Check if admin already exists
    admin = User.query.filter_by(username='admin').first()
    if admin:
        print("Admin user already exists!")
        return admin
    
    # Create admin user
    admin = User(
        username='admin',
        email='admin@respiratory-system.com',
        first_name='System',
        last_name='Administrator',
        role='admin',
        is_active=True,
        is_verified=True
    )
    admin.set_password('admin123')  # Change this in production!
    
    db.session.add(admin)
    db.session.commit()
    
    print(f"Admin user created: {admin.username}")
    return admin

def create_sample_users():
    """Create sample users for different roles"""
    
    users_data = [
        {
            'username': 'doctor1',
            'email': 'doctor1@respiratory-system.com',
            'first_name': 'Dr. Sarah',
            'last_name': 'Johnson',
            'role': 'doctor',
            'password': 'doctor123'
        },
        {
            'username': 'technician1',
            'email': 'tech1@respiratory-system.com',
            'first_name': 'Mike',
            'last_name': 'Chen',
            'role': 'technician',
            'password': 'tech123'
        },
        {
            'username': 'researcher1',
            'email': 'researcher1@respiratory-system.com',
            'first_name': 'Dr. Emily',
            'last_name': 'Rodriguez',
            'role': 'researcher',
            'password': 'research123'
        }
    ]
    
    created_users = []
    for user_data in users_data:
        # Check if user already exists
        existing_user = User.query.filter_by(username=user_data['username']).first()
        if existing_user:
            print(f"User {user_data['username']} already exists!")
            created_users.append(existing_user)
            continue
        
        # Create new user
        user = User(
            username=user_data['username'],
            email=user_data['email'],
            first_name=user_data['first_name'],
            last_name=user_data['last_name'],
            role=user_data['role'],
            is_active=True,
            is_verified=True
        )
        user.set_password(user_data['password'])
        
        db.session.add(user)
        created_users.append(user)
        print(f"User created: {user.username}")
    
    db.session.commit()
    return created_users

def create_sample_patients():
    """Create sample patients with medical information"""
    
    patients_data = [
        {
            'first_name': 'John',
            'last_name': 'Smith',
            'date_of_birth': date(1985, 3, 15),
            'gender': 'Male',
            'phone': '+1-555-0101',
            'email': 'john.smith@email.com',
            'address': '123 Main St, Anytown, USA',
            'height_cm': 175.0,
            'weight_kg': 70.0,
            'blood_type': 'O+',
            'allergies': 'None known',
            'current_medications': 'None',
            'smoking_history': 'Never',
            'symptoms': {
                'cough': True,
                'shortness_of_breath': False,
                'wheezing': False,
                'chest_pain': False
            },
            'symptom_duration': 3,
            'symptom_severity': 4
        },
        {
            'first_name': 'Maria',
            'last_name': 'Garcia',
            'date_of_birth': date(1972, 8, 22),
            'gender': 'Female',
            'phone': '+1-555-0102',
            'email': 'maria.garcia@email.com',
            'address': '456 Oak Ave, Somewhere, USA',
            'height_cm': 162.0,
            'weight_kg': 58.0,
            'blood_type': 'A-',
            'allergies': 'Dust, Pollen',
            'current_medications': 'Albuterol inhaler',
            'smoking_history': 'Former',
            'smoking_years': 15,
            'pack_years': 10.5,
            'symptoms': {
                'cough': True,
                'shortness_of_breath': True,
                'wheezing': True,
                'chest_pain': False
            },
            'symptom_duration': 7,
            'symptom_severity': 6
        },
        {
            'first_name': 'Robert',
            'last_name': 'Wilson',
            'date_of_birth': date(1958, 11, 8),
            'gender': 'Male',
            'phone': '+1-555-0103',
            'email': 'robert.wilson@email.com',
            'address': '789 Pine Rd, Elsewhere, USA',
            'height_cm': 180.0,
            'weight_kg': 85.0,
            'blood_type': 'B+',
            'allergies': 'None known',
            'current_medications': 'Tiotropium, Salmeterol',
            'smoking_history': 'Current',
            'smoking_years': 35,
            'pack_years': 35.0,
            'symptoms': {
                'cough': True,
                'shortness_of_breath': True,
                'wheezing': True,
                'chest_pain': True
            },
            'symptom_duration': 14,
            'symptom_severity': 8
        }
    ]
    
    # Get a user to assign as creator
    creator = User.query.filter_by(role='doctor').first()
    if not creator:
        creator = User.query.filter_by(role='admin').first()
    
    created_patients = []
    for patient_data in patients_data:
        # Check if patient already exists
        existing_patient = Patient.query.filter_by(
            first_name=patient_data['first_name'],
            last_name=patient_data['last_name'],
            date_of_birth=patient_data['date_of_birth']
        ).first()
        
        if existing_patient:
            print(f"Patient {patient_data['first_name']} {patient_data['last_name']} already exists!")
            created_patients.append(existing_patient)
            continue
        
        # Create new patient
        patient = Patient(
            created_by_id=creator.id,
            **patient_data
        )
        
        db.session.add(patient)
        created_patients.append(patient)
        print(f"Patient created: {patient.get_full_name()}")
    
    db.session.commit()
    return created_patients

def create_sample_medical_records():
    """Create sample medical records for patients"""
    
    # Get patients and users
    patients = Patient.query.all()
    doctors = User.query.filter_by(role='doctor').all()
    
    if not patients or not doctors:
        print("No patients or doctors found. Skipping medical records creation.")
        return []
    
    record_types = [
        'initial_consultation',
        'follow_up',
        'routine_checkup'
    ]
    
    created_records = []
    for patient in patients:
        # Create 2-3 medical records per patient
        for i in range(2):
            record_date = datetime.utcnow() - timedelta(days=30 * (i + 1))
            next_follow_up = record_date + timedelta(days=90)
            
            # Rotate through doctors
            doctor = doctors[i % len(doctors)]
            
            medical_record = MedicalRecord(
                patient_id=patient.id,
                record_type=record_types[i % len(record_types)],
                record_date=record_date,
                next_follow_up=next_follow_up,
                chief_complaint="Respiratory symptoms",
                present_illness="Patient presents with respiratory symptoms including cough and shortness of breath.",
                vital_signs={
                    'heart_rate': 75 + (i * 5),
                    'blood_pressure': f"{120 + (i * 5)}/{80 + (i * 2)}",
                    'oxygen_saturation': 95 - (i * 2),
                    'respiratory_rate': 16 + i,
                    'temperature': 36.8 + (i * 0.1)
                },
                respiratory_examination="Clear breath sounds bilaterally. No wheezing or crackles noted.",
                created_by_id=doctor.id
            )
            
            db.session.add(medical_record)
            created_records.append(medical_record)
    
    db.session.commit()
    print(f"Created {len(created_records)} medical records")
    return created_records

def seed_database():
    """Seed the database with all sample data"""
    
    print("Seeding database with sample data...")
    
    try:
        # Create users first
        print("Creating users...")
        create_admin_user()
        create_sample_users()
        
        # Create patients
        print("Creating patients...")
        create_sample_patients()
        
        # Create medical records
        print("Creating medical records...")
        create_sample_medical_records()
        
        print("Database seeding completed successfully!")
        
    except Exception as e:
        print(f"Error seeding database: {e}")
        db.session.rollback()
        raise

if __name__ == '__main__':
    from app import create_app
    
    app = create_app()
    with app.app_context():
        seed_database()