"""
Test models utility for testing database models and functionality
"""

from app import db
from app.models import User, Patient, AudioRecording, AnalysisResult, MedicalRecord
from datetime import datetime, date, timedelta

def test_all_models():
    """Test all database models and their functionality"""
    
    print("Testing all database models...")
    print("=" * 50)
    
    try:
        # Test User model
        test_user_model()
        
        # Test Patient model
        test_patient_model()
        
        # Test AudioRecording model
        test_audio_recording_model()
        
        # Test AnalysisResult model
        test_analysis_result_model()
        
        # Test MedicalRecord model
        test_medical_record_model()
        
        # Test relationships
        test_model_relationships()
        
        print("=" * 50)
        print("All model tests completed successfully!")
        
    except Exception as e:
        print(f"Error during model testing: {e}")
        db.session.rollback()
        raise

def test_user_model():
    """Test User model functionality"""
    
    print("Testing User model...")
    
    # Test user creation
    test_user = User(
        username='testuser',
        email='test@example.com',
        first_name='Test',
        last_name='User',
        role='technician'
    )
    test_user.set_password('testpass123')
    
    db.session.add(test_user)
    db.session.commit()
    
    # Test password checking
    assert test_user.check_password('testpass123') == True
    assert test_user.check_password('wrongpass') == False
    
    # Test role checking
    assert test_user.has_role('technician') == True
    assert test_user.has_role('admin') == False
    assert test_user.can_manage_patients() == True
    assert test_user.can_view_analytics() == False
    
    # Test user methods
    assert test_user.get_full_name() == 'Test User'
    
    # Test to_dict method
    user_dict = test_user.to_dict()
    assert 'username' in user_dict
    assert 'role' in user_dict
    assert 'created_at' in user_dict
    
    # Clean up
    db.session.delete(test_user)
    db.session.commit()
    
    print("✓ User model tests passed")

def test_patient_model():
    """Test Patient model functionality"""
    
    print("Testing Patient model...")
    
    # Create a test user first
    test_user = User(
        username='testuser2',
        email='test2@example.com',
        first_name='Test',
        last_name='User2',
        role='doctor'
    )
    test_user.set_password('testpass123')
    db.session.add(test_user)
    db.session.commit()
    
    # Test patient creation
    test_patient = Patient(
        first_name='John',
        last_name='Doe',
        date_of_birth=date(1990, 1, 1),
        gender='Male',
        phone='+1-555-0123',
        email='john.doe@example.com',
        height_cm=175.0,
        weight_kg=70.0,
        blood_type='O+',
        allergies='None',
        current_medications='None',
        smoking_history='Never',
        symptoms={
            'cough': True,
            'shortness_of_breath': False,
            'wheezing': False
        },
        symptom_duration=5,
        symptom_severity=3,
        created_by_id=test_user.id
    )
    
    db.session.add(test_patient)
    db.session.commit()
    
    # Test calculated fields
    assert test_patient.age is not None
    assert test_patient.bmi is not None
    assert test_patient.patient_id is not None
    
    # Test patient methods
    assert test_patient.get_full_name() == 'John Doe'
    assert test_patient.get_age_display() == f"{test_patient.age} years"
    assert test_patient.get_bmi_category() in ['Underweight', 'Normal weight', 'Overweight', 'Obese']
    assert 'cough' in test_patient.get_symptoms_list()
    
    # Test to_dict method
    patient_dict = test_patient.to_dict()
    assert 'patient_id' in patient_dict
    assert 'full_name' in patient_dict
    assert 'age' in patient_dict
    assert 'bmi_category' in patient_dict
    
    # Clean up
    db.session.delete(test_patient)
    db.session.delete(test_user)
    db.session.commit()
    
    print("✓ Patient model tests passed")

def test_audio_recording_model():
    """Test AudioRecording model functionality"""
    
    print("Testing AudioRecording model...")
    
    # Create test user and patient
    test_user = User(
        username='testuser3',
        email='test3@example.com',
        first_name='Test',
        last_name='User3',
        role='technician'
    )
    test_user.set_password('testpass123')
    db.session.add(test_user)
    
    test_patient = Patient(
        first_name='Jane',
        last_name='Smith',
        date_of_birth=date(1985, 5, 15),
        gender='Female',
        created_by_id=test_user.id
    )
    db.session.add(test_patient)
    db.session.commit()
    
    # Test audio recording creation
    test_recording = AudioRecording(
        filename='test_recording.wav',
        original_filename='test_recording_original.wav',
        file_path='/tmp/test_recording.wav',
        file_size=1024000,  # 1MB
        duration=30.5,
        sample_rate=22050,
        bit_depth=16,
        channels=1,
        format='wav',
        recording_location='anterior',
        equipment_used='Digital Stethoscope',
        audio_quality_score=0.85,
        noise_level=0.15,
        signal_strength=0.90,
        patient_id=test_patient.id,
        recorded_by_id=test_user.id
    )
    
    db.session.add(test_recording)
    db.session.commit()
    
    # Test audio recording methods
    assert test_recording.get_file_size_mb() == 1.0
    assert test_recording.get_duration_display() == '0m 30s'
    assert test_recording.get_quality_level() == 'Excellent'
    assert test_recording.get_noise_level_description() == 'Low'
    
    # Test to_dict method
    recording_dict = test_recording.to_dict()
    assert 'filename' in recording_dict
    assert 'duration_display' in recording_dict
    assert 'quality_level' in recording_dict
    assert 'file_size_mb' in recording_dict
    
    # Clean up
    db.session.delete(test_recording)
    db.session.delete(test_patient)
    db.session.delete(test_user)
    db.session.commit()
    
    print("✓ AudioRecording model tests passed")

def test_analysis_result_model():
    """Test AnalysisResult model functionality"""
    
    print("Testing AnalysisResult model...")
    
    # Create test user, patient, and audio recording
    test_user = User(
        username='testuser4',
        email='test4@example.com',
        first_name='Test',
        last_name='User4',
        role='doctor'
    )
    test_user.set_password('testpass123')
    db.session.add(test_user)
    
    test_patient = Patient(
        first_name='Bob',
        last_name='Johnson',
        date_of_birth=date(1975, 8, 20),
        gender='Male',
        created_by_id=test_user.id
    )
    db.session.add(test_patient)
    
    test_recording = AudioRecording(
        filename='test_recording2.wav',
        original_filename='test_recording2_original.wav',
        file_path='/tmp/test_recording2.wav',
        duration=25.0,
        patient_id=test_patient.id,
        recorded_by_id=test_user.id
    )
    db.session.add(test_recording)
    db.session.commit()
    
    # Test analysis result creation
    test_result = AnalysisResult(
        cnn_prediction='asthma',
        cnn_confidence=0.85,
        cnn_probabilities={'healthy': 0.05, 'asthma': 0.85, 'pneumonia': 0.10},
        lstm_prediction='asthma',
        lstm_confidence=0.78,
        lstm_probabilities={'healthy': 0.08, 'asthma': 0.78, 'pneumonia': 0.14},
        hybrid_prediction='asthma',
        hybrid_confidence=0.92,
        hybrid_probabilities={'healthy': 0.03, 'asthma': 0.92, 'pneumonia': 0.05},
        ensemble_prediction='asthma',
        ensemble_confidence=0.88,
        risk_level='Medium',
        clinical_notes='Patient shows signs of mild asthma',
        recommendations='Prescribe albuterol inhaler, follow up in 2 weeks',
        audio_recording_id=test_recording.id,
        patient_id=test_patient.id,
        analyzed_by_id=test_user.id,
        processing_time=12.5
    )
    
    db.session.add(test_result)
    db.session.commit()
    
    # Test analysis result methods
    primary_prediction, primary_confidence = test_result.get_primary_prediction()
    assert primary_prediction == 'asthma'
    assert primary_confidence == 0.92
    
    assert test_result.get_model_agreement() == True
    assert test_result.get_risk_assessment() == 'Medium risk - Some abnormalities detected, follow-up recommended'
    
    confidence_ranking = test_result.get_confidence_ranking()
    assert len(confidence_ranking) == 3
    assert confidence_ranking[0][0] == 'Hybrid'
    
    # Test to_dict method
    result_dict = test_result.to_dict()
    assert 'primary_prediction' in result_dict
    assert 'model_agreement' in result_dict
    assert 'confidence_ranking' in result_dict
    assert 'risk_assessment' in result_dict
    
    # Clean up
    db.session.delete(test_result)
    db.session.delete(test_recording)
    db.session.delete(test_patient)
    db.session.delete(test_user)
    db.session.commit()
    
    print("✓ AnalysisResult model tests passed")

def test_medical_record_model():
    """Test MedicalRecord model functionality"""
    
    print("Testing MedicalRecord model...")
    
    # Create test user and patient
    test_user = User(
        username='testuser5',
        email='test5@example.com',
        first_name='Test',
        last_name='User5',
        role='doctor'
    )
    test_user.set_password('testpass123')
    db.session.add(test_user)
    
    test_patient = Patient(
        first_name='Alice',
        last_name='Brown',
        date_of_birth=date(1980, 12, 10),
        gender='Female',
        created_by_id=test_user.id
    )
    db.session.add(test_patient)
    db.session.commit()
    
    # Test medical record creation
    test_record = MedicalRecord(
        patient_id=test_patient.id,
        record_type='initial_consultation',
        record_date=datetime.utcnow(),
        next_follow_up=datetime.utcnow() + timedelta(days=30),
        chief_complaint='Chest tightness and shortness of breath',
        present_illness='Patient reports chest tightness for the past week',
        vital_signs={
            'heart_rate': 80,
            'blood_pressure': '120/80',
            'oxygen_saturation': 96,
            'respiratory_rate': 18,
            'temperature': 36.8
        },
        respiratory_examination='Clear breath sounds, no wheezing',
        primary_diagnosis='Anxiety-related chest tightness',
        severity_assessment='Mild',
        treatment_recommendations='Breathing exercises, stress management',
        created_by_id=test_user.id
    )
    
    db.session.add(test_record)
    db.session.commit()
    
    # Test medical record methods
    assert test_record.get_record_type_display() == 'Initial Consultation'
    assert 'Heart Rate' in test_record.get_vital_signs_display()
    assert test_record.get_follow_up_status() == 'Follow-up in 30 days'
    assert test_record.is_overdue() == False
    assert test_record.get_urgency_level() == 'Routine'
    
    # Test to_dict method
    record_dict = test_record.to_dict()
    assert 'record_type_display' in record_dict
    assert 'vital_signs_display' in record_dict
    assert 'follow_up_status' in record_dict
    assert 'urgency_level' in record_dict
    
    # Clean up
    db.session.delete(test_record)
    db.session.delete(test_patient)
    db.session.delete(test_user)
    db.session.commit()
    
    print("✓ MedicalRecord model tests passed")

def test_model_relationships():
    """Test relationships between models"""
    
    print("Testing model relationships...")
    
    # Create test user
    test_user = User(
        username='testuser6',
        email='test6@example.com',
        first_name='Test',
        last_name='User6',
        role='doctor'
    )
    test_user.set_password('testpass123')
    db.session.add(test_user)
    
    # Create test patient
    test_patient = Patient(
        first_name='Charlie',
        last_name='Davis',
        date_of_birth=date(1995, 3, 25),
        gender='Male',
        created_by_id=test_user.id
    )
    db.session.add(test_patient)
    
    # Create test audio recording
    test_recording = AudioRecording(
        filename='test_recording3.wav',
        original_filename='test_recording3_original.wav',
        file_path='/tmp/test_recording3.wav',
        duration=20.0,
        patient_id=test_patient.id,
        recorded_by_id=test_user.id
    )
    db.session.add(test_recording)
    
    # Create test analysis result
    test_result = AnalysisResult(
        cnn_prediction='healthy',
        cnn_confidence=0.90,
        hybrid_prediction='healthy',
        hybrid_confidence=0.88,
        audio_recording_id=test_recording.id,
        patient_id=test_patient.id,
        analyzed_by_id=test_user.id
    )
    db.session.add(test_result)
    
    # Create test medical record
    test_record = MedicalRecord(
        patient_id=test_patient.id,
        record_type='routine_checkup',
        record_date=datetime.utcnow(),
        created_by_id=test_user.id
    )
    db.session.add(test_record)
    
    db.session.commit()
    
    # Test relationships
    assert test_user.patients.count() == 1
    assert test_user.audio_recordings.count() == 1
    assert test_user.analysis_results.count() == 1
    
    assert test_patient.audio_recordings.count() == 1
    assert test_patient.analysis_results.count() == 1
    assert test_patient.medical_records.count() == 1
    
    assert test_recording.analysis_results.count() == 1
    assert test_recording.patient == test_patient
    assert test_recording.recorded_by == test_user
    
    assert test_result.audio_recording == test_recording
    assert test_result.patient == test_patient
    assert test_result.analyzed_by == test_user
    
    assert test_record.patient == test_patient
    assert test_record.created_by == test_user
    
    # Clean up
    db.session.delete(test_result)
    db.session.delete(test_record)
    db.session.delete(test_recording)
    db.session.delete(test_patient)
    db.session.delete(test_user)
    db.session.commit()
    
    print("✓ Model relationship tests passed")

if __name__ == '__main__':
    from app import create_app
    
    app = create_app()
    with app.app_context():
        test_all_models()