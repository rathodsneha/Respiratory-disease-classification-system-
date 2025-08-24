from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from app import db

class Patient(db.Model):
    """Patient information model"""
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Auto-generated unique identifier
    patient_id = db.Column(db.String(20), unique=True, nullable=False, index=True)
    
    # Personal Information
    first_name = db.Column(db.String(64), nullable=False)
    last_name = db.Column(db.String(64), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)  # Male, Female, Other
    
    # Contact Information
    phone = db.Column(db.String(20))
    email = db.Column(db.String(120))
    address = db.Column(db.Text)
    
    # Emergency Contact
    emergency_contact_name = db.Column(db.String(128))
    emergency_contact_phone = db.Column(db.String(20))
    emergency_contact_relationship = db.Column(db.String(64))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    medical_history = db.relationship('MedicalHistory', backref='patient', lazy='dynamic')
    clinical_assessments = db.relationship('ClinicalAssessment', backref='patient', lazy='dynamic')
    recording_sessions = db.relationship('RecordingSession', backref='patient', lazy='dynamic')
    
    def __repr__(self):
        return f'<Patient {self.patient_id}: {self.get_full_name()}>'
    
    def get_full_name(self):
        """Get patient's full name"""
        return f"{self.first_name} {self.last_name}"
    
    def get_age(self):
        """Calculate patient's age"""
        from datetime import date
        today = date.today()
        return today.year - self.date_of_birth.year - ((today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day))
    
    @staticmethod
    def generate_patient_id():
        """Generate unique patient ID"""
        import random
        import string
        while True:
            patient_id = 'P' + ''.join(random.choices(string.digits, k=7))
            if not Patient.query.filter_by(patient_id=patient_id).first():
                return patient_id

class MedicalHistory(db.Model):
    """Patient medical history"""
    __tablename__ = 'medical_history'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    
    # Previous Respiratory Conditions
    has_asthma = db.Column(db.Boolean, default=False)
    has_copd = db.Column(db.Boolean, default=False)
    has_pneumonia_history = db.Column(db.Boolean, default=False)
    has_bronchitis_history = db.Column(db.Boolean, default=False)
    has_pleural_effusion_history = db.Column(db.Boolean, default=False)
    other_respiratory_conditions = db.Column(db.Text)
    
    # Current Medications
    current_medications = db.Column(db.Text)
    
    # Allergies and Sensitivities
    allergies = db.Column(db.Text)
    drug_sensitivities = db.Column(db.Text)
    
    # Smoking History
    smoking_status = db.Column(db.String(20))  # Never, Former, Current
    cigarettes_per_day = db.Column(db.Integer)
    years_smoked = db.Column(db.Integer)
    quit_date = db.Column(db.Date)
    
    # Occupational Exposure
    occupational_exposure = db.Column(db.Text)
    exposure_duration = db.Column(db.String(64))
    
    # Family Medical History
    family_respiratory_history = db.Column(db.Text)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<MedicalHistory for Patient {self.patient.patient_id}>'

class ClinicalAssessment(db.Model):
    """Clinical assessment for each visit"""
    __tablename__ = 'clinical_assessments'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    
    # Assessment Date
    assessment_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Symptoms Checklist
    has_cough = db.Column(db.Boolean, default=False)
    cough_severity = db.Column(db.Integer)  # 1-10 scale
    has_shortness_of_breath = db.Column(db.Boolean, default=False)
    shortness_severity = db.Column(db.Integer)  # 1-10 scale
    has_wheezing = db.Column(db.Boolean, default=False)
    wheezing_severity = db.Column(db.Integer)  # 1-10 scale
    has_chest_pain = db.Column(db.Boolean, default=False)
    chest_pain_severity = db.Column(db.Integer)  # 1-10 scale
    has_fever = db.Column(db.Boolean, default=False)
    has_fatigue = db.Column(db.Boolean, default=False)
    
    # Symptom Duration
    symptom_duration = db.Column(db.String(64))  # e.g., "2 weeks", "3 days"
    
    # Physical Examination
    examination_notes = db.Column(db.Text)
    
    # Vital Signs
    heart_rate = db.Column(db.Integer)
    blood_pressure_systolic = db.Column(db.Integer)
    blood_pressure_diastolic = db.Column(db.Integer)
    oxygen_saturation = db.Column(db.Float)
    temperature = db.Column(db.Float)
    respiratory_rate = db.Column(db.Integer)
    
    # Body Mass Index
    height_cm = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    bmi = db.Column(db.Float)
    
    # Healthcare Provider
    assessed_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ClinicalAssessment for Patient {self.patient.patient_id} on {self.assessment_date}>'
    
    def calculate_bmi(self):
        """Calculate BMI from height and weight"""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            self.bmi = round(self.weight_kg / (height_m ** 2), 1)

class RecordingSession(db.Model):
    """Audio recording session details"""
    __tablename__ = 'recording_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    
    # Session Information
    session_date = db.Column(db.DateTime, default=datetime.utcnow)
    session_id = db.Column(db.String(20), unique=True, nullable=False)
    
    # Recording Details
    recording_location = db.Column(db.String(64))  # anterior, posterior, lateral
    chest_position = db.Column(db.String(64))  # specific position on chest
    
    # Equipment Information
    equipment_used = db.Column(db.String(128))
    microphone_type = db.Column(db.String(64))
    
    # Environmental Conditions
    room_temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    ambient_noise_level = db.Column(db.String(20))  # Low, Medium, High
    
    # Quality Assessment
    recording_quality = db.Column(db.String(20))  # Excellent, Good, Fair, Poor
    quality_notes = db.Column(db.Text)
    
    # Healthcare Provider Notes
    technician_notes = db.Column(db.Text)
    recorded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    audio_files = db.relationship('AudioFile', backref='recording_session', lazy='dynamic')
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<RecordingSession {self.session_id} for Patient {self.patient.patient_id}>'
    
    @staticmethod
    def generate_session_id():
        """Generate unique session ID"""
        import random
        import string
        while True:
            session_id = 'S' + ''.join(random.choices(string.digits, k=7))
            if not RecordingSession.query.filter_by(session_id=session_id).first():
                return session_id

class AudioFile(db.Model):
    """Individual audio file information"""
    __tablename__ = 'audio_files'
    
    id = db.Column(db.Integer, primary_key=True)
    recording_session_id = db.Column(db.Integer, db.ForeignKey('recording_sessions.id'), nullable=False)
    
    # File Information
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255))
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)  # in bytes
    
    # Audio Properties
    format = db.Column(db.String(10))  # wav, mp3, flac, m4a
    duration = db.Column(db.Float)  # in seconds
    sample_rate = db.Column(db.Integer)
    bit_depth = db.Column(db.Integer)
    channels = db.Column(db.Integer)
    
    # Processing Status
    is_processed = db.Column(db.Boolean, default=False)
    processing_status = db.Column(db.String(20))  # pending, processing, completed, failed
    
    # Model Predictions
    predictions = db.relationship('ModelPrediction', backref='audio_file', lazy='dynamic')
    
    # Timestamps
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<AudioFile {self.filename}>'

class ModelPrediction(db.Model):
    """Model prediction results"""
    __tablename__ = 'model_predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    audio_file_id = db.Column(db.Integer, db.ForeignKey('audio_files.id'), nullable=False)
    
    # Model Information
    model_name = db.Column(db.String(64), nullable=False)  # CNN, LSTM, Hybrid
    model_version = db.Column(db.String(32))
    
    # Predictions
    predicted_class = db.Column(db.String(64), nullable=False)
    confidence_score = db.Column(db.Float, nullable=False)  # 0-1
    
    # All class probabilities (JSON format)
    class_probabilities = db.Column(db.Text)  # JSON string
    
    # Risk Assessment
    risk_level = db.Column(db.String(20))  # Low, Medium, High
    
    # Clinical Recommendations
    recommendations = db.Column(db.Text)
    
    # Processing Details
    processing_time = db.Column(db.Float)  # in seconds
    
    # Timestamps
    predicted_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelPrediction {self.model_name}: {self.predicted_class} ({self.confidence_score:.2f})>'