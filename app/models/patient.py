from datetime import datetime, date
from app import db

class Patient(db.Model):
    """Patient model for storing patient information and medical history"""
    
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(20), unique=True, nullable=False, index=True)
    first_name = db.Column(db.String(64), nullable=False)
    last_name = db.Column(db.String(64), nullable=False)
    date_of_birth = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    age = db.Column(db.Integer)
    
    # Contact Information
    phone = db.Column(db.String(20))
    email = db.Column(db.String(120))
    address = db.Column(db.Text)
    emergency_contact_name = db.Column(db.String(128))
    emergency_contact_phone = db.Column(db.String(20))
    emergency_contact_relationship = db.Column(db.String(50))
    
    # Medical Information
    height_cm = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    bmi = db.Column(db.Float)
    blood_type = db.Column(db.String(5))
    allergies = db.Column(db.Text)
    current_medications = db.Column(db.Text)
    
    # Respiratory History
    smoking_history = db.Column(db.String(20))  # Never, Former, Current
    smoking_years = db.Column(db.Integer)
    pack_years = db.Column(db.Float)
    occupational_exposure = db.Column(db.Text)
    family_respiratory_history = db.Column(db.Text)
    
    # Clinical Assessment
    symptoms = db.Column(db.JSON)  # Store symptoms as JSON
    symptom_duration = db.Column(db.Integer)  # Days
    symptom_severity = db.Column(db.Integer)  # 1-10 scale
    vital_signs = db.Column(db.JSON)  # Store vitals as JSON
    
    # System Fields
    created_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    notes = db.Column(db.Text)
    
    # Relationships
    audio_recordings = db.relationship('AudioRecording', backref='patient', lazy='dynamic')
    analysis_results = db.relationship('AnalysisResult', backref='patient', lazy='dynamic')
    medical_records = db.relationship('MedicalRecord', backref='patient', lazy='dynamic')
    
    # Gender options
    GENDER_CHOICES = ['Male', 'Female', 'Other', 'Prefer not to say']
    
    # Blood type options
    BLOOD_TYPES = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    
    # Smoking history options
    SMOKING_CHOICES = ['Never', 'Former', 'Current']
    
    def __init__(self, **kwargs):
        super(Patient, self).__init__(**kwargs)
        self.calculate_age()
        self.calculate_bmi()
        self.generate_patient_id()
    
    def generate_patient_id(self):
        """Generate unique patient ID"""
        if not self.patient_id:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            self.patient_id = f"P{timestamp}"
    
    def calculate_age(self):
        """Calculate age from date of birth"""
        if self.date_of_birth:
            today = date.today()
            self.age = today.year - self.date_of_birth.year - (
                (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
            )
    
    def calculate_bmi(self):
        """Calculate BMI from height and weight"""
        if self.height_cm and self.weight_kg:
            height_m = self.height_cm / 100
            self.bmi = round(self.weight_kg / (height_m ** 2), 2)
    
    def get_full_name(self):
        """Get patient's full name"""
        return f"{self.first_name} {self.last_name}"
    
    def get_age_display(self):
        """Get age with proper display format"""
        if self.age:
            return f"{self.age} years"
        return "Unknown"
    
    def get_bmi_category(self):
        """Get BMI category"""
        if not self.bmi:
            return "Unknown"
        
        if self.bmi < 18.5:
            return "Underweight"
        elif self.bmi < 25:
            return "Normal weight"
        elif self.bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def get_symptoms_list(self):
        """Get list of symptoms"""
        if isinstance(self.symptoms, dict):
            return [symptom for symptom, present in self.symptoms.items() if present]
        return []
    
    def get_vital_signs_display(self):
        """Get formatted vital signs"""
        if not self.vital_signs:
            return {}
        
        vitals = {}
        if 'heart_rate' in self.vital_signs:
            vitals['Heart Rate'] = f"{self.vital_signs['heart_rate']} bpm"
        if 'blood_pressure' in self.vital_signs:
            vitals['Blood Pressure'] = f"{self.vital_signs['blood_pressure']} mmHg"
        if 'oxygen_saturation' in self.vital_signs:
            vitals['O2 Saturation'] = f"{self.vital_signs['oxygen_saturation']}%"
        if 'temperature' in self.vital_signs:
            vitals['Temperature'] = f"{self.vital_signs['temperature']}°C"
        
        return vitals
    
    def to_dict(self):
        """Convert patient to dictionary"""
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': self.get_full_name(),
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'age': self.age,
            'age_display': self.get_age_display(),
            'gender': self.gender,
            'phone': self.phone,
            'email': self.email,
            'address': self.address,
            'emergency_contact_name': self.emergency_contact_name,
            'emergency_contact_phone': self.emergency_contact_phone,
            'emergency_contact_relationship': self.emergency_contact_relationship,
            'height_cm': self.height_cm,
            'weight_kg': self.weight_kg,
            'bmi': self.bmi,
            'bmi_category': self.get_bmi_category(),
            'blood_type': self.blood_type,
            'allergies': self.allergies,
            'current_medications': self.current_medications,
            'smoking_history': self.smoking_history,
            'smoking_years': self.smoking_years,
            'pack_years': self.pack_years,
            'occupational_exposure': self.occupational_exposure,
            'family_respiratory_history': self.family_respiratory_history,
            'symptoms': self.symptoms,
            'symptoms_list': self.get_symptoms_list(),
            'symptom_duration': self.symptom_duration,
            'symptom_severity': self.symptom_severity,
            'vital_signs': self.vital_signs,
            'vital_signs_display': self.get_vital_signs_display(),
            'created_by_id': self.created_by_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_active': self.is_active,
            'notes': self.notes
        }
    
    def __repr__(self):
        return f'<Patient {self.patient_id}: {self.get_full_name()}>'