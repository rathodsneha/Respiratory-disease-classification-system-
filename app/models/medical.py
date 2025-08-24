from datetime import datetime
from app import db

class MedicalRecord(db.Model):
    """Medical record model for storing comprehensive patient medical history"""
    
    __tablename__ = 'medical_records'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Patient Information
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    
    # Record Information
    record_type = db.Column(db.String(50), nullable=False)  # consultation, follow_up, emergency, etc.
    record_date = db.Column(db.DateTime, nullable=False)
    next_follow_up = db.Column(db.DateTime)
    
    # Clinical Assessment
    chief_complaint = db.Column(db.Text)
    present_illness = db.Column(db.Text)
    past_medical_history = db.Column(db.Text)
    family_history = db.Column(db.Text)
    social_history = db.Column(db.Text)
    
    # Physical Examination
    vital_signs = db.Column(db.JSON)  # Store vitals as JSON
    general_appearance = db.Column(db.Text)
    respiratory_examination = db.Column(db.Text)
    cardiovascular_examination = db.Column(db.Text)
    other_systems = db.Column(db.JSON)  # Store other system exams
    
    # Diagnostic Tests
    laboratory_tests = db.Column(db.JSON)  # Store lab results
    imaging_studies = db.Column(db.JSON)  # Store imaging results
    pulmonary_function_tests = db.Column(db.JSON)  # Store PFT results
    
    # Diagnosis and Assessment
    primary_diagnosis = db.Column(db.String(100))
    secondary_diagnoses = db.Column(db.JSON)  # Store multiple diagnoses
    differential_diagnosis = db.Column(db.JSON)  # Store differential diagnoses
    severity_assessment = db.Column(db.String(20))  # Mild, Moderate, Severe
    
    # Treatment Plan
    medications_prescribed = db.Column(db.JSON)  # Store medications
    treatment_recommendations = db.Column(db.Text)
    lifestyle_modifications = db.Column(db.Text)
    referral_recommendations = db.Column(db.Text)
    
    # Follow-up Plan
    follow_up_plan = db.Column(db.Text)
    monitoring_parameters = db.Column(db.JSON)  # Store monitoring requirements
    warning_signs = db.Column(db.Text)
    
    # Clinical Notes
    physician_notes = db.Column(db.Text)
    nursing_notes = db.Column(db.Text)
    patient_education = db.Column(db.Text)
    
    # System Fields
    created_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    created_by = db.relationship('User', backref='medical_records_created')
    
    # Record type options
    RECORD_TYPES = [
        'initial_consultation',
        'follow_up',
        'emergency_visit',
        'routine_checkup',
        'specialist_consultation',
        'hospital_admission',
        'discharge_summary',
        'procedure_note',
        'lab_result_review',
        'imaging_review'
    ]
    
    # Severity levels
    SEVERITY_LEVELS = ['Mild', 'Moderate', 'Severe', 'Critical']
    
    def __init__(self, **kwargs):
        super(MedicalRecord, self).__init__(**kwargs)
        if not self.record_date:
            self.record_date = datetime.utcnow()
    
    def get_record_type_display(self):
        """Get human-readable record type"""
        type_mapping = {
            'initial_consultation': 'Initial Consultation',
            'follow_up': 'Follow-up Visit',
            'emergency_visit': 'Emergency Visit',
            'routine_checkup': 'Routine Checkup',
            'specialist_consultation': 'Specialist Consultation',
            'hospital_admission': 'Hospital Admission',
            'discharge_summary': 'Discharge Summary',
            'procedure_note': 'Procedure Note',
            'lab_result_review': 'Lab Result Review',
            'imaging_review': 'Imaging Review'
        }
        return type_mapping.get(self.record_type, self.record_type.replace('_', ' ').title())
    
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
        if 'respiratory_rate' in self.vital_signs:
            vitals['Respiratory Rate'] = f"{self.vital_signs['respiratory_rate']} breaths/min"
        if 'weight' in self.vital_signs:
            vitals['Weight'] = f"{self.vital_signs['weight']} kg"
        
        return vitals
    
    def get_medications_display(self):
        """Get formatted medications list"""
        if not self.medications_prescribed:
            return []
        
        medications = []
        for med in self.medications_prescribed:
            if isinstance(med, dict):
                med_info = f"{med.get('name', 'Unknown')}"
                if med.get('dosage'):
                    med_info += f" - {med['dosage']}"
                if med.get('frequency'):
                    med_info += f" {med['frequency']}"
                if med.get('duration'):
                    med_info += f" for {med['duration']}"
                medications.append(med_info)
            else:
                medications.append(str(med))
        
        return medications
    
    def get_diagnoses_list(self):
        """Get list of all diagnoses"""
        diagnoses = []
        
        if self.primary_diagnosis:
            diagnoses.append(f"Primary: {self.primary_diagnosis}")
        
        if self.secondary_diagnoses:
            for i, diagnosis in enumerate(self.secondary_diagnoses, 1):
                diagnoses.append(f"Secondary {i}: {diagnosis}")
        
        return diagnoses
    
    def get_follow_up_status(self):
        """Get follow-up status"""
        if not self.next_follow_up:
            return "No follow-up scheduled"
        
        if self.next_follow_up < datetime.utcnow():
            return "Follow-up overdue"
        elif self.next_follow_up.date() == datetime.utcnow().date():
            return "Follow-up today"
        else:
            days_until = (self.next_follow_up - datetime.utcnow()).days
            return f"Follow-up in {days_until} days"
    
    def is_overdue(self):
        """Check if follow-up is overdue"""
        if self.next_follow_up:
            return self.next_follow_up < datetime.utcnow()
        return False
    
    def get_urgency_level(self):
        """Get urgency level based on severity and follow-up status"""
        if self.severity_assessment == 'Critical':
            return 'Immediate'
        elif self.severity_assessment == 'Severe':
            return 'Urgent'
        elif self.is_overdue():
            return 'Overdue'
        elif self.severity_assessment == 'Moderate':
            return 'Moderate'
        else:
            return 'Routine'
    
    def to_dict(self):
        """Convert medical record to dictionary"""
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'record_type': self.record_type,
            'record_type_display': self.get_record_type_display(),
            'record_date': self.record_date.isoformat() if self.record_date else None,
            'next_follow_up': self.next_follow_up.isoformat() if self.next_follow_up else None,
            'chief_complaint': self.chief_complaint,
            'present_illness': self.present_illness,
            'past_medical_history': self.past_medical_history,
            'family_history': self.family_history,
            'social_history': self.social_history,
            'vital_signs': self.vital_signs,
            'vital_signs_display': self.get_vital_signs_display(),
            'general_appearance': self.general_appearance,
            'respiratory_examination': self.respiratory_examination,
            'cardiovascular_examination': self.cardiovascular_examination,
            'other_systems': self.other_systems,
            'laboratory_tests': self.laboratory_tests,
            'imaging_studies': self.imaging_studies,
            'pulmonary_function_tests': self.pulmonary_function_tests,
            'primary_diagnosis': self.primary_diagnosis,
            'secondary_diagnoses': self.secondary_diagnoses,
            'differential_diagnosis': self.differential_diagnosis,
            'severity_assessment': self.severity_assessment,
            'medications_prescribed': self.medications_prescribed,
            'medications_display': self.get_medications_display(),
            'treatment_recommendations': self.treatment_recommendations,
            'lifestyle_modifications': self.lifestyle_modifications,
            'referral_recommendations': self.referral_recommendations,
            'follow_up_plan': self.follow_up_plan,
            'monitoring_parameters': self.monitoring_parameters,
            'warning_signs': self.warning_signs,
            'physician_notes': self.physician_notes,
            'nursing_notes': self.nursing_notes,
            'patient_education': self.patient_education,
            'created_by_id': self.created_by_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_active': self.is_active,
            'diagnoses_list': self.get_diagnoses_list(),
            'follow_up_status': self.get_follow_up_status(),
            'is_overdue': self.is_overdue(),
            'urgency_level': self.get_urgency_level()
        }
    
    def __repr__(self):
        return f'<MedicalRecord {self.id}: {self.get_record_type_display()} - {self.record_date.date()}>'