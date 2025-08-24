from datetime import datetime
import os
from app import db
from config import get_config

class AudioRecording(db.Model):
    """Audio recording model for storing respiratory audio files"""
    
    __tablename__ = 'audio_recordings'
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer)  # Size in bytes
    duration = db.Column(db.Float)  # Duration in seconds
    sample_rate = db.Column(db.Integer)
    bit_depth = db.Column(db.Integer)
    channels = db.Column(db.Integer)
    format = db.Column(db.String(10))  # wav, mp3, flac, etc.
    
    # Recording Information
    recording_date = db.Column(db.DateTime, nullable=False)
    recording_location = db.Column(db.String(50))  # anterior, posterior, lateral
    equipment_used = db.Column(db.String(100))
    environmental_conditions = db.Column(db.Text)
    technician_notes = db.Column(db.Text)
    
    # Quality Assessment
    audio_quality_score = db.Column(db.Float)  # 0-1 scale
    noise_level = db.Column(db.Float)  # 0-1 scale
    signal_strength = db.Column(db.Float)  # 0-1 scale
    
    # Relationships
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    recorded_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    analysis_results = db.relationship('AnalysisResult', backref='audio_recording', lazy='dynamic')
    
    # System Fields
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_processed = db.Column(db.Boolean, default=False)
    is_archived = db.Column(db.Boolean, default=False)
    
    # Recording location options
    LOCATIONS = ['anterior', 'posterior', 'lateral', 'other']
    
    def __init__(self, **kwargs):
        super(AudioRecording, self).__init__(**kwargs)
        if not self.recording_date:
            self.recording_date = datetime.utcnow()
    
    def get_file_size_mb(self):
        """Get file size in MB"""
        if self.file_size:
            return round(self.file_size / (1024 * 1024), 2)
        return 0
    
    def get_duration_display(self):
        """Get duration in human-readable format"""
        if not self.duration:
            return "Unknown"
        
        minutes = int(self.duration // 60)
        seconds = int(self.duration % 60)
        return f"{minutes}m {seconds}s"
    
    def get_quality_level(self):
        """Get quality level description"""
        if not self.audio_quality_score:
            return "Unknown"
        
        if self.audio_quality_score >= 0.8:
            return "Excellent"
        elif self.audio_quality_score >= 0.6:
            return "Good"
        elif self.audio_quality_score >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def get_noise_level_description(self):
        """Get noise level description"""
        if not self.noise_level:
            return "Unknown"
        
        if self.noise_level <= 0.2:
            return "Low"
        elif self.noise_level <= 0.5:
            return "Moderate"
        else:
            return "High"
    
    def file_exists(self):
        """Check if the audio file exists on disk"""
        return os.path.exists(self.file_path)
    
    def delete_file(self):
        """Delete the audio file from disk"""
        try:
            if self.file_exists():
                os.remove(self.file_path)
                return True
        except OSError:
            return False
        return False
    
    def to_dict(self):
        """Convert audio recording to dictionary"""
        return {
            'id': self.id,
            'filename': self.filename,
            'original_filename': self.original_filename,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'file_size_mb': self.get_file_size_mb(),
            'duration': self.duration,
            'duration_display': self.get_duration_display(),
            'sample_rate': self.sample_rate,
            'bit_depth': self.bit_depth,
            'channels': self.channels,
            'format': self.format,
            'recording_date': self.recording_date.isoformat() if self.recording_date else None,
            'recording_location': self.recording_location,
            'equipment_used': self.equipment_used,
            'environmental_conditions': self.environmental_conditions,
            'technician_notes': self.technician_notes,
            'audio_quality_score': self.audio_quality_score,
            'quality_level': self.get_quality_level(),
            'noise_level': self.noise_level,
            'noise_level_description': self.get_noise_level_description(),
            'signal_strength': self.signal_strength,
            'patient_id': self.patient_id,
            'recorded_by_id': self.recorded_by_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'is_processed': self.is_processed,
            'is_archived': self.is_archived,
            'file_exists': self.file_exists()
        }
    
    def __repr__(self):
        return f'<AudioRecording {self.filename}: {self.get_duration_display()}>'

class AnalysisResult(db.Model):
    """Analysis result model for storing model predictions and analysis data"""
    
    __tablename__ = 'analysis_results'
    
    id = db.Column(db.Integer, primary_key=True)
    
    # Model Predictions
    cnn_prediction = db.Column(db.String(50))
    cnn_confidence = db.Column(db.Float)
    cnn_probabilities = db.Column(db.JSON)
    
    lstm_prediction = db.Column(db.String(50))
    lstm_confidence = db.Column(db.Float)
    lstm_probabilities = db.Column(db.JSON)
    
    hybrid_prediction = db.Column(db.String(50))
    hybrid_confidence = db.Column(db.Float)
    hybrid_probabilities = db.Column(db.JSON)
    
    # Ensemble Results
    ensemble_prediction = db.Column(db.String(50))
    ensemble_confidence = db.Column(db.Float)
    ensemble_probabilities = db.Column(db.JSON)
    
    # Feature Analysis
    mfcc_features = db.Column(db.JSON)
    spectral_features = db.Column(db.JSON)
    temporal_features = db.Column(db.JSON)
    
    # Visualization Data
    spectrogram_path = db.Column(db.String(500))
    waveform_path = db.Column(db.String(500))
    feature_importance = db.Column(db.JSON)
    
    # Clinical Assessment
    risk_level = db.Column(db.String(20))  # Low, Medium, High
    clinical_notes = db.Column(db.Text)
    recommendations = db.Column(db.Text)
    
    # Relationships
    audio_recording_id = db.Column(db.Integer, db.ForeignKey('audio_recordings.id'), nullable=False)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'), nullable=False)
    analyzed_by_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # System Fields
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processing_time = db.Column(db.Float)  # Processing time in seconds
    model_versions = db.Column(db.JSON)  # Store model versions used
    
    # Risk level options
    RISK_LEVELS = ['Low', 'Medium', 'High']
    
    def __init__(self, **kwargs):
        super(AnalysisResult, self).__init__(**kwargs)
    
    def get_primary_prediction(self):
        """Get the primary prediction (highest confidence)"""
        predictions = [
            (self.cnn_prediction, self.cnn_confidence),
            (self.lstm_prediction, self.lstm_confidence),
            (self.hybrid_prediction, self.hybrid_confidence)
        ]
        
        # Filter out None predictions and find highest confidence
        valid_predictions = [(pred, conf) for pred, conf in predictions if pred and conf]
        if valid_predictions:
            return max(valid_predictions, key=lambda x: x[1])
        return None, None
    
    def get_model_agreement(self):
        """Check if all models agree on prediction"""
        predictions = [self.cnn_prediction, self.lstm_prediction, self.hybrid_prediction]
        valid_predictions = [p for p in predictions if p]
        
        if len(valid_predictions) < 2:
            return False
        
        return len(set(valid_predictions)) == 1
    
    def get_confidence_ranking(self):
        """Get models ranked by confidence"""
        models = [
            ('CNN', self.cnn_confidence),
            ('LSTM', self.lstm_confidence),
            ('Hybrid', self.hybrid_confidence)
        ]
        
        # Filter out None confidences and sort by confidence
        valid_models = [(name, conf) for name, conf in models if conf is not None]
        return sorted(valid_models, key=lambda x: x[1], reverse=True)
    
    def get_risk_assessment(self):
        """Get comprehensive risk assessment"""
        if not self.risk_level:
            return "Not assessed"
        
        risk_descriptions = {
            'Low': 'Low risk - Normal respiratory patterns detected',
            'Medium': 'Medium risk - Some abnormalities detected, follow-up recommended',
            'High': 'High risk - Significant abnormalities detected, immediate attention required'
        }
        
        return risk_descriptions.get(self.risk_level, 'Unknown risk level')
    
    def to_dict(self):
        """Convert analysis result to dictionary"""
        primary_prediction, primary_confidence = self.get_primary_prediction()
        
        return {
            'id': self.id,
            'cnn_prediction': self.cnn_prediction,
            'cnn_confidence': self.cnn_confidence,
            'cnn_probabilities': self.cnn_probabilities,
            'lstm_prediction': self.lstm_prediction,
            'lstm_confidence': self.lstm_confidence,
            'lstm_probabilities': self.lstm_probabilities,
            'hybrid_prediction': self.hybrid_prediction,
            'hybrid_confidence': self.hybrid_confidence,
            'hybrid_probabilities': self.hybrid_probabilities,
            'ensemble_prediction': self.ensemble_prediction,
            'ensemble_confidence': self.ensemble_confidence,
            'ensemble_probabilities': self.ensemble_probabilities,
            'primary_prediction': primary_prediction,
            'primary_confidence': primary_confidence,
            'model_agreement': self.get_model_agreement(),
            'confidence_ranking': self.get_confidence_ranking(),
            'mfcc_features': self.mfcc_features,
            'spectral_features': self.spectral_features,
            'temporal_features': self.temporal_features,
            'spectrogram_path': self.spectrogram_path,
            'waveform_path': self.waveform_path,
            'feature_importance': self.feature_importance,
            'risk_level': self.risk_level,
            'risk_assessment': self.get_risk_assessment(),
            'clinical_notes': self.clinical_notes,
            'recommendations': self.recommendations,
            'audio_recording_id': self.audio_recording_id,
            'patient_id': self.patient_id,
            'analyzed_by_id': self.analyzed_by_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'processing_time': self.processing_time,
            'model_versions': self.model_versions
        }
    
    def __repr__(self):
        return f'<AnalysisResult {self.id}: {self.primary_prediction}>'