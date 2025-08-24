# Import all models for easy access
from .user import User
from .patient import Patient
from .audio import AudioRecording, AnalysisResult
from .medical import MedicalRecord

# Export all models
__all__ = [
    'User',
    'Patient', 
    'AudioRecording',
    'AnalysisResult',
    'MedicalRecord'
]