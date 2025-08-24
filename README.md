# Respiratory Disease Classification System

A comprehensive deep learning system for analyzing lung sounds and classifying respiratory diseases using hybrid CNN-LSTM models with advanced audio signal processing and clinical decision support.

## 🏥 Project Overview

This system provides clinical decision support through advanced audio signal analysis and pattern recognition, automatically detecting and classifying various respiratory conditions including:

- **Asthma**
- **Pneumonia** 
- **Bronchitis**
- **COPD (Chronic Obstructive Pulmonary Disease)**
- **Pleural Effusion**
- **Healthy/Normal breathing patterns**

## 🚀 Key Features

### Core Functionality
- **Audio Signal Processing**: Multi-format respiratory audio analysis
- **Triple Model Architecture**: CNN, LSTM, and Hybrid CNN-LSTM models
- **Patient Management**: Comprehensive patient information system
- **Clinical Decision Support**: Confidence scores and recommendations
- **Advanced Visualizations**: Interactive spectrograms, waveforms, and analytics

### Technical Capabilities
- **Real-time Processing**: < 30 seconds per audio file
- **High Accuracy**: 85-95% classification accuracy
- **Multi-format Support**: WAV, MP3, FLAC, M4A
- **Scalable Architecture**: Support for 50+ concurrent users

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   Backend API   │    │   ML Pipeline   │
│                 │    │                 │    │                 │
│ • Patient Mgmt  │◄──►│ • Flask Server  │◄──►│ • CNN Model     │
│ • Audio Upload  │    │ • Authentication│    │ • LSTM Model    │
│ • Visualization │    │ • File Storage  │    │ • Hybrid Model  │
│ • Analytics     │    │ • Database      │    │ • Feature Ext.  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** with Flask framework
- **PostgreSQL/SQLite** database
- **TensorFlow/Keras** for deep learning models
- **Librosa** for audio processing
- **JWT** authentication with bcrypt

### Frontend
- **HTML5/CSS3/JavaScript** with Tailwind CSS
- **Chart.js/Plotly.js** for statistical visualizations
- **D3.js** for custom interactive visualizations
- **Web Audio API** for audio handling

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL (optional, SQLite for development)
- FFmpeg for audio processing
- 8GB+ RAM for model training
- GPU recommended for faster inference

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <repository-url>
cd respiratory-disease-classification-system
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Database Setup
```bash
python manage.py db upgrade
python manage.py create_admin_user
```

### 4. Run the Application
```bash
python app.py
```

The system will be available at `http://localhost:5000`

## 📁 Project Structure

```
respiratory-disease-classification-system/
├── app/                          # Main application package
│   ├── __init__.py              # Flask app initialization
│   ├── models/                  # Database models
│   ├── routes/                  # API endpoints
│   ├── services/                # Business logic
│   ├── utils/                   # Utility functions
│   └── templates/               # HTML templates
├── ml/                          # Machine learning components
│   ├── models/                  # Neural network architectures
│   ├── preprocessing/           # Audio preprocessing
│   ├── training/                # Model training scripts
│   └── evaluation/              # Model evaluation
├── static/                      # Static assets
│   ├── css/                     # Stylesheets
│   ├── js/                      # JavaScript files
│   └── uploads/                 # File uploads
├── tests/                       # Test suite
├── docs/                        # Documentation
├── requirements.txt              # Python dependencies
├── config.py                    # Configuration settings
└── app.py                       # Application entry point
```

## 🔧 Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/dbname
SECRET_KEY=your-secret-key

# File Storage
UPLOAD_FOLDER=./static/uploads
MAX_CONTENT_LENGTH=16777216

# Model Settings
MODEL_PATH=./ml/models/
AUDIO_SAMPLE_RATE=22050
```

## 📊 Model Performance

### Target Metrics
- **Overall Accuracy**: 85-95%
- **CNN Model**: 80-90%
- **LSTM Model**: 75-85%
- **Hybrid Model**: 85-95%
- **Processing Time**: < 30 seconds per file

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=app tests/

# Run specific test category
python -m pytest tests/test_models.py
```

## 📈 API Documentation

### Authentication Endpoints
- `POST /api/auth/login` - User login
- `POST /api/auth/register` - User registration
- `POST /api/auth/logout` - User logout

### Patient Management
- `GET /api/patients` - List patients
- `POST /api/patients` - Create patient
- `GET /api/patients/<id>` - Get patient details
- `PUT /api/patients/<id>` - Update patient

### Audio Analysis
- `POST /api/audio/upload` - Upload audio file
- `POST /api/audio/analyze` - Analyze audio with all models
- `GET /api/audio/results/<id>` - Get analysis results

## 🔒 Security Features

- **JWT Authentication** with refresh tokens
- **Role-based Access Control** (Doctor, Technician, Admin, Researcher)
- **Password Hashing** with bcrypt
- **HTTPS Enforcement** in production
- **Input Validation** and sanitization
- **Rate Limiting** for API endpoints

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation in `/docs`

## 🏥 Clinical Disclaimer

This system is designed to assist healthcare professionals and should not replace clinical judgment. All predictions and recommendations should be reviewed by qualified medical personnel before making clinical decisions.

---

**Version**: 1.0.0  
**Last Updated**: 2024  
**Maintainers**: Development Team