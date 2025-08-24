import os


class Config:
	SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-change-me')
	SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///instance/app.db')
	SQLALCHEMY_TRACK_MODIFICATIONS = False
	JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret-change-me')
	JSON_SORT_KEYS = False
	PREFERRED_URL_SCHEME = os.getenv('PREFERRED_URL_SCHEME', 'http')
	MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 128 * 1024 * 1024))  # 128 MB

	# CORS and security toggles for later
	DEBUG = os.getenv('FLASK_DEBUG', '0') == '1'
	ENV = os.getenv('FLASK_ENV', 'production')