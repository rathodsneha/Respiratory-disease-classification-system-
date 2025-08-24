from flask import Flask
from dotenv import load_dotenv
import os

from .extensions import db, migrate, jwt, bcrypt
from .blueprints.main import main_bp


def create_app(config_object: str | None = None) -> Flask:
	load_dotenv()
	app = Flask(__name__, instance_relative_config=True)

	# Base configuration
	app.config.from_object(config_object or 'app.config.Config')

	# Ensure instance folder exists
	try:
		os.makedirs(app.instance_path, exist_ok=True)
	except OSError:
		pass

	_register_extensions(app)
	_register_blueprints(app)

	return app


def _register_extensions(app: Flask) -> None:
	db.init_app(app)
	migrate.init_app(app, db)
	jwt.init_app(app)
	bcrypt.init_app(app)


def _register_blueprints(app: Flask) -> None:
	app.register_blueprint(main_bp)