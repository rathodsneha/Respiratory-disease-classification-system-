from flask import jsonify
from . import main_bp


@main_bp.get('/health')
def health():
	return jsonify({
		'status': 'ok'
	})