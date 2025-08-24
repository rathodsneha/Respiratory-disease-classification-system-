"""
Error handlers for the Flask application
"""

from flask import render_template, jsonify, request
from werkzeug.exceptions import HTTPException
import traceback

def register_error_handlers(app):
    """Register error handlers with the Flask app"""
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors"""
        if request.is_xhr:
            return jsonify({
                'error': 'Bad Request',
                'message': 'The request could not be processed due to invalid data.',
                'status_code': 400
            }), 400
        return render_template('errors/400.html'), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        """Handle 401 Unauthorized errors"""
        if request.is_xhr:
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Authentication is required to access this resource.',
                'status_code': 401
            }), 401
        return render_template('errors/401.html'), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        """Handle 403 Forbidden errors"""
        if request.is_xhr:
            return jsonify({
                'error': 'Forbidden',
                'message': 'You do not have permission to access this resource.',
                'status_code': 403
            }), 403
        return render_template('errors/403.html'), 403
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors"""
        if request.is_xhr:
            return jsonify({
                'error': 'Not Found',
                'message': 'The requested resource was not found.',
                'status_code': 404
            }), 404
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        """Handle 405 Method Not Allowed errors"""
        if request.is_xhr:
            return jsonify({
                'error': 'Method Not Allowed',
                'message': 'The HTTP method is not allowed for this resource.',
                'status_code': 405
            }), 405
        return render_template('errors/405.html'), 405
    
    @app.errorhandler(422)
    def unprocessable_entity(error):
        """Handle 422 Unprocessable Entity errors"""
        if request.is_xhr:
            return jsonify({
                'error': 'Unprocessable Entity',
                'message': 'The request was well-formed but contains invalid data.',
                'status_code': 422
            }), 422
        return render_template('errors/422.html'), 422
    
    @app.errorhandler(429)
    def too_many_requests(error):
        """Handle 429 Too Many Requests errors"""
        if request.is_xhr:
            return jsonify({
                'error': 'Too Many Requests',
                'message': 'Rate limit exceeded. Please try again later.',
                'status_code': 429
            }), 429
        return render_template('errors/429.html'), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 Internal Server Error errors"""
        if request.is_xhr:
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred. Please try again later.',
                'status_code': 500
            }), 500
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle 503 Service Unavailable errors"""
        if request.is_xhr:
            return jsonify({
                'error': 'Service Unavailable',
                'message': 'The service is temporarily unavailable. Please try again later.',
                'status_code': 503
            }), 503
        return render_template('errors/503.html'), 503
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle all unhandled exceptions"""
        # Log the error
        app.logger.error(f'Unhandled exception: {error}')
        app.logger.error(traceback.format_exc())
        
        if request.is_xhr:
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred. Please try again later.',
                'status_code': 500
            }), 500
        
        return render_template('errors/500.html'), 500
    
    @app.errorhandler(HTTPException)
    def handle_http_exception(error):
        """Handle HTTP exceptions"""
        if request.is_xhr:
            return jsonify({
                'error': error.name,
                'message': error.description,
                'status_code': error.code
            }), error.code
        
        return render_template('errors/generic.html', error=error), error.code