# DC Energy Chat API - Flask Application Factory
# Main application entry point for energy cost forecasting

from flask import Flask, jsonify, send_file
import os
import sys
import logging

# Add project root to Python path for module imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Application factory for Flask app."""
    # Configure Flask with proper static folder
    static_folder = os.path.join(project_root, 'chat_ui', 'static')
    
    app = Flask(__name__, 
                static_folder=static_folder,
                static_url_path='/static')
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    # Serve the main UI
    @app.route('/')
    def index():
        """Serve the main chat UI."""
        ui_path = os.path.join(project_root, 'chat_ui', 'index.html')
        return send_file(ui_path)
    
    # Health check endpoint
    @app.route('/health')
    def health():
            return jsonify({
        'status': 'healthy',
        'service': 'DC Energy Forecast API',
        'version': '3.0.0'
    })
    
    try:
        # Import and register blueprints
        from chat_api.routes import chat_bp
        app.register_blueprint(chat_bp)
        
        logger.info("Successfully registered chat blueprint")
        
    except ImportError as e:
        logger.error(f"Failed to import chat_api.routes: {e}")
        # Fallback: create a minimal chat route
        @app.route('/chat', methods=['POST'])
        def chat_fallback():
            return jsonify({
                'error': 'Chat service not properly configured',
                'message': 'Please check module imports'
            }), 500
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error'}), 500
    
    return app

# Create app instance for direct running
app = create_app()

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting DC Energy Forecast API on port {port}")
    logger.info(f"Debug mode: {debug}")
    logger.info(f"Forecast UI available at: http://localhost:{port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        use_reloader=debug
    ) 