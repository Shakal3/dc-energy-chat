# TODO: Wire this Flask app to real TimescaleDB and implement chat logic
# TODO: Add proper error handling, logging, and CORS configuration
# TODO: Integrate with forecast.cost_engine for actual predictions

import os
from flask import Flask
from dotenv import load_dotenv

def create_app(config=None):
    """
    Flask application factory pattern.
    Creates and configures the Flask app instance.
    """
    load_dotenv()
    
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'development')
    app.config['DATABASE_URL'] = os.getenv('DB_URL', 'sqlite:///telemetry.db')
    
    if config:
        app.config.update(config)
    
    # Register blueprints
    from chat_api.routes import chat_bp
    app.register_blueprint(chat_bp)
    
    @app.route('/health')
    def health_check():
        return {'status': 'healthy', 'service': 'dc-energy-chat'}
    
    return app

# Development server entry point
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True) 