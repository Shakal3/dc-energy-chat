# TODO: Implement real chat logic with embeddings and intent classification
# TODO: Integrate with forecast.cost_engine.cost_forecast()
# TODO: Add input validation and proper error responses
# TODO: Add authentication/rate limiting for production

from flask import Blueprint, request, jsonify

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/chat', methods=['POST'])
def chat_endpoint():
    """
    Main chat endpoint for DC Energy Chat API.
    Expects JSON: {"query": "user question about energy costs"}
    Returns: {"response": "AI generated response", "forecast": {...}}
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query field'}), 400
        
        user_query = data['query']
        
        # TODO: Process user query:
        # 1. Generate embeddings with sentence-transformers
        # 2. Classify intent (forecast request, general question, etc.)
        # 3. If forecast: call forecast.cost_engine.cost_forecast()
        # 4. Generate natural language response
        
        # Stub response for now
        response = {
            'ok': True,
            'query_received': user_query,
            'response': 'Chat functionality coming soon!',
            'next_steps': [
                'Implement embedding-based intent classification',
                'Wire to TimescaleDB for telemetry data',
                'Connect to cost_engine for forecasts'
            ]
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@chat_bp.route('/chat', methods=['GET'])
def chat_info():
    """Development endpoint to show API info"""
    return jsonify({
        'service': 'DC Energy Chat API',
        'endpoints': {
            'POST /chat': 'Main chat interface',
            'GET /health': 'Health check'
        },
        'example_request': {
            'query': 'What will my energy costs be next week?'
        }
    }) 