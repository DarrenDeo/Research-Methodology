from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import os

# Import your existing functions from model.py
from model import get_headline_from_url, categorize_news, predict_with_traditional, run_full_analysis

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/check-news', methods=['POST'])
def check_news():
    data = request.json
    url_or_headline = data.get('urlOrHeadline', '')
    
    # Use your existing functions
    if url_or_headline.startswith(('http://', 'https://')):
        headline = get_headline_from_url(url_or_headline)
    else:
        headline = url_or_headline
    
    # Load your trained model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'traditional_model_all.pkl')
        trad_model = joblib.load(model_path)
    except Exception as e:
        return jsonify({'error': f'Model not found: {str(e)}'}), 404
    
    # Make prediction
    category = categorize_news({'title': headline, 'content': ''})
    prediction = predict_with_traditional(trad_model, headline)
    
    # Return results
    return jsonify({
        'headline': headline,
        'category': category,
        'prediction': 'Hoax' if prediction == 1 else 'Not Hoax',
        'confidence': 85.5  # You might add confidence calculation to your model
    })

@app.route('/api/train-models', methods=['POST'])
def train_models():
    # Run your model training function
    try:
        results = run_full_analysis()
        return jsonify({
            'success': True,
            'message': 'Training completed successfully',
            'results': results.to_dict('records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during training: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Make sure the models folder exists
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, port=5000)