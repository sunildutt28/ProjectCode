from flask import Flask, request, jsonify, render_template
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load('svm_model.joblib')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract features in the correct order
        features = [
            float(data['age']),
            float(data['smoking']),
            float(data['yellowFingers']),
            float(data['anxiety']),
            float(data['peerPressure']),
            float(data['chronicDisease']),
            float(data['fatigue']),
            float(data['allergy']),
            float(data['wheezing']),
            float(data['alcoholConsuming']),
            float(data['coughing']),
            float(data['shortnessOfBreath']),
            float(data['swallowingDifficulty']),
            float(data['chestPain'])
        ]
        
        # Make prediction
        prediction = model.predict([features])
        result = "High Risk" if prediction[0] == 1 else "Low Risk"
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)