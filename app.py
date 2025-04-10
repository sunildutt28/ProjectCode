from flask import Flask, request, jsonify, render_template
import joblib
from flask_cors import CORS
import numpy as np

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
        print("Received data:", data)  # Debug print
        
        # Extract features in the correct order
        features = [
            float(data['gender']),
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
        
        # Debug prints
        print("Features array:", features)
        features_array = np.array(features).reshape(1, -1)
        print("Model feature names:", model.feature_names_in_)
        
        # Make prediction
        prediction = model.predict(features_array)
        probability = model.predict_proba(features_array)
        print("Raw prediction:", prediction)
        print("Prediction probability:", probability)
        
        result = "High Risk" if prediction[0] == 1 else "Low Risk"
        
        return jsonify({
            'prediction': result,
            'probability': probability.tolist(),
            'features': features
        })
    
    except Exception as e:
        print("Error during prediction:", str(e))  # Debug print
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Print model information
    print("Model information:")
    print("Number of support vectors:", model.n_support_)
    print("Classes:", model.classes_)
    
    app.run(debug=True, host='0.0.0.0', port=5000)