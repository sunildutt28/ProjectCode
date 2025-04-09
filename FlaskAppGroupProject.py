from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained SVM model
svm_model = joblib.load('svm_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()
        # Convert JSON input to DataFrame
        input_df = pd.DataFrame([data])

        # Ensure the input DataFrame has the same columns as the training data
        # You might need to handle missing columns or reorder them
        # Here's an example, adjust according to your data:
        expected_columns = ['GENDER', 'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']  # Replace with your actual column names
        missing_cols = set(expected_columns) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0  # Or handle missing columns differently (e.g., mean imputation)
        input_df = input_df[expected_columns]

        # Make prediction
        prediction = svm_model.predict(input_df)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})  # Convert prediction to int

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port='8080') # indent this line