from flask import Flask, request, render_template
import numpy as np
from model import load_model

app = Flask(__name__)

# Load the model
model = load_model('models/model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_heart_disease():
    try:
        # Extract features from the form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        # Make prediction
        prediction = model.predict(final_features)
        
        # Interpret prediction
        output = 'Heart Disease' if prediction == 1 else 'No Heart Disease'
        return render_template('index.html', prediction_text=f'Result: {output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)






