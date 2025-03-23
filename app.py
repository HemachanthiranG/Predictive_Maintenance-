from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load('predictive_maintenance_model.pkl')

def predict_failure(model, input_data):
    # Make prediction
    prediction = model.predict(input_data)
    return "Failure" if prediction[0] == 1 else "No Failure"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    Type = request.form['Type']
    air_temp = float(request.form['air_temp'])
    process_temp = float(request.form['process_temp'])
    rotational_speed = int(request.form['rotational_speed'])
    torque = float(request.form['torque'])
    tool_wear = int(request.form['tool_wear'])
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rotational_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear],
        'Type_H': [1 if Type == 'H' else 0],
        'Type_L': [1 if Type == 'L' else 0],
        'Type_M': [1 if Type == 'M' else 0]
    })
    
    # Make prediction
    prediction = predict_failure(model, input_data)
    
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
