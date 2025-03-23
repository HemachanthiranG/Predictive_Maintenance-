# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Create the dataset
df = pd.read_csv('predictive_maintenance.csv')
# Convert to DataFrame
df = pd.DataFrame(df)

# Drop columns that are not useful for prediction
df = df.drop(columns=['UDI', 'Product ID', 'Failure Type'])

# Handle categorical variables
df = pd.get_dummies(df, columns=['Type'])

# Split data into features and target
X = df.drop(columns=['Target'])
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Save the model to a file
model_filename = 'predictive_maintenance_model.pkl'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# Make predictions on the test set
predictions = model.predict(X_test)

# Create a DataFrame to display actual vs predicted values
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': predictions})
print(results.head(10))  # Display the results

# Optionally, save the predictions to a CSV file
results.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")

def get_user_input():
    # Get user input for each feature
    Type = input("Enter Type (H, L, M): ")
    air_temp = float(input("Enter Air temperature [K]: "))
    process_temp = float(input("Enter Process temperature [K]: "))
    rotational_speed = int(input("Enter Rotational speed [rpm]: "))
    torque = float(input("Enter Torque [Nm]: "))
    tool_wear = int(input("Enter Tool wear [min]: "))
    
    # Create a DataFrame from user input
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
    
    return input_data

def predict_failure(model, input_data):
    # Make prediction
    prediction = model.predict(input_data)
    return "Failure" if prediction[0] == 1 else "No Failure"

# Load the model
model = joblib.load('predictive_maintenance_model.pkl')

# Get user input and make prediction
input_data = get_user_input()
result = predict_failure(model, input_data)
print("Prediction:", result)