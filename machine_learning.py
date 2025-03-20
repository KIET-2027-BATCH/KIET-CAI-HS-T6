from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Load the dataset
df = pd.read_csv("C:/Users/Dell/OneDrive/Desktop/hackthon/traffic_dataset_5000.csv")

# Print the columns to debug
print(df.columns)

# Encode categorical data
location_encoder = LabelEncoder()
status_encoder = LabelEncoder()

df['Location'] = location_encoder.fit_transform(df['Location'])
df['Traffic_Status'] = status_encoder.fit_transform(df['Traffic_Status'])

# Feature selection and target
df['Date'] = pd.to_datetime(df['Date']).astype(int) / 10**9  # Convert date to timestamp
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M').dt.hour
df.dropna(inplace=True)

# Check if 'percentage' column exists
if 'percentage' in df.columns:
    df['percentage'] = df['percentage'].astype(float)
else:
    print("Column 'percentage' does not exist in the dataset")

# Adjust feature selection to available columns
X = df[['Location', 'Date', 'Time', 'percentage']] if 'percentage' in df.columns else df[['Location', 'Date', 'Time']]
y = df['Traffic_Status']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Flask App
app = Flask(__name__)

# Serve the HTML file
@app.route('/')
def home():
    return send_from_directory(os.path.dirname(__file__), 'ui.html')

# Prediction endpoint
@app.route('/submit', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate input data
        required_fields = ['location', 'date', 'time', 'percentage']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Encode location
        location = location_encoder.transform([data['location']])[0]
        date = pd.to_datetime(data['date']).timestamp()  # Convert date to timestamp
        time = int(data['time'].split(':')[0])  # Extract hour only
        percentage = float(data['percentage']) if 'percentage' in data else 0
        # Make prediction
        prediction = model.predict([[location, date, time, percentage]])[0]
        traffic_status = status_encoder.inverse_transform([prediction])[0]  # Decode prediction

        return jsonify({
            "location": data['location'],
            "date": data['date'],
            "time": data['time'],
            "percentage": data['percentage'],
            "predicted_traffic": traffic_status
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)