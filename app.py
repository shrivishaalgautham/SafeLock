import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, render_template, jsonify, session
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

app = Flask(__name__)
import os
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_secret')  # Use env variable for secret key

# Function to save the plot as a base64 string
def save_plot_as_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64

# Function to create plots
def create_plots(data):
    plots = {}
    sns.set(style="whitegrid")

    # Time Series Plot
    plt.figure(figsize=(14, 6))
    plt.plot(data['timestamp'], data['scaled_value'], label='Key Hold Time (Scaled)', color='blue', alpha=0.7)
    plt.title('Key Hold Time Series')
    plt.xlabel('Timestamp')
    plt.ylabel('Scaled Value')
    plt.legend(loc='upper right')
    plots['time_series'] = save_plot_as_base64()

    # Plot for Isolation Forest
    plt.figure(figsize=(14, 6))
    plt.plot(data['timestamp'], data['scaled_value'], label='Key Hold Time (Scaled)', color='blue', alpha=0.7)
    plt.scatter(data[data['iso_forest_outlier'] == True]['timestamp'], 
                data[data['iso_forest_outlier'] == True]['scaled_value'], 
                color='red', label='Anomalies (Isolation Forest)', s=50)
    plt.title('Anomalies Detected by Isolation Forest')
    plt.xlabel('Timestamp')
    plt.ylabel('Scaled Value')
    plt.legend(loc='upper right')
    plots['iso_forest'] = save_plot_as_base64()

    # Plot for One-Class SVM
    plt.figure(figsize=(14, 6))
    plt.plot(data['timestamp'], data['scaled_value'], label='Key Hold Time (Scaled)', color='blue', alpha=0.7)
    plt.scatter(data[data['svm_outlier'] == True]['timestamp'], 
                data[data['svm_outlier'] == True]['scaled_value'], 
                color='orange', label='Anomalies (One-Class SVM)', s=50)
    plt.title('Anomalies Detected by One-Class SVM')
    plt.xlabel('Timestamp')
    plt.ylabel('Scaled Value')
    plt.legend(loc='upper right')
    plots['svm'] = save_plot_as_base64()

    return plots

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_csv', methods=['POST'])
# Helper function: Validate CSV columns
REQUIRED_COLUMNS = {'timestamp', 'value'}
def validate_csv(data):
    if not REQUIRED_COLUMNS.issubset(data.columns):
        missing = REQUIRED_COLUMNS - set(data.columns)
        raise ValueError(f"Missing columns in CSV: {', '.join(missing)}")

# Helper function: Clean and preprocess data
def preprocess_data(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data[data['value'] != 0].copy()
    scaler = MinMaxScaler()
    data['scaled_value'] = scaler.fit_transform(data[['value']])
    return data

# Helper function: Detect anomalies
def detect_anomalies(data, contamination=0.05, nu=0.05):
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    data['iso_forest_outlier'] = iso_forest.fit_predict(data[['scaled_value']])
    data['iso_forest_outlier'] = data['iso_forest_outlier'].map({1: False, -1: True})
    one_class_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=nu)
    data['svm_outlier'] = one_class_svm.fit_predict(data[['scaled_value']])
    data['svm_outlier'] = data['svm_outlier'].map({1: False, -1: True})
    return data

@app.route('/process_csv', methods=['POST'])
def process_csv():
    try:
        if 'csv_data' not in request.form:
            return jsonify({'status': 'error', 'error': 'No CSV data provided'}), 400
        csv_data = request.form['csv_data']
        csv_file = io.StringIO(csv_data)
        data = pd.read_csv(csv_file)
        try:
            validate_csv(data)
        except Exception as ve:
            return jsonify({'status': 'error', 'error': str(ve)}), 400
        # Configurable parameters from request or default
        contamination = float(request.form.get('contamination', 0.05))
        nu = float(request.form.get('nu', 0.05))
        try:
            data = preprocess_data(data)
            data = detect_anomalies(data, contamination=contamination, nu=nu)
        except Exception as pe:
            return jsonify({'status': 'error', 'error': f'Data processing error: {pe}'}), 500
        plots = create_plots(data)
        iso_forest_anomalies = int(data['iso_forest_outlier'].sum())
        svm_anomalies = int(data['svm_outlier'].sum())
        session['plots'] = plots
        session['iso_forest_anomalies'] = iso_forest_anomalies
        session['svm_anomalies'] = svm_anomalies
        return jsonify({'status': 'success'})
    except Exception as e:
        # Log the error for server-side debugging
        import logging
        logging.exception("Error in process_csv")
        return jsonify({'status': 'error', 'error': 'Internal server error'}), 500


@app.route('/results')
def results():
    plots = session.get('plots', {})
    iso_forest_anomalies = session.get('iso_forest_anomalies', 0)
    svm_anomalies = session.get('svm_anomalies', 0)
    return render_template('results.html', plots=plots, 
                           iso_forest_anomalies=iso_forest_anomalies, 
                           svm_anomalies=svm_anomalies)

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode)

# Note: Set environment variables SECRET_KEY and FLASK_DEBUG for production use.
