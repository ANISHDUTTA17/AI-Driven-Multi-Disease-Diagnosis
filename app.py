from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import time
import joblib
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load image-based models
brain_model = load_model('models/brain_tumor.h5')
skin_model = load_model('models/skin_cancer_final.h5')

# Load tabular models
parkinsons_model = joblib.load('models/parkinsons_xgb_model.pkl')
parkinsons_scaler = joblib.load('models/scaler.pkl')

diabetes_model = joblib.load('models/diabetes_model.pkl')
diabetes_scaler = joblib.load('models/scaler_D.pkl')

# Helper for image preprocessing
def preprocess_image(image_path, target_size=(299, 299)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_disease(model, image_path, disease):
    img = preprocess_image(image_path)
    prediction = model.predict(img)

    if disease == 'brain_tumor':
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        return classes[np.argmax(prediction)]
    elif disease == 'skin_cancer':
        return "Malignant" if prediction[0][0] > 0.5 else "Benign"
    return "Unknown disease"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect/<disease>', methods=['GET', 'POST'])
def detect(disease):
    if disease == 'parkinsons':
        return redirect(url_for('parkinsons'))

    result = None
    image_path = None
    relative_image_path = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('detect.html', disease=disease, error="No file selected")

        filename = str(int(time.time())) + '_' + file.filename
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        relative_image_path = 'uploads/' + filename

        if disease == 'brain_tumor':
            result = predict_disease(brain_model, image_path, disease)
        elif disease == 'skin_cancer':
            result = predict_disease(skin_model, image_path, disease)
        else:
            return render_template('detect.html', disease=disease, error="Invalid disease type")

    return render_template('detect.html', disease=disease, result=result, image_path=relative_image_path)

@app.route('/parkinsons', methods=['GET', 'POST'])
def parkinsons():
    result = None
    input_values = {}

    if request.method == 'POST':
        features = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
            'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
            'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
            'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
        input_values = {f: float(request.form.get(f, 0)) for f in features}
        input_array = np.array([list(input_values.values())])
        input_scaled = parkinsons_scaler.transform(input_array)
        prediction = parkinsons_model.predict(input_scaled)
        result = "The person has Parkinson's Disease" if prediction[0] == 1 else "The person does NOT have Parkinson's Disease"

    return render_template('parkinsons.html', result=result, input_values=input_values)

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    result = None
    input_values = {}

    if request.method == 'POST':
        features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
            'AgeGroup', 'BMI_Category', 'Glucose_Insulin_Ratio',
            'Age_BMI_Product', 'Glucose_Squared'
        ]
        input_values = {f: float(request.form.get(f, 0)) for f in features}
        input_array = np.array([list(input_values.values())])
        input_scaled = diabetes_scaler.transform(input_array)
        prediction = diabetes_model.predict(input_scaled)[0]
        result = "🩺 The person is likely Diabetic" if prediction == 1 else "✅ The person is likely Not Diabetic"

    return render_template('diabetes.html', result=result, input_values=input_values)

@app.route('/heart', methods=['GET', 'POST'])
def heart():
    result = None
    input_values = {}

    if request.method == 'POST':
        # Use the exact column names used during model training
        cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
        num_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']

        # Collect inputs (note: collect thalach, we will rename it)
        input_values = {col: request.form.get(col, '') for col in cat_cols + ['thalach', 'age', 'trestbps', 'chol', 'oldpeak', 'ca']}

        # Convert numeric inputs to float
        for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']:
            input_values[col] = float(input_values[col])

        # Prepare input DataFrame
        df_input = pd.DataFrame([input_values])

        # ✅ Fix mismatch: Rename 'thalach' to 'thalch'
        df_input.rename(columns={'thalach': 'thalch'}, inplace=True)

        # Load preprocessing tools and model
        num_imputer = joblib.load('models/num_imputer.pkl')
        cat_imputer = joblib.load('models/cat_imputer.pkl')
        encoder = joblib.load('models/encoder.pkl')
        model = joblib.load('models/heart_model.pkl')

        # Preprocess numeric and categorical data
        X_num = pd.DataFrame(num_imputer.transform(df_input[num_cols]), columns=num_cols)
        X_cat_imputed = cat_imputer.transform(df_input[cat_cols])
        X_cat_encoded = pd.DataFrame(encoder.transform(X_cat_imputed),
                                     columns=encoder.get_feature_names_out(cat_cols))

        # Final combined input
        X_processed = pd.concat([X_num.reset_index(drop=True), X_cat_encoded.reset_index(drop=True)], axis=1)

        # Predict
        prediction = model.predict(X_processed)[0]
        result = "⚠️ The person is likely to have Heart Disease" if prediction == 1 else "✅ The person is unlikely to have Heart Disease"

    return render_template('heart.html', result=result, input_values=input_values)


if __name__ == '__main__':
    app.run(debug=True)
