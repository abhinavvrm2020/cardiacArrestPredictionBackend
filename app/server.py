from fastapi import FastAPI, HTTPException
import joblib
import numpy as np

# Load the model and class names
model = joblib.load('app/xgboost_model.joblib')
class_names = np.array(['Yes', 'No'])

# Initialize FastAPI app
app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Cardiac Arrest Prediction API'}

@app.post('/predict')
def predict(data: dict):
    try:
        print("Received data:", data)
        # Extract feature values from the request data
        features = [
            float(data['BMI']),
            1 if data['Smoking'] == 'Yes' else 0,
            1 if data['AlcoholDrinking'] == 'Yes' else 0,
            1 if data['Stroke'] == 'Yes' else 0,
            int(data['PhysicalHealth']),
            int(data['MentalHealth']),
            1 if data['DiffWalking'] == 'Yes' else 0,
            1 if data['Sex'] == 'Male' else 0,  # Encode Female as 0, Male as 1
            int(data['Age']),  
            1 if data['Diabetic'] == 'Yes' else 0,
            1 if data['PhysicalActivity'] == 'Yes' else 0,
            0 if data['GenHealth'] in ['Excellent'] else 4 if data['GenHealth'] in ['Very good'] else 2 if data['GenHealth'] in ['Good'] else 1 if data['GenHealth'] in ['Fair'] else 4,
            int(data['SleepTime']),
            1 if data['Asthma'] == 'Yes' else 0,
            1 if data['KidneyDisease'] == 'Yes' else 0,
            1 if data['SkinCancer'] == 'Yes' else 0
        ]
        
        # Reshape features into 2D array for prediction
        features_arr = np.array(features, dtype=object).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_arr)
        
        # Get predicted class name
        class_name = class_names[prediction][0]
        
        return {'predicted_class': class_name}
    
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
