
import joblib
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI,HTTPException,UploadFile,File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel,Field

app=FastAPI()

# Load Model and Features

model=joblib.load('house_model.joblib')
feat=joblib.load('house_features.joblib')
mae=joblib.load('house_mae.joblib')

# Define Input Schema

class houseFeatures(BaseModel):
    MedInc:float=Field(gt=0,description="Median income")
    HouseAge:float=Field(ge=0,description="Average age of houses")
    AveRooms:float=Field(gt=0,description="Average number of rooms")
    AveBedrms:float=Field(gt=0,description="Average number of bedrooms")
    Population:float=Field(ge=0,description="Total population")
    AveOccup:float=Field(gt=0,description="Average household occupancy")
    Latitude:float=Field(ge=32,le=42,description="Latitude")
    Longitude:float=Field(ge=-125,le=-114,description="Longitude")

# Home Endpoint

@app.get('/')
def home():
    return{
        'message':'California house price prediction API',
        'status':'running',
        'endpoint':'send POST request to /predict'
    }

# Health Check Endpoint

@app.get('/health')
def health():
    return{
        'status':'running',
        'model':'RandomForestRegressor',
        'features':feat,
        'test_mae_usd':round(mae,2)
    }

# Single House Prediction

@app.post('/predict')
def predict(house:houseFeatures):
    try:
        input_data=pd.DataFrame([{
            "MedInc":house.MedInc,
            "HouseAge":house.HouseAge,
            "AveRooms":house.AveRooms,
            "AveBedrms":house.AveBedrms,
            "Population":house.Population,
            "AveOccup":house.AveOccup,
            "Latitude":house.Latitude,
            "Longitude":house.Longitude
        }])

        prediction=(model.predict(input_data)[0])*100000

        return{
            'predicted_price':f'${prediction:.0f}',
            'test_mae_usd':f'${mae:.0f}'
        }

    except Exception as exp:
        raise HTTPException(
            status_code=500,
            detail=f'Prediction failed: {str(exp)}'
        )

# CSV Batch Prediction

@app.post('/predict-file')
async def predict_file(file:UploadFile=File(...)):
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail='Please upload CSV file only'
        )

    content=await file.read()

    if not content.strip():
        raise HTTPException(
            status_code=400,
            detail='Uploaded file is empty'
        )

    try:
        df=pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError,pd.errors.EmptyDataError,UnicodeDecodeError):
        raise HTTPException(
            status_code=400,
            detail='Invalid CSV file'
        )

    # Validate Required Columns

    req_col=['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']

    miss_col=[col for col in req_col if col not in df.columns]

    if miss_col:
        raise HTTPException(
            status_code=400,
            detail=f'These columns are missing: {miss_col}'
        )

    if len(df)==0:
        raise HTTPException(
            status_code=400,
            detail='No data present in file'
        )

    # Validate Numeric Data

    df[req_col]=df[req_col].apply(pd.to_numeric,errors='coerce')

    if df[req_col].isna().any().any():
        raise HTTPException(
            status_code=400,
            detail='Feature columns must contain valid numeric values without missing data'
        )

    if not np.isfinite(df[req_col].to_numpy()).all():
        raise HTTPException(
            status_code=400,
            detail='Feature columns cannot contain infinite values'
        )

    # Generate Batch Predictions

    try:
        prediction=model.predict(df[req_col])
        df['predicted_price_usd']=(prediction*100000).round(0).astype(int)

        output=df.to_csv(index=False)

        return StreamingResponse(
            io.StringIO(output),
            media_type='text/csv',
            headers={
                'Content-Disposition':'attachment; filename=predictions.csv'
            }
        )

    except Exception as exp:
        raise HTTPException(
            status_code=500,
            detail=f'Prediction failed: {str(exp)}'
        )
