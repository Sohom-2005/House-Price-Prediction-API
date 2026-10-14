
# California House Price Prediction API

A machine learning project using Random Forest Regression and FastAPI to predict California housing values. Supports individual predictions, batch CSV inference, and model evaluation.

## Features
- Data exploration using Pandas
- Random Forest Regression using Scikit-learn
- Model evaluation using MAE and R² Score
- Individual house price prediction through REST API
- Batch predictions through CSV uploads
- Input validation and error handling
- Downloadable CSV prediction results
- Model serialization using Joblib

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Joblib
- Uvicorn

## Project Structure
```text
california-house-price-prediction-api/
├── explore.py
├── train.py
├── main.py
├── house_model.joblib
├── house_features.joblib
├── house_mae.joblib
├── test_data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:
```bash
git clone <your-repository-url>
cd california-house-price-prediction-api
```

Create and activate a virtual environment:

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Training

Explore the dataset:
```bash
python explore.py
```

Train and evaluate the model:
```bash
python train.py
```

The training script saves the trained model, feature names, and test MAE using Joblib.

## Run the API

```bash
uvicorn main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Model and feature information |
| POST | `/predict` | Predict the price of one house |
| POST | `/predict-file` | Upload a CSV for batch predictions |

### Individual Prediction

Send a POST request to `/predict` with JSON input:

```json
{
  "MedInc": 5.0,
  "HouseAge": 20,
  "AveRooms": 6.0,
  "AveBedrms": 1.2,
  "Population": 800,
  "AveOccup": 3.0,
  "Latitude": 34.05,
  "Longitude": -118.25
}
```

The API returns the predicted house price and test-set MAE.

### Batch Prediction

Upload a CSV file to `/predict-file` containing these columns:

```text
MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude
```

The API validates the input, predicts prices for all rows, and returns a downloadable `predictions.csv` file containing the additional `predicted_price_usd` column.

## Model Evaluation

- **MAE:** Mean Absolute Error on the test set, converted to dollars.
- **R² Score:** Measures the model's predictive performance on the test set.

Add the actual MAE and R² values printed by `train.py` here.

## Limitations

- Built for learning and demonstration purposes.
- Predictions are based on the California Housing dataset and are not guaranteed real-world property valuations.
- Test-set MAE is an average error, not a guaranteed error margin for individual predictions.

## Future Improvements
- Automated API testing
- More comprehensive CSV validation
- Interactive frontend
- Model tuning and algorithm comparison

