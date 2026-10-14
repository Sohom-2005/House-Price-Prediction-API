
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
import pandas as pd
import joblib

# Load Data

data=fetch_california_housing()

# Prepare Features and Target

X=pd.DataFrame(data.data,columns=data.feature_names)
y=data.target

# Split Data

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# Train Model

model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

# Make Predictions

y_pred=model.predict(X_test)

# Evaluate Model

mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(f'MAE: ${mae*100000:.2f}')
print(f'R2 Score: {r2:.3f}')

# Save Model and Features

joblib.dump(model,'house_model.joblib')
joblib.dump(list(X.columns),'house_features.joblib')
joblib.dump(mae*100000,'house_mae.joblib')
