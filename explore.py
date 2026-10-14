from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load Data

data=fetch_california_housing()

# Create DataFrame

df=pd.DataFrame(data.data,columns=data.feature_names)

# Add Target

df['Price']=data.target

# Display Data

print(df.shape)
print(df.head())
print(df.describe())
