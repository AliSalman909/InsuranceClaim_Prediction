import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

#loading the data
df = pd.read_csv(r"C:\Users\Ali's HP\Desktop\Internship\task4\insurance.csv")
print("Data Overview:\n", df.head())

# creating graphs for visualization between different relationships
sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker')
plt.title("BMI vs Charges by Smoker")
plt.show()

sns.scatterplot(data=df, x='age', y='charges')
plt.title("Age vs Charges")
plt.show()

sns.boxplot(data=df, x='smoker', y='charges')
plt.title("Charges by Smoking Status")
plt.show()

# conversion into numeric forms 
df_encoded = pd.get_dummies(df, drop_first=True)

#x would have all columns other than charges
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges'] # last column is charges and y is set to that

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and Error evaluation
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Mean Absolute Error :", mae)
print("Root Mean Squared Error :", rmse)
