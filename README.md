# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Initialize parameters (θ vector), learning rate, and load dataset with multiple features.
2. Compute predictions using hypothesis ( h(x) = θ_0 + θ_1x_1 + θ_2x_2 + \dots + θ_nx_n ).
3. Update all parameters simultaneously using gradient descent to minimize error.
4. Repeat until convergence and use final θ values to make predictions.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Kamali V
RegisterNumber:  212225240066
*/
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("house.csv")
data.columns = data.columns.str.strip()
X = data[['Size', 'Bedrooms']]
y_price = data['Price']
y_occ = data['Occupants']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
price_model = SGDRegressor(max_iter=5000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=5000, learning_rate='constant', eta0=0.01)
price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)
size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))
new_data = pd.DataFrame([[size, bed]], columns=['Size', 'Bedrooms'])
new_scaled = scaler.transform(new_data)
pred_price = price_model.predict(new_scaled)
pred_occ = occ_model.predict(new_scaled)
print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
```

## Output:
<img width="474" height="96" alt="image" src="https://github.com/user-attachments/assets/2f64048d-1f90-4f52-875b-163db28792af" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
