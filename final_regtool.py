import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Necessary Libraries:
# Pandas
# Numpy
# Matplotlib
# SKLearn

# Gather the data using Pandas
mdata = pd.read_csv('data/number-of-measles-cases.csv')

mdata.head()


columns_to_select = ['Number of measles cases', 'Vaccination Rate']
mdata = mdata.loc[:, columns_to_select]

print(mdata.head())
print('Correlation Rate: ', mdata['Vaccination Rate'].corr(mdata['Number of measles cases']))

X = mdata['Vaccination Rate'].to_numpy().reshape(-1,1)
y = mdata['Number of measles cases']

X.shape


# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_degree = int(input('Degree of the polynomial (insert a number): '))



# Build a polynomial regression pipeline
pipeline = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression()) # HERE'S THE DEGREE

# Use pipeline to build a model
pipeline.fit(X_train,y_train)


# Test model with test data
pred = pipeline.predict(X_test)
pipeline.predict(X_test)

# Calc + print mean squared error
mse = np.sqrt(mean_squared_error(y_test,pred))
print(f'mean error: {mse:3.3} ({mse/np.mean(pred)*100:3.3}%)')

# Calculate coefficient of determination
score = pipeline.score(X_train,y_train)
print('Model determination: ', score)

# Coefficients + Intercept
print('Coefficients, lowest to highest degree: ', pipeline.steps[1][1].coef_)
print('Intercept: ', pipeline.steps[1][1].intercept_)

# Plot results
plt.scatter(X_test,y_test)
plt.plot(sorted(X_test),pipeline.predict(sorted(X_test)))
