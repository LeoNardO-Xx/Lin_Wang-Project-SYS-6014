

import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

plt.switch_backend('TkAgg')  

# Initialize two empty lists
dates = []
prices = []

def get_data(filename):
	
	with open(filename, 'r') as csvfile:
		# csvFileReader allows us to iterate over every row in our csv file
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0])) # Only gets day of the month which is at index 0
			prices.append(float(row[1])) # Convert to float for more precision

	return

def predict_price(dates, prices, x):
	
	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1
	
	
	svr_lin = SVR(kernel= 'linear', C= 1e3) # 1e3 denotes 1000
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
	
	svr_rbf.fit(dates, prices) 
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)


	plt.scatter(dates, prices, color= 'black', label= 'Data') 
	
	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') 
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') 
	plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') 
	plt.xlabel('Date') 
	plt.ylabel('Price') 
	plt.title('Support Vector Regression') # Setting title
	plt.legend() # Add legend
	plt.show() # To display result on screen

	return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0] # returns predictions from each of our models

get_data('/Users/leonardo/Downloads/stock-er-master/snap.csv') # calling get_data method by passing the csv file to it

predicted_price = predict_price(dates, prices, 29)

print('The predicted prices are:', predicted_price)
