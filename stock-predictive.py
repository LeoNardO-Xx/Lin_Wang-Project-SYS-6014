# -*- coding: utf-8 -*-
# @Time    : 4/27/20 12:26 上午
# @Author  : LeoNardO
# @FileName: Test.py
# @Software: PyCharm


import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score
plt.switch_backend('TkAgg')

# Initialize two empty lists
dates = []
prices = []


def get_data(filename):
    with open(filename, 'r') as csvfile:
        # csvFileReader allows us to iterate over every row in our csv file
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)  # skipping column names
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))  # Only gets day of the month which is at index 0
            prices.append(float(row[4]))  # Convert to float for more precision

    return

def predict_price(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))  # converting to matrix of n X 1
    # prices = np.reshape(prices, (len(prices), 1))

    # svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')  # 1e3 denotes 1000
    svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression models

    svr_rbf.fit(dates, prices)
    # svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)


    return svr_rbf.predict(np.array(x).reshape(-1, 1))[0], svr_poly.predict(np.array(x).reshape(-1, 1))[0]

def get_plot(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))  # converting to matrix of n X 1
    # prices = np.reshape(prices, (len(prices), 1))

    # svr_lin = SVR(kernel='linear', C=1e3, gamma='auto')  # 1e3 denotes 1000
    svr_poly = SVR(kernel='poly', C=1e3, degree=2, gamma='auto')
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)  # defining the support vector regression models

    svr_rbf.fit(dates, prices)
    # svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)

    # plt.scatter(dates, prices, color='black', label='Real Data')
    plt.plot(dates, prices, color='blue', label='Real Prices')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='SVR model Predicted Prices')
    # plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='green', label='Polynomial model Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')  # Setting title
    plt.legend()  # Add legend
    plt.show()  # To display result on screen
    return svr_rbf.predict(np.array(x).reshape(-1, 1))[0], svr_poly.predict(np.array(x).reshape(-1, 1))[0]

    # return svr_rbf.predict[x][0], svr_lin.predict[x][0], svr_poly.predict[x][0]  # returns predictions from each of our models
    # return svr_rbf.predict(x) # returns predictions from each of our models

predict_rbf=[]
predict_poly=[]
get_data('/Users/leonardo/Downloads/stock-er-master/snap.csv')  # calling get_data method by passing the csv file to it
# dates = np.reshape(dates, (len(dates), 1))  # converting to matrix of n X 1
# # print(dates)
for x in dates:
    predicted_price = predict_price(dates, prices, x)
    predict_rbf.append(float(predicted_price[0]))
    predict_poly.append(float(predicted_price[1]))
# print(predict_rbf)
# p=['19.44023847986434', '19.990404353075117', '20.66986058224376', '20.680387584211868', '20.990121253574657', '21.970172772312637', '22.809539658511902', '22.391307111943654', '21.53982429932977', '23.669442417946048', '27.190227468411376', '24.37978604067128']
# temp=[]
# print(prices)
# for i in p:
#     temp.append(float(i))

# print(temp)
mae_rbf=mean_absolute_error(prices,predict_rbf)
mse_rbf=mean_squared_error(prices,predict_rbf)
r2_rbf=r2_score(prices,predict_rbf)
print("The MAE of rbf is ",mae_rbf)
print("The MSE of rbf is " , mse_rbf)
print("The r2 score of rbf is ", r2_rbf)

mae_poly=mean_absolute_error(prices,predict_poly)
mse_poly=mean_squared_error(prices,predict_poly)
r2_poly=r2_score(prices,predict_poly)
print("The MAE of poly is ",mae_poly)
print("The MSE of poly is " , mse_poly)
print("The r2 score of poly is ", r2_poly)

a=predict_price(dates,prices,11)
# print(predicted_price)
# prices = np.reshape(prices, (len(prices), 1)
# # print(prices)
print("RBF preditcted price: $", str(a[0]))
# print("Linear kernel: $", str(predicted_price[1]))
print("Polynomial Prices : $", str(a[1]))
get_plot(dates,prices,11)

# print('The predicted prices are:', predicted_price)
# predicted_price, coefficient, constant=predict_price(dates, prices, 29)
# print(predicted_price)