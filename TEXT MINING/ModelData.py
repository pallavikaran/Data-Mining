
from sklearn import svm
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import math

def calculate_RMSLE(y_test,y_predict):
    #calculate rmsle
    error = [ math.pow((math.log(y_predict+1)-math.log(y_test+1)),2) for y_predict,y_test in zip(y_predict,y_test)]
    err_len = len(error)
    error_sum = sum(error)

    error = math.sqrt(error_sum/err_len)
    return error   


def SelectModel(regressor):

    if(regressor == 'svr'):
        model = svm.SVR()
    elif (regressor == 'nusvr'):
        model = svm.NuSVR()
    elif (regressor == 'linear'):
        model=LinearRegression()
    elif (regressor == 'RF'):        
        model = RandomForestRegressor(n_estimators = 1500, n_jobs=-1)    
    
    return model

def transform_features_log(x):
    return np.log(1+x)

# To Convert into Scaled data that has zero mean and unit variance:
def standardize_features(x):
    return preprocessing.scale(x)

