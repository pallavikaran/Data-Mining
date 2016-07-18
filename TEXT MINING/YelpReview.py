
#remove number of samples
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from nltk.collocations import* 

from YelpMining import *
from ModelData import *

#csv files
trainingCsv = 'consolidated_yelp_training_data.csv'
testingCsv = 'consolidated_yelp_testing_data.csv'
validationCsv = 'consolidated_yelp_validation_data.csv'

#select the regressor to run
#Regressor_Name = 'svr'
#Regressor_Name = 'nusvr'
#Regressor_Name = 'linear'
Regressor_Name = 'RF'

y_train = []
X_train = []

y_train, X_train = mineReviews(trainingCsv)
model = SelectModel(Regressor_Name)
model.fit(X_train,y_train)

#text preprocessing, compute features, fitting the data for validation  data
X_validation = []
y_validation = []
y_predict = []

y_validation, X_validation = mineReviews(validationCsv)
y_predict = model.predict(X_validation)
  
RMSLE = calculate_RMSLE(y_validation, y_predict) 
 
print "Mean Square Error on Validation data is: ",mean_squared_error(y_validation,y_predict)
print "R square error is: ",r2_score(y_validation, y_predict)                           
print "Root Mean Square Log Error: ", RMSLE 
      
review_id = []
X_test = []

review_id, X_test = mineReviews(testingCsv)
p_test = model.predict(X_test)

#write the predicted values for the test data
prediction = open("prediction.csv","w")
prediction.write(str(("id,votes"+"\n")))
for i,j in zip(review_id,p_test):
    prediction.write(str((i+","+str(j)+"\n")))
prediction.close()


