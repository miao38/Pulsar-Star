import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression #from website
from sklearn.metrics import accuracy_score #from website

data = pd.read_csv("C:\\Users\\Roy Miao\\Documents\\Projects\\Pulsar Star\\Pulsar Star Data.csv")
predictors = data[["Mean_of_the_integrated_profile", "Standard_deviation_of_the_integrated_profile", "Excess_kurtosis_of_the_integrated_profile",
                   "Skewness_of_the_integrated_profile", "Mean_of_the_DM-SNR_curve", "Standard_deviation_of_the_DM-SNR_curve",
                   "Excess_kurtosis_of_the_DM-SNR_curve", "Skewness_of_the_DM-SNR_curve", "target_class"]]
x = np.random.rand(len(data)) < .75
train = predictors[x]
test = predictors[~x]

train_x = train.drop(columns = ["target_class"], axis = 1)
train_y = train["target_class"]
test_x = test.drop(columns = ["target_class"], axis = 1)
test_y = test["target_class"]


#Logistic Regression from website
LR = LogisticRegression()
LR.fit(train_x, train_y)
print("Coefficient of model: ", LR.coef_)
print("Intercept of model: ", LR.intercept_)
predict_train = LR.predict(train_x)
print("Target on train data ", predict_train)
accuracy_train = accuracy_score(train_y, predict_train)
print("Accuracy score on train data: ", accuracy_train)
predict_test = LR.predict(test_x)
print("Target on test data ", predict_test)
accuracy_test = accuracy_score(test_y, predict_test)
print("Accuracy score on test data ", accuracy_test)