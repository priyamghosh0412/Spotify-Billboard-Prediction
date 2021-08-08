# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
import warnings
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

#Importing the dataset
df=pd.read_csv('final_dataset.csv')

#Droping unwanted columns
df.drop(['Unnamed: 0','Instrumentalness'],axis=1,inplace=True)

from scipy import stats
df['Liveness']=np.log(df['Liveness'])
df['Speechiness']=stats.boxcox(df['Speechiness'])[0]
df['Acousticness']=stats.boxcox(df['Acousticness'])[0]


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


counter=Counter(y)
print('before',counter)
smt=SMOTETomek()
balanced_x,balanced_y=smt.fit_resample(x,y)
counter=Counter(balanced_y)
print('after',counter)


x_train,x_test,y_train,y_test=train_test_split(balanced_x,balanced_y,test_size=0.2)

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
scaler=StandardScaler()
scaled_x_train=scaler.fit_transform(x_train)
scaled_x_test=scaler.transform(x_test)

##Decision Tree Classifier##
from sklearn.tree import DecisionTreeClassifier
Model=DecisionTreeClassifier()
Model.fit(scaled_x_train,y_train)
pred=Model.predict(scaled_x_test)
acc=accuracy_score(y_test,pred)
print(acc)

##Hyperparameter tuning on Decision tree classifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
param_dist={"criterion":["gini","entropy"],"max_depth":[1,2,3,4,5,6,7,8,9,10,11]}
grid=RandomizedSearchCV(DecisionTreeClassifier(),param_distributions=param_dist,cv=5,n_jobs=-1)
grid.fit(scaled_x_train,y_train)

grid.best_estimator_
grid.best_score_
final_model=grid.best_estimator_
y_pred=final_model.predict(scaled_x_test)
print('confusion matrix')
print(confusion_matrix(y_test,y_pred))
print('classification report')
print(classification_report(y_test,y_pred))
acc=accuracy_score(y_test,y_pred)
print(acc)

#plotting ROC curve
import matplotlib.pyplot as plt
from sklearn import metrics
metrics.plot_roc_curve(final_model,scaled_x_test,y_test)
plt.show()

##pickling the decision tree classifier model
import pickle
filename = 'decision_tree_model.pkl'
pickle.dump(final_model, open(filename, 'wb'))



##AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
boost=AdaBoostClassifier(base_estimator=final_model)
boost.fit(scaled_x_train,y_train)
y_predict=boost.predict(scaled_x_test)

#plotting ROC curve
print('score: ', boost.score)
print()
from sklearn import metrics
metrics.plot_roc_curve(boost,scaled_x_test,y_test)
plt.show()


print('classification report')
print(classification_report(y_test,y_predict))
print(confusion_matrix(y_test,y_predict))

##Hyperparameter tuning on Adaboost classifier
param_dist={'n_estimators':[40,50,60,70,80], 'learning_rate':[0.04,0.03,0.02,0.1],'algorithm':['SAMME', 'SAMME.R']}
grid_1=RandomizedSearchCV(boost,param_distributions=param_dist,cv=5,n_jobs=-1)
grid_1.fit(scaled_x_train,y_train)

#plotting ROC curve
print('score: ', grid_1.best_score_)
print('ROC-AUC curve')
metrics.plot_roc_curve(grid_1,scaled_x_test,y_test)
plt.show()

predict=grid_1.predict(scaled_x_test)
accuracy_score(y_test, predict)
print('classification report')
print(classification_report(y_test,predict))

boosted_model=grid_1.best_estimator_
grid_1.best_params_

print('confusion_matrix')
print(confusion_matrix(y_test,predict))

##pickling the Adaboost classifier model
Pkl_Filename = "Boosted_Model.pkl"  
pickle.dump(boosted_model, open(Pkl_Filename, 'wb'))



##Linear Discriminant Analysis##
lda=LinearDiscriminantAnalysis()
lda.fit(scaled_x_train,y_train)
y_pred=lda.predict(scaled_x_test)
print(accuracy_score(y_test,y_pred))

##Hyperparameter tuning on LDA
parameters={'solver':['svd', 'lsqr', 'eigen'],'shrinkage':['auto','None']}
grid=RandomizedSearchCV(LinearDiscriminantAnalysis(store_covariance=True),param_distributions=parameters,cv=5,n_jobs=-1)
grid.fit(scaled_x_train,y_train)

grid.best_params_
grid.best_score_
lda_model=grid.best_estimator_
y_pred=lda.predict(scaled_x_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

#Plotting ROC curve for LDA
metrics.plot_roc_curve(lda_model,scaled_x_test,y_test)
plt.show()

##pickling the LDA model
Pkl_Filename = "LDA_Model.pkl"  
pickle.dump(lda_model, open(Pkl_Filename, 'wb'))























































































































