# importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df_train = pd.read_csv('https://raw.githubusercontent.com/RohanRanshinge/churn-prediction/master/churn_train.csv',header=0)
#df_train.head(5)

df_test = pd.read_csv('https://raw.githubusercontent.com/RohanRanshinge/churn-prediction/master/churn_test.csv')
#df_test.head(5)

df_test.columns
df_train.columns

df_train.head(2)

df_train.loc[df_train['intplan']==' no',['intplan']] = 0
df_train.loc[df_train['intplan']==' yes',['intplan']] = 1
df_train.loc[df_train['voice']==' no',['voice']] = 0
df_train.loc[df_train['voice']==' yes',['voice']] = 1
df_train.loc[df_train['label']==' True.',['label']]=1
df_train.loc[df_train['label']==' False.',['label']]=0
df_train.head(2)

df_test.loc[df_test['intplan']==' no',['intplan']] = 0
df_test.loc[df_test['intplan']==' yes',['intplan']] = 1
df_test.loc[df_test['voice']==' no',['voice']] = 0
df_test.loc[df_test['voice']==' yes',['voice']] = 1
df_test.loc[df_test['label']==' True.',['label']]=1
df_test.loc[df_test['label']==' False.',['label']]=0
df_test.head(2)

df_train.drop('st',axis=1,inplace=True)
df_train.drop('phnum',axis=1,inplace=True)

df_test.drop('st',axis=1,inplace=True)
df_test.drop('phnum',axis=1,inplace=True)

df_train.columns

df_train.dtypes
df_test.dtypes

X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,-1]
X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:,-1]

df_train.columns

f, axes = plt.subplots(figsize=(20, 12))

cor = df_train.corr()
sns.heatmap(cor,annot=True,cmap = 'YlGnBu')

# plot
f, axes = plt.subplots(2, 5, figsize=(15, 5))
sns.distplot( df_train["acclen"] , color="skyblue", ax=axes[0, 0])
sns.distplot( df_train["arcode"] , color="olive", ax=axes[0, 1])
sns.distplot( df_train["intplan"] , color="gold", ax=axes[0, 2])
sns.distplot( df_train["voice"] , color="teal", ax=axes[0, 3])
sns.distplot( df_train["nummailmes"] , color="orange", ax=axes[0, 4])
sns.distplot( df_train["tdmin"] , color="purple", ax=axes[1, 0])
sns.distplot( df_train["tdcal"] , color="red",ax=axes[1, 1])
sns.distplot( df_train["tdchar"] , color="darkgreen", ax=axes[1, 2])
sns.distplot( df_train["temin"] , color="violet", ax=axes[1, 3])
sns.distplot( df_train["tecal"] , color="teal", ax=axes[1, 4])
f.tight_layout()
plt.show()

g = sns.FacetGrid(df_train,col='label',hue='label')
g.map(plt.hist,'ncsc').add_legend()
plt.subplots_adjust(top = 0.8)
g.fig.suptitle('Cholesterol distribution: FEMALE = 0; MALE = 1')

# MODELLING
# Logistic Regression

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X_train,y_train)
prediction_log = logreg.predict(X_test)

logreg_accuracy = logreg.score(X_test, y_test)*100
print('Accuracy of logistic regression: ',logreg_accuracy)


# Grid Search CV on Logistic Regression
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C' : np.logspace(0,4,10),
    'penalty': ['l1','l2']
}


best_model_cv = GridSearchCV(estimator=logreg, param_grid=param_grid, cv= 10)
best_model_cv.fit(X_train, y_train)
best_model_cv.best_estimator_

logreg_cv =LogisticRegression(C=21.544346900318832, class_weight=None, dual=False,
          fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='warn', n_jobs=None, penalty='l2', random_state=None,
          solver='warn', tol=0.0001, verbose=0, warm_start=False)

logreg_cv.fit(X_train,y_train)

prediction_log_cv = logreg_cv.predict(X_test)
logreg_cv_accuracy = logreg_cv.score(X_test, y_test)*100
print(logreg_cv_accuracy)

# SVM
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
pred_svm=svm.predict(X_test)
svm_accuracy = svm.score(X_test,y_test)*100
print(svm_accuracy)

# GridSearch CV on SVM
param_grid = {
    'gamma': [0.001, 0.01, 0.1]
}

best_svm_cv = GridSearchCV(estimator=svm, param_grid=param_grid, cv= 5)
best_svm_cv.fit(X_train, y_train)
best_svm_cv.best_estimator_

svm_cv = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

svm_cv.fit(X_train,y_train)

pred_svm_cv = svm_cv.predict(X_test)
svm_cv_accuracy = svm_cv.score(X_test, y_test)*100
print(svm_cv_accuracy)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()

RFC.fit(X_train,y_train)
pred_rfc = RFC.predict(X_test)
RFC_accuracy = RFC.score(X_test,y_test)*100
print('RFC Accuracy: ',RFC_accuracy)

# GridSearch CV on Random Forest Classifier
param_grid = {
    'n_estimators': [2,4]
}


best_rfc_cv = GridSearchCV(estimator=RFC, param_grid=param_grid, cv= 5)
best_rfc_cv.fit(X_train, y_train)
print (best_rfc_cv.best_params_)

rfc_cv = RandomForestClassifier(n_estimators = 4,random_state=60)

rfc_cv.fit(X_train,y_train)

pred_rfc_cv = rfc_cv.predict(X_test)
rfc_cv_accuracy = rfc_cv.score(X_test, y_test)*100
print('RFC accuracy: ',rfc_cv_accuracy)

# Assessing Model Performance
from sklearn.metrics import confusion_matrix

# Logistic Regression Confusion Matrix
cm_log = confusion_matrix(y_test, prediction_log)
cm_log
sns.heatmap(cm_log, annot = True, cmap = 'YlGnBu')
plt.show()

# GridSearchCV Logistic Regression Confusion Matrix
cm_log_cv = confusion_matrix(y_test, prediction_log_cv)
cm_log_cv
sns.heatmap(cm_log_cv, annot = True, cmap = 'BuPu')
plt.show()

# SVM Confusion Matrix
cm_svm = confusion_matrix(y_test, pred_svm)
cm_svm
sns.heatmap(cm_svm, annot = True, cmap = 'Greens')
plt.show()

# GridSearchCV SVM Confusion Matrix
cm_svm_cv = confusion_matrix(y_test, pred_svm_cv)
cm_svm_cv
sns.heatmap(cm_svm_cv, annot = True, cmap = 'Blues')
plt.show()

# Random Forest Classifer Confusion Matrix
cm_rf = confusion_matrix(y_test, pred_rfc)
cm_rf
sns.heatmap(cm_rf, annot = True, cmap = 'Reds')
plt.show()

# GridSearchCV Random Forest Classifier Confusion Matrix
cm_rf_cv = confusion_matrix(y_test, pred_rfc_cv)
cm_rf_cv
sns.heatmap(cm_rf_cv, annot = True, cmap = 'YlGnBu')
plt.show()

# Comparing accuracy of all the models
accuracy = [logreg_accuracy,logreg_cv_accuracy,svm_accuracy,svm_cv_accuracy,RFC_accuracy,rfc_cv_accuracy]
model_names = ['Logistic Reg','GridSearchCV Logistic Reg','SVM','GridSearchCV SVM','RFC','GridSearchCV RFC']

plt.figure(figsize=(10,5))
plt.yticks(np.arange(0,100,10))
sns.barplot(accuracy,model_names)
plt.xlim(70,100)
plt.xlabel('Accuracy %')
plt.ylabel('Classification Algorithms')

# Conclusions
# 1) RFC gives us the highest accuracy as compared to Logistic Regression and SVM
# 2) Taking care of false positives is very important for the business. We do not want to predict a customer assuming they will not change, when in reality they churn.
# 3) Having more leading indicators will give us better predictive powers and enable us to come up with even higher accuracy