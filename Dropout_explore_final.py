# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:13:30 2020

@author: J052311
"""



# Load libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#prepapre the environment for graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Users/J052311/Graphviz2.38/release/bin'
import graphviz 

plt.style.use('ggplot')
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

###################################################################

#Read data in a panda DATAframe + Format and explore data
df=pd.read_stata('C:\\Users\\J052311\Documents\\Drop_for_tree_bootstrapped.dta')
feature_cols_new=['term','f_score', 'pmt_type_string']
Xnew= df[feature_cols_new] # a data frame 
Y=df['dropout_acct']
Xdummies=pd.get_dummies(Xnew)
# Split data into 70% train and 30% test
SEED = 1
X_train, X_test, y_train, y_test = train_test_split(Xdummies,Y, test_size=0.3,
stratify=Y,
random_state=SEED)


#//////////////////////////////////////////////
#EDA
#//////////////////////////////////////////////////////////////
#//////////////////////////////////////////

"Learn the data
df.head()
df.info()
df.describe()
type(df)
df.shape
print(df.columns)
print(df.keys())
a=list(df.columns)
q=[0.25,0.75]
df.quantile(q)



#plotting series (padas) using series directly
df.boxplot('loanterm_final', 'dropout_acct', rot=60)
df.boxplot('effective_origfico_adj', 'dropout_acct', rot=60)
_ = pd.plotting.scatter_matrix(Xdummies, figsize = [8, 8],
s=150, marker = 'D')




#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
"EXPLORE KNN in a loop for n_neighbors
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////

#loop over k_neighbors
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()



#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
"EXPLORE KNN with hyperparameter tuning
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
knn = KNeighborsClassifier()
parameters = {'n_neighbors': np.arange(1, 10)}
knn_cv = GridSearchCV(knn, param_grid=parameters, cv=3)
knn_cv.fit(X_train, y_train)
knn_cv.best_params_ #{'n_neighbors': 8}
knn_cv.best_score_ #0.8198882752491589

# predict probabilities
knn_probs = knn_cv.predict_proba(X_test)
y_pred = knn_cv.predict(X_test)

#confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


"""

[[55081   668]
 [11447   317]]


              precision    recall  f1-score   support

           0       0.83      0.99      0.90     55749
           1       0.32      0.03      0.05     11764

   micro avg       0.82      0.82      0.82     67513
   macro avg       0.57      0.51      0.48     67513
weighted avg       0.74      0.82      0.75     67513
"""




knn_probs=knn_probs[:,1]
knn_auc = roc_auc_score(y_test, knn_probs)
print('KNN: ROC AUC=%.3f' % (knn_auc)) #KNN: ROC AUC=0.600
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(knn_fpr, knn_tpr, marker='.', label='KNN')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()




knn_precision, knn_recall, _ = precision_recall_curve(y_test, knn_probs)
knn_f1, knn_auc = f1_score(y_test, y_pred), auc(knn_recall, knn_precision)

# summarize scores
print('KNN: f1=%.3f auc=%.3f' % (knn_f1, knn_auc)) #KNN: f1=0.050 auc=0.240
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(knn_recall, knn_precision, marker='.', label='KNN')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()



#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
"EXPLORE decision tree
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier


param_dist = {"max_depth": [2, None],
              "max_features": randint(1, 4),
              "min_samples_leaf": randint(1, 4),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=3)

# Fit it to the data
tree_cv.fit(Xdummies,Y)

tree_cv.best_params_
tree_cv.best_score_ 

"""
{'criterion': 'entropy',
 'max_depth': 2,
 'max_features': 1,
 'min_samples_leaf': 2}

tree_cv.best_score_: 0.8257577440755767

"""

# predict probabilities
tree_probs = tree_cv.predict_proba(X_test)
y_pred = tree_cv.predict(X_test)

#confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
[[55749     0]
 [11764     0]]
              precision    recall  f1-score   support

           0       0.83      1.00      0.90     55749
           1       0.00      0.00      0.00     11764

   micro avg       0.83      0.83      0.83     67513
   macro avg       0.41      0.50      0.45     67513
weighted avg       0.68      0.83      0.75     67513

"""


tree_probs=tree_probs[:,1]
tree_auc = roc_auc_score(y_test, tree_probs)
print('tree: ROC AUC=%.3f' % (tree_auc)) #tree: ROC AUC=0.606
tree_fpr, tree_tpr, _ = roc_curve(y_test, tree_probs)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(tree_fpr, tree_tpr, marker='.', label='tree')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

tree_precision, tree_recall, _ = precision_recall_curve(y_test, tree_probs)
tree_f1, tree_auc = f1_score(y_test, y_pred), auc(tree_recall, tree_precision)

# summarize scores
print('tree: f1=%.3f auc=%.3f' % (tree_f1, tree_auc)) #tree: f1=0.000 auc=0.288
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(tree_recall, tree_precision, marker='.', label='tree')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()



# Create a pd.Series of features importances
importances_tree_cv = pd.Series(tree_cv.best_estimator_.feature_importances_, index = Xdummies.columns)
# Sort importances_rf
sorted_importances_tree_cv = importances_tree_cv.sort_values()
# Make a horizontal bar plot
sorted_importances_tree_cv.plot(kind='barh', color='lightgreen'); plt.show()

#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
"EXPLORE LOGISTIC REGRESSION
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='lbfgs')
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}


# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=3)
# Fit it to the data
logreg_cv.fit(Xdummies,Y)


logreg_cv.best_params_ 
logreg_cv.best_score_ 

"""
{'C': 0.0007196856730011522}
0.8257621876708007
"""

y_pred=logreg_cv.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


"""

[[55749     0]
 [11763     1]]
              precision    recall  f1-score   support

           0       0.83      1.00      0.90     55749
           1       1.00      0.00      0.00     11764

   micro avg       0.83      0.83      0.83     67513
   macro avg       0.91      0.50      0.45     67513
weighted avg       0.86      0.83      0.75     67513



"""

# predict probabilities
logreg_probs = logreg_cv.predict_proba(X_test)[:,1]
y_pred = logreg_cv.predict(X_test)

fpr, tpr, thresholds = roc_curve(y_test, logreg_probs)
#a previous example
#plot_roc_curve_alpha(y_test, RFC_probabilities[:,1],'g','Random Forest Test Results ',alpha)



plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, marker='.', label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
# show the legend
plt.legend()
plt.show()



logreg_cv_precision, logreg_cv_recall, _ = precision_recall_curve(y_test, logreg_probs)
logreg_cv_f1, logreg_cv_auc = f1_score(y_test, y_pred), auc(logreg_cv_recall, logreg_cv_precision)

# summarize scores
print('logreg_cv: f1=%.3f auc=%.3f' % (logreg_cv_f1, logreg_cv_auc)) logreg_cv: f1=0.000 auc=0.247
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(logreg_cv_recall, logreg_cv_precision, marker='.', label='logreg_cv')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()



#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
"EXPLORE SVM
#collecting probabilities takes too much time
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////


# Instantiate an RBF SVM
svm = SVC()

# Instantiate the GridSearchCV object and run the search
#parameters = {'C':[0.1, 1, 10], 'gamma':[0.00001, 0.0001, 0.001, 0.01, 0.1]}
parameters = {'C':[0.1, 10], 'gamma':[0.00001, 0.01]}
searcher = GridSearchCV(svm, parameters,cv=3)
searcher.fit(X_train, y_train)

# Report the best parameters and the corresponding score
print("Best CV params", searcher.best_params_)
print("Best CV accuracy", searcher.best_score_)
prediction2=svm.predict(X_test)
print(confusion_matrix(y_test, prediction2))
print(classification_report(y_test, prediction2))

# Report the test accuracy using these best parameters
print("Test accuracy of best grid search hypers:", searcher.score(X_test, y_test))
plot_classifier(X_train, y_train, svm_small, lims=(11,15,0,6))



#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
"EXPLORE random forest with variable importance
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////



# Basic imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as MSE
# Set seed for reproducibility
rf = RandomForestClassifier(max_depth=10,n_estimators=400, min_samples_leaf=0.12, random_state=0,verbose=2,n_jobs=-1)
# Fit 'rf' to the training set
rf.fit(X_train, y_train)
# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)
y_prob=rf.predict_proba(X_test)

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2) #0.41743014100842046
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

#confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
[[55749     0]
 [11764     0]]
              precision    recall  f1-score   support

           0       0.83      1.00      0.90     55749
           1       0.00      0.00      0.00     11764

   micro avg       0.83      0.83      0.83     67513
   macro avg       0.41      0.50      0.45     67513
weighted avg       0.68      0.83      0.75     67513



"""



rf_probs=y_prob[:,1]
rf_auc = roc_auc_score(y_test, rf_probs)
print('rf: ROC AUC=%.3f' % (rf_auc)) #rf: ROC AUC=0.605
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(rf_fpr, rf_tpr, marker='.', label='rf')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)
rf_f1, rf_auc = f1_score(y_test, y_pred), auc(rf_recall, rf_precision)

# summarize scores
print('rf: f1=%.3f auc=%.3f' % (rf_f1, rf_auc)) #rf: f1=0.000 auc=0.288
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(rf_recall, rf_precision, marker='.', label='rf')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

alpha=1
plot_roc_curve_alpha(y_test, rf_probs,'g','Random Forest Test Results ',alpha)

# Create a pd.Series of features importances
importances_rf = pd.Series(rf.feature_importances_, index = Xdummies.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()



#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
"EXPLORE random forest with variable importance&hypertuning+pther methods of improvong RF model
#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
"""random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
"""
random_grid = {'max_features': max_features,               
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               }

from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(random_grid)



rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


print("Best rf_random params", rf_random.best_params_)
print("Best rf_random accuracy", rf_random.best_score_)

"""
Best rf_random params {'min_samples_split': 10, 'min_samples_leaf': 4, 'max_features': 'auto'}
Best rf_random accuracy 0.819831143274297

"""
y_prob=rf_random.predict_proba(X_test)[:,1]
y_pred = rf_random.predict(X_test)


# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2) #0.42
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

#confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


"""
[[54873   876]
 [11267   497]]
              precision    recall  f1-score   support

           0       0.83      0.98      0.90     55749
           1       0.36      0.04      0.08     11764

   micro avg       0.82      0.82      0.82     67513
   macro avg       0.60      0.51      0.49     67513
weighted avg       0.75      0.82      0.76     67513

"""

rf_auc = roc_auc_score(y_test, y_prob)
print('rf: ROC AUC=%.3f' % (rf_auc)) #rf: ROC AUC=0.605
rf_fpr, rf_tpr, _ = roc_curve(y_test, y_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(rf_fpr, rf_tpr, marker='.', label='rf')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


rf_precision, rf_recall, _ = precision_recall_curve(y_test, y_prob)
rf_f1, rf_auc = f1_score(y_test, y_pred), auc(rf_recall, rf_precision)

# summarize scores
print('rf: f1=%.3f auc=%.3f' % (rf_f1, rf_auc)) rf: f1=0.076 auc=0.265
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(rf_recall, rf_precision, marker='.', label='rf')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()



# Create a pd.Series of features importances
importances_rf = pd.Series(rf_random.best_estimator_.feature_importances_, index = Xdummies.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()



"""
Evaluate Random Search
To determine if random search yielded a better model, we compare the base model with the best random search model.
"""

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_test, y_test)

best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, X_test, y_test)

print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))

"""
Model Performance
Average Error: 0.2692 degrees.
Accuracy = -inf%.
Model Performance
Average Error: 0.1799 degrees.
Accuracy = -inf%.

Improvement of nan%.

"""



"""//////////////////////////////////////////////////////////////
///////////////////////////////////////////////////
"EXPLORE Neural networks
"""
#runfile('C:\\Users\\J052311\\Documents\\libGeneral.py')
#this is from Selcuk
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(activation='relu', hidden_layer_sizes=(5),tol=1e-02,verbose=2,random_state=2,n_iter_no_change=5)
mlp.fit(X_train,y_train)

y_pred = mlp.predict(X_test)
y_prob=mlp.predict_proba(X_test)


# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2) 
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test)) #0.42


#confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
[[55749     0]
 [11764     0]]
              precision    recall  f1-score   support

           0       0.83      1.00      0.90     55749
           1       0.00      0.00      0.00     11764

   micro avg       0.83      0.83      0.83     67513
   macro avg       0.41      0.50      0.45     67513
weighted avg       0.68      0.83      0.75     67513

"""


mlp_auc = roc_auc_score(y_test, y_prob)
print('mlp: ROC AUC=%.3f' % (mlp_auc)) #rf: ROC AUC=0.644
mlp_fpr, mlp_tpr, _ = roc_curve(y_test, y_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(mlp_fpr, mlp_tpr, marker='.', label='rf')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


mlp_precision, mlp_recall, _ = precision_recall_curve(y_test, y_prob)
mlp_f1, mlp_auc = f1_score(y_test, y_pred), auc(mlp_recall, mlp_precision)

# summarize scores
print('rf: f1=%.3f auc=%.3f' % (mlp_f1, mlp_auc)) #rf: f1=0.000 auc=0.265
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(mlp_recall, mlp_precision, marker='.', label='rf')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()




"""
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
"EXPLORE Bagging + plot_classifier

#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
"""
# Import models and utility functions
from sklearn.ensemble import BaggingClassifier


# Instantiate a classification-tree 'dt'
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=0.16, random_state=SEED)
# Instantiate a BaggingClassifier 'bc'
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
# Fit 'bc' to the training set
bc.fit(X_train, y_train)


# Predict test set labels
y_pred = bc.predict(X_test)
y_prob=bc.predict_proba(X_test)[:,1]


# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2) #0.42
# Print the test set RMSE
print('Test set RMSE of bc: {:.2f}'.format(rmse_test))

# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(accuracy)) #0.826

#confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""

[[55749     0]
 [11764     0]]
              precision    recall  f1-score   support

           0       0.83      1.00      0.90     55749
           1       0.00      0.00      0.00     11764

   micro avg       0.83      0.83      0.83     67513
   macro avg       0.41      0.50      0.45     67513
weighted avg       0.68      0.83      0.75     67513

C:\Users\J052311\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\metrics\classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
    
    """
    
bc_auc = roc_auc_score(y_test, y_prob)
print('rf: ROC AUC=%.3f' % (bc_auc)) #rf: ROC AUC=0.598
bc_fpr, bc_tpr, _ = roc_curve(y_test, y_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(bc_fpr, bc_tpr, marker='.', label='rf')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()
    
    
bc_precision, bc_recall, _ = precision_recall_curve(y_test, y_prob)
bc_f1, bc_auc = f1_score(y_test, y_pred), auc(bc_recall, bc_precision)

# summarize scores
print('bc: f1=%.3f auc=%.3f' % (bc_f1, bc_auc)) #bc: f1=0.000 auc=0.418
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(bc_recall, bc_precision, marker='.', label='bc')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()


# Create a pd.Series of features importances
importances_bc = np.mean([
    dt.feature_importances_ for dt in bc.estimators_
], axis=0)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()


#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////
"EXPLORE Out-of-Bag evaluation medhod
#//////////////////////////////////////////////
#//////////////////////////////////////////////
#//////////////////////////////////////////////

# Instantiate a BaggingClassifier 'bc'; set oob_score= True
bc_oob = BaggingClassifier(base_estimator=dt, n_estimators=300,
oob_score=True, n_jobs=-1)
# Fit 'bc' to the traing set
bc_oob.fit(X_train, y_train)


# Predict test set labels
y_pred = bc_oob.predict(X_test)
y_prob=bc_oob.predict_proba(X_test)[:,1]


# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2) #0.42
# Print the test set RMSE
print('Test set RMSE of bc: {:.2f}'.format(rmse_test))

# Evaluate and print test-set accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of Bagging Classifier: {:.3f}'.format(test_accuracy)) #0.826
# Extract the OOB accuracy from 'bc'
oob_accuracy = bc_oob.oob_score_
# Print test set accuracy
print('Test set accuracy: {:.3f}'.format(test_accuracy))
# Print OOB accuracy
print('OOB accuracy: {:.3f}'.format(oob_accuracy))

"""
Test set accuracy: 0.826
OOB accuracy: 0.826
"""


#//////////////////////////////////////////////
"EXPLORE ADA BOOSTING
#//////////////////////////////////////////////

# Import models and utility functions
from sklearn.ensemble import AdaBoostClassifier

# Instantiate an AdaBoost classifier 'adab_clf'
adb_clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100)
# Fit 'adb_clf' to the training set
adb_clf.fit(X_train, y_train)
# Predict the test set probabilities of positive class
y_pred = adb_clf.predict(X_test)
y_prob=adb_clf.predict_proba(X_test)[:,1]


# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2) #0.42
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of ADA BOOSTING Classifier: {:.3f}'.format(accuracy)) #0.826

#confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
[[55749     0]
 [11764     0]]
              precision    recall  f1-score   support

           0       0.83      1.00      0.90     55749
           1       0.00      0.00      0.00     11764

   micro avg       0.83      0.83      0.83     67513
   macro avg       0.41      0.50      0.45     67513
weighted avg       0.68      0.83      0.75     67513

C:\Users\J052311\AppData\Local\Continuum\anaconda3\lib\site-packages\sklearn\metrics\classification.py:1143: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
  'precision', 'predicted', average, warn_for)
"""

adb_clf_auc = roc_auc_score(y_test, y_prob)
print('rf: ROC AUC=%.3f' % (adb_clf_auc)) #rf: ROC AUC=0.653
adb_clf_fpr, adb_clf_tpr, _ = roc_curve(y_test, y_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(adb_clf_fpr, adb_clf_tpr, marker='.', label='rf')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


adb_clf_precision, adb_clf_recall, _ = precision_recall_curve(y_test, y_prob)
adb_clf_f1, adb_clf_auc = f1_score(y_test, y_pred), auc(adb_clf_recall, adb_clf_precision)

# summarize scores
print('rf: f1=%.3f auc=%.3f' % (adb_clf_f1, adb_clf_auc)) #rf: f1=0.000 auc=0.288
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(adb_clf_recall, adb_clf_precision, marker='.', label='adb_clf')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

# Create a pd.Series of features importances
importances_bc = np.mean([
    dt.feature_importances_ for dt in adb_clf.estimators_
], axis=0)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()



#//////////////////////////////////////////////
"EXPLORE Gradient Boosting
#//////////////////////////////////////////////

# Import models and utility functions
from sklearn.ensemble import GradientBoostingClassifier

# Instantiate an AdaBoost classifier 'adab_clf'
# Instantiate a GradientBoostingRegressor 'gbt'
gbt = GradientBoostingClassifier(n_estimators=300, max_depth=1, random_state=SEED)
# Fit 'gbt' to the training set
gbt.fit(X_train, y_train)
# Predict the test set probabilities of positive class
y_pred = gbt.predict(X_test)
y_prob=gbt.predict_proba(X_test)[:,1]


# Predict the test set probabilities of positive class
y_pred = gbt.predict(X_test)
y_prob=gbt.predict_proba(X_test)[:,1]


# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2) #0.42
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of ADA BOOSTING Classifier: {:.3f}'.format(accuracy)) #0.826

#confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

gbt_auc = roc_auc_score(y_test, y_prob)
print('rf: ROC AUC=%.3f' % (gbt_auc)) #rf: ROC AUC=0.647
gbt_fpr, gbt_tpr, _ = roc_curve(y_test, y_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(gbt_fpr, gbt_tpr, marker='.', label='gbt')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()



gbt_precision, gbt_recall, _ = precision_recall_curve(y_test, y_prob)
gbt_f1, gbt_auc = f1_score(y_test, y_pred), auc(gbt_recall, gbt_precision)

# summarize scores
print('rf: f1=%.3f auc=%.3f' % (gbt_f1, gbt_auc)) #rf: f1=0.000 auc=0.278
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(gbt_recall, gbt_precision, marker='.', label='gbt')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()


# Create a pd.Series of features importances
importances_rf = pd.Series(gbt.feature_importances_, index = Xdummies.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()



#//////////////////////////////////////////////
"EXPLORE Stochastic Gradient Boosting
#//////////////////////////////////////////////

sgbt = GradientBoostingClassifier(max_depth=1,
subsample=0.8,
max_features=0.2,
n_estimators=300,
random_state=SEED)
# Fit 'sgbt' to the training set
sgbt.fit(X_train, y_train)
# Predict the test set probabilities of positive class
y_pred = sgbt.predict(X_test)
y_prob=sgbt.predict_proba(X_test)[:,1]

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2) #0.42
# Print the test set RMSE
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

# Evaluate and print test-set accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy of SGBT Classifier: {:.3f}'.format(accuracy)) #0.826

#confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
[[55743     6]
 [11754    10]]
              precision    recall  f1-score   support

           0       0.83      1.00      0.90     55749
           1       0.62      0.00      0.00     11764

   micro avg       0.83      0.83      0.83     67513
   macro avg       0.73      0.50      0.45     67513
weighted avg       0.79      0.83      0.75     67513
"""

sgbt_auc = roc_auc_score(y_test, y_prob)
print('rf: ROC AUC=%.3f' % (sgbt_auc)) #rf: ROC AUC=0.641
sgbt_fpr, sgbt_tpr, _ = roc_curve(y_test, y_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(sgbt_fpr, sgbt_tpr, marker='.', label='sgbt')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


sgbt_precision, sgbt_recall, _ = precision_recall_curve(y_test, y_prob)
sgbt_f1, sgbt_auc = f1_score(y_test, y_pred), auc(sgbt_recall, sgbt_precision)

# summarize scores
print('rf: f1=%.3f auc=%.3f' % (sgbt_f1, sgbt_auc)) #rf: f1=0.002 auc=0.273
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(sgbt_recall, sgbt_precision, marker='.', label='sgbt')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

# Create a pd.Series of features importances
importances_rf = pd.Series(sgbt.feature_importances_, index = Xdummies.columns)
# Sort importances_rf
sorted_importances_rf = importances_rf.sort_values()
# Make a horizontal bar plot
sorted_importances_rf.plot(kind='barh', color='lightgreen'); plt.show()



#//////////////////////////////////////////////
"EXPLORE Xgboost
#//////////////////////////////////////////////
import xgboost as xgb

D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)

param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 2} 

steps = 20  # The number of training iterations

model = xgb.train(param, D_train, steps)


preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])


print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))



clf = xgb.XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.1, 0.2 ],
     "colsample_bytree" : [ 0.3, 0.4 ]
     }
grid = GridSearchCV(clf,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)

grid.fit(X_train, y_train)
y_pred=grid.predict(X_test)
y_prob=grid.predict_proba(X_test)[:,1]

print("Precision = {}".format(precision_score(y_test, y_pred, average='macro')))
Precision = 0.6651552192784734
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
Recall = 0.5
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
Accuracy = 0.8257520773776902



print("Best xgb params", grid.best_params_)
Best xgb params {'colsample_bytree': 0.3, 'eta': 0.05, 'gamma': 0.2, 'max_depth': 8, 'min_child_weight': 3}

print("Best xgb accuracy", grid.best_score_)
Best xgb accuracy -0.4365497793469502

# Evaluate the test set RMSE
rmse_test = MSE(y_test, y_pred)**(1/2) #0.42
# Print the test set RMSE
print('Test set RMSE of xgb: {:.2f}'.format(rmse_test))

#confusion matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""
Test set RMSE of xgb: 0.42
[[55686    63]
 [11700    64]]
              precision    recall  f1-score   support

           0       0.83      1.00      0.90     55749
           1       0.50      0.01      0.01     11764

   micro avg       0.83      0.83      0.83     67513
   macro avg       0.67      0.50      0.46     67513
weighted avg       0.77      0.83      0.75     67513
"""

xgb_auc = roc_auc_score(y_test, y_prob)
print('xgb: ROC AUC=%.3f' % (xgb_auc)) #xgb: ROC AUC=0.667
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, y_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(xgb_fpr, xgb_tpr, marker='.', label='xgb')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, y_prob)
xgb_f1, xgb_auc = f1_score(y_test, y_pred), auc(xgb_recall, xgb_precision)

# summarize scores
print('xgb: f1=%.3f auc=%.3f' % (xgb_f1, xgb_auc)) #xgb: f1=0.011 auc=0.290
# plot the precision-recall curves
no_skill = len(y_test[y_test==1]) / len(y_test)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(xgb_recall, xgb_precision, marker='.', label='xgb')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
# show the legend
plt.legend()
# show the plot
plt.show()

# Create a pd.Series of features importances
importances_xgb = pd.Series(grid.best_estimator_.feature_importances_, index = Xdummies.columns)
# Sort importances_xgb
sorted_importances_xgb = importances_xgb.sort_values()
# Make a horizontal bar plot
sorted_importances_xgb.plot(kind='barh', color='lightgreen'); plt.show()