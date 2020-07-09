from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct
from sklearn.datasets import make_blobs, make_regression, make_circles
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE, RFECV
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import librosa
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, KFold
from sklearn import datasets
from sklearn.datasets import make_classification 
from sklearn.datasets import make_blobs, make_hastie_10_2
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_boston
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import pickle 
import os, re
from PIL import Image
import pathlib
import csv
import urllib
import numpy.random as npr
import numpy.linalg as npla
from scipy.optimize import minimize
import time
# pip install git+git://github.com/mgelbart/plot-classifier.git
from plot_classifier import plot_classifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import pickle
import pandas as pd
pd.set_option("display.max_colwidth", 200)
import altair as alt
import time
# pip install ipython-autotime
import autotime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
alt.data_transformers.disable_max_rows()

def fit_and_report_all(model, X, y, Xv, yv):
    """
    fits a model and returns train and validation score and f1 score
    
    Arguments
    ---------     
    model -- sklearn classifier model
        The sklearn model
    X -- numpy.ndarray        
        The X part of the train set
    y -- numpy.ndarray
        The y part of the train set
    Xv -- numpy.ndarray        
        The X part of the validation set
    yv -- numpy.ndarray
        The y part of the validation set       
    
    Keyword arguments 
    -----------------
    mode -- str 
        The mode for calculating error (default = 'regression') 
    
    Returns
    -------
    scores -- list
        A list containing train (on X, y) and validation (on Xv, yv) scores and f1 scores 
    
    """
    model.fit(X, y)
    pred_X = model.predict(X)
    pred_Xv = model.predict(Xv)
    scores = [model.score(X,y), model.score(Xv,yv), f1_score(y, pred_X), f1_score(yv, pred_Xv)]        
    return scores

def split_data(X, y):
    """
    Shows train and test error 
    Parameters
    ----------
    model: sklearn classifier model
        The sklearn model
    X: numpy.ndarray        
        The X part (features) of the dataset
    y numpy.ndarray
        The y part (target) of the dataset
    Returns
    -------        
        X_train: numpy.ndarray
            The X part of the train dataset
        y_train: numpy.ndarray  
            The y part of the train dataset
        X_valid: numpy.ndarray  
            The X part of the validation dataset        
        y_valid
            The y part of the validation dataset        
        X_trainvalid
            The X part of the train+validation dataset        
        y_trainvalid
            The y part of the train+validation dataset        
        X_test
            The X part of the test dataset        
        y_test            
            The y part of the test dataset        
    """
    X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(X, y, train_size=0.8, random_state=22)
    X_train, X_valid, y_train, y_valid = train_test_split(X_trainvalid, y_trainvalid, 
                                                        train_size=0.75, random_state=22)

    print("Number of training examples:", len(y_train))
    print("Number of validation examples:", len(y_valid))
    print("Number of test examples:", len(y_test))
    
    return X_train, y_train, X_valid, y_valid, X_trainvalid, y_trainvalid, X_test, y_test


# In[5]:


def display_confusion_matrix_classification_report(model, X_valid, y_valid, 
                                                labels=['Non fraud', 'Fraud']):
    """
    Displays confusion matrix and classification report. 
    
    Arguments
    ---------     
    model -- sklearn classifier model
        The sklearn model
    X_valid -- numpy.ndarray        
        The X part of the validation set
    y_valid -- numpy.ndarray
        The y part of the validation set       

    Keyword arguments:
    -----------
    labels -- list (default = ['Non fraud', 'Fraud'])
        The labels shown in the confusion matrix
    Returns
    -------
        None
    """
    ### Display confusion matrix 
    disp = plot_confusion_matrix(model, X_valid, y_valid,
                                display_labels=labels,
                                cmap=plt.cm.Blues, 
                                values_format = 'd')
    disp.ax_.set_title('Confusion matrix for the dataset')

    ### Print classification report
    print(classification_report(y_valid, model.predict(X_valid)))

credit_card_data = pd.read_excel('default of credit card clients.xls', header=1)

credit_card_data = credit_card_data.drop('ID', 1)
credit_card_data['y'] = credit_card_data['default payment next month']
credit_card_data = credit_card_data.drop('default payment next month', 1)

X = credit_card_data.drop('y', 1)
y = credit_card_data['y']

X_train, y_train, X_valid, y_valid, X_trainvalid, y_trainvalid, X_test, y_test = split_data(X,y)

whole_train = pd.merge(X_train, y_train, left_index=True, right_index=True)
print('There are', len(X_train[X_train['SEX'] == 1]), 'male in the training dataset')
print('There are', len(X_train[X_train['SEX'] == 2]), 'female in the training dataset')

print('There are', len(X_train[X_train['MARRIAGE'] == 1]), 'are married in the training dataset')
print('There are', len(X_train[X_train['MARRIAGE'] == 2]), 'are single in the training dataset')

print('There are', len(X_train[X_train['EDUCATION'] == 1]), 'got graduate school degree in the training dataset')
print('There are', len(X_train[X_train['EDUCATION'] == 2]), 'got university degree in the training dataset')
print('There are', len(X_train[X_train['EDUCATION'] == 3]), 'got high school degree in the training dataset')

alt.Chart(whole_train).mark_bar(opacity = 0.3).encode(
    alt.X("AGE:O", title = "Age"),
    alt.Y("count()", stack = None),
    color=alt.Color("y:O", 
                    legend=alt.Legend(title="Default Payment, 0: No, 1: Yes"), 
                    scale=alt.Scale(scheme='goldorange'))
).properties(title = "Age over Default Payment", width = 300, height = 300)

alt.Chart(whole_train).mark_bar(opacity = 0.3).encode(
    alt.X("SEX:O", title = "Sex, 1: Male, 2: Female"),
    alt.Y("count()", stack = None),
    color=alt.Color("y:O", 
                    legend=alt.Legend(title="Default Payment, 0: No, 1: Yes"), 
                    scale=alt.Scale(scheme='goldorange'))
).properties(title = "Sex over Default Payment", width = 300, height = 300)

alt.Chart(whole_train).mark_bar(opacity = 0.3).encode(
    alt.X("EDUCATION:O", title = "Education, 0: Unknown, 1: Graduate school,  2: University, 3: High school, 4: Others, 5: Unknown, 6: Unknown"),
    alt.Y("count()", stack = None),
    color=alt.Color("y:O", 
                    legend=alt.Legend(title="Default Payment, 0: No, 1: Yes"), 
                    scale=alt.Scale(scheme='goldorange'))
).properties(title = "Education over Default Payment", width = 300, height = 300)

alt.Chart(whole_train).mark_bar(opacity = 0.3).encode(
    alt.X("MARRIAGE:O", title = "Marriage, 0: Unkonwn, 1: Married, 2: Single, 3: Others"),
    alt.Y("count()", stack = None),
    color=alt.Color("y:O", 
                    legend=alt.Legend(title="Default Payment, 0: No, 1: Yes"), 
                    scale=alt.Scale(scheme='goldorange'))
).properties(title = "Marriage over Default Payment", width = 300, height = 300)

alt.Chart(whole_train).mark_bar(opacity = 0.3).encode(
    alt.X("y:O", title = "Default Payment, 0: No, 1: Yes"),
    alt.Y("count()")
).properties(title = "Default Payment", width = 300, height = 300)

corr = whole_train.corr()

plt.subplots(figsize=(20,15))

ax = plt.axes()
corr = whole_train.corr()

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap='Blues', annot = True)

plt.show()

base_model = DummyClassifier(strategy="most_frequent")

train, valid, f1_train, f1_valid = fit_and_report_all(base_model, X_train, y_train, X_valid, y_valid)

print('Training score: ', train)
print('Valid score: ', valid)
print('Training f1 score: ', f1_train)
print('Valid f1 score: ', f1_valid)

display_confusion_matrix_classification_report(base_model, X_valid, y_valid)

def search_and_plot_lr(score_dict, para, X_train, y_train, X_valid, y_valid):
    
    for c in para:
        lr = LogisticRegression(C=c, max_iter=1000)
        train, valid, f1_train, f1_valid = fit_and_report_all(lr, X_train, y_train, X_valid, y_valid)

        score_dict['C'].append(c)
        score_dict['train_score'].append(train)
        score_dict['val_score'].append(valid)
        score_dict['train_f1_score'].append(f1_train)
        score_dict['val_f1_score'].append(f1_valid)


    df = pd.DataFrame(score_dict)
    
    print(df)

    df = df.melt(id_vars='C', value_name='score')

    plot = alt.Chart(df).mark_line().encode(
        x = alt.X('C', axis = alt.Axis(title = 'C')),
        y = alt.Y('score', axis = alt.Axis(title = 'Scores')),
        color = 'variable'
    )

    display(plot)

score_dict = {'C' : [],'train_score' : [], 'val_score' : [], 'train_f1_score' : [], 'val_f1_score' : []}
para = 10.0**np.arange(-5,5)

search_and_plot_lr(score_dict, para, X_train, y_train, X_valid, y_valid)

numeric_features = ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']

preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), numeric_features),
        ('ohe', OneHotEncoder(), categorical_features)])

X_train = pd.DataFrame(preprocessor.fit_transform(X_train),
                    index=X_train.index,
                    columns=(numeric_features + list(preprocessor.named_transformers_['ohe'].get_feature_names(categorical_features))))
X_valid = pd.DataFrame(preprocessor.transform(X_valid),
                    index=X_valid.index,
                    columns=X_train.columns)

score_dict_1 = {'C' : [],'train_score' : [], 'val_score' : [], 'train_f1_score' : [], 'val_f1_score' : []}
para_1 = 10.0**np.arange(-5,5)

search_and_plot_lr(score_dict_1, para_1, X_train, y_train, X_valid, y_valid)

classifiers = {
    'knn'           : KNeighborsClassifier(),
    'random forest' : RandomForestClassifier(n_estimators=50),
    'sklearn NN'    : MLPClassifier(max_iter=10000)
}

train_scores = dict()
valid_scores = dict()
train_f1_scores = dict()
valid_f1_scores = dict()
training_times = dict()

for classifier_name, classifier_obj in classifiers.items():
    print("Fitting", classifier_name)
    t = time.time()
    classifier_obj.fit(X_train, y_train)
    
    pred_X = classifier_obj.predict(X_train)
    pred_Xv = classifier_obj.predict(X_valid)
    
    training_times[classifier_name] = time.time() - t
    train_scores[classifier_name] = classifier_obj.score(X_train, y_train)
    valid_scores[classifier_name] = classifier_obj.score(X_valid, y_valid)
    train_f1_scores[classifier_name] = f1_score(y_train, pred_X)
    valid_f1_scores[classifier_name] = f1_score(y_valid, pred_Xv)

data = {"train score": train_scores, "valid score" : valid_scores, "train f1 score": train_f1_scores, "valid f1 score" : valid_f1_scores, "training time (s)" : training_times}
df = pd.DataFrame(data, columns=data.keys())
df.index = list(classifiers.keys())

model_names = dict()
train_scores = dict()
valid_scores = dict()
train_f1_scores = dict()
valid_f1_scores = dict()
training_times = dict()

rgr_models = {
    'KNN': GridSearchCV(KNeighborsClassifier(),
                        param_grid = {'n_neighbors': [1, 3, 6, 9, 12]},
                        cv = 5),
    'RFC': GridSearchCV(RandomForestClassifier(n_estimators=50),
                        param_grid = {
                                    "n_estimators"     : np.arange(10,120,20),
                                    "max_depth"        : [2,5,10,20,None]
                                    },
                        cv = 5),
    'MLP': GridSearchCV(MLPClassifier(max_iter=10000),
                        param_grid = {'alpha': 10.0**np.arange(-5, 5)},
                        cv = 5)}

# Prediction loop
for name, model in rgr_models.items():
    print('Fitting', name)
    t = time.time()
    
    model.fit(X_train, y_train)
    pred_X = model.predict(X_train)
    pred_Xv = model.predict(X_valid)
    
    training_times[name] = time.time() - t
    train_scores[name] = model.score(X_train, y_train)
    valid_scores[name] = model.score(X_valid, y_valid)
    train_f1_scores[name] = f1_score(y_train, pred_X)
    valid_f1_scores[name] = f1_score(y_valid, pred_Xv)

data = {"train score": train_scores, "valid score" : valid_scores, "train f1 score": train_f1_scores, "valid f1 score" : valid_f1_scores, "training time (s)" : training_times}
df = pd.DataFrame(data, columns=data.keys())

num_of_features = dict()
train_scores = dict()
valid_scores = dict()
train_f1_scores = dict()
valid_f1_scores = dict()

for i in range(1, 30):

    lr = LogisticRegression(max_iter=1000)
    rfe = RFE(estimator = LogisticRegression(max_iter=1000), n_features_to_select = i)
    rfe.fit(X_train, y_train);
    X_train_sel = X_train.iloc[:, rfe.support_]
    X_valid_sel = X_valid.iloc[:, rfe.support_]
    
    train, valid, f1_train, f1_valid = fit_and_report_all(lr, X_train_sel, y_train, X_valid_sel, y_valid)
    num_of_features[str(i)] = i
    train_scores[str(i)] = train
    valid_scores[str(i)] = valid
    train_f1_scores[str(i)] = f1_train
    valid_f1_scores[str(i)] = f1_valid

data = {"num of features" : num_of_features, "train score": train_scores, "valid score" : valid_scores, "train f1 score": train_f1_scores, "valid f1 score" : valid_f1_scores}
df = pd.DataFrame(data, columns=data.keys())
features = df.melt(id_vars = 'num of features',
                    var_name = 'score_type',
                    value_name = 'score')

alt.Chart(features).mark_line().encode(
    alt.X('num of features'),
    alt.Y('score'),
    color = 'score_type'
)

lr_model = LogisticRegression(max_iter=1000)
rfe_model = RFE(estimator = LogisticRegression(max_iter=1000), n_features_to_select = 3)
rfe_model.fit(X_train, y_train);
X_train_sel = X_train.iloc[:, rfe_model.support_]
X_valid_sel = X_valid.iloc[:, rfe_model.support_]

train, valid, f1_train, f1_valid = fit_and_report_all(lr, X_train_sel, y_train, X_valid_sel, y_valid)
print('Training score: ', train)
print('Valid score: ', valid)
print('Training f1 score: ', f1_train)
print('Valid f1 score: ', f1_valid)
features = list(X_train_sel.columns)
print("Selected features: ", features)

import eli5

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_sel, y_train)
eli5.explain_weights(lr_model, feature_names = features)

lr_model_full = LogisticRegression(max_iter=1000)
lr_model_full.fit(X_train, y_train);

features = list(X_train.columns)
eli5.explain_weights(lr_model_full, feature_names = features)

import shap
from lightgbm import LGBMClassifier
lgb = LGBMClassifier()
lgb.fit(X_train_sel, y_train)

explainer = shap.TreeExplainer(lgb)
shap_values = explainer.shap_values(X_train_sel)
shap.summary_plot(shap_values, X_train_sel)

lgb_full = LGBMClassifier()
lgb_full.fit(X_train, y_train)

explainer = shap.TreeExplainer(lgb_full)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values, X_train)

preprocessor = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), numeric_features),
        ('ohe', OneHotEncoder(), categorical_features)])

X_trainvalid = pd.DataFrame(preprocessor.fit_transform(X_trainvalid),
                        index=X_trainvalid.index,
                        columns=(numeric_features + list(preprocessor.named_transformers_['ohe'].get_feature_names(categorical_features))))
X_test = pd.DataFrame(preprocessor.transform(X_test),
                        index=X_test.index,
                        columns=X_trainvalid.columns)
rfc = RandomForestClassifier(n_estimators=50)

rfc.fit(X_trainvalid, y_trainvalid)

pred = rfc.predict(X_test)

test_score = rfc.score(X_test, y_test)
test_f1_score = f1_score(pred, y_test)

print('Testing Score:', test_score)
print('Testing f1 Score:', test_f1_score)


# | model                   | Testing Score      | Testing f1 Score   |
# |-------------------------|--------------------|--------------------|
# | RandomForestClassifier  | 0.8178333333333333 | 0.4526790185277917 |
