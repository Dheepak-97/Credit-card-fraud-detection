#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler


# In[2]:


data = pd.read_csv('Credit_card.csv', index_col = 0)


# In[3]:


data.head()


# In[4]:


data.shape


# In[ ]:





# In[5]:


data.columns


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


fraud_data = data[data.Class == 1]
n_fraud_data = data[data.Class == 0]


# In[9]:


plt.figure(figsize = (7,5))
sns.countplot('Class', data = data)


# __Data is heavily imbalanced. It should be balanced to avoid bias that weakens the minority class__

# In[10]:


plt.figure(figsize = (9,5))
sns.distplot(data['Time'])


# In[11]:


#fraud transactions time distribution
plt.figure(figsize = (9,5))
sns.distplot(fraud_data['Time'])


# In[12]:


#Normal transactions time distribution
plt.figure(figsize = (9,5))
sns.distplot(n_fraud_data['Time'])


# In[13]:


plt.figure(figsize = (9,5.5))
sns.countplot('Amount', order = n_fraud_data['Amount'].value_counts().index[:10], data = n_fraud_data)
plt.title('Normal Transactions')


# In[14]:


plt.figure(figsize = (9,5.5))
sns.countplot('Amount', order = fraud_data['Amount'].value_counts().index[:10], data = fraud_data)
plt.title('Fraud Transactions')


# 1. It is quite interesting to see that __99.99 dollars__ is in __top 3__ of fraud transactions but __not in Normal transactions__ and It has __same frequency of 0 dollars__ 
# 2. Fraudsters still does the transaction with this amount even it is such a __large money__ to be unnoticed by the customer .

# In[15]:


sns.boxplot(y = data['Time'])


# In[16]:


fig, (axis1, axis2) = plt.subplots(1,2,figsize = (9,5.5))
sns.boxplot(y = data['Amount'], ax= axis1).set_title('Before log Transform')
sns.boxplot(y = np.log(data['Amount']), ax= axis2).set_title('After log Transform')


# # Undersampling

# In[17]:


# scaling the amount and time feature using minmaxscaler
m = MinMaxScaler()
data['scaled_amount'] = m.fit_transform(data['Amount'].values.reshape(-1,1))
data['scaled_time'] = m.fit_transform(data['Time'].values.reshape(-1,1))
data.drop(['Amount', 'Time'], axis = 1, inplace= True)


# In[18]:


majority_class = n_fraud_data.index
minority_class = fraud_data.index


# In[19]:


minority_class_len = len(minority_class)
minority_class_len


# In[20]:


random_majority_index = np.random.choice(majority_class, minority_class_len, replace = False)
len(random_majority_index)


# In[21]:


under_sampled_index = np.concatenate([random_majority_index,minority_class])


# In[22]:


under_sampled_data = data.loc[under_sampled_index]
under_sampled_data.index = range(0, len(under_sampled_data))


# In[23]:


plt.figure(figsize = (7,5))
sns.countplot('Class', data = under_sampled_data)


# In[24]:


# correlation on under sampled data
plt.figure(figsize = (20,12))
cor = under_sampled_data.corr()
sns.heatmap(cor, cmap='coolwarm_r', annot=True)


# In[25]:


X = under_sampled_data.drop('Class', axis= 1)
y = under_sampled_data['Class']


# In[26]:


#splitting the data into train and test data
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.33, random_state = 42, stratify = y)


# In[27]:


#train data
print('Non-fraud data is', len(y_train[y_train == 0]))
print('fraud data is',len(y_train[y_train == 1]))


# In[28]:


#test data
print('Non-fraud data is', len(y_test[y_test == 0]))
print('fraud data is',len(y_test[y_test == 1]))


# # Feature selection

# In[29]:


X1 = data.drop('Class', axis= 1)
y1 = data['Class']


# In[30]:


sm = SMOTE(random_state=42, ratio = 1)
X_new, y_new = sm.fit_sample(X1,y1)


# In[31]:


#after sampling
print('Non-fraud data is', len(y_new[y_new == 0]))
print('fraud data is',len(y_new[y_new == 1]))


# In[32]:


X1_df = data.drop('Class', axis= 1)
Smote_data = pd.DataFrame(X_new, columns= X1_df.columns)
Smote_data['Class'] = y_new


# In[33]:


rf_clf = RandomForestClassifier()
rf = rf_clf.fit(X_new,y_new) 


# In[67]:


fimp_col = []
fimp = []
for i,column in enumerate(X1):
    fimp_col.append(column)
    fimp.append(rf.feature_importances_[i])


# In[68]:


fimp_df = pd.DataFrame(zip(fimp_col,fimp), columns = ['Features', 'Feature Importance'])
fimp_df = fimp_df.sort_values('Feature Importance', ascending = False).reset_index()


# In[69]:


fimp_df['F_imp_cumulative'] = fimp_df['Feature Importance'].cumsum()
fimp_df


# In[37]:


Not_important_features = list(fimp_df['Features'][12:])
print(Not_important_features)


# In[38]:


# top 10 features based on feature importance
plt.figure(figsize = (10,5))
sns.barplot('Features','Feature Importance', data = fimp_df[:11])
plt.title('Feature Importance-SMOTE Data',size = 20)


# __From above graphs we can see these 11 features covers 85% of variance in the data__

# In[39]:


# correlation on SMOTE data
plt.figure(figsize = (19,11))
cor1 = Smote_data.corr()
sns.heatmap(cor1, cmap='coolwarm_r', annot = True)


# 1. V9,V10,V12,V14,V16,V17 are negatively correlated with the class
# 2. V2,V4,V11 are Positively Correlated with the class

# In[40]:


# dropping the insignificant features
Final_data = data.drop(Not_important_features, axis=1)


# In[43]:


Final_data.head(5)


# # Cross Validation - Under sampled data

# In[44]:


skfold = StratifiedKFold(n_splits = 5)
scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}
cv_performance_metrics = list(['Fit_time','score_time','test_accuracy', 'test_Ppecision', 'test_recall', 'test_F1'])


# In[45]:


# Algorithms
log_reg = LogisticRegression()
svc = SVC()
knn = KNeighborsClassifier()
d_tree = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()

# parameters
kfold = KFold(n_splits = 5)
algo_list = list([log_reg,knn,svc,d_tree,rf_clf])
algo_name = list(['Logistic regression','K Nearest Neighbor','Support vector classifier', 'Decision Tree', 'Random Forest'])
performance_metrics = list(['Accuracy', 'Precision', 'Recall', 'F1','AUC_ROC_Score'])


# In[46]:


# cross validation on various algorithms
def cross_validation(algo,X,y,fold,scoring):
    algo_score = []
    for i in algo:
        score = cross_validate(i, X, y, cv = fold, scoring = scoring)
        cv_metrics = dict(zip(cv_performance_metrics,[round(np.mean(score[j]),6) for j in score]))
        algo_score.append(cv_metrics)
    return algo_score


# In[47]:


result = cross_validation(algo_list, X, y,skfold,scoring)


# In[48]:


cv_score = dict(zip(algo_name,result))
cv_score


# In[49]:


def model(algo,X_train,y_train, X_test, y_test):
    algo_performance = []
    for i in algo:
        clf = i.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        Accuracy = accuracy_score(y_test, y_pred)
        Precision = precision_score(y_test, y_pred)
        Recall = recall_score(y_test, y_pred)
        F1 = f1_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred)
        metrics = dict(zip(performance_metrics,[Accuracy,Precision,Recall,F1,auc_score]))
        algo_performance.append(metrics)
    return algo_performance


# In[50]:


t_result = model(algo_list,X_train, y_train, X_test, y_test)


# In[51]:


final_score = dict(zip(algo_name,t_result))
final_score


# # SMOTE during Cross-validation 

# 1. we cannot do oversampling before cross validation because when cross validation the training and validation set will have        oversampled data (minority class) in equal proportion.
# 2. Model will learn about the oversampled data in the train data and easily predicts the minority class(Fraud transactions)        which result in higher precision score.
# 3. This model will not generalize well on test data or unseen data so we need to do Smote oversampling during cross-validation.

# In[56]:


new_X = Final_data.drop('Class', axis= 1).values
new_y = Final_data['Class'].values


# In[57]:


new_X


# In[58]:


# cross validation on various algorithms
def cross_validation(algo,X,y):
    
    algo_performance = []
    
    for train, test in kfold.split(X,y):
    
        x_train = X[train]
        y_train = y[train]
        x_test = X[test]
        y_test = y[test]
        
        sm = SMOTE(random_state=42, ratio = 1.0)
        X_train_sam, y_train_sam = sm.fit_sample(x_train, y_train)
        
        for i in algo:
            clf = i.fit(X_train_sam, y_train_sam)
            y_pred = clf.predict(x_test)
            Accuracy = accuracy_score(y_test, y_pred)
            Precision = precision_score(y_test, y_pred)
            Recall = recall_score(y_test, y_pred)
            F1 = f1_score(y_test, y_pred)
            auc_score = roc_auc_score(y_test, y_pred)
            metrics = dict(zip(performance_metrics,[Accuracy,Precision,Recall,F1,auc_score]))
            algo_performance.append(metrics)
    return algo_performance


# In[59]:


#training in SMOTE data with most important features # 9:10 pm
result = cross_validation(algo_list, new_X, new_y)


# In[60]:


#oversampled data
cv_score = dict(zip(algo_name,result))
cv_score


# ## Recall score of models
# 
# 1. Logistic regression - 0.955
# 2. Knn - 0.911
# 3. Random forest - 0.879
# 4. Decision tree - 0.847 
# 5. Svc - 0.669

# In[ ]:




