
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
from catboost import CatBoostRegressor
from tqdm import tqdm
import gc
import datetime as dt
from sklearn.preprocessing import LabelEncoder


# In[2]:


properties2016 = pd.read_csv('/Users/KUANGBixi/Downloads/properties_2016.csv', low_memory = False)
properties2017 = pd.read_csv('/Users/KUANGBixi/Downloads/properties_2017.csv', low_memory = False)


# In[3]:


train2016 = pd.read_csv('/Users/KUANGBixi/Downloads/train_2016_v2.csv', parse_dates=['transactiondate'], low_memory=False)
train2017 = pd.read_csv('/Users/KUANGBixi/Downloads/train_2017.csv', parse_dates=['transactiondate'], low_memory=False)


# In[4]:


def add_date_features(df):
    df["transaction_year"] = df["transactiondate"].dt.year
    df["transaction_month"] = (df["transactiondate"].dt.year - 2016)*12 + df["transactiondate"].dt.month
    df["transaction_day"] = df["transactiondate"].dt.day
    df["transaction_quarter"] = (df["transactiondate"].dt.year - 2016)*4 +df["transactiondate"].dt.quarter
    df.drop(["transactiondate"], inplace=True, axis=1)
    return df


# In[5]:


train2016 = add_date_features(train2016)
train2017 = add_date_features(train2017)


# In[6]:


sample_submission = pd.read_csv('/Users/KUANGBixi/Downloads/sample_submission-3.csv', low_memory = False)


# In[7]:


train2016 = pd.merge(train2016, properties2016, how = 'left', on = 'parcelid')
train2017 = pd.merge(train2017, properties2017, how = 'left', on = 'parcelid')


# In[8]:


train_df = pd.concat([train2016, train2017], axis = 0)
test_df = pd.merge(sample_submission[['ParcelId']], properties2016.rename(columns = {'parcelid': 'ParcelId'}), how = 'left', on = 'ParcelId')


# In[9]:


del properties2016, properties2017, train2016, train2017
gc.collect();


# In[10]:


id_feature = ['heatingorsystemtypeid','propertylandusetypeid', 'storytypeid', 'airconditioningtypeid',
        'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid', 'typeconstructiontypeid']
for c in train_df.columns:
        train_df[c]=train_df[c].fillna(-999)
        if train_df[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(train_df[c].values))
            train_df[c] = lbl.transform(list(train_df[c].values))
        if c in id_feature:
            lbl = LabelEncoder()
            lbl.fit(list(train_df[c].values))
            train_df[c] = lbl.transform(list(train_df[c].values))
            dum_df = pd.get_dummies(train_df[c])
            dum_df = dum_df.rename(columns=lambda x:c+str(x))
            train_df = pd.concat([train_df,dum_df],axis=1)
            train_df = train_df.drop([c], axis=1)


# In[11]:


id_feature = ['heatingorsystemtypeid','propertylandusetypeid', 'storytypeid', 'airconditioningtypeid',
        'architecturalstyletypeid', 'buildingclasstypeid', 'buildingqualitytypeid', 'typeconstructiontypeid']
for c in test_df.columns:
        test_df[c]=test_df[c].fillna(-999)
        if test_df[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(test_df[c].values))
            test_df[c] = lbl.transform(list(test_df[c].values))
        if c in id_feature:
            lbl = LabelEncoder()
            lbl.fit(list(test_df[c].values))
            test_df[c] = lbl.transform(list(test_df[c].values))
            dum_df = pd.get_dummies(test_df[c])
            dum_df = dum_df.rename(columns=lambda x:c+str(x))
            test_df = pd.concat([test_df,dum_df],axis=1)
            test_df = test_df.drop([c], axis=1)


# In[12]:


# drop outliers
train_df = train_df[train_df.logerror > -0.4]
train_df = train_df[train_df.logerror < 0.4]


# In[13]:


#exclude percentage of missing value > 98%
missing_perc_thresh = 0.98 
exclude_missing = []
num_rows = train_df.shape[0]
for c in train_df.columns:
    num_missing = train_df[c].isnull().sum()
    if num_missing == 0:
        continue
    missing_frac = num_missing / float(num_rows)
    if missing_frac > missing_perc_thresh:
        exclude_missing.append(c)


# In[14]:


del num_rows, missing_perc_thresh
gc.collect();


# In[15]:


#exclude unique value
exclude_unique = [] 
for c in train_df.columns:
    num_uniques = len(train_df[c].unique())
    if train_df[c].isnull().sum() != 0: #exclude nan value
        num_uniques -= 1
    if num_uniques == 1:
        exclude_unique.append(c)


# In[16]:


#exclude other features did not influent logerror
exclude_other = ['parcelid', 'logerror','propertyzoningdesc', 'propertycountylandusecode'] 
train_features = []
for c in train_df.columns:
    if c not in exclude_missing        and c not in exclude_other and c not in exclude_unique:
        train_features.append(c)


# In[17]:


train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)


# In[18]:


X_train = train_df[train_features]
y_train = train_df.logerror


# In[19]:


test_df['transactiondate'] = pd.Timestamp('2016-12-01') 
test_df = add_date_features(test_df)
X_test = test_df[train_features]


# In[20]:


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.grid_search import GridSearchCV


# In[21]:


n_folds = 10

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=100).split(X_train.values, y_train)
    rmse= -cross_val_score(model, X_train.values, y_train, scoring="mean_absolute_error", cv = kf)
    return(rmse)


# In[29]:


xgb = XGBRegressor()


# In[27]:


random = RandomForestRegressor()


# In[31]:


lgb = LGBMRegressor()


# In[32]:


extra = ExtraTreesRegressor()


# In[33]:


de = DecisionTreeRegressor()


# In[34]:


ada = AdaBoostRegressor()


# In[ ]:


score = rmsle_cv(xgb)
print("\nxgb score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(random)
print("\nrandom score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(lgb)
print("\nlgb score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(extra)
print("\nextra score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(de)
print("\nde score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


score = rmsle_cv(ada)
print("\nada score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[ ]:


param_test1 = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
gsearch1 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=60), 
   param_grid = param_test1, scoring='mean_absolute_error', cv=10)
gsearch1.fit(X_train,y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# In[ ]:


param_test2 = {'max_depth':[6, 8, 10, 12], 'min_samples_split':[80, 100, 120, 140]}
gsearch2 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=60), 
   param_grid = param_test2, scoring='mean_absolute_error', cv=10)
gsearch2.fit(X_train,y_train)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_


# In[ ]:


rf_params = {}
rf_params['n_estimators'] = 60
rf_params['max_depth'] = 6
rf_params['min_samples_split'] = 80
rf_params['min_samples_leaf'] = 30


# In[ ]:


random = RandomForestRegressor(**rf_params)


# In[29]:


random.fit(X_train, y_train)
y_pred = random.predict(X_test)


# In[30]:


submission = pd.DataFrame({'ParcelId': test_df['ParcelId'],})
test_dates = {
    '201610': pd.Timestamp('2016-09-30'),
    '201611': pd.Timestamp('2016-10-31'),
    '201612': pd.Timestamp('2016-11-30'),
    '201710': pd.Timestamp('2017-09-30'),
    '201711': pd.Timestamp('2017-10-31'),
    '201712': pd.Timestamp('2017-11-30')
}


# In[31]:


for label, test_date in test_dates.items():
    submission[label] = y_pred


# In[33]:


submission.to_csv("random_newparams.csv",index=False)


# In[34]:


sub = pd.read_csv('random_newparams.csv')


# In[35]:


sub.head()

