# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np        # linear algebra
import pandas as pd       # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

#%matplotlib inline

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


from subprocess import check_output
#print(check_output(["ls", "../kagglezillow/"]).decode("utf8"))


#-----------------------------------------------------------------------------
#makes a scatter plot of all the logerror values in a sorted order


train_df = pd.read_csv("train_2016_v2.csv", parse_dates=["transactiondate"])
print train_df.shape        #.shape returns the number of rows and columns
print train_df.head(1)      #returns the top n rows. Default value is 5
print train_df.tail(1)      #returns the bottom n rows

plt.figure(figsize=(8,6))   #Creates a figure of size (m,n)


plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.savefig("test.png")
#plt.show()


#-----------------------------------------------------------------------------
#Brings all the values beyond the top and bottom one percentile to the one 
#perentile points and plots the distribution


ulimit = np.percentile(train_df.logerror.values, 99) #retains the value of the 99th percentile
llimit = np.percentile(train_df.logerror.values, 1)  #retains the value of 1st percentile
train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit 
train_df['logerror'].ix[train_df['logerror']<llimit] = llimit

 #All the rows with logerror values greater or less than the value of the 99th or 1st percentile
#are changed to have logerror values of ulimit or llimit. Essentially the tails of the distribution
#are cut and the values are accumulated at the cutoff points.       
        
print train_df['logerror'].ix[train_df['logerror']==ulimit]                
print ulimit
print llimit        
  
plt.figure(figsize=(10,8))
sns.distplot(train_df.logerror.values, bins=100, kde=False)
plt.xlabel('logerror', fontsize=12)
plt.savefig("test2.png")
plt.show()


#-----------------------------------------------------------------------------
#plots the histogram of the number of transactions in each month

train_df['transaction_month'] = train_df['transactiondate'].dt.month
#A fourth column of transaction months is added to the train_df database   
print train_df.shape
        
        
cnt_srs = train_df['transaction_month'].value_counts() 
#Counts how many transactions there were in each of the 12 months
#vnt_srs has indeces 1 to 12 and each index has a value which represents the
#number of transactions in that month
print cnt_srs

plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
plt.xticks(rotation='vertical')  #rotates the xticks by 90 degrees
plt.xlabel('Month of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.savefig("test3.png")


#-----------------------------------------------------------------------------
#Counts whether any of the parcelid's is repeated more than once in the database


print train_df['parcelid'].value_counts() #some parcelid's are repeated twice! and
#one is repeated 3 times
print train_df['parcelid'].value_counts().reset_index()
#in this case reset_index() makes the parcelid's to be the index and the number
#of counts to be the parcelid

print ((train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts())
#It essentially counts how many parcelid's have appeared once, twice or three times
#123 parcelid's are repeated twice in the database, and one is repeated three times


#-----------------------------------------------------------------------------
#Read and display the properties data file


prop_df = pd.read_csv("properties_2016.csv")
prop_df.shape
prop_df.head()



# .isnull gives True or False for each element that is null or not
# .sum() adds the number of missing data in each column
# .reset_index() gives an index to each column title starting from 0 to 57
# missing_df is then updated to that having missing_count > 0. It essentially 
#removes the parcelId column as it doesn't have any null values
missing_df = prop_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
print missing_df





ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.savefig("test4.png")








plt.figure(figsize=(12,12))
sns.jointplot(x=prop_df.latitude.values, y=prop_df.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.savefig("test5.png")


#Merge the train and property dataframes based on parcelID's
train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
train_df.head()


pd.options.display.max_rows = 65

dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
print dtype_df

dtype_df.groupby("Column Type").aggregate('count').reset_index()


missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]
missing_df.ix[missing_df['missing_ratio']>0.999]

mean_values = train_df.mean(axis=0)
train_df_new = train_df.fillna(mean_values, inplace=True)

print train_df.shape

# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype == 'float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0, 1])
corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
corr_df = corr_df.sort_values(by='corr_values')
print corr_df





ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12, 40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
# autolabel(rects)
plt.savefig("test6.png")


corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']
for col in corr_zero_cols:
    print(col, len(train_df_new[col].unique()))


corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]
corr_df_sel

cols_to_use = corr_df_sel.col_labels.tolist()

temp_df = train_df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.savefig("test7.png")


col = "finishedsquarefeet12"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.finishedsquarefeet12.values, y=train_df.logerror.values, size=10, color=color[4])
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Finished Square Feet 12', fontsize=12)
plt.title("Finished square feet 12 Vs Log error", fontsize=15)
plt.savefig("test8.png")


col = "calculatedfinishedsquarefeet"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.calculatedfinishedsquarefeet.values, y=train_df.logerror.values, size=10, color=color[5])
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Calculated finished square feet', fontsize=12)
plt.title("Calculated finished square feet Vs Log error", fontsize=15)
plt.savefig("test9.png")

plt.figure(figsize=(12,8))
sns.countplot(x="bathroomcnt", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Bathroom', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bathroom count", fontsize=15)
plt.savefig("test10.png")


plt.figure(figsize=(12,8))
sns.boxplot(x="bathroomcnt", y="logerror", data=train_df)
plt.ylabel('Log error', fontsize=12)
plt.xlabel('Bathroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("How log error changes with bathroom count?", fontsize=15)
plt.savefig("test11.png")


plt.figure(figsize=(12,8))
sns.countplot(x="bedroomcnt", data=train_df)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Bedroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bedroom count", fontsize=15)
plt.savefig("test12.png")


train_df['bedroomcnt'].ix[train_df['bedroomcnt']>7] = 7
plt.figure(figsize=(12,8))
sns.violinplot(x='bedroomcnt', y='logerror', data=train_df)
plt.xlabel('Bedroom count', fontsize=12)
plt.ylabel('Log Error', fontsize=12)
plt.savefig("test13.png")

col = "taxamount"
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x=train_df['taxamount'].values, y=train_df['logerror'].values, size=10, color='g')
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Tax Amount', fontsize=12)
plt.title("Tax Amount Vs Log error", fontsize=15)
plt.savefig("test14.png")




"""
from ggplot import *
p = ggplot(aes(x='yearbuilt', y='logerror'), data=train_df) + \
    geom_point(color='steelblue', size=1) + \
    stat_smooth()

print(p)

p.save("test15p.png")

p = ggplot(aes(x='latitude', y='longitude', color='logerror'), data=train_df) + \
    geom_point() + \
    scale_color_gradient(low = 'red', high = 'blue')

p.save("test16p.png")

p = ggplot(aes(x='finishedsquarefeet12', y='taxamount', color='logerror'), data=train_df) + \
    geom_point(alpha=0.7) + \
    scale_color_gradient(low = 'pink', high = 'blue')

p.save("test17p.png")

p = ggplot(aes(x='finishedsquarefeet12', y='taxamount', color='logerror'), data=train_df) + \
    geom_now_its_art()

p.save("test18p.png")

train_y = train_df['logerror'].values
cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month']+cat_cols, axis=1)
feat_names = train_df.columns.values

from sklearn import ensemble
model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
model.fit(train_df, train_y)

## plot the importances ##
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.savefig("test15.png")


import xgboost as xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}
dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, height=0.8, ax=ax)
plt.savefig("test16.png")

"""








