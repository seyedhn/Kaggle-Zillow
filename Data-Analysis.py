import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns



data = pd.read_csv("properties_2016.csv")

#Change True/False to 1/0
data['hashottuborspa'] = data['hashottuborspa'].replace(True , 1)
data['hashottuborspa'] = data['hashottuborspa'].replace(np.nan , 0)
#Change the null values to 0
data['poolcnt'].fillna(0, inplace=True)




#Plot the missing data
missing_df = data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.sort_values(by='missing_count')
ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")



#Drop those columns which have more than 2.9 million (97%) missing data or have typedID values
for x in range(len(data.columns)):
    if missing_df.ix[x][1] > 2900000 or 'typeid' in missing_df.ix[x][0]:
        data = data.drop(missing_df.ix[x][0], 1)
 

       
#Drop those columns which have data types other than int or float        
x_cols = [col for col in data.columns if  data[col].dtype == 'float64' or data[col].dtype == 'int64']
data = data[x_cols]



#Plot a scatter plot of garagecarcount and garagetotalsqft
plt.figure(2)
plt.scatter(data['garagecarcnt'], data['garagetotalsqft'], s=120)
#Change the garagetotalsqft with values of 0 to Null
data['garagetotalsqft'] = data['garagetotalsqft'].replace(0 , np.nan)



#Plot the correlation
corr={}
sns.set(style="white")
size=data.shape[1]
z=data.columns
corr = data.loc[:,z[1:size]].corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True     
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(size, size))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})



#Drop the location columns for now
data = data.drop(['latitude', 'longitude', 'regionidcity', 'regionidcounty', 'regionidneighborhood', 'regionidzip', 'rawcensustractandblock', 'censustractandblock'] , 1)
#Drop the columns which have strong correlations
data = data.drop(['fullbathcnt', 'calculatedbathnbr'], 1)
data = data.drop(['finishedsquarefeet12', 'finishedsquarefeet15','finishedsquarefeet50','finishedfloor1squarefeet'], 1)
data = data.drop(['landtaxvaluedollarcnt', 'taxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'garagetotalsqft'] , 1)
#Drop the columns which the data are not useful or are not much meaningful.
data = data.drop(['fips', 'fireplacecnt', 'assessmentyear', 'threequarterbathnbr'] , 1)




#Remove the outlier entries from the column t
#Here we do it for numberofstories only. Each column requires a different coefficient to get the optimum number of entries
t='numberofstories'
a=data.loc[:,t]
q1=a.quantile(0.25)
m = a.median()
q3 = a.quantile(0.75)
r=q3-q1
a=list(a)
for i in range(len(a)):
    if a[i] > (m + (2* r)) or a[i] < (m - (2* r)):
        a[i]=np.nan
data.loc[:, t] = a


data.hist('yearbuilt', bins=100) 
data.hist('bedroomcnt', bins=20) 
data.hist('bathroomcnt', bins=20) 
data.hist('garagecarcnt', bins=20) 

print data.shape
print data.dropna().shape
print data.columns