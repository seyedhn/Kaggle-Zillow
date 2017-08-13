import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.formula.api as sm
data = pd.read_csv("C:\\Users\\Armando\\Downloads\\properties_2016.csv\\properties_2016.csv")
# Load the diabetes dataset
z=data.columns
n=len(z)
reg={}
corr={}
for i in range(n):
    for j in range(i+1,n):
        a=list(data.loc[:,z[i]])
        b=list(data.loc[:,z[j]])
        index=[i for i in range(len(a)) if (a[i]>0 and b[i]>0)]
        a=[a[i] for i in index]
        b=[b[i] for i in index]
        reg[i,j]= np.polyfit(a,b,1)
        corr[i,j]=np.corrcoef(a,b)[0,1]
        print("regression of %s and %s is :" % (z[i],z[j]),reg[i,j])


