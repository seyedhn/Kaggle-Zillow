
import pandas as pd       # data processing, CSV file I/O (e.g. pd.read_csv)

prop_df = pd.read_csv("properties_2016.csv")

#Create a new DataFrame which contains only a small amount of data
frame = pd.DataFrame(data=None, columns=prop_df.columns, index=None)
frame_size = 1000 #Set the number of properties that we would like to perform regression on
count = 0
for i in prop_df.index: #If the property has non-zero bedroom and total livng area, add it to frame.
   if prop_df.loc[i]['bedroomcnt'] > 0 and prop_df.loc[i]['finishedsquarefeet12'] > 0:
       frame.loc[count] = prop_df.loc[i]
       count += 1
   if count == frame_size:
       break

#print frame[['bedroomcnt', 'finishedsquarefeet12', 'taxvaluedollarcnt']]

#Set the initial guess values for our parameters and the learning rate alpha
theta_0 = -500
theta_1 = 80000
theta_2 = 200
alpha = 0.0000001

#Number of bedrooms and the total living area are the features, and the total tax value is the target

#Update the parameters
for i in frame.index:
    
    x1 = frame.loc[i]['bedroomcnt']
    x2 = frame.loc[i]['finishedsquarefeet12']
    h = theta_0 + theta_1*x1 + theta_2*x2
    
    theta_0 = theta_0 + alpha*(frame.loc[i]['taxvaluedollarcnt'] - h)
    theta_1 = theta_1 + alpha*(frame.loc[i]['taxvaluedollarcnt'] - h)*x1
    theta_2 = theta_2 + alpha*(frame.loc[i]['taxvaluedollarcnt'] - h)*x2


#Print the parameters                              
print "theta0 = " + str(theta_0)
print "theta_1 = " + str(theta_1)
print "theta_2 = " + str(theta_2)                             

#Calculate the average error 
error = 0
for i in frame.index:
    x1 = frame.loc[i]['bedroomcnt']
    x2 = frame.loc[i]['finishedsquarefeet12']
    h = theta_0 + theta_1*x1 + theta_2*x2
    y = frame.loc[i]['taxvaluedollarcnt']
    error = error + (y-h)/h
print "Average error " + str(round(error/len(frame), 3))
