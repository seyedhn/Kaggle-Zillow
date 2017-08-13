
import pandas as pd 
import statsmodels.formula.api as sm

prop_df = pd.read_csv("properties_2016.csv")

regression = sm.ols(formula = "taxvaluedollarcnt ~ bedroomcnt + finishedsquarefeet12", data=prop_df).fit()
print regression.params
print regression.summary()
