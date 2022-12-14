import pandas as pd
import numpy as np
from sklearn import linear_model

dic = {'zero': 0,'one': 1,'two':2,'three':3, 'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20}
df = pd.read_csv('C:\\Users\\rozhansystem\\Documents\\salary.csv')



df.fillna(df.mean(), inplace=True)# /Replacing missing numbers with mean's

df['experience'] = df['experience'].replace(np.nan,0)# /Replacing nan's with zero



x = df[['experience','test_score(out of 10)','interview_score(out of 10)']]
y = df['salary']

print(df)

print(pd.get_dummies(df))

#incomplete