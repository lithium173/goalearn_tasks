import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle

dic = {'zero': 0,'one': 1,'two':2,'three':3, 'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20}
df = pd.read_csv('C:\\Users\\rozhansystem\\Documents\\salary.csv')



df.fillna(df.mean(), inplace=True)# /Replacing missing numbers with mean's

df['experience'] = df['experience'].replace(np.nan,0)# /Replacing nan's with zero
for ele in df['experience']:
    if ele in dic:
        df['experience'] = df['experience'].replace(ele,dic[ele])# /Replacing name of number's with number's

print(df)
x = df[['experience','test_score(out of 10)','interview_score(out of 10)']]
y = df['salary']


model = linear_model.LinearRegression()
model.fit(x,y)


def prediction(exp,test,inter):
    return model.predict([[exp,test,inter]])

x1 = prediction(2,9,6)
x2 = prediction(12,10,10)

name = 'final_model.sav'
pickle.dump(model,open(name,'wb'))


loading_model = pickle.load(open(name, 'rb'))
result = loading_model.score(x, y)
print(result)