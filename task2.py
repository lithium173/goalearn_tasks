import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv('C:\\Users\\rozhansystem\\Documents\\salary.csv')

dic = {'zero': 0,'one': 1,'two':2,'three':3, 'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20}
df = pd.read_csv('C:\\Users\\rozhansystem\\Documents\\salary.csv')



df.fillna(df.mean(), inplace=True)# /Replacing missing numbers with mean's

df['experience'] = df['experience'].replace(np.nan,0)# /Replacing nan's with zero
for ele in df['experience']:
    if ele in dic:
        df['experience'] = df['experience'].replace(ele,dic[ele])# /Replacing name of number's with number's


x = df[['experience','test_score(out of 10)','interview_score(out of 10)']]
y = df['salary']


model = linear_model.LinearRegression()
model.fit(x,y)

m = model.coef_
b = model.intercept_


def gradient_descent(x, y):
    #initial value of m and b
    m_curr = b_curr = 0
    #initialize number of steps
    iterations = 1000
    #Number of data points n
    n = len(x)
    #Initialize learning rate
    learning_rate = 0.001
    
    for i in range(iterations):
        y_pred = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_pred)])
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m {}, b {}, cost {}, iteration {}".format(m_curr, b_curr, cost, i))

gradient_descent(x,y)
#incomplete