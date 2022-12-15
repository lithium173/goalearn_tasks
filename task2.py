import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv('C:\\Users\\rozhansystem\\Documents\\salary.csv')

dic = {'zero': 0,'one': 1,'two':2,'three':3, 'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,'nineteen':19,'twenty':20}



df.fillna(df.mean(), inplace=True)# /Replacing missing numbers with mean's

df['experience'] = df['experience'].replace(np.nan,0)# /Replacing nan's with zero
for ele in df['experience']:
    if ele in dic:
        df['experience'] = df['experience'].replace(ele,dic[ele])# /Replacing name of number's with number's


x = df[['experience','test_score','interview_score']]
y = df['salary']


model = linear_model.LinearRegression()
model.fit(x,y)




def prediction(exp,test,inter):
    return model.predict([[exp,test,inter]])

m1 ,m2 , m3 , b = 0 ,0 ,0 ,0 # m is coefficient & b is intercept

iterations = 100000000000000000000 # num of iterations

learning_rate = 0.0001  # default learning rate

n = 8 # num of data's in dataframe
exp = [] # list of experience
test = [] # list of test score's
interview = [] # list of interview score's
ans = [] # list of salaries
exp = df.experience
test = df.test_score
interview = df.interview_score
ans = df.salary

def MSE(): # Mean Squared Error
    
    sum = 0
    for i in range(n):
        y_predict = prediction(exp[i],test[i],interview[i])
        sum += (ans[i]-y_predict[0])**2
    return sum


def md_eq(exp):
    sum = 0
    for i in range(n):
        y_predict = prediction(exp[i],test[i],interview[i])
        sum += exp[i]*(ans[i]-y_predict[0])
    return sum
def b_eq():
    sum = 0
    for i in range(n):
        y_predict = prediction(exp[i],test[i],interview[i])
        sum += ans[i]-y_predict[0]
    return sum

for i in range(iterations):
    cost = (1/n) * MSE()
    md1 = -(2/n)* md_eq(exp)
    md2 = -(2/n)* md_eq(test)
    md3 = -(2/n)* md_eq(interview)
    bd = -(2/n)* b_eq()
    m1 = m1 - learning_rate * md1
    m2 = m2 - learning_rate * md2
    m3 = m3 - learning_rate * md3
    b = b - learning_rate * bd
    print("m1 {}, m2 {}, m3 {}, b {}, cost {}, iteration {}".format(m1, m2, m3, b, cost, i))

m_orgin = model.coef_
b_orgin = model.intercept_
print(m_orgin,'\n',b_orgin)
