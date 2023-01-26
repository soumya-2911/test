import pandas as pd
import numpy as np

#initialization

df=pd.read_excel("D:\SOUMYA\ml\Book1.xlsx")
x=np.transpose([df['area']/100,df['bedroom'],df['kitchen'],df['comfort']/100])
y=np.transpose(df['price'])




def train(x,y_label,iter,learning_rate):
    weight=np.zeros(len(x[0]))
    bias=0
    features=len(weight)
    m=len(y)
    for iteration in range(iter):

        
        
        slope_w=np.dot((np.dot(x,weight)+bias-y),(np.sum(x,axis=1)))
        weight=weight-learning_rate*slope_w
        
        slope_b=np.sum((np.dot(x,weight)+bias-y))/m
        print(slope_b)
        bias=bias-learning_rate*slope_b

    return weight,bias

#training
weight,bias=train(x,y,150,0.0001)

#prediction
df_test=pd.read_excel("D:\SOUMYA\ml\Book2.xlsx")
x_test=np.transpose([df_test['area']/100,df_test['bedroom'],df_test['kitchen'],df_test['comfort']/100])

price=(np.dot(x_test,weight))+bias
print('------------')
print(price)