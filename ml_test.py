
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#initialization

df=pd.read_excel("Emissions_train.xlsx")

x=np.transpose([df['Coal'],df['Oil'],df['Gas'],df['Per Capita']*10])

y=np.array(df['Total'])

#plt.scatter(np.transpose(x)[0],y)
#plt.show()


def train(x,y,iter,learning_rate):
    weight=np.zeros(len(x[0]))
    bias=0
    
    m=len(y)
    for iteration in range(iter):

        
        #print(weight)
        diff_vector_w=(np.dot(x,weight)+bias-y)/m
        
        
        
        slope_b=np.sum((np.dot(x,weight)+bias-y))/m

        for j in range(len(weight)):

            weight[j]=weight[j]-learning_rate*np.dot(diff_vector_w,np.transpose(x)[j])
            bias=bias-learning_rate*slope_b
        print(slope_b)

    return weight,bias


    

#training
weights,biases=train(x,y,400,0.000001)

#prediction
df_test=pd.read_excel("Emissions_test.xlsx")
x_test=np.transpose([df_test['Coal'],df_test['Oil'],df_test['Gas'],df_test['Per Capita']*10])

total=(np.dot(x_test,weights))+biases
print('------------')

#plt.scatter(np.transpose(x_test)[0],p)
#plt.show()

print(df_test['Country'])
print(total)
