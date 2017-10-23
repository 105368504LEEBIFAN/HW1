import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from keras.models import load_model
import matplotlib.pyplot as plt
import time
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train history')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train','validation'],loc='upper left')
    plt.ylim(60000,110000)
    plt.show()
# Read dataset into X and Y
df = pd.read_csv('./train-v3.csv')
dataset = df.values
df2 = pd.read_csv('./valid-v3.csv')
dataset2 = df2.values
df3 = pd.read_csv('./test-v3.csv')
dataset3 = df3.values
Z=time.strftime("%Y")

A=int(Z)-df['yr_built'].values
df['yr_built'] = A
B=int(Z)-df2['yr_built'].values
df2['yr_built'] = B
c=int(Z)-df3['yr_built'].values
df3['yr_built'] = c

X=df.drop(['price','id','sale_yr','sale_month','sale_day','waterfront','yr_renovated'],axis=1).values
#X=df['yr_built'].vaures
Y=df['price'].values
X_vaild=df2.drop(['price','id','sale_yr','sale_month','sale_day','waterfront','yr_renovated'],axis=1).values
Y_vaild=df2['price'].values
X_test=df3.drop(['id','sale_yr','sale_month','sale_day','waterfront','yr_renovated'],axis=1).values
mean=np.mean(X,axis=0)
std=np.std(X,axis=0)
print 'mean=',mean
print 'std=',std
X=preprocessing.scale(X)
X_vaild=(X_vaild-mean)/std
X_test=(X_test-mean)/std

model = load_model('./my_model.h5')
Y_pre=model.predict(X_test)
Y_pre_valid=model.predict(X_vaild)
print 'Y_pre',Y_pre
plt.plot(Y_vaild/1000,Y_pre_valid/1000,"o")
plt.plot(Y_vaild/1000,Y_vaild/1000)
plt.title('House Sale Price Validation Set')
plt.ylabel('Y_pre_valid(K)')
plt.xlabel('Y_vaild(K)')
plt.show()
print len(df3['id'])
Y_PRE=[]

for i in range(len(df3['id'])):
    Y_PRE.append(i+1)
Y_PRE_Tran=np.array(Y_PRE)
Y_PRE_Tran=Y_PRE_Tran.reshape((-1,1))

combine=np.append(Y_PRE_Tran,Y_pre,1)
combine3=pd.DataFrame(combine)
combine3.columns=['id','price']
combine3.shift()[1:]
combine3.to_csv('Y_pre_10230700.csv',index=False,float_format='%.0f')




    
