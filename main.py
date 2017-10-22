import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
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
#print "-----------------------------------"
#print df.drop(['price'], axis=1).values
#rint "-----------------------------------"
'''
X = dataset[:, 0:22]
Y = dataset[:, 1]
X_vaild = dataset2[:, 0:21]
Y_vaild = dataset2[:, 1]
X_test = dataset3[:, 0:21]

print "-----------------------------------"
print type(df.drop(['price','id'],axis=1).values)
print "-----------------------------------"
'''
print 'Z=',Z

A=int(Z)-df['yr_built'].values
df['yr_built'] = A
B=int(Z)-df2['yr_built'].values
df2['yr_built'] = B
c=int(Z)-df3['yr_built'].values
df3['yr_built'] = c
X=df.drop(['price','id','sale_yr','sale_month','sale_day','waterfront','yr_renovated','zipcode'],axis=1).values
#X=df['yr_built'].vaures
Y=df['price'].values
X_vaild=df2.drop(['price','id','sale_yr','sale_month','sale_day','waterfront','yr_renovated','zipcode'],axis=1).values
Y_vaild=df2['price'].values
X_test=df3.drop(['id','sale_yr','sale_month','sale_day','waterfront','yr_renovated','zipcode'],axis=1).values

print 'A=',type(A)
print 'X_p',df['yr_built'].values
print 'x=',X
print 'y=',Y
mean=np.mean(X,axis=0)
std=np.std(X,axis=0)
X=preprocessing.scale(X)
X_vaild=(X_vaild-mean)/std
X_test=(X_test-mean)/std
#print "X: ", X
#print "Y: ", Y


# Define the neural network

#
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], init='normal', activation='relu'))
model.add(Dense(128, input_dim=32, init='normal', activation='relu'))
model.add(Dense(256, input_dim=128, init='normal', activation='relu'))
model.add(Dense(128, input_dim=256, init='normal', activation='relu'))
model.add(Dense(32, input_dim=128,init='normal', activation='relu'))
model.add(Dense(X.shape[1], input_dim=32, init='normal', activation='relu'))
    # No activation needed in output layer (because regression)
model.add(Dense(1, init='normal'))

    # Compile Model
model.compile(loss='MAE', optimizer='adam')
train_history=model.fit(X,Y,batch_size=32,epochs=220,validation_data=(X_vaild,Y_vaild),verbose=2)
show_train_history(train_history,'loss','val_loss')
Y_pre=model.predict(X_test)
print 'Y_pre',Y_pre
np.savetxt('Y_pre_10211048.csv',Y_pre, delimiter=',')



    
