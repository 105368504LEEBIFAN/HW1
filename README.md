# HW1  
目的: 房價預測  
總共分為兩個檔案  
train.py為訓練與儲存模型  
test.py 呼叫訓練好的模型以及儲存預測好的CSV檔  
train.py 中給的train/valid/test data檔名為train-v3.csv/valid-v3.csv/test-v3.csv 並與檔案放於相同目錄下  

程式的區塊方塊圖  
![image](https://github.com/105368504LEEBIFAN/HW1/blob/master/5.PNG)  

數據導入:將數值資料讀進來

`df = pd.read_csv('./train-v3.csv')`  
`dataset = df.values`  
`df2 = pd.read_csv('./valid-v3.csv')`  
`dataset2 = df2.values`  
`df3 = pd.read_csv('./test-v3.csv')`  
`dataset3 = df3.values`       

資料處理與正規化:將認為需要的資料作預處理: 我是將建構年份與現在年份做相減，帶出屋齡後再將資料套回去做訓練  
`Z=time.strftime("%Y")`  
`A=int(Z)-df['yr_built'].values`  
`df['yr_built'] = A`  
`B=int(Z)-df2['yr_built'].values`  
`df2['yr_built'] = B`  
`c=int(Z)-df3['yr_built'].values`  
`df3['yr_built'] = c`  

將認為不需要的資料drop起來: 所有資料不見得都適用，故我將自己認為不適用的資料提出，以避免影響訓練結果  
`X=df.drop(['price','id','sale_yr','sale_month','sale_day','waterfront','yr_renovated','zipcode'],axis=1).values`   
`Y=df['price'].values`  
`X_vaild=df2.drop(['price','id','sale_yr','sale_month','sale_day','waterfront','yr_renovated','zipcode'],axis=1).values` `Y_vaild=df2['price'].values`  
`X_test=df3.drop(['id','sale_yr','sale_month','sale_day','waterfront','yr_renovated','zipcode'],axis=1).values`     

資料正規化:   
1. 將train data 透過scale函數做正規化處理  
2. valid data 與 test data 則是透過 減掉train data的平均值在除以其標準差  
`mean=np.mean(X,axis=0)`  
`std=np.std(X,axis=0)`  
`X=preprocessing.scale(X)`  
`X_vaild=(X_vaild-mean)/std`  
`X_test=(X_test-mean)/std`  

網路模型建構:   
層數架構  1:32:128:256:128:32:1  
`model = Sequential()`  
`model.add(Dense(32, input_dim=X.shape[1], init='normal', activation='relu'))`  
`model.add(Dense(128, input_dim=32, init='normal', activation='relu'))`  
`model.add(Dense(256, input_dim=128, init='normal', activation='relu'))`  
`model.add(Dense(128, input_dim=256, init='normal', activation='relu'))`  
`model.add(Dense(32, input_dim=128,init='normal', activation='relu'))`  
`model.add(Dense(X.shape[1], input_dim=32, init='normal', activation='relu'))`  
`model.add(Dense(1, init='normal'))`  
`model.compile(loss='MAE', optimizer='adam')`  

訓練模型  
透過fit函數進行模型訓練，並將valid data放進去驗證  
`train_history=model.fit(X,Y,batch_size=32,epochs=220,validation_data=(X_vaild,Y_vaild),verbose=2)`  

預測結果  
`Y_pre=model.predict(X_test)`  

再將其數據存檔  
`np.savetxt('Y_pre_10211048.csv',Y_pre, delimiter=',')`   

透過verbose=2的參數應用，觀察訓練期間loss的變化  
![image](https://raw.githubusercontent.com/105368504LEEBIFAN/HW1/2bc278740c4b65535b3369cb2fcdfbddff51bd3d/1.PNG)  
畫出EPOCH次與Train分數的關係圖，觀察訓練期間大約在何時可能出現overfitting，幫助我設定EPOCH次數調整的依據  
![image](https://raw.githubusercontent.com/105368504LEEBIFAN/HW1/2bc278740c4b65535b3369cb2fcdfbddff51bd3d/22.PNG)  
Epoch=10，觀察在不同次數下，評估的準確度  
![image](https://raw.githubusercontent.com/105368504LEEBIFAN/HW1/2bc278740c4b65535b3369cb2fcdfbddff51bd3d/3.PNG)  
Epoch=220，可以看出當Epoch=220大部分點靠近線的距離比Epoch=10 更近  
![image](https://raw.githubusercontent.com/105368504LEEBIFAN/HW1/2bc278740c4b65535b3369cb2fcdfbddff51bd3d/4.PNG)

心得:  
這是我第一次接觸Python，從基本的流程架構先了解外，也試著去了解語法以及各個函數參數所代表的意義   
因為dataset的資料量很多，所以我並沒有一筆一筆去觀察在不同條件之下的變化     
我只大略的取捨這些的重要性，其中我認為屋齡對於房價其實有很重要的權重，但檔案中的建築年份放進正規化過程去做平均等等的處理沒有意義    
所以我將其與現在年份相減來做預處理進而得到屋齡在放進去訓練，讓資料變得更有效，並捨棄部分我認位比較不重要的欄位，例如買賣日期等等     
並且透過圖片的輔助分析，來調整Epoch的次數，如此一來可以知道資料大約會在什麼時候overfitting，才可以知道訓練次數，避免過度或者尚未訓練完成    
在這次的作業中學習到很多，會再繼續研究更多的基礎資料處理語法與架構，讓自己更加上手!!  
     
  
  
  









