#先引入后面可能用到的包（package）
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
#保证画图时候的图像上的中文可以正常显示
import matplotlib as mpl
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
#统一使用微软雅黑
mpl.rcParams['font.sans-serif']=['SimHei']
mpl.rcParams['axes.unicode_minus']=False
import seaborn as sns  
import tushare as ts
import tensorflow as tf
from keras.models import Sequential
from tensorflow import keras
from keras import Model
from keras import layers
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
#引入相关内容
[in1]
data=ts.get_k_data(code='000001',ktype='D',
  autype='qfq', start='2008-01-01') 
data
ma_day = [20,52,252] # 月，季，年
# 计算工作日，所以天数不是30，120以及365
for ma in ma_day:
    column_name = "%s日均线" %(str(ma))
	data[column_name]data["close"].rolling(ma).mean()
#画出2008年以来收盘价和均线图，人为观察大致行走趋势
data.loc['2008-1-1':][["close","20日均线","52日均线","252日均线"]].plot(figsize=(12,6))
plt.title('平安银行走势图')
plt.xlabel('日期')
plt.show()
from scipy.stats import norm
# 首先利用传统的方法蒙特卡洛来进行模拟股票
# 计算对数收益率
log_returns = np.log(1 + data["close"].pct_change())
# 计算对数收益率平均值
u = log_returns.mean()
# 计算对数收益率方差
var = log_returns.var()
# 计算布朗运动的漂移（过去的收益率）
drift = u - (0.5 * var)
# 计算对数收益率标准差
stdev = log_returns.std()
# 一年交易252(预测一年（前文已有说明））
t_intervals = 252
# 进行迭代10次 
iterations = 10
# 重点 每天的回报 历史的收益率
daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))
daily_returns.shape # (252, 10)
# 将最后一天作为开始
S0 = data["close"].iloc[-1]
# 创建（252，10）的零矩阵
price_list = np.zeros_like(daily_returns)
# 将S0 赋予 price_list[0]
price_list[0] = S0
for t in range(1, t_intervals):
    price_list[t] = price_list[t - 1] * daily_returns[t]
plt.figure(figsize=(10,6))
plt.plot(price_list)
plt.title('平安银行预测与实际对比')
plt.xlabel('日期')
plt.show()
#画出图像，观察结果
#利用LSTM进行相关操作
data=ts.get_k_data(code='000001',ktype='D',
  autype='qfq', start='2008-01-01') 
data
#重新获取数据
data.to_csv('D:\python\data\\000001.csv’)
#预存数据在自己的硬盘上
look_back = 40
forward_days = 10
num_periods = 20
#进行赋值操作
#导入csv文件
df = pd.read_csv('D:\python\LSTM-Stock-Prices-master/data/000001.csv')
#设置index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
#保留Close一行,即只保留收盘价
df = df['Close']
df.head()
#测试代码是否正确
len(df)
#计算长度
plt.figure(figsize = (15,10))
plt.plot(df, label='000001')
plt.legend(loc='best')
plt.show()
#再次测试图像，并观察图像走势
array = df.values.reshape(df.shape[0],1)
array[:5]
from sklearn.preprocessing import MinMaxScaler
#补充导入所需要的包
scl = MinMaxScaler()
array = scl.fit_transform(array)
array[:5]
division = len(array) - num_periods*forward_days
array_test = array[division-look_back:]
array_train = array[:division]
def processData(data, look_back, forward_days,jump=1):
    X,Y = [],[]
    for i in range(0,len(data) -look_back -forward_days +1, jump):
        X.append(data[i:(i+look_back)])
        Y.append(data[(i+look_back):(i+look_back+forward_days)])
return np.array(X),np.array(Y)
X_test,y_test = processData(array_test,look_back,forward_days,forward_days)
y_test = np.array([list(a.ravel()) for a in y_test])
X,y = processData(array_train,look_back,forward_days)
y = np.array([list(a.ravel()) for a in y])
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.20, random_state=42)
print(X_train.shape)
print(X_validate.shape)
print(X_test.shape)
print(y_train.shape)
print(y_validate.shape)
print(y_test.shape)
model = load_model('D:\python\data\平安银行.h5’)
#或者导入已经做好的模型，后面代码为说明模型制作过程
NUM_NEURONS_FirstLayer = 50
NUM_NEURONS_SecondLayer = 30
EPOCHS = 50
model = Sequential()
model.add(LSTM(NUM_NEURONS_FirstLayer,input_shape=(look_back,1), return_sequences=True))
model.add(LSTM(NUM_NEURONS_SecondLayer,input_shape=(NUM_NEURONS_FirstLayer,1)))
model.add(Dense(forward_days))
model.compile(loss='mean_squared_error', optimizer='adam’)

history = model.fit(X_train,y_train,epochs=EPOCHS,validation_data=(X_validate,y_validate),shuffle=True,batch_size=2, verbose=2)
#赋值
plt.figure(figsize = (15,10))
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='best')
plt.title('损失函数')
plt.xlabel('进度')
plt.show()
#画图并观察图像走势
file_name = 'D:\python\data\\平安银行.h5'.format(look_back, forward_days, EPOCHS, NUM_NEURONS_FirstLayer, NUM_NEURONS_SecondLayer)
model.save(file_name)
print("Saved model `{}` to disk".format(file_name))
#文件保存

Xt = model.predict(X_test)
plt.figure(figsize = (15,10))
for i in range(0,len(Xt)):
    plt.plot([x + i*forward_days for x in range(len(Xt[i]))], scl.inverse_transform(Xt[i].reshape(-1,1)), color='r')
    
plt.plot(0, scl.inverse_transform(Xt[i].reshape(-1,1))[0], color='r', label='Prediction') #only to place the label
    
plt.plot(scl.inverse_transform(y_test.reshape(-1,1)), label='Target')
plt.legend(loc='best')
plt.title('平安银行预测与实际对比')
plt.xlabel('日期')
plt.show()
#画图并观察图像走势
division = len(array) - num_periods*forward_days
leftover = division%forward_days+1
array_test = array[division-look_back:]
array_train = array[leftover:division]
Xtrain,ytrain = processData(array_train,look_back,forward_days,forward_days)
Xtest,ytest = processData(array_test,look_back,forward_days,forward_days)
Xtrain = model.predict(Xtrain)
Xtrain = Xtrain.ravel()
Xtest = model.predict(Xtest)
Xtest = Xtest.ravel()
y = np.concatenate((ytrain, ytest), axis=0)

plt.figure(figsize = (15,10))
plt.plot([x for x in range(look_back+leftover, len(Xtrain)+look_back+leftover)], scl.inverse_transform(Xtrain.reshape(-1,1)), color='r', label='Train')
plt.plot([x for x in range(look_back +leftover+ len(Xtrain), len(Xtrain)+len(Xtest)+look_back+leftover)], scl.inverse_transform(Xtest.reshape(-1,1)), color='y', label='Test')
plt.plot([x for x in range(look_back+leftover, look_back+leftover+len(Xtrain)+len(Xtest))], scl.inverse_transform(y.reshape(-1,1)), color='b', label='Target')
plt.legend(loc='best')
plt.show()
#程序结束，获得最终图像
运行代码后所得结论及建议：
LSTM相较于蒙特卡洛效果出色很多，在预测的时候，LSTM在预测时候大致趋势是正确的，但是对于一小部分或者某一天的预测有些不太准确。从最后面的数据可见，模型在遇到突变的情况不能很好地预测，在这里选取的样本达到3000多，时间长达12年，尽可能将一些短时间突变的情况也带入其中，最后的训练效果仍然有些出入，但是预测长时间的趋势基本正确。在代码实例中，此次仅仅采用了收盘价来进行训练及预测，需要预测其它数据，可将代码中的‘close’进行替换成其它所需要预测的数据，训练后的结果与该模型所得结论基本一致。
因此个人建议，可以利用LSTM进行股票大致趋势的预测，但是不能过度依赖LSTM对于某一小段时间进行预测，一小段的时间可能因某些原因突变，不确定性过大，难以预测，但是放在长期中进行看，这只是长时间中的一小部分，因此在预测长时间的趋势时候，LSTM具有其适用性。

