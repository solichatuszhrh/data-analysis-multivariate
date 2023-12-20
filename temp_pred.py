### Import Packages ###
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklear.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn

### Import Data ###
data=pd.read_csv('C:/fix.csv', sep =';')
dataset=data.iloc[:,2:3].values #min_temperature
dataset=data.iloc[:,1:2].values #max_temperature
dataset=dataset.astype('float32')
dataset=np.reshape(dataset,(-1,1))

### Data Normalization ###
scaler=MinMaxScaler(feature_range=(0,1))
dataset=scaler.fit_transform(dataset)

### Data Division ###
train_size=int(len(dataset)*0.8)
test_size=len(dataset)-train_size
train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]

### Change Data Dimension ###
def create_dataset(dataset,look_back=1):
    X,Y=[],[]
    for i in range (len(dataset)-look_back-1):
        a=dataset[i:(i+look_back),0]
    X.append(a)
    Y.append(dataset[i+look_back,0])
    return np.array(X),np.array(Y)
look_back=1
X_train,Y_train=create_dataset(train,look_back)
X_test,Y_test=create_dataset(test,look_back)
#reshape input to be [samples, time steps, features]
X_train=np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
X_test=np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

### LSTM Model ###
model=Sequential()
model.add(LSTM(no_neuron,input_shape=(X_train.shape[1],X_train.shape[2])))
model.add(Dropout(0,2))
model.add(Dense(1,activation='tanh', kernel_initializer='glorot_uniform', bias_initializer='zero', use_bias=True))
model.add(Dense(1,activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zero'))
model.compile(loss='mean_squared_error',optimizer='adam')
history = model.fit(X_train,Y_train,epochs=10,batch_size=1,validation_data=(X_test,Y_test),callbacks=[EarlyStopping(monitor='val_loss',patience=10)],verbose=1,shuffle=False)
model.summary()

### Model Evaluation ###
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
#invert prediction
train_predict=scaler.inverse_transform(train_predict)
Y_train=scaler.inverse_transform([Y_train])
test_predict=scaler.inverse_transform(test_predict)
Y_test=scaler.inverse_transform([Y_test])
print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0],train_predict[:,0]))
print('Train Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_train[0],train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0],test_predict[:,0]))
print('Test Root Mean Squared Error:', np.sqrt(mean_squared_error(Y_test[0],test_predict[:,0])))
def mean_absolute_percentage_error (Y_test,test_predict) :
    Y_test, test_predict=np.array(Y_test),np.array(test_predict)
    return np.mean(np,abs((Y_test-test_predict)/Y_test))*100

### Plot Loss Function ###
plt.figure(figsize=(8,4))
plt.plot(history,history['loss'],label='Train Loss')
plt.plot(history,history['val_loss'],label='Test Loss')
plt.title('model,loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show()

### Prediction ###
df=data
df['Date'] = pd.to_datetime(df['Date'])
df.set_axis(df['Date'], inplace=True)
df.drop(columns=['Max_Temp'], inplace=True)
close_data = df['Min_Temp'].values
close_data = close_data.reshape((-1,1))
split_percent = 0.80
split = int(split_percent*len(close_data))
close_test = close_data[split:]
train_size=int(len(dataset)*0.8)
test_size=len(dataset)-train_size
train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]
look_back = 1
test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)
prediction = model.predict_generator(test_generator)
close_test = close_test.reshape((-1))
prediction = prediction.reshape((-1))
close_data = close_data.reshape((-1))
def predict(num_prediction, model):
    prediction_list = close_data[-look_back:]
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x,reshape((1,look_back, 1))
        out = model,predict(x)[0][0]
        prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
    return prediction_list
def num_prediction = 3
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)
    value=forecast.reshape((len(forecast),1))
    inversed=scaler.inverse_transform (value)
    return inversed

