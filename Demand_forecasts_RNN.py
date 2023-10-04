#Import data using python pandas library
import matplotlib.pyplot as plt
import numpy as np
import math
import statistics
import pandas as pd
import tensorflow as tf

# Set a random seed for TensorFlow
tf.random.set_seed(42)

# Set a random seed for NumPy (if you're using NumPy operations within your RNN)
np.random.seed(42)

#PRET FORECASTING
data = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df = pd.DataFrame(data, columns=['Pretoria'])
df.columns = ['Sales']
df.plot(figsize=(12,8))

dates = pd.date_range('2021-07-05', periods=361, freq='D')
values = df['Sales']
df = pd.DataFrame({'date_column': dates, 'Sales': values})
df.set_index('date_column',inplace=True)
df.index.freq='D'
df.head()
df.plot(figsize=(12,6))

test = df
train_val = df.iloc[:290]
test_val = df.iloc[280:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train_val)
#transform values between 0 and 1
scaled_test = scaler.transform(test)
scaled_train_val = scaler.transform(train_val)
scaled_test_val = scaler.transform(test_val)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#define generator
n_input = 10
n_features = 1
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length = n_input, batch_size = 1)
train_val_generator = TimeseriesGenerator(scaled_train_val, scaled_train_val, length = n_input, batch_size = 1)
test_val_generator = TimeseriesGenerator(scaled_test_val, scaled_test_val, length = n_input, batch_size = 1)

from tensorflow import keras
#define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', input_shape = (n_input, n_features), return_sequences=True))
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', return_sequences=False))
model.add(tf.keras.layers.Dense(64, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(1, activation='leaky_relu'))

from keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(optimizer = optimizer, loss = 'mse')

model.fit(train_val_generator, epochs=50)

train_pred=model.predict(train_val_generator)
train_pred=scaler.inverse_transform(train_pred)

train_sub = train_val[10:]
train_sub['RNN']=train_pred
train_sub.index = range(280)

plt.plot(train_sub.index, train_sub['RNN'], color='blue', label='RNN')
plt.plot(train_sub.index, train_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the training/validation dataset')

pred_pret=model.predict(test_generator)
pred_pret=scaler.inverse_transform(pred_pret)
total_pred_pret=[]
for i in range(len(pred_pret)):
    total_pred_pret.append(int(pred_pret[i]))
#print(total_pred_pret)

test_pred=model.predict(test_val_generator)
test_pred=scaler.inverse_transform(test_pred)
test_val_pret =[]
for i in range(len(test_pred)):
    test_val_pret.append(int(test_pred[i]))
#print(test_val_pret)
test_actual = test_val[10:]
test_actual['RNN']=test_val_pret
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
RNN_mse = mean_squared_error(test_actual['Sales'],test_actual['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_actual['Sales'],test_actual['RNN'])
RNN_r2 = r2_score(test_actual['Sales'],test_actual['RNN'])
print(f'test pret_mse: {RNN_mse}')
print(f'test pret_rmse: {RNN_rmse}')
print(f'test pret_mae: {RNN_mae}')
print(f'test pret_r2: {RNN_r2}')

test_sub = test[10:]
test_sub['RNN']=total_pred_pret
test_sub.index = range(351)

plt.plot(test_sub.index, test_sub['RNN'], color='blue', label='RNN')
plt.plot(test_sub.index, test_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the test dataset')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
RNN_mse = mean_squared_error(test_sub['Sales'],test_sub['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_sub['Sales'],test_sub['RNN'])
RNN_r2 = r2_score(test_sub['Sales'],test_sub['RNN'])
print(f'entire pret_mse: {RNN_mse}')
print(f'entire pret_rmse: {RNN_rmse}')
print(f'entire pret_mae: {RNN_mae}')
print(f'entire pret_r2: {RNN_r2}')

test_pred = test[10:]
test_pred = test_pred['Sales']
test_pred = test_pred.reset_index(drop=True)
y=[]
for i in range(len(test_pred)):
    y.append(total_pred_pret[i]-test_pred[i])
x=range(351)
# Plot the residual loss
plt.figure(figsize=(10, 6))
plt.plot(x,y)
plt.title('Residual Loss of RNN Model')
plt.xlabel('Day')
plt.ylabel('Residual Loss')
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[10:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df_pret= df2['Pretoria']
#print(df_pret)

#BLOEM FORECASTING
data = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df = pd.DataFrame(data, columns=['Bloemfontein'])
df.columns = ['Sales']
df.plot(figsize=(12,8))

dates = pd.date_range('2021-07-05', periods=361, freq='D')
values = df['Sales']
df = pd.DataFrame({'date_column': dates, 'Sales': values})
df.set_index('date_column',inplace=True)
df.index.freq='D'
df.head()
df.plot(figsize=(12,6))

test = df
train_val = df.iloc[:290]
test_val = df.iloc[280:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train_val)
#transform values between 0 and 1
scaled_test = scaler.transform(test)
scaled_train_val = scaler.transform(train_val)
scaled_test_val = scaler.transform(test_val)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#define generator
n_input = 10
n_features = 1
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length = n_input, batch_size = 1)
train_val_generator = TimeseriesGenerator(scaled_train_val, scaled_train_val, length = n_input, batch_size = 1)
test_val_generator = TimeseriesGenerator(scaled_test_val, scaled_test_val, length = n_input, batch_size = 1)

from tensorflow import keras
#define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', input_shape = (n_input, n_features), return_sequences=True))
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', return_sequences=False))
model.add(tf.keras.layers.Dense(64, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(1, activation='leaky_relu'))

from keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)
model.compile(optimizer = optimizer, loss = 'mse')

model.fit(train_val_generator, epochs=100)

train_pred=model.predict(train_val_generator)
train_pred=scaler.inverse_transform(train_pred)

train_sub = train_val[10:]
train_sub['RNN']=train_pred
train_sub.index = range(280)

plt.plot(train_sub.index, train_sub['RNN'], color='blue', label='RNN')
plt.plot(train_sub.index, train_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the training/validation dataset')

pred_bloem=model.predict(test_generator)
pred_bloem=scaler.inverse_transform(pred_bloem)
total_pred_bloem=[]
for i in range(len(pred_bloem)):
    total_pred_bloem.append(int(pred_bloem[i]))
#print(total_pred_bloem)

test_pred=model.predict(test_val_generator)
test_pred=scaler.inverse_transform(test_pred)
test_val_bloem =[]
for i in range(len(test_pred)):
    test_val_bloem.append(int(test_pred[i]))
#print(test_val_bloem)
test_actual = test_val[10:]
test_actual['RNN']=test_val_bloem
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
RNN_mse = mean_squared_error(test_actual['Sales'],test_actual['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_actual['Sales'],test_actual['RNN'])
RNN_r2 = r2_score(test_actual['Sales'],test_actual['RNN'])
print(f'test bloem_mse: {RNN_mse}')
print(f'test bloem_rmse: {RNN_rmse}')
print(f'test bloem_mae: {RNN_mae}')
print(f'test bloem_r2: {RNN_r2}')

test_sub = test[10:]
test_sub['RNN']=total_pred_bloem
test_sub.index = range(351)

plt.plot(test_sub.index, test_sub['RNN'], color='blue', label='RNN')
plt.plot(test_sub.index, test_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the test dataset')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
RNN_mse = mean_squared_error(test_sub['Sales'],test_sub['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_sub['Sales'],test_sub['RNN'])
RNN_r2 = r2_score(test_sub['Sales'],test_sub['RNN'])
print(f'entire bloem_mse: {RNN_mse}')
print(f'entire bloem_rmse: {RNN_rmse}')
print(f'entire bloem_mae: {RNN_mae}')
print(f'entire bloem_r2: {RNN_r2}')

test_pred = test[10:]
test_pred = test_pred['Sales']
test_pred = test_pred.reset_index(drop=True)
y=[]
for i in range(len(test_pred)):
    y.append(total_pred_bloem[i]-test_pred[i])
x=range(351)
# Plot the residual loss
plt.figure(figsize=(10, 6))
plt.plot(x,y)
plt.title('Residual Loss of RNN Model')
plt.xlabel('Day')
plt.ylabel('Residual Loss')
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[10:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df_bloem= df2['Bloemfontein']
#print(df_bloem)

#JHB FORECASTING
data = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df = pd.DataFrame(data, columns=['Johannesburg'])
df.columns = ['Sales']
df.plot(figsize=(12,8))

dates = pd.date_range('2021-07-05', periods=361, freq='D')
values = df['Sales']
df = pd.DataFrame({'date_column': dates, 'Sales': values})
df.set_index('date_column',inplace=True)
df.index.freq='D'
df.head()
df.plot(figsize=(12,6))

test = df
train_val = df.iloc[:290]
test_val = df.iloc[280:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train_val)
#transform values between 0 and 1
scaled_test = scaler.transform(test)
scaled_train_val = scaler.transform(train_val)
scaled_test_val = scaler.transform(test_val)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#define generator
n_input = 10
n_features = 1
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length = n_input, batch_size = 1)
train_val_generator = TimeseriesGenerator(scaled_train_val, scaled_train_val, length = n_input, batch_size = 1)
test_val_generator = TimeseriesGenerator(scaled_test_val, scaled_test_val, length = n_input, batch_size = 1)

from tensorflow import keras
#define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', input_shape = (n_input, n_features), return_sequences=True))
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', return_sequences=False))
model.add(tf.keras.layers.Dense(64, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(1, activation='leaky_relu'))

from keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)
model.compile(optimizer = optimizer, loss = 'mse')

model.fit(train_val_generator, epochs=100)

train_pred=model.predict(train_val_generator)
train_pred=scaler.inverse_transform(train_pred)

train_sub = train_val[10:]
train_sub['RNN']=train_pred
train_sub.index = range(280)

plt.plot(train_sub.index, train_sub['RNN'], color='blue', label='RNN')
plt.plot(train_sub.index, train_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the training/validation dataset')

pred_jhb=model.predict(test_generator)
pred_jhb=scaler.inverse_transform(pred_jhb)
total_pred_jhb=[]
for i in range(len(pred_jhb)):
    total_pred_jhb.append(int(pred_jhb[i]))
#print(total_pred_jhb)

test_pred=model.predict(test_val_generator)
test_pred=scaler.inverse_transform(test_pred)
test_val_jhb =[]
for i in range(len(test_pred)):
    test_val_jhb.append(int(test_pred[i]))
#print(test_val_jhb)
test_actual = test_val[10:]
test_actual['RNN']=test_val_jhb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
RNN_mse = mean_squared_error(test_actual['Sales'],test_actual['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_actual['Sales'],test_actual['RNN'])
RNN_r2 = r2_score(test_actual['Sales'],test_actual['RNN'])
print(f'test jhb_mse: {RNN_mse}')
print(f'test jhb_rmse: {RNN_rmse}')
print(f'test jhb_mae: {RNN_mae}')
print(f'test jhb_r2: {RNN_r2}')

test_sub = test[10:]
test_sub['RNN']=total_pred_jhb
test_sub.index = range(351)

plt.plot(test_sub.index, test_sub['RNN'], color='blue', label='RNN')
plt.plot(test_sub.index, test_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the test dataset')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
RNN_mse = mean_squared_error(test_sub['Sales'],test_sub['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_sub['Sales'],test_sub['RNN'])
RNN_r2 = r2_score(test_sub['Sales'],test_sub['RNN'])
print(f'entire jhb_mse: {RNN_mse}')
print(f'entire jhb_rmse: {RNN_rmse}')
print(f'entire jhb_mae: {RNN_mae}')
print(f'entire jhb_r2: {RNN_r2}')

test_pred = test[10:]
test_pred = test_pred['Sales']
test_pred = test_pred.reset_index(drop=True)
y=[]
for i in range(len(test_pred)):
    y.append(total_pred_jhb[i]-test_pred[i])
x=range(351)
# Plot the residual loss
plt.figure(figsize=(10, 6))
plt.plot(x,y)
plt.title('Residual Loss of RNN Model')
plt.xlabel('Day')
plt.ylabel('Residual Loss')
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[10:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df_jhb= df2['Johannesburg']
#print(df_jhb)

#Durban forecasting
data = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df = pd.DataFrame(data, columns=['Durban'])
df.columns = ['Sales']
df.plot(figsize=(12,8))

dates = pd.date_range('2021-07-05', periods=361, freq='D')
values = df['Sales']
df = pd.DataFrame({'date_column': dates, 'Sales': values})
df.set_index('date_column',inplace=True)
df.index.freq='D'
df.head()
df.plot(figsize=(12,6))

test = df
train_val = df.iloc[:290]
test_val = df.iloc[280:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train_val)
#transform values between 0 and 1
scaled_test = scaler.transform(test)
scaled_train_val = scaler.transform(train_val)
scaled_test_val = scaler.transform(test_val)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#define generator
n_input = 10
n_features = 1
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length = n_input, batch_size = 1)
train_val_generator = TimeseriesGenerator(scaled_train_val, scaled_train_val, length = n_input, batch_size = 1)
test_val_generator = TimeseriesGenerator(scaled_test_val, scaled_test_val, length = n_input, batch_size = 1)

from tensorflow import keras
#define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', input_shape = (n_input, n_features), return_sequences=True))
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', return_sequences=False))
model.add(tf.keras.layers.Dense(64, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(1, activation='leaky_relu'))

from keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)
model.compile(optimizer = optimizer, loss = 'mse')

model.fit(train_val_generator, epochs=100)

train_pred=model.predict(train_val_generator)
train_pred=scaler.inverse_transform(train_pred)

train_sub = train_val[10:]
train_sub['RNN']=train_pred
train_sub.index = range(280)

plt.plot(train_sub.index, train_sub['RNN'], color='blue', label='RNN')
plt.plot(train_sub.index, train_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the training/validation dataset')

pred_durb=model.predict(test_generator)
pred_durb=scaler.inverse_transform(pred_durb)
total_pred_durb=[]
for i in range(len(pred_durb)):
    total_pred_durb.append(int(pred_durb[i]))
#print(total_pred_durb)

test_pred=model.predict(test_val_generator)
test_pred=scaler.inverse_transform(test_pred)
test_val_durb =[]
for i in range(len(test_pred)):
    test_val_durb.append(int(test_pred[i]))
#print(test_val_durb)
test_actual = test_val[10:]
test_actual['RNN']=test_val_durb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
RNN_mse = mean_squared_error(test_actual['Sales'],test_actual['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_actual['Sales'],test_actual['RNN'])
RNN_r2 = r2_score(test_actual['Sales'],test_actual['RNN'])
print(f'test durb_mse: {RNN_mse}')
print(f'test durb_rmse: {RNN_rmse}')
print(f'test durb_mae: {RNN_mae}')
print(f'test durb_r2: {RNN_r2}')

test_sub = test[10:]
test_sub['RNN']=total_pred_durb
test_sub.index = range(351)

plt.plot(test_sub.index, test_sub['RNN'], color='blue', label='RNN')
plt.plot(test_sub.index, test_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the test dataset')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
RNN_mse = mean_squared_error(test_sub['Sales'],test_sub['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_sub['Sales'],test_sub['RNN'])
RNN_r2 = r2_score(test_sub['Sales'],test_sub['RNN'])
print(f'entire durb_mse: {RNN_mse}')
print(f'entire durb_rmse: {RNN_rmse}')
print(f'entire durb_mae: {RNN_mae}')
print(f'entire durb_r2: {RNN_r2}')

test_pred = test[10:]
test_pred = test_pred['Sales']
test_pred = test_pred.reset_index(drop=True)
y=[]
for i in range(len(test_pred)):
    y.append(total_pred_durb[i]-test_pred[i])
x=range(351)
# Plot the residual loss
plt.figure(figsize=(10, 6))
plt.plot(x,y)
plt.title('Residual Loss of RNN Model')
plt.xlabel('Day')
plt.ylabel('Residual Loss')
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[10:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df_durb= df2['Durban']
#print(df_durb)

#PE forecasting
data = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df = pd.DataFrame(data, columns=['Port Elizabeth'])
df.columns = ['Sales']
df.plot(figsize=(12,8))

dates = pd.date_range('2021-07-05', periods=361, freq='D')
values = df['Sales']
df = pd.DataFrame({'date_column': dates, 'Sales': values})
df.set_index('date_column',inplace=True)
df.index.freq='D'
df.head()
df.plot(figsize=(12,6))

test = df
train_val = df.iloc[:290]
test_val = df.iloc[280:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train_val)
#transform values between 0 and 1
scaled_test = scaler.transform(test)
scaled_train_val = scaler.transform(train_val)
scaled_test_val = scaler.transform(test_val)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#define generator
n_input = 10
n_features = 1
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length = n_input, batch_size = 1)
train_val_generator = TimeseriesGenerator(scaled_train_val, scaled_train_val, length = n_input, batch_size = 1)
test_val_generator = TimeseriesGenerator(scaled_test_val, scaled_test_val, length = n_input, batch_size = 1)

from tensorflow import keras
#define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', input_shape = (n_input, n_features), return_sequences=True))
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', return_sequences=False))
model.add(tf.keras.layers.Dense(64, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(1, activation='leaky_relu'))

from keras.optimizers import Adam
learning_rate = 0.0001
optimizer = Adam(lr=learning_rate)
model.compile(optimizer = optimizer, loss = 'mse')

model.fit(train_val_generator, epochs=75)

train_pred=model.predict(train_val_generator)
train_pred=scaler.inverse_transform(train_pred)

train_sub = train_val[10:]
train_sub['RNN']=train_pred
train_sub.index = range(280)

plt.plot(train_sub.index, train_sub['RNN'], color='blue', label='RNN')
plt.plot(train_sub.index, train_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the training/validation dataset')

pred_PE=model.predict(test_generator)
pred_PE=scaler.inverse_transform(pred_PE)
total_pred_PE=[]
for i in range(len(pred_PE)):
    total_pred_PE.append(int(pred_PE[i]))
#print(total_pred_PE)

test_pred=model.predict(test_val_generator)
test_pred=scaler.inverse_transform(test_pred)
test_val_PE =[]
for i in range(len(test_pred)):
    test_val_PE.append(int(test_pred[i]))
#print(test_val_PE)
test_actual = test_val[10:]
test_actual['RNN']=test_val_PE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
RNN_mse = mean_squared_error(test_actual['Sales'],test_actual['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_actual['Sales'],test_actual['RNN'])
RNN_r2 = r2_score(test_actual['Sales'],test_actual['RNN'])
print(f'test PE_mse: {RNN_mse}')
print(f'test PE_rmse: {RNN_rmse}')
print(f'test PE_mae: {RNN_mae}')
print(f'test PE_r2: {RNN_r2}')

test_sub = test[10:]
test_sub['RNN']=total_pred_PE
test_sub.index = range(351)

plt.plot(test_sub.index, test_sub['RNN'], color='blue', label='RNN')
plt.plot(test_sub.index, test_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the test dataset')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
RNN_mse = mean_squared_error(test_sub['Sales'],test_sub['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_sub['Sales'],test_sub['RNN'])
RNN_r2 = r2_score(test_sub['Sales'],test_sub['RNN'])
print(f'entire PE_mse: {RNN_mse}')
print(f'entire PE_rmse: {RNN_rmse}')
print(f'entire PE_mae: {RNN_mae}')
print(f'entire PE_r2: {RNN_r2}')

test_pred = test[10:]
test_pred = test_pred['Sales']
test_pred = test_pred.reset_index(drop=True)
y=[]
for i in range(len(test_pred)):
    y.append(total_pred_PE[i]-test_pred[i])
x=range(351)
# Plot the residual loss
plt.figure(figsize=(10, 6))
plt.plot(x,y)
plt.title('Residual Loss of RNN Model')
plt.xlabel('Day')
plt.ylabel('Residual Loss')
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[10:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df_PE= df2['Port Elizabeth']
#print(df_PE)

#CT Forecasting
data = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df = pd.DataFrame(data, columns=['Cape Town'])
df.columns = ['Sales']
df.plot(figsize=(12,8))

dates = pd.date_range('2021-07-05', periods=361, freq='D')
values = df['Sales']
df = pd.DataFrame({'date_column': dates, 'Sales': values})
df.set_index('date_column',inplace=True)
df.index.freq='D'
df.head()
df.plot(figsize=(12,6))

test = df
train_val = df.iloc[:290]
test_val = df.iloc[280:]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

scaler.fit(train_val)
#transform values between 0 and 1
scaled_test = scaler.transform(test)
scaled_train_val = scaler.transform(train_val)
scaled_test_val = scaler.transform(test_val)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
#define generator
n_input = 10
n_features = 1
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length = n_input, batch_size = 1)
train_val_generator = TimeseriesGenerator(scaled_train_val, scaled_train_val, length = n_input, batch_size = 1)
test_val_generator = TimeseriesGenerator(scaled_test_val, scaled_test_val, length = n_input, batch_size = 1)

from tensorflow import keras
#define model
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', input_shape = (n_input, n_features), return_sequences=True))
model.add(tf.keras.layers.LSTM(32, dropout=0, activation = 'tanh', return_sequences=False))
model.add(tf.keras.layers.Dense(64, activation='leaky_relu'))
model.add(tf.keras.layers.Dense(1, activation='leaky_relu'))

from keras.optimizers import Adam
learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(optimizer = optimizer, loss = 'mse')

model.fit(train_val_generator, epochs=100)

train_pred=model.predict(train_val_generator)
train_pred=scaler.inverse_transform(train_pred)

train_sub = train_val[10:]
train_sub['RNN']=train_pred
train_sub.index = range(280)

plt.plot(train_sub.index, train_sub['RNN'], color='blue', label='RNN')
plt.plot(train_sub.index, train_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the training/validation dataset')

pred_CT=model.predict(test_generator)
pred_CT=scaler.inverse_transform(pred_CT)
total_pred_CT=[]
for i in range(len(pred_CT)):
    total_pred_CT.append(int(pred_CT[i]))
#print(total_pred_CT)

test_pred=model.predict(test_val_generator)
test_pred=scaler.inverse_transform(test_pred)
test_val_CT =[]
for i in range(len(test_pred)):
    test_val_CT.append(int(test_pred[i]))
#print(test_val_CT)
test_actual = test_val[10:]
test_actual['RNN']=test_val_CT
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
RNN_mse = mean_squared_error(test_actual['Sales'],test_actual['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_actual['Sales'],test_actual['RNN'])
RNN_r2 = r2_score(test_actual['Sales'],test_actual['RNN'])
print(f'test CT_mse: {RNN_mse}')
print(f'test CT_rmse: {RNN_rmse}')
print(f'test CT_mae: {RNN_mae}')
print(f'test CT_r2: {RNN_r2}')

test_sub = test[10:]
test_sub['RNN']=total_pred_CT
test_sub.index = range(351)

plt.plot(test_sub.index, test_sub['RNN'], color='blue', label='RNN')
plt.plot(test_sub.index, test_sub['Sales'], color='orange', label='Actual')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Sales')
plt.title('Actual vs predicted sales for the test dataset')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
RNN_mse = mean_squared_error(test_sub['Sales'],test_sub['RNN'])
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(test_sub['Sales'],test_sub['RNN'])
RNN_r2 = r2_score(test_sub['Sales'],test_sub['RNN'])
print(f'entire CT_mse: {RNN_mse}')
print(f'entire CT_rmse: {RNN_rmse}')
print(f'entire CT_mae: {RNN_mae}')
print(f'entire CT_r2: {RNN_r2}')

test_pred = test[10:]
test_pred = test_pred['Sales']
test_pred = test_pred.reset_index(drop=True)
y=[]
for i in range(len(test_pred)):
    y.append(total_pred_CT[i]-test_pred[i])
x=range(351)
# Plot the residual loss
plt.figure(figsize=(10, 6))
plt.plot(x,y)
plt.title('Residual Loss of RNN Model')
plt.xlabel('Day')
plt.ylabel('Residual Loss')
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[10:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df_CT= df2['Cape Town']
#print(df_CT)