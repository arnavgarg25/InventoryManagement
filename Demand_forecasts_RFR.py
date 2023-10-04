#IMPORT LIBRARIES
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import os

#BLOEM FORECASTING
data = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df_Bloemfontein = pd.DataFrame(data, columns=['Bloemfontein'])

#sales forecasting using random forest
df_Bloemfontein['Sale_1dayago']=df_Bloemfontein['Bloemfontein'].shift(+1)
df_Bloemfontein['Sale_2daysago']=df_Bloemfontein['Bloemfontein'].shift(+2)
df_Bloemfontein['Sale_3daysago']=df_Bloemfontein['Bloemfontein'].shift(+3)
df_Bloemfontein['Sale_4daysago']=df_Bloemfontein['Bloemfontein'].shift(+4)

df_Bloemfontein = df_Bloemfontein.dropna()

#preprocessing
x1,x2,x3,x4,y = (df_Bloemfontein['Sale_1dayago'],df_Bloemfontein['Sale_2daysago'],df_Bloemfontein['Sale_3daysago'],
                 df_Bloemfontein['Sale_4daysago'],df_Bloemfontein['Bloemfontein'])
x1,x2,x3,x4,y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(y)
x1,x2,x3,x4,y = x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1), x4.reshape(-1,1), y.reshape(-1,1)
final_x = np.concatenate((x1,x2,x3,x4),axis=1)

X_train, X_test, y_train, y_test = final_x[:-50], final_x[-50:], y[:-50], y[-50:]

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=300, max_features=1, max_depth=30, random_state=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
total_pred_bloem = model.predict(final_x)
#total_pred_bloem

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(total_pred_bloem, label ='Random_Forest_Predictions')
plt.plot(y, label = 'Actual Sales')
plt.legend(loc="upper left")
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[4:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df2
df_bloem= df2['Bloemfontein']
df_bloem

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
RNN_mse = mean_squared_error(y, total_pred_bloem)
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(y, total_pred_bloem)
RNN_r2 = r2_score(y, total_pred_bloem)
#print(f'Bloem_mse: {RNN_mse}')
#print(f'Bloem_rmse: {RNN_rmse}')
#print(f'Bloem_mae: {RNN_mae}')
#print(f'Bloem_r2: {RNN_r2}')

#JHB FORECASTING
data2 = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df_Johannesburg = pd.DataFrame(data2, columns=['Johannesburg'])

#sales forecasting using random forest
df_Johannesburg['Sale_1dayago']=df_Johannesburg['Johannesburg'].shift(+1)
df_Johannesburg['Sale_2daysago']=df_Johannesburg['Johannesburg'].shift(+2)
df_Johannesburg['Sale_3daysago']=df_Johannesburg['Johannesburg'].shift(+3)
df_Johannesburg['Sale_4daysago']=df_Johannesburg['Johannesburg'].shift(+4)

df_Johannesburg=df_Johannesburg.dropna()

#preprocessing
x1,x2,x3,x4,y = df_Johannesburg['Sale_1dayago'],df_Johannesburg['Sale_2daysago'],df_Johannesburg['Sale_3daysago'],df_Johannesburg['Sale_4daysago'],df_Johannesburg['Johannesburg']
x1,x2,x3,x4,y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(y)
x1,x2,x3,x4,y = x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1), x4.reshape(-1,1), y.reshape(-1,1)
final_x = np.concatenate((x1,x2,x3,x4),axis=1)

X_train, X_test, y_train, y_test = final_x[:-50], final_x[-50:], y[:-50], y[-50:]

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=50, max_features=1, max_depth=20, random_state=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
total_pred_jhb = model.predict(final_x)
total_pred_jhb

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(total_pred_jhb, label ='Random_Forest_Predictions')
plt.plot(y, label = 'Actual Sales')
plt.legend(loc="upper left")
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[4:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df2
df_jhb= df2['Johannesburg']
df_jhb

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
RNN_mse = mean_squared_error(y,total_pred_jhb)
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(y,total_pred_jhb)
RNN_r2 = r2_score(y,total_pred_jhb)
#print(f'JHB_mse: {RNN_mse}')
#print(f'JHB_rmse: {RNN_rmse}')
#print(f'JHB_mae: {RNN_mae}')
#print(f'JHB_r2: {RNN_r2}')

#DURB FORECASTING
data3 = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df_Durban = pd.DataFrame(data3, columns=['Durban'])

#sales forecasting using random forest
df_Durban['Sale_1dayago']=df_Durban['Durban'].shift(+1)
df_Durban['Sale_2daysago']=df_Durban['Durban'].shift(+2)
df_Durban['Sale_3daysago']=df_Durban['Durban'].shift(+3)
df_Durban['Sale_4daysago']=df_Durban['Durban'].shift(+4)

df_Durban=df_Durban.dropna()

#preprocessing
x1,x2,x3,x4,y = df_Durban['Sale_1dayago'],df_Durban['Sale_2daysago'],df_Durban['Sale_3daysago'],df_Durban['Sale_4daysago'],df_Durban['Durban']
x1,x2,x3,x4,y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(y)
x1,x2,x3,x4,y = x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1), x4.reshape(-1,1), y.reshape(-1,1)
final_x = np.concatenate((x1,x2,x3,x4),axis=1)

X_train, X_test, y_train, y_test = final_x[:-50], final_x[-50:], y[:-50], y[-50:]

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=50, max_features=1, max_depth=10, random_state=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
total_pred_durb = model.predict(final_x)
total_pred_durb

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(total_pred_durb, label ='Random_Forest_Predictions')
plt.plot(y, label = 'Actual Sales')
plt.legend(loc="upper left")
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[4:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df2
df_durb= df2['Durban']
df_durb

#PE FORECASTING
data4 = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df_PortElizabeth = pd.DataFrame(data4, columns=['Port Elizabeth'])

#sales forecasting using random forest
df_PortElizabeth['Sale_1dayago']=df_PortElizabeth['Port Elizabeth'].shift(+1)
df_PortElizabeth['Sale_2daysago']=df_PortElizabeth['Port Elizabeth'].shift(+2)
df_PortElizabeth['Sale_3daysago']=df_PortElizabeth['Port Elizabeth'].shift(+3)
df_PortElizabeth['Sale_4daysago']=df_PortElizabeth['Port Elizabeth'].shift(+4)

df_PortElizabeth=df_PortElizabeth.dropna()

#preprocessing
x1,x2,x3,x4,y = df_PortElizabeth['Sale_1dayago'],df_PortElizabeth['Sale_2daysago'],df_PortElizabeth['Sale_3daysago'],df_PortElizabeth['Sale_4daysago'],df_PortElizabeth['Port Elizabeth']
x1,x2,x3,x4,y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(y)
x1,x2,x3,x4,y = x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1), x4.reshape(-1,1), y.reshape(-1,1)
final_x = np.concatenate((x1,x2,x3,x4),axis=1)

X_train, X_test, y_train, y_test = final_x[:-50], final_x[-50:], y[:-50], y[-50:]

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=100, max_features=1, max_depth=30, random_state=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
total_pred_PE = model.predict(final_x)
total_pred_EL = total_pred_PE

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(total_pred_PE, label ='Random_Forest_Predictions')
plt.plot(y, label = 'Actual Sales')
plt.legend(loc="upper left")
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[4:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df2
df_PE= df2['Port Elizabeth']
df_EL = df_PE

#CT FORECASTING
data5 = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df_CapeTown = pd.DataFrame(data5, columns=['Cape Town'])

#sales forecasting using random forest
df_CapeTown['Sale_1dayago']=df_CapeTown['Cape Town'].shift(+1)
df_CapeTown['Sale_2daysago']=df_CapeTown['Cape Town'].shift(+2)
df_CapeTown['Sale_3daysago']=df_CapeTown['Cape Town'].shift(+3)
df_CapeTown['Sale_4daysago']=df_CapeTown['Cape Town'].shift(+4)

df_CapeTown=df_CapeTown.dropna()

#preprocessing
x1,x2,x3,x4,y = df_CapeTown['Sale_1dayago'],df_CapeTown['Sale_2daysago'],df_CapeTown['Sale_3daysago'],df_CapeTown['Sale_4daysago'],df_CapeTown['Cape Town']
x1,x2,x3,x4,y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(y)
x1,x2,x3,x4,y = x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1), x4.reshape(-1,1), y.reshape(-1,1)
final_x = np.concatenate((x1,x2,x3,x4),axis=1)

X_train, X_test, y_train, y_test = final_x[:-50], final_x[-50:], y[:-50], y[-50:]

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=50, max_features=1, max_depth=30, random_state=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
total_pred_CT = model.predict(final_x)
total_pred_CT

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(total_pred_CT, label ='Random_Forest_Predictions')
plt.plot(y, label = 'Actual Sales')
plt.legend(loc="upper left")
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[4:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df2
df_CT= df2['Cape Town']
df_CT

#Pret FORECASTING
data6 = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand',index_col='Date', parse_dates=True)
df_Pretoria = pd.DataFrame(data6, columns=['Pretoria'])

#sales forecasting using random forest
df_Pretoria['Sale_1dayago']=df_Pretoria['Pretoria'].shift(+1)
df_Pretoria['Sale_2daysago']=df_Pretoria['Pretoria'].shift(+2)
df_Pretoria['Sale_3daysago']=df_Pretoria['Pretoria'].shift(+3)
df_Pretoria['Sale_4daysago']=df_Pretoria['Pretoria'].shift(+4)

df_Pretoria=df_Pretoria.dropna()

#preprocessing
x1,x2,x3,x4,y = df_Pretoria['Sale_1dayago'],df_Pretoria['Sale_2daysago'],df_Pretoria['Sale_3daysago'],df_Pretoria['Sale_4daysago'],df_Pretoria['Pretoria']
x1,x2,x3,x4,y = np.array(x1), np.array(x2), np.array(x3), np.array(x4), np.array(y)
x1,x2,x3,x4,y = x1.reshape(-1,1), x2.reshape(-1,1), x3.reshape(-1,1), x4.reshape(-1,1), y.reshape(-1,1)
final_x = np.concatenate((x1,x2,x3,x4),axis=1)

X_train, X_test, y_train, y_test = final_x[:-50], final_x[-50:], y[:-50], y[-50:]

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=50, max_features=5, max_depth=10, random_state=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
total_pred_pret = model.predict(final_x)
total_pred_pret

plt.rcParams["figure.figsize"] = (12,8)
plt.plot(total_pred_pret, label ='Random_Forest_Predictions')
plt.plot(y, label = 'Actual Sales')
plt.legend(loc="upper left")
#plt.show()

df = pd.read_excel(r'C:\Users\ARNAV GARG\Desktop\MSc Eng\MECN7018A Research Project\Wits Student Project\arnav parameters updated.xlsx', sheet_name = 'Updated Demand')
df2 = df[4:]
df2['Index'] = range(len(df2))
df2.set_index("Index", inplace = True)
df2
df_pret= df2['Pretoria']
df_pret

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
RNN_mse = mean_squared_error(y, total_pred_pret)
RNN_rmse = math.sqrt(RNN_mse)
RNN_mae = mean_absolute_error(y, total_pred_pret)
RNN_r2 = r2_score(y, total_pred_pret)
#print(f'pret_mse: {RNN_mse}')
#print(f'pret_rmse: {RNN_rmse}')
#print(f'pret_mae: {RNN_mae}')
#print(f'pret_r2: {RNN_r2}')
