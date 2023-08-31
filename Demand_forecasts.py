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
model=RandomForestRegressor(n_estimators=100, max_features=3, random_state=1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
total_pred_bloem = model.predict(final_x)
total_pred_bloem

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
df_Bloem= df2['Bloemfontein']
df_Bloem

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
model=RandomForestRegressor(n_estimators=100, max_features=3, random_state=1)
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
print('F')