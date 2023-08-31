import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import tensorflow as tf
import os
import time
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
start_time = time.time()

# Set the random seed for the entire environment
#from stable_baselines3.common.utils import set_random_seed
#seed = 123
#set_random_seed(seed)

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
df_Bloem= df2['Bloemfontein']
#df_Bloem

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
#total_pred_jhb

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
#df_jhb

#Input parameters
initial_bloem_units = 1000 #starting stock
#durb_units = 1000
#EL_units = 1000
#pret_units = 1000
#CT_units = 1000

min_production_limit = 1000 #units   #reset to 30000
max_production_limit = 150000 #units

prod_storage_capacity = 3175200 #units
ware_storage_capacity = 14175000
bloem_storage_capacity = 1268400
#durb_storage_capacity = 630000
#EL_storage_capacity = 987000
#pret_storage_capacity = 3990000
#CT_storage_capacity = 745500

manufacture_cost = 25000/70/30 #per unit
total_manufacture_cost = 0

production_processing_time = 2 #maturation period in days

production_order_backlog = []
warehouse_order_backlog = []

produce_vector = []
initial_prod_units = 5000 #starting stock
initial_ware_units = 1000 #starting stock

small_truck_capacity = 14000
large_truck_capacity = 58800

transport_cost_prod_ware = 5031.942839
transport_cost_prodware_Bloem = 16116.92355
#transport_cost_prodware_Durb = 14042.60801
#transport_cost_prodware_EL = 27968,98425
#transport_cost_prod_Pret = 5031.942839
#transport_cost_ware_Pret = 8739.340276
#transport_cost_prodware_CT = 32701,39211
total_delivery_cost = 0

transport_time_prod_ware = 1
transport_time_prodware_Bloem = 2
#transport_time_prodware_Durb = 2
#transport_time_prodware_EL = 3
#transport_time_prodware_Pret = 1
#transport_time_prodware_CT = 3

prod_storage_cost = 224.29/7/70/30 #per unit per day
ware_storage_cost = 54.79/7/70/30 #per unit per day
bloem_storage_cost = 65.748/7/70/30 #per unit per day
#durb_storage_cost = 65.748/7/70/30 #per unit per day
#EL_storage_cost = 65.748/7/70/30 #per unit per day
#pret_storage_cost = 65.748/7/70/30 #per unit per day
#CT_storage_cost = 65.748/7/70/30 #per unit per day
total_storage_cost = 0

distribution_percent_bloem = 82
#distribution_percent_durb = 95
#distribution_percent_jhb = 0
#distribution_percent_EL = 93
#distribution_percent_pret = 54
#distribution_percent_CT = 25

units_satisfied = 0
units_unsatisfied = 0
selling_price = 25
revenue_gained = 0

units_moving_prodware_bloem_vector = []

import gym
from gym import Env
from gym import spaces
from gym.spaces import Box, Discrete, MultiDiscrete
import numpy as np
import random

class InventoryEnvironment(Env):
    def __init__(self, initial_bloem_units, initial_ware_units, initial_prod_units, bloem_storage_capacity,
                 ware_storage_capacity, prod_storage_capacity, large_truck_capacity, small_truck_capacity, df_Bloem,
                 df_jhb, selling_price, bloem_storage_cost, ware_storage_cost, prod_storage_cost, manufacture_cost,
                 distribution_percent_bloem, production_processing_time, transport_time_prod_ware,
                 transport_time_prodware_Bloem, transport_cost_prod_ware, transport_cost_prodware_Bloem,
                 min_production_limit, max_production_limit):

        self.initial_bloem_units = initial_bloem_units
        self.initial_ware_units = initial_ware_units
        self.initial_prod_units = initial_prod_units
        self.bloem_storage_capacity = bloem_storage_capacity
        self.ware_storage_capacity = ware_storage_capacity
        self.prod_storage_capacity = prod_storage_capacity
        self.small_truck_capacity = small_truck_capacity
        self.large_truck_capacity = large_truck_capacity
        self.demand_bloem = df_Bloem
        self.demand_ware = df_jhb
        self.selling_price = selling_price
        self.bloem_storage_cost = bloem_storage_cost
        self.ware_storage_cost = ware_storage_cost
        self.prod_storage_cost = prod_storage_cost
        self.manufacture_cost = manufacture_cost
        self.distribution_percent_bloem = distribution_percent_bloem
        self.production_processing_time = production_processing_time
        self.transport_time_prod_ware = transport_time_prod_ware
        self.transport_time_prodware_Bloem = transport_time_prodware_Bloem
        self.transport_cost_prod_ware = transport_cost_prod_ware
        self.transport_cost_prodware_Bloem = transport_cost_prodware_Bloem
        self.min_production_limit = min_production_limit
        self.max_production_limit = max_production_limit

        # define action space (continuous, ordering quantity)
        self.num_stock_points = 3  # initially just considering bloem, production, warehouse
        #self.action_space = Box(low=np.array([0, 0, 0]), high=np.array([1, 1, 1]), shape=(self.num_stock_points,),dtype=np.float32)  # self.action_space = Box(low=np.array([self.min_production_limit,0,0,0,0,0,0]), high=np.array([self.max_production_limit,self.warehouse_storage_capacity,self.bloem_storage_capacity,self.durb_storage_cost,self.EL_storage_cost,self.pret_storage_capacity,self.CT_storage_capacity]), shape=(num_stock_points,))
        self.action_space = Box(low=0, high=1, shape=(self.num_stock_points,), dtype=np.float32)
        #self.action_space = MultiDiscrete([10,10,10])

        # define observation space
        self.num_obs_points = 30
        self.observation_space = Box(low=0, high=100000000, shape=(self.num_obs_points,),dtype=np.float32)  # self.observation_space = Box(low=np.array([0]), high=np.array([20000]), shape=(self.num_stock_points,), dtype=np.float32) [bloem/prod/ware_units, units_satisfied/unsatisfied, revenue_gained, total_delivery/manufacture/storage_cost, step_count]

        # set starting inventory
        self.bloem_units = initial_bloem_units  # state
        self.ware_units = initial_ware_units  # state
        self.prod_units = initial_prod_units  # state
        # set days length
        self.days_length = 342
        # current day
        self.day = 5
        # set initial performance
        self.units_satisfied = 0
        self.units_unsatisfied = 0
        self.fill_rate = 0
        self.revenue_gained = 0
        self.total_storage_cost = 0
        self.total_manufacture_cost = 0
        self.total_delivery_cost = 0
        self.net_profit = 0

        self.production_order_backlog = []
        self.warehouse_order_backlog = []
        self.units_moving_prodware_bloem_vector = []
        self.produce_vector = []
        self.units_moving_prod_ware_vector = []

        self.no_of_trucks_prod_ware = 0
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_ware_bloem = 0

        self.prod_action = 0
        self.ware_action = 0
        self.bloem_action = 0

        self.current_cost = 0
        self.current_revenue = 0

        self.units_transit_prod_ware = 0
        self.units_transit_prod_bloem = 0
        self.units_transit_ware_bloem = 0

    def reset(self):
        # Reset starting inventory
        self.bloem_units = self.initial_bloem_units
        self.ware_units = self.initial_ware_units
        self.prod_units = self.initial_prod_units  # state
        # reset days length
        self.days_length = 342
        self.day = 5
        # reset initial performance
        self.units_satisfied = 0
        self.units_unsatisfied = 0
        self.fill_rate = 0
        self.revenue_gained = 0
        self.total_storage_cost = 0
        self.total_manufacture_cost = 0
        self.total_delivery_cost = 0
        self.net_profit = 0

        self.production_order_backlog = []
        self.warehouse_order_backlog = []
        self.units_moving_prodware_bloem_vector = []
        self.produce_vector = []
        self.units_moving_prod_ware_vector = []

        self.no_of_trucks_prod_ware = 0
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_ware_bloem = 0

        self.prod_action = 0
        self.ware_action = 0
        self.bloem_action = 0

        self.current_cost = 0
        self.current_revenue = 0

        self.units_transit_prod_ware = 0
        self.units_transit_prod_bloem = 0
        self.units_transit_ware_bloem = 0

        obs = [self.prod_units, self.ware_units, self.bloem_units, self.fill_rate,
               self.no_of_trucks_prod_ware, self.no_of_trucks_prod_bloem, self.no_of_trucks_ware_bloem,
               self.units_transit_prod_ware, self.units_transit_prod_bloem, self.units_transit_ware_bloem]
        bloem_forecast = self.demand_bloem[self.day-5:self.day + 5]
        obs.extend(bloem_forecast)
        ware_forecast = self.demand_ware[self.day-5:self.day + 5]
        obs.extend(ware_forecast)

        return obs

    def step(self, action):
        self.current_revenue = 0
        self.current_cost = 0

        prod, ware, bloem = action   #MULTIDISCRETE action space
        #self.prod_action = prod  # MULTIDISCRETE action space
        # 1) production producing quantity action
        self.prod_action = ((prod * (self.max_production_limit - self.min_production_limit)) + self.min_production_limit)
        if self.prod_action < self.min_production_limit:
            self.prod_action = self.min_production_limit
        elif self.prod_action > self.max_production_limit:
            self.prod_action = self.max_production_limit

        if self.prod_units + self.prod_action > self.prod_storage_capacity:
            self.prod_action = self.prod_storage_capacity - self.prod_units
            if self.prod_action < self.min_production_limit:
                self.prod_action = 0
            elif self.prod_action > self.max_production_limit:
                self.prod_action = self.max_production_limit

        self.total_manufacture_cost += self.prod_action * self.manufacture_cost
        self.current_cost += self.prod_action * self.manufacture_cost
        self.produce_vector.append(self.prod_action)

        # processing time before turning into finished inventory
        if self.day >= 5 + self.production_processing_time:
            self.prod_units += self.produce_vector[0]
            del self.produce_vector[0]

        # 2) Ware ordering quantity action
        #self.ware_action = ware    #MULTIDISCRETE action space
        self.ware_action = ware * 28000

        # send units from production to warehouse
        if self.ware_units + self.ware_action > self.ware_storage_capacity:
            self.ware_action = self.ware_storage_capacity - self.ware_units

        if self.ware_action <= self.prod_units:
            self.prod_units -= self.ware_action
        else:
            self.ware_action = self.prod_units
            self.prod_units=0

        self.units_moving_prod_ware_vector.append(self.ware_action)

        # delivery time before reaching ware
        if self.day >= 5 + self.transport_time_prod_ware:
            self.ware_units += self.units_moving_prod_ware_vector[0]
            self.units_transit_prod_ware = self.units_moving_prod_ware_vector[0]
            del self.units_moving_prod_ware_vector[0]

        self.no_of_trucks_prod_ware = math.ceil(self.ware_action / self.small_truck_capacity)
        self.total_delivery_cost += self.no_of_trucks_prod_ware * self.transport_cost_prod_ware
        self.current_cost += self.no_of_trucks_prod_ware * self.transport_cost_prod_ware

        # 3) Bloem ordering quantity action
        #self.bloem_action = bloem     #MULTIDISCRETE action space
        self.bloem_action = bloem * 588000

        # Determine whether bloem order is from prod or ware
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_ware_bloem = 0
        choice = random.choice(range(1, 101))

        # sending units from production to bloem
        if choice > self.distribution_percent_bloem:
            self.production_order_backlog.append(self.bloem_action)
            units_moving_prod_bloem = 0
            for n in range(len(self.production_order_backlog) - 1):
                if self.prod_units > self.production_order_backlog[n] and self.production_order_backlog[n] + self.bloem_units + units_moving_prod_bloem < self.bloem_storage_capacity:  # checking storage capacity of bloem outlet:
                    self.prod_units -= self.production_order_backlog[n]
                    units_moving_prod_bloem += self.production_order_backlog[n]
                    self.production_order_backlog[n] = 0
                else:  # (dont continue satisfying orders)  #production_order_backlog = [500 200]
                    break
            self.production_order_backlog = [i for i in self.production_order_backlog if i != 0]

            self.units_moving_prodware_bloem_vector.append(units_moving_prod_bloem)
            self.no_of_trucks_prod_bloem = math.ceil(units_moving_prod_bloem / self.large_truck_capacity)
            self.units_transit_prod_bloem = units_moving_prod_bloem
            self.total_delivery_cost += self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem
            self.current_cost += self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem

        # sending units from warehouse to bloem
        else:
            self.warehouse_order_backlog.append(self.bloem_action)
            units_moving_ware_bloem = 0
            for n in range(len(self.warehouse_order_backlog) - 1):
                if self.ware_units > self.warehouse_order_backlog[n] and self.warehouse_order_backlog[n] + self.bloem_units + units_moving_ware_bloem < self.bloem_storage_capacity:  # checking storage capacity of bloem outlet
                    self.ware_units -= self.warehouse_order_backlog[n]
                    units_moving_ware_bloem += self.warehouse_order_backlog[n]
                    self.warehouse_order_backlog[n] = 0
                else:
                    break
            self.warehouse_order_backlog = [i for i in self.warehouse_order_backlog if i != 0]

            self.units_moving_prodware_bloem_vector.append(units_moving_ware_bloem)
            self.no_of_trucks_ware_bloem = math.ceil(units_moving_ware_bloem / self.large_truck_capacity)
            self.units_transit_ware_bloem = units_moving_ware_bloem
            self.total_delivery_cost += self.no_of_trucks_ware_bloem * self.transport_cost_prodware_Bloem
            self.current_cost += self.no_of_trucks_ware_bloem * self.transport_cost_prodware_Bloem

        # delivery time before reaching bloem
        if self.day >= 5 + self.transport_time_prodware_Bloem:
            self.bloem_units += self.units_moving_prodware_bloem_vector[0]
            del self.units_moving_prodware_bloem_vector[0]

        # 4) Apply customer fulfilment at bloem
        if self.bloem_units > self.demand_bloem[self.day]:
            self.bloem_units -= self.demand_bloem[self.day]
            self.units_satisfied += self.demand_bloem[self.day]
            self.revenue_gained += self.demand_bloem[self.day] * self.selling_price
            self.current_revenue += self.demand_bloem[self.day] * self.selling_price
        else:
            self.units_satisfied += self.bloem_units
            self.units_unsatisfied += self.demand_bloem[self.day] - self.bloem_units
            self.revenue_gained += self.bloem_units * self.selling_price
            self.current_revenue += self.bloem_units * self.selling_price
            self.bloem_units = 0

        # 5) Apply customer fulfilment at ware
        if self.ware_units > self.demand_ware[self.day]:
            self.ware_units -= self.demand_ware[self.day]
            self.units_satisfied += self.demand_ware[self.day]
            self.revenue_gained += self.demand_ware[self.day] * self.selling_price
            self.current_revenue += self.demand_ware[self.day] * self.selling_price
        else:
            self.units_satisfied += self.ware_units
            self.units_unsatisfied += self.demand_ware[self.day] - self.ware_units
            self.revenue_gained += self.ware_units * self.selling_price
            self.current_revenue += self.ware_units * self.selling_price
            self.ware_units = 0

        # 6) net profit and fill rate
        self.fill_rate = (self.units_satisfied / (self.units_satisfied + self.units_unsatisfied)) * 100
        self.net_profit = (abs(self.revenue_gained) - self.total_storage_cost - self.total_manufacture_cost -
                           self.total_delivery_cost)

        # 7) storage costs for remaining inventory
        self.total_storage_cost += self.bloem_units * self.bloem_storage_cost
        self.total_storage_cost += self.ware_units * self.ware_storage_cost
        self.total_storage_cost += self.prod_units * self.prod_storage_cost
        self.current_cost += self.bloem_units * self.bloem_storage_cost + self.ware_units * self.ware_storage_cost + self.prod_units * self.prod_storage_cost

        # 8) calculate reward
        reward = self.current_revenue - self.current_cost  # (Profit-based reward)
        min_reward = -100000
        max_reward = 100000
        target_min = -10
        target_max = 10
        reward = (reward - min_reward) / (max_reward - min_reward) * (target_max - target_min) + target_min

        #if self.fill_rate > 90: #(service-based reward)
        #    reward = 9
        #elif self.fill_rate > 80:
        #    reward = 8
        #elif self.fill_rate > 70:
        #    reward = 7
        #elif self.fill_rate > 60:
        #    reward = 6
        #elif self.fill_rate > 50:
        #    reward = 5
        #elif self.fill_rate > 40:
        #    reward = -1
        #elif self.fill_rate > 30:
        #    reward = -2
        #elif self.fill_rate > 20:
        #    reward = -3
        #elif self.fill_rate > 10:
        #    reward = -4
        #else:
        #    reward = -5

        # check if days are complete
        if self.days_length <= 0:
            done = True
        else:
            done = False

        # increase current day
        self.day += 1
        # reduce number of days by 1
        self.days_length -= 1

        obs = [self.prod_units, self.ware_units, self.bloem_units, self.fill_rate,
               self.no_of_trucks_prod_ware, self.no_of_trucks_prod_bloem, self.no_of_trucks_ware_bloem,
               self.units_transit_prod_ware, self.units_transit_prod_bloem, self.units_transit_ware_bloem]
        bloem_forecast = self.demand_bloem[self.day-5:self.day + 5]
        obs.extend(bloem_forecast)
        ware_forecast = self.demand_ware[self.day-5:self.day + 5]
        obs.extend(ware_forecast)

        # placeholder for additional info
        info = {}

        # return step information
        return obs, reward, done, info

    def render(self):
        # Implement viz
        pass

class MeticLogger(BaseCallback):
    def __init__(self, verbose=0, log_freq=1000):
        super(MeticLogger, self).__init__(verbose)
        self.log_freq = log_freq
        self.logger = SummaryWriter()

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            # add metrics that are on a timestep basis
            stats = {'financials/net_profit': self.training_env.get_attr("net_profit")[0],
                     'financials/fill_rate': self.training_env.get_attr("fill_rate")[0],
                     'financials/prod_units': self.training_env.get_attr("prod_units")[0],
                     'financials/ware_units': self.training_env.get_attr("ware_units")[0],
                     'financials/bloem_units': self.training_env.get_attr("bloem_units")[0],
                     'financials/prod_action': self.training_env.get_attr("prod_action")[0],
                     'financials/ware_action': self.training_env.get_attr("ware_action")[0],
                     'financials/bloem_action': self.training_env.get_attr("bloem_action")[0],
                    }
            for key in stats.keys():
                self.logger.record(key, stats[key])

env = InventoryEnvironment(initial_bloem_units=initial_bloem_units, initial_ware_units=initial_ware_units, initial_prod_units=initial_prod_units, bloem_storage_capacity=bloem_storage_capacity, ware_storage_capacity=ware_storage_capacity, prod_storage_capacity=prod_storage_capacity, large_truck_capacity=large_truck_capacity, small_truck_capacity=small_truck_capacity, df_Bloem=df_Bloem, df_jhb=df_jhb, selling_price=selling_price, bloem_storage_cost=bloem_storage_cost, ware_storage_cost=ware_storage_cost, prod_storage_cost=prod_storage_cost, manufacture_cost=manufacture_cost, distribution_percent_bloem=distribution_percent_bloem, production_processing_time=production_processing_time, transport_time_prod_ware=transport_time_prod_ware, transport_time_prodware_Bloem=transport_time_prodware_Bloem, transport_cost_prod_ware=transport_cost_prod_ware, transport_cost_prodware_Bloem=transport_cost_prodware_Bloem, min_production_limit=min_production_limit, max_production_limit=max_production_limit)

import shutil
import os

log_dir = "logs"  # Specify the path to your log directory
models_dir = "models/PPO"  # Specify the path to your log directory

# Check if the log directory exists
if os.path.exists(log_dir):
    # Delete the contents of the log directory
    shutil.rmtree(log_dir)
    print("Logs deleted successfully.")
else:
    print("Log directory does not exist.")
# Check if the model directory exists
if os.path.exists(models_dir):
    # Delete the contents of the log directory
    shutil.rmtree(models_dir)
    print("models deleted successfully.")
else:
    print("model directory does not exist.")

models_dir = "models/PPO"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

from stable_baselines3 import PPO, DDPG
ent_coef = 0.1
learning_rate=0.003
log_freq = 100

model = PPO('MlpPolicy',env,verbose=1, tensorboard_log = logdir, ent_coef=ent_coef, learning_rate=learning_rate)

TIMESTEPS = 1000000
model.learn(total_timesteps=TIMESTEPS, callback=[MeticLogger(log_freq=log_freq)])
model.save("ppo_IM")
del model
model=PPO.load("ppo_IM")

#for i in range(1,10):
#    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=[MeticLogger(log_freq=log_freq)])
#    model.save(f"{models_dir}/{TIMESTEPS*i}")

# Create a summary writer
#logdir2 = 'logs2'

summary_writer = tf.summary.create_file_writer(logdir)

# Write the summary data for the line graphs
with summary_writer.as_default():

    episodes = 1
    total_reward_vector = []
    for episode in range(episodes):
        # print("Episode: {}".format(episode+1))
        obs = env.reset()
        done = False
        total_reward = 0
        i = 1

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            print("Day {}".format(i))
            print("Action: ", action)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            print('obs=', obs, 'reward=', reward, 'done=', done)
            i += 1

            tf.summary.scalar('Production/Production units available', env.prod_units, step=i)
            tf.summary.scalar('Production/Action', env.prod_action, step=i)

            tf.summary.scalar('Ware/Warehouse current demand', df_jhb[i], step=i)
            tf.summary.scalar('Ware/Action', env.ware_action, step=i)
            tf.summary.scalar('Ware/Warehouse units available', env.ware_units, step=i)

            tf.summary.scalar('Bloem/Bloem current demand', df_Bloem[i], step=i)
            tf.summary.scalar('Bloem/Action', env.bloem_action, step=i)
            tf.summary.scalar('Bloem/Bloem units available', env.bloem_units, step=i)

            tf.summary.scalar('Trucks/Number of trucks in operation prod to ware', env.no_of_trucks_prod_ware, step=i)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to bloem', env.no_of_trucks_prod_bloem, step=i)
            tf.summary.scalar('Trucks/Number of trucks in operation ware to bloem', env.no_of_trucks_ware_bloem, step=i)

            tf.summary.scalar('Cost/Total manufacturing cost', env.total_manufacture_cost, step=i)
            tf.summary.scalar('Cost/Total delivery cost', env.total_delivery_cost, step=i)
            tf.summary.scalar('Cost/Total storage cost', env.total_storage_cost, step=i)
            tf.summary.scalar('Cost/Overall cost', env.total_delivery_cost+env.total_manufacture_cost+env.total_storage_cost,step=i)

            tf.summary.scalar('Profitability/Revenue', env.revenue_gained, step=i)
            tf.summary.scalar('Profitability/Total cost', env.total_delivery_cost+env.total_manufacture_cost+env.total_storage_cost, step=i)
            tf.summary.scalar('Profitability/Net profit', env.net_profit, step=i)

            tf.summary.scalar('Units/Units satisfied', env.units_satisfied, step=i)
            tf.summary.scalar('Units/Units unsatisfied', env.units_unsatisfied, step=i)
            tf.summary.scalar('Order fulfilment rate', env.fill_rate, step=i)

        print('Net profit =', env.net_profit)
        print('Fill rate =', env.fill_rate)
        print('Episode:{} Reward:{}'.format(episode + 1, total_reward))
        total_reward_vector.append(total_reward)
        obs = env.reset()

    plt.plot(range(1, episodes + 1), total_reward_vector)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

#TRY USING ALTERNATIVE REWARD