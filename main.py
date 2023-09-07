import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import tensorflow as tf
import os
import shutil
import time
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from gym import Env
from gym.spaces import Box, Discrete, MultiDiscrete
from stable_baselines3 import PPO, DDPG

start_time = time.time()

# Set the random seed for the entire environment
from stable_baselines3.common.utils import set_random_seed
seed = 123
set_random_seed(seed)

# Import demand forecasts
from Demand_forecasts import df_Bloem, df_jhb, total_pred_bloem, total_pred_jhb

# Import input parameters
from Input_parameters import (initial_prod_units, initial_bloem_units, initial_ware_units, manufacture_cost,
                              production_processing_time, min_production_limit, max_production_limit,
                              prod_storage_capacity, ware_storage_capacity, bloem_storage_capacity,
                              small_truck_capacity, large_truck_capacity, transport_cost_prod_ware,
                              transport_cost_prodware_Bloem, transport_time_prod_ware, transport_time_prodware_Bloem,
                              prod_storage_cost, ware_storage_cost, bloem_storage_cost, distribution_percent_bloem,
                              selling_price)


class InventoryEnvironment(Env):
    def __init__(self, initial_bloem_units, initial_ware_units, initial_prod_units, bloem_storage_capacity,
                 ware_storage_capacity, prod_storage_capacity, large_truck_capacity, small_truck_capacity, df_Bloem,
                 df_jhb, total_pred_bloem, total_pred_jhb, selling_price, bloem_storage_cost, ware_storage_cost,
                 prod_storage_cost, manufacture_cost, distribution_percent_bloem, production_processing_time,
                 transport_time_prod_ware, transport_time_prodware_Bloem, transport_cost_prod_ware,
                 transport_cost_prodware_Bloem, min_production_limit, max_production_limit):

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
        self.total_pred_bloem = total_pred_bloem
        self.total_pred_jhb = total_pred_jhb
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
        self.action_space = Box(low=0, high=1, shape=(self.num_stock_points,), dtype=np.float32)
        # self.action_space = MultiDiscrete([10,10,10])

        # define observation space
        self.num_obs_points = 30
        self.observation_space = Box(low=0, high=10000000, shape=(self.num_obs_points,), dtype=np.float32)

        # set starting inventory
        self.bloem_units = initial_bloem_units  # state
        self.ware_units = initial_ware_units  # state
        self.prod_units = initial_prod_units  # state
        # set days length
        self.days_length = 346
        # current day
        self.day = 5
        # set initial performance
        self.units_satisfied = 0
        self.units_unsatisfied = 0
        self.fill_rate = 0
        self.revenue_gained = 0
        self.net_profit = 0
        self.total_storage_cost = 0
        self.total_manufacture_cost = 0
        self.total_delivery_cost = 0

        self.production_order_backlog = []
        self.warehouse_order_backlog = []
        self.units_moving_prodware_bloem_vector = []
        self.produce_vector = []
        self.units_moving_prod_ware_vector = []

        # Part of observation space
        self.no_of_trucks_prod_ware = 0
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_ware_bloem = 0
        self.units_transit_prod_ware = 0
        self.units_transit_prod_bloem = 0
        self.units_transit_ware_bloem = 0
        self.choice = 0

    def reset(self, **kwargs):
        # Reset starting inventory
        self.bloem_units = self.initial_bloem_units
        self.ware_units = self.initial_ware_units
        self.prod_units = self.initial_prod_units
        # reset days length
        self.days_length = 346
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

        # Part of observation space
        self.no_of_trucks_prod_ware = 0
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_ware_bloem = 0
        self.units_transit_prod_ware = 0
        self.units_transit_prod_bloem = 0
        self.units_transit_ware_bloem = 0
        self.choice = 0

        obs = [self.prod_units, self.ware_units, self.bloem_units, self.choice,
               self.no_of_trucks_prod_ware, self.no_of_trucks_prod_bloem, self.no_of_trucks_ware_bloem,
               self.units_transit_prod_ware, self.units_transit_prod_bloem, self.units_transit_ware_bloem]
        bloem_forecast = self.total_pred_bloem[self.day-5:self.day + 5]
        obs.extend(bloem_forecast)
        ware_forecast = self.total_pred_jhb[self.day-5:self.day + 5]
        obs.extend(ware_forecast)
        return obs

    def step(self, action):
        self.current_revenue = 0
        self.current_cost = 0
        self.current_units_sold = 0
        self.current_units_available = 0
        reward = 0

        prod, ware, bloem = action
        # 1) production producing quantity action
        self.prod_action = ((prod * (self.max_production_limit-self.min_production_limit)) + self.min_production_limit)
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
        self.ware_action = ware * 28000

        # send units from production to warehouse
        if self.ware_units + self.ware_action > self.ware_storage_capacity:
            self.ware_action = self.ware_storage_capacity - self.ware_units

        if self.ware_action <= self.prod_units:
            self.prod_units -= self.ware_action
        else:
            self.ware_action = self.prod_units
            self.prod_units = 0

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
        self.bloem_action = bloem * 58800

        # Determine whether bloem order is from prod or ware
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_ware_bloem = 0
        self.units_moving_ware_bloem = 0
        self.units_moving_prod_bloem = 0
        self.choice = random.choice(range(1, 101))

        # sending units from production to bloem
        if self.choice > self.distribution_percent_bloem:
            self.choice = 1
            self.production_order_backlog.append(self.bloem_action)
            for n in range(len(self.production_order_backlog)-1):
                if (self.prod_units > self.production_order_backlog[n] and self.production_order_backlog[n] +
                        self.bloem_units + self.units_moving_prod_bloem < self.bloem_storage_capacity):
                    self.prod_units -= self.production_order_backlog[n]
                    self.units_moving_prod_bloem += self.production_order_backlog[n]
                    self.production_order_backlog[n] = 0
                else:  # (don't continue satisfying orders)  #production_order_backlog = [500 200]
                    break
            self.production_order_backlog = [i for i in self.production_order_backlog if i != 0]

            self.units_moving_prodware_bloem_vector.append(self.units_moving_prod_bloem)
            self.no_of_trucks_prod_bloem = math.ceil(self.units_moving_prod_bloem / self.large_truck_capacity)
            self.units_transit_prod_bloem = self.units_moving_prod_bloem
            self.total_delivery_cost += self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem
            self.current_cost += self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem

        # sending units from warehouse to bloem
        else:
            self.choice = 0
            self.warehouse_order_backlog.append(self.bloem_action)
            for n in range(len(self.warehouse_order_backlog)-1):
                if (self.ware_units > self.warehouse_order_backlog[n] and self.warehouse_order_backlog[n] +
                        self.bloem_units + self.units_moving_ware_bloem < self.bloem_storage_capacity):
                    self.ware_units -= self.warehouse_order_backlog[n]
                    self.units_moving_ware_bloem += self.warehouse_order_backlog[n]
                    self.warehouse_order_backlog[n] = 0
                else:
                    break
            self.warehouse_order_backlog = [i for i in self.warehouse_order_backlog if i != 0]

            self.units_moving_prodware_bloem_vector.append(self.units_moving_ware_bloem)
            self.no_of_trucks_ware_bloem = math.ceil(self.units_moving_ware_bloem / self.large_truck_capacity)
            self.units_transit_ware_bloem = self.units_moving_ware_bloem
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
            self.current_units_sold += self.demand_bloem[self.day]
            self.revenue_gained += self.demand_bloem[self.day] * self.selling_price
            self.current_revenue += self.demand_bloem[self.day] * self.selling_price
        else:
            self.units_satisfied += self.bloem_units
            self.current_units_sold += self.bloem_units
            self.units_unsatisfied += self.demand_bloem[self.day] - self.bloem_units
            self.revenue_gained += self.bloem_units * self.selling_price
            self.current_revenue += self.bloem_units * self.selling_price
            self.bloem_units = 0

        # 5) Apply customer fulfilment at ware
        if self.ware_units > self.demand_ware[self.day]:
            self.ware_units -= self.demand_ware[self.day]
            self.units_satisfied += self.demand_ware[self.day]
            self.current_units_sold += self.demand_ware[self.day]
            self.revenue_gained += self.demand_ware[self.day] * self.selling_price
            self.current_revenue += self.demand_ware[self.day] * self.selling_price
        else:
            self.units_satisfied += self.ware_units
            self.current_units_sold += self.ware_units
            self.units_unsatisfied += self.demand_ware[self.day] - self.ware_units
            self.revenue_gained += self.ware_units * self.selling_price
            self.current_revenue += self.ware_units * self.selling_price
            self.ware_units = 0

        # 6) net profit and fill rate
        self.fill_rate = (self.units_satisfied / (self.units_satisfied + self.units_unsatisfied)) * 100
        self.net_profit = (self.revenue_gained - self.total_storage_cost - self.total_manufacture_cost -
                           self.total_delivery_cost)

        # 7) storage costs for remaining inventory
        self.total_storage_cost += self.bloem_units * self.bloem_storage_cost
        self.total_storage_cost += self.ware_units * self.ware_storage_cost
        self.total_storage_cost += self.prod_units * self.prod_storage_cost
        self.current_cost += self.bloem_units * self.bloem_storage_cost
        self.current_cost += self.ware_units * self.ware_storage_cost
        self.current_cost += self.prod_units * self.prod_storage_cost

        # 8) calculate reward
        reward += self.current_revenue - self.current_cost  # 1(Profit-based reward)

        if self.fill_rate > 90:  # 2(service-based reward)
            reward += 90
        elif self.fill_rate > 80:
            reward += 80
        elif self.fill_rate > 70:
            reward += 70
        elif self.fill_rate > 60:
            reward += 60
        elif self.fill_rate > 50:
            reward += 50
        elif self.fill_rate > 40:
            reward += -10
        elif self.fill_rate > 30:
            reward += -20
        elif self.fill_rate > 20:
            reward += -30
        elif self.fill_rate > 10:
            reward += -40
        else:
            reward += -50

        # Normalize reward
        min_reward = -100000
        max_reward = 100000
        target_min = -10
        target_max = 10
        reward = (reward - min_reward) / (max_reward - min_reward) * (target_max - target_min) + target_min

        # 3(Maximizing units sold while minimizing units available)
        #reward += self.current_units_sold - (self.bloem_units + self.ware_units + self.prod_units)

        # check if days are complete
        if self.days_length <= 0:
            done = True
        else:
            done = False

        # increase current day
        self.day += 1
        # reduce number of days by 1
        self.days_length -= 1

        # Normalize prod units
        min_produnits = 0
        max_produnits = 1000000  # self.prod_storage_capacity
        target_min = 0
        target_max = 100
        prev_prod = self.prod_units
        self.prod_units = ((self.prod_units-min_produnits)/(max_produnits-min_produnits)*(target_max-target_min)
                           + target_min)
        # Normalize ware units
        min_wareunits = 0
        max_wareunits = 1000000  # self.ware_storage_capacity
        target_min = 0
        target_max = 100
        prev_ware = self.ware_units
        self.ware_units = ((self.ware_units-min_wareunits)/(max_wareunits-min_wareunits)*(target_max-target_min)
                           + target_min)
        # Normalize bloem units
        min_bloemunits = 0
        max_bloemunits = 1000000  # self.bloem_storage_capacity
        target_min = 0
        target_max = 100
        prev_bloem = self.bloem_units
        self.bloem_units = ((self.bloem_units-min_bloemunits)/(max_bloemunits-min_bloemunits)*(target_max - target_min)
                            + target_min)

        obs = [self.prod_units, self.ware_units, self.bloem_units, self.choice,
               self.no_of_trucks_prod_ware, self.no_of_trucks_prod_bloem, self.no_of_trucks_ware_bloem,
               self.units_transit_prod_ware, self.units_transit_prod_bloem, self.units_transit_ware_bloem]
        bloem_forecast = self.total_pred_bloem[self.day-5:self.day + 5]
        obs.extend(bloem_forecast)
        ware_forecast = self.total_pred_jhb[self.day-5:self.day + 5]
        obs.extend(ware_forecast)

        self.prod_units = prev_prod
        self.ware_units = prev_ware
        self.bloem_units = prev_bloem

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
            stats = {'Training/net_profit': self.training_env.get_attr("net_profit")[0],
                     'Training/fill_rate': self.training_env.get_attr("fill_rate")[0],
                     'Training/prod_units': self.training_env.get_attr("prod_units")[0],
                     'Training/ware_units': self.training_env.get_attr("ware_units")[0],
                     'Training/bloem_units': self.training_env.get_attr("bloem_units")[0],
                     'Training/prod_action': self.training_env.get_attr("prod_action")[0],
                     'Training/ware_action': self.training_env.get_attr("ware_action")[0],
                     'Training/bloem_action': self.training_env.get_attr("bloem_action")[0], }
            for key in stats.keys():
                self.logger.record(key, stats[key])
'''
# PPO HYPERPARAMETER TUNING START
param_grid = {
    'ent_coef': [0, 0.005, 0.01, 0.05, 0.1, 0.5],
    'learning_rate': [0.0001, 0.001, 0.003, 0.005, 0.01, 0.1],
    'clip_range': [0.1, 0.2, 0.3],
    'n_steps': [32, 128, 256, 512],
    'batch_size': [32, 64, 128, 4096],
    'n_epochs': [3, 5, 8],
    'gamma': [0.8, 0.9, 0.95, 0.99],
    'gae_lambda': [0.9, 0.95, 1],
    'vf_coef': [0.25, 0.5, 0.75], }

def evaluate_agent(agent, env, num_episodes=10):
    episode_rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            action, _ = agent.predict(obs)  # Get the action from the policy
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        episode_rewards.append(episode_reward)
    mean_reward = sum(episode_rewards) / num_episodes
    return mean_reward

def train_ppo(params):
    env = InventoryEnvironment(initial_bloem_units=initial_bloem_units, initial_ware_units=initial_ware_units,
                               initial_prod_units=initial_prod_units, bloem_storage_capacity=bloem_storage_capacity,
                               ware_storage_capacity=ware_storage_capacity, prod_storage_capacity=prod_storage_capacity,
                               large_truck_capacity=large_truck_capacity, small_truck_capacity=small_truck_capacity,
                               df_Bloem=df_Bloem, df_jhb=df_jhb, total_pred_bloem=total_pred_bloem,
                               total_pred_jhb=total_pred_jhb, selling_price=selling_price,
                               bloem_storage_cost=bloem_storage_cost, ware_storage_cost=ware_storage_cost,
                               prod_storage_cost=prod_storage_cost, manufacture_cost=manufacture_cost,
                               distribution_percent_bloem=distribution_percent_bloem,
                               production_processing_time=production_processing_time,
                               transport_time_prod_ware=transport_time_prod_ware,
                               transport_time_prodware_Bloem=transport_time_prodware_Bloem,
                               transport_cost_prod_ware=transport_cost_prod_ware,
                               transport_cost_prodware_Bloem=transport_cost_prodware_Bloem,
                               min_production_limit=min_production_limit, max_production_limit=max_production_limit)
    model = PPO("MlpPolicy", env, verbose=0, **params)
    model.learn(total_timesteps=1000000)  # Adjust total_timesteps as needed
    mean_reward = evaluate_agent(model, env)  # Implement this function to calculate the mean reward
    return mean_reward

best_mean_reward = -float("inf")
best_hyperparams = None
num_iterations = 30  # Adjust as needed

for _ in range(num_iterations):
    hyperparams = {
        'learning_rate': random.choice(param_grid['learning_rate']),
        'n_steps': random.choice(param_grid['n_steps']),
        'batch_size': random.choice(param_grid['batch_size']),
        'ent_coef': random.choice(param_grid['ent_coef']),
        'gamma': random.choice(param_grid['gamma']),
        'gae_lambda': random.choice(param_grid['gae_lambda']),
        'clip_range': random.choice(param_grid['clip_range']),
        'vf_coef': random.choice(param_grid['vf_coef']),
        'n_epochs': random.choice(param_grid['n_epochs'])
    }
    mean_reward = train_ppo(hyperparams)
    if mean_reward > best_mean_reward:
        best_mean_reward = mean_reward
        best_hyperparams = hyperparams
print("Best Hyperparameters:", best_hyperparams)
print("Best Mean Reward:", best_mean_reward)
# PPO HYPERPARAMETER TUNING END
'''
env = InventoryEnvironment(initial_bloem_units=initial_bloem_units, initial_ware_units=initial_ware_units,
                               initial_prod_units=initial_prod_units, bloem_storage_capacity=bloem_storage_capacity,
                               ware_storage_capacity=ware_storage_capacity, prod_storage_capacity=prod_storage_capacity,
                               large_truck_capacity=large_truck_capacity, small_truck_capacity=small_truck_capacity,
                               df_Bloem=df_Bloem, df_jhb=df_jhb, total_pred_bloem=total_pred_bloem,
                               total_pred_jhb=total_pred_jhb, selling_price=selling_price,
                               bloem_storage_cost=bloem_storage_cost, ware_storage_cost=ware_storage_cost,
                               prod_storage_cost=prod_storage_cost, manufacture_cost=manufacture_cost,
                               distribution_percent_bloem=distribution_percent_bloem,
                               production_processing_time=production_processing_time,
                               transport_time_prod_ware=transport_time_prod_ware,
                               transport_time_prodware_Bloem=transport_time_prodware_Bloem,
                               transport_cost_prod_ware=transport_cost_prod_ware,
                               transport_cost_prodware_Bloem=transport_cost_prodware_Bloem,
                               min_production_limit=min_production_limit, max_production_limit=max_production_limit)

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

TIMESTEPS = 100000
log_freq = 100

''''# PPO model
ent_coef = 0.1  # 0 to 0.01 [0.005, 0.01, 0.05, 0.1] default=0.1
learning_rate = 0.003  # 0.003 to 5e-6 [0.0001, 0.001, 0.003, 0.005, 0.01] default=0.003
clip_range = 0.2  # 0.1,0.2,0.3 [0.1, 0.2] default=0.2
n_steps = 2048  # 32 to 5000 [32, 128, 256, 512, 2048, 5000] default=2048
batch_size = 64  # 4 to 4096 [32, 64, 128, 4096] default=64
n_epochs = 10  # 3 to 30 [3, 5, 8, 10] default=10
gamma = 0.99  # 0.8 to 0.9997 [0.9, 0.95, 0.99] default=0.99
gae_lambda = 0.95  # 0.9 to 1 [0.95, 1] default=0.95
vf_coef = 0.5  # 0.5, 1 [0.25, 0.5, 0.75] default=0.5

model = PPO(policy='MlpPolicy', env=env, verbose=1, tensorboard_log=logdir, ent_coef=ent_coef,
            learning_rate=learning_rate, clip_range=clip_range, n_steps=n_steps, batch_size=batch_size,
            n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, vf_coef=vf_coef)

model.learn(total_timesteps=TIMESTEPS, callback=[MeticLogger(log_freq=log_freq)])
model.save("ppo_IM")
print(f"Train net profit: {env.net_profit}")
print(f"Train fill rate: {env.fill_rate}")
loaded_model = PPO.load("ppo_IM")'''

# DDPG model
learning_rate = 0.001  # [1e-4, 1e-3, 1e-2],
buffer_size = 1000000  # [int(1e4), int(1e5), int(1e6)],
learning_starts = 100
batch_size = 100  # [32, 64, 128],
tau = 0.005  # [0.001, 0.005, 0.01]
gamma = 0.99  # [0.95, 0.99],

model = DDPG(policy='MlpPolicy', env=env, verbose=1, tensorboard_log=logdir, learning_rate=learning_rate,
             buffer_size=buffer_size, learning_starts=learning_starts, batch_size=batch_size, tau=tau, gamma=gamma)

model.learn(total_timesteps=TIMESTEPS, callback=[MeticLogger(log_freq=log_freq)])
model.save("ddpg_IM")
print(f"Train net profit: {env.net_profit}")
print(f"Train fill rate: {env.fill_rate}")
loaded_model = DDPG.load("ddpg_IM")

# Evaluate the loaded model
summary_writer = tf.summary.create_file_writer(logdir)
with summary_writer.as_default():
    obs = env.reset()
    done = False
    for day in range(346):
        action, _state = loaded_model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(f"Day: {day}")
        print(f"Action: {action}")
        # print(f"Obs: {obs}")
        print(f"Prod units {env.prod_units} Ware units {env.ware_units} Bloem units {env.bloem_units}")
        print(f"choice: {env.choice}")
        print(f"production order backlog: {env.production_order_backlog}")
        print(f"warehouse order backlog: {env.warehouse_order_backlog}")
        print(f"units moving prod bloem: {env.units_moving_prod_bloem}")
        print(f"units moving ware bloem: {env.units_moving_ware_bloem}")
        print(f"units moving prodware bloem: {env.units_moving_prodware_bloem_vector}")
        print(f"Trucks prod bloem: {env.no_of_trucks_prod_bloem}")
        print(f"Trucks ware bloem: {env.no_of_trucks_ware_bloem}")

        tf.summary.scalar('Production/Production units available', env.prod_units, step=day)
        tf.summary.scalar('Production/Action', env.prod_action, step=day)
        tf.summary.scalar('Ware/Warehouse current demand', total_pred_jhb[day], step=day)
        tf.summary.scalar('Ware/Action', env.ware_action, step=day)
        tf.summary.scalar('Ware/Warehouse units available', env.ware_units, step=day)
        tf.summary.scalar('Bloem/Bloem current demand', total_pred_bloem[day], step=day)
        tf.summary.scalar('Bloem/Action', env.bloem_action, step=day)
        tf.summary.scalar('Bloem/Bloem units available', env.bloem_units, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation prod to ware', env.no_of_trucks_prod_ware, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation prod to bloem', env.no_of_trucks_prod_bloem, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation ware to bloem', env.no_of_trucks_ware_bloem, step=day)
        tf.summary.scalar('Cost/Total manufacturing cost', env.total_manufacture_cost, step=day)
        tf.summary.scalar('Cost/Total delivery cost', env.total_delivery_cost, step=day)
        tf.summary.scalar('Cost/Total storage cost', env.total_storage_cost, step=day)
        tf.summary.scalar('Cost/Overall cost',
                          env.total_delivery_cost + env.total_manufacture_cost + env.total_storage_cost, step=day)
        tf.summary.scalar('Profitability/Revenue', env.revenue_gained, step=day)
        tf.summary.scalar('Profitability/Total cost',
                          env.total_delivery_cost + env.total_manufacture_cost + env.total_storage_cost, step=day)
        tf.summary.scalar('Profitability/Net profit', env.net_profit, step=day)
        tf.summary.scalar('Units/Units satisfied', env.units_satisfied, step=day)
        tf.summary.scalar('Units/Units unsatisfied', env.units_unsatisfied, step=day)
        tf.summary.scalar('Order fulfilment rate', env.fill_rate, step=day)

        if done:
            print('DONE')
            obs = env.reset()
    print(f"Test net profit: {env.net_profit}")
    print(f"Test fill rate: {env.fill_rate}")

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
