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
import gym
from gym import Env
from gym.spaces import Box, Discrete, MultiDiscrete
from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv

start_time = time.time()

# Set the random seed for the entire environment
from stable_baselines3.common.utils import set_random_seed
seed = 123
set_random_seed(seed)

# Import demand forecasts
from Demand_forecasts_RFR import (df_bloem, total_pred_bloem, df_jhb, total_pred_jhb, df_durb, total_pred_durb, df_EL,
                                  total_pred_EL, df_CT, total_pred_CT, df_pret, total_pred_pret)

# Import input parameters
from Input_parameters import (initial_prod_units, initial_bloem_units, initial_durb_units,
                              initial_EL_units, initial_pret_units, initial_CT_units, manufacture_cost,
                              production_processing_time, min_production_limit, max_production_limit,
                              prod_storage_capacity, bloem_storage_capacity,
                              durb_storage_capacity, EL_storage_capacity, pret_storage_capacity, CT_storage_capacity,
                              large_truck_capacity,
                              transport_cost_prodware_Bloem, transport_cost_prodware_Durb, transport_cost_prodware_EL,
                              transport_cost_prod_Pret, transport_cost_prodware_CT,
                              transport_time_prodware_Bloem, transport_time_prodware_Durb,
                              transport_time_prodware_EL, transport_time_prodware_Pret, transport_time_prodware_CT,
                              prod_storage_cost, bloem_storage_cost, durb_storage_cost,
                              EL_storage_cost, pret_storage_cost, CT_storage_cost, selling_price, product_lifespan)


class InventoryEnvironment(gym.Env):
    def __init__(self, initial_bloem_units, initial_prod_units, initial_durb_units,
                 initial_EL_units, initial_pret_units, initial_CT_units, bloem_storage_capacity,
                 prod_storage_capacity, durb_storage_capacity, EL_storage_capacity, pret_storage_capacity,
                 CT_storage_capacity, large_truck_capacity, df_bloem, df_jhb, df_durb, df_EL,
                 df_CT, df_pret, total_pred_bloem, total_pred_jhb, total_pred_durb, total_pred_EL, total_pred_CT,
                 total_pred_pret, selling_price, bloem_storage_cost, prod_storage_cost,
                 durb_storage_cost, EL_storage_cost, pret_storage_cost, CT_storage_cost, manufacture_cost,
                 production_processing_time, transport_time_prodware_Bloem, transport_time_prodware_Durb,
                 transport_time_prodware_EL, transport_time_prodware_Pret, transport_time_prodware_CT,
                 transport_cost_prodware_Bloem, transport_cost_prodware_Durb,
                 transport_cost_prodware_EL, transport_cost_prod_Pret,
                 transport_cost_prodware_CT, min_production_limit, max_production_limit, product_lifespan):

        self.initial_bloem_units = initial_bloem_units
        self.initial_prod_units = initial_prod_units
        self.initial_durb_units = initial_durb_units
        self.initial_EL_units = initial_EL_units
        self.initial_pret_units = initial_pret_units
        self.initial_CT_units = initial_CT_units

        self.bloem_storage_capacity = bloem_storage_capacity
        self.prod_storage_capacity = prod_storage_capacity
        self.durb_storage_capacity = durb_storage_capacity
        self.EL_storage_capacity = EL_storage_capacity
        self.pret_storage_capacity = pret_storage_capacity
        self.CT_storage_capacity = CT_storage_capacity

        self.large_truck_capacity = large_truck_capacity

        self.demand_bloem = df_bloem
        self.demand_jhb = df_jhb
        self.demand_durb = df_durb
        self.demand_EL = df_EL
        self.demand_pret = df_pret
        self.demand_CT = df_CT

        self.total_pred_bloem = total_pred_bloem
        self.total_pred_jhb = total_pred_jhb
        self.total_pred_durb = total_pred_durb
        self.total_pred_EL = total_pred_EL
        self.total_pred_pret = total_pred_pret
        self.total_pred_CT = total_pred_CT

        self.prod_storage_cost = prod_storage_cost
        self.bloem_storage_cost = bloem_storage_cost
        self.durb_storage_cost = durb_storage_cost
        self.EL_storage_cost = EL_storage_cost
        self.pret_storage_cost = pret_storage_cost
        self.CT_storage_cost = CT_storage_cost
        self.manufacture_cost = manufacture_cost

        self.production_processing_time = production_processing_time
        self.transport_time_prodware_Bloem = transport_time_prodware_Bloem
        self.transport_time_prodware_Durb = transport_time_prodware_Durb
        self.transport_time_prodware_EL = transport_time_prodware_EL
        self.transport_time_prodware_Pret = transport_time_prodware_Pret
        self.transport_time_prodware_CT = transport_time_prodware_CT

        self.transport_cost_prodware_Bloem = transport_cost_prodware_Bloem
        self.transport_cost_prodware_Durb = transport_cost_prodware_Durb
        self.transport_cost_prodware_EL = transport_cost_prodware_EL
        self.transport_cost_prod_Pret = transport_cost_prod_Pret
        self.transport_cost_prodware_CT = transport_cost_prodware_CT

        self.min_production_limit = min_production_limit
        self.max_production_limit = max_production_limit
        self.selling_price = selling_price
        self.product_lifespan=product_lifespan

        # define action space (continuous, ordering quantity)
        self.num_stock_points = 6  # initially just considering bloem, production, durb, EL, CT, pret
        self.action_space = Box(low=0, high=1, shape=(self.num_stock_points,), dtype=np.float32)
        # self.action_space = MultiDiscrete([10,10,10])

        # define observation space
        self.num_obs_points = 69
        self.observation_space = Box(low=0, high=10000000, shape=(self.num_obs_points,), dtype=np.float32)

        # set starting inventory
        self.bloem_units_sum = initial_bloem_units  # state
        self.prod_units_sum = initial_prod_units  # state
        self.durb_units_sum = initial_durb_units
        self.EL_units_sum = initial_EL_units
        self.CT_units_sum = initial_CT_units
        self.pret_units_sum = initial_pret_units

        # set days length
        self.days_length = 341  #FOR RFR 347, for RNN 341
        # current day
        self.day = 5
        # set initial performance
        self.units_satisfied = 0
        self.units_unsatisfied = 0
        self.current_units_unsatisfied = 0
        self.current_bloem_units_unsatisfied = 0
        self.current_durb_units_unsatisfied = 0
        self.current_EL_units_unsatisfied = 0
        self.current_CT_units_unsatisfied = 0
        self.current_pret_units_unsatisfied = 0
        self.current_prod_units_unsatisfied = 0

        self.current_bloem_units_satisfied = 0
        self.current_CT_units_satisfied = 0
        self.current_EL_units_satisfied = 0
        self.current_pret_units_satisfied = 0
        self.current_durb_units_satisfied = 0
        self.current_prod_units_satisfied = 0

        self.fill_rate = 0
        self.revenue_gained = 0
        self.net_profit = 0
        self.total_storage_cost = 0
        self.total_manufacture_cost = 0
        self.total_delivery_cost = 0

        self.units_satisfied_prod = 0
        self.units_satisfied_bloem = 0
        self.units_satisfied_durb = 0
        self.units_satisfied_EL = 0
        self.units_satisfied_CT = 0
        self.units_satisfied_pret = 0

        self.units_unsatisfied_prod = 0
        self.units_unsatisfied_bloem = 0
        self.units_unsatisfied_durb = 0
        self.units_unsatisfied_EL = 0
        self.units_unsatisfied_CT = 0
        self.units_unsatisfied_pret = 0

        self.fill_rate_prod = 0
        self.fill_rate_bloem = 0
        self.fill_rate_durb = 0
        self.fill_rate_EL = 0
        self.fill_rate_CT = 0
        self.fill_rate_pret = 0

        self.net_profit_prod = 0
        self.net_profit_bloem = 0
        self.net_profit_durb = 0
        self.net_profit_EL = 0
        self.net_profit_CT = 0
        self.net_profit_pret = 0

        self.bloem_production_order_backlog = []
        self.durb_production_order_backlog = []
        self.EL_production_order_backlog = []
        self.CT_production_order_backlog = []
        self.pret_production_order_backlog = []

        self.produce_vector = []

        self.units_moving_prodware_bloem_vector = []
        self.units_moving_prodware_durb_vector = []
        self.units_moving_prodware_EL_vector = []
        self.units_moving_prodware_CT_vector = []
        self.units_moving_prodware_pret_vector = []

        # Part of observation space
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_prod_durb = 0
        self.no_of_trucks_prod_EL = 0
        self.no_of_trucks_prod_CT = 0
        self.no_of_trucks_prod_pret = 0

        self.units_transit_prod_bloem = 0
        self.units_transit_prod_durb = 0
        self.units_transit_prod_EL = 0
        self.units_transit_prod_CT = 0
        self.units_transit_prod_pret = 0

        #product lifespan
        self.prod_units = []
        self.prod_units.append([initial_prod_units, product_lifespan])
        self.bloem_units = []
        self.bloem_units.append([initial_bloem_units, product_lifespan])
        self.durb_units = []
        self.durb_units.append([initial_durb_units, product_lifespan])
        self.EL_units = []
        self.EL_units.append([initial_EL_units, product_lifespan])
        self.CT_units = []
        self.CT_units.append([initial_CT_units, product_lifespan])
        self.pret_units = []
        self.pret_units.append([initial_pret_units, product_lifespan])

        self.obsolete_inventory = 0
        self.current_obsolete_inventory = 0
        self.units_exp_1day = 0
        self.units_exp_2day = 0
        self.units_exp_3day = 0

        self.current_revenue = 0
        self.current_cost = 0
        self.reward = 0

        self.bloem_backlog_sum = 0
        self.durb_backlog_sum = 0
        self.EL_backlog_sum = 0
        self.CT_backlog_sum = 0
        self.pret_backlog_sum = 0

        self.total_delivery_cost_bloem = 0
        self.total_delivery_cost_durb = 0
        self.total_delivery_cost_EL = 0
        self.total_delivery_cost_CT = 0
        self.total_delivery_cost_pret = 0

        self.total_storage_cost_bloem = 0
        self.total_storage_cost_durb = 0
        self.total_storage_cost_EL = 0
        self.total_storage_cost_CT = 0
        self.total_storage_cost_pret = 0

        self.total_obsolete_inventory_bloem = 0
        self.total_obsolete_inventory_durb = 0
        self.total_obsolete_inventory_EL = 0
        self.total_obsolete_inventory_CT = 0
        self.total_obsolete_inventory_pret = 0
        self.total_obsolete_inventory_prod = 0

        # Rest of your environment initialization
        super(InventoryEnvironment, self).__init__()

    def reset(self, **kwargs):
        # Reset starting inventory
        self.bloem_units_sum = self.initial_bloem_units
        self.prod_units_sum = self.initial_prod_units
        self.durb_units_sum = self.initial_durb_units
        self.EL_units_sum = self.initial_EL_units
        self.CT_units_sum = self.initial_CT_units
        self.pret_units_sum = self.initial_pret_units

        # reset days length
        self.days_length = 341
        self.day = 5
        # reset initial performance
        self.units_satisfied = 0
        self.units_unsatisfied = 0
        self.current_units_unsatisfied = 0
        self.current_bloem_units_unsatisfied = 0
        self.current_durb_units_unsatisfied = 0
        self.current_EL_units_unsatisfied = 0
        self.current_CT_units_unsatisfied = 0
        self.current_pret_units_unsatisfied = 0
        self.current_prod_units_unsatisfied = 0

        self.current_bloem_units_satisfied = 0
        self.current_CT_units_satisfied = 0
        self.current_EL_units_satisfied = 0
        self.current_pret_units_satisfied = 0
        self.current_durb_units_satisfied = 0
        self.current_prod_units_satisfied = 0

        self.fill_rate = 0
        self.revenue_gained = 0
        self.total_storage_cost = 0
        self.total_manufacture_cost = 0
        self.total_delivery_cost = 0
        self.net_profit = 0

        self.units_satisfied_prod = 0
        self.units_satisfied_bloem = 0
        self.units_satisfied_durb = 0
        self.units_satisfied_EL = 0
        self.units_satisfied_CT = 0
        self.units_satisfied_pret = 0

        self.units_unsatisfied_prod = 0
        self.units_unsatisfied_bloem = 0
        self.units_unsatisfied_durb = 0
        self.units_unsatisfied_EL = 0
        self.units_unsatisfied_CT = 0
        self.units_unsatisfied_pret = 0

        self.fill_rate_prod = 0
        self.fill_rate_bloem = 0
        self.fill_rate_durb = 0
        self.fill_rate_EL = 0
        self.fill_rate_CT = 0
        self.fill_rate_pret = 0

        self.net_profit_prod = 0
        self.net_profit_bloem = 0
        self.net_profit_durb = 0
        self.net_profit_EL = 0
        self.net_profit_CT = 0
        self.net_profit_pret = 0

        self.bloem_production_order_backlog = []
        self.durb_production_order_backlog = []
        self.EL_production_order_backlog = []
        self.CT_production_order_backlog = []
        self.pret_production_order_backlog = []

        self.produce_vector = []

        self.units_moving_prodware_bloem_vector = []
        self.units_moving_prodware_durb_vector = []
        self.units_moving_prodware_EL_vector = []
        self.units_moving_prodware_CT_vector = []
        self.units_moving_prodware_pret_vector = []

        # Part of observation space
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_prod_durb = 0
        self.no_of_trucks_prod_EL = 0
        self.no_of_trucks_prod_CT = 0
        self.no_of_trucks_prod_pret = 0

        self.units_transit_prod_bloem = 0
        self.units_transit_prod_durb = 0
        self.units_transit_prod_EL = 0
        self.units_transit_prod_CT = 0
        self.units_transit_prod_pret = 0

        # product lifespan
        self.prod_units = []
        self.prod_units.append([initial_prod_units, product_lifespan])
        self.bloem_units = []
        self.bloem_units.append([initial_bloem_units, product_lifespan])
        self.durb_units = []
        self.durb_units.append([initial_durb_units, product_lifespan])
        self.EL_units = []
        self.EL_units.append([initial_durb_units, product_lifespan])
        self.CT_units = []
        self.CT_units.append([initial_CT_units, product_lifespan])
        self.pret_units = []
        self.pret_units.append([initial_pret_units, product_lifespan])

        self.obsolete_inventory = 0
        self.current_obsolete_inventory = 0
        self.units_exp_1day = 0
        self.units_exp_2day = 0
        self.units_exp_3day = 0

        self.current_revenue = 0
        self.current_cost = 0
        self.reward = 0

        self.bloem_backlog_sum = 0
        self.durb_backlog_sum = 0
        self.EL_backlog_sum = 0
        self.CT_backlog_sum = 0
        self.pret_backlog_sum = 0

        self.total_delivery_cost_bloem = 0
        self.total_delivery_cost_durb = 0
        self.total_delivery_cost_EL = 0
        self.total_delivery_cost_CT = 0
        self.total_delivery_cost_pret = 0

        self.total_storage_cost_bloem = 0
        self.total_storage_cost_durb = 0
        self.total_storage_cost_EL = 0
        self.total_storage_cost_CT = 0
        self.total_storage_cost_pret = 0

        self.total_obsolete_inventory_bloem = 0
        self.total_obsolete_inventory_durb = 0
        self.total_obsolete_inventory_EL = 0
        self.total_obsolete_inventory_CT = 0
        self.total_obsolete_inventory_pret = 0
        self.total_obsolete_inventory_prod = 0

        obs = [self.prod_units_sum, self.bloem_units_sum, self.durb_units_sum, self.EL_units_sum,
               self.CT_units_sum, self.pret_units_sum, self.no_of_trucks_prod_bloem, self.no_of_trucks_prod_durb,
               self.no_of_trucks_prod_EL, self.no_of_trucks_prod_CT, self.no_of_trucks_prod_pret,
               self.units_transit_prod_bloem, self.units_transit_prod_durb, self.units_transit_prod_EL,
               self.units_transit_prod_CT, self.units_transit_prod_pret, self.bloem_backlog_sum, self.durb_backlog_sum,
               self.EL_backlog_sum, self.CT_backlog_sum, self.pret_backlog_sum]
        bloem_forecast = self.total_pred_bloem[self.day-4:self.day + 4]
        obs.extend(bloem_forecast)
        prod_forecast = self.total_pred_jhb[self.day-4:self.day + 4]
        obs.extend(prod_forecast)
        durb_forecast = self.total_pred_durb[self.day - 4:self.day + 4]
        obs.extend(durb_forecast)
        EL_forecast = self.total_pred_EL[self.day - 4:self.day + 4]
        obs.extend(EL_forecast)
        CT_forecast = self.total_pred_CT[self.day - 4:self.day + 4]
        obs.extend(CT_forecast)
        pret_forecast = self.total_pred_pret[self.day - 4:self.day + 4]
        obs.extend(pret_forecast)
        return obs

    def step(self, action):
        self.current_revenue = 0
        self.current_cost = 0
        self.current_units_sold = 0
        self.current_units_available = 0
        self.current_units_unsatisfied = 0
        self.current_bloem_units_satisfied = 0
        self.current_CT_units_satisfied = 0
        self.current_EL_units_satisfied = 0
        self.current_pret_units_satisfied = 0
        self.current_durb_units_satisfied = 0
        self.current_prod_units_satisfied = 0

        self.current_bloem_units_unsatisfied = 0
        self.current_durb_units_unsatisfied = 0
        self.current_EL_units_unsatisfied = 0
        self.current_CT_units_unsatisfied = 0
        self.current_pret_units_unsatisfied = 0
        self.current_prod_units_unsatisfied = 0
        reward = 0
        self.reward = 0

        self.prod_units_sum = 0
        for i in range(len(self.prod_units)):
            self.prod_units_sum += self.prod_units[i][0]
        self.bloem_units_sum = 0
        for i in range(len(self.bloem_units)):
            self.bloem_units_sum += self.bloem_units[i][0]
        self.durb_units_sum = 0
        for i in range(len(self.durb_units)):
            self.durb_units_sum += self.durb_units[i][0]
        self.EL_units_sum = 0
        for i in range(len(self.EL_units)):
            self.EL_units_sum += self.EL_units[i][0]
        self.CT_units_sum = 0
        for i in range(len(self.CT_units)):
            self.CT_units_sum += self.CT_units[i][0]
        self.pret_units_sum = 0
        for i in range(len(self.pret_units)):
            self.pret_units_sum += self.pret_units[i][0]

        prod, bloem, durb, EL, CT, pret = action
        # 1) production producing quantity action
        self.prod_action = (prod * (self.max_production_limit-self.min_production_limit)) + self.min_production_limit
        if self.prod_action < self.min_production_limit:
            self.prod_action = self.min_production_limit
        elif self.prod_action > self.max_production_limit:
            self.prod_action = self.max_production_limit

        if self.prod_units_sum + self.prod_action > self.prod_storage_capacity:
            self.prod_action = self.prod_storage_capacity - self.prod_units_sum
            if self.prod_action < self.min_production_limit:
                self.prod_action = 0
            elif self.prod_action > self.max_production_limit:
                self.prod_action = self.max_production_limit

        self.total_manufacture_cost += self.prod_action * self.manufacture_cost
        self.current_cost += self.prod_action * self.manufacture_cost
        self.produce_vector.append(self.prod_action)

        # processing time before turning into finished inventory
        if self.day >= 5 + self.production_processing_time:
            self.prod_units_sum += self.produce_vector[0]
            self.prod_units.append([self.produce_vector[0], self.product_lifespan])
            del self.produce_vector[0]

        # 3) Bloem ordering quantity action
        self.bloem_action = bloem * 10000

        self.no_of_trucks_prod_bloem = 0
        self.units_moving_prod_bloem = []
        self.units_transit_prod_bloem = 0
        ss=0

        # sending units from production to bloem
        self.bloem_production_order_backlog.append(self.bloem_action)
        for n in range(len(self.bloem_production_order_backlog)):
            if (self.prod_units_sum > self.bloem_production_order_backlog[n] and self.bloem_production_order_backlog[n] +
                    self.bloem_units_sum + ss < self.bloem_storage_capacity):

                send = self.bloem_production_order_backlog[n]
                for i in range(len(self.prod_units)):
                    if send > 0:
                        if self.prod_units[i][0] > send:
                            self.prod_units[i][0] -= send
                            ss += send
                            self.units_moving_prod_bloem.append([send, self.prod_units[i][1]])
                            send -= send
                        else:
                            send -= self.prod_units[i][0]
                            ss += self.prod_units[i][0]
                            self.units_moving_prod_bloem.append(self.prod_units[i])
                            self.prod_units[i] = []
                self.prod_units = [x for x in self.prod_units if x != []]
                self.prod_units_sum = 0
                for i in range(len(self.prod_units)):
                    self.prod_units_sum += self.prod_units[i][0]

                self.bloem_production_order_backlog[n] = 0
            else:  # (don't continue satisfying orders)  #production_order_backlog = [500 200]
                break
        self.bloem_production_order_backlog = [i for i in self.bloem_production_order_backlog if i != 0]

        self.units_moving_prodware_bloem_vector.append(self.units_moving_prod_bloem)
        self.no_of_trucks_prod_bloem = math.ceil(ss / self.large_truck_capacity)
        self.units_transit_prod_bloem = ss
        self.total_delivery_cost += self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem
        self.total_delivery_cost_bloem += self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem
        self.current_cost += self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem
        self.net_profit_bloem -= self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem

        # delivery time before reaching bloem
        if self.day >= 5 + self.transport_time_prodware_Bloem:
            for i in range(len(self.units_moving_prodware_bloem_vector[0])):
                self.bloem_units.append(self.units_moving_prodware_bloem_vector[0][i])
            del self.units_moving_prodware_bloem_vector[0]
        self.bloem_units_sum = 0
        for i in range(len(self.bloem_units)):
            self.bloem_units_sum += self.bloem_units[i][0]

        self.bloem_backlog_sum = sum(self.bloem_production_order_backlog)

        # 4) Durb ordering quantity action
        self.durb_action = durb * 8000 #5

        self.no_of_trucks_prod_durb = 0
        self.units_moving_prod_durb = []
        self.units_transit_prod_durb = 0
        ss=0

        # sending units from production to durb
        self.durb_production_order_backlog.append(self.durb_action)
        for n in range(len(self.durb_production_order_backlog)):
            if (self.prod_units_sum > self.durb_production_order_backlog[n] and self.durb_production_order_backlog[n] +
                    self.durb_units_sum + ss < self.durb_storage_capacity):

                send = self.durb_production_order_backlog[n]
                for i in range(len(self.prod_units)):
                    if send > 0:
                        if self.prod_units[i][0] > send:
                            self.prod_units[i][0] -= send
                            ss += send
                            self.units_moving_prod_durb.append([send, self.prod_units[i][1]])
                            send -= send
                        else:
                            send -= self.prod_units[i][0]
                            ss += self.prod_units[i][0]
                            self.units_moving_prod_durb.append(self.prod_units[i])
                            self.prod_units[i] = []
                self.prod_units = [x for x in self.prod_units if x != []]
                self.prod_units_sum = 0
                for i in range(len(self.prod_units)):
                    self.prod_units_sum += self.prod_units[i][0]

                self.durb_production_order_backlog[n] = 0
            else:  # (don't continue satisfying orders)  #production_order_backlog = [500 200]
                break
        self.durb_production_order_backlog = [i for i in self.durb_production_order_backlog if i != 0]

        self.units_moving_prodware_durb_vector.append(self.units_moving_prod_durb)
        self.no_of_trucks_prod_durb = math.ceil(ss / self.large_truck_capacity)
        self.units_transit_prod_durb = ss
        self.total_delivery_cost += self.no_of_trucks_prod_durb * self.transport_cost_prodware_Durb
        self.total_delivery_cost_durb += self.no_of_trucks_prod_durb * self.transport_cost_prodware_Durb
        self.current_cost += self.no_of_trucks_prod_durb * self.transport_cost_prodware_Durb
        self.net_profit_durb -= self.no_of_trucks_prod_durb * self.transport_cost_prodware_Durb

        # delivery time before reaching durb
        if self.day >= 5 + self.transport_time_prodware_Durb:
            for i in range(len(self.units_moving_prodware_durb_vector[0])):
                self.durb_units.append(self.units_moving_prodware_durb_vector[0][i])
            del self.units_moving_prodware_durb_vector[0]
        self.durb_units_sum = 0
        for i in range(len(self.durb_units)):
            self.durb_units_sum += self.durb_units[i][0]

        self.durb_backlog_sum = sum(self.durb_production_order_backlog)

        # 5) EL ordering quantity action
        self.EL_action = EL * 1000

        self.no_of_trucks_prod_EL = 0
        self.units_moving_prod_EL = []
        self.units_transit_prod_EL = 0
        ss=0

        # sending units from production to EL
        self.EL_production_order_backlog.append(self.EL_action)
        for n in range(len(self.EL_production_order_backlog)):
            if (self.prod_units_sum > self.EL_production_order_backlog[n] and self.EL_production_order_backlog[n] +
                    self.EL_units_sum + ss < self.EL_storage_capacity):

                send = self.EL_production_order_backlog[n]
                for i in range(len(self.prod_units)):
                    if send > 0:
                        if self.prod_units[i][0] > send:
                            self.prod_units[i][0] -= send
                            ss += send
                            self.units_moving_prod_EL.append([send, self.prod_units[i][1]])
                            send -= send
                        else:
                            send -= self.prod_units[i][0]
                            ss += self.prod_units[i][0]
                            self.units_moving_prod_EL.append(self.prod_units[i])
                            self.prod_units[i] = []
                self.prod_units = [x for x in self.prod_units if x != []]
                self.prod_units_sum = 0
                for i in range(len(self.prod_units)):
                    self.prod_units_sum += self.prod_units[i][0]

                self.EL_production_order_backlog[n] = 0
            else:  # (don't continue satisfying orders)  #production_order_backlog = [500 200]
                break
        self.EL_production_order_backlog = [i for i in self.EL_production_order_backlog if i != 0]

        self.units_moving_prodware_EL_vector.append(self.units_moving_prod_EL)
        self.no_of_trucks_prod_EL = math.ceil(ss / self.large_truck_capacity)
        self.units_transit_prod_EL = ss
        self.total_delivery_cost += self.no_of_trucks_prod_EL * self.transport_cost_prodware_EL
        self.total_delivery_cost_EL += self.no_of_trucks_prod_EL * self.transport_cost_prodware_EL
        self.current_cost += self.no_of_trucks_prod_EL * self.transport_cost_prodware_EL
        self.net_profit_EL -= self.no_of_trucks_prod_EL * self.transport_cost_prodware_EL

        # delivery time before reaching EL
        if self.day >= 5 + self.transport_time_prodware_EL:
            for i in range(len(self.units_moving_prodware_EL_vector[0])):
                self.EL_units.append(self.units_moving_prodware_EL_vector[0][i])
            del self.units_moving_prodware_EL_vector[0]
        self.EL_units_sum = 0
        for i in range(len(self.EL_units)):
            self.EL_units_sum += self.EL_units[i][0]

        self.EL_backlog_sum = sum(self.EL_production_order_backlog)

        # 6) CT ordering quantity action
        self.CT_action = CT * 1000

        self.no_of_trucks_prod_CT = 0
        self.units_moving_prod_CT = []
        self.units_transit_prod_CT = 0
        ss=0

        # sending units from production to CT
        self.CT_production_order_backlog.append(self.CT_action)
        for n in range(len(self.CT_production_order_backlog)):
            if (self.prod_units_sum > self.CT_production_order_backlog[n] and self.CT_production_order_backlog[n] +
                    self.CT_units_sum + ss < self.CT_storage_capacity):

                send = self.CT_production_order_backlog[n]
                for i in range(len(self.prod_units)):
                    if send > 0:
                        if self.prod_units[i][0] > send:
                            self.prod_units[i][0] -= send
                            ss += send
                            self.units_moving_prod_CT.append([send, self.prod_units[i][1]])
                            send -= send
                        else:
                            send -= self.prod_units[i][0]
                            ss += self.prod_units[i][0]
                            self.units_moving_prod_CT.append(self.prod_units[i])
                            self.prod_units[i] = []
                self.prod_units = [x for x in self.prod_units if x != []]
                self.prod_units_sum = 0
                for i in range(len(self.prod_units)):
                    self.prod_units_sum += self.prod_units[i][0]

                self.CT_production_order_backlog[n] = 0
            else:  # (don't continue satisfying orders)  #production_order_backlog = [500 200]
                break
        self.CT_production_order_backlog = [i for i in self.CT_production_order_backlog if i != 0]

        self.units_moving_prodware_CT_vector.append(self.units_moving_prod_CT)
        self.no_of_trucks_prod_CT = math.ceil(ss / self.large_truck_capacity)
        self.units_transit_prod_CT = ss
        self.total_delivery_cost += self.no_of_trucks_prod_CT * self.transport_cost_prodware_CT
        self.total_delivery_cost_CT += self.no_of_trucks_prod_CT * self.transport_cost_prodware_CT
        self.current_cost += self.no_of_trucks_prod_CT * self.transport_cost_prodware_CT
        self.net_profit_CT -= self.no_of_trucks_prod_CT * self.transport_cost_prodware_CT

        # delivery time before reaching CT
        if self.day >= 5 + self.transport_time_prodware_CT:
            for i in range(len(self.units_moving_prodware_CT_vector[0])):
                self.CT_units.append(self.units_moving_prodware_CT_vector[0][i])
            del self.units_moving_prodware_CT_vector[0]
        self.CT_units_sum = 0
        for i in range(len(self.CT_units)):
            self.CT_units_sum += self.CT_units[i][0]

        self.CT_backlog_sum = sum(self.CT_production_order_backlog)

        # 7) pret ordering quantity action
        self.pret_action = pret * 6000

        self.no_of_trucks_prod_pret = 0
        self.units_moving_prod_pret = []
        self.units_transit_prod_pret = 0
        ss=0

        # sending units from production to pret
        self.pret_production_order_backlog.append(self.pret_action)
        for n in range(len(self.pret_production_order_backlog)):
            if (self.prod_units_sum > self.pret_production_order_backlog[n] and self.pret_production_order_backlog[n] +
                    self.pret_units_sum + ss < self.pret_storage_capacity):

                send = self.pret_production_order_backlog[n]
                for i in range(len(self.prod_units)):
                    if send > 0:
                        if self.prod_units[i][0] > send:
                            self.prod_units[i][0] -= send
                            ss += send
                            self.units_moving_prod_pret.append([send, self.prod_units[i][1]])
                            send -= send
                        else:
                            send -= self.prod_units[i][0]
                            ss += self.prod_units[i][0]
                            self.units_moving_prod_pret.append(self.prod_units[i])
                            self.prod_units[i] = []
                self.prod_units = [x for x in self.prod_units if x != []]
                self.prod_units_sum = 0
                for i in range(len(self.prod_units)):
                    self.prod_units_sum += self.prod_units[i][0]

                self.pret_production_order_backlog[n] = 0
            else:  # (don't continue satisfying orders)  #production_order_backlog = [500 200]
                break
        self.pret_production_order_backlog = [i for i in self.pret_production_order_backlog if i != 0]

        self.units_moving_prodware_pret_vector.append(self.units_moving_prod_pret)
        self.no_of_trucks_prod_pret = math.ceil(ss / self.large_truck_capacity)
        self.units_transit_prod_pret = ss
        self.total_delivery_cost += self.no_of_trucks_prod_pret * self.transport_cost_prod_Pret
        self.total_delivery_cost_pret += self.no_of_trucks_prod_pret * self.transport_cost_prod_Pret
        self.current_cost += self.no_of_trucks_prod_pret * self.transport_cost_prod_Pret
        self.net_profit_pret -= self.no_of_trucks_prod_pret * self.transport_cost_prod_Pret

        # delivery time before reaching pret
        if self.day >= 5 + self.transport_time_prodware_Pret:
            for i in range(len(self.units_moving_prodware_pret_vector[0])):
                self.pret_units.append(self.units_moving_prodware_pret_vector[0][i])
            del self.units_moving_prodware_pret_vector[0]
        self.pret_units_sum = 0
        for i in range(len(self.pret_units)):
            self.pret_units_sum += self.pret_units[i][0]

        self.pret_backlog_sum = sum(self.pret_production_order_backlog)

        # 8) Apply customer fulfilment at bloem
        if self.bloem_units_sum > self.demand_bloem[self.day]:

            send = self.demand_bloem[self.day]
            for i in range(len(self.bloem_units)):
                if send > 0:
                    if self.bloem_units[i][0] > send:
                        self.bloem_units[i][0] -= send
                        send -= send
                    else:
                        send -= self.bloem_units[i][0]
                        self.bloem_units[i] = []
            self.bloem_units = [x for x in self.bloem_units if x != []]
            self.bloem_units_sum = 0
            for i in range(len(self.bloem_units)):
                self.bloem_units_sum += self.bloem_units[i][0]

            self.units_satisfied += self.demand_bloem[self.day]
            self.current_bloem_units_satisfied += self.demand_bloem[self.day]
            self.units_satisfied_bloem += self.demand_bloem[self.day]
            self.current_units_sold += self.demand_bloem[self.day]
            self.revenue_gained += self.demand_bloem[self.day] * self.selling_price
            self.current_revenue += self.demand_bloem[self.day] * self.selling_price
            self.net_profit_bloem += self.demand_bloem[self.day] * self.selling_price
        else:
            self.units_satisfied += self.bloem_units_sum
            self.current_bloem_units_satisfied += self.bloem_units_sum
            self.units_satisfied_bloem += self.bloem_units_sum
            self.current_units_sold += self.bloem_units_sum
            self.units_unsatisfied += self.demand_bloem[self.day] - self.bloem_units_sum
            self.units_unsatisfied_bloem += self.demand_bloem[self.day] - self.bloem_units_sum
            self.current_units_unsatisfied += self.demand_bloem[self.day] - self.bloem_units_sum
            self.current_bloem_units_unsatisfied = self.demand_bloem[self.day] - self.bloem_units_sum
            self.revenue_gained += self.bloem_units_sum * self.selling_price
            self.current_revenue += self.bloem_units_sum * self.selling_price
            self.net_profit_bloem += self.bloem_units_sum * self.selling_price
            self.bloem_units_sum = 0
            self.bloem_units = []

        # 9) Apply customer fulfilment at durb
        if self.durb_units_sum > self.demand_durb[self.day]:

            send = self.demand_durb[self.day]
            for i in range(len(self.durb_units)):
                if send > 0:
                    if self.durb_units[i][0] > send:
                        self.durb_units[i][0] -= send
                        send -= send
                    else:
                        send -= self.durb_units[i][0]
                        self.durb_units[i] = []
            self.durb_units = [x for x in self.durb_units if x != []]
            self.durb_units_sum = 0
            for i in range(len(self.durb_units)):
                self.durb_units_sum += self.durb_units[i][0]

            self.units_satisfied += self.demand_durb[self.day]
            self.current_durb_units_satisfied += self.demand_durb[self.day]
            self.units_satisfied_durb += self.demand_durb[self.day]
            self.current_units_sold += self.demand_durb[self.day]
            self.revenue_gained += self.demand_durb[self.day] * self.selling_price
            self.current_revenue += self.demand_durb[self.day] * self.selling_price
            self.net_profit_durb += self.demand_durb[self.day] * self.selling_price
        else:
            self.units_satisfied += self.durb_units_sum
            self.current_durb_units_satisfied += self.durb_units_sum
            self.units_satisfied_durb += self.durb_units_sum
            self.current_units_sold += self.durb_units_sum
            self.units_unsatisfied += self.demand_durb[self.day] - self.durb_units_sum
            self.units_unsatisfied_durb += self.demand_durb[self.day] - self.durb_units_sum
            self.current_units_unsatisfied += self.demand_durb[self.day] - self.durb_units_sum
            self.current_durb_units_unsatisfied = self.demand_durb[self.day] - self.durb_units_sum
            self.revenue_gained += self.durb_units_sum * self.selling_price
            self.current_revenue += self.durb_units_sum * self.selling_price
            self.net_profit_durb += self.durb_units_sum * self.selling_price
            self.durb_units_sum = 0
            self.durb_units = []

        # 10) Apply customer fulfilment at EL
        if self.EL_units_sum > self.demand_EL[self.day]:

            send = self.demand_EL[self.day]
            for i in range(len(self.EL_units)):
                if send > 0:
                    if self.EL_units[i][0] > send:
                        self.EL_units[i][0] -= send
                        send -= send
                    else:
                        send -= self.EL_units[i][0]
                        self.EL_units[i] = []
            self.EL_units = [x for x in self.EL_units if x != []]
            self.EL_units_sum = 0
            for i in range(len(self.EL_units)):
                self.EL_units_sum += self.EL_units[i][0]

            self.units_satisfied += self.demand_EL[self.day]
            self.current_EL_units_satisfied += self.demand_EL[self.day]
            self.units_satisfied_EL += self.demand_EL[self.day]
            self.current_units_sold += self.demand_EL[self.day]
            self.revenue_gained += self.demand_EL[self.day] * self.selling_price
            self.current_revenue += self.demand_EL[self.day] * self.selling_price
            self.net_profit_EL += self.demand_EL[self.day] * self.selling_price
        else:
            self.units_satisfied += self.EL_units_sum
            self.current_EL_units_satisfied += self.EL_units_sum
            self.units_satisfied_EL += self.EL_units_sum
            self.current_units_sold += self.EL_units_sum
            self.units_unsatisfied += self.demand_EL[self.day] - self.EL_units_sum
            self.units_unsatisfied_EL += self.demand_EL[self.day] - self.EL_units_sum
            self.current_units_unsatisfied += self.demand_EL[self.day] - self.EL_units_sum
            self.current_EL_units_unsatisfied = self.demand_EL[self.day] - self.EL_units_sum
            self.revenue_gained += self.EL_units_sum * self.selling_price
            self.current_revenue += self.EL_units_sum * self.selling_price
            self.net_profit_EL += self.EL_units_sum * self.selling_price
            self.EL_units_sum = 0
            self.EL_units = []

        # 11) Apply customer fulfilment at CT
        if self.CT_units_sum > self.demand_CT[self.day]:

            send = self.demand_CT[self.day]
            for i in range(len(self.CT_units)):
                if send > 0:
                    if self.CT_units[i][0] > send:
                        self.CT_units[i][0] -= send
                        send -= send
                    else:
                        send -= self.CT_units[i][0]
                        self.CT_units[i] = []
            self.CT_units = [x for x in self.CT_units if x != []]
            self.CT_units_sum = 0
            for i in range(len(self.CT_units)):
                self.CT_units_sum += self.CT_units[i][0]

            self.units_satisfied += self.demand_CT[self.day]
            self.current_CT_units_satisfied += self.demand_CT[self.day]
            self.units_satisfied_CT += self.demand_CT[self.day]
            self.current_units_sold += self.demand_CT[self.day]
            self.revenue_gained += self.demand_CT[self.day] * self.selling_price
            self.current_revenue += self.demand_CT[self.day] * self.selling_price
            self.net_profit_CT += self.demand_CT[self.day] * self.selling_price
        else:
            self.units_satisfied += self.CT_units_sum
            self.current_CT_units_satisfied += self.CT_units_sum
            self.units_satisfied_CT += self.CT_units_sum
            self.current_units_sold += self.CT_units_sum
            self.units_unsatisfied += self.demand_CT[self.day] - self.CT_units_sum
            self.units_unsatisfied_CT += self.demand_CT[self.day] - self.CT_units_sum
            self.current_units_unsatisfied += self.demand_CT[self.day] - self.CT_units_sum
            self.current_CT_units_unsatisfied = self.demand_CT[self.day] - self.CT_units_sum
            self.revenue_gained += self.CT_units_sum * self.selling_price
            self.current_revenue += self.CT_units_sum * self.selling_price
            self.net_profit_CT += self.CT_units_sum * self.selling_price
            self.CT_units_sum = 0
            self.CT_units = []

        # 12) Apply customer fulfilment at pret
        if self.pret_units_sum > self.demand_pret[self.day]:

            send = self.demand_pret[self.day]
            for i in range(len(self.pret_units)):
                if send > 0:
                    if self.pret_units[i][0] > send:
                        self.pret_units[i][0] -= send
                        send -= send
                    else:
                        send -= self.pret_units[i][0]
                        self.pret_units[i] = []
            self.pret_units = [x for x in self.pret_units if x != []]
            self.pret_units_sum = 0
            for i in range(len(self.pret_units)):
                self.pret_units_sum += self.pret_units[i][0]

            self.units_satisfied += self.demand_pret[self.day]
            self.current_pret_units_satisfied += self.demand_pret[self.day]
            self.units_satisfied_pret += self.demand_pret[self.day]
            self.current_units_sold += self.demand_pret[self.day]
            self.revenue_gained += self.demand_pret[self.day] * self.selling_price
            self.current_revenue += self.demand_pret[self.day] * self.selling_price
            self.net_profit_pret += self.demand_pret[self.day] * self.selling_price
        else:
            self.units_satisfied += self.pret_units_sum
            self.current_pret_units_satisfied += self.pret_units_sum
            self.units_satisfied_pret += self.pret_units_sum
            self.current_units_sold += self.pret_units_sum
            self.units_unsatisfied += self.demand_pret[self.day] - self.pret_units_sum
            self.units_unsatisfied_pret += self.demand_pret[self.day] - self.pret_units_sum
            self.current_units_unsatisfied += self.demand_pret[self.day] - self.pret_units_sum
            self.current_pret_units_unsatisfied = self.demand_pret[self.day] - self.pret_units_sum
            self.revenue_gained += self.pret_units_sum * self.selling_price
            self.current_revenue += self.pret_units_sum * self.selling_price
            self.net_profit_pret += self.pret_units_sum * self.selling_price
            self.pret_units_sum = 0
            self.pret_units = []

        # 13) Apply customer fulfilment at prod
        if self.prod_units_sum > self.demand_jhb[self.day]:

            send = self.demand_jhb[self.day]
            for i in range(len(self.prod_units)):
                if send > 0:
                    if self.prod_units[i][0] > send:
                        self.prod_units[i][0] -= send
                        send -= send
                    else:
                        send -= self.prod_units[i][0]
                        self.prod_units[i] = []
            self.prod_units = [x for x in self.prod_units if x != []]
            self.prod_units_sum = 0
            for i in range(len(self.prod_units)):
                self.prod_units_sum += self.prod_units[i][0]

            self.units_satisfied += self.demand_jhb[self.day]
            self.current_prod_units_satisfied += self.demand_jhb[self.day]
            self.units_satisfied_prod += self.demand_jhb[self.day]
            self.current_units_sold += self.demand_jhb[self.day]
            self.revenue_gained += self.demand_jhb[self.day] * self.selling_price
            self.current_revenue += self.demand_jhb[self.day] * self.selling_price
            self.net_profit_prod += self.demand_jhb[self.day] * self.selling_price
        else:
            self.units_satisfied += self.prod_units_sum
            self.current_prod_units_satisfied += self.prod_units_sum
            self.units_satisfied_prod += self.prod_units_sum
            self.current_units_sold += self.prod_units_sum
            self.units_unsatisfied += self.demand_jhb[self.day] - self.prod_units_sum
            self.units_unsatisfied_prod += self.demand_jhb[self.day] - self.prod_units_sum
            self.current_units_unsatisfied += self.demand_jhb[self.day] - self.prod_units_sum
            self.current_prod_units_unsatisfied = self.demand_jhb[self.day] - self.prod_units_sum
            self.revenue_gained += self.prod_units_sum * self.selling_price
            self.current_revenue += self.prod_units_sum * self.selling_price
            self.net_profit_prod += self.prod_units_sum * self.selling_price
            self.prod_units_sum = 0
            self.prod_units = []

        # 14) reduce product lifespan by 1 and check for obsolete
        for i in range(len(self.prod_units)):
            if self.prod_units[i][1] == 0:
                self.obsolete_inventory += self.prod_units[i][0]
                self.total_obsolete_inventory_prod += self.prod_units[i][0]
                self.prod_units[i] = []
            else:
                self.prod_units[i][1] -= 1
        self.prod_units = [x for x in self.prod_units if x != []]

        for i in range(len(self.prod_units)):
            if self.prod_units[i][1] == 1:
                self.units_exp_1day += self.prod_units[i][0]
            elif self.prod_units[i][1] == 2:
                self.units_exp_2day += self.prod_units[i][0]
            elif self.prod_units[i][1] == 3:
                self.units_exp_3day += self.prod_units[i][0]

        self.prod_units_sum = 0
        for i in range(len(self.prod_units)):
            self.prod_units_sum += self.prod_units[i][0]

        for i in range(len(self.bloem_units)):
            if self.bloem_units[i][1] == 0:
                self.obsolete_inventory += self.bloem_units[i][0]
                self.total_obsolete_inventory_bloem += self.bloem_units[i][0]
                self.bloem_units[i] = []
            else:
                self.bloem_units[i][1] -= 1
        self.bloem_units = [x for x in self.bloem_units if x != []]

        for i in range(len(self.bloem_units)):
            if self.bloem_units[i][1] == 1:
                self.units_exp_1day += self.bloem_units[i][0]
            elif self.bloem_units[i][1] == 2:
                self.units_exp_2day += self.bloem_units[i][0]
            elif self.bloem_units[i][1] == 3:
                self.units_exp_3day += self.bloem_units[i][0]

        self.bloem_units_sum = 0
        for i in range(len(self.bloem_units)):
            self.bloem_units_sum += self.bloem_units[i][0]

        for i in range(len(self.durb_units)):
            if self.durb_units[i][1] == 0:
                self.obsolete_inventory += self.durb_units[i][0]
                self.total_obsolete_inventory_durb += self.durb_units[i][0]
                self.durb_units[i] = []
            else:
                self.durb_units[i][1] -= 1
        self.durb_units = [x for x in self.durb_units if x != []]

        for i in range(len(self.durb_units)):
            if self.durb_units[i][1] == 1:
                self.units_exp_1day += self.durb_units[i][0]
            elif self.durb_units[i][1] == 2:
                self.units_exp_2day += self.durb_units[i][0]
            elif self.durb_units[i][1] == 3:
                self.units_exp_3day += self.durb_units[i][0]

        self.durb_units_sum = 0
        for i in range(len(self.durb_units)):
            self.durb_units_sum += self.durb_units[i][0]

        for i in range(len(self.EL_units)):
            if self.EL_units[i][1] == 0:
                self.obsolete_inventory += self.EL_units[i][0]
                self.total_obsolete_inventory_EL += self.EL_units[i][0]
                self.EL_units[i] = []
            else:
                self.EL_units[i][1] -= 1
        self.EL_units = [x for x in self.EL_units if x != []]

        for i in range(len(self.EL_units)):
            if self.EL_units[i][1] == 1:
                self.units_exp_1day += self.EL_units[i][0]
            elif self.EL_units[i][1] == 2:
                self.units_exp_2day += self.EL_units[i][0]
            elif self.EL_units[i][1] == 3:
                self.units_exp_3day += self.EL_units[i][0]

        self.EL_units_sum = 0
        for i in range(len(self.EL_units)):
            self.EL_units_sum += self.EL_units[i][0]

        for i in range(len(self.CT_units)):
            if self.CT_units[i][1] == 0:
                self.obsolete_inventory += self.CT_units[i][0]
                self.total_obsolete_inventory_CT += self.CT_units[i][0]
                self.current_obsolete_inventory = self.CT_units[i][0]
                self.CT_units[i] = []
            else:
                self.CT_units[i][1] -= 1
        self.CT_units = [x for x in self.CT_units if x != []]

        for i in range(len(self.CT_units)):
            if self.CT_units[i][1] == 1:
                self.units_exp_1day += self.CT_units[i][0]
            elif self.CT_units[i][1] == 2:
                self.units_exp_2day += self.CT_units[i][0]
            elif self.CT_units[i][1] == 3:
                self.units_exp_3day += self.CT_units[i][0]

        self.CT_units_sum = 0
        for i in range(len(self.CT_units)):
            self.CT_units_sum += self.CT_units[i][0]

        for i in range(len(self.pret_units)):
            if self.pret_units[i][1] == 0:
                self.obsolete_inventory += self.pret_units[i][0]
                self.total_obsolete_inventory_pret += self.pret_units[i][0]
                self.pret_units[i] = []
            else:
                self.pret_units[i][1] -= 1
        self.pret_units = [x for x in self.pret_units if x != []]

        for i in range(len(self.pret_units)):
            if self.pret_units[i][1] == 1:
                self.units_exp_1day += self.pret_units[i][0]
            elif self.pret_units[i][1] == 2:
                self.units_exp_2day += self.pret_units[i][0]
            elif self.pret_units[i][1] == 3:
                self.units_exp_3day += self.pret_units[i][0]

        self.pret_units_sum = 0
        for i in range(len(self.pret_units)):
            self.pret_units_sum += self.pret_units[i][0]

        # 15) storage costs for remaining inventory
        self.total_storage_cost += self.bloem_units_sum * self.bloem_storage_cost
        self.total_storage_cost += self.prod_units_sum * self.prod_storage_cost
        self.total_storage_cost += self.durb_units_sum * self.durb_storage_cost
        self.total_storage_cost += self.EL_units_sum * self.EL_storage_cost
        self.total_storage_cost += self.CT_units_sum * self.CT_storage_cost
        self.total_storage_cost += self.pret_units_sum * self.pret_storage_cost

        self.current_cost += self.bloem_units_sum * self.bloem_storage_cost
        self.current_cost += self.prod_units_sum * self.prod_storage_cost
        self.current_cost += self.durb_units_sum * self.durb_storage_cost
        self.current_cost += self.EL_units_sum * self.EL_storage_cost
        self.current_cost += self.CT_units_sum * self.CT_storage_cost
        self.current_cost += self.pret_units_sum * self.pret_storage_cost

        self.total_storage_cost_bloem += self.bloem_units_sum * self.bloem_storage_cost
        self.total_storage_cost_durb += self.durb_units_sum * self.durb_storage_cost
        self.total_storage_cost_EL += self.EL_units_sum * self.EL_storage_cost
        self.total_storage_cost_CT += self.CT_units_sum * self.CT_storage_cost
        self.total_storage_cost_pret += self.pret_units_sum * self.pret_storage_cost

        self.net_profit_prod -= self.prod_units_sum * self.prod_storage_cost
        self.net_profit_bloem -= self.bloem_units_sum * self.bloem_storage_cost
        self.net_profit_durb -= self.durb_units_sum * self.durb_storage_cost
        self.net_profit_EL -= self.EL_units_sum * self.EL_storage_cost
        self.net_profit_CT -= self.CT_units_sum * self.CT_storage_cost
        self.net_profit_pret -= self.pret_units_sum * self.pret_storage_cost

        # 16) net profit and fill rate
        self.fill_rate = (self.units_satisfied / (self.units_satisfied + self.units_unsatisfied)) * 100
        self.net_profit = (self.revenue_gained - self.total_storage_cost - self.total_manufacture_cost -
                           self.total_delivery_cost)

        self.fill_rate_prod = (self.units_satisfied_prod/(self.units_satisfied_prod + self.units_unsatisfied_prod)) * 100
        self.fill_rate_bloem = (self.units_satisfied_bloem/(self.units_satisfied_bloem + self.units_unsatisfied_bloem)) * 100
        self.fill_rate_durb = (self.units_satisfied_durb/(self.units_satisfied_durb + self.units_unsatisfied_durb)) * 100
        self.fill_rate_EL = (self.units_satisfied_EL/(self.units_satisfied_EL + self.units_unsatisfied_EL)) * 100
        self.fill_rate_CT = (self.units_satisfied_CT/(self.units_satisfied_CT + self.units_unsatisfied_CT)) * 100
        self.fill_rate_pret = (self.units_satisfied_pret/(self.units_satisfied_pret + self.units_unsatisfied_pret)) * 100

        # 17) calculate reward
        reward += self.current_revenue - self.current_cost

        if self.fill_rate > 90:  # 2(service-based reward)
            reward += self.fill_rate
        elif self.fill_rate > 80:
            reward += 50
        elif self.fill_rate > 70:
            reward += 0
        elif self.fill_rate > 60:
            reward += -10
        elif self.fill_rate > 50:
            reward += -20
        elif self.fill_rate > 40:
            reward += -30
        elif self.fill_rate > 30:
            reward += -40
        elif self.fill_rate > 20:
            reward += -50
        elif self.fill_rate > 10:
            reward += -100
        else:
            reward += -100

        reward -= self.bloem_units_sum * 2.2
        reward -= self.current_bloem_units_unsatisfied

        reward -= self.CT_units_sum * 2.8
        reward -= self.current_CT_units_unsatisfied

        reward -= self.durb_units_sum * 3
        reward -= self.current_durb_units_unsatisfied * 10

        reward -= self.EL_units_sum * 2.4
        reward -= self.current_EL_units_unsatisfied

        reward -= self.pret_units_sum * 2
        reward -= self.current_pret_units_unsatisfied * 35

        reward -= self.prod_units_sum / 1000
        reward -= self.current_prod_units_unsatisfied

        self.reward = reward

        # Normalize reward
        min_reward = -2500000
        max_reward = 1000000
        target_min = -1000
        target_max = 1000
        reward = (reward - min_reward) / (max_reward - min_reward) * (target_max - target_min) + target_min

        '''reward += self.current_revenue - self.current_cost

        if self.fill_rate > 90:  # 2(service-based reward)
            reward += self.fill_rate
        elif self.fill_rate > 80:
            reward += 50
        elif self.fill_rate > 70:
            reward += 0
        elif self.fill_rate > 60:
            reward += -10
        elif self.fill_rate > 50:
            reward += -20
        elif self.fill_rate > 40:
            reward += -30
        elif self.fill_rate > 30:
            reward += -40
        elif self.fill_rate > 20:
            reward += -50
        elif self.fill_rate > 10:
            reward += -100
        else:
            reward += -100

        reward -= self.bloem_units_sum
        reward -= self.current_bloem_units_unsatisfied

        reward -= self.CT_units_sum * 2
        reward -= self.current_CT_units_unsatisfied

        reward -= self.durb_units_sum
        reward -= self.current_durb_units_unsatisfied * 10

        reward -= self.EL_units_sum
        reward -= self.current_EL_units_unsatisfied * 15

        reward -= self.pret_units_sum
        reward -= self.current_pret_units_unsatisfied * 15

        reward -= self.prod_units_sum/70
        reward -= self.current_prod_units_unsatisfied

        self.reward = reward

        # Normalize reward
        min_reward = -3000000
        max_reward = 1000000
        target_min = -3000
        target_max = 1000
        reward = (reward - min_reward) / (max_reward - min_reward) * (target_max - target_min) + target_min'''

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
        prev_prod = self.prod_units_sum
        self.prod_units_sum = ((self.prod_units_sum-min_produnits)/(max_produnits-min_produnits)*(target_max-target_min) + target_min)
        # Normalize bloem units
        min_bloemunits = 0
        max_bloemunits = 1000000  # self.bloem_storage_capacity
        target_min = 0
        target_max = 100
        prev_bloem = self.bloem_units_sum
        self.bloem_units_sum = ((self.bloem_units_sum-min_bloemunits)/(max_bloemunits-min_bloemunits)*(target_max - target_min) + target_min)
        # Normalize durb units
        min_durbunits = 0
        max_durbunits = 1000000  # self.durb_storage_capacity
        target_min = 0
        target_max = 100
        prev_durb = self.durb_units_sum
        self.durb_units_sum = ((self.durb_units_sum-min_durbunits)/(max_durbunits-min_durbunits)*(target_max - target_min) + target_min)
        # Normalize EL units
        min_ELunits = 0
        max_ELunits = 1000000  # self.EL_storage_capacity
        target_min = 0
        target_max = 100
        prev_EL = self.EL_units_sum
        self.EL_units_sum = ((self.EL_units_sum-min_ELunits)/(max_ELunits-min_ELunits)*(target_max - target_min) + target_min)
        # Normalize CT units
        min_CTunits = 0
        max_CTunits = 1000000  # self.CT_storage_capacity
        target_min = 0
        target_max = 100
        prev_CT = self.CT_units_sum
        self.CT_units_sum = ((self.CT_units_sum-min_CTunits)/(max_CTunits-min_CTunits)*(target_max - target_min) + target_min)
        # Normalize pret units
        min_pretunits = 0
        max_pretunits = 1000000  # self.pret_storage_capacity
        target_min = 0
        target_max = 100
        prev_pret = self.pret_units_sum
        self.pret_units_sum = ((self.pret_units_sum-min_pretunits)/(max_pretunits-min_pretunits)*(target_max - target_min) + target_min)

        obs = [self.prod_units_sum, self.bloem_units_sum, self.durb_units_sum, self.EL_units_sum, self.CT_units_sum,
               self.pret_units_sum, self.no_of_trucks_prod_bloem, self.no_of_trucks_prod_durb, self.no_of_trucks_prod_EL,
               self.no_of_trucks_prod_CT, self.no_of_trucks_prod_pret, self.units_transit_prod_bloem,
               self.units_transit_prod_durb, self.units_transit_prod_EL, self.units_transit_prod_CT,
               self.units_transit_prod_pret, self.bloem_backlog_sum, self.durb_backlog_sum, self.EL_backlog_sum,
               self.CT_backlog_sum, self.pret_backlog_sum]
        bloem_forecast = self.total_pred_bloem[self.day-4:self.day + 4]
        obs.extend(bloem_forecast)
        prod_forecast = self.total_pred_jhb[self.day-4:self.day + 4]
        obs.extend(prod_forecast)
        durb_forecast = self.total_pred_durb[self.day - 4:self.day + 4]
        obs.extend(durb_forecast)
        EL_forecast = self.total_pred_EL[self.day - 4:self.day + 4]
        obs.extend(EL_forecast)
        CT_forecast = self.total_pred_CT[self.day - 4:self.day + 4]
        obs.extend(CT_forecast)
        pret_forecast = self.total_pred_pret[self.day - 4:self.day + 4]
        obs.extend(pret_forecast)

        self.prod_units_sum = prev_prod
        self.bloem_units_sum = prev_bloem
        self.durb_units_sum = prev_durb
        self.EL_units_sum = prev_EL
        self.CT_units_sum = prev_CT
        self.pret_units_sum = prev_pret

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
                     'Training/reward': self.training_env.get_attr("reward")[0],
                     'Training/prod_units': self.training_env.get_attr("prod_units_sum")[0],
                     'Training/bloem_units': self.training_env.get_attr("bloem_units_sum")[0],
                     'Training/durb_units': self.training_env.get_attr("durb_units_sum")[0],
                     'Training/EL_units': self.training_env.get_attr("EL_units_sum")[0],
                     'Training/CT_units': self.training_env.get_attr("CT_units_sum")[0],
                     'Training/pret_units': self.training_env.get_attr("pret_units_sum")[0],
                     'Training/prod_action': self.training_env.get_attr("prod_action")[0],
                     'Training/bloem_action': self.training_env.get_attr("bloem_action")[0],
                     'Training/durb_action': self.training_env.get_attr("durb_action")[0],
                     'Training/EL_action': self.training_env.get_attr("EL_action")[0],
                     'Training/CT_action': self.training_env.get_attr("CT_action")[0],
                     'Training/pret_action': self.training_env.get_attr("pret_action")[0],}
            for key in stats.keys():
                self.logger.record(key, stats[key])

env = InventoryEnvironment(initial_bloem_units=initial_bloem_units,
                           initial_prod_units=initial_prod_units, initial_durb_units=initial_durb_units,
                           initial_EL_units=initial_EL_units, initial_pret_units=initial_pret_units,
                           initial_CT_units=initial_CT_units, bloem_storage_capacity=bloem_storage_capacity,
                           prod_storage_capacity=prod_storage_capacity,
                           durb_storage_capacity=durb_storage_capacity, EL_storage_capacity=EL_storage_capacity,
                           pret_storage_capacity=pret_storage_capacity, CT_storage_capacity=CT_storage_capacity,
                           large_truck_capacity=large_truck_capacity,
                           df_bloem=df_bloem, df_jhb=df_jhb, df_durb=df_durb, df_EL=df_EL, df_pret=df_pret, df_CT=df_CT,
                           total_pred_bloem=total_pred_bloem, total_pred_jhb=total_pred_jhb,
                           total_pred_durb=total_pred_durb, total_pred_EL=total_pred_EL,
                           total_pred_pret=total_pred_pret, total_pred_CT=total_pred_CT, selling_price=selling_price,
                           bloem_storage_cost=bloem_storage_cost,
                           prod_storage_cost=prod_storage_cost, durb_storage_cost=durb_storage_cost,
                           EL_storage_cost=EL_storage_cost, pret_storage_cost=pret_storage_cost,
                           CT_storage_cost=CT_storage_cost, manufacture_cost=manufacture_cost,
                           production_processing_time=production_processing_time,
                           transport_time_prodware_Bloem=transport_time_prodware_Bloem,
                           transport_time_prodware_Durb=transport_time_prodware_Durb, transport_time_prodware_EL=transport_time_prodware_EL,
                           transport_time_prodware_Pret=transport_time_prodware_Pret, transport_time_prodware_CT=transport_time_prodware_CT,
                           transport_cost_prodware_Bloem=transport_cost_prodware_Bloem,
                           transport_cost_prodware_Durb=transport_cost_prodware_Durb, transport_cost_prodware_EL=transport_cost_prodware_EL,
                           transport_cost_prod_Pret=transport_cost_prod_Pret,
                           transport_cost_prodware_CT=transport_cost_prodware_CT, min_production_limit=min_production_limit,
                           max_production_limit=max_production_limit, product_lifespan=product_lifespan)


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

def make_env(env_class):
    def _init():
        # Pass parameters here when creating an instance
        env = env_class(initial_bloem_units, initial_prod_units, initial_durb_units,
                        initial_EL_units, initial_pret_units, initial_CT_units, bloem_storage_capacity,
                        prod_storage_capacity, durb_storage_capacity, EL_storage_capacity,
                        pret_storage_capacity, CT_storage_capacity, large_truck_capacity,
                        df_bloem, df_jhb, df_durb, df_EL, df_pret, df_CT, total_pred_bloem, total_pred_jhb,
                        total_pred_durb, total_pred_EL, total_pred_pret, total_pred_CT, selling_price,
                        bloem_storage_cost, prod_storage_cost, durb_storage_cost, EL_storage_cost,
                        pret_storage_cost, CT_storage_cost, manufacture_cost, production_processing_time,
                        transport_time_prodware_Bloem, transport_time_prodware_Durb, transport_time_prodware_EL,
                        transport_time_prodware_Pret, transport_time_prodware_CT,
                        transport_cost_prodware_Bloem, transport_cost_prodware_Durb, transport_cost_prodware_EL,
                        transport_cost_prod_Pret, transport_cost_prodware_CT,
                        min_production_limit, max_production_limit, product_lifespan)
        return env
    return _init

TIMESTEPS = 2000000
log_freq = 100
n_envs = 8

# PPO model
ent_coef = 0.1  # 0 to 0.01 [0.005, 0.01, 0.05, 0.1] default=0.1
learning_rate = 0.003  # 0.003 to 5e-6 [0.0001, 0.001, 0.003, 0.005, 0.01] default=0.003
clip_range = 0.2  # 0.1,0.2,0.3 [0.1, 0.2] default=0.2
n_steps = 2048  # 32 to 5000 [32, 128, 256, 512, 2048, 5000] default=2048
batch_size = 64  # 4 to 4096 [32, 64, 128, 4096] default=64
n_epochs = 10  # 3 to 30 [3, 5, 8, 10] default=10
gamma = 0.99  # 0.8 to 0.9997 [0.9, 0.95, 0.99] default=0.99
gae_lambda = 0.95  # 0.9 to 1 [0.95, 1] default=0.95
vf_coef = 0.5  # 0.5, 1 [0.25, 0.5, 0.75] default=0.5

if __name__ == "__main__":
    envs = SubprocVecEnv([make_env(InventoryEnvironment)] * n_envs)
    model = PPO(policy='MlpPolicy', env=envs, verbose=1, tensorboard_log=logdir, ent_coef=ent_coef,
                learning_rate=learning_rate, clip_range=clip_range, n_steps=n_steps, batch_size=batch_size,
                n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, vf_coef=vf_coef)
    model.learn(total_timesteps=TIMESTEPS, callback=[MeticLogger(log_freq=log_freq)])
    model.save("ppo_IM")
    print(f"Train net profit: {env.net_profit}")
    print(f"Train fill rate: {env.fill_rate}")
    loaded_model = PPO.load("ppo_IM")

    # Evaluate the loaded model
    summary_writer = tf.summary.create_file_writer(logdir)
    with summary_writer.as_default():
        obs = env.reset()
        done = False

        prod_fill_rate_vector = []
        prod_units_available_vector = []
        bloem_current_demand_vector = []
        bloem_units_available_vector = []
        bloem_fill_rate_vector = []
        bloem_units_required_vector = []
        bloem_obsolete_inventory_vector = []
        durb_current_demand_vector = []
        durb_units_available_vector = []
        durb_fill_rate_vector = []
        durb_units_required_vector = []
        durb_obsolete_inventory_vector = []
        prod_action_vector = []

        for day in range(341):
            action, _state = loaded_model.predict(obs)
            obs, reward, done, info = env.step(action)
            print(f"Day: {day}")
            print(f"Action: {action}")
            # print(f"Obs: {obs}")
            print(f"Prod units {env.prod_units_sum} Bloem units {env.bloem_units_sum} Durb units {env.durb_units_sum} EL units {env.EL_units_sum} CT units {env.CT_units_sum} pret units {env.pret_units_sum}")

            tf.summary.scalar('Production/Prod units available', env.prod_units_sum, step=day)
            prod_units_available_vector.append(env.prod_units_sum)
            tf.summary.scalar('Production/Action', env.prod_action, step=day)
            prod_action_vector.append(env.prod_action)
            tf.summary.scalar('Production/Prod current demand', total_pred_jhb[day], step=day)
            tf.summary.scalar('Production/Prod fill rate', env.fill_rate_prod, step=day)
            prod_fill_rate_vector.append(env.fill_rate_prod)
            tf.summary.scalar('Production/Prod Current units satisfied', env.current_prod_units_satisfied, step=day)
            tf.summary.scalar('Production/Prod Current units unsatisfied', env.current_prod_units_unsatisfied, step=day)
            tf.summary.scalar('Production/Total obsolete inventory', env.total_obsolete_inventory_prod, step=day)
            tf.summary.scalar('Bloem/Bloem current demand', total_pred_bloem[day], step=day)
            bloem_current_demand_vector.append(total_pred_bloem[day])
            tf.summary.scalar('Bloem/Action', env.bloem_action, step=day)
            bloem_units_required_vector.append(env.bloem_action)
            tf.summary.scalar('Bloem/Bloem units available', env.bloem_units_sum, step=day)
            bloem_units_available_vector.append(env.bloem_units_sum)
            tf.summary.scalar('Bloem/Bloem fill rate', env.fill_rate_bloem, step=day)
            bloem_fill_rate_vector.append(env.fill_rate_bloem)
            tf.summary.scalar('Bloem/bloem Current units satisfied', env.current_bloem_units_satisfied, step=day)
            tf.summary.scalar('Bloem/bloem Current units unsatisfied', env.current_bloem_units_unsatisfied, step=day)
            tf.summary.scalar('Bloem/Total obsolete inventory', env.total_obsolete_inventory_bloem, step=day)
            bloem_obsolete_inventory_vector.append(env.total_obsolete_inventory_bloem)
            tf.summary.scalar('Bloem/Total delivery cost', env.total_delivery_cost_bloem, step=day)
            tf.summary.scalar('Bloem/Total storage cost', env.total_storage_cost_bloem, step=day)
            tf.summary.scalar('Durb/Durb current demand', total_pred_durb[day], step=day)
            durb_current_demand_vector.append(total_pred_durb[day])
            tf.summary.scalar('Durb/Action', env.durb_action, step=day)
            durb_units_required_vector.append(env.durb_action)
            tf.summary.scalar('Durb/Durb units available', env.durb_units_sum, step=day)
            durb_units_available_vector.append(env.durb_units_sum)
            tf.summary.scalar('Durb/Durb fill rate', env.fill_rate_durb, step=day)
            durb_fill_rate_vector.append(env.fill_rate_durb)
            tf.summary.scalar('Durb/Total obsolete inventory', env.total_obsolete_inventory_durb, step=day)
            durb_obsolete_inventory_vector.append(env.total_obsolete_inventory_durb)
            tf.summary.scalar('Durb/Total delivery cost', env.total_delivery_cost_durb, step=day)
            tf.summary.scalar('Durb/Total storage cost', env.total_storage_cost_durb, step=day)
            tf.summary.scalar('EL/EL current demand', total_pred_EL[day], step=day)
            tf.summary.scalar('EL/Action', env.EL_action, step=day)
            tf.summary.scalar('EL/EL units available', env.EL_units_sum, step=day)
            tf.summary.scalar('EL/EL fill rate', env.fill_rate_EL, step=day)
            tf.summary.scalar('EL/Total obsolete inventory', env.total_obsolete_inventory_EL, step=day)
            tf.summary.scalar('EL/Total delivery cost', env.total_delivery_cost_EL, step=day)
            tf.summary.scalar('EL/Total storage cost', env.total_storage_cost_EL, step=day)
            tf.summary.scalar('CT/CT current demand', total_pred_CT[day], step=day)
            tf.summary.scalar('CT/Action', env.CT_action, step=day)
            tf.summary.scalar('CT/CT units available', env.CT_units_sum, step=day)
            tf.summary.scalar('CT/CT fill rate', env.fill_rate_CT, step=day)
            tf.summary.scalar('CT/Total obsolete inventory', env.total_obsolete_inventory_CT, step=day)
            tf.summary.scalar('CT/Total delivery cost', env.total_delivery_cost_CT, step=day)
            tf.summary.scalar('CT/Total storage cost', env.total_storage_cost_CT, step=day)
            tf.summary.scalar('pret/pret current demand', total_pred_pret[day], step=day)
            tf.summary.scalar('pret/Action', env.pret_action, step=day)
            tf.summary.scalar('pret/pret units available', env.pret_units_sum, step=day)
            tf.summary.scalar('pret/pret fill rate', env.fill_rate_pret, step=day)
            tf.summary.scalar('pret/Total obsolete inventory', env.total_obsolete_inventory_pret, step=day)
            tf.summary.scalar('pret/Total delivery cost', env.total_delivery_cost_pret, step=day)
            tf.summary.scalar('pret/Total storage cost', env.total_storage_cost_pret, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to bloem', env.no_of_trucks_prod_bloem, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to durb', env.no_of_trucks_prod_durb, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to EL', env.no_of_trucks_prod_EL, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to CT', env.no_of_trucks_prod_CT, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to pret', env.no_of_trucks_prod_pret, step=day)
            tf.summary.scalar('Cost/Total manufacturing cost', env.total_manufacture_cost, step=day)
            tf.summary.scalar('Cost/Total delivery cost', env.total_delivery_cost, step=day)
            tf.summary.scalar('Cost/Total storage cost', env.total_storage_cost, step=day)
            tf.summary.scalar('Cost/Overall cost', env.total_delivery_cost + env.total_manufacture_cost + env.total_storage_cost, step=day)
            tf.summary.scalar('Profitability/Revenue', env.revenue_gained, step=day)
            tf.summary.scalar('Profitability/Total cost', env.total_delivery_cost + env.total_manufacture_cost + env.total_storage_cost, step=day)
            tf.summary.scalar('Profitability/Net profit', env.net_profit, step=day)
            tf.summary.scalar('Profitability/Reward', env.reward, step=day)
            tf.summary.scalar('Units/Units satisfied', env.units_satisfied, step=day)
            tf.summary.scalar('Units/Units unsatisfied', env.units_unsatisfied, step=day)
            tf.summary.scalar('Units/obsolete inventory', env.obsolete_inventory, step=day)
            tf.summary.scalar('Order fulfilment rate', env.fill_rate, step=day)
            tf.summary.scalar('Current/Current revenue', env.current_revenue, step=day)
            tf.summary.scalar('Current/Current cost', env.current_cost, step=day)
            tf.summary.scalar('Current/Current profit', env.current_revenue - env.current_cost, step=day)
            tf.summary.scalar('Current/Current units unsatisfied', env.current_units_unsatisfied, step=day)

            if done:
                print('DONE')
                obs = env.reset()
        print(f"Test net profit: {env.net_profit}")
        print(f"Test fill rate: {env.fill_rate}")
        print(f"Obsolete inventory: {env.obsolete_inventory}")
        print(f"Units unsatisfied: {env.units_unsatisfied}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")

        # Scatter plots
        x = range(341)

        y1 = prod_units_available_vector
        plt.figure()
        plt.plot(x, y1)
        plt.xlabel('Day')
        plt.ylabel('Butter units')
        plt.title('Production/Units available')
        plt.show()

        y2 = prod_fill_rate_vector
        plt.figure()
        plt.plot(x, y2)
        plt.xlabel('Day')
        plt.ylabel('Order fulfilment rate (%)')
        plt.title('Production/Order fulfilment rate')
        plt.show()

        y3 = bloem_current_demand_vector
        plt.figure()
        plt.plot(x, y3)
        plt.xlabel('Day')
        plt.ylabel('Butter units')
        plt.title('Bloemfontein/Current demand')
        plt.show()

        y4 = bloem_units_available_vector
        plt.figure()
        plt.plot(x, y4)
        plt.xlabel('Day')
        plt.ylabel('Butter units')
        plt.title('Bloemfontein/Units available')
        plt.show()

        y5 = bloem_fill_rate_vector
        plt.figure()
        plt.plot(x, y5)
        plt.xlabel('Day')
        plt.ylabel('Order fulfilment rate (%)')
        plt.title('Bloemfontein/Order fulfilment rate')
        plt.show()

        y6 = bloem_units_required_vector
        plt.figure()
        plt.scatter(x, y6)
        plt.xlabel('Day')
        plt.ylabel('Butter units')
        plt.title('Bloemfontein/Order quantity')
        plt.show()

        y7 = bloem_obsolete_inventory_vector
        plt.figure()
        plt.plot(x, y7)
        plt.xlabel('Day')
        plt.ylabel('Butter units')
        plt.title('Bloemfontein/Obsolete inventory')
        plt.show()

        y8 = durb_current_demand_vector
        plt.figure()
        plt.plot(x, y8)
        plt.xlabel('Day')
        plt.ylabel('Butter units')
        plt.title('Durban/Current demand')
        plt.show()

        y9 = durb_units_available_vector
        plt.figure()
        plt.plot(x, y9)
        plt.xlabel('Day')
        plt.ylabel('Butter units')
        plt.title('Durban/Units available')
        plt.show()

        y10 = durb_fill_rate_vector
        plt.figure()
        plt.plot(x, y10)
        plt.xlabel('Day')
        plt.ylabel('Order fulfilment rate (%)')
        plt.title('Durban/Order fulfilment rate')
        plt.show()

        y11 = durb_units_required_vector
        plt.figure()
        plt.scatter(x, y11)
        plt.xlabel('Day')
        plt.ylabel('Butter units')
        plt.title('Durban/Order quantity')
        plt.show()

        y12 = durb_obsolete_inventory_vector
        plt.figure()
        plt.plot(x, y12)
        plt.xlabel('Day')
        plt.ylabel('Butter units')
        plt.title('Durban/Obsolete inventory')
        plt.show()

        y13 = prod_action_vector
        plt.figure()
        plt.scatter(x, y13)
        plt.xlabel('Day')
        plt.ylabel('Butter units')
        plt.title('Production/Units produced')
        plt.show()
