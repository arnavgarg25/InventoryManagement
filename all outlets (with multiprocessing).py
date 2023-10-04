import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from Input_parameters import (initial_prod_units, initial_bloem_units, initial_ware_units, initial_durb_units,
                              initial_EL_units, initial_pret_units, initial_CT_units, manufacture_cost,
                              production_processing_time, min_production_limit, max_production_limit,
                              prod_storage_capacity, ware_storage_capacity, bloem_storage_capacity,
                              durb_storage_capacity, EL_storage_capacity, pret_storage_capacity, CT_storage_capacity,
                              small_truck_capacity, large_truck_capacity, transport_cost_prod_ware,
                              transport_cost_prodware_Bloem, transport_cost_prodware_Durb, transport_cost_prodware_EL,
                              transport_cost_prod_Pret, transport_cost_ware_Pret, transport_cost_prodware_CT,
                              transport_time_prod_ware, transport_time_prodware_Bloem, transport_time_prodware_Durb,
                              transport_time_prodware_EL, transport_time_prodware_Pret, transport_time_prodware_CT,
                              prod_storage_cost, ware_storage_cost, bloem_storage_cost, durb_storage_cost,
                              EL_storage_cost, pret_storage_cost, CT_storage_cost, distribution_percent_bloem,
                              distribution_percent_durb, distribution_percent_EL, distribution_percent_pret,
                              distribution_percent_CT, selling_price, product_lifespan)


class InventoryEnvironment(Env):
    def __init__(self, initial_bloem_units, initial_ware_units, initial_prod_units, initial_durb_units,
                 initial_EL_units, initial_pret_units, initial_CT_units, bloem_storage_capacity, ware_storage_capacity,
                 prod_storage_capacity, durb_storage_capacity, EL_storage_capacity, pret_storage_capacity,
                 CT_storage_capacity, large_truck_capacity, small_truck_capacity, df_bloem, df_jhb, df_durb, df_EL,
                 df_CT, df_pret, total_pred_bloem, total_pred_jhb, total_pred_durb, total_pred_EL, total_pred_CT,
                 total_pred_pret, selling_price, bloem_storage_cost, ware_storage_cost, prod_storage_cost,
                 durb_storage_cost, EL_storage_cost, pret_storage_cost, CT_storage_cost, manufacture_cost,
                 distribution_percent_bloem, distribution_percent_durb, distribution_percent_EL,
                 distribution_percent_pret, distribution_percent_CT, production_processing_time,
                 transport_time_prod_ware, transport_time_prodware_Bloem, transport_time_prodware_Durb,
                 transport_time_prodware_EL, transport_time_prodware_Pret, transport_time_prodware_CT,
                 transport_cost_prod_ware, transport_cost_prodware_Bloem, transport_cost_prodware_Durb,
                 transport_cost_prodware_EL, transport_cost_prod_Pret, transport_cost_ware_Pret,
                 transport_cost_prodware_CT, min_production_limit, max_production_limit):

        self.initial_bloem_units = initial_bloem_units
        self.initial_ware_units = initial_ware_units
        self.initial_prod_units = initial_prod_units
        self.initial_durb_units = initial_durb_units
        self.initial_EL_units = initial_EL_units
        self.initial_pret_units = initial_pret_units
        self.initial_CT_units = initial_CT_units

        self.bloem_storage_capacity = bloem_storage_capacity
        self.ware_storage_capacity = ware_storage_capacity
        self.prod_storage_capacity = prod_storage_capacity
        self.durb_storage_capacity = durb_storage_capacity
        self.EL_storage_capacity = EL_storage_capacity
        self.pret_storage_capacity = pret_storage_capacity
        self.CT_storage_capacity = CT_storage_capacity

        self.small_truck_capacity = small_truck_capacity
        self.large_truck_capacity = large_truck_capacity

        self.demand_bloem = df_bloem
        self.demand_ware = df_jhb
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

        self.bloem_storage_cost = bloem_storage_cost
        self.ware_storage_cost = ware_storage_cost
        self.prod_storage_cost = prod_storage_cost
        self.durb_storage_cost = durb_storage_cost
        self.EL_storage_cost = EL_storage_cost
        self.pret_storage_cost = pret_storage_cost
        self.CT_storage_cost = CT_storage_cost
        self.manufacture_cost = manufacture_cost

        self.distribution_percent_bloem = distribution_percent_bloem
        self.distribution_percent_durb = distribution_percent_durb
        self.distribution_percent_EL = distribution_percent_EL
        self.distribution_percent_pret = distribution_percent_pret
        self.distribution_percent_CT = distribution_percent_CT

        self.production_processing_time = production_processing_time
        self.transport_time_prod_ware = transport_time_prod_ware
        self.transport_time_prodware_Bloem = transport_time_prodware_Bloem
        self.transport_time_prodware_Durb = transport_time_prodware_Durb
        self.transport_time_prodware_EL = transport_time_prodware_EL
        self.transport_time_prodware_Pret = transport_time_prodware_Pret
        self.transport_time_prodware_CT = transport_time_prodware_CT

        self.transport_cost_prod_ware = transport_cost_prod_ware
        self.transport_cost_prodware_Bloem = transport_cost_prodware_Bloem
        self.transport_cost_prodware_Durb = transport_cost_prodware_Durb
        self.transport_cost_prodware_EL = transport_cost_prodware_EL
        self.transport_cost_prod_Pret = transport_cost_prod_Pret
        self.transport_cost_ware_Pret = transport_cost_ware_Pret
        self.transport_cost_prodware_CT = transport_cost_prodware_CT

        self.min_production_limit = min_production_limit
        self.max_production_limit = max_production_limit
        self.selling_price = selling_price

        # define action space (continuous, ordering quantity)
        self.num_stock_points = 7  #considering bloem, production, warehouse, durban, EL, pret, CT
        self.action_space = Box(low=0, high=1, shape=(self.num_stock_points,), dtype=np.float32)
        # self.action_space = MultiDiscrete([10,10,10])

        # define observation space
        self.num_obs_points = 94
        self.observation_space = Box(low=0, high=10000000, shape=(self.num_obs_points,), dtype=np.float32)

        # set starting inventory
        self.bloem_units = initial_bloem_units  # state
        self.ware_units = initial_ware_units  # state
        self.prod_units = initial_prod_units  # state
        self.durb_units = initial_durb_units  # state
        self.EL_units = initial_EL_units
        self.pret_units = initial_pret_units
        self.CT_units = initial_CT_units

        # set days length
        self.days_length = 347
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

        self.units_satisfied_ware = 0
        self.units_satisfied_bloem = 0
        self.units_satisfied_durb = 0
        self.units_satisfied_EL = 0
        self.units_satisfied_CT = 0
        self.units_satisfied_pret = 0
        self.units_unsatisfied_ware = 0
        self.units_unsatisfied_bloem = 0
        self.units_unsatisfied_durb = 0
        self.units_unsatisfied_EL = 0
        self.units_unsatisfied_CT = 0
        self.units_unsatisfied_pret = 0
        self.fill_rate_ware = 0
        self.fill_rate_bloem = 0
        self.fill_rate_durb = 0
        self.fill_rate_EL = 0
        self.fill_rate_CT = 0
        self.fill_rate_pret = 0
        self.net_profit_ware = 0
        self.net_profit_bloem = 0
        self.net_profit_durb = 0
        self.net_profit_EL = 0
        self.net_profit_CT = 0
        self.net_profit_pret = 0

        self.bloem_production_order_backlog = []
        self.durb_production_order_backlog = []
        self.EL_production_order_backlog = []
        self.pret_production_order_backlog = []
        self.CT_production_order_backlog = []

        self.bloem_warehouse_order_backlog = []
        self.durb_warehouse_order_backlog = []
        self.EL_warehouse_order_backlog = []
        self.pret_warehouse_order_backlog = []
        self.CT_warehouse_order_backlog = []

        self.units_moving_prodware_bloem_vector = []
        self.units_moving_prodware_durb_vector = []
        self.units_moving_prodware_EL_vector = []
        self.units_moving_prodware_pret_vector = []
        self.units_moving_prodware_CT_vector = []

        self.produce_vector = []
        self.units_moving_prod_ware_vector = []

        # Part of observation space
        self.no_of_trucks_prod_ware = 0
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_ware_bloem = 0
        self.no_of_trucks_prod_durb = 0
        self.no_of_trucks_ware_durb = 0
        self.no_of_trucks_prod_EL = 0
        self.no_of_trucks_ware_EL = 0
        self.no_of_trucks_prod_pret = 0
        self.no_of_trucks_ware_pret = 0
        self.no_of_trucks_prod_CT = 0
        self.no_of_trucks_ware_CT = 0

        self.units_transit_prod_ware = 0
        self.units_transit_prod_bloem = 0
        self.units_transit_ware_bloem = 0
        self.units_transit_prod_durb = 0
        self.units_transit_ware_durb = 0
        self.units_transit_prod_EL = 0
        self.units_transit_ware_EL = 0
        self.units_transit_prod_pret = 0
        self.units_transit_ware_pret = 0
        self.units_transit_prod_CT = 0
        self.units_transit_ware_CT = 0

        self.choice_bloem = 0
        self.choice_durb = 0
        self.choice_EL = 0
        self.choice_pret = 0
        self.choice_CT = 0

    def reset(self, **kwargs):
        # Reset starting inventory
        self.bloem_units = self.initial_bloem_units
        self.ware_units = self.initial_ware_units
        self.prod_units = self.initial_prod_units
        self.durb_units = self.initial_durb_units
        self.EL_units = initial_EL_units
        self.pret_units = initial_pret_units
        self.CT_units = initial_CT_units

        # reset days length
        self.days_length = 347
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

        self.units_satisfied_ware = 0
        self.units_satisfied_bloem = 0
        self.units_satisfied_durb = 0
        self.units_satisfied_EL = 0
        self.units_satisfied_CT = 0
        self.units_satisfied_pret = 0
        self.units_unsatisfied_ware = 0
        self.units_unsatisfied_bloem = 0
        self.units_unsatisfied_durb = 0
        self.units_unsatisfied_EL = 0
        self.units_unsatisfied_CT = 0
        self.units_unsatisfied_pret = 0
        self.fill_rate_ware = 0
        self.fill_rate_bloem = 0
        self.fill_rate_durb = 0
        self.fill_rate_EL = 0
        self.fill_rate_CT = 0
        self.fill_rate_pret = 0
        self.net_profit_ware = 0
        self.net_profit_bloem = 0
        self.net_profit_durb = 0
        self.net_profit_EL = 0
        self.net_profit_CT = 0
        self.net_profit_pret = 0

        self.bloem_production_order_backlog = []
        self.durb_production_order_backlog = []
        self.EL_production_order_backlog = []
        self.pret_production_order_backlog = []
        self.CT_production_order_backlog = []

        self.bloem_warehouse_order_backlog = []
        self.durb_warehouse_order_backlog = []
        self.EL_warehouse_order_backlog = []
        self.pret_warehouse_order_backlog = []
        self.CT_warehouse_order_backlog = []

        self.units_moving_prodware_bloem_vector = []
        self.units_moving_prodware_durb_vector = []
        self.units_moving_prodware_EL_vector = []
        self.units_moving_prodware_pret_vector = []
        self.units_moving_prodware_CT_vector = []

        self.produce_vector = []
        self.units_moving_prod_ware_vector = []

        # Part of observation space
        self.no_of_trucks_prod_ware = 0
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_ware_bloem = 0
        self.no_of_trucks_prod_durb = 0
        self.no_of_trucks_ware_durb = 0
        self.no_of_trucks_prod_EL = 0
        self.no_of_trucks_ware_EL = 0
        self.no_of_trucks_prod_pret = 0
        self.no_of_trucks_ware_pret = 0
        self.no_of_trucks_prod_CT = 0
        self.no_of_trucks_ware_CT = 0

        self.units_transit_prod_ware = 0
        self.units_transit_prod_bloem = 0
        self.units_transit_ware_bloem = 0
        self.units_transit_prod_durb = 0
        self.units_transit_ware_durb = 0
        self.units_transit_prod_EL = 0
        self.units_transit_ware_EL = 0
        self.units_transit_prod_pret = 0
        self.units_transit_ware_pret = 0
        self.units_transit_prod_CT = 0
        self.units_transit_ware_CT = 0

        self.choice_bloem = 0
        self.choice_durb = 0
        self.choice_EL = 0
        self.choice_pret = 0
        self.choice_CT = 0

        obs = [self.prod_units, self.ware_units, self.bloem_units, self.durb_units, self.EL_units, self.pret_units,
               self.CT_units, self.choice_bloem, self.choice_durb, self.choice_EL, self.choice_pret, self.choice_CT,
               self.no_of_trucks_prod_ware, self.no_of_trucks_prod_bloem, self.no_of_trucks_ware_bloem,
               self.no_of_trucks_prod_durb, self.no_of_trucks_ware_durb, self.no_of_trucks_prod_EL,
               self.no_of_trucks_ware_EL, self.no_of_trucks_prod_pret, self.no_of_trucks_ware_pret,
               self.no_of_trucks_prod_CT, self.no_of_trucks_ware_CT, self.units_transit_prod_ware,
               self.units_transit_prod_bloem, self.units_transit_ware_bloem, self.units_transit_prod_durb,
               self.units_transit_ware_durb, self.units_transit_prod_EL, self.units_transit_ware_EL,
               self.units_transit_prod_pret, self.units_transit_ware_pret, self.units_transit_prod_CT,
               self.units_transit_ware_CT]
        bloem_forecast = self.total_pred_bloem[self.day-5:self.day + 5]
        obs.extend(bloem_forecast)
        ware_forecast = self.total_pred_jhb[self.day-5:self.day + 5]
        obs.extend(ware_forecast)
        durb_forecast = self.total_pred_durb[self.day-5:self.day + 5]
        obs.extend(durb_forecast)
        EL_forecast = self.total_pred_EL[self.day - 5:self.day + 5]
        obs.extend(EL_forecast)
        pret_forecast = self.total_pred_pret[self.day - 5:self.day + 5]
        obs.extend(pret_forecast)
        CT_forecast = self.total_pred_CT[self.day - 5:self.day + 5]
        obs.extend(CT_forecast)
        return obs

    def step(self, action):
        self.current_revenue = 0
        self.current_cost = 0
        self.current_units_sold = 0
        self.current_units_available = 0
        reward = 0

        prod, ware, bloem, durb, EL, pret, CT = action
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

        # 2) Warehouse ordering quantity action
        self.ware_action = ware * self.small_truck_capacity  # 28000

        # send units from production to warehouse
        if self.ware_units + self.ware_action > self.ware_storage_capacity:
            self.ware_action = self.ware_storage_capacity - self.ware_units

        if self.ware_action <= self.prod_units:
            self.prod_units -= self.ware_action
        else:
            self.ware_action = self.prod_units
            self.prod_units = 0

        self.units_moving_prod_ware_vector.append(self.ware_action)

        # delivery time before reaching warehouse
        if self.day >= 5 + self.transport_time_prod_ware:
            self.ware_units += self.units_moving_prod_ware_vector[0]
            self.units_transit_prod_ware = self.units_moving_prod_ware_vector[0]
            del self.units_moving_prod_ware_vector[0]

        self.no_of_trucks_prod_ware = math.ceil(self.ware_action / self.small_truck_capacity)
        self.total_delivery_cost += self.no_of_trucks_prod_ware * self.transport_cost_prod_ware
        self.current_cost += self.no_of_trucks_prod_ware * self.transport_cost_prod_ware
        self.net_profit_ware -= self.no_of_trucks_prod_ware * self.transport_cost_prod_ware

        # 3) Bloemfontein ordering quantity action
        self.bloem_action = bloem * self.large_truck_capacity  # 58800

        # Determine whether Bloemfontein order is from production or warehouse
        self.no_of_trucks_prod_bloem = 0
        self.no_of_trucks_ware_bloem = 0
        self.units_moving_ware_bloem = 0
        self.units_moving_prod_bloem = 0
        self.choice_bloem = random.choice(range(1, 101))

        # sending units from production to Bloemfontein
        if self.choice_bloem > self.distribution_percent_bloem:
            self.choice_bloem = 1
            self.bloem_production_order_backlog.append(self.bloem_action)
            for n in range(len(self.bloem_production_order_backlog)-1):
                if (self.prod_units > self.bloem_production_order_backlog[n] and self.bloem_production_order_backlog[n]
                        + self.bloem_units + self.units_moving_prod_bloem < self.bloem_storage_capacity):
                    self.prod_units -= self.bloem_production_order_backlog[n]
                    self.units_moving_prod_bloem += self.bloem_production_order_backlog[n]
                    self.bloem_production_order_backlog[n] = 0
                else:
                    break
            self.bloem_production_order_backlog = [i for i in self.bloem_production_order_backlog if i != 0]

            self.units_moving_prodware_bloem_vector.append(self.units_moving_prod_bloem)
            self.no_of_trucks_prod_bloem = math.ceil(self.units_moving_prod_bloem / self.large_truck_capacity)
            self.units_transit_prod_bloem = self.units_moving_prod_bloem
            self.total_delivery_cost += self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem
            self.current_cost += self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem
            self.net_profit_bloem -= self.no_of_trucks_prod_bloem * self.transport_cost_prodware_Bloem

        # sending units from warehouse to bloem
        else:
            self.choice_bloem = 0
            self.bloem_warehouse_order_backlog.append(self.bloem_action)
            for n in range(len(self.bloem_warehouse_order_backlog)-1):
                if (self.ware_units > self.bloem_warehouse_order_backlog[n] and self.bloem_warehouse_order_backlog[n] +
                        self.bloem_units + self.units_moving_ware_bloem < self.bloem_storage_capacity):
                    self.ware_units -= self.bloem_warehouse_order_backlog[n]
                    self.units_moving_ware_bloem += self.bloem_warehouse_order_backlog[n]
                    self.bloem_warehouse_order_backlog[n] = 0
                else:
                    break
            self.bloem_warehouse_order_backlog = [i for i in self.bloem_warehouse_order_backlog if i != 0]

            self.units_moving_prodware_bloem_vector.append(self.units_moving_ware_bloem)
            self.no_of_trucks_ware_bloem = math.ceil(self.units_moving_ware_bloem / self.large_truck_capacity)
            self.units_transit_ware_bloem = self.units_moving_ware_bloem
            self.total_delivery_cost += self.no_of_trucks_ware_bloem * self.transport_cost_prodware_Bloem
            self.current_cost += self.no_of_trucks_ware_bloem * self.transport_cost_prodware_Bloem
            self.net_profit_bloem -= self.no_of_trucks_ware_bloem * self.transport_cost_prodware_Bloem

        # delivery time before reaching Bloemfontein
        if self.day >= 5 + self.transport_time_prodware_Bloem:
            self.bloem_units += self.units_moving_prodware_bloem_vector[0]
            del self.units_moving_prodware_bloem_vector[0]

        # 4) Durban ordering quantity action
        self.durb_action = durb * self.large_truck_capacity  # 58800

        # Determine whether Durban order is from production or warehouse
        self.no_of_trucks_prod_durb = 0
        self.no_of_trucks_ware_durb = 0
        self.units_moving_ware_durb = 0
        self.units_moving_prod_durb = 0
        self.choice_durb = random.choice(range(1, 101))

        # sending units from production to Durban
        if self.choice_durb > self.distribution_percent_durb:
            self.choice_durb = 1
            self.durb_production_order_backlog.append(self.durb_action)
            for n in range(len(self.durb_production_order_backlog)-1):
                if (self.prod_units > self.durb_production_order_backlog[n] and self.durb_production_order_backlog[n] +
                        self.durb_units + self.units_moving_prod_durb < self.durb_storage_capacity):
                    self.prod_units -= self.durb_production_order_backlog[n]
                    self.units_moving_prod_durb += self.durb_production_order_backlog[n]
                    self.durb_production_order_backlog[n] = 0
                else:
                    break
            self.durb_production_order_backlog = [i for i in self.durb_production_order_backlog if i != 0]

            self.units_moving_prodware_durb_vector.append(self.units_moving_prod_durb)
            self.no_of_trucks_prod_durb = math.ceil(self.units_moving_prod_durb / self.large_truck_capacity)
            self.units_transit_prod_durb = self.units_moving_prod_durb
            self.total_delivery_cost += self.no_of_trucks_prod_durb * self.transport_cost_prodware_Durb
            self.current_cost += self.no_of_trucks_prod_durb * self.transport_cost_prodware_Durb
            self.net_profit_durb -= self.no_of_trucks_prod_durb * self.transport_cost_prodware_Durb

        # sending units from warehouse to Durban
        else:
            self.choice_durb = 0
            self.durb_warehouse_order_backlog.append(self.durb_action)
            for n in range(len(self.durb_warehouse_order_backlog)-1):
                if (self.ware_units > self.durb_warehouse_order_backlog[n] and self.durb_warehouse_order_backlog[n] +
                        self.durb_units + self.units_moving_ware_durb < self.durb_storage_capacity):
                    self.ware_units -= self.durb_warehouse_order_backlog[n]
                    self.units_moving_ware_durb += self.durb_warehouse_order_backlog[n]
                    self.durb_warehouse_order_backlog[n] = 0
                else:
                    break
            self.durb_warehouse_order_backlog = [i for i in self.durb_warehouse_order_backlog if i != 0]

            self.units_moving_prodware_durb_vector.append(self.units_moving_ware_durb)
            self.no_of_trucks_ware_durb = math.ceil(self.units_moving_ware_durb / self.large_truck_capacity)
            self.units_transit_ware_durb = self.units_moving_ware_durb
            self.total_delivery_cost += self.no_of_trucks_ware_durb * self.transport_cost_prodware_Durb
            self.current_cost += self.no_of_trucks_ware_durb * self.transport_cost_prodware_Durb
            self.net_profit_durb -= self.no_of_trucks_ware_durb * self.transport_cost_prodware_Durb

        # delivery time before reaching Durban
        if self.day >= 5 + self.transport_time_prodware_Durb:
            self.durb_units += self.units_moving_prodware_durb_vector[0]
            del self.units_moving_prodware_durb_vector[0]

        # 4) East London ordering quantity action
        self.EL_action = EL * self.large_truck_capacity  # 58800

        # Determine whether East London order is from production or warehouse
        self.no_of_trucks_prod_EL = 0
        self.no_of_trucks_ware_EL = 0
        self.units_moving_ware_EL = 0
        self.units_moving_prod_EL = 0
        self.choice_EL = random.choice(range(1, 101))

        # sending units from production to East London
        if self.choice_EL > self.distribution_percent_EL:
            self.choice_EL = 1
            self.EL_production_order_backlog.append(self.EL_action)
            for n in range(len(self.EL_production_order_backlog) - 1):
                if (self.prod_units > self.EL_production_order_backlog[n] and self.EL_production_order_backlog[n] +
                        self.EL_units + self.units_moving_prod_EL < self.EL_storage_capacity):
                    self.prod_units -= self.EL_production_order_backlog[n]
                    self.units_moving_prod_EL += self.EL_production_order_backlog[n]
                    self.EL_production_order_backlog[n] = 0
                else:
                    break
            self.EL_production_order_backlog = [i for i in self.EL_production_order_backlog if i != 0]

            self.units_moving_prodware_EL_vector.append(self.units_moving_prod_EL)
            self.no_of_trucks_prod_EL = math.ceil(self.units_moving_prod_EL / self.large_truck_capacity)
            self.units_transit_prod_EL = self.units_moving_prod_EL
            self.total_delivery_cost += self.no_of_trucks_prod_EL * self.transport_cost_prodware_EL
            self.current_cost += self.no_of_trucks_prod_EL * self.transport_cost_prodware_EL
            self.net_profit_EL -= self.no_of_trucks_prod_EL * self.transport_cost_prodware_EL

        # sending units from warehouse to East London
        else:
            self.choice_EL = 0
            self.EL_warehouse_order_backlog.append(self.EL_action)
            for n in range(len(self.EL_warehouse_order_backlog) - 1):
                if (self.ware_units > self.EL_warehouse_order_backlog[n] and self.EL_warehouse_order_backlog[n] +
                        self.EL_units + self.units_moving_ware_EL < self.EL_storage_capacity):
                    self.ware_units -= self.EL_warehouse_order_backlog[n]
                    self.units_moving_ware_EL += self.EL_warehouse_order_backlog[n]
                    self.EL_warehouse_order_backlog[n] = 0
                else:
                    break
            self.EL_warehouse_order_backlog = [i for i in self.EL_warehouse_order_backlog if i != 0]

            self.units_moving_prodware_EL_vector.append(self.units_moving_ware_EL)
            self.no_of_trucks_ware_EL = math.ceil(self.units_moving_ware_EL / self.large_truck_capacity)
            self.units_transit_ware_EL = self.units_moving_ware_EL
            self.total_delivery_cost += self.no_of_trucks_ware_EL * self.transport_cost_prodware_EL
            self.current_cost += self.no_of_trucks_ware_EL * self.transport_cost_prodware_EL
            self.net_profit_EL -= self.no_of_trucks_ware_EL * self.transport_cost_prodware_EL

        # delivery time before reaching East London
        if self.day >= 5 + self.transport_time_prodware_EL:
            self.EL_units += self.units_moving_prodware_EL_vector[0]
            del self.units_moving_prodware_EL_vector[0]

        # 4) Cape Town ordering quantity action
        self.CT_action = CT * self.large_truck_capacity  # 58800

        # Determine whether Cape Town order is from production or warehouse
        self.no_of_trucks_prod_CT = 0
        self.no_of_trucks_ware_CT = 0
        self.units_moving_ware_CT = 0
        self.units_moving_prod_CT = 0
        self.choice_CT = random.choice(range(1, 101))

        # sending units from production to Cape Town
        if self.choice_CT > self.distribution_percent_CT:
            self.choice_CT = 1
            self.CT_production_order_backlog.append(self.CT_action)
            for n in range(len(self.CT_production_order_backlog) - 1):
                if (self.prod_units > self.CT_production_order_backlog[n] and self.CT_production_order_backlog[n] +
                        self.CT_units + self.units_moving_prod_CT < self.CT_storage_capacity):
                    self.prod_units -= self.CT_production_order_backlog[n]
                    self.units_moving_prod_CT += self.CT_production_order_backlog[n]
                    self.CT_production_order_backlog[n] = 0
                else:
                    break
            self.CT_production_order_backlog = [i for i in self.CT_production_order_backlog if i != 0]

            self.units_moving_prodware_CT_vector.append(self.units_moving_prod_CT)
            self.no_of_trucks_prod_CT = math.ceil(self.units_moving_prod_CT / self.large_truck_capacity)
            self.units_transit_prod_CT = self.units_moving_prod_CT
            self.total_delivery_cost += self.no_of_trucks_prod_CT * self.transport_cost_prodware_CT
            self.current_cost += self.no_of_trucks_prod_CT * self.transport_cost_prodware_CT
            self.net_profit_CT -= self.no_of_trucks_prod_CT * self.transport_cost_prodware_CT

        # sending units from warehouse to Cape Town
        else:
            self.choice_CT = 0
            self.CT_warehouse_order_backlog.append(self.CT_action)
            for n in range(len(self.CT_warehouse_order_backlog) - 1):
                if (self.ware_units > self.CT_warehouse_order_backlog[n] and self.CT_warehouse_order_backlog[n] +
                        self.CT_units + self.units_moving_ware_CT < self.CT_storage_capacity):
                    self.ware_units -= self.CT_warehouse_order_backlog[n]
                    self.units_moving_ware_CT += self.CT_warehouse_order_backlog[n]
                    self.CT_warehouse_order_backlog[n] = 0
                else:
                    break
            self.CT_warehouse_order_backlog = [i for i in self.CT_warehouse_order_backlog if i != 0]

            self.units_moving_prodware_CT_vector.append(self.units_moving_ware_CT)
            self.no_of_trucks_ware_CT = math.ceil(self.units_moving_ware_CT / self.large_truck_capacity)
            self.units_transit_ware_CT = self.units_moving_ware_CT
            self.total_delivery_cost += self.no_of_trucks_ware_CT * self.transport_cost_prodware_CT
            self.current_cost += self.no_of_trucks_ware_CT * self.transport_cost_prodware_CT
            self.net_profit_CT -= self.no_of_trucks_ware_CT * self.transport_cost_prodware_CT

        # delivery time before reaching Cape Town
        if self.day >= 5 + self.transport_time_prodware_CT:
            self.CT_units += self.units_moving_prodware_CT_vector[0]
            del self.units_moving_prodware_CT_vector[0]

        # 4) Pretoria ordering quantity action
        self.pret_action = pret * self.large_truck_capacity  # 58800

        # Determine whether Pretoria order is from production or warehouse
        self.no_of_trucks_prod_pret = 0
        self.no_of_trucks_ware_pret = 0
        self.units_moving_ware_pret = 0
        self.units_moving_prod_pret = 0
        self.choice_pret = random.choice(range(1, 101))

        # sending units from production to Pretoria
        if self.choice_pret > self.distribution_percent_pret:
            self.choice_pret = 1
            self.pret_production_order_backlog.append(self.pret_action)
            for n in range(len(self.pret_production_order_backlog) - 1):
                if (self.prod_units > self.pret_production_order_backlog[n] and self.pret_production_order_backlog[n] +
                        self.pret_units + self.units_moving_prod_pret < self.pret_storage_capacity):
                    self.prod_units -= self.pret_production_order_backlog[n]
                    self.units_moving_prod_pret += self.pret_production_order_backlog[n]
                    self.pret_production_order_backlog[n] = 0
                else:
                    break
            self.pret_production_order_backlog = [i for i in self.pret_production_order_backlog if i != 0]

            self.units_moving_prodware_pret_vector.append(self.units_moving_prod_pret)
            self.no_of_trucks_prod_pret = math.ceil(self.units_moving_prod_pret / self.large_truck_capacity)
            self.units_transit_prod_pret = self.units_moving_prod_pret
            self.total_delivery_cost += self.no_of_trucks_prod_pret * self.transport_cost_prod_Pret
            self.current_cost += self.no_of_trucks_prod_pret * self.transport_cost_prod_Pret
            self.net_profit_pret -= self.no_of_trucks_prod_pret * self.transport_cost_prod_Pret

        # sending units from warehouse to Pretoria
        else:
            self.choice_pret = 0
            self.pret_warehouse_order_backlog.append(self.pret_action)
            for n in range(len(self.pret_warehouse_order_backlog) - 1):
                if (self.ware_units > self.pret_warehouse_order_backlog[n] and self.pret_warehouse_order_backlog[n] +
                        self.pret_units + self.units_moving_ware_pret < self.pret_storage_capacity):
                    self.ware_units -= self.pret_warehouse_order_backlog[n]
                    self.units_moving_ware_pret += self.pret_warehouse_order_backlog[n]
                    self.pret_warehouse_order_backlog[n] = 0
                else:
                    break
            self.pret_warehouse_order_backlog = [i for i in self.pret_warehouse_order_backlog if i != 0]

            self.units_moving_prodware_pret_vector.append(self.units_moving_ware_pret)
            self.no_of_trucks_ware_pret = math.ceil(self.units_moving_ware_pret / self.large_truck_capacity)
            self.units_transit_ware_pret = self.units_moving_ware_pret
            self.total_delivery_cost += self.no_of_trucks_ware_pret * self.transport_cost_ware_Pret
            self.current_cost += self.no_of_trucks_ware_pret * self.transport_cost_ware_Pret
            self.net_profit_pret -= self.no_of_trucks_ware_pret * self.transport_cost_ware_Pret

        # delivery time before reaching Pretoria
        if self.day >= 5 + self.transport_time_prodware_Pret:
            self.pret_units += self.units_moving_prodware_pret_vector[0]
            del self.units_moving_prodware_pret_vector[0]

        # 5) Apply customer fulfilment at Bloemfontein
        if self.bloem_units > self.demand_bloem[self.day]:
            self.bloem_units -= self.demand_bloem[self.day]
            self.units_satisfied += self.demand_bloem[self.day]
            self.units_satisfied_bloem += self.demand_bloem[self.day]
            self.current_units_sold += self.demand_bloem[self.day]
            self.revenue_gained += self.demand_bloem[self.day] * self.selling_price
            self.current_revenue += self.demand_bloem[self.day] * self.selling_price
            self.net_profit_bloem += self.demand_bloem[self.day] * self.selling_price
        else:
            self.units_satisfied += self.bloem_units
            self.units_satisfied_bloem += self.bloem_units
            self.current_units_sold += self.bloem_units
            self.units_unsatisfied += self.demand_bloem[self.day] - self.bloem_units
            self.units_unsatisfied_bloem += self.demand_bloem[self.day] - self.bloem_units
            self.revenue_gained += self.bloem_units * self.selling_price
            self.current_revenue += self.bloem_units * self.selling_price
            self.net_profit_bloem += self.bloem_units * self.selling_price
            self.bloem_units = 0

        # 6) Apply customer fulfilment at Durban
        if self.durb_units > self.demand_durb[self.day]:
            self.durb_units -= self.demand_durb[self.day]
            self.units_satisfied += self.demand_durb[self.day]
            self.units_satisfied_durb += self.demand_durb[self.day]
            self.current_units_sold += self.demand_durb[self.day]
            self.revenue_gained += self.demand_durb[self.day] * self.selling_price
            self.current_revenue += self.demand_durb[self.day] * self.selling_price
            self.net_profit_durb += self.demand_durb[self.day] * self.selling_price
        else:
            self.units_satisfied += self.durb_units
            self.units_satisfied_durb += self.durb_units
            self.current_units_sold += self.durb_units
            self.units_unsatisfied += self.demand_durb[self.day] - self.durb_units
            self.units_unsatisfied_durb += self.demand_durb[self.day] - self.durb_units
            self.revenue_gained += self.durb_units * self.selling_price
            self.current_revenue += self.durb_units * self.selling_price
            self.net_profit_durb += self.durb_units * self.selling_price
            self.durb_units = 0

        # 7) Apply customer fulfilment at East London
        if self.EL_units > self.demand_EL[self.day]:
            self.EL_units -= self.demand_EL[self.day]
            self.units_satisfied += self.demand_EL[self.day]
            self.units_satisfied_EL += self.demand_EL[self.day]
            self.current_units_sold += self.demand_EL[self.day]
            self.revenue_gained += self.demand_EL[self.day] * self.selling_price
            self.current_revenue += self.demand_EL[self.day] * self.selling_price
            self.net_profit_EL += self.demand_EL[self.day] * self.selling_price
        else:
            self.units_satisfied += self.EL_units
            self.units_satisfied_EL += self.EL_units
            self.current_units_sold += self.EL_units
            self.units_unsatisfied += self.demand_EL[self.day] - self.EL_units
            self.units_unsatisfied_EL += self.demand_EL[self.day] - self.EL_units
            self.revenue_gained += self.EL_units * self.selling_price
            self.current_revenue += self.EL_units * self.selling_price
            self.net_profit_EL += self.EL_units * self.selling_price
            self.EL_units = 0

        # 9) Apply customer fulfilment at Pretoria
        if self.pret_units > self.demand_pret[self.day]:
            self.pret_units -= self.demand_pret[self.day]
            self.units_satisfied += self.demand_pret[self.day]
            self.units_satisfied_pret += self.demand_pret[self.day]
            self.current_units_sold += self.demand_pret[self.day]
            self.revenue_gained += self.demand_pret[self.day] * self.selling_price
            self.current_revenue += self.demand_pret[self.day] * self.selling_price
            self.net_profit_pret += self.demand_pret[self.day] * self.selling_price
        else:
            self.units_satisfied += self.pret_units
            self.units_satisfied_pret += self.pret_units
            self.current_units_sold += self.pret_units
            self.units_unsatisfied += self.demand_pret[self.day] - self.pret_units
            self.units_unsatisfied_pret += self.demand_pret[self.day] - self.pret_units
            self.revenue_gained += self.pret_units * self.selling_price
            self.current_revenue += self.pret_units * self.selling_price
            self.net_profit_pret += self.pret_units * self.selling_price
            self.pret_units = 0

        # 10) Apply customer fulfilment at Cape Town
        if self.CT_units > self.demand_CT[self.day]:
            self.CT_units -= self.demand_CT[self.day]
            self.units_satisfied += self.demand_CT[self.day]
            self.units_satisfied_CT += self.demand_CT[self.day]
            self.current_units_sold += self.demand_CT[self.day]
            self.revenue_gained += self.demand_CT[self.day] * self.selling_price
            self.current_revenue += self.demand_CT[self.day] * self.selling_price
            self.net_profit_CT += self.demand_CT[self.day] * self.selling_price
        else:
            self.units_satisfied += self.CT_units
            self.units_satisfied_CT += self.CT_units
            self.current_units_sold += self.CT_units
            self.units_unsatisfied += self.demand_CT[self.day] - self.CT_units
            self.units_unsatisfied_CT += self.demand_CT[self.day] - self.CT_units
            self.revenue_gained += self.CT_units * self.selling_price
            self.current_revenue += self.CT_units * self.selling_price
            self.net_profit_CT += self.CT_units * self.selling_price
            self.CT_units = 0

        # 11) Apply customer fulfilment at the Warehouse
        if self.ware_units > self.demand_ware[self.day]:
            self.ware_units -= self.demand_ware[self.day]
            self.units_satisfied += self.demand_ware[self.day]
            self.units_satisfied_ware += self.demand_ware[self.day]
            self.current_units_sold += self.demand_ware[self.day]
            self.revenue_gained += self.demand_ware[self.day] * self.selling_price
            self.current_revenue += self.demand_ware[self.day] * self.selling_price
            self.net_profit_ware += self.demand_ware[self.day] * self.selling_price
        else:
            self.units_satisfied += self.ware_units
            self.units_satisfied_ware += self.ware_units
            self.current_units_sold += self.ware_units
            self.units_unsatisfied += self.demand_ware[self.day] - self.ware_units
            self.units_unsatisfied_ware += self.demand_ware[self.day] - self.ware_units
            self.revenue_gained += self.ware_units * self.selling_price
            self.current_revenue += self.ware_units * self.selling_price
            self.net_profit_ware += self.ware_units * self.selling_price
            self.ware_units = 0

        # 12) Net profit and Fill rate
        self.fill_rate = (self.units_satisfied / (self.units_satisfied + self.units_unsatisfied)) * 100
        self.net_profit = (self.revenue_gained - self.total_storage_cost - self.total_manufacture_cost -
                           self.total_delivery_cost)

        self.fill_rate_ware = (self.units_satisfied_ware/(self.units_satisfied_ware + self.units_unsatisfied_ware)) * 100
        self.fill_rate_bloem = (self.units_satisfied_bloem/(self.units_satisfied_bloem + self.units_unsatisfied_bloem)) * 100
        self.fill_rate_durb = (self.units_satisfied_durb/(self.units_satisfied_durb + self.units_unsatisfied_durb)) * 100
        self.fill_rate_EL = (self.units_satisfied_EL/(self.units_satisfied_EL + self.units_unsatisfied_EL)) * 100
        self.fill_rate_CT = (self.units_satisfied_CT/(self.units_satisfied_CT + self.units_unsatisfied_CT)) * 100
        self.fill_rate_pret = (self.units_satisfied_pret/(self.units_satisfied_pret + self.units_unsatisfied_pret)) * 100

        # 13) storage costs for remaining inventory
        self.total_storage_cost += self.bloem_units * self.bloem_storage_cost
        self.total_storage_cost += self.ware_units * self.ware_storage_cost
        self.total_storage_cost += self.prod_units * self.prod_storage_cost
        self.total_storage_cost += self.durb_units * self.durb_storage_cost
        self.total_storage_cost += self.EL_units * self.EL_storage_cost
        self.total_storage_cost += self.pret_units * self.pret_storage_cost
        self.total_storage_cost += self.CT_units * self.CT_storage_cost
        self.current_cost += self.bloem_units * self.bloem_storage_cost
        self.current_cost += self.ware_units * self.ware_storage_cost
        self.current_cost += self.prod_units * self.prod_storage_cost
        self.current_cost += self.durb_units * self.durb_storage_cost
        self.current_cost += self.EL_units * self.EL_storage_cost
        self.current_cost += self.pret_units * self.pret_storage_cost
        self.current_cost += self.CT_units * self.CT_storage_cost
        self.net_profit_ware -= self.ware_units * self.ware_storage_cost
        self.net_profit_bloem -= self.bloem_units * self.bloem_storage_cost
        self.net_profit_durb -= self.durb_units * self.durb_storage_cost
        self.net_profit_EL -= self.EL_units * self.EL_storage_cost
        self.net_profit_CT -= self.CT_units * self.CT_storage_cost
        self.net_profit_pret -= self.pret_units * self.pret_storage_cost

        # 14) calculate reward
        reward += self.current_revenue - self.current_cost  # (Profit-based reward)

        if self.fill_rate > 90:  # (service-based reward)
            reward += 100
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

        # (Maximizing units sold while minimizing units available)
        # reward += self.current_units_sold - (self.bloem_units + self.ware_units + self.prod_units)
        # reward -= self.bloem_units
        # reward -= self.ware_units
        # reward -= self.prod_units

        # Normalize reward
        min_reward = -100000
        max_reward = 100000
        target_min = -10
        target_max = 10
        reward = (reward - min_reward) / (max_reward - min_reward) * (target_max - target_min) + target_min

        # check if days are complete
        if self.days_length <= 0:
            done = True
        else:
            done = False

        # increase current day
        self.day += 1
        # reduce number of days by 1
        self.days_length -= 1

        # Normalize Production units
        min_produnits = 0
        max_produnits = 1000000  # self.prod_storage_capacity
        target_min = 0
        target_max = 100
        prev_prod = self.prod_units
        self.prod_units = ((self.prod_units-min_produnits)/(max_produnits-min_produnits)*(target_max-target_min)
                           + target_min)
        # Normalize Warehouse units
        min_wareunits = 0
        max_wareunits = 1000000  # self.ware_storage_capacity
        target_min = 0
        target_max = 100
        prev_ware = self.ware_units
        self.ware_units = ((self.ware_units-min_wareunits)/(max_wareunits-min_wareunits)*(target_max-target_min)
                           + target_min)
        # Normalize Bloemfontein units
        min_bloemunits = 0
        max_bloemunits = 1000000  # self.bloem_storage_capacity
        target_min = 0
        target_max = 100
        prev_bloem = self.bloem_units
        self.bloem_units = ((self.bloem_units-min_bloemunits)/(max_bloemunits-min_bloemunits)*(target_max - target_min)
                            + target_min)
        # Normalize Durban units
        min_durbunits = 0
        max_durbunits = 1000000  # self.durb_storage_capacity
        target_min = 0
        target_max = 100
        prev_durb = self.durb_units
        self.durb_units = ((self.durb_units-min_durbunits)/(max_durbunits-min_durbunits)*(target_max - target_min)
                           + target_min)
        # Normalize East London units
        min_ELunits = 0
        max_ELunits = 1000000  # self.EL_storage_capacity
        target_min = 0
        target_max = 100
        prev_EL = self.EL_units
        self.EL_units = ((self.EL_units-min_ELunits)/(max_ELunits-min_ELunits)*(target_max - target_min)
                         + target_min)
        # Normalize Pretoria units
        min_pretunits = 0
        max_pretunits = 1000000  # self.pret_storage_capacity
        target_min = 0
        target_max = 100
        prev_pret = self.pret_units
        self.pret_units = ((self.pret_units - min_pretunits)/(max_pretunits-min_pretunits)*(target_max - target_min)
                           + target_min)
        # Normalize Cape Town units
        min_CTunits = 0
        max_CTunits = 1000000  # self.CT_storage_capacity
        target_min = 0
        target_max = 100
        prev_CT = self.CT_units
        self.CT_units = ((self.CT_units - min_CTunits)/(max_CTunits-min_CTunits)*(target_max - target_min)
                         + target_min)

        obs = [self.prod_units, self.ware_units, self.bloem_units, self.durb_units, self.EL_units, self.pret_units,
               self.CT_units, self.choice_bloem, self.choice_durb, self.choice_EL, self.choice_pret, self.choice_CT,
               self.no_of_trucks_prod_ware, self.no_of_trucks_prod_bloem, self.no_of_trucks_ware_bloem,
               self.no_of_trucks_prod_durb, self.no_of_trucks_ware_durb, self.no_of_trucks_prod_EL,
               self.no_of_trucks_ware_EL, self.no_of_trucks_prod_pret, self.no_of_trucks_ware_pret,
               self.no_of_trucks_prod_CT, self.no_of_trucks_ware_CT, self.units_transit_prod_ware,
               self.units_transit_prod_bloem, self.units_transit_ware_bloem, self.units_transit_prod_durb,
               self.units_transit_ware_durb, self.units_transit_prod_EL, self.units_transit_ware_EL,
               self.units_transit_prod_pret, self.units_transit_ware_pret, self.units_transit_prod_CT,
               self.units_transit_ware_CT]
        bloem_forecast = self.total_pred_bloem[self.day - 5:self.day + 5]
        obs.extend(bloem_forecast)
        ware_forecast = self.total_pred_jhb[self.day - 5:self.day + 5]
        obs.extend(ware_forecast)
        durb_forecast = self.total_pred_durb[self.day - 5:self.day + 5]
        obs.extend(durb_forecast)
        EL_forecast = self.total_pred_EL[self.day - 5:self.day + 5]
        obs.extend(EL_forecast)
        pret_forecast = self.total_pred_pret[self.day - 5:self.day + 5]
        obs.extend(pret_forecast)
        CT_forecast = self.total_pred_CT[self.day - 5:self.day + 5]
        obs.extend(CT_forecast)

        self.prod_units = prev_prod
        self.ware_units = prev_ware
        self.bloem_units = prev_bloem
        self.durb_units = prev_durb
        self.EL_units = prev_EL
        self.pret_units = prev_pret
        self.CT_units = prev_CT

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
                     'Training/EL_units': self.training_env.get_attr("EL_units")[0],
                     'Training/pret_units': self.training_env.get_attr("pret_units")[0],
                     'Training/CT_units': self.training_env.get_attr("CT_units")[0],
                     'Training/prod_action': self.training_env.get_attr("prod_action")[0],
                     'Training/ware_action': self.training_env.get_attr("ware_action")[0],
                     'Training/bloem_action': self.training_env.get_attr("bloem_action")[0],
                     'Training/EL_action': self.training_env.get_attr("EL_action")[0],
                     'Training/pret_action': self.training_env.get_attr("pret_action")[0],
                     'Training/CT_action': self.training_env.get_attr("CT_action")[0], }
            for key in stats.keys():
                self.logger.record(key, stats[key])

env = InventoryEnvironment(initial_bloem_units=initial_bloem_units, initial_ware_units=initial_ware_units,
                           initial_prod_units=initial_prod_units, initial_durb_units=initial_durb_units,
                           initial_EL_units=initial_EL_units, initial_pret_units=initial_pret_units,
                           initial_CT_units=initial_CT_units, bloem_storage_capacity=bloem_storage_capacity,
                           ware_storage_capacity=ware_storage_capacity, prod_storage_capacity=prod_storage_capacity,
                           durb_storage_capacity=durb_storage_capacity, EL_storage_capacity=EL_storage_capacity,
                           pret_storage_capacity=pret_storage_capacity, CT_storage_capacity=CT_storage_capacity,
                           large_truck_capacity=large_truck_capacity, small_truck_capacity=small_truck_capacity,
                           df_bloem=df_bloem, df_jhb=df_jhb, df_durb=df_durb, df_EL=df_EL, df_pret=df_pret, df_CT=df_CT,
                           total_pred_bloem=total_pred_bloem, total_pred_jhb=total_pred_jhb,
                           total_pred_durb=total_pred_durb, total_pred_EL=total_pred_EL,
                           total_pred_pret=total_pred_pret, total_pred_CT=total_pred_CT, selling_price=selling_price,
                           bloem_storage_cost=bloem_storage_cost, ware_storage_cost=ware_storage_cost,
                           prod_storage_cost=prod_storage_cost, durb_storage_cost=durb_storage_cost,
                           EL_storage_cost=EL_storage_cost, pret_storage_cost=pret_storage_cost,
                           CT_storage_cost=CT_storage_cost, manufacture_cost=manufacture_cost,
                           distribution_percent_bloem=distribution_percent_bloem, distribution_percent_durb=distribution_percent_durb,
                           distribution_percent_EL=distribution_percent_EL, distribution_percent_pret=distribution_percent_pret,
                           distribution_percent_CT=distribution_percent_CT, production_processing_time=production_processing_time,
                           transport_time_prod_ware=transport_time_prod_ware, transport_time_prodware_Bloem=transport_time_prodware_Bloem,
                           transport_time_prodware_Durb=transport_time_prodware_Durb, transport_time_prodware_EL=transport_time_prodware_EL,
                           transport_time_prodware_Pret=transport_time_prodware_Pret, transport_time_prodware_CT=transport_time_prodware_CT,
                           transport_cost_prod_ware=transport_cost_prod_ware, transport_cost_prodware_Bloem=transport_cost_prodware_Bloem,
                           transport_cost_prodware_Durb=transport_cost_prodware_Durb, transport_cost_prodware_EL=transport_cost_prodware_EL,
                           transport_cost_prod_Pret=transport_cost_prod_Pret, transport_cost_ware_Pret=transport_cost_ware_Pret,
                           transport_cost_prodware_CT=transport_cost_prodware_CT,
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

def make_env(env_class):
    def _init():
        # Pass parameters here when creating an instance
        env = env_class(initial_bloem_units, initial_ware_units, initial_prod_units, initial_durb_units,
                        initial_EL_units, initial_pret_units, initial_CT_units, bloem_storage_capacity,
                        ware_storage_capacity, prod_storage_capacity, durb_storage_capacity, EL_storage_capacity,
                        pret_storage_capacity, CT_storage_capacity, large_truck_capacity, small_truck_capacity,
                        df_bloem, df_jhb, df_durb, df_EL, df_pret, df_CT, total_pred_bloem, total_pred_jhb,
                        total_pred_durb, total_pred_EL, total_pred_pret, total_pred_CT, selling_price,
                        bloem_storage_cost, ware_storage_cost, prod_storage_cost, durb_storage_cost, EL_storage_cost,
                        pret_storage_cost, CT_storage_cost, manufacture_cost, distribution_percent_bloem,
                        distribution_percent_durb, distribution_percent_EL, distribution_percent_pret,
                        distribution_percent_CT, production_processing_time, transport_time_prod_ware,
                        transport_time_prodware_Bloem, transport_time_prodware_Durb, transport_time_prodware_EL,
                        transport_time_prodware_Pret, transport_time_prodware_CT, transport_cost_prod_ware,
                        transport_cost_prodware_Bloem, transport_cost_prodware_Durb, transport_cost_prodware_EL,
                        transport_cost_prod_Pret, transport_cost_ware_Pret, transport_cost_prodware_CT,
                        min_production_limit, max_production_limit)
        return env
    return _init

TIMESTEPS = 1000000
log_freq = 100
n_envs = 8

# PPO model
ent_coef = 0.1  # 0 to 0.01 [0.005, 0.01, 0.05, 0.1] default=0.1
learning_rate = 0.003  # 0.003 to 5e-6 [0.0001, 0.001, 0.003, 0.005, 0.01] default=0.003
clip_range = 0.3  # 0.1,0.2,0.3 [0.1, 0.2] default=0.2
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
        for day in range(347):
            action, _state = loaded_model.predict(obs)
            obs, reward, done, info = env.step(action)
            print(f"Day: {day}")
            print(f"Action: {action}")
            # print(f"Obs: {obs}")
            print(f"Prod units {env.prod_units} Ware units {env.ware_units} Bloem units {env.bloem_units} "
                  f"Durban units {env.durb_units} EL units {env.EL_units} Pret units {env.pret_units} "
                  f"CT units {env.CT_units}")

            tf.summary.scalar('Production/Production units available', env.prod_units, step=day)
            tf.summary.scalar('Production/Action', env.prod_action, step=day)
            tf.summary.scalar('Ware/Warehouse current demand', total_pred_jhb[day], step=day)
            tf.summary.scalar('Ware/Action', env.ware_action, step=day)
            tf.summary.scalar('Ware/Warehouse units available', env.ware_units, step=day)
            tf.summary.scalar('Bloem/Bloem current demand', total_pred_bloem[day], step=day)
            tf.summary.scalar('Bloem/Action', env.bloem_action, step=day)
            tf.summary.scalar('Bloem/Bloem units available', env.bloem_units, step=day)
            tf.summary.scalar('Durb/Durb current demand', total_pred_durb[day], step=day)
            tf.summary.scalar('Durb/Action', env.durb_action, step=day)
            tf.summary.scalar('Durb/Durb units available', env.durb_units, step=day)
            tf.summary.scalar('EL/EL current demand', total_pred_EL[day], step=day)
            tf.summary.scalar('EL/Action', env.EL_action, step=day)
            tf.summary.scalar('EL/EL units available', env.EL_units, step=day)
            tf.summary.scalar('Pret/Pret current demand', total_pred_pret[day], step=day)
            tf.summary.scalar('Pret/Action', env.pret_action, step=day)
            tf.summary.scalar('Pret/Pret units available', env.pret_units, step=day)
            tf.summary.scalar('CT/CT current demand', total_pred_CT[day], step=day)
            tf.summary.scalar('CT/Action', env.CT_action, step=day)
            tf.summary.scalar('CT/CT units available', env.CT_units, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to ware', env.no_of_trucks_prod_ware, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to bloem', env.no_of_trucks_prod_bloem, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation ware to bloem', env.no_of_trucks_ware_bloem, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to durb', env.no_of_trucks_prod_durb, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation ware to durb', env.no_of_trucks_ware_durb, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to EL', env.no_of_trucks_prod_EL, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation ware to EL', env.no_of_trucks_ware_EL, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to pret', env.no_of_trucks_prod_pret, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation ware to pret', env.no_of_trucks_ware_pret, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation prod to CT', env.no_of_trucks_prod_CT, step=day)
            tf.summary.scalar('Trucks/Number of trucks in operation ware to CT', env.no_of_trucks_ware_CT, step=day)
            tf.summary.scalar('Cost/Total manufacturing cost', env.total_manufacture_cost, step=day)
            tf.summary.scalar('Cost/Total delivery cost', env.total_delivery_cost, step=day)
            tf.summary.scalar('Cost/Total storage cost', env.total_storage_cost, step=day)
            tf.summary.scalar('Cost/Overall cost',env.total_delivery_cost + env.total_manufacture_cost + env.total_storage_cost, step=day)
            tf.summary.scalar('Profitability/Revenue', env.revenue_gained, step=day)
            tf.summary.scalar('Profitability/Total cost',env.total_delivery_cost + env.total_manufacture_cost + env.total_storage_cost, step=day)
            tf.summary.scalar('Profitability/Global Net profit', env.net_profit, step=day)
            tf.summary.scalar('Profitability/Ware Net profit', env.net_profit_ware, step=day)
            tf.summary.scalar('Profitability/Bloem Net profit', env.net_profit_bloem, step=day)
            tf.summary.scalar('Profitability/Durb profit', env.net_profit_durb, step=day)
            tf.summary.scalar('Profitability/EL Net profit', env.net_profit_EL, step=day)
            tf.summary.scalar('Profitability/CT Net profit', env.net_profit_CT, step=day)
            tf.summary.scalar('Profitability/Pret Net profit', env.net_profit_pret, step=day)
            tf.summary.scalar('Units/Units satisfied', env.units_satisfied, step=day)
            tf.summary.scalar('Units/Units unsatisfied', env.units_unsatisfied, step=day)
            tf.summary.scalar('Order fulfilment rate/Global fill rate', env.fill_rate, step=day)
            tf.summary.scalar('Order fulfilment rate/ware fill rate', env.fill_rate_ware, step=day)
            tf.summary.scalar('Order fulfilment rate/bloem fill rate', env.fill_rate_bloem, step=day)
            tf.summary.scalar('Order fulfilment rate/durb fill rate', env.fill_rate_durb, step=day)
            tf.summary.scalar('Order fulfilment rate/EL fill rate', env.fill_rate_EL, step=day)
            tf.summary.scalar('Order fulfilment rate/CT fill rate', env.fill_rate_CT, step=day)
            tf.summary.scalar('Order fulfilment rate/pret fill rate', env.fill_rate_pret, step=day)
            tf.summary.scalar('Current/Current revenue', env.current_revenue, step=day)
            tf.summary.scalar('Current/Current cost', env.current_cost, step=day)
            tf.summary.scalar('Current/Current profit', env.current_revenue-env.current_cost, step=day)

            if done:
                print('DONE')
                obs = env.reset()
        print(f"Test net profit: {env.net_profit}")
        print(f"Test fill rate: {env.fill_rate}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
