import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import tensorflow as tf

# Import demand forecasts
from Demand_forecasts_RFR import (df_bloem, total_pred_bloem, df_jhb, total_pred_jhb, df_durb, total_pred_durb, df_EL,
                                  total_pred_EL, df_CT, total_pred_CT, df_pret, total_pred_pret)

product_lifespan = 60

#Input parameters
production_units = []
production_units.append([5000, product_lifespan])

ware_units = []
ware_units.append([16710, product_lifespan])

bloem_units = []
bloem_units.append([5028, product_lifespan])

durb_units = []
durb_units.append([6333, product_lifespan])

PE_units = []
PE_units.append([1241, product_lifespan])

CT_units = []
CT_units.append([927, product_lifespan])

pret_units = []
pret_units.append([8938, product_lifespan])

min_production_limit = 5000 #units   #reset to 30000
max_production_limit = 150000 #units

production_storage_capacity = 3175200 #units
warehouse_storage_capacity = 14175000
bloem_storage_capacity = 1268400
durb_storage_capacity = 630000
PE_storage_capacity = 987000
CT_storage_capacity = 745500
pret_storage_capacity = 3990000

manufacture_cost = 25000/70/30 #per unit
total_manufacture_cost = 0

production_processing_time = 2 #maturation period in days

produce_vector = []
prod_ware_order_backlog = []

bloem_production_order_backlog = []
durb_production_order_backlog = []
PE_production_order_backlog = []
CT_production_order_backlog = []
pret_production_order_backlog = []

bloem_warehouse_order_backlog = []
durb_warehouse_order_backlog = []
PE_warehouse_order_backlog = []
CT_warehouse_order_backlog = []
pret_warehouse_order_backlog = []

small_truck_capacity = 14000
large_truck_capacity = 58800

transport_cost_prod_ware = 5031.942839
transport_cost_prodware_Bloem = 16116.92355
transport_cost_prodware_Durb = 14042.60801
transport_cost_prodware_PE = 27968.98425
transport_cost_prodware_CT = 32701.39211
transport_cost_prod_Pret = 5031.942839
transport_cost_ware_Pret = 8739.340276
total_delivery_cost = 0

transport_time_prod_ware = 1
transport_time_prodware_Bloem = 2
transport_time_prodware_Durb = 2
transport_time_prodware_PE = 3
transport_time_prodware_CT = 3
transport_time_prodware_Pret = 1

prod_storage_cost = 224.29/7/70/30 #per unit per day
ware_storage_cost = 54.79/7/70/30 #per unit per day
bloem_storage_cost = 65.748/7/70/30 #per unit per day
durb_storage_cost = 65.748/7/70/30 #per unit per day
PE_storage_cost = 65.748/7/70/30 #per unit per day
CT_storage_cost = 65.748/7/70/30 #per unit per day
pret_storage_cost = 65.748/7/70/30 #per unit per day
total_storage_cost = 0

distribution_percent_bloem = 82
distribution_percent_durb = 95
distribution_percent_jhb = 0
distribution_percent_PE = 93
distribution_percent_CT = 25
distribution_percent_pret = 54

units_satisfied = 0
units_unsatisfied = 0
obsolete_inventory = 0
selling_price = 25
revenue_gained = 0

units_moving_prod_ware_vector =[]
units_moving_prodware_bloem_vector = []
units_moving_prodware_durb_vector = []
units_moving_prodware_PE_vector = []
units_moving_prodware_CT_vector = []
units_moving_prodware_pret_vector = []

import shutil
import os

log_dir = "./logs"  # Specify the path to your log directory

# Check if the log directory exists
if os.path.exists(log_dir):
    # Delete the contents of the log directory
    shutil.rmtree(log_dir)
    print("Logs deleted successfully.")
else:
    print("Log directory does not exist.")

# Create a summary writer
log_dir = './logs'
summary_writer = tf.summary.create_file_writer(log_dir)

# Write the summary data for the line graphs
with (((summary_writer.as_default()))):
    for day in range(len(df_bloem) - 9):  # -7 is because you cannot forecast for the last 7 days #-9 to make consistent with RL

        production_units_sum = 0
        for i in range(len(production_units)):
            production_units_sum += production_units[i][0]
        ware_units_sum = 0
        for i in range(len(ware_units)):
            ware_units_sum += ware_units[i][0]
        bloem_units_sum = 0
        for i in range(len(bloem_units)):
            bloem_units_sum += bloem_units[i][0]
        durb_units_sum = 0
        for i in range(len(durb_units)):
            durb_units_sum += durb_units[i][0]
        PE_units_sum = 0
        for i in range(len(PE_units)):
            PE_units_sum += PE_units[i][0]
        CT_units_sum = 0
        for i in range(len(CT_units)):
            CT_units_sum += CT_units[i][0]
        pret_units_sum = 0
        for i in range(len(pret_units)):
            pret_units_sum += pret_units[i][0]

        # Units required at bloem
        bloem_inventory_target = math.ceil(sum(total_pred_bloem[day + 1:day + 8]))
        #s = 0
        #for i in range(len(units_moving_prodware_bloem_vector)):
        #    if units_moving_prodware_bloem_vector[i] != []:
        #        s += units_moving_prodware_bloem_vector[i][0][0]
        bloem_units_required = bloem_inventory_target - bloem_units_sum #- s
        if bloem_units_required < 0:
            bloem_units_required = 0
        print(f'Bloem inventory_target: {bloem_inventory_target}, Bloem units_required: {bloem_units_required}')
        # Units required at durb
        durb_inventory_target = math.ceil(sum(total_pred_durb[day + 1:day + 8]))
        durb_units_required = durb_inventory_target - durb_units_sum
        if durb_units_required < 0:
            durb_units_required = 0
        print(f'Durb inventory_target: {durb_inventory_target}, Durb units_required: {durb_units_required}')
        # Units required at PE
        PE_inventory_target = math.ceil(sum(total_pred_EL[day + 1:day + 8]))
        PE_units_required = PE_inventory_target - PE_units_sum
        if PE_units_required < 0:
            PE_units_required = 0
        print(f'PE inventory_target: {PE_inventory_target}, PE units_required: {PE_units_required}')
        # Units required at CT
        CT_inventory_target = math.ceil(sum(total_pred_CT[day + 1:day + 8]))
        CT_units_required = CT_inventory_target - CT_units_sum
        if CT_units_required < 0:
            CT_units_required = 0
        print(f'CT inventory_target: {CT_inventory_target}, CT units_required: {CT_units_required}')
        # Units required at pret
        pret_inventory_target = math.ceil(sum(total_pred_pret[day + 1:day + 8]))
        pret_units_required = pret_inventory_target - pret_units_sum
        if pret_units_required < 0:
            pret_units_required = 0
        print(f'pret inventory_target: {pret_inventory_target}, pret units_required: {pret_units_required}')
        # Units required at ware
        ware_inventory_target = math.ceil(sum(total_pred_jhb[day + 1:day + 13]))
        ware_units_required = ware_inventory_target - ware_units_sum
        if ware_units_required < 0:
            ware_units_required = 0
        print(f'Ware inventory_target: {ware_inventory_target}, Ware units_required: {ware_units_required}')

        units_to_produce = 0
        # production starts, determine number of units to produce on current day
        units_to_produce += sum(prod_ware_order_backlog) + sum(bloem_production_order_backlog) + sum(durb_production_order_backlog) + sum(PE_production_order_backlog) + sum(CT_production_order_backlog) + sum(pret_production_order_backlog)
        units_to_produce -= production_units_sum
        if units_to_produce < min_production_limit:
            produce = min_production_limit
        elif units_to_produce > max_production_limit:
            produce = max_production_limit
        else:
            produce = units_to_produce
        if production_units_sum + produce > production_storage_capacity:
            produce = production_storage_capacity - production_units_sum
            if produce < min_production_limit:
                produce = 0
            elif produce > max_production_limit:
                produce = max_production_limit

        total_manufacture_cost += produce * manufacture_cost
        produce_vector.append(produce)
        print(f'Units to produce: {produce}, produce_vector: {produce_vector}, total_manufacture_cost: {total_manufacture_cost}')

        # processing time before turning into finished inventory
        if day >= production_processing_time:
            production_units_sum += produce_vector[0]
            production_units.append([produce_vector[0], product_lifespan])
            del produce_vector[0]
        print(f'Production units available: {production_units_sum}, produce_vector: {produce_vector}')

        # send units to warehouse
        ware_units_required += sum(bloem_warehouse_order_backlog) + sum(durb_warehouse_order_backlog) + sum(PE_warehouse_order_backlog) + sum(CT_warehouse_order_backlog) + sum(pret_warehouse_order_backlog)  # - ware_units #can remove this - warehouse units i think
        prod_ware_order_backlog.append(ware_units_required)
        units_moving_prod_ware = []
        no_of_trucks_prod_ware = 0
        ss=0

        for n in range(len(prod_ware_order_backlog)):
            if production_units_sum > prod_ware_order_backlog[n] and prod_ware_order_backlog[n] + ware_units_sum + ss < warehouse_storage_capacity:

                send = prod_ware_order_backlog[n]
                for i in range(len(production_units)):
                    if send > 0:
                        if production_units[i][0] > send:
                            production_units[i][0] -= send
                            ss += send
                            units_moving_prod_ware.append([send, production_units[i][1]])
                            send -= send
                        else:
                            send -= production_units[i][0]
                            ss += production_units[i][0]
                            units_moving_prod_ware.append(production_units[i])
                            production_units[i] = []
                production_units = [x for x in production_units if x != []]
                production_units_sum = 0
                for i in range(len(production_units)):
                    production_units_sum += production_units[i][0]

                prod_ware_order_backlog[n] = 0
            else:
                break
        prod_ware_order_backlog = [i for i in prod_ware_order_backlog if i != 0]

        units_moving_prod_ware_vector.append(units_moving_prod_ware)
        no_of_trucks_prod_ware = math.ceil(ss / small_truck_capacity)
        total_delivery_cost += no_of_trucks_prod_ware * transport_cost_prod_ware

        # delivery time before reaching ware
        if day >= transport_time_prod_ware:
            for i in range(len(units_moving_prod_ware_vector[0])):
                ware_units.append(units_moving_prod_ware_vector[0][i])
            del units_moving_prod_ware_vector[0]
        ware_units_sum = 0
        for i in range(len(ware_units)):
            ware_units_sum += ware_units[i][0]

        print(f'units sent from prod to ware: {units_moving_prod_ware}, production units available: {production_units_sum}, warehouse units available: {ware_units_sum}')
        print(f'no. of trucks_prod_ware: {no_of_trucks_prod_ware}, total delivery cost: {total_delivery_cost}')

        # Determine whether order is from prod or ware
        no_of_trucks_prod_bloem = 0  # can consider moving these to above cell
        no_of_trucks_ware_bloem = 0
        no_of_trucks_prod_durb = 0
        no_of_trucks_ware_durb = 0
        no_of_trucks_prod_PE = 0
        no_of_trucks_ware_PE = 0
        no_of_trucks_prod_CT = 0
        no_of_trucks_ware_CT = 0
        no_of_trucks_prod_pret = 0
        no_of_trucks_ware_pret = 0

        choice = random.choice(range(1, 101))
        # sending units from production to bloem
        if choice > distribution_percent_bloem:
            units_to_produce += bloem_units_required
            bloem_production_order_backlog.append(bloem_units_required)
            units_moving_prod_bloem = []
            ss = 0
            for n in range(len(bloem_production_order_backlog)):
                if production_units_sum > bloem_production_order_backlog[n] and bloem_production_order_backlog[
                    n] + bloem_units_sum + ss < bloem_storage_capacity:  # checking storage capacity of bloem outlet:

                    send = bloem_production_order_backlog[n]
                    for i in range(len(production_units)):
                        if send > 0:
                            if production_units[i][0] > send:
                                production_units[i][0] -= send
                                ss += send
                                units_moving_prod_bloem.append([send, production_units[i][1]])
                                send -= send
                            else:
                                send -= production_units[i][0]
                                ss += production_units[i][0]
                                units_moving_prod_bloem.append(production_units[i])
                                production_units[i] = []
                    production_units = [x for x in production_units if x != []]
                    production_units_sum = 0
                    for i in range(len(production_units)):
                        production_units_sum += production_units[i][0]

                    bloem_production_order_backlog[n] = 0
                else:  # (dont continue satisfying orders)  #bloem_production_order_backlog = [500 200]
                    break
            bloem_production_order_backlog = [i for i in bloem_production_order_backlog if i != 0]

            units_moving_prodware_bloem_vector.append(units_moving_prod_bloem)
            no_of_trucks_prod_bloem = math.ceil(ss / large_truck_capacity)
            total_delivery_cost += no_of_trucks_prod_bloem * transport_cost_prodware_Bloem

            print(f'bloem production order backlog: {bloem_production_order_backlog}, production units available: {production_units_sum}, units moving prodbloem: {units_moving_prod_bloem}, units moving prod bloem vector: {units_moving_prodware_bloem_vector}')
            print(f'no. of trucks_prod_bloem: {no_of_trucks_prod_bloem}, total delivery cost: {total_delivery_cost}')
        # sending units from warehouse to bloem
        else:
            if bloem_units_required > ware_units_sum:
                units_to_produce += bloem_units_required - ware_units_sum
            bloem_warehouse_order_backlog.append(bloem_units_required)
            units_moving_ware_bloem = []
            ss=0
            for n in range(len(bloem_warehouse_order_backlog)):
                if ware_units_sum > bloem_warehouse_order_backlog[n] and bloem_warehouse_order_backlog[
                    n] + bloem_units_sum + ss < bloem_storage_capacity:  # checking storage capacity of bloem outlet

                    send = bloem_warehouse_order_backlog[n]
                    for i in range(len(ware_units)):
                        if send > 0:
                            if ware_units[i][0] > send:
                                ware_units[i][0] -= send
                                ss += send
                                units_moving_ware_bloem.append([send, ware_units[i][1]])
                                send -= send
                            else:
                                send -= ware_units[i][0]
                                ss += ware_units[i][0]
                                units_moving_ware_bloem.append(ware_units[i])
                                ware_units[i] = []
                    ware_units = [x for x in ware_units if x != []]
                    ware_units_sum = 0
                    for i in range(len(ware_units)):
                        ware_units_sum += ware_units[i][0]

                    bloem_warehouse_order_backlog[n] = 0
                else:
                    break
            bloem_warehouse_order_backlog = [i for i in bloem_warehouse_order_backlog if i != 0]

            units_moving_prodware_bloem_vector.append(units_moving_ware_bloem)
            no_of_trucks_ware_bloem = math.ceil(ss / large_truck_capacity)
            total_delivery_cost += no_of_trucks_ware_bloem * transport_cost_prodware_Bloem

            print(f'bloem warehouse order backlog: {bloem_warehouse_order_backlog}, warehouse units available: {ware_units_sum}, units moving warebloem: {units_moving_ware_bloem}, units moving ware bloem vector: {units_moving_prodware_bloem_vector}')
            print(f'no. of trucks_ware_bloem: {no_of_trucks_ware_bloem}, total delivery cost: {total_delivery_cost}')

        print(f'units moving prod ware to bloem vector: {units_moving_prodware_bloem_vector}')

        # sending units from production to durb
        if choice > distribution_percent_durb:
            units_to_produce += durb_units_required
            durb_production_order_backlog.append(durb_units_required)
            units_moving_prod_durb = []
            ss=0
            for n in range(len(durb_production_order_backlog)):
                if production_units_sum > durb_production_order_backlog[n] and durb_production_order_backlog[
                    n] + durb_units_sum + ss < durb_storage_capacity:  # checking storage capacity of durb outlet:

                    send = durb_production_order_backlog[n]
                    for i in range(len(production_units)):
                        if send > 0:
                            if production_units[i][0] > send:
                                production_units[i][0] -= send
                                ss += send
                                units_moving_prod_durb.append([send, production_units[i][1]])
                                send -= send
                            else:
                                send -= production_units[i][0]
                                ss += production_units[i][0]
                                units_moving_prod_durb.append(production_units[i])
                                production_units[i] = []
                    production_units = [x for x in production_units if x != []]
                    production_units_sum = 0
                    for i in range(len(production_units)):
                        production_units_sum += production_units[i][0]

                    durb_production_order_backlog[n] = 0
                else:  # (dont continue satisfying orders)  #durb_production_order_backlog = [500 200]
                    break
            durb_production_order_backlog = [i for i in durb_production_order_backlog if i != 0]

            units_moving_prodware_durb_vector.append(units_moving_prod_durb)
            no_of_trucks_prod_durb = math.ceil(ss / large_truck_capacity)
            total_delivery_cost += no_of_trucks_prod_durb * transport_cost_prodware_Durb

            print(f'durb production order backlog: {durb_production_order_backlog}, production units available: {production_units_sum}, units moving prod durb: {units_moving_prod_durb}, units moving prod durb vector: {units_moving_prodware_durb_vector}')
            print(f'no. of trucks_prod_durb: {no_of_trucks_prod_durb}, total delivery cost: {total_delivery_cost}')
        # sending units from warehouse to durb
        else:
            if durb_units_required > ware_units_sum:
                units_to_produce += durb_units_required - ware_units_sum
            durb_warehouse_order_backlog.append(durb_units_required)
            units_moving_ware_durb = []
            ss=0
            for n in range(len(durb_warehouse_order_backlog)):
                if ware_units_sum > durb_warehouse_order_backlog[n] and durb_warehouse_order_backlog[
                    n] + durb_units_sum + ss < durb_storage_capacity:  # checking storage capacity of durb outlet

                    send = durb_warehouse_order_backlog[n]
                    for i in range(len(ware_units)):
                        if send > 0:
                            if ware_units[i][0] > send:
                                ware_units[i][0] -= send
                                ss += send
                                units_moving_ware_durb.append([send, ware_units[i][1]])
                                send -= send
                            else:
                                send -= ware_units[i][0]
                                ss += ware_units[i][0]
                                units_moving_ware_durb.append(ware_units[i])
                                ware_units[i] = []
                    ware_units = [x for x in ware_units if x != []]
                    ware_units_sum = 0
                    for i in range(len(ware_units)):
                        ware_units_sum += ware_units[i][0]

                    durb_warehouse_order_backlog[n] = 0
                else:
                    break
            durb_warehouse_order_backlog = [i for i in durb_warehouse_order_backlog if i != 0]

            units_moving_prodware_durb_vector.append(units_moving_ware_durb)
            no_of_trucks_ware_durb = math.ceil(ss / large_truck_capacity)
            total_delivery_cost += no_of_trucks_ware_durb * transport_cost_prodware_Durb

            print(f'durb warehouse order backlog: {durb_warehouse_order_backlog}, warehouse units available: {ware_units_sum}, units moving ware durb: {units_moving_ware_durb}, units moving ware durb vector: {units_moving_prodware_durb_vector}')
            print(f'no. of trucks_ware_durb: {no_of_trucks_ware_durb}, total delivery cost: {total_delivery_cost}')

        # sending units from production to PE
        if choice > distribution_percent_PE:
            units_to_produce += PE_units_required
            PE_production_order_backlog.append(PE_units_required)
            units_moving_prod_PE = []
            ss=0
            for n in range(len(PE_production_order_backlog)):
                if production_units_sum > PE_production_order_backlog[n] and PE_production_order_backlog[
                    n] + PE_units_sum + ss < PE_storage_capacity:  # checking storage capacity of PE outlet:

                    send = PE_production_order_backlog[n]
                    for i in range(len(production_units)):
                        if send > 0:
                            if production_units[i][0] > send:
                                production_units[i][0] -= send
                                ss += send
                                units_moving_prod_PE.append([send, production_units[i][1]])
                                send -= send
                            else:
                                send -= production_units[i][0]
                                ss += production_units[i][0]
                                units_moving_prod_PE.append(production_units[i])
                                production_units[i] = []
                    production_units = [x for x in production_units if x != []]
                    production_units_sum = 0
                    for i in range(len(production_units)):
                        production_units_sum += production_units[i][0]

                    PE_production_order_backlog[n] = 0
                else:  # (dont continue satisfying orders)  #PE_production_order_backlog = [500 200]
                    break
            PE_production_order_backlog = [i for i in PE_production_order_backlog if i != 0]

            units_moving_prodware_PE_vector.append(units_moving_prod_PE)
            no_of_trucks_prod_PE = math.ceil(ss / large_truck_capacity)
            total_delivery_cost += no_of_trucks_prod_PE * transport_cost_prodware_PE

            print(f'PE production order backlog: {PE_production_order_backlog}, production units available: {production_units_sum}, units moving prod PE: {units_moving_prod_PE}, units moving prod PE vector: {units_moving_prodware_PE_vector}')
            print(f'no. of trucks_prod_PE: {no_of_trucks_prod_PE}, total delivery cost: {total_delivery_cost}')
        # sending units from warehouse to PE
        else:
            if PE_units_required > ware_units_sum:
                units_to_produce += PE_units_required - ware_units_sum
            PE_warehouse_order_backlog.append(PE_units_required)
            units_moving_ware_PE = []
            ss=0
            for n in range(len(PE_warehouse_order_backlog)):
                if ware_units_sum > PE_warehouse_order_backlog[n] and PE_warehouse_order_backlog[
                    n] + PE_units_sum + ss < PE_storage_capacity:  # checking storage capacity of PE outlet

                    send = PE_warehouse_order_backlog[n]
                    for i in range(len(ware_units)):
                        if send > 0:
                            if ware_units[i][0] > send:
                                ware_units[i][0] -= send
                                ss += send
                                units_moving_ware_PE.append([send, ware_units[i][1]])
                                send -= send
                            else:
                                send -= ware_units[i][0]
                                ss += ware_units[i][0]
                                units_moving_ware_PE.append(ware_units[i])
                                ware_units[i] = []
                    ware_units = [x for x in ware_units if x != []]
                    ware_units_sum = 0
                    for i in range(len(ware_units)):
                        ware_units_sum += ware_units[i][0]

                    PE_warehouse_order_backlog[n] = 0
                else:
                    break
            PE_warehouse_order_backlog = [i for i in PE_warehouse_order_backlog if i != 0]

            units_moving_prodware_PE_vector.append(units_moving_ware_PE)
            no_of_trucks_ware_PE = math.ceil(ss / large_truck_capacity)
            total_delivery_cost += no_of_trucks_ware_PE * transport_cost_prodware_PE

            print(f'PE warehouse order backlog: {PE_warehouse_order_backlog}, warehouse units available: {ware_units_sum}, units moving ware PE: {units_moving_ware_PE}, units moving ware PE vector: {units_moving_prodware_PE_vector}')
            print(f'no. of trucks_ware_PE: {no_of_trucks_ware_PE}, total delivery cost: {total_delivery_cost}')

        # sending units from production to CT
        if choice > distribution_percent_CT:
            units_to_produce += CT_units_required
            CT_production_order_backlog.append(CT_units_required)
            units_moving_prod_CT = []
            ss= 0
            for n in range(len(CT_production_order_backlog)):
                if production_units_sum > CT_production_order_backlog[n] and CT_production_order_backlog[
                    n] + CT_units_sum + ss < CT_storage_capacity:  # checking storage capacity of CT outlet:

                    send = CT_production_order_backlog[n]
                    for i in range(len(production_units)):
                        if send > 0:
                            if production_units[i][0] > send:
                                production_units[i][0] -= send
                                ss += send
                                units_moving_prod_CT.append([send, production_units[i][1]])
                                send -= send
                            else:
                                send -= production_units[i][0]
                                ss += production_units[i][0]
                                units_moving_prod_CT.append(production_units[i])
                                production_units[i] = []
                    production_units = [x for x in production_units if x != []]
                    production_units_sum = 0
                    for i in range(len(production_units)):
                        production_units_sum += production_units[i][0]

                    CT_production_order_backlog[n] = 0
                else:  # (dont continue satisfying orders)  #CT_production_order_backlog = [500 200]
                    break
            CT_production_order_backlog = [i for i in CT_production_order_backlog if i != 0]

            units_moving_prodware_CT_vector.append(units_moving_prod_CT)
            no_of_trucks_prod_CT = math.ceil(ss / large_truck_capacity)
            total_delivery_cost += no_of_trucks_prod_CT * transport_cost_prodware_CT

            print(f'CT production order backlog: {CT_production_order_backlog}, production units available: {production_units_sum}, units moving prod CT: {units_moving_prod_CT}, units moving prod CT vector: {units_moving_prodware_CT_vector}')
            print(f'no. of trucks_prod_CT: {no_of_trucks_prod_CT}, total delivery cost: {total_delivery_cost}')
        # sending units from warehouse to CT
        else:
            if CT_units_required > ware_units_sum:
                units_to_produce += CT_units_required - ware_units_sum
            CT_warehouse_order_backlog.append(CT_units_required)
            units_moving_ware_CT = []
            ss=0
            for n in range(len(CT_warehouse_order_backlog)):
                if ware_units_sum > CT_warehouse_order_backlog[n] and CT_warehouse_order_backlog[
                    n] + CT_units_sum + ss < CT_storage_capacity:  # checking storage capacity of CT outlet

                    send = CT_warehouse_order_backlog[n]
                    for i in range(len(ware_units)):
                        if send > 0:
                            if ware_units[i][0] > send:
                                ware_units[i][0] -= send
                                ss += send
                                units_moving_ware_CT.append([send, ware_units[i][1]])
                                send -= send
                            else:
                                send -= ware_units[i][0]
                                ss += ware_units[i][0]
                                units_moving_ware_CT.append(ware_units[i])
                                ware_units[i] = []
                    ware_units = [x for x in ware_units if x != []]
                    ware_units_sum = 0
                    for i in range(len(ware_units)):
                        ware_units_sum += ware_units[i][0]

                    CT_warehouse_order_backlog[n] = 0
                else:
                    break
            CT_warehouse_order_backlog = [i for i in CT_warehouse_order_backlog if i != 0]

            units_moving_prodware_CT_vector.append(units_moving_ware_CT)
            no_of_trucks_ware_CT = math.ceil(ss / large_truck_capacity)
            total_delivery_cost += no_of_trucks_ware_CT * transport_cost_prodware_CT

            print(f'CT warehouse order backlog: {CT_warehouse_order_backlog}, warehouse units available: {ware_units_sum}, units moving ware CT: {units_moving_ware_CT}, units moving ware CT vector: {units_moving_prodware_CT_vector}')
            print(f'no. of trucks_ware_CT: {no_of_trucks_ware_CT}, total delivery cost: {total_delivery_cost}')

        # sending units from production to pret
        if choice > distribution_percent_pret:
            units_to_produce += pret_units_required
            pret_production_order_backlog.append(pret_units_required)
            units_moving_prod_pret = []
            ss=0
            for n in range(len(pret_production_order_backlog)):
                if production_units_sum > pret_production_order_backlog[n] and pret_production_order_backlog[
                    n] + pret_units_sum + ss < pret_storage_capacity:  # checking storage capacity of pret outlet:

                    send = pret_production_order_backlog[n]
                    for i in range(len(production_units)):
                        if send > 0:
                            if production_units[i][0] > send:
                                production_units[i][0] -= send
                                ss += send
                                units_moving_prod_pret.append([send, production_units[i][1]])
                                send -= send
                            else:
                                send -= production_units[i][0]
                                ss += production_units[i][0]
                                units_moving_prod_pret.append(production_units[i])
                                production_units[i] = []
                    production_units = [x for x in production_units if x != []]
                    production_units_sum = 0
                    for i in range(len(production_units)):
                        production_units_sum += production_units[i][0]

                    pret_production_order_backlog[n] = 0
                else:  # (dont continue satisfying orders)  #pret_production_order_backlog = [500 200]
                    break
            pret_production_order_backlog = [i for i in pret_production_order_backlog if i != 0]

            units_moving_prodware_pret_vector.append(units_moving_prod_pret)
            no_of_trucks_prod_pret = math.ceil(ss / large_truck_capacity)
            total_delivery_cost += no_of_trucks_prod_pret * transport_cost_prod_Pret

            print(f'pret production order backlog: {pret_production_order_backlog}, production units available: {production_units_sum}, units moving prod pret: {units_moving_prod_pret}, units moving prod pret vector: {units_moving_prodware_pret_vector}')
            print(f'no. of trucks_prod_pret: {no_of_trucks_prod_pret}, total delivery cost: {total_delivery_cost}')
        # sending units from warehouse to pret
        else:
            if pret_units_required > ware_units_sum:
                units_to_produce += pret_units_required - ware_units_sum
            pret_warehouse_order_backlog.append(pret_units_required)
            units_moving_ware_pret = []
            ss=0
            for n in range(len(pret_warehouse_order_backlog)):
                if ware_units_sum > pret_warehouse_order_backlog[n] and pret_warehouse_order_backlog[
                    n] + pret_units_sum + ss < pret_storage_capacity:  # checking storage capacity of pret outlet

                    send = pret_warehouse_order_backlog[n]
                    for i in range(len(ware_units)):
                        if send > 0:
                            if ware_units[i][0] > send:
                                ware_units[i][0] -= send
                                ss += send
                                units_moving_ware_pret.append([send, ware_units[i][1]])
                                send -= send
                            else:
                                send -= ware_units[i][0]
                                ss += ware_units[i][0]
                                units_moving_ware_pret.append(ware_units[i])
                                ware_units[i] = []
                    ware_units = [x for x in ware_units if x != []]
                    ware_units_sum = 0
                    for i in range(len(ware_units)):
                        ware_units_sum += ware_units[i][0]

                    pret_warehouse_order_backlog[n] = 0
                else:
                    break
            pret_warehouse_order_backlog = [i for i in pret_warehouse_order_backlog if i != 0]

            units_moving_prodware_pret_vector.append(units_moving_ware_pret)
            no_of_trucks_ware_pret = math.ceil(ss / large_truck_capacity)
            total_delivery_cost += no_of_trucks_ware_pret * transport_cost_ware_Pret

            print(f'pret warehouse order backlog: {pret_warehouse_order_backlog}, warehouse units available: {ware_units_sum}, units moving ware pret: {units_moving_ware_pret}, units moving ware pret vector: {units_moving_prodware_pret_vector}')
            print(f'no. of trucks_ware_pret: {no_of_trucks_ware_pret}, total delivery cost: {total_delivery_cost}')

        # delivery time before reaching bloem
        if day >= transport_time_prodware_Bloem:
            for i in range(len(units_moving_prodware_bloem_vector[0])):
                bloem_units.append(units_moving_prodware_bloem_vector[0][i])
            del units_moving_prodware_bloem_vector[0]
        bloem_units_sum = 0
        for i in range(len(bloem_units)):
            bloem_units_sum += bloem_units[i][0]
        print(f'bloem units: {bloem_units}')
        print(f'bloem units: {bloem_units_sum}')

        # delivery time before reaching durb
        if day >= transport_time_prodware_Durb:
            for i in range(len(units_moving_prodware_durb_vector[0])):
                durb_units.append(units_moving_prodware_durb_vector[0][i])
            del units_moving_prodware_durb_vector[0]
        durb_units_sum = 0
        for i in range(len(durb_units)):
            durb_units_sum += durb_units[i][0]
        print(f'durb units: {durb_units}')
        print(f'durb units: {durb_units_sum}')

        # delivery time before reaching PE
        if day >= transport_time_prodware_PE:
            for i in range(len(units_moving_prodware_PE_vector[0])):
                PE_units.append(units_moving_prodware_PE_vector[0][i])
            del units_moving_prodware_PE_vector[0]
        PE_units_sum = 0
        for i in range(len(PE_units)):
            PE_units_sum += PE_units[i][0]
        print(f'PE units: {PE_units}')
        print(f'PE units: {PE_units_sum}')

        # delivery time before reaching CT
        if day >= transport_time_prodware_CT:
            for i in range(len(units_moving_prodware_CT_vector[0])):
                CT_units.append(units_moving_prodware_CT_vector[0][i])
            del units_moving_prodware_CT_vector[0]
        CT_units_sum = 0
        for i in range(len(CT_units)):
            CT_units_sum += CT_units[i][0]
        print(f'CT units: {CT_units}')
        print(f'CT units: {CT_units_sum}')

        # delivery time before reaching pret
        if day >= transport_time_prodware_Pret:
            for i in range(len(units_moving_prodware_pret_vector[0])):
                pret_units.append(units_moving_prodware_pret_vector[0][i])
            del units_moving_prodware_pret_vector[0]
        pret_units_sum = 0
        for i in range(len(pret_units)):
            pret_units_sum += pret_units[i][0]
        print(f'pret units: {pret_units}')
        print(f'pret units: {pret_units_sum}')

        # units after replenishment
        print(f'bloem units: {bloem_units_sum}, durb units: {durb_units_sum}, PE units: {PE_units_sum}, CT units: {CT_units_sum}, pret inits: {pret_units_sum}')

        # serve customer at bloem
        print(f'*Day: {day}, Bloem units: {bloem_units_sum}, Bloem current demand: {df_bloem[day]}')
        if bloem_units_sum > df_bloem[day]:

            send = df_bloem[day]
            for i in range(len(bloem_units)):
                if send > 0:
                    if bloem_units[i][0] > send:
                        bloem_units[i][0] -= send
                        send -= send
                    else:
                        send -= bloem_units[i][0]
                        bloem_units[i] = []
            bloem_units = [x for x in bloem_units if x != []]
            bloem_units_sum = 0
            for i in range(len(bloem_units)):
                bloem_units_sum += bloem_units[i][0]

            units_satisfied += df_bloem[day]
            revenue_gained += df_bloem[day] * 25
        else:
            units_satisfied += bloem_units_sum
            units_unsatisfied += df_bloem[day] - bloem_units_sum
            revenue_gained += bloem_units_sum * 25
            bloem_units_sum = 0
            bloem_units = []
        # serve customer at durb
        print(f'Durb units: {durb_units_sum}, Durb current demand: {df_durb[day]}')
        if durb_units_sum > df_durb[day]:

            send = df_durb[day]
            for i in range(len(durb_units)):
                if send > 0:
                    if durb_units[i][0] > send:
                        durb_units[i][0] -= send
                        send -= send
                    else:
                        send -= durb_units[i][0]
                        durb_units[i] = []
            durb_units = [x for x in durb_units if x != []]
            durb_units_sum = 0
            for i in range(len(durb_units)):
                durb_units_sum += durb_units[i][0]

            units_satisfied += df_durb[day]
            revenue_gained += df_durb[day] * 25
        else:
            units_satisfied += durb_units_sum
            units_unsatisfied += df_durb[day] - durb_units_sum
            revenue_gained += durb_units_sum * 25
            durb_units_sum = 0
            durb_units = []
        # serve customer at PE
        print(f'PE units: {PE_units_sum}, PE current demand: {df_EL[day]}')
        if PE_units_sum > df_EL[day]:

            send = df_EL[day]
            for i in range(len(PE_units)):
                if send > 0:
                    if PE_units[i][0] > send:
                        PE_units[i][0] -= send
                        send -= send
                    else:
                        send -= PE_units[i][0]
                        PE_units[i] = []
            PE_units = [x for x in PE_units if x != []]
            PE_units_sum = 0
            for i in range(len(PE_units)):
                PE_units_sum += PE_units[i][0]

            units_satisfied += df_EL[day]
            revenue_gained += df_EL[day] * 25
        else:
            units_satisfied += PE_units_sum
            units_unsatisfied += df_EL[day] - PE_units_sum
            revenue_gained += PE_units_sum * 25
            PE_units_sum = 0
            PE_units = []
        # serve customer at CT
        print(f'CT units: {CT_units_sum}, CT current demand: {df_CT[day]}')
        if CT_units_sum > df_CT[day]:

            send = df_CT[day]
            for i in range(len(CT_units)):
                if send > 0:
                    if CT_units[i][0] > send:
                        CT_units[i][0] -= send
                        send -= send
                    else:
                        send -= CT_units[i][0]
                        CT_units[i] = []
            CT_units = [x for x in CT_units if x != []]
            CT_units_sum = 0
            for i in range(len(CT_units)):
                CT_units_sum += CT_units[i][0]

            units_satisfied += df_CT[day]
            revenue_gained += df_CT[day] * 25
        else:
            units_satisfied += CT_units_sum
            units_unsatisfied += df_CT[day] - CT_units_sum
            revenue_gained += CT_units_sum * 25
            CT_units_sum = 0
            CT_units = []
        # serve customer at pret
        print(f'pret units: {pret_units_sum}, pret current demand: {df_pret[day]}')
        if pret_units_sum > df_pret[day]:

            send = df_pret[day]
            for i in range(len(pret_units)):
                if send > 0:
                    if pret_units[i][0] > send:
                        pret_units[i][0] -= send
                        send -= send
                    else:
                        send -= pret_units[i][0]
                        pret_units[i] = []
            pret_units = [x for x in pret_units if x != []]
            pret_units_sum = 0
            for i in range(len(pret_units)):
                pret_units_sum += pret_units[i][0]

            units_satisfied += df_pret[day]
            revenue_gained += df_pret[day] * 25
        else:
            units_satisfied += pret_units_sum
            units_unsatisfied += df_pret[day] - pret_units_sum
            revenue_gained += pret_units_sum * 25
            pret_units_sum = 0
            pret_units = []
        # serve customer at ware
        print(f'Ware units: {ware_units_sum}, Ware current demand: {df_jhb[day]}')
        if ware_units_sum > df_jhb[day]:

            send = df_jhb[day]
            for i in range(len(ware_units)):
                if send > 0:
                    if ware_units[i][0] > send:
                        ware_units[i][0] -= send
                        send -= send
                    else:
                        send -= ware_units[i][0]
                        ware_units[i] = []
            ware_units = [x for x in ware_units if x != []]
            ware_units_sum = 0
            for i in range(len(ware_units)):
                ware_units_sum += ware_units[i][0]

            units_satisfied += df_jhb[day]
            revenue_gained += df_jhb[day] * 25
        else:
            units_satisfied += ware_units_sum
            units_unsatisfied += df_jhb[day] - ware_units_sum
            revenue_gained += ware_units_sum * 25
            ware_units_sum = 0
            ware_units = []
        # units available after sale
        print(f'Bloem units: {bloem_units_sum}, Durb units: {durb_units_sum}, PE units: {PE_units_sum}, CT units: {CT_units_sum}, Pret units: {pret_units_sum}, Ware units: {ware_units_sum}, units satisfied: {units_satisfied}, units unsatisfied: {units_unsatisfied}, revenue gained: {revenue_gained}')

        # reduce product lifespan by 1 and check for obsolete
        for i in range(len(production_units)):
            if production_units[i][1] == 0:
                units_unsatisfied += production_units[i][0]
                obsolete_inventory += production_units[i][0]
                production_units[i] = []
            else:
                production_units[i][1] -= 1
        production_units = [x for x in production_units if x != []]
        print(f'production units {production_units}')
        production_units_sum = 0
        for i in range(len(production_units)):
            production_units_sum += production_units[i][0]

        for i in range(len(ware_units)):
            if ware_units[i][1] == 0:
                units_unsatisfied += ware_units[i][0]
                obsolete_inventory += ware_units[i][0]
                ware_units[i] = []
            else:
                ware_units[i][1] -= 1
        ware_units = [x for x in ware_units if x != []]
        print(f'ware units {ware_units}')
        ware_units_sum = 0
        for i in range(len(ware_units)):
            ware_units_sum += ware_units[i][0]

        for i in range(len(bloem_units)):
            if bloem_units[i][1] == 0:
                units_unsatisfied += bloem_units[i][0]
                obsolete_inventory += bloem_units[i][0]
                bloem_units[i] = []
            else:
                bloem_units[i][1] -= 1
        bloem_units = [x for x in bloem_units if x != []]
        print(f'bloem units {bloem_units}')
        bloem_units_sum = 0
        for i in range(len(bloem_units)):
            bloem_units_sum += bloem_units[i][0]

        for i in range(len(durb_units)):
            if durb_units[i][1] == 0:
                units_unsatisfied += durb_units[i][0]
                obsolete_inventory += durb_units[i][0]
                durb_units[i] = []
            else:
                durb_units[i][1] -= 1
        durb_units = [x for x in durb_units if x != []]
        print(f'durb units {durb_units}')
        durb_units_sum = 0
        for i in range(len(durb_units)):
            durb_units_sum += durb_units[i][0]

        for i in range(len(PE_units)):
            if PE_units[i][1] == 0:
                units_unsatisfied += PE_units[i][0]
                obsolete_inventory += PE_units[i][0]
                PE_units[i] = []
            else:
                PE_units[i][1] -= 1
        PE_units = [x for x in PE_units if x != []]
        print(f'PE units {PE_units}')
        PE_units_sum = 0
        for i in range(len(PE_units)):
            PE_units_sum += PE_units[i][0]

        for i in range(len(CT_units)):
            if CT_units[i][1] == 0:
                units_unsatisfied += CT_units[i][0]
                obsolete_inventory += CT_units[i][0]
                CT_units[i] = []
            else:
                CT_units[i][1] -= 1
        CT_units = [x for x in CT_units if x != []]
        print(f'CT units {CT_units}')
        CT_units_sum = 0
        for i in range(len(CT_units)):
            CT_units_sum += CT_units[i][0]

        for i in range(len(pret_units)):
            if pret_units[i][1] == 0:
                units_unsatisfied += pret_units[i][0]
                obsolete_inventory += pret_units[i][0]
                pret_units[i] = []
            else:
                pret_units[i][1] -= 1
        pret_units = [x for x in pret_units if x != []]
        print(f'pret units {pret_units}')
        pret_units_sum = 0
        for i in range(len(pret_units)):
            pret_units_sum += pret_units[i][0]

        print(f'Obsolete inventory {obsolete_inventory}')

        # storage costs for remaining inventory
        total_storage_cost += production_units_sum * prod_storage_cost
        total_storage_cost += ware_units_sum * ware_storage_cost
        total_storage_cost += bloem_units_sum * bloem_storage_cost
        total_storage_cost += durb_units_sum * durb_storage_cost
        total_storage_cost += PE_units_sum * PE_storage_cost
        total_storage_cost += CT_units_sum * CT_storage_cost
        total_storage_cost += pret_units_sum * pret_storage_cost
        print(f'Total storage cost: {total_storage_cost}')

        # net profit and fill rate
        net_profit = abs(revenue_gained) - total_manufacture_cost - total_storage_cost - total_delivery_cost
        fill_rate = (units_satisfied / (units_satisfied + units_unsatisfied)) * 100
        print(f'net profit: {net_profit}, fill rate: {fill_rate}')

        print(f'FINAL bloem units available: {bloem_units_sum}, FINAL durb units available: {durb_units_sum}, FINAL PE units available: {PE_units_sum}, FINAL CT units available: {CT_units_sum}, FINAL pret units available: {pret_units_sum}, FINAL ware units available: {ware_units_sum}, FINAL prod units available: {production_units_sum}')
        # Create graphs in tensorboard
        tf.summary.scalar('Production/Production units available', production_units_sum, step=day)
        tf.summary.scalar('Production/Units to produce', produce, step=day)

        tf.summary.scalar('Ware/Warehouse current demand', df_jhb[day], step=day)
        tf.summary.scalar('Ware/Warehouse inventory target', ware_inventory_target, step=day)
        tf.summary.scalar('Ware/Warehouse units required', ware_units_required, step=day)
        tf.summary.scalar('Ware/Warehouse units available', ware_units_sum, step=day)

        tf.summary.scalar('Bloem/Bloem current demand', df_bloem[day], step=day)
        tf.summary.scalar('Bloem/Bloem inventory target', bloem_inventory_target, step=day)
        tf.summary.scalar('Bloem/Bloem units required', bloem_units_required, step=day)
        tf.summary.scalar('Bloem/Bloem units available', bloem_units_sum, step=day)

        tf.summary.scalar('Durb/Durb current demand', df_durb[day], step=day)
        tf.summary.scalar('Durb/Durb inventory target', durb_inventory_target, step=day)
        tf.summary.scalar('Durb/Durb units required', durb_units_required, step=day)
        tf.summary.scalar('Durb/Durb units available', durb_units_sum, step=day)

        tf.summary.scalar('PE/PE current demand', df_EL[day], step=day)
        tf.summary.scalar('PE/PE inventory target', PE_inventory_target, step=day)
        tf.summary.scalar('PE/PE units required', PE_units_required, step=day)
        tf.summary.scalar('PE/PE units available', PE_units_sum, step=day)

        tf.summary.scalar('CT/CT current demand', df_CT[day], step=day)
        tf.summary.scalar('CT/CT inventory target', CT_inventory_target, step=day)
        tf.summary.scalar('CT/CT units required', CT_units_required, step=day)
        tf.summary.scalar('CT/CT units available', CT_units_sum, step=day)

        tf.summary.scalar('pret/pret current demand', df_pret[day], step=day)
        tf.summary.scalar('pret/pret inventory target', pret_inventory_target, step=day)
        tf.summary.scalar('pret/pret units required', pret_units_required, step=day)
        tf.summary.scalar('pret/pret units available', pret_units_sum, step=day)

        tf.summary.scalar('Trucks/Number of trucks in operation prod to ware', no_of_trucks_prod_ware, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation prod to bloem', no_of_trucks_prod_bloem, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation ware to bloem', no_of_trucks_ware_bloem, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation prod to durb', no_of_trucks_prod_durb, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation ware to durb', no_of_trucks_ware_durb, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation prod to PE', no_of_trucks_prod_PE, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation ware to PE', no_of_trucks_ware_PE, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation prod to CT', no_of_trucks_prod_CT, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation ware to CT', no_of_trucks_ware_CT, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation prod to pret', no_of_trucks_prod_pret, step=day)
        tf.summary.scalar('Trucks/Number of trucks in operation ware to pret', no_of_trucks_ware_pret, step=day)

        tf.summary.scalar('Cost/Total manufacturing cost', total_manufacture_cost, step=day)
        tf.summary.scalar('Cost/Total delivery cost', total_delivery_cost, step=day)
        tf.summary.scalar('Cost/Total storage cost', total_storage_cost, step=day)
        tf.summary.scalar('Cost/Overall cost', total_manufacture_cost + total_delivery_cost + total_storage_cost,
                          step=day)

        tf.summary.scalar('Profitability/Revenue', revenue_gained, step=day)
        tf.summary.scalar('Profitability/Total cost', total_manufacture_cost + total_delivery_cost + total_storage_cost,
                          step=day)
        tf.summary.scalar('Profitability/Net profit', net_profit, step=day)

        tf.summary.scalar('Units/Units satisfied', units_satisfied, step=day)
        tf.summary.scalar('Units/Units unsatisfied', units_unsatisfied, step=day)
        tf.summary.scalar('Units/Order fulfilment rate', fill_rate, step=day)
        tf.summary.scalar('Units/Obsolete inventory', obsolete_inventory, step=day)
# Flush and close the summary writer to clear the logs
summary_writer.flush()
summary_writer.close()
# Launch TensorBoard from cmd
# C:\Users\ARNAV GARG\My Python Stuff\Inventory management using AI
# tensorboard --logdir=./logs
