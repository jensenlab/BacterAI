import csv
import random

import model

import scipy.stats as sp
from tqdm import trange 

if __name__ == "__main__":
    
    m = model.load_cobra("models/iSMUv01_CDM_LOO_v2.xml")
    with open('CDM_leave_out_training.csv', mode='a') as file:
        writer = csv.writer(file, delimiter=',')
        for _ in trange(100000):
            # n = random.randint(0, len(model.KO_RXN_IDS))
            n = sp.poisson.rvs(5)
            grow, reactions = model.knockout_and_simulate(m, n)
            reactions = list(reactions)
            reactions.append(grow)
            writer.writerow(reactions)
            # print(grow, reactions)
        
        for _ in trange(100000):
            n = random.randint(0, len(model.KO_RXN_IDS))
            grow, reactions = model.knockout_and_simulate(m, n)
            reactions = list(reactions)
            reactions.append(grow)
            writer.writerow(reactions)
            # print(grow, reactions)