import simulation
import pdb
import warnings
import copy
import pickle
import torch
#import matplotlib as plt

PRERUNS = 1000
POSTRUNS = 1200
SIMULACAO = 100

warnings.filterwarnings("ignore")
prerun = simulation.simulation() 

prerun.train(PRERUNS)

torch.save(prerun.cohortModels[0].state_dict(), "tempNN0")
torch.save(prerun.cohortModels[1].state_dict(), "tempNN1")

#baseline    = copy.deepcopy(prerun)
#wealth      = copy.deepcopy(prerun)
#consumption = copy.deepcopy(prerun)
#labor       = copy.deepcopy(prerun)
#
#baseline.rebaseParameters(neuralnets= ["tempNN0", "tempNN1"])
#wealth.rebaseParameters(neuralnets = ["tempNN0", "tempNN1"])
#consumption.rebaseParameters(neuralnets = ["tempNN0", "tempNN1"])
#labor.rebaseParameters(neuralnets = ["tempNN0", "tempNN1"])

#prerun.train(POSTRUNS)
#baseline.train(POSTRUNS)
#wealth.train(POSTRUNS, wealthTax = 0.05)
#consumption.train(POSTRUNS, consumptionClothTax = 0.5)
#labor.train(POSTRUNS, laborClothTax = 0.25, laborFoodTax = 0.10)

#lista = [prerun, baseline, wealth, consumption, labor]
lista = [prerun]

for i in range(len(lista)):
    nome = "Final_" + str(SIMULACAO) + "_arquivo_" + str(i) + ".pkl"
    file = open(nome, "wb")
    pickle.dump(lista[i], file)
    file.close()

pdb.set_trace()



