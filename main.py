import simulation
import pdb
import warnings
import copy
import pickle
#import matplotlib as plt

PRERUNS = 800
POSTRUNS = 400
SIMULACAO = 1

warnings.filterwarnings("ignore")
prerun = simulation.simulation() 

prerun.train(PRERUNS)


baseline    = copy.deepcopy(prerun)
wealth      = copy.deepcopy(prerun)
consumption = copy.deepcopy(prerun)
labor       = copy.deepcopy(prerun)


baseline.train(POSTRUNS)
wealth.train(POSTRUNS, wealthTax = 0.1)
consumption.train(POSTRUNS, consumptionClothTax = 0.5)
labor.train(POSTRUNS, laborClothTax = 0.25, laborFoodTax = 0.10)


lista = [prerun, baseline, wealth, consumption, labor]
for i in range(len(lista)):
    nome = "simulacao_" + str(SIMULACAO) + "_arquivo_" + str(i) + ".pkl"
    file = open(nome, "wb")
    pickle.dump(lista[i], file)
    file.close()

pdb.set_trace()




