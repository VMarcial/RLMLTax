# Arquivo "excecutável" do projeto
# Faz a seguinte ordem de ações
# Roda uma pre-simulação de PRERUNS rodadas
# Copia a pre-simulação e roda em paralelo os choques e um baseline

import simulation
import pdb
import warnings
import copy
import pickle
import torch


PRERUNS = 1000  # Épocas da présimulação
POSTRUNS = 1200 # Épocas da simulação pós intervenção
SIMULACAO = 100 # Número do teste, para nomear o arquivo


# Ignorar um aviso do numpy
warnings.filterwarnings("ignore")

# Criamos o objeto da simulação
prerun = simulation.simulation() 

# Rodamos a pré-simulação
prerun.train(PRERUNS)

# Salvamos as redes neurais
# Isto ta aqui por garantia, em tese o deepcopy ja faria isso
torch.save(prerun.cohortModels[0].state_dict(), "tempNN0")
torch.save(prerun.cohortModels[1].state_dict(), "tempNN1")

# Copiamos a présimulação em outros objetos
baseline    = copy.deepcopy(prerun)
wealth      = copy.deepcopy(prerun)
consumption = copy.deepcopy(prerun)
labor       = copy.deepcopy(prerun)

# Em tese o deepcopy, copia perfeitamente as redes neurais.
# Entretanto isto está aqui para garantir que a rede foi apropriadamente copiada
# e que as referencias internas de maximização estão corretas
baseline.rebaseParameters(neuralnets= ["tempNN0", "tempNN1"])
wealth.rebaseParameters(neuralnets = ["tempNN0", "tempNN1"])
consumption.rebaseParameters(neuralnets = ["tempNN0", "tempNN1"])
labor.rebaseParameters(neuralnets = ["tempNN0", "tempNN1"])

# Realizamos as simulações das intervenções e do baseline
baseline.train(POSTRUNS)
wealth.train(POSTRUNS, wealthTax = 0.05)
consumption.train(POSTRUNS, consumptionClothTax = 0.5)
labor.train(POSTRUNS, laborClothTax = 0.25, laborFoodTax = 0.10)

lista = [prerun, baseline, wealth, consumption, labor]

# Salvamos os objetos em pkl para que possam ser acessados posteriormente
for i in range(len(lista)):
    nome = "Final_" + str(SIMULACAO) + "_arquivo_" + str(i) + ".pkl"
    file = open(nome, "wb")
    pickle.dump(lista[i], file)
    file.close()

# Abre o python interativo
pdb.set_trace()



