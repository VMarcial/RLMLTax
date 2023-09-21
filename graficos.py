import simulation
import pdb
import warnings
import copy
import pickle
import pandas as pd
import matplotlib

FILENAME = "Final_100_arquivo_0.pkl"
OUTPUT = "DadosFinalFinalFinal"

file = open(FILENAME, "rb")

sim = pickle.load(file)

for j in range(2):
    utilidadesFinais = []
    utilidades = []
    trabalho = []
    consumo = []
    trocas = []
    dist0 = []
    dist1 = []
    dist2 = []
    dist3 = []
    dist4 = []
    dist5 = []
    dist6 = []
    dist7 = []
    dist8 = []
    agente1 = []
    agente2 = []
    agente3 = []
    agente4 = []
    agente5 = []
    vida1 = []
    vida2 = []
    vida3 = []
    vida4 = []
    vida5 = []

    for i in range(len(sim.epsTrackers)):
        temp = sim.epsTrackers[i].cTrackers[j]
        agente1.append(temp.utilities[0][-1])
        agente2.append(temp.utilities[1][-1])
        agente3.append(temp.utilities[2][-1])
        agente4.append(temp.utilities[3][-1])
        agente5.append(temp.utilities[4][-1])
        vida1.append(len(temp.individualActions[0]))
        vida2.append(len(temp.individualActions[1]))
        vida3.append(len(temp.individualActions[2]))
        vida4.append(len(temp.individualActions[3]))
        vida5.append(len(temp.individualActions[4]))
        dist0.append(temp.actionDistribution[0])
        dist1.append(temp.actionDistribution[1])
        dist2.append(temp.actionDistribution[2])
        dist3.append(temp.actionDistribution[3])
        dist4.append(temp.actionDistribution[4])
        dist5.append(temp.actionDistribution[5])
        dist6.append(temp.actionDistribution[6])
        dist7.append(temp.actionDistribution[7])
        dist8.append(temp.actionDistribution[8])
    
    df = pd.DataFrame([agente1, agente2, agente3, agente4, agente5, vida1, vida2, vida3, vida4, vida5, dist0, dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8]).transpose()
    df.columns = ["Utilidade Agente 1","Utilidade Agente 2","Utilidade Agente 3","Utilidade Agente 4","Utilidade Agente 5",
                  "Vida Agente 1","Vida Agente 2","Vida Agente 3","Vida Agente 4","Vida Agente 5",
                  "Trabalhar", "Cima", "Baixo", "Esquerda", "Direita", "ComidaPorTecido", "TecidoporComida", "Consumir", "Nada"]
    nome  = OUTPUT + "_cohort_" + str(j) + ".xlsx"
    df.to_excel(nome)


file.close()

