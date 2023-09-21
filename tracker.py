import numpy as np
import copy

class cohortTracker():

    def __init__(self, cohortId, cohortSize) -> None:
        self.cohortID = cohortId
        self.actionsTaken = 0
        self.actionDistribution = [0,0,0,0,0,0,0,0,0]
        self.cohortSize = cohortSize
        self.sampleAgent = None
        
        self.utilities = [[] for _ in range(cohortSize)]
        self.individualActions = [[] for _ in range(cohortSize)]
    
    def update(self, action, agent):
        loc = self._getCohortLocation(agent)
        loc = loc[1]

        self.utilities[loc].append(agent.acumulatedUtil)
        self.individualActions[loc].append(action)
        self.actionDistribution[action] += 1
        self.actionsTaken += 1

        if self.sampleAgent == None: self.sampleAgent = copy.copy(agent)
    

    def _getCohortLocation(self, agent):
        cohort = (agent.rg // self.cohortSize) -1
        relativeLocation = agent.rg % self.cohortSize
        return (cohort, relativeLocation)
    
    def report(self):
        utilidadesFinais = [inner_list[-1] for inner_list in self.utilities]
        media = np.mean(utilidadesFinais)
        maior = np.max(utilidadesFinais)
        menor = min(utilidadesFinais)
        print(f"Cohort {self.cohortID}")
        print(f"Ações tomadas:{self.actionsTaken}")
        print(f"Função Utilidades: {self.sampleAgent.utilName}")
        print(f"Utilidades: {utilidadesFinais}")
        print(f"Média das utilidades: {media}")
        print(f"Maior das utilidades: {maior}")
        print(f"Menor das utilidades: {menor}")
        print(f"Distribuição das ações:")
        print(f"Nada/Lazer/Trabalhar: {self.actionDistribution[0]}")
        print(f"Mover para cima: {self.actionDistribution[1]}")
        print(f"Mover para baixo: {self.actionDistribution[2]}")
        print(f"Mover para esquerda: {self.actionDistribution[3]}")
        print(f"Mover para direita: {self.actionDistribution[4]}")
        print(f"Trocar Comida por Tecido: {self.actionDistribution[5]}")
        print(f"Trocar Tecido por Comida: {self.actionDistribution[6]}")
        print(f"Consumir: {self.actionDistribution[7]}")
        print(f"Investir: {self.actionDistribution[8]}")


class episodeTracker():

    def __init__(self, episodeId, geography, nCohorts, cohortSize) -> None:
        self.episodeId = episodeId + 1
        self.maxUtility = None
        self.actionsTaken = 0
        self.actionDistribution = [0,0,0,0,0,0,0,0,0]
        self.geography = geography
        self.foodMap = geography.food
        self.clothMap = geography.cloth
        self.cTrackers = [cohortTracker(i, cohortSize) for i in range(nCohorts)]
        self.cohortSize = cohortSize
        self.nCohorts = nCohorts

    def update(self, agent, action, mapInfo):
        self.actionsTaken += 1
        self.actionDistribution[action] += 1

        if self.maxUtility == None: self.maxUtility = agent.acumulatedUtil
        if agent.acumulatedUtil > self.maxUtility: self.maxUtility = agent.acumulatedUtil

        cohort = self._getCohortLocation(agent)
        cohort = cohort[0]


        self.cTrackers[cohort].update(action, agent)


    def report(self):
        print(f"Episódio: {self.episodeId}")
        print(f"Utilidade Máxima: {self.maxUtility}")
        print(f"Total de ações: {self.actionsTaken}")
        print(f"Distribuição das ações:")
        print(f"Trabalhar: {self.actionDistribution[0]}")
        print(f"Mover para cima: {self.actionDistribution[1]}")
        print(f"Mover para baixo: {self.actionDistribution[2]}")
        print(f"Mover para esquerda: {self.actionDistribution[3]}")
        print(f"Mover para direita: {self.actionDistribution[4]}")
        print(f"Trocar Comida por Tecido: {self.actionDistribution[5]}")
        print(f"Trocar Tecido por Comida: {self.actionDistribution[6]}")
        print(f"Consumir: {self.actionDistribution[7]}")
        print(f"Nada: {self.actionDistribution[8]}")


    def _getCohortLocation(self, agent):
        cohort = ((agent.rg -1) // self.cohortSize)
        relativeLocation = (agent.rg-1) % self.cohortSize
        return (cohort, relativeLocation)

    def showMaps(self):
        print("Mapa de comida:")
        print(self.foodMap)

        print("Mapa de tecido:")
        print(self.clothMap)

        print(f"Recursos localizados em: {self.geography.resourceLocations}")

