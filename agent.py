import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical

class Cohort():

    def __init__(self, population = 5):
        self.population = population
    


class Agent():

    def __init__(self, rg, center, initialFood, initialCloth, productivities, utilFun):
        
        self.alive = True 
        self.rg = rg
        self.age = 0
        self.x = center
        self.y = center
        self.center = center
        self.food = initialFood
        self.initialFood = initialFood
        self.cloth = initialCloth
        self.initialCloth = initialCloth
        self.productivity = productivities 
        self.wrongChoicePunishment = -1
        self.deathPunishment = -10
        
        self.storagedFood = 0
        self.storagedCloth = 0
        
        self.acumulatedUtil = 0

        self.actionsTaken = []
        
        self.utilName = utilFun
        self.util = self._getUtilFun(utilFun)


    def _getUtilFun(self, utilFun):
        # TODO dar algum jeito de normalizar isso
        if utilFun == "cobb douglas":
            return self.cobbDouglas
        elif utilFun == "quasilinear":
            return self.quasiLinear
        else:
            return self.cobbDouglas
    
    def getState(self):
        temp = [self.x, self.y, self.food, self.cloth, self.productivity[0], self.productivity[1]]
        return temp


    def reset(self):
        """"
        Reseta o agente ao seu estado inicial para reinicar uma simulação
        """
        self.food = self.initialFood
        self.cloth = self.initialCloth
        self.alive = True
        self.age = 0
        self.x = self.center
        self.y = self.center
        self.productivity = [10,10]
        self.storagedCloth = 0
        self.storagedFood = 0
        self.acumulatedUtil = 0
    

    def collect(self, tile, foodLaborTax = 0, clothLaborTax = 0):

        """
        Coleta recursos do ambiente

        Parametros:
        * tile():
        * foodLaborTax():
        * clothLaborTax():

        Retorna:
        
        0) foodTaxed(float): comida taxada
        1) clothTaxed(float): tecido taxado
        
        """

        self.exhaustion = 0 #TODO adicionar exaustão
        good = tile[1]
        capital = tile[2]
        foodTaxed = 0
        clothTaxed = 0

        if good == 0:
            self.food += capital * self.productivity[0] * (1-foodLaborTax)
            foodTaxed += capital * self.productivity[0] * foodLaborTax
        elif good == 1:
            self.cloth += capital * self.productivity[1] * (1-clothLaborTax)
            clothTaxed += capital * self.productivity[1] * clothLaborTax
        
        return foodTaxed, clothTaxed


    def predict(self, mapInfo, marketInfo, agentInfo, model, positionalInfo):
        return model(mapInfo, marketInfo, agentInfo, positionalInfo)


    def update(self, action, newMapInfo, consumedCloth = 0, wealthTax = None, foodConsumptionTax = 0, clothConsumptionTax = 0):
        
        self.food -= 1
        utilReward = 2
        t1foodTax = 0
        t1clothTax = 0

        self.actionsTaken.append(action) #

        if wealthTax != None:
            t1foodTax = self.food - (self.food * (1 - wealthTax))
            self.food = self.food * (1 - wealthTax)
            t1clothTax = self.cloth -  (self.cloth * (1 - wealthTax))
            self.cloth = self.cloth * (1 - wealthTax)

        #Checando se continuará vivo
        if self.food <= 0:
            self.alive = False
            utilReward = self.deathPunishment

        utilReward += self.util(1*(1-foodConsumptionTax), consumedCloth*(1-clothConsumptionTax)) 
        self.acumulatedUtil += utilReward

        foodTaxed = t1foodTax + (1 * foodConsumptionTax)
        clothTaxed = t1clothTax + (consumedCloth * clothConsumptionTax)

        return self.getState(), utilReward, foodTaxed, clothTaxed
    
    def cobbDouglas(self, foodConsumed, clothConsumed, alpha=0.5, beta=0.5, gamma = 20):
        #TODO arranjar algum jeito de normalizar isso aqui
        temp = (foodConsumed ** (alpha)) * ((gamma *clothConsumed) ** (beta))
        return temp
    
    def quasiLinear(self, foodConsumed, clothConsumed, alpha = 0.5, beta = 0.5, gamma = 20):
        #TODO arranjar algum jeito de normalizar isso aqui
        temp = foodConsumed + np.log((gamma * clothConsumed)+1)
        return temp



class ActorCritic(nn.Module):

    def __init__(self, mapSize, marketSize, agentInfoSize, actionsPossible, positionalInfo, mapOutputSize = None) -> None:
        super().__init__()

        if mapOutputSize == None: mapOutputSize = 3 #TODO mudar isso depois

        self.mapLayer = nn.Sequential(nn.Conv2d(3, mapOutputSize, kernel_size=2)
                                      )

        sharedInputSize = (9*mapOutputSize) + marketSize + agentInfoSize + positionalInfo

        self.sharedLayers = nn.Sequential(nn.Linear(sharedInputSize, 512),
                                          nn.ReLU(),
                                          nn.Linear(512, 512),
                                          nn.ReLU(),
                                          nn.Linear(512,256),
                                          nn.ReLU(),
                                          #nn.LSTM(256, 256), #TODO tem que arrumar a LSTM
                                          nn.Linear(256,64),
                                          nn.Softmax()
                                          )

        self.policyLayers = nn.Sequential(nn.Linear(64,64),
                                          nn.ReLU(),
                                          nn.Linear(64, actionsPossible),
                                          nn.Softmax())
        
        self.marketLayers = nn.Sequential(nn.Linear(64,8),
                                         nn.Linear(8,2),
                                         nn.ReLU()) #Ultimo tem que ser ReLU pra não ter possibilidade de ser menor que zero

        self.valueLayers= nn.Sequential(nn.Linear(64,32), 
                                        nn.Linear(32,1))
    

    def marketNN(self, mapinfo, marketinfo, agentinfo, positionalInfo):
        # Passamos o mapa pela rede convulucional
        x = self.mapLayer(mapinfo)


        # Juntamos com os dados de mercado e do agente
        x = torch.flatten(x,start_dim=1)
        
        inputs = torch.concat([x, marketinfo, agentinfo, positionalInfo], dim = 1)

        # Passamos pelas camadas compartilhadas
        z = self.sharedLayers(inputs)

        # Camada final individual
        market = self.marketLayers(z)

        return market    
    
    
    def policyNN(self, mapinfo, marketinfo, agentinfo, positionalInfo):
        # Passamos o mapa pela rede convulucional
        x = self.mapLayer(mapinfo)


        # Juntamos com os dados de mercado e do agente
        x = torch.flatten(x,start_dim=1)
        
        inputs = torch.concat([x, marketinfo, agentinfo, positionalInfo], dim = 1)

        # Passamos pelas camadas compartilhadas
        z = self.sharedLayers(inputs)

        # Camada final individual
        policy = self.policyLayers(z)

        return policy
    
    def valuesNN(self, mapinfo, marketinfo, agentinfo, positionalInfo):
        # Passamos o mapa pela rede convulucional
        x = self.mapLayer(mapinfo)

        # Juntamos com os dados de mercado e do agente
        x = torch.flatten(x,start_dim=1)
        
        inputs = torch.concat([x, marketinfo, agentinfo, positionalInfo], dim = 1)

        # Passamos pelas camadas compartilhadas
        z = self.sharedLayers(inputs)

        # Camada final individual
        values = self.valueLayers(z)

        return values

    def forward(self, mapinfo, marketinfo, agentinfo, positionalInfo):

        #inputs
        #mapinfo: atualmente 3x4x4=48, mas é pra ser 3x5x5=75
        #agentInfo: 6
        #maketInfo: 5
        #total = 6+5+48 = 59

        # Passamos o mapa pela rede convulucional


        mapinfo = torch.tensor(mapinfo, dtype=torch.float32)
        marketinfo = torch.tensor(marketinfo, dtype=torch.float32)
        agentinfo = torch.tensor(agentinfo, dtype=torch.float32)
        positionalInfo = torch.tensor(positionalInfo, dtype=torch.float32)
        

        x = self.mapLayer(mapinfo)

        # Juntamos com os dados de mercado e do agente
 
        x = torch.flatten(x)
        inputs = torch.concat([x, marketinfo, agentinfo, positionalInfo])

        # Passamos pelas camadas compartilhadas
        z = self.sharedLayers(inputs)

        # Camadas finais individuais para cada elemento
        policy = self.policyLayers(z)
        values = self.valueLayers(z)
        market = self.marketLayers(z)

        return policy, values, market
    

class PPOTrainer:

    def __init__(self, actorCritic, clipValue, targetDivergence, maxIterations, valueTrainIterations, policyLR, valueLR):
        self.actorCritic = actorCritic
        self.clipValue = clipValue
        self.targetDivergence = targetDivergence
        self.maxIterations = maxIterations
        self.valueTrainIterations = valueTrainIterations
        self.policyLR = policyLR
        self.valueLR = valueLR

        policyParams = list(self.actorCritic.mapLayer.parameters()) + list(self.actorCritic.sharedLayers.parameters()) + list(self.actorCritic.policyLayers.parameters())
        self.policyOptimizer = optim.Adam(policyParams, lr = self.policyLR)

        valueParams = list(self.actorCritic.mapLayer.parameters()) + list(self.actorCritic.sharedLayers.parameters()) + list(self.actorCritic.valueLayers.parameters())
        self.valueOptimizer = optim.Adam(valueParams, lr  = self.valueLR)

        marketParams = list(self.actorCritic.mapLayer.parameters()) + list(self.actorCritic.sharedLayers.parameters()) + list(self.actorCritic.marketLayers.parameters())
        self.marketOptimizer = optim.Adam(marketParams, lr  = self.valueLR)

    def rebaseParams(self):

        """
        Isso aqui é para garantir que está otimizando as coisas certas depois do deepcopy. Possivelmente não precisa, mas é pra garantir.
        """

        policyParams = list(self.actorCritic.mapLayer.parameters()) + list(self.actorCritic.sharedLayers.parameters()) + list(self.actorCritic.policyLayers.parameters())
        self.policyOptimizer = optim.Adam(policyParams, lr = self.policyLR)

        valueParams = list(self.actorCritic.mapLayer.parameters()) + list(self.actorCritic.sharedLayers.parameters()) + list(self.actorCritic.valueLayers.parameters())
        self.valueOptimizer = optim.Adam(valueParams, lr  = self.valueLR)

        marketParams = list(self.actorCritic.mapLayer.parameters()) + list(self.actorCritic.sharedLayers.parameters()) + list(self.actorCritic.marketLayers.parameters())
        self.marketOptimizer = optim.Adam(marketParams, lr  = self.valueLR)


    def trainPolicy(self, mapInfo, marketInfo, agentInfo, actions, oldLogProbs, gaes, positionalInfo):

        for i in range(self.maxIterations):
            self.policyOptimizer.zero_grad()

            newProbabilities = self.actorCritic.policyNN(mapInfo, marketInfo, agentInfo, positionalInfo)
            newProbabilities = Categorical(logits= newProbabilities)
            entropy = newProbabilities.entropy().mean()
            newLogProbs = newProbabilities.log_prob(actions)

            policyRatio = torch.exp(newLogProbs - oldLogProbs)
            clippedRatio = policyRatio.clamp(1- self.clipValue, 1+self.clipValue)

            clippedLoss = clippedRatio * gaes
            actualLoss = policyRatio * gaes

            loss = -torch.min(clippedLoss, actualLoss).mean() - 0.01 * entropy

            loss.backward()
            self.policyOptimizer.step()

    def trainValue(self, mapInfo, marketInfo, agentInfo, returns, positionalInfo):
        for i in range(self.valueTrainIterations):
            self.valueOptimizer.zero_grad()

            values = self.actorCritic.valuesNN(mapInfo, marketInfo, agentInfo, positionalInfo)
            loss = ((returns - values) ** 2).mean()

            loss.backward()
            self.valueOptimizer.step()


    def trainMarket(self, mapInfo, marketInfo, agentInfo, positionalInfo):
        for i in range(self.valueTrainIterations):
            self.marketOptimizer.zero_grad()
            minPrice = marketInfo[4]

            values = self.actorCritic.marketNN(mapInfo, marketInfo, agentInfo, positionalInfo)
            loss2 = ((minPrice - values) ** 2)
            loss = torch.min(loss2).mean()

            loss.backward()
            self.marketOptimizer.step()


