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
    
    def evolve():
        pass


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
        self.productivity = productivities #TODO tornar variavel
        self.wrongChoicePunishment = -1
        self.deathPunishment = -10
        
        self.storagedFood = 0
        self.storagedCloth = 0
        
        self.acumulatedUtil = 0

        self.actionsTaken = []
        
        self.utilName = utilFun
        self.util = self._getUtilFun(utilFun)


    def _getUtilFun(self, utilFun):
        if utilFun == "cobb douglas":
            return self.cobbDouglas
        elif utilFun == "quasilinear":
            return self.quasiLinear
        else:
            return self.cobbDouglas
    
    def getState(self):
        temp = [self.x, self.y, self.food, self.cloth, self.storagedCloth, self.storagedFood]
        return temp

    def reset(self):
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
    
    def _randomName(self):
        pass


    def _randomUtilFun(self):
        pass


    def getAction(self):
        pass


    def movement(self):
        pass


    def consume(self):
        pass


    def collect(self, tile, foodLaborTax = 0, clothLaborTax = 0):
        self.exhaustion = 0 #TODO adicionar exaustão
        good = tile[1]
        capital = tile[2]

        if good == 0:
            self.food += capital * self.productivity[0] * (1-foodLaborTax)
        elif good == 1:
            self.cloth += capital * self.productivity[1] * (1-clothLaborTax)


    def educate(self):
        pass


    def invest(self):
        pass



    def predict(self, mapInfo, marketInfo, agentInfo, model, positionalInfo):
        return model(mapInfo, marketInfo, agentInfo, positionalInfo)


    def update(self, action, newMapInfo, consumedCloth = 0, wealthTax = None, foodConsumptionTax = 0, clothConsumptionTax = 0):
        
        self.food -= 1
        utilReward = 2

        self.actionsTaken.append(action)

        if wealthTax != None:
            self.food = self.food * (1 - wealthTax)

        #Checando se continuará vivo
        if self.food <= 0:
            self.alive = False
            utilReward = self.deathPunishment

        utilReward += self.util(1*(1-foodConsumptionTax), consumedCloth*(1-clothConsumptionTax)) 
        self.acumulatedUtil += utilReward

        return self.getState(), utilReward
    
    def cobbDouglas(self, foodConsumed, clothConsumed, alpha=0.5, beta=0.5):

        temp = (foodConsumed ** (alpha)) * (clothConsumed ** (beta))
        return temp
    
    def quasiLinear(self, foodConsumed, clothConsumed, alpha = 0.5, beta = 0.5):
        temp = foodConsumed + np.log(clothConsumed+1)
        return temp



class ActorCritic(nn.Module):

    def __init__(self, mapSize, marketSize, agentInfoSize, actionsPossible, positionalInfo, mapOutputSize = None) -> None:
        super().__init__()

        if mapOutputSize == None: mapOutputSize = 3 #TODO mudar isso depois

        self.mapLayer = nn.Sequential(nn.Conv2d(3, mapOutputSize, kernel_size=2)
                                      )
        #TODO verificar a convulução de 3 imagens

        sharedInputSize = (9*mapOutputSize) + marketSize + agentInfoSize + positionalInfo

        self.sharedLayers = nn.Sequential(nn.Linear(sharedInputSize, 256),
                                          nn.Tanh(),
                                          nn.Linear(256, 256),
                                          nn.Tanh(),
                                          #nn.LSTM(256, 256),
                                          nn.Linear(256,64),
                                          nn.Tanh()
                                          )

        self.policyLayers = nn.Sequential(nn.Linear(64,64),
                                          nn.Linear(64, actionsPossible),
                                          nn.Tanh())
        
        self.marketLayers = nn.Sequential(nn.Linear(64,8),
                                         nn.Linear(8,2))

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

        marketParams = list(self.actorCritic.mapLayer.parameters()) + list(self.actorCritic.sharedLayers.parameters()) + list(self.actorCritic.valueLayers.parameters())
        self.marketOptimizer = optim.Adam(marketParams, lr  = self.valueLR)


    def trainPolicy(self, mapInfo, marketInfo, agentInfo, actions, oldLogProbs, gaes, positionalInfo):

        for i in range(self.maxIterations):
            self.policyOptimizer.zero_grad()

            newProbabilities = self.actorCritic.policyNN(mapInfo, marketInfo, agentInfo, positionalInfo)
            newProbabilities = Categorical(logits= newProbabilities)
            newLogProbs = newProbabilities.log_prob(actions)

            policyRatio = torch.exp(newLogProbs - oldLogProbs)
            clippedRatio = policyRatio.clamp(1- self.clipValue, 1+self.clipValue)

            clippedLoss = clippedRatio * gaes
            actualLoss = policyRatio * gaes

            loss = -torch.min(clippedLoss, actualLoss).mean()

            loss.backward()
            self.policyOptimizer.step()

            divergence = (oldLogProbs - newLogProbs).mean()
            if divergence >= self.targetDivergence: break

    def trainValue(self, mapInfo, marketInfo, agentInfo, returns, positionalInfo):
        for i in range(self.valueTrainIterations):
            self.valueOptimizer.zero_grad()

            values = self.actorCritic.valuesNN(mapInfo, marketInfo, agentInfo, positionalInfo)
            loss = ((returns - values) ** 2).mean()

            loss.backward()
            self.valueOptimizer.step()


    def trainMarket(self, mapInfo, marketInfo, agentInfo, maxPrice, minPrice, positionalInfo):
        for i in range(self.valueTrainIterations):
            self.marketOptimizer.zero_grad()

            values = self.actorCritic.marketNN(mapInfo, marketInfo, agentInfo, positionalInfo)
            loss1 = ((maxPrice - values) ** 2)
            loss2 = ((minPrice - values) ** 2)
            loss = torch.min(loss1, loss2).mean()

            loss.backward()
            self.markerOptimizer.step()


