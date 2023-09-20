import random
import enviroment
import numpy as np
import agent
import torch
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import tracker

class simulation():

    def __init__(self, load =  False, gaesGamma = 0.998, gaesDecay = 0.97, nCohorts = 2, cohortSize = 5, trainers = None):
        self.population = 0
        self.gaesGamma = gaesGamma
        self.gaesDecay = gaesDecay
        self.cohortSize = cohortSize
        self.nCohorts = nCohorts
        self.device = "cpu"

        self.epsTrackers= []

        self.ambient = enviroment.enviroment()
        self.cohorts = []
        self._generateCohorts(self.nCohorts, self.cohortSize, self.ambient.geography.center, [10,20], [1,3], [[10,10], [5,5]], ["cobb douglas", "quasilinear"])
        self.cohortModels = []
        self._getModels()
        self.cohortTrainer = []
        self._getTrainers()
    

    def _getRG(self):
        self.population += 1
        return self.population


    def _getCohortLocation(self, rg):
        cohort = (rg // self.cohortSize) - 1
        relativeLocation = rg % self.cohortSize
        return (cohort, relativeLocation)


    def _generateCohorts(self, quantity, cohortSize, center , initialFood, initialCloth, productivities, utilities):
        #intialFood lista de ints
        #initialCloth lista de ints
        #productivities lista de lista de ints[[x,y],[a,b]]
        #utilFun lista de strings
        
        for i in range(quantity):
            temp = []
            for k in range(cohortSize): 
                rg = self._getRG()
                temp.append(agent.Agent(rg = rg, 
                                        center =  center,
                                        initialFood = initialFood[i],
                                        initialCloth = initialCloth[i],
                                        productivities = productivities[i],
                                        utilFun = utilities[i]))
            self.cohorts.append(temp)


    def _getModels(self):
        visionSize      = self.ambient.geography.visionInput
        marketSize      = self.ambient.market.infoSize
        agentInfoSize   = 6 #TODO colocar isso como a200utoomatico
        actionsPossible = 9
        positionalInfoSize = 2
        for i in range(self.cohortSize): 
            model = agent.ActorCritic(visionSize, marketSize, agentInfoSize, actionsPossible, positionalInfoSize)
            self.cohortModels.append(model)

    
    def _resetAgents(self):
        for coh in self.cohorts:
            for agent in coh:
                agent.reset()
    

    def _resetEnviroment(self):
        self.ambient = enviroment.enviroment()


    def _getTrainers(self, trainers = None):
        if trainers == None:
            for i in range(self.nCohorts):
                self.cohortTrainer.append(agent.PPOTrainer(self.cohortModels[i], clipValue= 0.2, targetDivergence= 0.01, maxIterations= 50, valueTrainIterations= 50, policyLR=3e-4, valueLR=1e-4))
    

    def _calculateGaes(self, rewards, values):

        next_values = np.concatenate([values[1:], [0]]) 
        deltas = [rew + self.gaesGamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

        gaes = [deltas[-1]]
        for i in reversed(range(len(deltas)-1)):
            gaes.append(deltas[i] + self.gaesDecay * self.gaesGamma * gaes[-1])

        return gaes[::-1]


    def _runEpisode(self, maxTurns = 100, consumptionFoodTax = 0, consumptionClothTax = 0, wealthTax = 0, laborClothTax = 0, laborFoodTax = 0):

        
        done = False
        agentOrder = [(x,y) for x in range(len(self.cohorts)) for y in range(len(self.cohorts[0]))]
        cohortData = [[], [], [], [], [], [], [], [], []] #mapInfo, marketInfo, agentInfo, action, reward, values, logProbs, cohort
        trainingData = []
        for i in range(len(self.cohorts)): trainingData.append(cohortData)
        self._resetAgents()
        self._resetEnviroment()
        epTracker = tracker.episodeTracker(len(self.epsTrackers), self.ambient.geography, self.nCohorts, self.cohortSize)
        progressbar = tqdm(range(maxTurns), leave= False, desc = f"running turn 0")
        marketorderBuffer = []

        for eps in progressbar:
            random.shuffle(agentOrder)
            for i in agentOrder:
                # Selecionando o agente da vez
                x = i[0]
                y = i[1]
                agent = self.cohorts[x][y]

                marketorderBuffer = self.ambient.market.clear_market()  #, eps
                
                for transaction in marketorderBuffer:
                    t1 = transaction[0]
                    t2 = transaction[1]
                    loc = self._getCohortLocation(t1.agent)
                    if t1.request_item == "food":
                        self.cohorts[loc[0]][loc[1]].food += t1[2]
                    elif t1.request_item == "cloth":
                        self.cohorts[loc[0]][loc[1]].cloth += t1[2]
                    
                    loc = self._getCohortLocation(t2.agent)
                    if t2.request_item == "food":
                        self.cohorts[loc[0]][loc[1]].food += t2[2]
                    elif t2.request_item == "cloth":
                        self.cohorts[loc[0]][loc[1]].cloth += t2[2]
                    

                if agent.alive == True:

                    # Pegando as informações publicas e privadas
                    agentInfo = agent.getState()
                    mapInfo, marketInfo, positionalInfo = self.ambient.getInfo(agent.x, agent.y)
                    model = self.cohortModels[x]
                    
                    # Prevendo a ação do agente
                    logits, val, market = agent.predict(mapInfo, marketInfo, agentInfo, model, positionalInfo)
                    price = market[0].item()
                    quantity = market[1].item()
                    distribution = Categorical(logits = logits)
                    prediction = distribution.sample()
                    logProb = distribution.log_prob(prediction).item()
                    prediction = prediction.item()
                    val = val.item()
                    
                    # Checamos se o agente pode fazer a ação com seus recursos
                    action = prediction
                    
                    # Caso possa, ele faz a ação no ambiente, caso contrario não faz nada

                    newMapinfo,newMarketInfo, newAgentInfo, reward, done = self.ambient.step(agent, action, mapInfo, marketInfo, agentInfo, consumptionFoodTax, consumptionClothTax, wealthTax, laborFoodTax, laborClothTax, eps, price, quantity)
                    
                    
                    # Atualizamos com os ganhos e perdas desta etapa
                    epTracker.update(agent, action, newMapinfo)

                    # Preparação dos dados para utilização no treinamento


                    for i, item in enumerate((newMapinfo, newMarketInfo, newAgentInfo, action, reward, val, logProb, positionalInfo, x)):
                        try:
                            trainingData[x][i].append(item)
                        except:
                            np.append(trainingData[x][i], item)
                    

                    
                    #TODO adicionar o tracker aqui
                    
            for x in range(2): 
                trainingData[x][5] = self._calculateGaes(trainingData[x][4], trainingData[x][5])
            
            progressbar.set_description(f"running turn {eps+1}")
        return trainingData, epTracker
    

    def train(self, epochs, consumptionFoodTax = 0, consumptionClothTax = 0, wealthTax = 0, laborClothTax = 0, laborFoodTax = 0):

        episodesRewards = []
        progressbar = tqdm(range(epochs), leave = True, desc = f"running epoch 0")
        for eps in progressbar:
            trainingData, temp = self._runEpisode( consumptionFoodTax = consumptionFoodTax, consumptionClothTax = consumptionClothTax, wealthTax = wealthTax, laborClothTax = laborClothTax, laborFoodTax = laborFoodTax)
            self.epsTrackers.append(temp)
            for i in tqdm(range(len(self.cohorts)), desc = f"training neural networks", leave = False):
                
                tdata = trainingData[i]

                # Aleatorizamos a ordem dos dados para reduzir overfitting
                permutation = np.random.permutation(len(tdata[0])).tolist()


                # Transformamos no tipo necessário para passar pelo ML

                #TODO colocar a randomização
                #temp = list(zip(tdata[0], tdata[1], tdata[2], tdata[3], tdata[4], tdata[5], tdata[6]))
                #temp = []
                #random.shuffle(temp)

                mapInfo, marketInfo, agentInfo, actions, returns, gaes, logProbs,positionalInfo = tdata[0], tdata[1], tdata[2], tdata[3], tdata[4], tdata[5], tdata[6], tdata[7]

                mapInfo    = torch.tensor(mapInfo, dtype=torch.float32, device=self.device)
                marketInfo = torch.tensor(marketInfo, dtype=torch.float32, device=self.device)
                agentInfo  = torch.tensor(agentInfo, dtype=torch.float32, device=self.device)
                actions    = torch.tensor(actions, dtype=torch.float32, device=self.device)
                returns    = torch.tensor(returns, dtype=torch.float32, device=self.device)
                gaes       = torch.tensor(gaes, dtype=torch.float32, device=self.device) #problema
                logProbs   = torch.tensor(logProbs, dtype=torch.float32, device=self.device)
                positionalInfo   = torch.tensor(positionalInfo, dtype=torch.float32, device=self.device)

                self.cohortTrainer[i].trainPolicy(mapInfo, marketInfo, agentInfo, actions, logProbs, gaes, positionalInfo)

                self.cohortTrainer[i].trainValue(mapInfo, marketInfo, agentInfo, returns, positionalInfo)

            progressbar.set_description(f"running epoch {eps+1} bestUtility:{temp.maxUtility} nActions {temp.actionsTaken}")
        obs = []
            
    