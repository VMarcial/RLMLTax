import random
import numpy as np
import math

class enviroment():
    
    def __init__(self, seed = None, mapSize = 5, visionSize = 2):
        self.market = market()
        self.mapSize = mapSize
        if seed == None: self.seed = random.randint(0, 10000)
        self.geography = geography(seed= self.seed, mapSize = self.mapSize, visionSize= visionSize)


    def getInfo(self, x, y):
        maps = self.geography.subMap(x,y)
        workingTile = self.workingTile(x,y)
        #TODO adicionar as informações de mercado
        y = [0,0,0,0,0] # melhorpreco, quantidademelhorprecofood, quantidademelhorprecocloth, quantidadefood, quantidadecloth
        #x = np.concatenate([maps[0].flatten(), maps[1].flatten(), maps[2].flatten()])        
        return maps, y, [int(workingTile[2]), int(workingTile[2])]
    

    def workingTile(self, x, y):

        if self.geography.food[x, y] != 0:
            return [True, 0, self.geography.food[x, y]]
        elif self.geography.cloth[x, y] != 0:
            return [True, 0, self.geography.cloth[x, y]]
        else:
            return [False, False, False]


    def step(self, agent, action, mapInfo, marketInfo, agentInfo,consumptionFoodTax, consumptionClothTax, wealthTax, laborFoodTax, laborClothTax, eps, fquantity, cquantity):

        # 0: Nada
        # 1: Up
        # 2: Down
        # 3: Left
        # 4: Right
        # 5: Trade Food for Cloth
        # 6: Trade Cloth for Food
        # 7: Consume
        # 8: Invest

        #TODO colocar resolução do mercado aqui

        newMarketInfo = marketInfo

        if action == 0:
            if self.workingTile(agent.x, agent.y)[0] == True:
                agent.collect(self.workingTile(agent.x, agent.y), laborFoodTax, laborClothTax)
                newAgentInfo, utilityReward = agent.update(action, mapInfo, wealthTax)

                return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive
            else:
                newAgentInfo, utilityReward = agent.update(action, mapInfo, wealthTax = wealthTax, foodConsumptionTax = consumptionFoodTax)

                return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive

        elif action == 1:
            tempX, tempY = agent.x, agent.y + 1

            if self.geography.food[tempX, tempY] != -1:
                newMapInfo = self.geography.updatePeople(agent.x, agent.y, tempX, tempY)
                agent.x, agent.y = tempX, tempY
                newAgentInfo, utilityReward = agent.update(action, newMapInfo, wealthTax =wealthTax)

                return newMapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive 
            
            else:
                return mapInfo, newMarketInfo, agentInfo, agent.wrongChoicePunishment, agent.alive 

        elif action == 2:     
            tempX, tempY = agent.x, agent.y - 1

            if self.geography.food[tempX, tempY] != -1:
                newMapInfo = self.geography.updatePeople(agent.x, agent.y, tempX, tempY)
                agent.x, agent.y = tempX, tempY
                newAgentInfo, utilityReward = agent.update(action, newMapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)

                return newMapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive 
            
            else:
                return mapInfo, newMarketInfo, agentInfo, agent.wrongChoicePunishment, agent.alive 
            
        elif action == 3:
            tempX, tempY = agent.x + 1, agent.y

            if self.geography.food[tempX, tempY] != -1:
                newMapInfo = self.geography.updatePeople(agent.x, agent.y, tempX, tempY)
                agent.x, agent.y = tempX, tempY
                newAgentInfo, utilityReward = agent.update(action, newMapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)

                return newMapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive 
            
            else:
                return mapInfo, newMarketInfo, agentInfo, agent.wrongChoicePunishment, agent.alive 
        
        elif action == 4:           
            tempX, tempY = agent.x - 1, agent.y

            if self.geography.food[tempX, tempY] != -1:
                newMapInfo = self.geography.updatePeople(agent.x, agent.y, tempX, tempY)
                agent.x, agent.y = tempX, tempY
                newAgentInfo, utilityReward = agent.update(action, newMapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)

                return newMapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive 
            
            else:
                return mapInfo, newMarketInfo, agentInfo, agent.wrongChoicePunishment, agent.alive 
            
        elif action == 5: #TODO adicionar mercado
            
            order = Offer(agent.rg, "food", fquantity, "cloth", cquantity, eps)

            self.market.orders.append(order)

            newAgentInfo, utilityReward = agent.update(action, mapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)
                
            return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive
        
        elif action == 6: #TODO adicionar mercado

            order = Offer(agent.rg, "cloth", cquantity, "food", fquantity, eps)
            
            self.market.orders.append(order)
            
            newAgentInfo, utilityReward = agent.update(action, mapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)
            
            return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive
        
        elif action == 7:
            
            #tempUtilityReward = agent.consume()#TODO adicionar quantidade variavel
            newAgentInfo, utilityReward = agent.update(action, mapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax, consumedCloth = 1)

            #utilityReward =  tempUtilityReward + utilityReward

            return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive

        elif action == 8:
            newAgentInfo, utilityReward = agent.update(action, mapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax, clothConsumptionTax = consumptionClothTax)
                
            return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive
        else:
            print(action)
            print("deu errado")
            pass


class market():

    # Preços são dados em comida/cloth

    def __init__(self) -> None:
        self.infoSize = 5
        self.orders = []
        self.buys = []
        self.sells = []
        self.maxPrice = None
        self.minPrice = None


    def addBuy(self, agent, fquantity, cquantity, eps):
        total = fquantity
        if self.agent.food >= total:
            self.agent.food -= total
            self.orders.append(Offer(agent, "food", fquantity, "cloth", cquantity, eps))
            return 0 
        else:
            return agent.wrongChoicePunishment

    def addSell(self, agent, fquantity, cquantity, eps):
        total = cquantity
        if self.agent.cloth >= total:
            self.agent.cloth -= total
            self.orders.append(Offer(agent, "cloth", cquantity, "food", fquantity, eps))
            return 0 
        else:
            return agent.wrongChoicePunishment


    def clear_market(self):
        offers = self.orders
        # Separate offers based on item types
        offers_A_to_B = [o for o in offers if o.offer_item == 'itemA' and o.request_item == 'itemB']
        offers_B_to_A = [o for o in offers if o.offer_item == 'itemB' and o.request_item == 'itemA']

        matched_offers = []

        for offer_A in offers_A_to_B[:]:  # We use slicing [:] to loop over a copy of the list
            for offer_B in offers_B_to_A[:]:
                if offer_A.offer_quantity / offer_A.request_quantity >= offer_B.request_quantity / offer_B.offer_quantity:
                    transact_quantity = min(offer_A.offer_quantity, offer_B.request_quantity)
                    matched_offers.append((offer_A, offer_B, transact_quantity))
                    offer_A.offer_quantity -= transact_quantity
                    offer_B.request_quantity -= transact_quantity

                    # Remove offers that have been completely matched
                    if offer_A.offer_quantity == 0:
                        offers_A_to_B.remove(offer_A)
                    if offer_B.request_quantity == 0:
                        offers_B_to_A.remove(offer_B)
                    break

        return matched_offers


class Offer:
    def __init__(self, agent, offer_item, offer_quantity, request_item, request_quantity, eps):
        self.agent = agent
        self.offer_item = offer_item
        self.offer_quantity = offer_quantity
        self.request_item = request_item
        self.request_quantity = request_quantity
        self.eps = eps




class geography():

    def __init__(self, seed, mapSize, nagents= 10, foodQuantity = 8, clothQuantity = 2, visionSize = 2):
        self.seed = seed
        self.mapSize = mapSize
        self.foodQuantity = foodQuantity
        self.clothQuantity =  clothQuantity
        self.visionSize = visionSize
        self.nagents = nagents
        self.visionInput = 48
        self._getCenterMap()
        self.generateMap()

    def generateMap(self):
        # Checando se é possível criar todos os recursos com o tamanho do mapa
        if self.foodQuantity + self.clothQuantity > self.mapSize**2:
            raise OverflowError
        random.seed(self.seed)
        
        resourceLocations = []
        foodPlaced = 0
        clothPlaced = 0

        # Inicializamos o mapa com -1, com um tamanho tal que o mapa esteja no centro do mapa e
        # nas bordas a visão retorne -1 até o tamanho máximo da visão
        self.food   = np.full([self.mapSize + (2*self.visionSize), self.mapSize + (2*self.visionSize)], -1)
        self.cloth  = np.full([self.mapSize + (2*self.visionSize), self.mapSize + (2*self.visionSize)], -1)
        self.people = np.full([self.mapSize + (2*self.visionSize), self.mapSize + (2*self.visionSize)], -1)

        self.people[self.center, self.center] = self.nagents

        # Colocamos no centro dos -1 o mapa real inicializado com 0
        self.food[self.visionSize:self.visionSize+self.mapSize, self.visionSize:self.visionSize+self.mapSize]   = 0 
        self.cloth[self.visionSize:self.visionSize+self.mapSize, self.visionSize:self.visionSize+self.mapSize]  = 0 
        self.people[self.visionSize:self.visionSize+self.mapSize, self.visionSize:self.visionSize+self.mapSize] = 0 
        

        # Colocando os recursos no mapa, não é permitido mais de um recurso no mesmo local

        
        while foodPlaced < self.foodQuantity:
            x = random.randint(self.visionSize, self.visionSize + self.mapSize - 1)
            y = random.randint(self.visionSize, self.visionSize + self.mapSize - 1)
    
            if [x,y] in resourceLocations:
                pass
            else:
                foodPlaced += 1
                resourceLocations.append([x,y])
                self.food[x, y] = 1
        
        while clothPlaced < self.clothQuantity:
            x = random.randint(self.visionSize, self.visionSize + self.mapSize - 1)
            y = random.randint(self.visionSize, self.visionSize + self.mapSize - 1)
            
            if [x,y] in resourceLocations:
                pass
            else:
                clothPlaced += 1
                resourceLocations.append([x,y])
                self.cloth[x, y] = 1
        self.resourceLocations = resourceLocations


    def updatePeople(self, oldX, oldY, newX, newY):
        #TODO arrumar inicialização
        self.people[oldX, oldY] -= 1
        self.people[newX, newY] += 1
        
        return self.subMap(newX, newY)


    def _getCenterMap(self):
        self.center = ((self.visionSize * 2) + self.mapSize)//2


    def subMap(self, x, y):
        lx = x - self.visionSize
        ly = y - self.visionSize
        hx = x + self.visionSize #+1
        hy = y + self.visionSize #+1

        #TODO checar porque ta saindo 4x4 em vez de 5x5

        # Caso algo de errado, isso aqui restringe os erros
        #if lx < 0: lx = self.visionSize
        #if ly < 0: ly = self.visionSize
        #if hx > self.mapSize + (2*self.visionSize): hx = self.mapSize + self.visionSize
        #if hy > self.mapSize + (2*self.visionSize): hy = self.mapSize + self.visionSize

        maps = [self.food[lx:hx, ly:hy], self.cloth[lx:hx, ly:hy], self.people[lx:hx, ly:hy]]
        return maps
    

        
