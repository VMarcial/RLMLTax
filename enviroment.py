import random
import numpy as np
import math

class enviroment():
    """
    Essa classe cria o ambiente da simulação. Ela é composta por dois elementos: market e geography.
    Parametros:
    * seed(int): seed para o random. permite reproducibilidade. Por padrão None, que leva a uma seed aleatória.
    * mapSize(int): tamanho do mapa mapSize x mapSize. Por padrão 5.
    * visionSize(int): tamanho do espaço ao seu redor que o agente vê. Por padrão 2.
    """
    
    def __init__(self, seed = None, mapSize = 5, visionSize = 2):
        
        if seed == None: self.seed = random.randint(0, 10000)
        else: self.seed = seed
        self.mapSize = mapSize
        self.visionSize = visionSize
        
        # Inicialização do mercado
        self.market = market()

        # Inicialização do mapa
        self.geography = geography(seed= self.seed, mapSize = self.mapSize, visionSize= visionSize)


    def getInfo(self, x, y):
        """
        Retorna informações do ambiente para o agente
        Parametros:
        * x(int): posição x do agente no mapa 
        * y(int): posição y do agente no mapa

        Retorna:
        * mapInfo: array de formato mapSize x mapSize x 3. 
        * marketInfo: lista de tamanho y. Contendo os dados de preço e quantidade do mercado.
        * workingInfo(tupla):  
        """
        
        maps = self.geography.subMap(x,y)
        
        workingTile = self.workingTile(x,y)
        workingInfo = [int(workingTile[1]), int(workingTile[2])]
        
        marketInfo = self.market.getInfo() # melhorpreco, quantidademelhorprecofood, quantidademelhorprecocloth, quantidadefood, quantidadecloth
        
        return maps, marketInfo, workingInfo
    

    

    def workingTile(self, x, y):

        """
        Retorna a informação se o indivíduo está em um campo em que possa produzir. 
        Caso possa, retorna uma lista contendo a produtividade do local onde está.
        
        Parametros:
        * x(int): posição x do agente no mapa 
        * y(int): posição y do agente no mapa

        Retorna lista com elementos que:

        0) True caso seja possível trabalhar nesse campo, False caso contrário
        1) 0 caso não seja possível produzir comida neste campo. int da produtividade do campo caso contrário
        2) 0 caso não seja possível produzir tecido neste campo. int da produtividade do campo caso contrário
        """

        if self.geography.food[x, y] != 0:
            return [True, self.geography.food[x, y], 0]
        elif self.geography.cloth[x, y] != 0:
            return [True, 0, self.geography.cloth[x, y]]
        else:
            return [False, False, False]


    def step(self, agent, action, mapInfo, marketInfo, agentInfo,consumptionFoodTax, consumptionClothTax, wealthTax, laborFoodTax, laborClothTax, eps, fquantity, cquantity):

        #TODO organizar isso, principalemente os inputs

        # actions
        # 0: Nada
        # 1: Up
        # 2: Down
        # 3: Left
        # 4: Right
        # 5: Trade Food for Cloth
        # 6: Trade Cloth for Food
        # 7: Consume
        # 8: Invest

        """
        Realiza uma ação no ambiente

        Parametros:

        """


        newMarketInfo = marketInfo #TODO arrumar isso
        
        # Ficar parado. Caso seja um local de trabalho, permite que o agente produza comida
        #TODO adicionar alguma fadiga por trabalhar demais
        if action == 0:
            if self.workingTile(agent.x, agent.y)[0] == True:
                t1foodTaxed, t1clothTaxed = agent.collect(self.workingTile(agent.x, agent.y), laborFoodTax, laborClothTax)
                newAgentInfo, utilityReward, t2foodTaxed, t2clothTaxed = agent.update(action, mapInfo, wealthTax)
                
                foodTaxed = t1foodTaxed + t2foodTaxed 
                clothTaxed = t1clothTaxed + t2clothTaxed
                
                return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive, foodTaxed, clothTaxed
            else:
                newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, mapInfo, wealthTax = wealthTax, foodConsumptionTax = consumptionFoodTax)

                return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive, foodTaxed, clothTaxed

        # Movimenta para cima
        elif action == 1:
            tempX, tempY = agent.x, agent.y + 1

            if self.geography.food[tempX, tempY] != -1:
                newMapInfo = self.geography.updatePeople(agent.x, agent.y, tempX, tempY)
                agent.x, agent.y = tempX, tempY
                newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, newMapInfo, wealthTax =wealthTax)

                return newMapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive, foodTaxed, clothTaxed 
            
            else:
                return mapInfo, newMarketInfo, agentInfo, agent.wrongChoicePunishment, agent.alive, 0, 0 

        # Movimenta para baixo
        elif action == 2:     
            tempX, tempY = agent.x, agent.y - 1

            if self.geography.food[tempX, tempY] != -1:
                newMapInfo = self.geography.updatePeople(agent.x, agent.y, tempX, tempY)
                agent.x, agent.y = tempX, tempY
                newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, newMapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)

                return newMapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive, foodTaxed, clothTaxed 
            
            else:
                return mapInfo, newMarketInfo, agentInfo, agent.wrongChoicePunishment, agent.alive, 0, 0 

        # Movimenta para direita
        elif action == 3:
            tempX, tempY = agent.x + 1, agent.y

            if self.geography.food[tempX, tempY] != -1:
                newMapInfo = self.geography.updatePeople(agent.x, agent.y, tempX, tempY)
                agent.x, agent.y = tempX, tempY
                newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, newMapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)

                return newMapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive, foodTaxed, clothTaxed 
            
            else:
                return mapInfo, newMarketInfo, agentInfo, agent.wrongChoicePunishment, agent.alive, 0, 0 

        # Movimenta para a esqueda      
        elif action == 4:           
            tempX, tempY = agent.x - 1, agent.y

            if self.geography.food[tempX, tempY] != -1:
                newMapInfo = self.geography.updatePeople(agent.x, agent.y, tempX, tempY)
                agent.x, agent.y = tempX, tempY
                newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, newMapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)

                return newMapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive, foodTaxed, clothTaxed 
            
            else:
                return mapInfo, newMarketInfo, agentInfo, agent.wrongChoicePunishment, agent.alive, 0, 0 
        
        # Faz uma oferta de comida por tecido
        elif action == 5:

            if fquantity > 0 and cquantity > 0:
            
                order = Offer(agent.rg, "food", fquantity, "cloth", cquantity, eps) #TODO verificar se não ta ofertando mais comida/tecido do que tem

                self.market.orders.append(order)

                newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, mapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)

                return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive, foodTaxed, clothTaxed
            else:
                newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, mapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)

                return mapInfo, newMarketInfo, agentInfo, agent.wrongChoicePunishment, agent.alive, 0, 0 

        # Faz uma oferta de tecido por comida
        elif action == 6:

            if fquantity > 0 and cquantity > 0: #TODO verificar se não ta ofertando mais comida/tecido do que tem

                order = Offer(agent.rg, "cloth", cquantity, "food", fquantity, eps)

                self.market.orders.append(order)

                newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, mapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)

                return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive, foodTaxed, clothTaxed
            
            else:
                newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, mapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax)

                return mapInfo, newMarketInfo, agentInfo, agent.wrongChoicePunishment, agent.alive, 0, 0 

        # Consome tecido  
        elif action == 7:
            
            newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, mapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax, consumedCloth = 1)

            return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive, foodTaxed, clothTaxed

        # Investe em um posto de trabalho aumentando sua produtividade para todos
        elif action == 8:
            newAgentInfo, utilityReward, foodTaxed, clothTaxed = agent.update(action, mapInfo, wealthTax =wealthTax, foodConsumptionTax = consumptionFoodTax, clothConsumptionTax = consumptionClothTax)
                
            return mapInfo, newMarketInfo, newAgentInfo, utilityReward, agent.alive, foodTaxed, clothTaxed
        
        #TODO adicionar aumento de capital humano
        else:
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
    

    def getInfo(self):

        """
        Retorna informações do mercado de preço e quantidade disponíveis

        Parametros:

        Retorna:
        

        """

        offers = self.orders
        
        offers_A_to_B = [o for o in offers if o.offer_item == 'cloth' and o.request_item == 'food']
        offers_B_to_A = [o for o in offers if o.offer_item == 'food' and o.request_item == 'cloth']
        
        offers_A_to_B = sorted(offers_A_to_B, key=lambda x: (x.offer_quantity / x.request_quantity)) #TODO checar se isso ta realmente fazendo os preços certos
        offers_B_to_A = sorted(offers_B_to_A, key=lambda x: (x.request_quantity / x.offer_quantity))
        
        # Calculando a quantidade total ofertada no mercado
        quantidadeA = sum([x.offer_quantity for x in offers_A_to_B])
        quantidadeB = sum([x.offer_quantity  for x in offers_B_to_A])
        
        # Inicizaliamos os preços implicitos nas ofertas
        precoImplicitoA = None
        precoImplicitoB = None
        
        # Calcula o melhor preço implicito nas ofertas
        if len(offers_A_to_B)>0: precoImplicitoA = (offers_A_to_B[0].offer_quantity / offers_A_to_B[0].request_quantity)
        if len(offers_B_to_A)>0: precoImplicitoB = (offers_B_to_A[0].request_quantity / offers_B_to_A[0].offer_quantity)
        
        # Calcula o melhor preço e a quantidade disponível aquele preço, retorna -1 caso não haja
        if precoImplicitoA != None:
            if precoImplicitoB != None:
                    melhorPreco = min(precoImplicitoA, precoImplicitoB)
                    melhorQuantidadeA = offers_A_to_B[0].offer_quantity
                    melhorQuantidadeB = offers_B_to_A[0].request_quantity
            else:
                melhorPreco = precoImplicitoA
                precoImplicitoB = -1
                melhorQuantidadeB = -1
                melhorQuantidadeA = offers_A_to_B[0].offer_quantity
        else:
            if precoImplicitoB != None:
                melhorPreco = precoImplicitoB
                melhorQuantidadeB = offers_B_to_A[0].request_quantity
                precoImplicitoA = -1
                melhorQuantidadeA = -1
            else:
                melhorPreco = -1
                precoImplicitoA = -1
                precoImplicitoB = -1
                melhorQuantidadeA = -1
                melhorQuantidadeB = -1
        
        return [quantidadeA, quantidadeB, melhorQuantidadeA, melhorQuantidadeB, melhorPreco]

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
    
    def updatePrices(self):
        pass

    def clear_market(self):
        offers = self.orders
        # Separate offers based on item types
        offers_A_to_B = [o for o in offers if o.offer_item == 'cloth' and o.request_item == 'food']
        offers_B_to_A = [o for o in offers if o.offer_item == 'food' and o.request_item == 'cloth']

        matched_offers = []

        for offer_A in offers_A_to_B[:]:  # We use slicing [:] to loop over a copy of the list
            for offer_B in offers_B_to_A[:]:
                if offer_A.offer_quantity / offer_A.request_quantity >= offer_B.request_quantity / offer_B.offer_quantity:
                    transact_quantity = min(offer_A.offer_quantity, offer_B.request_quantity)
                    cquantity =  transact_quantity * offer_A.offer_quantity / offer_A.request_quantity
                    matched_offers.append((offer_A, offer_B, transact_quantity, cquantity))
                    offer_A.offer_quantity -= transact_quantity
                    offer_B.request_quantity -= transact_quantity

                    # Remove offers that have been completely matched
                    if offer_A.offer_quantity == 0:
                        offers_A_to_B.remove(offer_A)
                    if offer_B.request_quantity == 0:
                        offers_B_to_A.remove(offer_B)
                    break
        self.orders = offers_A_to_B + offers_B_to_A
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
        """
        Retorna a posição x,y do meio do mapa"""
        self.center = ((self.visionSize * 2) + self.mapSize)//2


    def subMap(self, x, y):
        lx = x - self.visionSize
        ly = y - self.visionSize
        hx = x + self.visionSize #+1
        hy = y + self.visionSize #+1

        #TODO checar porque ta saindo 4x4 em vez de 5x5

        # TODO não sei porque funciona sem isso mas funciona, não deveria
        # Caso algo de errado, isso aqui restringe os erros
        #if lx < 0: lx = self.visionSize
        #if ly < 0: ly = self.visionSize
        #if hx > self.mapSize + (2*self.visionSize): hx = self.mapSize + self.visionSize
        #if hy > self.mapSize + (2*self.visionSize): hy = self.mapSize + self.visionSize

        maps = [self.food[lx:hx, ly:hy], self.cloth[lx:hx, ly:hy], self.people[lx:hx, ly:hy]]
        return maps
    

        
