##
# MiniMaxing Agent for HW 3
# CS 421
#
# Authors: Geryl Vinoya and Samuel Nguyen
##
import random
import sys
import time
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *
from typing import Dict, List
import unittest
from pathlib import Path
import os
import ast

##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "Genetic")
        self.popSize = 8
        self.currPop = self.initPopulation()
        self.nextEval = 1
        self.currFit = []
        self.numGames = 20
        

    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    

    ##
    #initGene
    #Description: Method to initialize a gene
    #             A gene is a list of ints displaying the values
    #
    #Parameters:
    #   self
    #
    #Return: List of ints
    ##
    def initGene(self) -> List[int]:
        newGene = [None]*15 #initialize a list with 15 empty spots
        for i in range(0, 12, 1):
            newGene[i] = random.uniform(-10.0, 10.0)

        newGene[14] = 0
        
        return newGene


    ##
    #initPopulation
    #Description: Method to initialize the population of genes
    #
    #Parameters:
    #   self
    #
    #Return: None
    ##
    def initPopulation(self):
        #get the path for the population.txt file and read if it exists
        myPath = Path("./vinoya21_grohm22_population.txt")
        if myPath.is_file():
            f = open(myPath, 'r')
            contents = f.read().splitlines()
            for line in contents:
                gene = ast.literal_eval(line)
                self.currPop.append(gene)
                self.currFit.append(gene[14])
            f.close()
        
        #initialize the gene list with random values from -10 to 10
        else:
            for i in range(self.popSize):
                self.currPop.append(self.initGene())
                self.currFit.append(0)

        self.nextEval = 1 #set to the first in the list
        return self.currPop


    ##
    #generateChildren
    #Description: Generates two children from two parents
    #
    #Parameters:
    #   parent1 - the first parent gene
    #   parent2 - the second parent gene
    #
    #Return: List of the new children
    ##
    def generateChildren(self, parent1, parent2):
        #maybe make the children creation more random?
        listChildren = []
        child1 = [None]*15
        child2 = [None]*15
        slice1 = parent1[0:6]
        slice2 = parent2[6:12]
        slice3 = parent2[0:6]
        slice4 = parent1[6:12]
        child1[0:6] = slice1
        child1[6:12] = slice2
        child2[0:6] = slice3
        child2[6:12] = slice4
        listChildren.append(child1)
        listChildren.append(child2)
        #set default fitness (in spot 14) to 0
        child1[14] = 0
        child2[14] = 0
        return listChildren

    ##
    #nextGeneration
    #Description: Creates the next generation of genes
    #
    #Parameters:
    #   self
    #
    #Return: None
    ##
    def nextGeneration(self):
        fittestParents = []
        newGeneration = []

        #select the top 4 in the population sorted by fitness
        
        fittestParents = sorted(self.currPop, key=lambda x: x[14])

        fittestParents = fittestParents[0:4]

        #might need to fix the currFit list to reflect on this
        #or the currFit list might not even be needed

        #will need to adjust for a larger population size
        #this gives us 24 children
        for i in range(0, 4, 1):
            for j in range(0, 4, 1):
                if i != j:
                    for child in self.generateChildren(fittestParents[i], fittestParents[j]):
                        newGeneration.append(child)

        self.currPop = newGeneration

        f = open("vinoya21_grohm22_population.txt", "w")
        for gene in newGeneration:
            f.write(str(gene) + "\n")
        f.close()



    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):

        # create a root node object with a move, current state, eval, parent
        root = {
            "move": None,
            "state": currentState,
            "evaluation": self.utility(currentState),
            "parent": None
        }

        return self.getMiniMaxMove(root, 0, True, -100, 100)

    ##
    #getMiniMaxMove
    #Description: Recursively gets the best minimax move at depth 2
    #
    #Parameters:
    #   currentNode - the current node being worked on
    #   currentDepth - the current depth 
    #   myTurn - boolean if it is the AIPlayer's turn or not
    #   alpha - alpha value for pruning
    #   beta - beta value for pruning
    #
    #Return: The Move to be made
    ##
    def getMiniMaxMove (self, currentNode, currentDepth, myTurn, alpha, beta):
        maxDepth = 2 #the max depth for the minimax tree

        #get the current node's eval and state to be used later
        currNodeEval = currentNode.get("evaluation")
        currentState = currentNode.get("state")

        #if we are at max depth or at a win/loss, return the eval number
        if currentDepth == maxDepth or currNodeEval == -1 or currNodeEval == 1:
            return currNodeEval

        #create a list of children nodes for the current node
        children = []
        legalMoves = listAllLegalMoves(currentState)
        for move in legalMoves:
            newState = getNextStateAdversarial(currentState, move)
            node = {
                "move": move,
                "state": newState,
                "evaluation": self.utility(newState),
                "parent": None,
            }
            children.append(node)

        #sort the children in descending order and only pick the highest 2
        children = sorted(children, key=lambda child: child.get("evaluation"), reverse=True)
        children = children[:2]

        #create a list of nodes that have an end move type
        endNodes = []
        for node in children:
            endNodes.append(self.getEndNode(node))

        #now we have all the information we need, recursion will start below    
        #   if this is the root node, then it is our turn and we will recursively get the
        #   move with the best eval
        if currentDepth == 0:
            for node in endNodes:
                node["evaluation"] = self.getMiniMaxMove(node, 1, False, -10, 10)
        
            best = max(endNodes, key=lambda node: node.get("evaluation"))
            while best.get("parent"):
                best = best.get("parent")
            
            return best.get("move")
        
        #if this is not the root node, then we travel through the tree
        #   and evaluate the values while performing alpha-beta pruning
        else:
            if myTurn:
                #set to an arbitrary "low" number to compare the high score to get the highest score
                maxScore = -10

                #continue to recurse through the endNodes to find the best move and change playerturn
                for node in endNodes:
                    miniMaxScore = self.getMiniMaxMove(node, currentDepth + 1, False, alpha, beta)
                    if miniMaxScore > maxScore: 
                        maxScore = miniMaxScore
                    if maxScore > alpha:
                        alpha = maxScore

                    #check if we can prune
                    if beta <= alpha:
                        break
                return maxScore
            
            else:
                #set to an arbitrary "high" number to compare the high score to get the lowest score
                minScore = 10

                #continue to recurse through the endNodes to find the best move and change playerturn
                for node in endNodes:
                    miniMaxScore = self.getMiniMaxMove(node, currentDepth + 1, True, alpha, beta)
                    if miniMaxScore < minScore:
                        minScore = miniMaxScore
                    if minScore < beta:
                        beta = minScore
                
                    #check if we can prune
                    if beta <= alpha:
                        break

                return minScore

    ##
    #getEndNode
    #Description: Recursively finds the highest scoring end node
    #
    #Parameters:
    #   currentNode - node to find
    #
    #Return: The Move to be made
    ##
    def getEndNode(self, currentNode):

        #base case
        #   if we find a node that is an end type return that node
        currentMove = currentNode.get("move")
        if currentMove.moveType == END:
            return currentNode

        currentState = currentNode.get("state")

        #create list of children of the current node
        children = []
        legalMoves = listAllLegalMoves(currentState)
        for move in legalMoves:
            newState = getNextStateAdversarial(currentState, move)
            node = {
                "move": move,
                "state": newState,
                "evaluation": self.utility(newState),
                "parent": currentNode,
            }
            children.append(node)
        #find the best child and return it
        bestChild = max(children, key=lambda node: node.get("evaluation"))
        return self.getEndNode(bestChild)

    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        if hasWon:
            self.currPop[self.nextEval - 1][12] += 1
        else:
            self.currPop[self.nextEval - 1][13] += 1
        self.currPop[self.nextEval - 1][14] = self.currPop[self.nextEval - 1][12] / self.currPop[self.nextEval - 1][13]
        if self.currPop[self.nextEval - 1][12] + self.currPop[self.nextEval - 1][13] == self.numGames:
            self.nextEval += 1
        if self.nextEval > self.popSize:
            self.nextGeneration()
            self.nextEval = 1
        pass



    ##
    #utility
    #Description: examines GameState object and returns a heuristic guess of how
    #               "good" that game state is on a scale of 0 to 1
    #
    #               a player will win if his opponent’s queen is killed, his opponent's
    #               anthill is captured, or if the player collects 11 units of food
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: the "guess" of how good the game state is
    ##
    """def utility(self, currentState):
        WEIGHT = 10 #weight value for moves

        #will modify this toRet value based off of gamestate
        toRet = 0

        #get my id and enemy id
        me = currentState.whoseTurn
        enemy = 1 - me

        #get the values of the anthill, tunnel, and foodcount
        myTunnel = getConstrList(currentState, me, (TUNNEL,))[0]
        myAnthill = getConstrList(currentState, me, (ANTHILL,))[0]
        myFoodList = getConstrList(currentState, 2, (FOOD,))
        enemyTunnel = getConstrList(currentState, enemy, (TUNNEL,))[0]

        #get my soldiers and workers
        mySoldiers = getAntList(currentState, me, (SOLDIER,))
        myWorkerList = getAntList(currentState, me, (WORKER,))

        #get enemy worker and queen
        enemyWorkerList = getAntList(currentState, enemy, (WORKER,))
        enemyQueenList = getAntList(currentState, enemy, (QUEEN,))

        for worker in myWorkerList:

            #if a worker is carrying food, go to tunnel
            if worker.carrying:
                tunnelDist = stepsToReach(currentState, worker.coords, myTunnel.coords)
                #anthillDist = stepsToReach(currentState, worker.coords, myAnthill.coords)

                #if tunnelDist <= anthillDist:
                toRet = toRet + (1 / (tunnelDist + (4 * WEIGHT)))
                #else:
                    #toRet = toRet + (1 / (anthillDist + (4 * WEIGHT)))

                #add to the eval if a worker is carrying food
                toRet = toRet + (1 / WEIGHT)

            #if a worker isn't carrying food, get to the food
            else:
                foodDist = 1000
                for food in myFoodList:
                    # Updates the distance if its less than the current distance
                    dist = stepsToReach(currentState, worker.coords, food.coords)
                    if (dist < foodDist):
                        foodDist = dist
                toRet = toRet + (1 / (foodDist + (4 * WEIGHT)))
        
        #try to get only 1 worker
        if len(myWorkerList) == 1:
            toRet = toRet + (2 / WEIGHT)
        

        #try to get only one soldier
        if len(mySoldiers) == 1:
            toRet = toRet + (WEIGHT * 0.2)
            enemyWorkerLength = len(enemyWorkerList)
            enemyQueenLength = len(enemyQueenList)
            
            #we want the soldier to go twoards the enemy tunnel/workers
            if enemyWorkerList:
                distToEnemyWorker = stepsToReach(currentState, mySoldiers[0].coords, enemyWorkerList[0].coords)
                distToEnemyTunnel = stepsToReach(currentState, mySoldiers[0].coords, enemyTunnel.coords)
                toRet = toRet + (1 / (distToEnemyWorker + (WEIGHT * 0.2))) + (1 / (distToEnemyTunnel + (WEIGHT * 0.5)))
            
            #reward the agent for killing enemy workers
            #try to kill the queen if enemy workers dead
            else:
                toRet = toRet + (2 * WEIGHT)
                if enemyQueenLength > 0:
                    enemyQueenDist = stepsToReach(currentState, mySoldiers[0].coords, enemyQueenList[0].coords)
                    toRet = toRet + (1 / (1 + enemyQueenDist))
            

            toRet = toRet + (1 / (enemyWorkerLength + 1)) + (1 / (enemyQueenLength + 1))

        #try to get higher food score
        foodCount = currentState.inventories[me].foodCount
        toRet = toRet + foodCount

        #set the correct bounds for the toRet
        toRet = 1 - (1 / (toRet + 1))
        if toRet <= 0:
            toRet = 0.01
        if toRet >= 1:
            toRet = 0.99

        #convert the previous score of [0,1] to [-1, 1]
        if toRet == 0.5:
            toRet = 0
        elif toRet > 0.5:
            toRet = (2 * toRet) - 1
        elif toRet < 0.5:
            toRet = -(1 - (2 * toRet))

        return toRet"""


    # Difference in food between me and opponent
    def foodDiff(self, currentState, weight):
        me = currentState.whoseTurn
        myFood = currentState.inventories[me].foodCount
        enemyFood = currentState.inventories[1 - me].foodCount
        return weight * (myFood - enemyFood)


    # Difference in health between my queen and enemy queen
    def queenHealthDiff(self, currentState, weight):
        me = currentState.whoseTurn
        myQueenHealth = currentState.inventories[me].getQueen().health
        enemyQueenHealth = currentState.inventories[1 - me].getQueen().health
        return weight * (myQueenHealth - enemyQueenHealth)


    # Difference between my drones and enemy's drones
    def droneCompare(self, currentState, weight):
        me = currentState.whoseTurn
        myDrones = getAntList(currentState, me, (DRONE,))
        enemyDrones = getAntList(currentState, (1 - me), (DRONE,))
        return weight * (len(myDrones) - len(enemyDrones))


    # Difference between my workers and enemy's workers
    def workerCompare(self, currentState, weight):
        me = currentState.whoseTurn
        myWorkers = getAntList(currentState, me, (WORKER,))
        enemyWorkers = getAntList(currentState, (1 - me), (WORKER,))
        return weight * (len(myWorkers) - len(enemyWorkers))


    # Difference between my soldiers and enemy's soldiers
    def soldierCompare(self, currentState, weight):
        me = currentState.whoseTurn
        mySoldiers = getAntList(currentState, me, (SOLDIER,))
        enemySoldiers = getAntList(currentState, (1 - me), (SOLDIER,))
        return weight * (len(mySoldiers) - len(enemySoldiers))


    # Comparing if me or enemy has more offensive capability
    def offensiveCompare(self, currentState, weight):
        myDrones = getAntList(currentState, me, (DRONE,))
        mySoldiers = getAntList(currentState, me, (SOLDIER,))
        myRSoldiers = getAntList(currentState, me, (R_SOLDIER,))
        enemyDrones = getAntList(currentState, me, (DRONE,))
        enemySoldiers = getAntList(currentState, me, (SOLDIER,))
        enemyRSoldiers = getAntList(currentState, me, (R_SOLDIER,))

        myOffense = len(myDrones) + len(mySoldiers) + len(myRSoldiers)
        enemyOffense = len(enemyDrones) + len(enemySoldiers) + len(myRSoldiers)
        if myOffense > enemyOffense:
            return weight * 1
        else:
            return 0


    # Average dist between enemy queen and my offensive ants
    def attackingQueenDist(self, currentState, weight):
        stepsAway = []
        me = currentState.whoseTurn
        myDrones = getAntList(currentState, me, (DRONE,))
        enemyQueenCoords = currentState.inventories[1 - me].getQueen().coords
        for drone in myDrones:
            stepsAway.append(approxDist(drone.coords, enemyQueenCoords))
        return weight * (sum(stepsAway) / len(stepsAway))


    # Average dist between my queen and enemy's offensive ants
    def defendingQueenDist(self, currentState, weight):
        stepsAway = []
        me = currentState.whoseTurn
        enemyDrones = getAntList(currentState, (1 - me), (DRONE,))
        myQueenCoords = currentState.inventories[me].getQueen().coords
        for drone in enemyDrones:
            stepsAway.append(approxDist(drone.coords, myQueenCoords))
        return weight * (sum(stepsAway) / len(stepsAway))


    # Average dist between enemy anthill and my offensive ants
    def attackingAnthillDist(self, currentState, weight):
        stepsAway = []
        me = currentState.whoseTurn
        myDrones = getAntList(currentState, me, (DRONE,))
        enemyAnthillCoords = getConstrList(currentState, (1 - me), (ANTHILL,))[0].coords
        for drone in myDrones:
            stepsAway.append(approxDist(drone.coords, enemyAnthillCoords))
        return weight * (sum(stepsAway) / len(stepsAway))


    # Average dist between my anthill and enemy's offensive ants
    def defendingAnthillDist(self, currentState, weight):
        stepsAway = []
        me = currentState.whoseTurn
        enemyDrones = getAntList(currentState, (1 - me), (DRONE,))
        myAnthillCoords = getConstrList(currentState, me, (ANTHILL,))[0].coords
        for drone in enemyDrones:
            stepsAway.append(approxDist(drone.coords, myAnthillCoords))
        return weight * (sum(stepsAway) / len(stepsAway))


    # Average dist between my workers and my queen
    def workersFromQueen(self, currentState, weight):
        stepsAway = []
        me = currentState.whoseTurn
        myWorkers = getAntList(currentState, me, (WORKER,))
        myQueenCoords = currentState.inventories[me].getQueen().coords
        for worker in myWorkers:
            stepsAway.append(approxDist(worker.coords, myQueenCoords))
        return weight * (sum(stepsAway) / len(stepsAway))


    # Distance between my queen and enemy queen
    def queenSeparation(self, currentState, weight):
        myQueenCoords = currentState.inventories[me].getQueen().coords
        enemyQueenCoords = currentState.inventories[1 - me].getQueen().coords
        return weight * approxDist(myQueenCoords, enemyQueenCoords)


    def utility(self, currentState):
        foodDiffScore = self.foodDiff(currentState, self.currPop[self.nextEval - 1][0])
        queenDiffScore = self.queenHealthDiff(currentState, self.currPop[self.nextEval - 1][1])
        droneDiffScore = self.droneCompare(currentState, self.currPop[self.nextEval - 1][2])
        workerDiffScore = self.workerCompare(currentState, self.currPop[self.nextEval - 1][3])
        soldierDiffScore = self.soldierCompare(currentState, self.currPop[self.nextEval - 1][4])
        offenseScore = self.offensiveCompare(currentState, self.currPop[self.nextEval - 1][5])
        queenAtkScore = self.attackingQueenDist(currentState, self.currPop[self.nextEval - 1][6])
        queenDefScore = self.defendingQueenDist(currentState, self.currPop[self.nextEval - 1][7])
        anthillAtkScore = self.attackingAnthillDist(currentState, self.currPop[self.nextEval - 1][8])
        anthillDefScore = self.defendingAnthillDist(currentState, self.currPop[self.nextEval - 1][9])
        queenDistScore = self.workersFromQueen(currentState, self.currPop[self.nextEval - 1][10])
        queenSepScore = self.queenSeparation(currentState, self.currPop[self.nextEval - 1][11])

        return foodDiffScore + queenDiffScore + droneDiffScore + workerDiffScore + soldierDiffScore \
            + offenseScore + queenAtkScore + queenDefScore + anthillAtkScore + anthillDefScore \
            + queenDistScore + queenSepScore


class TestCreateNode(unittest.TestCase):
    # Queens, anthills, and tunnels only.
    def test_utility_basic(self):
        player = AIPlayer(0)

        # Create game state.
        gameState = GameState.getBasicState()

        # Calculations below.

        # toRet = 0

        # toRet = 1 - (1 / (toRet + 1)) = 1
        # if toRet <= 0:
        #     toRet = 0.01 = 0.01
        # if toRet >= 1:
        #     toRet = 0.99
        #
        # if toRet == 0.5:
        #     toRet = 0
        # elif toRet > 0.5:
        #     toRet = (2 * toRet) - 1
        # elif toRet < 0.5:
        #     toRet = -(1 - (2 * toRet)) = -0.98
        self.assertEqual(player.utility(gameState), -0.98)

    # Worker and food added.
    def test_utility_worker_and_food(self):
        player = AIPlayer(0)

        # Create game state with food.
        gameState = GameState.getBasicState()
        p1Food1 = Building((1, 1), FOOD, 0)
        p1Food2 = Building((2, 2), FOOD, 0)
        gameState.board[1][1] = p1Food1
        gameState.board[2][2] = p1Food2
        gameState.inventories[2].constrs += [p1Food1, p1Food2]
        p1Food1 = Building((7, 7), FOOD, 1)
        p1Food2 = Building((8, 8), FOOD, 1)
        gameState.board[7][7] = p1Food1
        gameState.board[8][8] = p1Food2
        gameState.inventories[2].constrs += [p1Food1, p1Food2]

        # Add worker.
        worker = Ant((1, 0), WORKER, 0)
        gameState.board[1][0] = worker
        gameState.inventories[0].ants.append(worker)

        # Calculations below.

        # toRet = 0
        # Dist from closer food is 1, toRet = toRet + (1 / (foodDist + (4 * WEIGHT))) = 1/41.
        # toRet = toRet + (2 / WEIGHT) = 46/205
        # toRet = 1 - (1 / (toRet + 1)) = 46/251
        # if toRet <= 0:
        #     toRet = 0.01
        # if toRet >= 1:
        #     toRet = 0.99
        #
        # if toRet == 0.5:
        #     toRet = 0
        # elif toRet > 0.5:
        #     toRet = (2 * toRet) - 1
        # elif toRet < 0.5:
        #     toRet = -(1 - (2 * toRet)) = -159/251

        # Help from https://stackoverflow.com/questions/33199548/how-to-perform-unittest-for-floating-point-outputs-python
        # assertAlmostEqual exists.
        self.assertAlmostEqual(player.utility(gameState), -159/251)

    # Soldier and enemy queen.
    def test_utility_soldier_and_queen(self):
        player = AIPlayer(0)

        # Create game state.
        gameState = GameState.getBasicState()

        # Add soldier.
        soldier = Ant((1, 0), SOLDIER, 0)
        gameState.board[1][0] = soldier
        gameState.inventories[0].ants.append(soldier)

        # Calculations below.

        # toRet = 0
        # One soldier, toRet = toRet + (WEIGHT * 0.2) = 2.
        # No workers, toRet = toRet + (2 * WEIGHT) = 22.
        # Dist to enemy queen is 17, toRet = toRet + (1 / (1 + 17)) = 397/18.
        # No enemy workers, toRet = toRet + (1 / (0 + 1)) + (1 / (1 + 1)) = 212/9.
        # toRet = 1 - (1 / (toRet + 1)) = 212/221
        # if toRet <= 0:
        #     toRet = 0.01
        # if toRet >= 1:
        #     toRet = 0.99
        #
        # if toRet == 0.5:
        #     toRet = 0
        # elif toRet > 0.5:
        #     toRet = (2 * toRet) - 1 = 203/221
        # elif toRet < 0.5:
        #     toRet = -(1 - (2 * toRet))
        self.assertAlmostEqual(player.utility(gameState), 203/221)

    # Soldier and enemy worker.
    def test_utility_soldier_and_worker(self):
        player = AIPlayer(0)

        # Create game state.
        gameState = GameState.getBasicState()

        # Add soldier and enemy worker.
        soldier = Ant((1, 0), SOLDIER, 0)
        enemyWorker = Ant((2, 0), WORKER, 1)
        gameState.board[1][0] = soldier
        gameState.board[2][0] = enemyWorker
        gameState.inventories[0].ants.append(soldier)
        gameState.inventories[1].ants.append(enemyWorker)

        # Calculations below.

        # toRet = 0
        # One soldier, toRet = toRet + (WEIGHT * 0.2) = 2.
        # Dist to enemy worker is 1, dist to enemy tunnel is 10.
        # toRet = toRet + (1 / (1 + (WEIGHT * 0.2))) + (1 / (10 + (WEIGHT * 0.5))) = 12/5
        # toRet = toRet + (1 / (1 + 1)) + (1 / (1 + 1)) = 17/5
        # toRet = 1 - (1 / (toRet + 1)) = 17/22
        # if toRet <= 0:
        #     toRet = 0.01
        # if toRet >= 1:
        #     toRet = 0.99
        #
        # if toRet == 0.5:
        #     toRet = 0
        # elif toRet > 0.5:
        #     toRet = (2 * toRet) - 1 = 6/11
        # elif toRet < 0.5:
        #     toRet = -(1 - (2 * toRet))
        self.assertAlmostEqual(player.utility(gameState), 6/11)

    def test_initPopulation(self):
        player = AIPlayer(0)

        self.assertEqual(player.currPop, [])
        self.assertEqual(player.nextEval, None)
        self.assertEqual(player.currFit, [])
        self.assertEqual(player.numGames, 20)
        player.initPopulation()
        self.assertNotEqual(player.currPop, [])
        self.assertNotEqual(player.currFit, [])
        self.assertEqual(player.nextEval, player.currPop[0])

    def test_initGene(self):
        player = AIPlayer(0)

        gene = player.initGene()

        for i in range(0, 12, 1):

            self.assertAlmostEqual(gene[i], 0, delta=10)

        self.assertEqual(gene[14], 0)

    def test_generateChildren(self):
        player = AIPlayer(0)
        parent1 = player.initGene()
        parent2 = player.initGene()

        children = player.generateChildren(parent1, parent2)

        child1 = children[0]
        child2 = children[1]

        self.assertEqual(child1[0:5], parent1[0:5])
        self.assertEqual(child1[6:11], parent2[6:11])
        self.assertEqual(child2[0:5], parent2[0:5])
        self.assertEqual(child2[6:11], parent1[6:11])

    def test_nextGeneration(self):
        player = AIPlayer (0)

        player.initPopulation()

        old = player.currPop

        player.nextGeneration()

        new = player.currPop

        self.assertNotEqual(old, new)


if __name__ == '__main__':
    unittest.main()