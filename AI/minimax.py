##
# Genetic Algorithm Agent for HW 4
# CS 421
#
# Authors: Geryl Vinoya and Matthew Groh
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
        super(AIPlayer,self).__init__(inputPlayerId, "Genetic_vinoya21_grohm22")
        self.popSize = 8 #sets the INITIAL population size if no population file is given
        self.currFit = []
        self.nextEval = 1
        self.numGames = 5
        self.numPlayed = 0

        self.currPop = self.initPopulation()
        
        while (self.currPop[self.nextEval-1][12] + self.currPop[self.nextEval-1][13]) >= self.numGames:
                    self.nextEval += 1
        
        self.numPlayed = self.currPop[self.nextEval-1][12] + self.currPop[self.nextEval-1][13]

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
        newGene = [0]*15 #initialize a list with 15 empty spots
        for i in range(0, 12, 1):
            newGene[i] = random.uniform(-10.0, 10.0)
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
        pop = []
        #get the path for the population.txt file and read if it exists
        myPath = Path("./vinoya21_grohm22_population.txt")
        if myPath.is_file():
            f = open(myPath, 'r')
            contents = f.read().splitlines()
            for line in contents:
                gene = ast.literal_eval(line)
                pop.append(gene)
            f.close()
        #initialize the gene list with random values from -10 to 10
        else:
            for i in range(self.popSize):
                pop.append(self.initGene())

            #after initializing new genes, put that in the text
            f = open("vinoya21_grohm22_population.txt", "w")
            f.truncate(0)
            for gene in pop:
                f.write(str(gene) + "\n")
            f.close()

            

        self.currFit.clear()
        for i in range(0, self.popSize + 1, 1):
            self.currFit.append(0)
        self.nextEval = 1 #set to the first in the list
        return pop


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
        #make new children using slicing and step
        listChildren = []
        child1 = [0]*15
        child2 = [0]*15
        
        child1[0:12:2] = parent1[0:12:2]
        child1[1:12:2] = parent2[1:12:2]
        child2[0:12:2] = parent2[0:12:2]
        child2[1:12:2] = parent1[1:12:2]
        
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
        fittestValues = []
        fittestParents = []
        newGeneration = []

        #select the top 4 in the population sorted by fitness
        
        fittestParents = sorted(self.currPop, key=lambda x: x[14])
        #fittestValues = sorted(self.currFit)
        fittestValues = fittestValues[0:4]
        #for value in fittestValues:
           #fittestParents.append(self.currPop[value])


        #might need to fix the currFit list to reflect on this
        #or the currFit list might not even be needed

        #will need to adjust for a larger population size
        #this gives us 24 children
        for i in range(0, 4, 1):
            for j in range(0, 4, 1):
                if i != j:
                    for child in self.generateChildren(fittestParents[i], fittestParents[j]):
                        newGeneration.append(child)

        newGeneration = newGeneration[:self.popSize]

        self.currPop.clear()

        for child in newGeneration:
            self.currPop.append(child)

        f = open("AI/vinoya21_grohm22_population.txt", "w")
        f.truncate(0) 
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
    #Description: function to either update the next gene to be evaluated or
    #             create a new generation of genes
    #             this happens at the end of each game
    #
    #Parameters:
    #   currentState - current state
    #   hasWon - boolean if the play has won or lost
    #
    #Return: None
    ##
    def registerWin(self, hasWon):
        # Update fitness score based on win/lose
        if hasWon:
            self.currFit[self.nextEval - 1] += 1
            self.currPop[self.nextEval - 1][12] += 1
        else:
            self.currPop[self.nextEval - 1][13] += 1
        self.numPlayed += 1
        self.currFit[self.nextEval - 1] = self.currFit[self.nextEval - 1] / self.numPlayed

        #Calculate the W/L ratio and store it in spot 14 of the gene
        if self.currPop[self.nextEval - 1][13] == 0:
            self.currPop[self.nextEval - 1][14] = self.currPop[self.nextEval - 1][12]
        else:
            self.currPop[self.nextEval - 1][14] = self.currPop[self.nextEval - 1][12] / self.currPop[self.nextEval - 1][13]
        


        if self.numPlayed  >= self.numGames:
            self.nextEval += 1
            self.numPlayed = 0

        if self.nextEval > self.popSize:
            self.nextGeneration()
            self.nextEval = 1
            self.numPlayed = 0
        else:
            f = open("AI/vinoya21_grohm22_population.txt", "w")
            f.truncate(0)
            for gene in self.currPop:
                f.write(str(gene) + "\n")
            f.close()
        pass

    ##
    #foodDiff
    #Description: utility helper function for
    #             difference in food between me and opponent
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def foodDiff(self, currentState, weight):
        me = currentState.whoseTurn
        myFood = currentState.inventories[me].foodCount
        enemyFood = currentState.inventories[1 - me].foodCount
        return weight * (myFood - enemyFood)

    ##
    #queenHealthDiff
    #Description: utility helper function for
    #             difference in health between my queen and enemy queen
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def queenHealthDiff(self, currentState, weight):
        me = currentState.whoseTurn
        myQueen = currentState.inventories[me].getQueen()
        myQueenHealth = 0
        if myQueen:
            myQueenHealth = myQueen.health

        enemyQueen = currentState.inventories[1 - me].getQueen()
        enemyQueenHealth = 0
        if enemyQueen:
            enemyQueenHealth = enemyQueen.health
        return weight * (myQueenHealth - enemyQueenHealth)

    ##
    #droneCompare
    #Description: utility helper function for
    #             difference between my drones and enemy drones
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def droneCompare(self, currentState, weight):
        me = currentState.whoseTurn
        myDrones = getAntList(currentState, me, (DRONE,))
        enemyDrones = getAntList(currentState, (1 - me), (DRONE,))
        return weight * (len(myDrones) - len(enemyDrones))

    ##
    #workerCompare
    #Description: utility helper function for
    #             difference between my workers and enemy workers
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def workerCompare(self, currentState, weight):
        me = currentState.whoseTurn
        myWorkers = getAntList(currentState, me, (WORKER,))
        enemyWorkers = getAntList(currentState, (1 - me), (WORKER,))
        return weight * (len(myWorkers) - len(enemyWorkers))

    ##
    #soldierCompare
    #Description: utility helper function for
    #             difference between my soldiers and enemy soldiers
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def soldierCompare(self, currentState, weight):
        me = currentState.whoseTurn
        mySoldiers = getAntList(currentState, me, (SOLDIER,))
        enemySoldiers = getAntList(currentState, (1 - me), (SOLDIER,))
        return weight * (len(mySoldiers) - len(enemySoldiers))

    ##
    #offensiveCompare
    #Description: utility helper function for
    #             comparing if me or enemy has more offensive capability
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def offensiveCompare(self, currentState, weight):
        me = currentState.whoseTurn
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

    ##
    #attackingQueenDist
    #Description: utility helper function for
    #             average distance between enemy queen and my offensive ants
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def attackingQueenDist(self, currentState, weight):
        stepsAway = []
        me = currentState.whoseTurn
        myDrones = getAntList(currentState, me, (DRONE,))

        enemyQueen = currentState.inventories[1 - me].getQueen()
        if enemyQueen:
            enemyQueenCoords = enemyQueen.coords
            for drone in myDrones:
                stepsAway.append(approxDist(drone.coords, enemyQueenCoords))
            if len(stepsAway) == 0:
                return 0
            else:
                return weight * (sum(stepsAway) / len(stepsAway))
        return 0

    ##
    #defendingQueenDist
    #Description: utility helper function for
    #             average distance between my queen and enemy offensive ants
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def defendingQueenDist(self, currentState, weight):
        stepsAway = []
        me = currentState.whoseTurn
        enemyDrones = getAntList(currentState, (1 - me), (DRONE,))

        myQueen = currentState.inventories[me].getQueen()
        if myQueen:       
            myQueenCoords = myQueen.coords
            for drone in enemyDrones:
                stepsAway.append(approxDist(drone.coords, myQueenCoords))
            if len(stepsAway) == 0:
                return 0
            else:
                return weight * (sum(stepsAway) / len(stepsAway))
        return 0

    ##
    #attackingAnthillDist
    #Description: utility helper function for
    #             average distance between enemy anthill and my offensive ants
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def attackingAnthillDist(self, currentState, weight):
        stepsAway = []
        me = currentState.whoseTurn
        myDrones = getAntList(currentState, me, (DRONE,))
        enemyAnthillCoords = getConstrList(currentState, (1 - me), (ANTHILL,))[0].coords
        for drone in myDrones:
            stepsAway.append(approxDist(drone.coords, enemyAnthillCoords))
        if len(stepsAway) == 0:
            return 0
        else:
            return weight * (sum(stepsAway) / len(stepsAway))

    ##
    #defendingAnthillDist
    #Description: utility helper function for
    #             average distance between my anthill and enemy offensive ants
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def defendingAnthillDist(self, currentState, weight):
        stepsAway = []
        me = currentState.whoseTurn
        enemyDrones = getAntList(currentState, (1 - me), (DRONE,))
        myAnthillCoords = getConstrList(currentState, me, (ANTHILL,))[0].coords
        for drone in enemyDrones:
            stepsAway.append(approxDist(drone.coords, myAnthillCoords))
        if len(stepsAway) == 0:
            return 0
        else:
            return weight * (sum(stepsAway) / len(stepsAway))

    ##
    #workersFromQueen
    #Description: utility helper function avg distanve between my workers and queen
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def workersFromQueen(self, currentState, weight):
        stepsAway = []
        me = currentState.whoseTurn
        myWorkers = getAntList(currentState, me, (WORKER,))

        myQueen = currentState.inventories[me].getQueen()
        if myQueen:
            myQueenCoords = myQueen.coords
            for worker in myWorkers:
                stepsAway.append(approxDist(worker.coords, myQueenCoords))
            if len(stepsAway) == 0:
                return 0
            else:
                return weight * (sum(stepsAway) / len(stepsAway))
        return 0


    ##
    #queenSeparation
    #Description: utility helper function for how far away the queens are
    #
    #Parameters:
    #   currentState - current state
    #   weight - given weight to multiply the value
    #
    #Return: The utility value
    ##
    def queenSeparation(self, currentState, weight):
        me = currentState.whoseTurn
        myQueen = currentState.inventories[me].getQueen()
        enemyQueen = currentState.inventories[1 - me].getQueen()
        
        if myQueen and enemyQueen:
            myQueenCoords = currentState.inventories[me].getQueen().coords
            enemyQueenCoords = currentState.inventories[1 - me].getQueen().coords
            return weight * approxDist(myQueenCoords, enemyQueenCoords)
        
        return 0

    ##
    #utility
    #Description: Method to return a value of how good a game state is
    #
    #Parameters:
    #   currentState - current state
    #
    #Return: The utility value
    ##
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

##
#TestCreateNode
#Description: Unit tests for functions
#
#Variables:
#   unittest.TestCase
##
class TestCreateNode(unittest.TestCase):
    # test for queens, anthills, and tunnels only.
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
        #self.assertEqual(player.utility(gameState), -0.98)

    # test for worker and food.
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
        #self.assertAlmostEqual(player.utility(gameState), -159/251)

    # test for soldier and enemy queen.
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
        #self.assertAlmostEqual(player.utility(gameState), 203/221)

    # test for soldier and enemy worker.
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
        #self.assertAlmostEqual(player.utility(gameState), 6/11)

    # test initPopulation method
    def test_initPopulation(self):
        player = AIPlayer(0)
        player.initPopulation()
        self.assertNotEqual(player.currPop, [])
        self.assertNotEqual(player.currFit, [])
        #self.assertEqual(player.nextEval, player.currPop[0])

    #test initGene method
    def test_initGene(self):
        player = AIPlayer(0)

        gene = player.initGene()

        for i in range(0, 12, 1):

            self.assertAlmostEqual(gene[i], 0, delta=10)

        self.assertEqual(gene[14], 0)

    #test generateChildren method
    def test_generateChildren(self):
        player = AIPlayer(0)
        parent1 = [7.507215762345602, -0.9353204286408445, 5.476848785448352, 
            4.294255252678781, 3.6872182103715083, 5.420527128934285, -5.397806227288577, 
            0.405377902684922, -5.139527196741747, 6.783147929282464, -0.47716375391582844, 0.32226207693290654, 0, 0, 0]
        parent2 = [2.9292292505301205, -8.774524899413418, 4.393742330335932,
            7.8408059326901025, 7.888871502914284, 7.362293911864427, 4.930590254141119,
            1.8456828797033573, -2.676689047090191, -1.245851380097914, -4.583554164498778, 9.202898392277866, 0, 0, 0]

        children = player.generateChildren(parent1, parent2)

        child1 = children[0]

        self.assertEqual(child1[0], 7.507215762345602)
        self.assertEqual(child1[1], -8.774524899413418)
        self.assertEqual(child1[2], 5.476848785448352)
        self.assertEqual(child1[3], 7.8408059326901025)
        self.assertEqual(child1[4], 3.6872182103715083)
        self.assertEqual(child1[5], 7.362293911864427)
        self.assertEqual(child1[6], -5.397806227288577)
        self.assertEqual(child1[7], 1.8456828797033573)
        self.assertEqual(child1[8], -5.139527196741747)
        self.assertEqual(child1[9], -1.245851380097914)
        self.assertEqual(child1[10], -0.47716375391582844)
        self.assertEqual(child1[11], 9.202898392277866)

    #test nextGeneration method
    def test_nextGeneration(self):
        player = AIPlayer (0)

        player.initPopulation()

        old = player.currPop

        #player.nextGeneration()

        #new = player.currPop

        #self.assertNotEqual(old, new)


if __name__ == '__main__':
    unittest.main()