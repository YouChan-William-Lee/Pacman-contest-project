# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game


# Own imports
from util import manhattanDistance, nearestPoint
import math
import sys

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent', numTraining = 0):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]



"""
  BRAINSTORMING 30-08-2022

  Must use 2 AI-related techniques.
  The first we plan to use is Heuristic Search Algoritms.
  The second we plan to use is Monte Carlo Tree Search.
  Could also possibly use Reinforcement Learning since that will be covered relatively soon.

  Try using 2 Offensive agents and also 1 Offensive and 1 Defensive agent.

  We can try using A* algorithm for heuristic search. We will need to come up with a heuristic.

  General Ideas:
    - Try having the offensive agent(s) rush for a food pellet and take as many food pellets as it can before returning.
    - Maybe try switching from Offensive to Defense
    - Can try alpha-beta search
    - Can try blocking path to half through defensive agents.
    - Have to come up with a way to evaluate when to go home.
    - When we have two offensive agents, can try to divide the food among them.

"""

##########
# Agents #
##########


############################# ORIGINAL MYTEAM CODE USE IN CASE OF ANY REVERTS #############################

# class DummyAgent(CaptureAgent):
#   """
#   A Dummy agent to serve as an example of the necessary agent structure.
#   You should look at baselineTeam.py for more details about how to
#   create an agent as this is the bare minimum.
#   """

#   def registerInitialState(self, gameState):
#     """
#     This method handles the initial setup of the
#     agent to populate useful fields (such as what team
#     we're on).

#     A distanceCalculator instance caches the maze distances
#     between each pair of positions, so your agents can use:
#     self.distancer.getDistance(p1, p2)

#     IMPORTANT: This method may run for at most 15 seconds.
#     """

#     '''
#     Make sure you do not delete the following line. If you would like to
#     use Manhattan distances instead of maze distances in order to save
#     on initialization time, please take a look at
#     CaptureAgent.registerInitialState in captureAgents.py.
#     '''
#     CaptureAgent.registerInitialState(self, gameState)

#     '''
#     Your initialization code goes here, if you need any.
#     '''


#   def chooseAction(self, gameState):
#     """
#     Picks among actions randomly.
#     """
#     actions = gameState.getLegalActions(self.index)

#     '''
#     You should change this in your own agent.
#     '''

#     return random.choice(actions)

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.entrances = findEntrances(gameState.isOnRedTeam(self.index), self.index, gameState)
    self.middle = findMiddleOfMap(gameState.isOnRedTeam(self.index), self.index, gameState)
    self.walls = gameState.getWalls().asList()
    self.entranceToPatrol = self.middle   
    # self.entranceToPatrol = self.entrances[len(self.entrances)-1]   # Testing
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # print("agent", str(self.index), gameState.getAgentPosition(self.index))

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    # print(bestActions)

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.entrances = findEntrances(gameState.isOnRedTeam(self.index), self.index, gameState)
    self.offensiveEntrances = findOffensiveEntrances(gameState.isOnRedTeam(self.index), gameState, self.entrances)
    self.middle = findMiddleOfMap(gameState.isOnRedTeam(self.index), self.index, gameState)
    self.offensiveMiddle = findOffensiveMiddleOfMap(gameState.isOnRedTeam(self.index), gameState, self.middle)
    self.walls = gameState.getWalls().asList()  
    self.nextFoodToEat = None
    self.isScared = False
    self.foodEaten = 0
    self.entranceToReturnTo = None
    # Setting allow return State as true but this may cause issues in the future. Will need to double check.
    self.offensivePositions = getAllOffensivePositions(gameState.isOnRedTeam(self.index), gameState, True)
    self.legalOffensiveActions = getLegalOffensiveActions(gameState, gameState.isOnRedTeam(self.index))
    self.discountFactor = 0.9 # Gamma/Discount factor for MDP time steps.
    
    CaptureAgent.registerInitialState(self, gameState)
  
  
  ####################################################################################
  # The choose action for the MDP Pacman Offensive Agent
  ####################################################################################

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    # THIS IS JUST A STAR SO FAR. WE WILL DEVELOP MDP INTO THIS OFFENSIVE AGENT. JUST WANT TO GET
    # SOMETHING DONE SO WE CAN GET FEEDBACK ASAP FROM THE FEEDBACK CONTESTS ON OUR DEFENSIVE AGENT.
    start = time.time()
    # actions = gameState.getLegalActions(self.index)

    atEntrance = False

    currentPosition = gameState.getAgentPosition(self.index)
    currentAgentState = gameState.getAgentState(self.index)

    foodToEatList = self.getFood(gameState).asList()

    capsuleList = self.getCapsules(gameState)

    # print("self.middle: ", self.middle)
    # print("self.offensiveMiddle: ", self.offensiveMiddle)
    # print("self.entrances", self.entrances)
    # print("self.offensiveEntrances", self.offensiveEntrances)

    
    # print(self.offensiveMiddle)

    # print("offensive positions")
    # print(self.offensivePositions)
    # print("legal offensive actions")
    # print(self.legalOffensiveActions)

    if currentAgentState.isPacman:
      action = performValueIteration(self.offensivePositions,self.legalOffensiveActions,self.discountFactor,currentPosition,foodToEatList,capsuleList)
      print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
      return action
          


      

      

    # If this occurs, then agent is making its way to the center of the map to attack.
    action = aStarSearchToLocation(gameState, self.index, self.offensiveMiddle, self.isScared)




    print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    return action

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    self.entrances = findEntrances(gameState.isOnRedTeam(self.index), self.index, gameState)
    self.middle = findMiddleOfMap(gameState.isOnRedTeam(self.index), self.index, gameState)
    self.walls = gameState.getWalls().asList()
    self.entranceToPatrol = self.middle   
    # self.entranceToPatrol = self.entrances[len(self.entrances)-1]   # Testing
    self.lastFoodEaten = None
    self.isScared = False
    CaptureAgent.registerInitialState(self, gameState)

  # # print("Implement Defensive agent here")

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    # # print("NEW ACTION ----------")

    start = time.time()

    # # print("LAst eaten food: ", self.lastFoodEaten)

    action = None

    currentPosition = gameState.getAgentPosition(self.index)
    currentAgentState = gameState.getAgentState(self.index)

    # update last food eaten
    if self.getPreviousObservation():
      # # print("Check food here")
      eatenFoods = checkEatenFoods(self.red, self.getPreviousObservation(), gameState)
      closestFood = None
      
      # Decide which food to go for when a food is eated on our side.
      # We don't do this action straight away, as it is important we update the last food eaten
      # BEFORE we a star to any invaders. And then, if there are no invaders, THEN we A* to food.
      if len(eatenFoods) == 2:
        distanceToFirstFood = util.manhattanDistance(currentPosition, eatenFoods[0])
        distanceToSecondFood = util.manhattanDistance(currentPosition, eatenFoods[1])
        if distanceToFirstFood < distanceToSecondFood:
          closestFood = eatenFoods[0]
        else:
          closestFood = eatenFoods[1]
      elif len(eatenFoods) == 1:
        closestFood = eatenFoods[0]

      if closestFood != None:
        # # print("Setting last eaten food")
        self.lastFoodEaten = closestFood
        # self.foodEaten = True
        # # print("DOING FOOD ASTAR")

    # If somehow you become pacman, go back to the entrance
    if currentAgentState.isPacman:
      action = aStarSearchToLocation(gameState, self.index, self.entranceToPatrol)
      return action

    if currentAgentState.scaredTimer > 0:
      # actions = gameState.getLegalActions(self.index)
      # # print(actions)
      # # return "Stop"
      # # print("Agent is scared")
      self.isScared = True
    else:
      self.isScared = False

    # # print("SEEN INVADERS")
    # Find all seen invaders
    seenInvaders = []
    for enemy in self.getOpponents(gameState):
      invader = gameState.getAgentState(enemy)
      if invader.isPacman and invader.getPosition() != None:
        seenInvaders.append(invader)

    # If the defensive agent knows where the enemies are
    if len(seenInvaders) > 0:
      closestInvader = None
      closestDistanceToInvader = sys.maxsize
      for invader in seenInvaders:
        distance = util.manhattanDistance(currentPosition, invader.getPosition())
        if distance < closestDistanceToInvader:
          closestInvader = invader
          closestDistanceToInvader = distance
      
      # # print("ASTAR to invader:")
      # # print(closestInvader.getPosition())
      action = aStarSearchToLocation(gameState, self.index, closestInvader.getPosition(), self.isScared)
      # # print(action)

      # In the case that we eat an invader, set the last food eaten to none so the agent can go back to patrolling.
      successor = gameState.generateSuccessor(self.index, action)
      if successor.getAgentPosition(self.index) == closestInvader.getPosition():
        self.lastFoodEaten = None
      return action

    # If the defensive agent doesn't know where the enemies are, then wait them at one of the entrances 
    # until they get close to 5 manhattan distances 
    if self.isScared:
      action = aStarSearchToLocation(gameState, self.index, self.entranceToPatrol)
      if action == 'Stop':
        # self.entranceToPatrol = random.choice(self.entrances)
        self.entranceToPatrol = (self.entrances[math.floor(len(self.entrances)/2)])

      # print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
      # # print(action)
      return action

    # Ensuring that the ghost checks the last food eaten still.
    if self.lastFoodEaten != None:
      
      action = aStarSearchToLocation(gameState, self.index, self.lastFoodEaten)
      if currentPosition == self.lastFoodEaten:
        self.lastFoodEaten = None
        # instantly go to patrol - DONT STOP
        action = aStarSearchToLocation(gameState, self.index, self.entranceToPatrol)
        # print("Moving back to patrol")
      # # print(action)
      return action

    # # print("A STAR TO ENTRANCE")
    # agent = gameState.getAgentState(agentIndex)
    # agent.isPacman
    action = aStarSearchToLocation(gameState, self.index, self.entranceToPatrol)
    if action == 'Stop':
      # self.entranceToPatrol = random.choice(self.entrances)
      self.entranceToPatrol = getNextEntranceToPatrol(self.entrances, self.entranceToPatrol)
      # print(self.entranceToPatrol)
      # Do another a star so it doesn't stop. Takes more calculation time though.
      action = aStarSearchToLocation(gameState, self.index, self.entranceToPatrol)
      # self.entranceToPatrol = (self.entrances[len(self.entrances) - 1])

    # print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    # # print(action)
    return action

# Method to aStar to any location on the map given the game state, and the agent index.
# Takes in facts such as if the agent is scared, and also if it is an offensive agent.
def aStarSearchToLocation(gameState, agentIndex, location, isScared=False, isOffensive=False):
  """Search the node that has the lowest combined cost and heuristic first."""
  "*** YOUR CODE HERE ***"

  # # print("INSIDE OF ASTAR agent is", agentIndex, location)

  heuristic = util.manhattanDistance

  POSITION_INDEX = 0
  PATH_INDEX = 1
  GAME_STATE_INDEX = 2
  START_STATE_PRIORITY = 0

  startState = (gameState.getAgentPosition(agentIndex), [], gameState)
  frontier = util.PriorityQueue()
  frontier.update(startState, START_STATE_PRIORITY)
  visitedKey = [startState[POSITION_INDEX]]
  visitedValue = [START_STATE_PRIORITY]

  while(not frontier.isEmpty()):
      currentState = frontier.pop()
      currentStatePosition = currentState[POSITION_INDEX]
      currentStatePath = currentState[PATH_INDEX]
      currentGameState = currentState[GAME_STATE_INDEX]

      # Separate goal for if the agent is scared, and if it is not scared.
      if not isScared:
        if currentStatePosition == location:

            if len(currentStatePath) == 0: # Return STOP if the location has been reached and there is no more path.
              return "Stop"
            return currentStatePath[0] # Return the first action on the path  
      else:
        if 2 <= util.manhattanDistance(currentStatePosition, location) <= 3:

            if len(currentStatePath) == 0: # Return STOP if the location has been reached and there is no more path.
              return "Stop"
            return currentStatePath[0] # Return the first action on the path  

      # legalActions = currentGameState.getLegalActions(agentIndex)
      # if len(legalActions) == 0:
      #   # # print("No legal actions")

      for action in currentGameState.getLegalActions(agentIndex):
        successor = currentGameState.generateSuccessor(agentIndex, action)
        # # print("Trying Action ", action)

        

        childPosition = successor.getAgentPosition(agentIndex)

        # This will make the defensive agent never go onto the other side.
        agent = currentGameState.getAgentState(agentIndex)
        if agent.isPacman and not isOffensive:
          # # print("isPacman")
          continue
        # else:
          # # print("NotPacman")

    # for childPosition, action, cost in currentGameState.getSuccessors(currentStatePosition):
        pathToChild = currentStatePath + [action]
        # In A* search, also include the heuristic function value when calculating cost.
        # This is the ONLY difference to the normal UCS algorithm above.
        costToChild = len(pathToChild) + heuristic(childPosition, location)
        if (childPosition not in visitedKey): 
            childState = (childPosition, pathToChild, successor)
            frontier.update(childState, costToChild)
            visitedKey.append(childPosition)
            visitedValue.append(costToChild)
        else:
            visitedIndex = visitedKey.index(childPosition)
            if (costToChild < visitedValue[visitedIndex]):
                childState = (childPosition, pathToChild, successor)
                frontier.update(childState, costToChild)
                visitedKey[visitedIndex] = childPosition
                visitedValue[visitedIndex] = costToChild
  
  # return []
  return "Stop"

# Method to find all entrances. First finds them, then checks that they are valid.
def findEntrances(teamIsRed, agentIndex, gameState):
  
  location = None

  walls = gameState.getWalls().asList()

  if teamIsRed:
    middleWidth = math.floor((gameState.data.layout.width / 2)) - 1

    entrances = []

    # Red team checks their end and the slot to the right which is the blue teams end.
    # If they are both empty, then it is an entrance.
    for i in range(gameState.data.layout.height - 1):
      currentPosition = (middleWidth, i)
      currentPositionToRight = (middleWidth + 1, i)
      
      if currentPosition not in walls and currentPositionToRight not in walls:
        entrances.append(currentPosition)

    # # print("RED TEAM # printING ENTRANCES")
    # # print(entrances)
    
  else:
    middleWidth = math.ceil((gameState.data.layout.width / 2))

    entrances = []

    # Blue team checks their end and the slot to the left which is the red teams end.
    # If they are both empty, then it is an entrance.
    for i in range(gameState.data.layout.height - 1):
      currentPosition = (middleWidth, i)
      currentPositionToLeft = (middleWidth - 1, i)
      
      if currentPosition not in walls and currentPositionToLeft not in walls:
        entrances.append(currentPosition)

    # # print("BLUE TEAM # printING ENTRANCES")
    # # print(entrances)

  # Check validity of entrances

  validEntrances = []
  for entrance in entrances:
    if aStarSearchToLocation(gameState, agentIndex, entrance) != 'Stop':
      validEntrances.append(entrance)

  return validEntrances


# Method that finds all offensive entrances
def findOffensiveEntrances(teamIsRed, gameState, entrances):

  offensiveEntrances = []

  for entrance in entrances:
    if teamIsRed:
      offensiveEntrance = (entrance[0] + 1, entrance[1])
    else:
      offensiveEntrance = (entrance[0] - 1, entrance[1])

    offensiveEntrances.append(offensiveEntrance)

  return offensiveEntrances

# Find the middle of the map by finding all entrances and getting the middle entrance.
# WARNING: This may mess up in cases where an entrance is unreachable
def findMiddleOfMap(teamIsRed, agentIndex, gameState):
  entrances = findEntrances(teamIsRed, agentIndex, gameState)
  return entrances[math.ceil(len(entrances)/2)] 

# Find the offensive middle of the map
def findOffensiveMiddleOfMap(teamIsRed, gameState, defensiveMiddle):
  if teamIsRed:
    return (defensiveMiddle[0]+1, defensiveMiddle[1])
  else:
    return (defensiveMiddle[0]-1, defensiveMiddle[1])


# Method that will check for food that is eaten on the agent's own team.
# Return list of food.
def checkEatenFoods(teamIsRed, previousState, currentState):
  previousFood = []
  currentFood = []
  if teamIsRed:
    previousFood = previousState.getRedFood().asList()
    currentFood = currentState.getRedFood().asList()
  else:
    previousFood = previousState.getBlueFood().asList()
    currentFood = currentState.getBlueFood().asList()

  return list(set(previousFood) - set(currentFood))

# Method that will choose the next entrance for the defender agent to go to.
def getNextEntranceToPatrol(entrances, currentEntrance):

  # Patrol randomly between upper and lower entrances
  upperEntrances = entrances[:len(entrances)//2]
  lowerEntrances = entrances[len(entrances)//2:]

  if currentEntrance in upperEntrances:
    return random.choice(lowerEntrances)
  else:
    return random.choice(upperEntrances)


# OFFENSIVE MDP METHODS!!!!

# Method that gets all offensive positions of an agent.
def getAllOffensivePositions(teamIsRed, gameState, returnState=False):
  positions = []
  width = gameState.data.layout.width
  height = gameState.data.layout.height

  startingX = 0
  endingX = math.floor((gameState.data.layout.width / 2))
  if returnState:
    endingX = math.floor((gameState.data.layout.width / 2)) + 1

  if teamIsRed:
    startingX = math.ceil((gameState.data.layout.width / 2))
    if returnState:
      startingX = math.ceil((gameState.data.layout.width / 2)) - 1
    endingX = width

  for x in range(startingX, endingX):
    for y in range(height):
      if not gameState.hasWall(x,y):
        positions.append((x,y))

  # print(positions)

  return positions

# Method that returns all legal offensive actions for a state
def getLegalOffensiveActions(gameState, teamIsRed):
  
  allStateLegalActions = {}

  possibleStates = getAllOffensivePositions(teamIsRed, gameState, True) # True because we want to include returning to base

  # print(possibleStates)

  for state in possibleStates:

    xPosition = state[0]
    yPosition = state[1]

    legalActions = []

    moveUpPosition = (xPosition, yPosition + 1)
    if not gameState.hasWall(moveUpPosition[0], moveUpPosition[1]) and moveUpPosition in possibleStates:
      legalActions.append(Directions.NORTH)
    moveRightPosition = (xPosition + 1, yPosition)
    if not gameState.hasWall(moveRightPosition[0], moveRightPosition[1]) and moveRightPosition in possibleStates:
      legalActions.append(Directions.EAST)
    moveDownPosition = (xPosition, yPosition - 1)
    if not gameState.hasWall(moveDownPosition[0], moveDownPosition[1]) and moveDownPosition in possibleStates:
      legalActions.append(Directions.SOUTH)
    moveLeftPosition = (xPosition - 1, yPosition)
    if not gameState.hasWall(moveLeftPosition[0], moveLeftPosition[1]) and moveLeftPosition in possibleStates:
      legalActions.append(Directions.WEST)

    allStateLegalActions[state] = legalActions

  return allStateLegalActions

# MDP method to generate the successor of a game state and an action made.
# Game state is the position, action is which direction it moves to.
def generateSuccessor(gameState, action):
  x = gameState[0]
  y = gameState[1]
  if action == Directions.NORTH:
    return (x,y+1)
  elif action == Directions.EAST:
    return (x+1,y)
  elif action == Directions.SOUTH:
    return (x,y-1)
  elif action == Directions.WEST:
    return (x-1,y)
  else:
    return (x,y)

# MDP function to calculate the reward.
def calculateMDPReward(state, foodList, capsuleList):
  reward = 0
  if state in foodList:
    reward += 1
  if state in capsuleList:
    reward += 5




  return reward

# Method to perform the value iteration. Returns an action.
def performValueIteration(offensivePositions, legalOffensiveActions, discountFactor, currentPosition, foodList, capsuleList):
  # if pacman, then do MDP
  # print("Do MDP")

  # optimal policy which stores action and q value as a tuple
  optimalPolicies = {state: ("Stop", 0.0) for state in offensivePositions}

  # Constants for indexes of the actions and q values
  ACTION_INDEX = 0
  Q_VALUE_INDEX = 1

  numIterations = 100

  # Value iteration
  for i in range(numIterations):
    previousPolicies = optimalPolicies.copy()

    # print("Iteration:", i)

    for state in offensivePositions:
      QDict = {}

      for action in legalOffensiveActions[state]:
        # print("Current State:", state)
        childState = generateSuccessor(state, action) #Method that gets the child state when applying action to state
        # print("Child State:", childState)
        # calculateReward function, bellmans equation here. R(s) + gamma * next state utility
        QDict[action] = calculateMDPReward(state, foodList, capsuleList) + discountFactor * previousPolicies[childState][Q_VALUE_INDEX]

      optimalPolicies[state] = (getActionOfMaxQValue(QDict), max(QDict.values()))

  return optimalPolicies[currentPosition][ACTION_INDEX]

# Method used to quickly get the key of the max q value in the q dictionary.
def getActionOfMaxQValue(QDict):
  qValues = list(QDict.values())
  actions = list(QDict.keys())
  return actions[qValues.index(max(qValues))]
