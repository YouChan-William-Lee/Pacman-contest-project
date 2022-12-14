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
from distanceCalculator import DistanceCalculator
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
               first = 'OffensiveReflexAgent', second = 'OffensiveReflexAgent', numTraining = 0):
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
    self.entrancesDict = makeEntrancesDict(self.entrances)
    self.offensiveEntrances = findOffensiveEntrances(gameState.isOnRedTeam(self.index), gameState, self.entrances)
    self.middle = findMiddleOfMap(gameState.isOnRedTeam(self.index), self.index, gameState)
    self.offensiveMiddle = findOffensiveMiddleOfMap(gameState.isOnRedTeam(self.index), gameState, self.middle)
    self.walls = gameState.getWalls().asList()  
    self.wallsDict = makeWallsDict(self.walls)
    self.nextFoodToEat = None
    self.isScared = False
    self.foodEaten = 0
    self.entranceToReturnTo = None
    # Setting allow return State as true but this may cause issues in the future. Will need to double check.
    self.offensivePositions = getAllOffensivePositions(gameState.isOnRedTeam(self.index), gameState, True)
    self.legalOffensiveActions = getLegalOffensiveActions(gameState, gameState.isOnRedTeam(self.index))
    self.discountFactor = 0.9 # Gamma/Discount factor for MDP time steps.
    self.offensiveFoodEaten = 0
    self.storeFood = False
    self.nextAttackingPoint = None
    self.teamIndex = getTeamIndex(gameState.isOnRedTeam(self.index), self.getTeam(gameState), self.index)
    self.ownOffensiveEntrances = getOwnEntrances(self.offensiveEntrances, self.teamIndex)
    self.ownDefensiveEntrances = getOwnEntrances(self.entrances, self.teamIndex)
    self.numWallsDict = makeNumWallsDict(self.offensivePositions, self.wallsDict)
    self.totalFoodCount = len(self.getFood(gameState).asList())
    self.lastFoodEaten = None
    # print(len(self.getFood(gameState).asList()))
    print("double offensive and defensive mdp agent V3.2 - remove prints and increase num iterations to 150")
    # print(self.numWallsDict)
    # print("Team index:", self.teamIndex)
    # print("agent index:", self.index)
    # print(self.ownOffensiveEntrances)
    # print(self.ownDefensiveEntrances)
    
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
    currentScore = self.getScore(gameState)

    if not currentAgentState.isPacman:
      self.offensiveFoodEaten = 0
    
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv  Copied from down (CTRL + Z) for going back
    team = self.getTeam(gameState)
    # get index of teammate
    teammateIndex = -1
    for teammate in team:
      if teammate != self.index:
        teammateIndex = teammate
    teammatePosition = gameState.getAgentPosition(teammateIndex)
    teammateState = gameState.getAgentState(teammateIndex)
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    currentDirection = currentAgentState.getDirection()

    foodToEatList = self.getFood(gameState).asList()

    # Defend when team mate agent is Pacman and wining 
    # certainAmountOfScores = self.totalFoodCount / 4
    certainAmountOfScores = 1

    # if teammateState.isPacman and currentScore >= certainAmountOfScores:
    #   print("defend")
      # One idea when only few food left and we are defending
      # We can get the left food's location by self.getFoodYouAreDefending(gameState)
      # Only petroling these foods

      # do defensive -> probably copy some codes from defensive agent?

    # If foodtoeatlist is less than or equal to 2, just stay on defense.
    if (len(foodToEatList) <= 2 and not currentAgentState.isPacman) or (currentScore >= certainAmountOfScores and not currentAgentState.isPacman and currentAgentState.scaredTimer <= 10):
      # For now, just stop But we want them to play defense now
      # return DefensiveReflexAgent.getAction(self, gameState)

      
      
      # DO DEFENSIVE ACTION HERE

      # If we see a pacman, then we a star to them

      # print("AGENT DEFENDING: ", self.index)

      seenInvaders = []
      for enemy in self.getOpponents(gameState):
        invader = gameState.getAgentState(enemy)
        if invader.isPacman and invader.getPosition() != None:
        # if invader.getPosition() != None: # Try also checking for ghost and not just pacman
          seenInvaders.append(invader)

      # If the defensive agent knows where the enemies are
      if len(seenInvaders) > 0:
        closestInvader = None
        closestDistanceToInvader = sys.maxsize
        for invader in seenInvaders:
          distance = util.manhattanDistance(currentPosition, invader.getPosition())
          teammateDistance = util.manhattanDistance(teammatePosition, invader.getPosition())
          if distance <= teammateDistance:
            if distance < closestDistanceToInvader:
              closestInvader = invader
              closestDistanceToInvader = distance

        # Try to block the pacman position if u can see
        if closestInvader != None:
          pacmanBlockingPosition = getPacmanBlockingPosition(gameState.isOnRedTeam(self.index), closestInvader.getPosition(), currentPosition, self.wallsDict)
          action = aStarSearchToLocation(gameState, self.index, pacmanBlockingPosition, self.isScared, False, True)
          # # print(action)
          return action
      
      # # Try to get to the closest entrance to lost food
      # if self.getPreviousObservation():
      #   # # print("Check food here")
      #   eatenFoods = checkEatenFoods(self.red, self.getPreviousObservation(), gameState)
      #   closestFood = None
        
      #   # Decide which food to go for when a food is eated on our side.
      #   # We don't do this action straight away, as it is important we update the last food eaten
      #   # BEFORE we a star to any invaders. And then, if there are no invaders, THEN we A* to food.
      #   if len(eatenFoods) == 2:
      #     distanceToFirstFood = util.manhattanDistance(currentPosition, eatenFoods[0])
      #     distanceToSecondFood = util.manhattanDistance(currentPosition, eatenFoods[1])
      #     if distanceToFirstFood < distanceToSecondFood:
      #       closestFood = eatenFoods[0]
      #     else:
      #       closestFood = eatenFoods[1]
      #   elif len(eatenFoods) == 1:
      #     closestFood = eatenFoods[0]

      #   if closestFood != None:
      #   # # print("Setting last eaten food")
      #     self.lastFoodEaten = closestFood

      #   # A star to this
      #   if self.lastFoodEaten != None:
      #     closestEntranceToLostFood = getLostFoodClosestEntrance(self.lastFoodEaten, self.entrancesDict)

      #     action = aStarSearchToLocation(gameState, self.index, closestEntranceToLostFood)
      #     return action
      #   else:
      action = aStarSearchToLocation(gameState, self.index, self.ownDefensiveEntrances[len(self.ownDefensiveEntrances)//2])
      return action

    foodDict = {}
    capsuleDict = {}

    for food in foodToEatList:
      foodDict[food] = True

    capsuleList = self.getCapsules(gameState)

    for capsule in capsuleList:
      capsuleDict[capsule] = True

    if self.nextAttackingPoint == None:
      # Set a random attacking point
      self.nextAttackingPoint = random.choice(self.ownOffensiveEntrances)

    
    previousState = self.getPreviousObservation()
    if previousState != None:
      previousFoodList = self.getFood(previousState).asList()
      if currentPosition in previousFoodList:
        self.offensiveFoodEaten += 1
        # print("Ate food")
      previousAgentState = previousState.getAgentState(self.index)
      if previousAgentState.isPacman and not currentAgentState.isPacman:
        # If pacman returned back for any reason, set a new attacking point
        self.nextAttackingPoint = random.choice(self.ownOffensiveEntrances)

    beingChased = False
    teammateBeingChased = False
    

    ghostAgents = []
    seenInvaders = []
    closestInvader = None
    ghostPositions = []

    # If scared, just play offense
    if currentAgentState.scaredTimer > 0 and not currentAgentState.isPacman:
      self.offensiveFoodEaten = 0
      action = aStarSearchToLocation(gameState, self.index, self.nextAttackingPoint, False, True)
      return action

    for agent in self.getOpponents(gameState):
      enemy = gameState.getAgentState(agent)
      if enemy.getPosition() != None:
        if currentAgentState.isPacman:
          if enemy.scaredTimer <= 5:
            if util.manhattanDistance(currentPosition, enemy.getPosition()) <= 3:
              beingChased = True
              ghostAgents.append(enemy)
              ghostPositions.append(enemy.getPosition())
            if teammateState.isPacman and util.manhattanDistance(teammatePosition, enemy.getPosition()) <= 3:
              teammateBeingChased = True
              # print("Teammate being chased!")
            # ghostDistances.append(self.distancer.getDistance(currentPosition, enemy.getPosition()))

        # If currently the agent is on defense and it sees a pacman, chase
        else:
          if enemy.isPacman and currentScore < 0:
            # Only chase pacman if the defender teammate is not closer to the enemy or it is very close to the enemy

            # if (util.manhattanDistance(gameState.getAgentPosition(teammateIndex), enemy.getPosition()) > util.manhattanDistance(currentPosition, enemy.getPosition())) or (util.manhattanDistance(currentPosition, enemy.getPosition()) <= 3):
            if (util.manhattanDistance(currentPosition, enemy.getPosition()) <= 2 and (util.manhattanDistance(teammatePosition, enemy.getPosition()) > util.manhattanDistance(currentPosition, enemy.getPosition()))):
              pacmanBlockingPosition = getPacmanBlockingPosition(gameState.isOnRedTeam(self.index), enemy.getPosition(), currentPosition, self.wallsDict)
              action = aStarSearchToLocation(gameState, self.index, pacmanBlockingPosition, self.isScared)
              # print ('eval time for phantomtroupe offensive mdp agent %d: %.4f' % (self.index, time.time() - start))
              # print("Chasing pacman")
              return action
          else:
            # If there is a defender very close to this entrance, choose antother entrance
            if util.manhattanDistance(enemy.getPosition(), currentPosition) <= 3 and currentPosition in self.entrancesDict and enemy.scaredTimer <= 1:
              # Choosing another entrance
              # print("Choose another entrance")
              self.nextAttackingPoint = random.choice(self.ownOffensiveEntrances)
              


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
      # Currently just checking if its eaten a certain number of food. Maybe we can come up with a formula
      # to let it decide on its own when to go back to store food.
      if self.offensiveFoodEaten >= 6:
        self.storeFood = True

      # if self.index == 0:
      #   print("is pacman")

      ghostDistanceRewardDict = ghostDistancesRewardDict(self.offensivePositions, ghostPositions)
      # if self.index == 0:
      #   print("Performing value iterations for red agent")
      closeToGhostFoodDict = findCloseFoodsToGhost(ghostPositions, foodDict, 5)
      action = performValueIteration(self.offensivePositions,self.legalOffensiveActions,self.discountFactor,currentPosition,
      foodDict, capsuleDict, self.entrancesDict, self.storeFood, beingChased, ghostAgents, self.offensiveFoodEaten, gameState, currentDirection,
      ghostPositions, self.wallsDict, teammatePosition, self.numWallsDict, ghostDistanceRewardDict, self.totalFoodCount, teammateBeingChased,
      closeToGhostFoodDict, currentScore, currentAgentState.scaredTimer)
      # print ('eval time for phantomtroupe offensive mdp agent %d: %.4f' % (self.index, time.time() - start))
      # if self.index == 0:
      #   print("Policy for red agent: ", action)
      return action[0]
    else:
      self.offensiveFoodEaten = 0
      self.storeFood = False
          


      

      

    # If this occurs, then agent is making its way to the center of the map to attack.
    # action = aStarSearchToLocation(gameState, self.index, self.offensiveMiddle, self.isScared)
    # action = aStarSearchToLocation(gameState, self.index, self.entrances[len(self.entrances)-1], self.isScared)
    # if action == 'Stop':
    #   if gameState.isOnRedTeam(self.index):
    #     action = 'East'
    #   else:
    #     action = 'West'

    action = aStarSearchToLocation(gameState, self.index, self.nextAttackingPoint, self.isScared)




    # print ('eval time for phantomtroupe offensive mdp agent %d: %.4f' % (self.index, time.time() - start))
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
def aStarSearchToLocation(gameState, agentIndex, location, isScared=False, isOffensive=False, blocking=False):
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
        if not blocking:
          if currentStatePosition == location:

              if len(currentStatePath) == 0: # Return STOP if the location has been reached and there is no more path.
                return "Stop"
              return currentStatePath[0] # Return the first action on the path  
        else:
          if currentStatePosition[1] == location[1]:
            teamIsRed = gameState.isOnRedTeam(agentIndex)

            if (teamIsRed and currentStatePosition[0] >= location[0]) or (not teamIsRed and currentStatePosition[0] <= location[0]):

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

# turn entrances into a dictionary for O(1) access
def makeEntrancesDict(entrances):
  entrancesDict = {}
  for entrance in entrances:
    entrancesDict[entrance] = True

  return entrancesDict

# turn walls into a dictionary for O(1) access
def makeWallsDict(walls):
  wallDict = {}
  for wall in walls:
    wallDict[wall] = True

  return wallDict


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

# Get the entrances of this specific agent
def getOwnEntrances(entrances, teamIndex):
  topEntrances = entrances[:len(entrances)//2]
  bottomEntrances = entrances[len(entrances)//2:]

  if teamIndex == 0:
    return topEntrances
  return bottomEntrances

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
    xPosition, yPosition = state
    # xPosition = state[0]
    # yPosition = state[1]

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
def generateSuccessor(gameState, action, currentDirection):
  x, y = gameState
  # x = gameState[0]
  # y = gameState[1]
  if action == Directions.NORTH:
    return ((x,y+1), Directions.NORTH)
  elif action == Directions.EAST:
    return ((x+1,y), Directions.EAST)
  elif action == Directions.SOUTH:
    return ((x,y-1), Directions.SOUTH)
  elif action == Directions.WEST:
    return ((x-1,y), Directions.WEST)
  else:
    return ((x,y), currentDirection)

# MDP function to calculate the reward.
def calculateMDPReward(state, foodDict, capsuleDict, entrancesDict, storeFood, beingChased, ghostAgents, offensiveFoodEaten, gameState, previousDirection,
currentDirection, totalFeatureCalculatingTime, ghostPositions, wallsDict, teammatePosition, numWallsDict,
ghostDistanceRewardDict, totalFoodCount, teammateBeingChased, closeToGhostFoodDict, currentScore, agentScaredTimer):
  reward = 0

  # foodAndCapsuleTime = time.time()

  foodReward = 0.0
  # if not beingChased:
  #   foodReward += 10 + 5 * (totalFoodCount - len(foodDict))
  #   # reward += (len(foodDict) + offensiveFoodEaten) * 2
  # else:
  foodReward += 10 + 5 * (totalFoodCount - len(foodDict)) / (offensiveFoodEaten+1)
  
  if state in foodDict:
    if state in closeToGhostFoodDict:
      reward += foodReward/2
    else:
      reward += foodReward

    # reward += 5
    # reward += len(foodDict) / (offensiveFoodEaten + 1)
      # reward += len(foodDict) / (offensiveFoodEaten + 1)
    
  if state in capsuleDict:
    # Very good to get capsule if being chased
    if beingChased or teammateBeingChased:
      reward += foodReward * totalFoodCount * 6
    # reward += foodReward / 2

  # Try to split up teammates
  distanceToTeammate = util.manhattanDistance(state, teammatePosition)
  # if distanceToTeammate > 3:
  if distanceToTeammate <= 3:
    # other agent is being chased than - 50
    reward -= 5 * (10/(distanceToTeammate+1))

    if teammateBeingChased:
      # reward -= (totalFoodCount*10)/(distanceToTeammate+1)
      reward -= 1000

  # if teammateBeingChased:
  #   reward -= (totalFoodCount*10)/(distanceToTeammate+1)

  if state in entrancesDict:
    # reward +=len(foodDict) * totalFoodCount/10
    
    # # If you've eaten a quarter of the food, going back to entrance is even better
    # if offensiveFoodEaten >= totalFoodCount/4 and beingChased:
    if offensiveFoodEaten >= totalFoodCount/4:
      reward += foodReward * totalFoodCount * 2

    if currentScore + offensiveFoodEaten > 0 and agentScaredTimer <= 10:
      reward += foodReward * totalFoodCount * 4

    # If there is only 2 food left, go back as soon as possible.
    if len(foodDict) <= 2:
      reward += foodReward * totalFoodCount * 10


  if beingChased:
    # If the ghost is being chased, give a lot of reward for returning home to store the food.


    # if state in entrancesDict:
    if state in entrancesDict and offensiveFoodEaten > 0:
      # reward += 10*offensiveFoodEaten
      reward += foodReward * 2
      # reward += foodReward * totalFoodCount * 6

    # To save time, calculating reward of ghost distance reward dict in SEPARATE function called
    # ghostDistancesRewardDict and simple grabbing the reward to subtract from this state.
    # To edit the ghost reward, PLEASE LOOK AT THIS FUNCTION!
    reward -= ghostDistanceRewardDict[state]

    if numWallsDict[state] >= 3:
      reward -= 10



  return reward

# Method to perform the value iteration. Returns an action.
def performValueIteration(offensivePositions, legalOffensiveActions, discountFactor,
currentPosition, foodDict, capsuleDict, entrancesDict, storeFood, beingChased, ghostAgents, offensiveFoodEaten, gameState, currentDirection,
ghostPositions, wallsDict, teammatePosition, numWallsDict, ghostDistanceRewardDict, totalFoodCount, teammateBeingChased,
closeToGhostFoodDict, currentScore, agentScaredTimer):

  # Check how long it takes to perform value iteration
  start = time.time()
  
  # optimal policy which stores action and q value as a tuple
  optimalPolicies = {state: ("Stop", 0.0) for state in offensivePositions}

  # Constants for indexes of the actions and q values
  ACTION_INDEX = 0
  Q_VALUE_INDEX = 1

  numIterations = 150

  # print("offensive positions length:", len(offensivePositions))
  # print("actions length:", len(offensivePositions))

  # totalRewardTime = 0

  totalFeatureCalculatingTime = [0, 0, 0, 0, 0, 0, 0]


  # Value iteration
  for i in range(numIterations):
    previousPolicies = optimalPolicies.copy()

    for state in offensivePositions:
      QDict = {}

      # print("QDict now: ",QDict)
      # print("State now:", state)
      # print("legal offensive actions for this state: ",legalOffensiveActions[state])

      # If there are no legal offensive actions then just continue and dont bother with this state
      if len(legalOffensiveActions[state]) == 0:
        continue
      for action in legalOffensiveActions[state]:
        childState = generateSuccessor(state, action, currentDirection) #Method that gets the child state when applying action to state
        # calculateReward function, bellmans equation here. R(s) + gamma * next state utility
        # startCalculatingReward = time.time()
        QDict[action] = calculateMDPReward(state, foodDict, capsuleDict, entrancesDict, storeFood, beingChased,
        ghostAgents, offensiveFoodEaten, gameState, currentDirection, childState[1], totalFeatureCalculatingTime, ghostPositions,
        wallsDict, teammatePosition, numWallsDict, ghostDistanceRewardDict, totalFoodCount, teammateBeingChased,
        closeToGhostFoodDict, currentScore, agentScaredTimer) + discountFactor * previousPolicies[childState[0]][Q_VALUE_INDEX]
        # totalRewardTime += (time.time() - startCalculatingReward)
        # print("action set:", QDict[action])

      # print("QDict after: ",QDict)
      
      # if len(QDict) == 0:
      #  print("QDict is empty!")
      optimalPolicies[state] = (getActionOfMaxQValue(QDict), max(QDict.values()))
  
  # Print how many enemy ghosts the offensive agent can see
  # print('Number of ghost: ', str(len(ghostAgents))) 

  # Print how long it takes to perform value iteration
  # print('Value Iteration time for phantomtroupe offensive mpd agent: ', str(time.time() - start))

  # print('Total time to calculate rewards: ', totalRewardTime)
  # print('Total time to calculate ghosts reward: ', totalFeatureCalculatingTime[0])
  # print('Total time to calculate entrances reward: ', totalFeatureCalculatingTime[1])
  # print('Total time to calculate food and capsules reward: ', totalFeatureCalculatingTime[2])
  # print('Total time to calculate walls reward: ', totalFeatureCalculatingTime[3])
  # print('Total time to calculate opposite directions reward: ', totalFeatureCalculatingTime[4])
  # print('Total time to calculate entrances when scared reward: ', totalFeatureCalculatingTime[5])
  # print('Total time to calculate being scared reward: ', totalFeatureCalculatingTime[6])

  # return optimalPolicies[currentPosition][ACTION_INDEX]
  # return whole optimal policy for testing
  return optimalPolicies[currentPosition]

# Method used to quickly get the key of the max q value in the q dictionary.
def getActionOfMaxQValue(QDict):
  qValues = list(QDict.values())
  actions = list(QDict.keys())
  return actions[qValues.index(max(qValues))]

# Returns true if the new direction is the opposite of the previous direction
def checkOppositeDirections(previousDirection, currentDirection):
  if previousDirection == Directions.NORTH:
    if currentDirection == Directions.SOUTH:
      return True
  if previousDirection == Directions.EAST:
    if currentDirection == Directions.WEST:
      return True
  if previousDirection == Directions.SOUTH:
    if currentDirection == Directions.NORTH:
      return True
  if previousDirection == Directions.WEST:
    if currentDirection == Directions.EAST:
      return True
  return False

# Checks how many walls are around pacman
def checkSurroundingWalls(position, wallsDict):
  x, y = position
  # x = position[0]
  # y = position[1]
  numWalls = 0

  walls = [(x, y + 1), (x + 1, y), (x, y - 1), (x - 1, y)]

  for wall in walls:
    if wall in wallsDict:
      numWalls += 1

  return numWalls

  # if numWalls >= 3:
  #   return True
  # return False

# Method that creates a dictionary which takes a position as a key and returns a value that is how many walls surround it
def makeNumWallsDict(offensiveStates, wallsDict):
  numWallsDict = {}
  for state in offensiveStates:
    numWallsDict[state] = checkSurroundingWalls(state, wallsDict)
  return numWallsDict


# This method returns the index of the agent in their team
def getTeamIndex(teamIsRed, team, index):
  # add 1 if red team
  if teamIsRed:
    index += 1
  # print("agent index:", index)
  # print("team: ", team)
  # print("Team:", team)
  # print(team[0])
  # print(index)
  if team[0] == index:
    # print("Returning 0")
    return 0
  else:
    # print("Returning 1")
    return 1

# Method that returns ghost distances reward dictionary given a position and all the ghost positions
def ghostDistancesRewardDict(possibleOffensivePositions, ghostPositions):
  ghostDistanceRewardDict = {position: 0 for position in possibleOffensivePositions}
  for position in possibleOffensivePositions:
    for ghostPosition in ghostPositions:
      distance = util.manhattanDistance(position, ghostPosition)
      if distance <= 1:
        ghostDistanceRewardDict[position] = sys.maxsize
      else:
        ghostDistanceRewardDict[position] += 20/distance

  return ghostDistanceRewardDict

# get location of enemy pacman but be ahead of it to block it
def getPacmanBlockingPosition(teamIsRed, pacmanPosition, currentPosition, wallsDict):
  # get pacman position
  # if team is red
  #   check pacmanPosition x value + 1 is not a wall
  #   if not a wall
  #     return pacmanPosition but with x value + 1
  #   else
  #     return pacmanPosition
  # else (if team is blue)
  #   check pacmanPosition x value - 1 is not a wall
  #   if not a wall
  #     return pacman position but with x value - 1
  #   else
  #     return pacmanPostion

  # return currentPosition x value, but y position of pacman

  blockingPosition = pacmanPosition
  if teamIsRed:
    blockingPosition = (pacmanPosition[0]+1, pacmanPosition[1])
  else:
    blockingPosition = (pacmanPosition[0]-1, pacmanPosition[1])

  if blockingPosition in wallsDict:
    return pacmanPosition
  else:
    return blockingPosition
  

# This will return the closest entrance to a food that is lost
def getLostFoodClosestEntrance(lostFoodPosition, entrancesDict):
  # entrance with the closest y value to the food
  # return currentPosition x value, but y position of pacman
  closestDistance = sys.maxsize
  closestEntrance = None
  for entrance in entrancesDict:
    distance = util.manhattanDistance(lostFoodPosition, entrance)
    if distance < closestDistance:
      closestDistance = distance
      closestEntrance = entrance
  
  return closestEntrance

# Find all foods that are within a certain range of the ghost
def findCloseFoodsToGhost(ghostPositions, foodDict, distanceRange):
  closeToGhostFoodDict = {}

  for ghostPosition in ghostPositions:

    ghostX, ghostY = int(ghostPosition[0]), int(ghostPosition[1])

    # print(ghostX-distanceRange)
    # print(ghostX+distanceRange+1)

    for x in range(ghostX-distanceRange, ghostX+distanceRange+1):
      for y in range(ghostY-distanceRange, ghostY+distanceRange+1):
        position = (x,y)
        if position in foodDict:
          closeToGhostFoodDict[position] = True

  return closeToGhostFoodDict
