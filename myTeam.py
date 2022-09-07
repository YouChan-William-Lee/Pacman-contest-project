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
from util import nearestPoint
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
    self.entrances = findEntrances(gameState.isOnRedTeam(self.index), gameState)
    self.middle = findMiddleOfMap(gameState.isOnRedTeam(self.index), gameState)
    self.walls = gameState.getWalls().asList()
    # self.entranceToPatrol = self.middle   
    self.entranceToPatrol = self.entrances[len(self.entrances)-1]   
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

    print(bestActions)

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
  
  print("Implement Offensive Agent here")

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)



    return "Stop"

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  # print("Implement Defensive agent here")

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    print("NEW ACTION ----------")

    start = time.time()

    action = None

    currentPosition = gameState.getAgentPosition(self.index)
    # if currentPosition in self.entrances:
    #   print("Yeet")
    # # In this case, 
    # else:


    print("SEEN INVADERS")
    # Find all seen invaders
    seenInvaders = []
    for enemy in self.getOpponents(gameState):
      invader = gameState.getAgentState(enemy)
      if invader.isPacman and invader.getPosition() != None:
        seenInvaders.append(invader)


    if len(seenInvaders) > 0:
      closestInvader = None
      closestDistanceToInvader = sys.maxsize
      for invader in seenInvaders:
        distance = util.manhattanDistance(currentPosition, invader.getPosition())
        if distance < closestDistanceToInvader:
          closestInvader = invader
          closestDistanceToInvader = distance
      
      print("ASTAR to invader:")
      print(closestInvader.getPosition())
      action = aStarSearchToLocation(gameState, self.index, closestInvader.getPosition())
      print(action)
      return action

    print("GETTING PREV OBSERVATION")
    # If it reaches here, no direct invaders were found
    if self.getPreviousObservation():
      print("Check food here")
      eatenFoods = checkEatenFoods(self.red, self.getPreviousObservation(), gameState)
      closestFood = None
      
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
        print("DOING FOOD ASTAR")
        action = aStarSearchToLocation(gameState, self.index, closestFood)
        print(action)
        return action
    

    print("A STAR TO ENTRANCE")
    # agent = gameState.getAgentState(agentIndex)
    # agent.isPacman
    action = aStarSearchToLocation(gameState, self.index, self.entranceToPatrol)
    if action == 'Stop':
      # self.entranceToPatrol = random.choice(self.entrances)
      self.entranceToPatrol = (self.entrances[len(self.entrances) - 1])

    print ('eval time for agent %d: %.4f' % (self.index, time.time() - start))
    print(action)
    return action
    
# def aStarSearchToLocation(gameState, agentIndex, location):
#     """Search the node that has the lowest combined cost and heuristic first."""
#     "*** YOUR CODE HERE ***"

#     heuristic = util.manhattanDistance
#     def expand(agentIndex, node):
#         # A list for all successors' node of current node
#         successors = []
#         currentState, currentParentState, currentAction, currentCost = node
#         for action in currentState.getLegalActions(agentIndex):
#           successor = currentState.generateSuccessor(agentIndex, action)
#           successors.append((successor, currentState, currentAction + [action], currentCost + stepCost))

#         return successors

#     # pathToGoal includes all directions to get to the goal state
#     pathToGoal = []
#     # costToGoal includes all costs to get to the goal state
#     costToGoal = 0
#     # Assign problem's initial state
#     initialState = gameState.getAgentPosition(agentIndex)
#     # There is no parent state of the start state, so None
#     parentState = None
    
#     # Use a priority queue for A*
#     # structure of frontier's node -> (state, parent state, path, cost), heuristics
#     # If cost is not included in the tuple, then when it is popped, the cost cannot be reachable
#     frontier = util.PriorityQueue()
#     frontier.push((initialState, parentState, pathToGoal, costToGoal), heuristic(initialState, problem))

#     # reachedStates includes all reached states
#     # Dictionary is used to save state with corresponding cost
#     reachedStates = {}

#     # Keep searching until frontier is empty
#     while not frontier.isEmpty():
#         node = frontier.pop()
#         state, parentState, currentPath, currentCost = node
#         # Assgin the cost to the state
#         reachedStates[state] = currentCost

#         # Check if current state is the goal state
#         if currentStatePosition == location:

#           if len(currentPath) == 0: # Return STOP if the location has been reached and there is no more path.
#             return "Stop"
#           return currentPath[0]

#         # Check successors of current state
#         for successor, parentState, action, stepCost in expand(problem, node):
#             # Find successors satisfying either condition1 or condition2
#             # condition1 -> Do not visit again
#             # conditoin2 -> Visit again if only new cost is less than current cost including heuristics
#             condition1 = successor not in reachedStates
#             condition2 = (successor in reachedStates) and (reachedStates[successor] > (stepCost + heuristic(successor, problem)))
            
#             if condition1 or condition2:
#                 # Add current successor to reachedStates with corresponding cost, so not to visit again
#                 reachedStates[successor] = stepCost
#                 # Push current successor's state, parent state, path, and cost including heuristics to frontier
#                 frontier.push((successor, state, action, stepCost), stepCost + heuristic(successor, problem))
                
#     return pathToGoal

def aStarSearchToLocation(gameState, agentIndex, location):
  """Search the node that has the lowest combined cost and heuristic first."""
  "*** YOUR CODE HERE ***"

  print("INSIDE OF ASTAR agent is", agentIndex, location)

  heuristic=util.manhattanDistance

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

      if currentStatePosition == location:

          if len(currentStatePath) == 0: # Return STOP if the location has been reached and there is no more path.
            return "Stop"
          return currentStatePath[0] # Return the first action on the path   

      for action in currentGameState.getLegalActions(agentIndex):
        successor = currentGameState.generateSuccessor(agentIndex, action)

        childPosition = successor.getAgentPosition(agentIndex)

        # This will make the defensive agent never go onto the other side.
        agent = currentGameState.getAgentState(agentIndex)
        if agent.isPacman:
          # print("isPacman")
          continue
        # else:
          # print("NotPacman")

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
  
  return []
  # return "Stop"

# Method that finds all entrances
def findEntrances(teamIsRed, gameState):
  
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

    print("RED TEAM PRINTING ENTRANCES")
    print(entrances)
    
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

    print("BLUE TEAM PRINTING ENTRANCES")
    print(entrances)

  return entrances

# Find the middle of the map by finding all entrances and getting the middle entrance.
# WARNING: This may mess up in cases where an entrance is unreachable
def findMiddleOfMap(teamIsRed, gameState):
  entrances = findEntrances(teamIsRed, gameState)
  return entrances[math.ceil(len(entrances)/2)] 

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
