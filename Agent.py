"""
This file implements the Agent class.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar,Iterator
from queue import Queue, LifoQueue, PriorityQueue
from Maze import Action, Maze
from util import Coord, SearchResult
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt



class Node:
    """
    First, we define the Node class, that imitates the nodes mentioned in lecture. 
    A node represents a step along the way of searching for the goal state in a search tree/graph.
    Remember - a node is not the same as a coordinate. It contains more information.
    """

    def __init__(self, coord: Coord, parent:'Node' | None = None,
                 action: Action | None = None, path_cost=0):
        self.coord = coord
        self.parent = parent
        self.action = action
        self.path_cost = path_cost


    # Given a node, get a path back to the start node by recording the actions taken from the parent node
    def trace_back(self) -> tuple[list[Coord], list[Action]]:
        coord_path = []
        actions = []
        trace: Node = self
        while trace is not None:
            coord_path.append(trace.coord)
            if trace.action is not None:
                actions.append(trace.action)
            trace = trace.parent
        coord_path.reverse()
        actions.reverse()
        return coord_path, actions


    """
    Algorithms like UCS require us to compare and order nodes.
    Since nodes are objects, we can define a custom comparator via operator overriding.
    The three functions below override the standard "==", "<" and ">" operators for the node class.
    This allows us to implement logic like node1 > node2.
    More importantly, these operators are used by the queue data structures we imported at the start.
    """
    def __eq__(self, other):
        return self.path_cost == other.path_cost


    def __lt__(self, other):
        return self.path_cost < other.path_cost


    def __gt__(self, other):
        return self.path_cost > other.path_cost
    
    
    """
    The ability to hash a node allows you to use them as members of a `set` or
    as a key in a `dict`.
    """
    def __hash__(self):
        return hash((self.coord,
                     self.parent.coord if self.parent is not None else None,
                     self.action if self.action is not None else None,
                     self.path_cost))


"""
Next we define the AStar node. Note that the AStar node is identical to a regular node but has two key differences
1. The AStar node has an end coordinate
2. The AStar node has a heuristic value, computed using a heuristic function
"""
class AStarNode(Node):
    """
    HINT: Make sure this is set first before using the heuristics function
    Remember that you need to set this for the entire class, not just one object
    """
    END_COORDS = ClassVar[set[Coord]]
    
    def __init__(self, coord: Coord, parent: Node | None = None,
                    action: Action | None = None, cost: int = 0):
        super().__init__(coord, parent, action, cost)
        # store the heuristics value for this node
        self.h_val = self.heuristic_function()
        # print(self.h_val)


    """
    TODO - complete the heuristic function
    Implement the heuristic function for the AStar Node by returning the minimum euclidean (straight line) distance
    between the AStar Node's coordinate and the END_COORDS。
    Hint: You may use numpy's sqrt function, and the builtin min function
    """
    def heuristic_function(self):
        # TODO here
       res = float('inf')
       for end in AStarNode.END_COORDS:
        dis = np.sqrt((self.coord[0] - end[0]) ** 2 + (self.coord[1] - end[1]) ** 2) 
        res = min(dis, res)
                 
       return res


    """
    TODO - complete the operator overloads for the AStar node class
    Unlike the regular node, AStar nodes are not compared using just the path costs。
    Complete the operator overloads for the AStar nodes.
    Hint: You can use the same syntax as the overload in the base node class。
    """
    def __eq__(self, other):
        # TODO here
        return (self.path_cost + self.h_val) == (other.path_cost + other.h_val)


    def __lt__(self, other):
        # TODO here
        return (self.path_cost + self.h_val) < (other.path_cost + other.h_val)


    def __gt__(self, other):
        # TODO here
        return (self.path_cost + self.h_val) > (other.path_cost + other.h_val)
    
    
    """
    The ability to hash a node allows you to use them as members of a `set` or
    as a key in a `dict`.
    """
    def __hash__(self):
        return hash((self.coord,
                     self.parent.coord if self.parent is not None else None,
                     self.action if self.action is not None else None,
                     self.path_cost,
                     self.h_val,
                     self.END_COORDS))


@dataclass
class Agent:
    """This is the Agent class. This class mimics an agent that explores the
    maze. The agent has just two attributes - the maze that it is in, and the
    node expansion history. The expansion history is used just to evaluate your
    implementation.
    """
    maze: Maze
    expansion_history: list[Coord] = field(default_factory=list)

    # We want to reset this every time we use a new algorithm
    def clear_expansion_history(self):
        self.expansion_history.clear()


    # Visualize the maze and how it was explored in matplotlib
    def visualize_expansion(self, path):
        plt.subplot(1, 1, 1)
        blocks = np.zeros((self.maze.height, self.maze.width))
        blocks[:] = np.nan
        for co_ord in self.maze.pits:
            blocks[co_ord[1], co_ord[0]] = 2

        expansion_cval = np.zeros((self.maze.height, self.maze.width))

        for i, coord in enumerate(self.expansion_history):
            expansion_cval[coord[1], coord[0]] = len(self.expansion_history) - i + len(self.expansion_history)

        plt.pcolormesh(
            expansion_cval,
            shading='flat',
            edgecolors='k', linewidths=1, cmap='Blues')

        cmap = colors.ListedColormap(['grey', 'grey'])

        plt.pcolormesh(
            blocks,
            shading='flat',
            edgecolors='k', linewidths=1, cmap=cmap)

        start = self.maze.start
        ends = self.maze.end

        # Plot start and end points
        plt.scatter(start[0] + 0.5, start[1] + 0.5, color='red', s=100, marker='o', label='Start')
        for end in ends:
            plt.scatter(end[0] + 0.5, end[1] + 0.5, color='gold', s=100, marker=(5, 1), label='End')

        plt.title("Maze Plot")
        plt.xlabel("X")
        plt.ylabel("Y", rotation=0)

        plt.xticks(np.arange(0 + 0.5, expansion_cval.shape[1] + 0.5), np.arange(0, expansion_cval.shape[1]))
        plt.yticks(np.arange(0 + 0.5, expansion_cval.shape[0] + 0.5), np.arange(0, expansion_cval.shape[0]))

        # Plot the path only if it exists
        if path is not None:
            for i in range(len(path) - 1):
                x, y = path[i]
                next_x, next_y = path[i + 1]
                plt.annotate('', xy=(next_x + 0.5, next_y + 0.5), xytext=(x + 0.5, y + 0.5),
                             arrowprops=dict(color='g', arrowstyle='->', lw=2))
        plt.show()


    """
    TODO: Complete the goal_test function
    Input: a node object
    Returns: A boolean indicating whether or not the node corresponds to a goal
    Hint: The agent has the maze object as an attribute. The maze object has a set of end coordinates
    You can use the 'in' operator to determine if an object is in a set
    """
    def goal_test(self, node: Node) -> bool:
        # Your implementation here :)
        return node.coord in self.maze.end
    

    """
    TODO: Complete the expand_node function
    For each neighbouring node that can be reached from the current node within one action, 'yield' the node.
    Hints:
    1.  Use self.maze.valid_ordered_action(...) to obtain all valid action for a given coordinate
    2.  Use Maze.resulting_coord(...) to compute the new states. Note the capital M is important, since it's a
        class function
    3.  Use yield to temporarily return a node to the caller function while saving the context of this function.
        When expand is called again, execution will be resumed from where it stopped.
        Follow this link to learn more:
        https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    4.  Take advantage of polymorphic construction. You need to ensure that a Node object yields Node objects, while
        an AStarNode object yields AStarNodes. You can use type(node) to do this.
    """
    def expand_node(self, node: Node) -> Iterator[Node]:
        self.expansion_history.append(node.coord)
        s = node.coord
        for action in self.maze.valid_ordered_action(s):
            # TODO Here
            # Use the maze object's resulting_coord function to compute the new state after taking the action
            # Compute the new cost of getting to this state via the current path and action
            new_state = self.maze.resulting_coord(node.coord, action)
            new_cost = node.path_cost + self.maze.action_cost(action)
            # print(f"Expanding Node: {node.coord}, Action: {action} -> New State: {new_state}, New Cost: {new_cost}")
            yield type(node)(new_state, node, action, new_cost)
        # Your implementation :)


    """
    TODO : Complete the search algorithms below!
    You will need to complete the following search algorithms: BFS, DFS, UCS and AStar

    Hints:
    1. Aside from the algorithms that you'll be evaluated on, we've included a function called "best_first_search"
    Best first search is a general search method, that is optional to implement, however we would highly recommend it
    You are free to use best_first_search in any other search algorithm
    2. We've provided three types of Queues for you to use: Queue, LifoQueue and PriorityQueue. You do not need to
    worry about implementing these data structures
    3. A rough template for each algorithm is as follows
    - create a start node using the starting coordinates of the maze
    - initialize an appropriate frontier
    - expand nodes and add to the frontier as needed
    - goal test as needed
    4. You might need a way to keep track of where you've already been
    


    Implement the generic best-first-search algorithm here. 

    Inputs: 
    1. A start node
    2. A frontier (i.e, a queue)

    Return a tuple of three items:
    1. A boolean indicating whether or not the goal is reachable from the start
    2. The path from the start of the maze to the end of the maze. If no path exists, return None 
       (Hint: see the trace_back function in the node class)    
    3. The list of expanded nodes
    """
    def best_first_search(self, node: Node, frontier: Queue[Node]) -> SearchResult:
        self.clear_expansion_history()
        # TODO : Your Implementation :)
        frontier.put(node)
        explored = set()
        while not frontier.empty():
            curr = frontier.get()
            if self.goal_test(curr):
                res = curr.trace_back()
                return True, res, self.expansion_history
            explored.add(curr.coord)
            for neighbor in self.expand_node(curr):
                if neighbor.coord not in explored:
                    frontier.put(neighbor)
        return False, None, self.expansion_history


    """
    Implement breadth-first-search here
    Return a tuple of three items:
    1. A boolean indicating whether or not the goal is reachable from the start
    2. The path from the start of the maze to the end of the maze. If no path exists, return None
        (Hint: see the trace_back function in the node class)
    3. The list of expanded nodes
    """
    def bfs(self) -> SearchResult:
        self.clear_expansion_history()
        # TODO : Your Implementation :)
        start_node = Node(self.maze.start, None, None, 0)
        frontier = Queue()
        frontier.put(start_node)
        explored = set()
        explored.add(start_node.coord)
        i = 1
        while not frontier.empty():
            # print("Current frontier contents:")
            # for f in list(frontier.queue): 
            #     print(f.coord)
            curr = frontier.get()
            # if curr.coord not in explored
            # print("neighbors:")
            for neighbor in self.expand_node(curr):
                if self.goal_test(neighbor):
                    res = neighbor.trace_back()
                    return True, res, self.expansion_history
                # print(neighbor.coord)
                if neighbor.coord not in explored:
                    frontier.put(neighbor)
                    explored.add(neighbor.coord)

        return False, None, self.expansion_history



    """
    Implement depth-first-search here
    Return a tuple of three items:
    1. A boolean indicating whether or not the goal is reachable from the start
    2. The path from the start of the maze to the end of the maze. If no path exists, return None
       (Hint: see the trace_back function in the node class)
    3. The list of expanded nodes
    
    NOTE：Although DFS is often implemented as a tree-like search, as discussed in lecture,
    for this programming assignment please implement it as a graph search, since we are applying
    it to a state space with cycles that must be handled in some way.
    """
    def dfs(self) -> SearchResult:
        self.clear_expansion_history()
        # TODO: Your Implementation :)
        start_node = Node(self.maze.start, None, None, 0)
        # print(start_node.coord)
        stack = LifoQueue()
        stack.put(start_node)
        reached = {start_node.coord: start_node}
        while not stack.empty(): 
            curr = stack.get()
            if self.goal_test(curr):
                res = curr.trace_back()
                return True, res, self.expansion_history
            for neighbor in self.expand_node(curr):
                s = neighbor.coord
                if s not in reached or neighbor.path_cost < reached[s].path_cost:
                    stack.put(neighbor)
                    reached[s] = neighbor 
        return False, None, self.expansion_history 


    """
    Implement uniform-cost-search here
    Return a tuple of three items:
    1. A boolean indicating whether or not the goal is reachable from the start
    2. The path from the start of the maze to the end of the maze. If no path exists, return None
       (Hint: see the trace_back function in the node class)
    3. The list of expanded nodes
    """
    def ucs(self) -> SearchResult:
        self.clear_expansion_history()
        # TODO: Your Implementation :)
        start_node = Node(self.maze.start, None, None, 0)
        frontier = PriorityQueue()
        frontier.put(start_node)
        reached = {start_node.coord: start_node}
        while not frontier.empty():
            curr = frontier.get()  
            if self.goal_test(curr):
                res = curr.trace_back()
                return True, res, self.expansion_history
        
            for neighbor in self.expand_node(curr):
                s = neighbor.coord
                if s not in reached or neighbor.path_cost < reached[s].path_cost:
                    frontier.put(neighbor)
                    reached[s] = neighbor
        return False, None, self.expansion_history 


    """
    Implement A* search here
    Return a tuple of three items:
    1. A boolean indicating whether or not the goal is reachable from the start
    2. The path from the start of the maze to the end of the maze. If no path exists, return None
        (Hint: see the trace_back function in the node class)
    3. The list of expanded nodes
    """
    def astar(self) -> SearchResult:
        self.clear_expansion_history()
        # TODO : Your Implementation :)
        AStarNode.END_COORDS = self.maze.end
        start_node = AStarNode(self.maze.start, None, None, 0)
        frontier = PriorityQueue()
        frontier.put((start_node.path_cost + start_node.h_val, start_node))  
        reached = {start_node.coord: start_node}

        while not frontier.empty():
            curr_cost, curr = frontier.get()
            if self.goal_test(curr):
                res = curr.trace_back()
                return True, res, self.expansion_history
            
            for neighbor in self.expand_node(curr):
                s = neighbor.coord
                total_cost = neighbor.path_cost + neighbor.h_val
                if s not in reached or neighbor.path_cost < reached[s].path_cost:
                    frontier.put((total_cost, neighbor))
                    reached[s] = neighbor
        return False, None, self.expansion_history

