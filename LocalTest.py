"""
The test runner - use this to check if your implementation seems sound.
This test runner does NOT automatically evaluate your implementation, it just visualizes it.
"""
import argparse
from Maze import Maze
from Agent import Agent
from util import SearchResult
import sys


# Load a test maze and create the agent
def load_agent(test_number):
    testfile = "./Tests/Test"+test_number+".test"
    test_maze = Maze(testfile)
    agent = Agent(test_maze)
    return agent


def print_result(test_number: int, algo: str, result: SearchResult):
    print(f"Results for {algo} for test case {test_number}")
    print(f"Path found: {result[0]}")
    if result[1] is None:
        print(f"No path to goal found")
    else:
        print(f"Final path: {result[1][0]}")
        print(f"Action Sequence: {result[1][1]}")
    print(f"Expansion history: {result[2]}")

def run_test(test_number, algo=None):
    test_agent = load_agent(test_number)
    algo = algo.upper()
    if algo == "BFS":
        result = test_agent.bfs()
    elif algo == "DFS":
        result = test_agent.dfs()
    elif algo == "UCS":
        result = test_agent.ucs()
    elif algo == "ASTAR":
        result = test_agent.astar()
    else:
        result = None
    if result is not None:
        print_result(test_number, algo, result)
        test_agent.visualize_expansion(result[1][0])


# Use this block of code to run the test via a command line
# Example: python3 LocalTest.py 0 BFS
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('test_number', type=str,
                        help="Test number in Tests/")
    parser.add_argument('algo', choices=['BFS', 'DFS', 'UCS', 'ASTAR'],
                        help="The search algorithm to test")
    args = parser.parse_args()
    run_test(args.test_number, args.algo)

"""
If you would rather run this test from an IDE instead of the command line, feel free to comment out the If block
You can simply call the run_test function directly. For example:
run_test('0', 'BFS')
"""
