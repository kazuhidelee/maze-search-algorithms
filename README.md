# 📌 Maze Search Algorithms

This repository contains implementations of four search algorithms to navigate a maze:

- **Breadth-First Search (BFS)**  
- **Depth-First Search (DFS)**  
- **Uniform Cost Search (UCS)**  
- **A* Search (A-Star)**  

The algorithms are implemented in `Agent.py`, and they solve mazes defined in `.test` files.

## 📂 Project Structure

- `Agent.py` – Contains the search algorithms and helper functions.  
- `Maze.py` – Handles maze parsing and representation.
- `util.py` – Utility functions.
- `tests/` – Contains `.test` files representing mazes.  

## 🚀 Getting Started

### 1️⃣ Setup Environment  
Create a virtual environment and install dependencies:

```bash
python -m venv venv  
source venv/bin/activate  # On Windows use `venv\Scripts\activate`  
python -m pip install -r requirements.txt  
```

### 2️⃣ Running the Agent  
Run the search algorithms on a maze file:

```bash
python LocalTest.py {TESTNUMBER} {SEARCH ALGORITHM}.
```
You can also create your own maze and put them as a test file

## 🔍 Search Algorithms

| Algorithm | Description |
|-----------|------------|
| **BFS** | Explores nodes level by level using a FIFO queue. |
| **DFS** | Explores as deep as possible using a stack. |
| **UCS** | Expands the least-cost node first using a priority queue. |
| **A*** | Uses a heuristic (Euclidean distance) to find the shortest path efficiently. |

## 📝 Notes

- The A* heuristic function uses the Euclidean distance to the closest goal.  
- The agent expands nodes without updating existing ones in the frontier.  


