## Assignment Goals:

- Deepen understanding of state space generation
- Practice implementation of an efficient search algorithm

## Assignment Summary:
This assignment is about solving the 8-tile puzzle we have discussed in class. The 8-tile puzzle was invented and popularized by Noyes Palmer Chapman in the 1870s. It is played on a 3x3 grid with 8 tiles labeled 1 through 8 and an empty grid. The goal is to rearrange the tiles so that they are in order.

You solve the puzzle by moving the tiles around. For each step, you can only move one of the neighbor tiles (left, right, top, bottom) into an empty grid. And all tiles must stay in the 3x3 grid. An example is shown in the picture below. In this example, there are only 2 valid moves, i.e., either moving 6 down or moving 1 right.

Given these rules for the puzzle, you will generate a state space and solve this puzzle using the A* search algorithm. Note that not all 8-tile puzzles are solvable. For this assignment, it is safe to assume that the input puzzle is always solvable.