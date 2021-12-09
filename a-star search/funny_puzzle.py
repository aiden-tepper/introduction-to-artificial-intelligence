import heapq


def succ_list(state):
    index = state.index(0)
    middle_boundary_swaps = {1: [0, 2, 4], 3: [0, 4, 6], 5: [2, 4, 8], 7: [4, 6, 8]}
    corner_boundary_swaps = {0: [1, 3], 2: [1, 5], 6: [3, 7], 8: [5, 7]}
    successors = []
    if index == 4:  # if empty grid in center, 4 successor states
        for i in range(1, 9, 2):
            successor = state[:]
            successor[index] = successor[i]
            successor[i] = 0
            successors.append(successor)
    elif index in middle_boundary_swaps:  # if empty grid in middle of boundary, 3 successor states
        for i in middle_boundary_swaps[index]:
            successor = state[:]
            successor[index] = successor[i]
            successor[i] = 0
            successors.append(successor)
    elif index in corner_boundary_swaps:  # if empty grid in corner, 2 successor states
        for i in corner_boundary_swaps[index]:
            successor = state[:]
            successor[index] = successor[i]
            successor[i] = 0
            successors.append(successor)
    return successors


def h_value(state):
    solved_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    h = 0
    for i in range(len(state)):
        if state[i] == 0:
            continue
        curr_x = int(i / 3)
        curr_y = i % 3
        goal_x = int(solved_state.index(state[i]) / 3)
        goal_y = solved_state.index(state[i]) % 3
        h += abs(curr_x - goal_x) + abs(curr_y - goal_y)
    return h


def print_succ(state):
    successors = succ_list(state)
    sorted_successors = []
    for successor in successors:
        h = h_value(successor)
        sorted_successors.append((successor, h))
    sorted_successors = sorted(sorted_successors, key=lambda s: s[0])
    for successor in sorted_successors:
        print(successor[0], 'h=' + str(successor[1]))


def print_solve(closed_list, max_len):
    solution = []
    move = closed_list[-1]
    while move[2][2] != -1:
        solution.append((move[1], move[2][1]))
        move = closed_list[move[2][2]]
    move = closed_list[0]
    solution.append((move[1], move[2][1]))
    solution = reversed(solution)
    for i, move in enumerate(solution):
        print(move[0], 'h=' + str(move[1]), 'moves:', i)
    # print('Max queue length:', max_len)


def solve(state):
    solved_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    open_list = []
    closed_list = []
    visited = {tuple(state)}
    h = h_value(state)
    heapq.heappush(open_list, (h, state, (0, h, -1)))
    max_len = 1

    while open_list:
        q = heapq.heappop(open_list)
        closed_list.append(q)
        parent_index = closed_list.index(q)
        if q[1] == solved_state:
            print_solve(closed_list, max_len)
            return
        successors = succ_list(q[1])
        for successor in successors:
            if tuple(successor) in visited:
                continue
            visited.add(tuple(successor))
            g = q[2][0] + 1
            h = h_value(successor)
            f = g+h
            heapq.heappush(open_list, (f, successor, (g, h, parent_index)))
        max_len = max(max_len, len(open_list))
