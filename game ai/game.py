import random

class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def succ_main_phase(self, state):
        adjacent_tiles = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]

        locations = []
        for a in range(len(self.board)):
            for b in range(len(self.board)):
                if state[a][b] == self.my_piece:
                    locations.append((a, b))

        successors = []
        for location in locations:
            for tile in adjacent_tiles:
                x = location[0] + tile[0]
                y = location[1] + tile[1]
                if 0 <= x <= 4 and 0 <= y <= 4 and state[x][y] == ' ':
                    board_copy = [row[:] for row in state]
                    board_copy[location[0]][location[1]] = ' '
                    board_copy[x][y] = self.my_piece
                    if board_copy not in successors:
                        successors.append(board_copy)

        return successors

    def succ_drop_phase(self, state):
        successors = []
        for a in range(len(self.board)):
            for b in range(len(self.board)):
                if state[a][b] == ' ':
                    state_copy = [row[:] for row in state]
                    state_copy[a][b] = self.my_piece
                    successors.append(state_copy)

        return successors

    def succ(self, state):
        if self.drop_phase(state):
            return self.succ_drop_phase(state)
        else:
            return self.succ_main_phase(state)

    def heuristic_game_value(self, state):
        score = 0

        # check horizontal positions
        for row in state:
            for i in range(2):
                count = 0
                for j in range (4):
                    if row[i+j] == self.my_piece:
                        count += 1
                if count != 0:
                    score = max(score, count)

        # check vertical positions
        for col in range(5):
            for i in range(2):
                count = 0
                for j in range(4):
                    if state[i + j][col] == self.my_piece:
                        count += 1
                if count != 0:
                    score = max(score, count)

        # # check \ diagonal positions
        # diag_left_starting_squares = [[3, 3], [3, 4], [4, 3], [4, 4]]
        # for a, b in diag_left_starting_squares:
        #     count = 0
        #     for j in range(4):
        #         if state[a - j][b - j] == self.my_piece:
        #             count += 1
        #     if count != 0:
        #         score = max(score, count)
        #
        # # check / diagonal positions
        # diag_right_starting_squares = [[3, 0], [3, 1], [4, 0], [4, 1]]
        # for a, b in diag_right_starting_squares:
        #     count = 0
        #     for j in range(4):
        #         if state[a - j][b + j] == self.my_piece:
        #             count += 1
        #     if count != 0:
        #         score = max(score, count)
        #
        # # check 3x3 square corners positions
        # for a in range(3):
        #     for b in range(3):
        #         count = 0
        #         if state[a][b] == self.my_piece:
        #             count += 1
        #         if state[a][b + 2] == self.my_piece:
        #             count += 1
        #         if state[a + 2][b] == self.my_piece:
        #             count += 1
        #         if state[a + 2][b + 2] == self.my_piece:
        #             count += 1
        #         if count != 0:
        #             score = max(score, count)

        return (score * 2) / 4 - 1

    def max_value(self, state, depth):
        if self.game_value(state)**2 == 1:
            return self.game_value(state)
        if depth == 2:
            return self.heuristic_game_value(state)
        elif depth % 2 == 1:
            return max(self.max_value(successor, depth+1) for successor in self.succ_main_phase(state))
        else:
            return min(self.max_value(successor, depth+1) for successor in self.succ_main_phase(state))

    def drop_phase(self, state):
        num_pieces = 0

        for a in range(len(self.board)):
            for b in range(len(self.board)):
                if state[a][b] != ' ':
                    num_pieces += 1

        return num_pieces != 8

    def first_move(self, state):
        num_pieces = 0

        for a in range(len(self.board)):
            for b in range(len(self.board)):
                if state[a][b] != ' ':
                    num_pieces += 1

        return num_pieces == 0 or num_pieces == 1

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """

        if self.first_move(state):
            if state[2][2] == ' ':
                return [(2, 2)]
            else:
                return [(1, 1)]

        drop_phase = self.drop_phase(state)
        move = []

        next_move = None
        max_val = -1
        for successor in self.succ(state):
            val = self.max_value(successor, 0)
            if val >= max_val:
                next_move = successor
                max_val = val

        delete = None
        place = None
        for a in range(5):
            for b in range(5):
                if state[a][b] != next_move[a][b]:
                    if state[a][b] == self.my_piece:
                        delete = (a, b)
                    else:
                        place = (a, b)

        # ensure the destination (row,col) tuple is at the beginning of the move list
        if not drop_phase:
            move.append(delete)
        move.insert(0, place)
        return move

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # check \ diagonal wins
        diag_left_starting_squares = [[3, 3], [3, 4], [4, 3], [4, 4]]
        for a, b in diag_left_starting_squares:
            if state[a][b] != ' ' and  state[a][b] == state[a-1][b-1] == state[a-2][b-2] == state[a-3][b-3]:
                return 1 if state[a][b] == self.my_piece else -1

        # check / diagonal wins
        diag_right_starting_squares = [[3, 0], [3, 1], [4, 0], [4, 1]]
        for a, b in diag_right_starting_squares:
            if state[a][b] != ' ' and state[a][b] == state[a-1][b+1] == state[a-2][b+2] == state[a-3][b+3]:
                return 1 if state[a][b] == self.my_piece else -1

        # check 3x3 square corners wins
        for a in range(3):
            for b in range(3):
                if state[a][b] != ' ' and state[a][b] == state[a][b+2] == state[a+2][b] == state[a+2][b+2]:
                    return 1 if state[a][b] == self.my_piece else -1

        return 0 # no winner yet

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")


if __name__ == "__main__":
    main()
