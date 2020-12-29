from collections import deque
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


class Piece:
    def __init__(self, name):
        if name == 'O':
            number = 1
            occupied = [
                [(0, 0), (1, 0), (0, 1), (1, 1)]
            ]
        elif name == 'I':
            number = 2
            occupied = [
                [(0, 0), (0, 1), (0, 2), (0, 3)],
                [(0, 0), (1, 0), (2, 0), (3, 0)]
            ]
        elif name == 'T':
            number = 3
            occupied = [
                [(0, 1), (1, 0), (1, 2), (1, 1)],
                [(0, 0), (1, 0), (2, 0), (1, 1)],
                [(0, 0), (0, 1), (0, 2), (1, 1)],
                [(0, 1), (1, 1), (2, 1), (1, 0)]
            ]
        elif name == 'S':
            number = 4
            occupied = [
                [(1, 0), (0, 1), (1, 1), (0, 2)],
                [(0, 0), (1, 0), (1, 1), (2, 1)]
            ]
        elif name == 'Z':
            number = 5
            occupied = [
                [(0, 0), (0, 1), (1, 1), (1, 2)],
                [(0, 1), (1, 1), (1, 0), (2, 0)]
            ]
        elif name == 'L':
            number = 6
            occupied = [
                [(1, 2), (1, 1), (0, 2), (1, 0)],
                [(0, 0), (1, 0), (2, 0), (2, 1)],
                [(0, 0), (0, 1), (0, 2), (1, 0)],
                [(0, 0), (0, 1), (1, 1), (2, 1)]
            ]
        elif name == 'J':
            number = 7
            occupied = [
                [(0, 0), (1, 0), (1, 2), (1, 1)],
                [(0, 0), (0, 1), (1, 0), (2, 0)],
                [(0, 0), (0, 1), (0, 2), (1, 2)],
                [(1, 1), (0, 1), (2, 1), (2, 0)]
            ]
        elif name is None:
            self.name = None
            self.occupied = None
            self.number = None
            return

        self.name = name
        self.occupied = np.array(occupied)
        self.number = number


def show_piece(piece):
    board = np.zeros((4, 4))
    for pos in piece:
        board[pos] = 1
    plt.imshow(board)


class TetrisPCState:
    def __init__(self, board, cur_piece, hold_piece, next_queue, hold_available, path):
        self.board = board
        self.cur_piece = cur_piece
        self.hold_piece = hold_piece
        self.next_queue = next_queue
        self.hold_available = hold_available
        self.path = path

        # self.solved = board.shape == (0, 0)
        # self.solved = board.shape[0] == 0
        self.solved = not np.any(board)

    def generate_successors(self):
        """Generates all possible successors states given this board state. UNFINISHED!

        Returns
        -------
        List[TetrisPCState]
            List of all possible successor states
        """

        # Technically hold piece may exist but be unsable (should fix this to help optimize)
        remaining_squares = np.sum(self.board == 0) - (10 * np.sum(~self.board.any(axis=1)))
        squares_left = 4 * (1 + len(self.next_queue) +
                            (1 if self.hold_piece is not None else 0))

        if remaining_squares > squares_left:
            return []

        new_states = []

        if self.cur_piece is None:
            return []

        # Trying to put the current piece into the board in any orientation
        for occupied_spots in self.cur_piece.occupied:
            # TODO this is a questionable way of detecting where a piece can be placed
            # Should write a function to generate all possible places a piece can go
            for r in range(4):
                for c in range(10):
                    offset_occupied_spots = occupied_spots + [r, c]
                    # TODO should optimize out obviously impossible placments (i.e. all unfilled contiguous chunks must have multiple of four squares)
                    if all(self.valid_pos(*pos) for pos in offset_occupied_spots):
                        new_states.append(
                            self.place_cur_piece(offset_occupied_spots))

        # TODO Try to use hold to swap cur and hold piece and try placing that everywhere
        if self.hold_available:
            pass

        return new_states

    def valid_pos(self, r, c):
        """Determines whether the position (r, c) is valid i.e. empty and in-bounds

        Parameters
        ----------
        r : int
            Row value
        c : int
            Column value

        Returns
        -------
        bool
            Returns whether or not (r, c) is valid
        """
        return 0 <= r < self.board.shape[0] and 0 <= c < self.board.shape[1] and self.board[r, c] == 0

    def place_cur_piece(self, occupied_spots):
        """Simulates placing the current piece into the board, returning a new state for it

        Parameters
        ----------
        occupied_spots : List[Tuple[Int]]
            [List of four (r, c) positions to place cur_piece into]

        Returns
        -------
        TetrisPCState
            New state with cur_piece placed at occupied_spots in it
        """
        # Inserting cur_piece into the board
        new_board = self.board.copy()
        for pos in occupied_spots:
            new_board[tuple(pos)] = self.cur_piece.number

        # Clearing rows by removing filled rows
        new_board = new_board[~new_board.all(axis=1)]

        # Generating new next queue and new cur_piece
        new_next_queue = deepcopy(self.next_queue)
        new_cur_piece = None if not new_next_queue else new_next_queue.popleft()

        # Adding piece placement to path
        new_path = self.path + [(self.cur_piece.number, occupied_spots)]
        return TetrisPCState(
            new_board, new_cur_piece, self.hold_piece, new_next_queue, True, new_path)


def dfs(starting_state):
    fringe = [starting_state]

    while fringe:
        next_state = fringe.pop()

        if next_state.solved:
            return next_state

        fringe += next_state.generate_successors()

    raise ValueError('Un-PC-able board!')


def plot_solution(starting_board, solution_path):
    board = starting_board.copy()
    original_board_row_map = np.arange(board.shape[0])
    path = deque(solution_path)

    while path:
        val, occupied = path.popleft()

        for pos in occupied:
            board[original_board_row_map[pos[0]], pos[1]] = val

        original_board_row_map = original_board_row_map[~board[original_board_row_map].all(
            axis=1)]

    plt.imshow(board)


def plot_fringe(fringe):
    if len(fringe) == 0:
        return
    elif len(fringe) == 1:
        plt.imshow(fringe[0].board)
    else:
        _, axs = plt.subplots(len(fringe), figsize=(10, 2 * len(fringe)))
        for ax, next_state in zip(axs, fringe):
            ax.imshow(next_state.board)

        plt.show()


if __name__ == '__main__':
    # test_board = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 9]], dtype=np.uint8) * 9
    test_board = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                           [1, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                           [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                           [1, 1, 1, 1, 0, 0, 0, 1, 1, 1]], dtype=np.uint8) * 9

    # next_queue = deque(Piece(l) for l in ['L', 'T', 'I'])
    # next_queue = deque(Piece(l) for l in ['T', 'S', 'Z'])
    next_queue = deque(Piece(l) for l in ['J', 'Z', 'S'])
    cur_piece = next_queue.popleft()
    hold_piece = None
    hold_available = True
    path = []

    state = TetrisPCState(test_board, cur_piece, hold_piece,
                          next_queue, hold_available, path)
    next_states = state.generate_successors()

    res = dfs(state)
    plot_solution(state.board, res.path)
