import numpy as np

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.done = False
        self.winner = None
        return self.board.flatten()

    def available_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3: return np.sign(sum(self.board[i, :]))
            if abs(sum(self.board[:, i])) == 3: return np.sign(sum(self.board[:, i]))
        if abs(sum([self.board[i, i] for i in range(3)])) == 3: return np.sign(sum([self.board[i, i] for i in range(3)]))
        if abs(sum([self.board[i, 2 - i] for i in range(3)])) == 3: return np.sign(sum([self.board[i, 2 - i] for i in range(3)]))
        return None

    def step(self, action, player):
        if self.done:
            print("----->Game is Over")
            raise ValueError("Game is over")
        self.board.flat[action] = player
        self.winner = self.check_winner()
        self.done = self.winner is not None or len(self.available_actions()) == 0
        reward = 1 if self.winner == 1 else (-1 if self.winner == -1 else 0)
        return self.board.flatten(), reward, self.done
