import numpy as np
import random

class CheckersEnv:
    """
    Entorno de damas inglesas en tablero 8x8 con 12 piezas por jugador.
    - Jugador 1 (humano): -1
    - Jugador -1 (agente): 1
    - Casillas vacías: 0
    """
    def __init__(self):
        self.board_size = 8
        self.reset()

    def reset(self):
        """Reinicia el tablero al estado inicial"""
        self.board = np.zeros((8, 8), dtype=int)

        for row in range(3):
            for col in range(8):
                if (row + col) % 2 != 0:
                    self.board[row, col] = -1  # Humano

        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 != 0:
                    self.board[row, col] = 1  # Agente

        self.current_player = -1
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return str(self.board.flatten().tolist())

    def get_board_state(self):
        return self.board.copy()

    def is_done(self):
        return self.done

    def get_winner(self):
        return self.winner

    def get_valid_actions(self, player=None):
        if player is None:
            player = self.current_player

        actions = []
        pieces = np.where(self.board == player)

        for i in range(len(pieces[0])):
            row, col = pieces[0][i], pieces[1][i]
            moves = self._get_piece_moves(row, col, player)
            actions.extend(moves)

        return actions

    def _get_piece_moves(self, row, col, player):
        moves = []
        directions = [(1, -1), (1, 1)] if player == 1 else [(-1, -1), (-1, 1)]
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Opcional: permitir reinas

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                if self.board[new_row, new_col] == 0:
                    moves.append((row, col, new_row, new_col))
                elif self.board[new_row, new_col] == -player:
                    jump_row, jump_col = new_row + dr, new_col + dc
                    if 0 <= jump_row < 8 and 0 <= jump_col < 8 and self.board[jump_row, jump_col] == 0:
                        moves.append((row, col, jump_row, jump_col, new_row, new_col))

        return moves

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True

        reward = 0

        if len(action) == 4:
            from_row, from_col, to_row, to_col = action
            self.board[to_row, to_col] = self.board[from_row, from_col]
            self.board[from_row, from_col] = 0
            reward = 0.1

        elif len(action) == 6:
            from_row, from_col, to_row, to_col, cap_row, cap_col = action
            self.board[to_row, to_col] = self.board[from_row, from_col]
            self.board[from_row, from_col] = 0
            self.board[cap_row, cap_col] = 0
            reward = 10 if self.current_player == 1 else -10

        # Verificar coronación (reina opcional)
        if self.board[to_row, to_col] == 1 and to_row == 0:
            pass  # self.board[to_row, to_col] = 2  # Reina del agente (opcional)
        elif self.board[to_row, to_col] == -1 and to_row == 7:
            pass  # self.board[to_row, to_col] = -2  # Reina del humano (opcional)

        # Verificar fin del juego
        human_pieces = np.sum(self.board == -1)
        agent_pieces = np.sum(self.board == 1)

        if human_pieces == 0:
            self.done = True
            self.winner = 1
            reward = 100 if self.current_player == 1 else -100
        elif agent_pieces == 0:
            self.done = True
            self.winner = -1
            reward = -100 if self.current_player == 1 else 100
        elif len(self.get_valid_actions(-self.current_player)) == 0:
            self.done = True
            self.winner = self.current_player
            reward = 100 if self.current_player == 1 else -100

        self.current_player *= -1
        return self.get_state(), reward, self.done
