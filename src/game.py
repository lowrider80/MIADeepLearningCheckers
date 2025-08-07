
from src.agent import QLearningAgent
from src.env import CheckersEnv


class Game:
    def __init__(self):
        self.env = CheckersEnv()
        self.agent = QLearningAgent()
        self.state = self.env.reset()
        self.selected_piece = None
        self.game_mode = "human_vs_agent"

    def reset(self):
        self.state = self.env.reset()
        self.selected_piece = None
        return self.state

    def get_board_state(self):
        return self.env.get_board_state()

    def is_done(self):
        return self.env.is_done()

    def get_current_player(self):
        return self.env.current_player

    def get_winner(self):
        return self.env.get_winner()

    def result_text(self):
        if not self.is_done():
            return f"Turno del {'Humano' if self.env.current_player == -1 else 'Agente'}"

        winner = self.get_winner()
        if winner == -1:
            return "¡Humano Gana!"
        elif winner == 1:
            return "¡Agente Gana!"
        else:
            return "¡Empate!"

    def get_valid_actions(self):
        return self.env.get_valid_actions()

    def make_move(self, action):
        self.state, reward, done = self.env.step(action)
        return reward, done

    def handle_click(self, row, col):
        if self.is_done() or self.env.current_player != -1:
            return False

        board = self.get_board_state()

        if self.selected_piece is None:
            if board[row, col] == -1:
                self.selected_piece = (row, col)
                return True
        else:
            from_row, from_col = self.selected_piece
            valid_actions = self.get_valid_actions()
            simple_move = (from_row, from_col, row, col)
            if simple_move in valid_actions:
                self.make_move(simple_move)
                self.selected_piece = None
                return True

            for action in valid_actions:
                if (len(action) == 6 and
                    action[0] == from_row and action[1] == from_col and
                    action[2] == row and action[3] == col):
                    self.make_move(action)
                    self.selected_piece = None
                    return True

            if board[row, col] == -1:
                self.selected_piece = (row, col)
                return True

            self.selected_piece = None

        return False

    def agent_move(self):
        if self.is_done() or self.env.current_player != 1:
            return

        valid_actions = self.get_valid_actions()
        if valid_actions:
            action = self.agent.choose_action(self.state, valid_actions)
            self.make_move(action)
