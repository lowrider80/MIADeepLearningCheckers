from src.agent import QLearningAgent, MonteCarloAgent  # Asegúrate de tener MonteCarloAgent en agent.py
from src.env import CheckersEnv


class Game:
    def __init__(self, agent_cls=MonteCarloAgent):  # Monte Carlo por defecto
        self.env = CheckersEnv()
        self.agent = agent_cls()
        self.state = self.env.reset()
        self.selected_piece = None
        self.game_mode = "human_vs_agent"

    # -------- métodos utilitarios que requiere UI --------
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

    # -------- juego --------
    def make_move(self, action):
        self.state, reward, done = self.env.step(action)
        return reward, done

    def handle_click(self, row, col):
        # turno del humano
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

            # Movimiento simple
            if simple_move in valid_actions:
                reward, done = self.make_move(simple_move)
                # MC: registrar recompensa también cuando mueve el humano (si el agente la implementa)
                if hasattr(self.agent, "record_reward"):
                    self.agent.record_reward(reward)
                self.selected_piece = None
                if done and hasattr(self.agent, "end_episode"):
                    self.agent.end_episode()
                return True

            # Captura (si tu entorno usa tuplas de 6)
            for action in valid_actions:
                if (len(action) == 6 and
                    action[0] == from_row and action[1] == from_col and
                    action[2] == row and action[3] == col):
                    reward, done = self.make_move(action)
                    if hasattr(self.agent, "record_reward"):
                        self.agent.record_reward(reward)
                    self.selected_piece = None
                    if done and hasattr(self.agent, "end_episode"):
                        self.agent.end_episode()
                    return True

            if board[row, col] == -1:
                self.selected_piece = (row, col)
                return True

            self.selected_piece = None

        return False

    def agent_move(self):
        # turno del agente
        if self.is_done() or self.env.current_player != 1:
            return

        valid_actions = self.get_valid_actions()
        if not valid_actions:
            return

        # Elegir acción
        action = self.agent.choose_action(self.state, valid_actions)

        # MC: recordar (s,a) ANTES del step si el agente lo soporta
        if hasattr(self.agent, "remember_action"):
            self.agent.remember_action(self.state, action)

        # Ejecutar y registrar recompensa
        reward, done = self.make_move(action)
        if hasattr(self.agent, "record_reward"):
            self.agent.record_reward(reward)

        if done and hasattr(self.agent, "end_episode"):
            self.agent.end_episode()
