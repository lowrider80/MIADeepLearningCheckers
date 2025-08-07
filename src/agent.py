import random
from collections import defaultdict

class QLearningAgent:
    """
    Agente de Q-learning para jugar damas
    """
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha          # Tasa de aprendizaje
        self.gamma = gamma          # Factor de descuento
        self.epsilon = epsilon      # Probabilidad de exploración
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state, valid_actions):
        """Elige una acción usando política epsilon-greedy"""
        if not valid_actions:
            return None
            
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        q_values = [self.q_table[state][str(a)] for a in valid_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, next_valid_actions, done):
        """Actualiza la Q-table usando la ecuación de Bellman"""
        if done:
            max_next_q = 0
        else:
            max_next_q = max([self.q_table[next_state][str(a)] for a in next_valid_actions] or [0])
        
        current_q = self.q_table[state][str(action)]
        self.q_table[state][str(action)] = (
            current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        )
        
        # Reducir epsilon después de cada episodio
        if done:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)