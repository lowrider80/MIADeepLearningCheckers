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



class MonteCarloAgent:
    """
        Monte Carlo Control on-policy ε-soft (retornos acumulando TODAS las recompensas).
        - Guarda la secuencia del episodio: R1, R2, ..., RT
        - Para cada (s,a) del agente en el tiempo t, calcula:
              G_t = R_{t+1} + γ R_{t+2} + ... + γ^{T-t-1} R_T
        - Soporta primera-visita (por defecto) o every-visit.
        """

    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05,
                 first_visit=True):
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Acumuladores Monte Carlo
        self._returns_sum = defaultdict(float)  # key=(state_str, action_str) → suma de retornos
        self._returns_count = defaultdict(int)  # key=(state_str, action_str) → cantidad

        # Episodio en curso
        self._rewards = []  # [R1, R2, ..., RT]
        self._episode_sa_t = []  # [(state_str, action_str, t_index), ...]
        # t_index = índice en _rewards que corresponde a R_{t+1}

        # Parámetros de control
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.first_visit = first_visit  # True: primera-visita; False: every-visit

    # ---------- Política ε-greedy sobre Q ----------
    def choose_action(self, state, valid_actions):
        if not valid_actions:
            return None
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        q_values = [self.q_table[state][str(a)] for a in valid_actions]
        max_q = max(q_values)
        best = [a for a, q in zip(valid_actions, q_values) if q == max_q]
        return random.choice(best)

    # ---------- Registro del episodio ----------
    def remember_action(self, state, action):
        """
        Llamar justo ANTES de ejecutar la acción del agente.
        t_index = len(_rewards) → posición donde caerá R_{t+1}.
        """
        self._episode_sa_t.append((state, str(action), len(self._rewards)))

    def record_reward(self, reward):
        """
        Llamar DESPUÉS de cada step del entorno (mueva quien mueva).
        Esto agrega R_{t+1}, R_{t+2}, ... en orden.
        """
        self._rewards.append(float(reward))

    # ---------- Actualización Monte Carlo al final del episodio ----------
    def end_episode(self):
        T = len(self._rewards)
        if T == 0 or not self._episode_sa_t:
            # Nada que actualizar
            self._clear_and_decay()
            return

        if self.first_visit:
            seen = set()
            for (s, a, t) in self._episode_sa_t:
                if (s, a) in seen:
                    continue
                seen.add((s, a))
                G = self._discounted_return_from(t, T)
                key = (s, a)
                self._returns_sum[key] += G
                self._returns_count[key] += 1
                self.q_table[s][a] = self._returns_sum[key] / self._returns_count[key]
        else:
            # every-visit: actualiza incluso si (s,a) aparece varias veces
            for (s, a, t) in self._episode_sa_t:
                G = self._discounted_return_from(t, T)
                key = (s, a)
                self._returns_sum[key] += G
                self._returns_count[key] += 1
                self.q_table[s][a] = self._returns_sum[key] / self._returns_count[key]

        self._clear_and_decay()

    # ---------- Utilidades ----------
    def _discounted_return_from(self, t, T):
        """G_t = sum_{k=t}^{T-1} γ^{k-t} R_{k+1}   (aquí R_{k+1} ≡ _rewards[k])"""
        G, factor = 0.0, 1.0
        # _rewards[t] es R_{t+1}
        for k in range(t, T):
            G += factor * self._rewards[k]
            factor *= self.gamma
        return G

    def _clear_and_decay(self):
        self._rewards.clear()
        self._episode_sa_t.clear()
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)