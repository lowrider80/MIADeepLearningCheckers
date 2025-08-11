import numpy as np

class CheckersEnv:
    """
    Damas inglesas 8x8.
    Piezas:
      - Agente:  1 (hombre),  2 (reina)
      - Humano: -1 (hombre), -2 (reina)
      - Vacío:   0
    Reglas:
      - Captura obligatoria.
      - Multi-captura: si tras capturar hay otra captura posible con la misma pieza,
        el mismo jugador debe continuar (no se alterna el turno).
    """
    def __init__(self, no_capture_draw=80):
        self.board_size = 8
        self.no_capture_draw = no_capture_draw
        self.reset()

    def reset(self):
        self.board = np.zeros((8, 8), dtype=int)

        # Filas 0..2: humano (-1), 5..7: agente (1)
        for row in range(3):
            for col in range(8):
                if (row + col) % 2 != 0:
                    self.board[row, col] = -1
        for row in range(5, 8):
            for col in range(8):
                if (row + col) % 2 != 0:
                    self.board[row, col] = 1

        self.current_player = -1   # humano comienza
        self.done = False
        self.winner = None
        self.must_continue_from = None   # (r,c) si debe seguir multi-captura
        self.no_capture_moves = 0
        return self.get_state()

    def get_state(self):
        # Incluir current_player para diferenciar el turno
        return f"{self.current_player}|{self.board.flatten().tolist()}"

    def get_board_state(self):
        return self.board.copy()

    def is_done(self): return self.done
    def get_winner(self): return self.winner

    # ----------------- Generación de movimientos -----------------
    def get_valid_actions(self, player=None):
        if player is None:
            player = self.current_player

        actions = []
        if self.must_continue_from is not None:
            r, c = self.must_continue_from
            if self.board[r, c] * player <= 0:  # pieza ya no pertenece al jugador
                self.must_continue_from = None
            else:
                caps = self._get_piece_captures(r, c, player)
                return caps  # solo puede seguir capturando con esa pieza

        # Si no hay multi-captura pendiente, buscar movimientos/capturas de  el tablero
        piece_positions = zip(*np.where(self.board * player > 0))
        captures = []
        moves = []
        for r, c in piece_positions:
            captures.extend(self._get_piece_captures(r, c, player))
            if not captures:  # solo computar moves si aún no hay capturas globales
                moves.extend(self._get_piece_moves(r, c, player))

        # Captura obligatoria
        return captures if captures else moves

    def _directions_for(self, piece):
        is_king = abs(piece) == 2
        sign = 1 if piece > 0 else -1
        if is_king:
            return [(-1,-1), (-1,1), (1,-1), (1,1)]
        # hombres: humano(-1) baja (+1 en fila); agente(+1) sube (-1 en fila)
        return ([(1,-1), (1,1)] if sign < 0 else [(-1,-1), (-1,1)])

    def _get_piece_moves(self, row, col, player):
        piece = self.board[row, col]
        if piece * player <= 0:  # no es tu pieza
            return []
        moves = []
        for dr, dc in self._directions_for(piece):
            nr, nc = row + dr, col + dc
            if 0 <= nr < 8 and 0 <= nc < 8 and self.board[nr, nc] == 0:
                moves.append((row, col, nr, nc))
        return moves

    def _get_piece_captures(self, row, col, player):
        piece = self.board[row, col]
        if piece * player <= 0:
            return []
        caps = []
        for dr, dc in self._directions_for(piece):
            mr, mc = row + dr, col + dc
            jr, jc = row + 2*dr, col + 2*dc
            if 0 <= jr < 8 and 0 <= jc < 8 and 0 <= mr < 8 and 0 <= mc < 8:
                if self.board[mr, mc] * player < 0 and self.board[jr, jc] == 0:
                    caps.append((row, col, jr, jc, mr, mc))
        return caps

    # ----------------- Dinámica del entorno -----------------
    def step(self, action):
        if self.done:
            return self.get_state(), 0.0, True

        reward = 0.0
        player = self.current_player

        if len(action) == 4:
            fr, fc, tr, tc = action
            capture = False
        else:
            fr, fc, tr, tc, cr, cc = action
            capture = True

        # Validaciones mínimas (asumimos que la UI/agent pasa acciones válidas)
        piece = self.board[fr, fc]
        assert piece * player > 0, "La pieza no pertenece al jugador actual"

        # Ejecutar movimiento
        if capture:
            self.board[tr, tc] = piece
            self.board[fr, fc] = 0
            self.board[cr, cc] = 0
            # Recompensa centrada en el agente (jugador +1)
            reward += (1.0 if player == 1 else -1.0)
            self.no_capture_moves = 0
        else:
            self.board[tr, tc] = piece
            self.board[fr, fc] = 0
            # Movimiento simple: sin recompensa densa (0) para no sesgar cuando mueve el humano
            self.no_capture_moves += 1

        # Coronación
        promoted = False
        if self.board[tr, tc] == 1 and tr == 0:
            self.board[tr, tc] = 2
            promoted = True
        elif self.board[tr, tc] == -1 and tr == 7:
            self.board[tr, tc] = -2
            promoted = True
        if promoted:
            reward += (0.5 if player == 1 else -0.5)

        # ¿Sigue la multi-captura?
        self.must_continue_from = None
        if capture:
            next_caps = self._get_piece_captures(tr, tc, player)
            if next_caps:
                # Mismo jugador continúa; acotar a esa pieza
                self.must_continue_from = (tr, tc)

        # ¿Fin de juego?
        human_pieces = np.sum(self.board < 0)
        agent_pieces = np.sum(self.board > 0)

        if human_pieces == 0:
            self.done, self.winner = True, 1
            reward += (5.0 if player == 1 else -5.0)
        elif agent_pieces == 0:
            self.done, self.winner = True, -1
            reward += (-5.0 if player == 1 else 5.0)
        elif self.no_capture_moves >= self.no_capture_draw:
            self.done, self.winner = True, None  # tablas
            # reward += 0.0
        else:
            # Bloqueo del oponente: si el oponente no puede mover, gana quien movió
            opp = -player
            if self.must_continue_from is None:
                if len(self.get_valid_actions(opp)) == 0:
                    self.done, self.winner = True, player
                    reward += (5.0 if player == 1 else -5.0)

        # Alternancia de turno (solo si no hay multi-captura)
        if not self.done and self.must_continue_from is None:
            self.current_player *= -1

        return self.get_state(), reward, self.done
