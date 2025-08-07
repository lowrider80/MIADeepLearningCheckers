import pygame

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_BROWN = (240, 217, 181)
DARK_BROWN = (181, 136, 99)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

BOARD_SIZE = 8


def render_board(screen, board_state, selected_piece=None):
    """
    Dibuja el tablero 8x8 y las piezas según el estado board_state.
    board_state: matriz 8x8 con valores: 0=vacío, 1=agente, -1=humano
    selected_piece: tupla (row, col) de la pieza seleccionada
    """
    cell_size = screen.get_width() // BOARD_SIZE

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
            color = LIGHT_BROWN if (row + col) % 2 == 0 else DARK_BROWN
            if selected_piece and selected_piece == (row, col):
                color = GREEN
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 2)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            piece = board_state[row][col]
            if piece != 0:
                center_x = col * cell_size + cell_size // 2
                center_y = row * cell_size + cell_size // 2
                radius = cell_size // 2 - 10

                if piece == 1:
                    color = WHITE
                    border_color = BLACK
                else:
                    color = BLACK
                    border_color = WHITE

                pygame.draw.circle(screen, color, (center_x, center_y), radius)
                pygame.draw.circle(screen, border_color, (center_x, center_y), radius, 3)


def display_message(screen, text, font_size=36, color=RED):
    font = pygame.font.SysFont('Arial', font_size)
    surf = font.render(text, True, color)
    rect = surf.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 - 50))

    background = pygame.Surface((rect.width + 20, rect.height + 10))
    background.set_alpha(180)
    background.fill(WHITE)
    screen.blit(background, (rect.x - 10, rect.y - 5))

    screen.blit(surf, rect)


class UI:
    def __init__(self, screen, game):
        self.screen = screen
        self.game = game
        self.cell_size = screen.get_width() // BOARD_SIZE

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_x, mouse_y = event.pos
            col = mouse_x // self.cell_size
            row = mouse_y // self.cell_size
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                self.game.handle_click(row, col)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.game.reset()

    def update(self):
        self.screen.fill(WHITE)
        board = self.game.get_board_state()
        render_board(self.screen, board, self.game.selected_piece)
        self._draw_game_info()

        if not self.game.is_done() and self.game.get_current_player() == 1:
            self.game.agent_move()

    def _draw_game_info(self):
        message = self.game.result_text()
        display_message(self.screen, message, 24, BLACK)

        if not self.game.is_done():
            instruction = "Click en tu pieza, luego en destino" if self.game.get_current_player() == -1 else "Turno del agente..."
        else:
            instruction = "Presiona 'R' para reiniciar"

        font = pygame.font.SysFont('Arial', 18)
        surf = font.render(instruction, True, BLUE)
        rect = surf.get_rect(center=(self.screen.get_width() // 2, self.screen.get_height() - 30))
        self.screen.blit(surf, rect)
