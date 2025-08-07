import pygame
from pygame.locals import *
import sys
import os

from src.game import Game
from src.ui import UI

# Añadir el directorio padre al path si es necesario
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    pygame.init()

    width, height = 640, 700
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Damas 8x8 - Humano vs Agente IA")

    game = Game()
    ui = UI(screen, game)

    clock = pygame.time.Clock()

    print("=== Damas 8x8 - Humano vs Agente ===")
    print("Controles:")
    print("- Click en pieza negra, luego en destino")
    print("- R: reiniciar, ESC: salir")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False

            ui.handle_event(event)

        ui.update()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("¡Gracias por jugar!")


if __name__ == "__main__":
    main()
