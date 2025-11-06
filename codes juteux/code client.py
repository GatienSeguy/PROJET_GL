import pygame
import socket
import sys

SERVER_HOST = "127.0.0.1" 
SERVER_PORT = 8081
CELL_SIZE = 100
GRID_SIZE = 3
WINDOW_SIZE = CELL_SIZE * GRID_SIZE

BG_COLOR = (230, 230, 230)
LINE_COLOR = (50, 50, 50)
X_COLOR = (200, 50, 50)
O_COLOR = (50, 50, 200)
FONT_COLOR = (10, 10, 10)

def recv_line(rfile):
    line = rfile.readline()
    if line == "":
        raise RuntimeError("Connexion fermée par le serveur.")
    return line.rstrip("\n")

def send_line(wfile, text: str):
    print(text, file=wfile, flush=True)

def check_winner(board):
    # Lignes
    for r in board:
        if r[0] != " " and r[0] == r[1] == r[2]:
            return r[0]
    # Colonnes
    for c in range(3):
        if board[0][c] != " " and board[0][c] == board[1][c] == board[2][c]:
            return board[0][c]
    # Diagonales
    if board[0][0] != " " and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][2] != " " and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    return None

def board_full(board):
    return all(cell != " " for row in board for cell in row)

def draw_board(screen, board, message):
    screen.fill(BG_COLOR)
    font = pygame.font.SysFont(None, 72)
    msg_font = pygame.font.SysFont(None, 36)

    # Lignes
    for i in range(1, 3):
        pygame.draw.line(screen, LINE_COLOR, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE), 4)
        pygame.draw.line(screen, LINE_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE), 4)

    # Symboles
    for r in range(3):
        for c in range(3):
            x = c * CELL_SIZE + CELL_SIZE // 2
            y = r * CELL_SIZE + CELL_SIZE // 2
            symbol = board[r][c]
            if symbol == "X":
                text = font.render("X", True, X_COLOR)
                screen.blit(text, text.get_rect(center=(x, y)))
            elif symbol == "O":
                text = font.render("O", True, O_COLOR)
                screen.blit(text, text.get_rect(center=(x, y)))

    # Message
    if message:
        text = msg_font.render(message, True, FONT_COLOR)
        screen.blit(text, text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE + 25)))

def main():
    # Connexion serveur
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, SERVER_PORT))
    rfile = sock.makefile("r", encoding="utf-8", newline="\n")
    wfile = sock.makefile("w", encoding="utf-8", newline="\n")

    # Réception ID|SYMBOLE
    txt = recv_line(rfile)
    parts = txt.split("|")
    player_id = int(parts[0])
    symbol = parts[1]
    opponent_symbol = "O" if symbol == "X" else "X"

    print(f"Connecté ! Tu joues {symbol}")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
    pygame.display.set_caption("Morpion en ligne")

    board = [[" "] * 3 for _ in range(3)]
    running = True
    message = ""
    turn = "X"  # Le joueur X commence

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if turn != symbol:
                    continue
                if check_winner(board) or board_full(board):
                    continue
                x, y = event.pos
                if y > WINDOW_SIZE:
                    continue
                c, r = x // CELL_SIZE, y // CELL_SIZE
                if board[r][c] == " ":
                    board[r][c] = symbol
                    send_line(wfile, f"{r}|{c}")
                    turn = opponent_symbol

        # Réception d'un coup adverse
        if turn != symbol and not check_winner(board) and not board_full(board):
            sock.setblocking(False)
            try:
                line = recv_line(rfile)
                r, c = map(int, line.split("|"))
                board[r][c] = opponent_symbol
                turn = symbol
            except BlockingIOError:
                pass
            except Exception:
                pass
            sock.setblocking(True)

        winner = check_winner(board)
        if winner:
            message = f"{winner} gagne !"
        elif board_full(board):
            message = "Match nul !"
        else:
            message = f"À {"toi" if (turn == symbol) else "l'adversaire"} de jouer"

        draw_board(screen, board, message)
        pygame.display.flip()

    rfile.close()
    wfile.close()
    sock.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
