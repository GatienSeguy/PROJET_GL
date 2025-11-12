import pygame
import socket
import sys

SERVER_HOST = "rpi2-7.zpq.ens-paris-saclay.fr"
SERVER_PORT = 8080
CELL_SIZE = 100
GRID_SIZE = 3
WINDOW_SIZE = CELL_SIZE * GRID_SIZE

BG_COLOR = (230, 230, 230)
LINE_COLOR = (50, 50, 50)
X_COLOR = (200, 50, 50)
O_COLOR = (50, 50, 200)
FONT_COLOR = (10, 10, 10)


def send_line(sock, text):
    sock.sendall((text + "\n").encode("utf-8"))

def recv_line(sock, buffer):
    while True:
        if "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            return line, buffer
        data = sock.recv(1024).decode("utf-8")
        if not data:
            raise ConnectionError("Connexion fermée.")
        buffer += data


def check_winner(board):
    for r in board:
        if r[0] != " " and r[0] == r[1] == r[2]: return r[0]
    for c in range(3):
        if board[0][c] != " " and board[0][c] == board[1][c] == board[2][c]:
            return board[0][c]
    if board[0][0] != " " and board[0][0] == board[1][1] == board[2][2]: return board[0][0]
    if board[0][2] != " " and board[0][2] == board[1][1] == board[2][0]: return board[0][2]
    return None

def board_full(board):
    return all(cell != " " for row in board for cell in row)

def draw_board(screen, board, message):
    screen.fill(BG_COLOR)
    font = pygame.font.SysFont(None, 72)
    msg_font = pygame.font.SysFont(None, 36)
    for i in range(1, 3):
        pygame.draw.line(screen, LINE_COLOR, (0, i * CELL_SIZE), (WINDOW_SIZE, i * CELL_SIZE), 4)
        pygame.draw.line(screen, LINE_COLOR, (i * CELL_SIZE, 0), (i * CELL_SIZE, WINDOW_SIZE), 4)
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
    if message:
        text = msg_font.render(message, True, FONT_COLOR)
        screen.blit(text, text.get_rect(center=(WINDOW_SIZE // 2, WINDOW_SIZE + 25)))


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((SERVER_HOST, SERVER_PORT))
    buffer = ""

    txt, buffer = recv_line(sock, buffer)
    parts = txt.split("|")
    symbol = parts[1]
    opponent_symbol = "O" if symbol == "X" else "X"
    print(f"Connecté ! Tu joues {symbol}")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
    pygame.display.set_caption("Morpion en ligne")

    board = [[" "] * 3 for _ in range(3)]
    running = True
    message = ""
    turn = "X"

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if turn != symbol or check_winner(board) or board_full(board): continue
                x, y = event.pos
                if y > WINDOW_SIZE: continue
                c, r = x // CELL_SIZE, y // CELL_SIZE
                if board[r][c] == " ":
                    board[r][c] = symbol
                    send_line(sock, f"{r}|{c}")
                    turn = opponent_symbol


        if turn != symbol and not check_winner(board) and not board_full(board):
            sock.setblocking(False)
            try:
                line, buffer = recv_line(sock, buffer)
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
            part = "L adversaire a" if (turn == symbol) else "Tu as"
            message = f"{part} gagné !"
        elif board_full(board):
            message = "Match nul !"
        else:
            message = f"À {'toi' if (turn == symbol) else 'l adversaire'} de jouer"

        draw_board(screen, board, message)
        pygame.display.flip()

    sock.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
