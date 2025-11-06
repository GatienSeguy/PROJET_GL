import socket
import threading

HOST = ""
PORT = 8080
BACKLOG = 50

# ----- Helpers pour ligne par ligne avec socket brut -----
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

# ----- Classe Player -----
class Player:
    def __init__(self, sock, addr, pid):
        self.sock = sock
        self.addr = addr
        self.id = pid
        self.symbol = None  # "X" ou "O"
        self.buffer = ""    # tampon pour recv_line

    def close(self):
        try: self.sock.close()
        except Exception: pass

# ----- Liste d'attente -----
waiting = []
wait_lock = threading.Lock()

# ----- Relais P1 -> P2 -----
def forward(src: Player, dst: Player):
    try:
        while True:
            line, src.buffer = recv_line(src.sock, src.buffer)
            send_line(dst.sock, line)
    except Exception:
        pass

# ----- Thread par partie -----
def game_session(p1: Player, p2: Player):
    try:
        p1.id, p1.symbol = 1, "X"
        p2.id, p2.symbol = 2, "O"

        send_line(p1.sock, f"{p1.id}|{p1.symbol}")
        send_line(p2.sock, f"{p2.id}|{p2.symbol}")

        t12 = threading.Thread(target=forward, args=(p1, p2), daemon=True)
        t21 = threading.Thread(target=forward, args=(p2, p1), daemon=True)
        t12.start()
        t21.start()
        t12.join()
        t21.join()
    finally:
        p1.close()
        p2.close()

# ----- Gestion d’une nouvelle connexion -----
def handle_new_client(conn, addr):
    print(f"[+] Nouveau client {addr}")
    player = Player(conn, addr, pid=0)

    with wait_lock:
        waiting.append(player)
        if len(waiting) >= 2:
            p1 = waiting.pop(0)
            p2 = waiting.pop(0)
            print(f"[=] Match lancé: {p1.addr} vs {p2.addr}")
            threading.Thread(target=game_session, args=(p1, p2), daemon=True).start()
        else:
            print("[ ] En attente d’un autre joueur...")

# ----- Boucle principale -----
def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(BACKLOG)
        print(f"[Serveur] En écoute sur {HOST or '0.0.0.0'}:{PORT} ...")
        while True:
            conn, addr = s.accept()
            threading.Thread(target=handle_new_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    main()