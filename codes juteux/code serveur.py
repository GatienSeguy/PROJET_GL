import socket
import threading

HOST = ""
PORT = 8081
BACKLOG = 50

def recv_line(rfile):
    line = rfile.readline()
    if line == "":
        raise ConnectionError("Connexion fermée.")
    return line.rstrip("\n")

def send_line(wfile, text: str):
    print(text, file=wfile, flush=True)

class Player:
    def __init__(self, sock, addr, pid):
        self.sock = sock
        self.addr = addr
        self.rfile = sock.makefile("r", encoding="utf-8", newline="\n")
        self.wfile = sock.makefile("w", encoding="utf-8", newline="\n")
        self.id = pid
        self.symbol = None  # "X" ou "O"

    def close(self):
        for f in (self.rfile, self.wfile):
            try:
                f.close()
            except Exception:
                pass
        try:
            self.sock.close()
        except Exception:
            pass

waiting = []
wait_lock = threading.Lock()

def forward(src: Player, dst: Player):
    try:
        while True:
            line = recv_line(src.rfile)
            send_line(dst.wfile, line)
    except Exception:
        pass

def game_session(p1: Player, p2: Player):
    try:
        p1.id, p1.symbol = 1, "X"
        p2.id, p2.symbol = 2, "O"

        # Envoi infos d'identité
        send_line(p1.wfile, f"{p1.id}|{p1.symbol}")
        send_line(p2.wfile, f"{p2.id}|{p2.symbol}")

        # Lancement des relais bidirectionnels
        t12 = threading.Thread(target=forward, args=(p1, p2), daemon=True)
        t21 = threading.Thread(target=forward, args=(p2, p1), daemon=True)
        t12.start()
        t21.start()
        t12.join()
        t21.join()
    finally:
        p1.close()
        p2.close()

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
