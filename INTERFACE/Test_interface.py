import tkinter as tk

class Fenetre(tk.Tk):
    def __init__(self):
        super().__init__()  # initialise tk.Tk
        self.title("Ma fenêtre perso")
        self.geometry("400x300")

        label = tk.Label(self, text="Hello Maxime 👋")
        label.pack(pady=20)

# Création et lancement
app = Fenetre()
app.mainloop()
