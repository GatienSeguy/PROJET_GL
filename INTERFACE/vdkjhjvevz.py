import customtkinter as ctk

# Initialisation CTk
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class FenetreBoutons(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Fenêtre simple")
        self.geometry("400x300")  # largeur x hauteur initiales
        self.minsize(300, 200)
        self.focus_force()

        # Frame principale
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        # Rendre la grille de la fenêtre responsive
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Rendre les colonnes du frame responsive
        frame.grid_columnconfigure((0,1), weight=1)
        frame.grid_rowconfigure((0,1), weight=1)

        # Boutons
        ctk.CTkButton(frame, text="Bouton 1", command=lambda: print("Bouton 1"))\
            .grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkButton(frame, text="Bouton 2", command=lambda: print("Bouton 2"))\
            .grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        ctk.CTkButton(frame, text="Bouton 3", command=lambda: print("Bouton 3"))\
            .grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        ctk.CTkButton(frame, text="Bouton 4", command=lambda: print("Bouton 4"))\
            .grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

# ------------------ Application ------------------
if __name__ == "__main__":
    root = ctk.CTk()
    root.geometry("600x400")
    FenetreBoutons(root)
    root.mainloop()
