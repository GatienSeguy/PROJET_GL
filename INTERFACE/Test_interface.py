import tkinter as tk

# Classe 2 : un cadre personnalisé
class CadreSecondaire(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(bg="#eaf2f8", padx=20, pady=20, highlightbackground="black", highlightthickness=2)

        # Exemple de contenu dans le cadre
        label = tk.Label(self, text="Je suis dans le cadre secondaire", bg="#eaf2f8")
        label.pack()

# Classe 1 : la fenêtre principale
class FenetrePrincipale(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fenêtre principale")
        self.geometry("600x400")

        # Création du cadre secondaire (classe 2)
        self.cadre_secondaire = CadreSecondaire(self)
        self.cadre_secondaire.pack(side="left", fill="y", padx=10, pady=10)

        # Un autre cadre pour les résultats
        self.cadre_results = tk.Frame(self, bg="#d5f5e3", highlightbackground="black", highlightthickness=2)
        self.cadre_results.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        label_results = tk.Label(self.cadre_results, text="Zone des résultats", bg="#d5f5e3")
        label_results.pack()

# Lancement de l'application
if __name__ == "__main__":
    app = FenetrePrincipale()
    app.mainloop()