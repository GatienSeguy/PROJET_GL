import tkinter as tk

class Fenetre(tk.Tk):
    def __init__(self):
        super().__init__()

        couleur_fond = "#d9d9d9"
        self.configure(bg=couleur_fond)
        self.title("Paramétrage du Réseau de Neuronnes")
        self.geometry("520x1")

        self.font_titre = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 11)

        self.cadre = tk.Frame(self, bg=couleur_fond, padx=20, pady=20)
        self.cadre.pack(fill="both", expand=True)

        # Titre simulé
        tk.Label(self.cadre, text="Paramètres", font=self.font_titre, bg=couleur_fond).pack(anchor="w", pady=(0, 10))

         # Cadre des paramètres
        self.CadreParams = tk.LabelFrame(
            self.cadre, text="", font=self.font_titre,
            bg="#ffffff", fg="#333333", bd=3, relief="ridge", padx=15, pady=15
        )
        self.CadreParams.pack(fill="both", expand=True, pady=(0, 20))

        boutons = [
            ("Paramètres temporels et de découpage de données", self.Params_temporels),
            ("Choix du modèle de réseau de neurones", self.Params_choix_reseau_neurones),
            ("Paramétrage de l'architecture réseau", self.Params_archi_reseau),
            ("Choix de la fonction perte (loss)", self.Params_choix_loss_fct),
            ("Choix et paramétrage de l'optimisateur", self.Params_optimisateur),
            ("Paramètres d'entraînement", self.Params_entrainement),
            ("Paramétrage des métriques et visualisations de suivi", self.Params_visualisation_suivi),
        ]

        for texte, commande in boutons:
            tk.Button(
                self.CadreParams, text=texte, font=self.font_bouton,
                height=2, bg="#ececec", fg="#000000", relief="groove", bd=2,
                command=commande
            ).pack(fill="x", pady=6)

        # Boutons d'action
        tk.Button(
            self.cadre, text="🚀 Envoyer la configuration au serveur", font=self.font_bouton,
            height=2, bg="#b4d9b2", fg="#0f5132", relief="raised", bd=3,
            command=self.EnvoyerConfig
        ).pack(fill="x", pady=10)

        tk.Button(
            self.cadre, text="❌ Quitter", font=self.font_bouton,
            height=2, bg="#f7b2b2", fg="#842029", relief="raised", bd=3,
            command=self.destroy
        ).pack(fill="x", pady=(0, 10))

        self.update_idletasks()
        self.geometry(f"520x{self.winfo_reqheight()}")

    # Méthodes fictives à définir
    def Params_temporels(self): pass
    def Params_choix_reseau_neurones(self): pass
    def Params_archi_reseau(self): pass
    def Params_choix_loss_fct(self): pass
    def Params_optimisateur(self): pass
    def Params_entrainement(self): pass
    def Params_visualisation_suivi(self): pass
    def EnvoyerConfig(self): pass

if __name__ == "__main__":
    Fenetre().mainloop()