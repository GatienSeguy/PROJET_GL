import tkinter as tk
from tkinter import ttk


# Paramètres et variables






# Créer la fenêtre principale
class Fenetre(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.title("Paramétrage du Réseau de Neuronnes")
        self.geometry("500x700")  # largeur x hauteur
        #self.cadre = tk.Frame(self, borderwidth=50)
        self.cadre = tk.Frame(self, borderwidth=30)
        self.cadre.pack(fill="both", expand="yes")
        self.CadreParams = tk.LabelFrame(self.cadre, text="Paramètres", borderwidth=3)
        self.CadreParams.pack(fill="both", expand="yes")
        tk.Button(self.CadreParams, text='Paramètres temporels et de découpage de données', height=3,command=self.Params_temporels).pack(fill="both",pady=10,padx=20)
        tk.Button(self.CadreParams, text='Choix du modèle de réseau de neurones', height=3,command=self.Params_choix_reseau_neurones).pack(fill="both",pady=10,padx=20)
        tk.Button(self.CadreParams, text="Paramétrage de l'architechture réseau", height=3,command=self.Params_archi_reseau).pack(fill="both",pady=10,padx=20)
        tk.Button(self.CadreParams, text="Choix de la fonction perte (loss)", height=3,command=self.Params_choix_loss_fct).pack(fill="both",pady=10,padx=20)
        tk.Button(self.CadreParams, text="Choix et paramétrage de l'optimisateur", height=3,command=self.Params_optimisateur).pack(fill="both",pady=10,padx=20)
        tk.Button(self.CadreParams, text="Paramètres d'entrainement", height=3,command=self.Params_entrainement).pack(fill="both",pady=10,padx=20)
        tk.Button(self.CadreParams, text="Paramétrage des métriques et visualisations de suivi", height=3,command=self.Params_visualisation_suivi).pack(fill="both",pady=10,padx=20)
        


    def Params_temporels(self):
        self.fenetre_params_temporels = tk.Tk()
        self.fenetre_params_temporels.title("Paramètres temporels et de découpage de données")
        self.fenetre_params_temporels.geometry("300x200")  # largeur x hauteur

        # Ajouter un label
        label = tk.Label(self.fenetre_params_temporels, text="Bonjour, Maxime !")
        label.pack(pady=20)

        # Ajouter un bouton pour fermer
        bouton_quitter = tk.Button(self.fenetre_params_temporels, text="Quitter", command=self.fenetre_params_temporels.destroy)
        bouton_quitter.pack()

        # Lancer la boucle principale
        self.fenetre_params_temporels.mainloop()
        pass
    def Params_choix_reseau_neurones(self):
        pass
    def Params_archi_reseau(self):
        pass
    def Params_choix_loss_fct(self):
        pass
    def Params_optimisateur(self):
        pass
    def Params_entrainement(self):
        pass
    def Params_visualisation_suivi(self):
        pass

class Fenetre_Params(tk.Tk):
    pass



# # Ajouter un label
# label = tk.Label(fenetre, text="Bonjour, Maxime !")
# label.pack()

# # Ajouter un bouton
# def clic():
#     label.config(text="Tu as cliqué!")

# bouton = tk.Button(fenetre, text="Clique-moi", command=clic)
# bouton.pack()

# Lancer la boucle principale
fenetree = Fenetre()
fenetree.mainloop()
