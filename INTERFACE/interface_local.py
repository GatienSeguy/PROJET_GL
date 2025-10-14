import tkinter as tk
import requests, json
 

from tkinter import ttk


# Paramètres et variables

class Parametres_temporels():
    def __init__(self):
        self.horizon=None # int
        self.dates=None # variable datetime
        self.pas_temporel=None # int
        self.portion_decoupage=None# float entre 0 et 1
class Parametres_choix_reseau_neurones():
    def __init__(self):
        self.modele=None # str ['RNN','LSTM','GRU','CNN']
class Parametres_archi_reseau():
    def __init__(self):
        self.nb_couches=None # int
        self.hidden_size=None # int
        self.dropout_rate=None # float entre 0.0 et 0.9
        #self.nb_neurones_par_couche=None # list d'int
        self.fonction_activation=None # fontion ReLU/GELU/tanh
class Parametres_choix_loss_fct():
    def __init__(self):
        self.fonction_perte=None # fonction MSE/MAE/Huber
        self.params=None # paramètres de la fonction perte (dépend de la fonction)
class Parametres_optimisateur():
    def __init__(self):
        self.optimisateur=None # fonction Adam/SGD/RMSprop/Adagrad/Adadelta
        self.learning_rate=None # float
        self.decroissance=None # float
        self.scheduler=None # fonction Plateau/Cosine/OneCycle/None
        self.patience=None # int
class Parametres_entrainement():
    def __init__(self):
        self.nb_epochs=None # int
        self.batch_size=None # int
        #self.nb_workers=None # int
        self.clip_grandient=None # float
        #self.seed=None # int
        #self.Device=None # CPU/CUDA/AUTO
        #self.sauvegarde_checkpoints=None # best/last/all
        #self.early_stopping=None # metric/mode/patience
class Parametres_visualisation_suivi():
    def __init__(self):
        self.metriques=None # list de fonctions ['MSE','MAE'...]
        # self.Frequence_affichage=None # int
        # self.Visualisation_predictions=None # bool
        # self.Frequence_visualisation_predictions=None # int
        # self.Nb_exemples_visualises=None # int

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


def Formatter_JSON():
    config_totale={}
    config_totale["Parametres_temporels"]=Parametres_temporels().__dict__
    config_totale["Parametres_choix_reseau_neurones"]=Parametres_choix_reseau_neurones().__dict__
    config_totale["Parametres_archi_reseau"]=Parametres_archi_reseau().__dict__
    config_totale["Parametres_choix_loss_fct"]=Parametres_choix_loss_fct().__dict__
    config_totale["Parametres_optimisateur"]=Parametres_optimisateur().__dict__
    config_totale["Parametres_entrainement"]=Parametres_entrainement().__dict__
    config_totale["Parametres_visualisation_suivi"]=Parametres_visualisation_suivi().__dict__

    return config_totale

c=Formatter_JSON()
print(c)


# Lancer la boucle principale
fenetree = Fenetre()
fenetree.mainloop()
