import tkinter as tk
from tkinter import ttk
import requests, json

URL = "http://138.231.149.81:8000" 

# Paramètres et variables

class Parametres_temporels_class():
    def __init__(self):
        self.horizon=None # int
        self.dates=None # variable datetime
        self.pas_temporel=None # int
        self.portion_decoupage=None# float entre 0 et 1

class Parametres_choix_reseau_neurones_class():
    def __init__(self):
        self.modele=None # str ['RNN','LSTM','GRU','CNN']
class Parametres_archi_reseau_class():
    def __init__(self):
        self.nb_couches=1 #None # int
        self.hidden_size=1 # int
        self.dropout_rate=1 # float entre 0.0 et 0.9
        #self.nb_neurones_par_couche=None # list d'int
        self.fonction_activation="ReLU" # fontion ReLU/GELU/tanh
class Parametres_choix_loss_fct_class():
    def __init__(self):
        self.fonction_perte=None # fonction MSE/MAE/Huber
        self.params=None # paramètres de la fonction perte (dépend de la fonction)
class Parametres_optimisateur_class():
    def __init__(self):
        self.optimisateur=None # fonction Adam/SGD/RMSprop/Adagrad/Adadelta
        self.learning_rate=None # float
        self.decroissance=None # float
        self.scheduler=None # fonction Plateau/Cosine/OneCycle/None
        self.patience=None # int
class Parametres_entrainement_class():
    def __init__(self):
        self.nb_epochs=None # int
        self.batch_size=None # int
        #self.nb_workers=None # int
        self.clip_grandient=None # float
        #self.seed=None # int
        #self.Device=None # CPU/CUDA/AUTO
        #self.sauvegarde_checkpoints=None # best/last/all
        #self.early_stopping=None # metric/mode/patience
class Parametres_visualisation_suivi_class():
    def __init__(self):
        self.metriques=None # list de fonctions ['MSE','MAE'...]
        # self.Frequence_affichage=None # int
        # self.Visualisation_predictions=None # bool
        # self.Frequence_visualisation_predictions=None # int
        # self.Nb_exemples_visualises=None # int

Parametres_temporels=Parametres_temporels_class()
Parametres_choix_reseau_neurones=Parametres_choix_reseau_neurones_class()
Parametres_archi_reseau=Parametres_archi_reseau_class()
Parametres_choix_loss_fct=Parametres_choix_loss_fct_class()
Parametres_optimisateur=Parametres_optimisateur_class()
Parametres_entrainement=Parametres_entrainement_class()
Parametres_visualisation_suivi=Parametres_visualisation_suivi_class()

# Créer la fenêtre principale
class Fenetre(tk.Tk):
    # bouton: tk.Button(self.fenetre, text="Texte du bouton", command=self.fct_bouton)
    # entrée int: tk.Entry(self.fenetre, validate="key", validatecommand=(self.fenetre.register(self.validate_int), "%P"))
    def __init__(self):
        tk.Tk.__init__(self)


        self.title("Paramétrage du Réseau de Neuronnes")
        self.geometry("500x800")  # largeur x hauteur
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
        tk.Button(self.cadre, text="Envoyer la configuration au serveur", height=3, command=self.EnvoyerConfig).pack(fill="both",pady=20,padx=50)
        tk.Button(self.cadre, text="Quitter", command=self.destroy).pack(fill="both",pady=20,padx=50)

        # Variables pour les paramètres temporels
        self.Params_temporels_horizon = tk.IntVar()
        self.Params_temporels_horizon.set(Parametres_temporels.horizon if Parametres_temporels.horizon is not None else 0)
        # self.Params_temporels_dates = tk.StringVar()
        self.Params_temporels_pas_temporel = tk.IntVar()
        self.Params_temporels_pas_temporel.set(Parametres_temporels.pas_temporel if Parametres_temporels.pas_temporel is not None else 0)
        self.Params_temporels_portion_decoupage = tk.IntVar()
        self.Params_temporels_portion_decoupage.set(Parametres_temporels.portion_decoupage*100 if Parametres_temporels.portion_decoupage is not None else 0)
    
    def Params_temporels(self):

        # Fenêtre secondaire
        self.fenetre_params_temporels = tk.Toplevel(self)
        self.fenetre_params_temporels.title("Paramètres temporels et de découpage de données")
        self.fenetre_params_temporels.geometry("360x300")

        # Message d'accueil
        # tk.Label(self.fenetre_params_temporels, text="Bonjour, Maxime !", font=("Helvetica", 12, "bold")).pack(pady=10)

        # Cadre principal
        cadre = tk.LabelFrame(self.fenetre_params_temporels, text="Configuration", padx=10, pady=10)
        cadre.pack(padx=10, pady=10, fill="both", expand=True)

        # Validation d'entiers
        vcmd = (self.fenetre_params_temporels.register(self.validate_int_fct), "%P")

        # Ligne 1 : Horizon temporel
        tk.Label(cadre, text="Horizon temporel (int) :").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=self.Params_temporels_horizon, validate="key", validatecommand=vcmd).grid(row=0, column=1, pady=5)

        # Ligne 2 : Pas temporel
        tk.Label(cadre, text="Pas temporel (int) :").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=self.Params_temporels_pas_temporel, validate="key", validatecommand=vcmd).grid(row=1, column=1, pady=5)

        # Ligne 3 : Portion découpage
        tk.Label(cadre, text="Portion découpage (%) :").grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=self.Params_temporels_portion_decoupage, validate="key", validatecommand=vcmd).grid(row=2, column=1, pady=5)

        # Boutons
        bouton_frame = tk.Frame(self.fenetre_params_temporels)
        bouton_frame.pack(pady=10)

        def afficher():
            Parametres_temporels.horizon = self.Params_temporels_horizon.get()
            Parametres_temporels.pas_temporel = self.Params_temporels_pas_temporel.get()
            Parametres_temporels.portion_decoupage = self.Params_temporels_portion_decoupage.get() / 100
            print(f"Horizon temporel : {Parametres_temporels.horizon}")
            print(f"Pas temporel : {Parametres_temporels.pas_temporel}")
            print(f"Portion découpage : {Parametres_temporels.portion_decoupage}")

        tk.Button(bouton_frame, text="Afficher", command=afficher).grid(row=0, column=0, padx=10)
        tk.Button(bouton_frame, text="Quitter", command=self.fenetre_params_temporels.destroy).grid(row=0, column=1, padx=10)

        self.fenetre_params_temporels.mainloop()
    
    
    
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
    
    def validate_int_fct(self, text):
        return text.isdigit() or text == ""

    def Formatter_JSON(self):
        self.config_totale={}
        self.config_totale["Parametres_temporels"]=Parametres_temporels.__dict__
        self.config_totale["Parametres_choix_reseau_neurones"]=Parametres_choix_reseau_neurones.__dict__
        self.config_totale["Parametres_archi_reseau"]=Parametres_archi_reseau.__dict__
        self.config_totale["Parametres_choix_loss_fct"]=Parametres_choix_loss_fct.__dict__
        self.config_totale["Parametres_optimisateur"]=Parametres_optimisateur.__dict__
        self.config_totale["Parametres_entrainement"]=Parametres_entrainement.__dict__
        self.config_totale["Parametres_visualisation_suivi"]=Parametres_visualisation_suivi.__dict__
        print(self.config_totale)
        return self.config_totale
    
    def EnvoyerConfig(self):
        self.payload=self.Formatter_JSON()
        r = requests.post(f"{URL}/train_full", json=self.payload)
        print("POST /train_full ->", r.status_code)
        print(json.dumps(r.json(), indent=2))


# Lancer la boucle principale
fenetree = Fenetre()
fenetree.mainloop()

#print(Parametres_temporels.__dict__)