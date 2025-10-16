import tkinter as tk
from tkcalendar import Calendar
from datetime import datetime
import requests, json

URL = "http://138.231.149.81:8000" 

# Paramètres et variables

class Parametres_temporels_class():
    def __init__(self):
        self.horizon=1 # int
        self.dates=["2025-01-01", "2025-01-01"] # variable datetime
        self.pas_temporel=60 # int
        self.portion_decoupage=0.8# float entre 0 et 1
class Parametres_choix_reseau_neurones_class():
    def __init__(self):
        self.modele="RNN" # str ['RNN','LSTM','GRU','CNN']
class Parametres_archi_reseau_class():
    def __init__(self):
        self.nb_couches=2 #None # int
        self.hidden_size=64 # int
        self.dropout_rate=0.0 # float entre 0.0 et 0.9
        #self.nb_neurones_par_couche=None # list d'int
        self.fonction_activation="ReLU" # fontion ReLU/GELU/tanh
class Parametres_choix_loss_fct_class():
    def __init__(self):
        self.fonction_perte="MSE" # fonction MSE/MAE/Huber
        self.params=None # paramètres de la fonction perte (dépend de la fonction)
class Parametres_optimisateur_class():
    def __init__(self):
        self.optimisateur="Adam" # fonction Adam/SGD/RMSprop/Adagrad/Adadelta
        self.learning_rate=0.001 # float
        self.decroissance=0.0 # float
        self.scheduler=None # fonction Plateau/Cosine/OneCycle/None
        self.patience=5 # int
class Parametres_entrainement_class():
    def __init__(self):
        self.nb_epochs=1000 # int
        self.batch_size=4 # int
        #self.nb_workers=None # int
        self.clip_grandient=None # float
        #self.seed=None # int
        #self.Device=None # CPU/CUDA/AUTO
        #self.sauvegarde_checkpoints=None # best/last/all
        #self.early_stopping=None # metric/mode/patience
class Parametres_visualisation_suivi_class():
    def __init__(self):
        self.metriques=["loss"] # list de fonctions ['MSE','MAE'...]
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

        self.Params_temporels_dates = tk.StringVar()
        self.Params_temporels_dates.set(Parametres_temporels.dates)

        self.date_debut_str = tk.StringVar(value=Parametres_temporels.dates[0])
        self.date_fin_str = tk.StringVar(value=Parametres_temporels.dates[1])


        self.Params_temporels_pas_temporel = tk.IntVar()
        self.Params_temporels_pas_temporel.set(Parametres_temporels.pas_temporel if Parametres_temporels.pas_temporel is not None else 0)
        self.Params_temporels_portion_decoupage = tk.IntVar()
        self.Params_temporels_portion_decoupage.set(Parametres_temporels.portion_decoupage*100 if Parametres_temporels.portion_decoupage is not None else 0)
    
    # Fonctions des fenêtres de paramétrage
    
    def Params_temporels(self):

        # Fenêtre secondaire
        fenetre_params_temporels = tk.Toplevel(self)
        fenetre_params_temporels.title("Paramètres temporels et de découpage de données")
        fenetre_params_temporels.geometry("360x300")

        # Message d'accueil
        # tk.Label(self.fenetre_params_temporels, text="Bonjour, Maxime !", font=("Helvetica", 12, "bold")).pack(pady=10)

        # Cadre principal
        cadre = tk.LabelFrame(fenetre_params_temporels, text="Configuration", padx=10, pady=10)
        cadre.pack(padx=10, pady=10, fill="both", expand=True)

        # Validation d'entiers
        vcmd = (fenetre_params_temporels.register(self.validate_int_fct), "%P")

        # Ligne 1 : Horizon temporel
        tk.Label(cadre, text="Horizon temporel (int) :").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=self.Params_temporels_horizon, validate="key", validatecommand=vcmd).grid(row=0, column=1, pady=5)

        # Ligne 2 : Pas temporel
        tk.Label(cadre, text="Pas temporel (int) :").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=self.Params_temporels_pas_temporel, validate="key", validatecommand=vcmd).grid(row=1, column=1, pady=5)

        # Ligne 3 : Portion découpage
        tk.Label(cadre, text="Portion découpage (%) :").grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=self.Params_temporels_portion_decoupage, validate="key", validatecommand=vcmd).grid(row=2, column=1, pady=5)

        # Ligne 4 : Boutons pour sélectionner les dates
        tk.Label(cadre, text="Date de début :").grid(row=3, column=0, sticky="w", pady=5)
        tk.Button(cadre, textvariable=self.date_debut_str, command=self.ouvrir_calendrier_debut).grid(row=3, column=1, pady=5)

        tk.Label(cadre, text="Date de fin :").grid(row=4, column=0, sticky="w", pady=5)
        tk.Button(cadre, textvariable=self.date_fin_str, command=self.ouvrir_calendrier_fin).grid(row=4, column=1, pady=5)


        # Boutons
        bouton_frame = tk.Frame(fenetre_params_temporels)
        bouton_frame.pack(pady=10)

        def afficher():
            Parametres_temporels.horizon = self.Params_temporels_horizon.get()
            Parametres_temporels.pas_temporel = self.Params_temporels_pas_temporel.get()
            Parametres_temporels.portion_decoupage = self.Params_temporels_portion_decoupage.get() / 100
            Parametres_temporels.dates = [self.date_debut_str.get(), self.date_fin_str.get()]
            fenetre_params_temporels.destroy()

        tk.Button(bouton_frame, text="Sauvegarder et quitter", command=afficher).grid(row=0, column=0, padx=10)
        tk.Button(bouton_frame, text="Quitter", command=fenetre_params_temporels.destroy).grid(row=0, column=1, padx=10)

        fenetre_params_temporels.mainloop()
    


    def ouvrir_calendrier_debut(self):
        top = tk.Toplevel(self)
        top.title("Sélectionner la date de début")
        try:
            date_obj = datetime.strptime(self.date_fin_str.get(), "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.today()
        cal = Calendar(top, selectmode='day', date_pattern='yyyy-mm-dd',year=date_obj.year,month=date_obj.month,day=date_obj.day)
        cal.pack(padx=10, pady=10)

        def valider():
            self.date_debut_str.set(cal.get_date())
            top.destroy()

        tk.Button(top, text="Valider", command=valider).pack(pady=10)

    def ouvrir_calendrier_fin(self):
        top = tk.Toplevel(self)
        top.title("Sélectionner la date de fin")
        try:
            date_obj = datetime.strptime(self.date_fin_str.get(), "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.today()

        cal = Calendar(top, selectmode='day', date_pattern='yyyy-mm-dd',year=date_obj.year,month=date_obj.month,day=date_obj.day)
        cal.pack(padx=10, pady=10)

        def valider():
            self.date_fin_str.set(cal.get_date())
            top.destroy()

        tk.Button(top, text="Valider", command=valider).pack(pady=10)






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
    
    # Fonctions utilitaires

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