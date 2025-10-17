import tkinter as tk
from tkcalendar import Calendar
from datetime import datetime
import requests, json
from tkinter import ttk

#URL = "http://138.231.149.81:8000" 
URL = "http://192.168.27.66:8000"


# Paramètres et variables

class Parametres_temporels_class():
    def __init__(self):
        self.horizon=1 # int
        self.dates=["2025-01-01", "2025-01-01"] # variable datetime
        self.pas_temporel=60 # int
        self.portion_decoupage=0.8# float entre 0 et 1
class Parametres_choix_reseau_neurones_class():
    def __init__(self):
        self.modele="MLP" # str ['MLP','LSTM','GRU','CNN']
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
        self.clip_gradient=None # float
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
    def __init__(self):
        tk.Tk.__init__(self)
        couleur_fond = "#d9d9d9"
        self.title("🧠 Paramétrage du Réseau de Neuronnes")

        # Définir une police personnalisée
        self.font_titre = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 11)

        self.geometry("500x1")  # largeur fixe, hauteur minimale

        self.cadre = tk.Frame(self, borderwidth=30)
        self.cadre.configure(bg=couleur_fond)
        self.cadre.pack(fill="both", expand="yes")
        
        # Titre simulé
        tk.Label(self.cadre, text="Paramètres", font=self.font_titre, bg=couleur_fond).pack(anchor="w", pady=(0, 10))

        # Cadre des paramètres
        self.CadreParams = tk.LabelFrame(
            self.cadre, text="", font=self.font_titre,
            bg="#ffffff", fg="#333333", bd=3, relief="ridge", padx=15, pady=15
        )
        self.CadreParams.pack(fill="both", expand=True, pady=(0, 20))

        # Liste des boutons
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
                height=2, bg="#e6e6e6", fg="#000000", relief="groove", bd=2,
                command=commande
            ).pack(fill="x", pady=6, padx=12)

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
        self.geometry(f"500x{self.winfo_reqheight()}")

    # Fonctions des fenêtres de paramétrage
    
    def Params_temporels(self):
        # Variables pour les paramètres temporels
        Params_temporels_horizon = tk.IntVar(value=Parametres_temporels.horizon)
        date_debut_str = tk.StringVar(value=Parametres_temporels.dates[0])
        date_fin_str = tk.StringVar(value=Parametres_temporels.dates[1])
        Params_temporels_pas_temporel = tk.IntVar(value=Parametres_temporels.pas_temporel)
        Params_temporels_portion_decoupage = tk.IntVar(value=Parametres_temporels.portion_decoupage*100)

        def ouvrir_calendrier_debut():
            top = tk.Toplevel(self)
            top.title("Sélectionner la date de début")
            try:
                date_obj = datetime.strptime(date_debut_str.get(), "%Y-%m-%d")
            except ValueError:
                date_obj = datetime.today()
            cal = Calendar(top, selectmode='day', date_pattern='yyyy-mm-dd',
                        year=date_obj.year, month=date_obj.month, day=date_obj.day)
            cal.pack(padx=10, pady=10)
            tk.Button(top, text="Valider", command=lambda: (date_debut_str.set(cal.get_date()), top.destroy())).pack(pady=10)

        # Fonction locale : ouvrir calendrier fin
        def ouvrir_calendrier_fin():
            top = tk.Toplevel(self)
            top.title("Sélectionner la date de fin")
            try:
                date_obj = datetime.strptime(date_fin_str.get(), "%Y-%m-%d")
            except ValueError:
                date_obj = datetime.today()
            cal = Calendar(top, selectmode='day', date_pattern='yyyy-mm-dd',
                        year=date_obj.year, month=date_obj.month, day=date_obj.day)
            cal.pack(padx=10, pady=10)
            tk.Button(top, text="Valider", command=lambda: (date_fin_str.set(cal.get_date()), top.destroy())).pack(pady=10)

        def Save_quit():
            Parametres_temporels.horizon = Params_temporels_horizon.get()
            Parametres_temporels.pas_temporel = Params_temporels_pas_temporel.get()
            Parametres_temporels.portion_decoupage = Params_temporels_portion_decoupage.get() / 100
            Parametres_temporels.dates = [date_debut_str.get(), date_fin_str.get()]
            fenetre_params_temporels.destroy()
        
        def Quit():
            Params_temporels_horizon.set(Parametres_temporels.horizon)
            date_debut_str.set(Parametres_temporels.dates[0])
            date_fin_str.set(value=Parametres_temporels.dates[1])
            Params_temporels_pas_temporel.set(Parametres_temporels.pas_temporel)
            Params_temporels_portion_decoupage.set(Parametres_temporels.portion_decoupage*100)
        
            fenetre_params_temporels.destroy()


        # Fenêtre secondaire
        fenetre_params_temporels = tk.Toplevel(self)
        fenetre_params_temporels.title("Paramètres temporels et de découpage de données")
        fenetre_params_temporels.geometry("")

        # Cadre principal
        cadre = tk.LabelFrame(fenetre_params_temporels, text="Configuration", padx=10, pady=10)
        cadre.pack(padx=10, pady=10, fill="both", expand=True)

        # Validation d'entiers
        vcmd = (fenetre_params_temporels.register(self.validate_int_fct), "%P")

        # Ligne 1 : Horizon temporel
        tk.Label(cadre, text="Horizon temporel (int) :").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_temporels_horizon, validate="key", validatecommand=vcmd).grid(row=0, column=1, pady=5)

        # Ligne 2 : Pas temporel
        tk.Label(cadre, text="Pas temporel (int) :").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_temporels_pas_temporel, validate="key", validatecommand=vcmd).grid(row=1, column=1, pady=5)

        # Ligne 3 : Portion découpage
        tk.Label(cadre, text="Portion découpage (%) :").grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_temporels_portion_decoupage, validate="key", validatecommand=vcmd).grid(row=2, column=1, pady=5)

        # Ligne 4 : Boutons pour sélectionner les dates
        tk.Label(cadre, text="Date de début :").grid(row=3, column=0, sticky="w", pady=5)
        tk.Button(cadre, textvariable=date_debut_str, command=ouvrir_calendrier_debut).grid(row=3, column=1, pady=5)

        tk.Label(cadre, text="Date de fin :").grid(row=4, column=0, sticky="w", pady=5)
        tk.Button(cadre, textvariable=date_fin_str, command=ouvrir_calendrier_fin).grid(row=4, column=1, pady=5)


        # Boutons
        bouton_frame = tk.Frame(fenetre_params_temporels)
        bouton_frame.pack(pady=10)
        tk.Button(bouton_frame, text="Sauvegarder et quitter", command=Save_quit).grid(row=0, column=0, padx=10)
        tk.Button(bouton_frame, text="Quitter", command=Quit).grid(row=0, column=1, padx=10)
        
        fenetre_params_temporels.mainloop()

    def Params_choix_reseau_neurones(self):
        # Variables pour les paramètres
        Params_choix_reseau_neurones_modele = tk.StringVar(value=Parametres_choix_reseau_neurones.modele) # str ['MLP','LSTM','GRU','CNN']

        def Save_quit():
            Parametres_choix_reseau_neurones.modele = Params_choix_reseau_neurones_modele.get()
            fenetre_params_choix_reseau_neurones.destroy()
        
        def Quit():
            Params_choix_reseau_neurones_modele.set(Parametres_choix_reseau_neurones.modele)
            fenetre_params_choix_reseau_neurones.destroy()

        # Fenêtre secondaire
        fenetre_params_choix_reseau_neurones = tk.Toplevel(self)
        fenetre_params_choix_reseau_neurones.title("Paramètres de Choix du réseau de neurones")
        fenetre_params_choix_reseau_neurones.geometry("")
        
        # Cadre principal
        cadre = tk.LabelFrame(fenetre_params_choix_reseau_neurones, text="Configuration", padx=10, pady=10)
        cadre.pack(padx=10, pady=10, fill="both", expand=True)

        # Ligne 1 : Horizon temporel
        tk.Label(cadre, text="Choix du modèle :").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Combobox(cadre, values =["MLP","LSTM","GRU","CNN"],textvariable=Params_choix_reseau_neurones_modele,state="readonly").grid(row=0, column=1, pady=5)

        # Boutons
        bouton_frame = tk.Frame(fenetre_params_choix_reseau_neurones)
        bouton_frame.pack(pady=10)
        tk.Button(bouton_frame, text="Sauvegarder et quitter", command=Save_quit).grid(row=0, column=0, padx=10)
        tk.Button(bouton_frame, text="Quitter", command=Quit).grid(row=0, column=1, padx=10)
        
        fenetre_params_choix_reseau_neurones.mainloop()
    
    def Params_archi_reseau(self):
        # Variables pour les paramètres
        Params_archi_reseau_nb_couches = tk.IntVar(value=Parametres_archi_reseau.nb_couches) # int
        Params_archi_reseau_hidden_size = tk.IntVar(value=Parametres_archi_reseau.hidden_size) # int
        Params_archi_reseau_dropout_rate = tk.DoubleVar(value=Parametres_archi_reseau.dropout_rate) # float entre 0.0 et 0.9
        Params_archi_reseau_fonction_activation = tk.StringVar(value=Parametres_archi_reseau.fonction_activation) # fontion ReLU/GELU/tanh

        def Save_quit():
            Parametres_archi_reseau.nb_couches = Params_archi_reseau_nb_couches.get()
            Parametres_archi_reseau.hidden_size = Params_archi_reseau_hidden_size.get()
            Parametres_archi_reseau.dropout_rate = Params_archi_reseau_dropout_rate.get()
            Parametres_archi_reseau.fonction_activation = Params_archi_reseau_fonction_activation.get()
            fenetre_params_archi_reseau.destroy()
        
        def Quit():
            Params_archi_reseau_nb_couches.set(Parametres_archi_reseau.nb_couches)
            Params_archi_reseau_hidden_size.set(Parametres_archi_reseau.hidden_size)
            Params_archi_reseau_dropout_rate.set(Parametres_archi_reseau.dropout_rate)
            Params_archi_reseau_fonction_activation.set(Parametres_archi_reseau.fonction_activation)
            fenetre_params_archi_reseau.destroy()

        # Fenêtre secondaire
        fenetre_params_archi_reseau = tk.Toplevel(self)
        fenetre_params_archi_reseau.title("Paramètres de l'architechture du réseau de neurones")
        fenetre_params_archi_reseau.geometry("")
        
        # Cadre principal
        cadre = tk.LabelFrame(fenetre_params_archi_reseau, text="Configuration", padx=10, pady=10)
        cadre.pack(padx=10, pady=10, fill="both", expand=True)


        # Validation d'entiers
        vcmd = (fenetre_params_archi_reseau.register(self.validate_int_fct), "%P")

        # Ligne 1 : Nombre de couches de neurones
        tk.Label(cadre, text="Nombre de couches de neurones :").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_archi_reseau_nb_couches, validate="key", validatecommand=vcmd).grid(row=0, column=1, pady=5)

        # Ligne 2 : Taille des couches cachées
        tk.Label(cadre, text="Taille des couches cachées :").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_archi_reseau_hidden_size, validate="key", validatecommand=vcmd).grid(row=1, column=1, pady=5)

        # Ligne 3 : Taux de dropout
        tk.Label(cadre, text="Taux de dropout (0.0 - 0.9) :").grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_archi_reseau_dropout_rate).grid(row=2, column=1, pady=5)

        # Ligne 4 : Fonction d'activation
        tk.Label(cadre, text="Fonction d'activation :").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Combobox(cadre, values =["ReLU","GELU","tanh"],textvariable=Params_archi_reseau_fonction_activation,state="readonly").grid(row=3, column=1, pady=5)

        # Boutons
        bouton_frame = tk.Frame(fenetre_params_archi_reseau)
        bouton_frame.pack(pady=10)
        tk.Button(bouton_frame, text="Sauvegarder et quitter", command=Save_quit).grid(row=0, column=0, padx=10)
        tk.Button(bouton_frame, text="Quitter", command=Quit).grid(row=0, column=1, padx=10)
        
        fenetre_params_archi_reseau.mainloop()
    
    def Params_choix_loss_fct(self):
        # Variables pour les paramètres
        Params_choix_loss_fct_fonction_perte = tk.StringVar(value=Parametres_choix_loss_fct.fonction_perte) # fonction MSE/MAE/Huber

        def Save_quit():
            Parametres_choix_loss_fct.fonction_perte = Params_choix_loss_fct_fonction_perte.get()
            fenetre_params_choix_loss_fct.destroy()
        
        def Quit():
            Params_choix_loss_fct_fonction_perte.set(Parametres_choix_loss_fct.fonction_perte)
            fenetre_params_choix_loss_fct.destroy()
        
        # Fenêtre secondaire
        fenetre_params_choix_loss_fct = tk.Toplevel(self)
        fenetre_params_choix_loss_fct.title("Paramètres de Choix de la fonction perte (loss)")
        fenetre_params_choix_loss_fct.geometry("")

        # Cadre principal
        cadre = tk.LabelFrame(fenetre_params_choix_loss_fct, text="Configuration", padx=10, pady=10)
        cadre.pack(padx=10, pady=10, fill="both", expand=True)

        # Ligne 1 : Choix de la fonction perte
        tk.Label(cadre, text="Choix de la fonction perte :").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Combobox(cadre, values =["MSE","MAE","Huber"],textvariable=Params_choix_loss_fct_fonction_perte,state="readonly").grid(row=0, column=1, pady=5)

        # Boutons
        bouton_frame = tk.Frame(fenetre_params_choix_loss_fct)
        bouton_frame.pack(pady=10)
        tk.Button(bouton_frame, text="Sauvegarder et quitter", command=Save_quit).grid(row=0, column=0, padx=10)
        tk.Button(bouton_frame, text="Quitter", command=Quit).grid(row=0, column=1, padx=10)

        fenetre_params_choix_loss_fct.mainloop()
    
    def Params_optimisateur(self):
        # Variables pour les paramètres
        Params_optimisateur_optimisateur = tk.StringVar(value=Parametres_optimisateur.optimisateur) # fonction Adam/SGD/RMSprop/Adagrad/Adadelta
        Params_optimisateur_learning_rate = tk.DoubleVar(value=Parametres_optimisateur.learning_rate) # float
        Params_optimisateur_decroissance = tk.DoubleVar(value=Parametres_optimisateur.decroissance) # float
        Params_optimisateur_scheduler = tk.StringVar(value=Parametres_optimisateur.scheduler) # fonction Plateau/Cosine/OneCycle/None
        Params_optimisateur_patience = tk.IntVar(value=Parametres_optimisateur.patience) # int

        def Save_quit():
            Parametres_optimisateur.optimisateur = Params_optimisateur_optimisateur.get()
            Parametres_optimisateur.learning_rate = Params_optimisateur_learning_rate.get()
            Parametres_optimisateur.decroissance = Params_optimisateur_decroissance.get()
            Parametres_optimisateur.scheduler = Params_optimisateur_scheduler.get()
            Parametres_optimisateur.patience = Params_optimisateur_patience.get()
            fenetre_params_optimisateur.destroy()
        
        def Quit():
            Params_optimisateur_optimisateur.set(Parametres_optimisateur.optimisateur)
            Params_optimisateur_learning_rate.set(Parametres_optimisateur.learning_rate)
            Params_optimisateur_decroissance.set(Parametres_optimisateur.decroissance)
            Params_optimisateur_scheduler.set(Parametres_optimisateur.scheduler)
            Params_optimisateur_patience.set(Parametres_optimisateur.patience)
            fenetre_params_optimisateur.destroy()

        # Fenêtre secondaire
        fenetre_params_optimisateur = tk.Toplevel(self)
        fenetre_params_optimisateur.title("Paramètres de l'Optimisation")
        fenetre_params_optimisateur.geometry("")
        
        # Cadre principal
        cadre = tk.LabelFrame(fenetre_params_optimisateur, text="Configuration", padx=10, pady=10)
        cadre.pack(padx=10, pady=10, fill="both", expand=True)

        # Validation d'entiers
        vcmd_int = (fenetre_params_optimisateur.register(self.validate_int_fct), "%P")

        # Ligne 1 : Choix de l'optimisateur
        tk.Label(cadre, text="Choix de l'optimisateur :").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Combobox(cadre, values =["Adam","SGD","RMSprop","Adagrad","Adadelta"],textvariable=Params_optimisateur_optimisateur,state="readonly").grid(row=0, column=1, pady=5)

        # Ligne 2 : Taux d'apprentissage
        tk.Label(cadre, text="Taux d'apprentissage :").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_optimisateur_learning_rate).grid(row=1, column=1, pady=5)

        # Ligne 3 : Décroissance
        tk.Label(cadre, text="Décroissance :").grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_optimisateur_decroissance).grid(row=2, column=1, pady=5)

        # Ligne 4 : Scheduler
        tk.Label(cadre, text="Scheduler :").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Combobox(cadre, values =["Plateau","Cosine","OneCycle","None"],textvariable=Params_optimisateur_scheduler,state="readonly").grid(row=3, column=1, pady=5)

        # Ligne 5 : Patience
        tk.Label(cadre, text="Patience (int) :").grid(row=4, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_optimisateur_patience, validate="key", validatecommand=vcmd_int).grid(row=4, column=1, pady=5)

        # Boutons
        bouton_frame = tk.Frame(fenetre_params_optimisateur)
        bouton_frame.pack(pady=10)
        tk.Button(bouton_frame, text="Sauvegarder et quitter", command=Save_quit).grid(row=0, column=0, padx=10)
        tk.Button(bouton_frame, text="Quitter", command=Quit).grid(row=0, column=1, padx=10)

        fenetre_params_optimisateur.mainloop()
    
    def Params_entrainement(self):
        # Variables pour les paramètres
        Params_entrainement_nb_epochs = tk.IntVar(value=Parametres_entrainement.nb_epochs) # int
        Params_entrainement_batch_size = tk.IntVar(value=Parametres_entrainement.batch_size) # int
        Params_entrainement_clip_gradient = tk.DoubleVar(value=Parametres_entrainement.clip_gradient if Parametres_entrainement.clip_gradient is not None else 0.0) # float

        def Save_quit():
            Parametres_entrainement.nb_epochs = Params_entrainement_nb_epochs.get()
            Parametres_entrainement.batch_size = Params_entrainement_batch_size.get()
            Parametres_entrainement.clip_gradient = Params_entrainement_clip_gradient.get() if Params_entrainement_clip_gradient.get() != 0.0 else None
            fenetre_params_entrainement.destroy()
        
        def Quit():
            Params_entrainement_nb_epochs.set(Parametres_entrainement.nb_epochs)
            Params_entrainement_batch_size.set(Parametres_entrainement.batch_size)
            Params_entrainement_clip_gradient.set(Parametres_entrainement.clip_gradient if Parametres_entrainement.clip_gradient is not None else 0.0)
            fenetre_params_entrainement.destroy()

        # Fenêtre secondaire
        fenetre_params_entrainement = tk.Toplevel(self)
        fenetre_params_entrainement.title("Paramètres d'Entrainement")
        fenetre_params_entrainement.geometry("")
        
        # Cadre principal
        cadre = tk.LabelFrame(fenetre_params_entrainement, text="Configuration", padx=10, pady=10)
        cadre.pack(padx=10, pady=10, fill="both", expand=True)

        # Validation d'entiers
        vcmd_int = (fenetre_params_entrainement.register(self.validate_int_fct), "%P")

        # Ligne 1 : Nombre d'epochs
        tk.Label(cadre, text="Nombre d'epochs:").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_entrainement_nb_epochs, validate="key", validatecommand=vcmd_int).grid(row=0, column=1, pady=5)

        # Ligne 2 : Taille du batch
        tk.Label(cadre, text="Taille du batch:").grid(row=1, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_entrainement_batch_size, validate="key", validatecommand=vcmd_int).grid(row=1, column=1, pady=5)

        # Ligne 3 : Clip des gradients
        tk.Label(cadre, text="Clip des gradients (0.0 pour None):").grid(row=2, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_entrainement_clip_gradient).grid(row=2, column=1, pady=5)

        # Boutons
        bouton_frame = tk.Frame(fenetre_params_entrainement)
        bouton_frame.pack(pady=10)
        tk.Button(bouton_frame, text="Sauvegarder et quitter", command=Save_quit).grid(row=0, column=0, padx=10)
        tk.Button(bouton_frame, text="Quitter", command=Quit).grid(row=0, column=1, padx=10)

        fenetre_params_entrainement.mainloop()
    
    def Params_visualisation_suivi(self):
        # Variables pour les paramètres
        Params_visualisation_suivi_metriques = tk.StringVar(value=",".join(Parametres_visualisation_suivi.metriques)) # list de fonctions ['MSE','MAE'...]

        def Save_quit():
            Parametres_visualisation_suivi.metriques = [m.strip() for m in Params_visualisation_suivi_metriques.get().split(",") if m.strip()]
            fenetre_params_visualisation_suivi.destroy()
        
        def Quit():
            Params_visualisation_suivi_metriques.set(",".join(Parametres_visualisation_suivi.metriques))
            fenetre_params_visualisation_suivi.destroy()
        
        # Fenêtre secondaire
        fenetre_params_visualisation_suivi = tk.Toplevel(self)
        fenetre_params_visualisation_suivi.title("Paramètres de Visualisation et Suivi")
        fenetre_params_visualisation_suivi.geometry("")
        
        # Cadre principal
        cadre = tk.LabelFrame(fenetre_params_visualisation_suivi, text="Configuration", padx=10, pady=10)
        cadre.pack(padx=10, pady=10, fill="both", expand=True)
        
        # Ligne 1 : Choix des métriques
        tk.Label(cadre, text="Choix des métriques (séparées par des virgules) :").grid(row=0, column=0, sticky="w", pady=5)
        tk.Entry(cadre, textvariable=Params_visualisation_suivi_metriques).grid(row=0, column=1, pady=5)
        
        # Boutons
        bouton_frame = tk.Frame(fenetre_params_visualisation_suivi)
        bouton_frame.pack(pady=10)
        tk.Button(bouton_frame, text="Sauvegarder et quitter", command=Save_quit).grid(row=0, column=0, padx=10)
        tk.Button(bouton_frame, text="Quitter", command=Quit).grid(row=0, column=1, padx=10)

        fenetre_params_visualisation_suivi.mainloop()

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
        print(self.config_totale)
        return self.config_totale
    
    def EnvoyerConfig(self):
        self.payload=self.Formatter_JSON()
        with requests.post(f"{URL}/train_full", json=self.payload, stream=True) as r:
            r.raise_for_status()
            print("Content-Type:", r.headers.get("content-type"))  # doit être text/event-stream
            for line in r.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    msg = json.loads(line[6:].decode("utf-8"))
                    # msg = {"epoch": i, "avg_loss": ...} puis {"done": True, "final_loss": ...}
                    print("EVENT:", msg)
                    if msg.get("done"):
                        break

# Lancer la boucle principale
fenetree = Fenetre()
fenetree.mainloop()
