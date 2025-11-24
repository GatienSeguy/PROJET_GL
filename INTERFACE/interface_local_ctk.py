import tkinter as tk
from tkcalendar import Calendar
from datetime import datetime
import requests, json
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import threading
import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
from tkinter.filedialog import asksaveasfilename
matplotlib.use("TkAgg")  # backend Tkinter
import time
import customtkinter as ctk



# URL = "http://192.168.27.66:8000"
# URL = "http://138.231.149.81:8000"
URL = "http://192.168.1.190:8000"

# Paramètres et variables

class Parametres_temporels_class():
    def __init__(self):
        self.nom_dataset="" # str
        self.horizon=1 # int
        self.dates=["2001-01-01", "2025-01-02"] # variable datetime
        self.pas_temporel=1 # int
        self.portion_decoupage=0.8# float entre 0 et 1
        # self.dataset="" # str

class Parametres_choix_reseau_neurones_class():
    def __init__(self):
        self.modele="MLP" # str ['MLP','LSTM','GRU','CNN']
class Parametres_archi_reseau_class():
    class MLP_params():
        def __init__(self):
            self.nb_couches=2 #None # int
            self.hidden_size=64 # int
            self.dropout_rate=0.0 # float entre 0.0 et 0.9
            #self.nb_neurones_par_couche=None # list d'int
            self.fonction_activation="ReLU" # fontion ReLU/GELU/tanh
    
    class CNN_params():
        def __init__(self):
            self.nb_couches=2 #None # int
            self.hidden_size=64 # int
            self.dropout_rate=0.0 # float entre 0.0 et 0.9
            #self.nb_neurones_par_couche=None # list d'int
            self.fonction_activation="ReLU" # fontion ReLU/GELU/tanh
            self.kernel_size=3 # int
            self.stride=1 # int
            self.padding=0 # int
    
    class LSTM_params():
        def __init__(self):
            self.nb_couches=2 #None # int
            self.hidden_size=64 # int
            self.bidirectional=False # bool
            self.batch_first=False # bool

    # def __init__(self):
        # self.nb_couches=2 #None # int
        # self.hidden_size=64 # int
        # self.dropout_rate=0.0 # float entre 0.0 et 0.9
        # #self.nb_neurones_par_couche=None # list d'int
        # self.fonction_activation="ReLU" # fontion ReLU/GELU/tanh

        # self.bidirectional=False # bool
        # self.batch_first=False # bool

        # self.kernel_size=3 # int
        # self.stride=1 # int
        # self.padding=0 # int
class Parametres_choix_loss_fct_class():
    def __init__(self):
        self.fonction_perte="MSE" # fonction MSE/MAE/Huber
        self.params=None # paramètres de la fonction perte (dépend de la fonction)
class Parametres_optimisateur_class():
    def __init__(self):
        self.optimisateur="Adam" # fonction Adam/SGD/RMSprop/Adagrad/Adadelta
        self.learning_rate=0.001 # float
        self.decroissance=0.0 # float
        self.scheduler="None" # fonction Plateau/Cosine/OneCycle/None
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

class Selected_Dataset_class():
    def __init__(self):
        self.name="" # str
        self.dates=[] # list de str
        self.pas_temporel=0 # int

Datasets_list=[]
Dataset=""

Parametres_temporels=Parametres_temporels_class()
Parametres_choix_reseau_neurones=Parametres_choix_reseau_neurones_class()
Parametres_archi_reseau_MLP=Parametres_archi_reseau_class.MLP_params()
Parametres_archi_reseau_CNN=Parametres_archi_reseau_class.CNN_params()
Parametres_archi_reseau_LSTM=Parametres_archi_reseau_class.LSTM_params()
Parametres_choix_loss_fct=Parametres_choix_loss_fct_class()
Parametres_optimisateur=Parametres_optimisateur_class()
Parametres_entrainement=Parametres_entrainement_class()
Parametres_visualisation_suivi=Parametres_visualisation_suivi_class()

Selected_Dataset=Selected_Dataset_class()

# Couleurs style IRMA Conseil
BG_DARK = "#2c3e50"
BG_CARD = "#34495e"
BG_INPUT = "#273747"
TEXT_PRIMARY = "#ecf0f1"
TEXT_SECONDARY = "#bdc3c7"
ACCENT_PRIMARY = "#e74c3c"
ACCENT_SECONDARY = "#3498db"
BORDER_COLOR = "#4a5f7f"

# Créer la fenêtre d'accueil
class Fenetre_Acceuil(ctk.CTk):
    def __init__(self):
        self.JSON_Datasets=self.obtenir_datasets()
        self.cadres_bg="#eaf2f8"
        self.cadres_fg="#e4eff8"
        self.fenetre_bg="#f0f4f8"
        self.stop_training = False  # drapeau d’annulation
        self.Payload={}
        self.Fenetre_Params_instance = None
        self.Fenetre_Params_horizon_instance = None
        self.Fenetre_Choix_datasets_instance = None
        self.Fenetre_Choix_metriques_instance = None
        # self.feur_instance = None

        ctk.CTk.__init__(self)
        self.grid_columnconfigure(0, weight=0, minsize=600)  # largeur fixe
        self.grid_columnconfigure(1, weight=1)                # extensible
        self.grid_rowconfigure(0, weight=1)
        self.title("🧠 Paramétrage du Réseau de Neuronnes")
        #self.configure(bg=self.fenetre_bg)
        # self.geometry("1200x700")

        #feur
        # style = ttk.Style()
        # style.theme_use('default')

        # style.configure('TNotebook', background=self.cadres_bg, borderwidth=0)
        # style.configure('TNotebook.Tab', background=self.cadres_bg, padding=[10, 5])
        # style.configure('TNotebook.Tab', foreground="black")

        # Couleur du texte quand l'onglet est sélectionné
        # style.map('TNotebook.Tab', foreground=[('selected', 'black')])



        # Polices
        self.font_titre = ("Roboto Medium", 28,"bold")
        self.font_section = ("Roboto Medium", 24,"bold")
        self.font_bouton = ("Roboto", 20)

        # Cadre principal de configuration
        self.cadre = ctk.CTkFrame(self, corner_radius=10)
        self.cadre.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        self.cadre.grid_columnconfigure(0, weight=1)




        # Cadre des résultats
        # self.Cadre_results_global = ctk.CTkFrame(self, corner_radius=10)
        # self.Cadre_results_global.grid(row=0, column=1, sticky="nsew", padx=(0,20), pady=20)
    
        self.Results_notebook = ctk.CTkTabview(self,corner_radius=10)
        self.Results_notebook.grid(row=0, column=1, sticky="nsew", padx=(0,20),pady=(0,20))

        #Créer les onglets
        self.Results_notebook.add("Training")
        self.Results_notebook.add("Testing")
        self.Results_notebook.add("Metrics")
        self.Results_notebook.add("Prediction")

        # Créer les cadres dans les onglets
        self.Cadre_results_Entrainement = Cadre_Entrainement(
            self, self.Results_notebook.tab("Training")
        )
        self.Cadre_results_Entrainement.pack(fill="both", expand=True)
        self.Cadre_results_Testing = Cadre_Testing(
            self, self.Results_notebook.tab("Testing")
        )
        self.Cadre_results_Metrics = Cadre_Metrics(
            self, self.Results_notebook.tab("Metrics")
        )
        self.Cadre_results_Prediction = Cadre_Prediction(
            self, self.Results_notebook.tab("Prediction")
        )


        # Titre
        ctk.CTkLabel(
            self.cadre, 
            text="MLApp", 
            font=("Roboto Medium", 30)
        ).pack(pady=(20, 10))
        # Sous-titre
        ctk.CTkLabel(
            self.cadre,
            text="Machine Learning Application",
            font=("Roboto", 20)
        ).pack(pady=(0, 40))


        # Section 1 : Modèle
        Label_frame_Modele,Titre_frame_Modele = self.label_frame(self.cadre, title="🧬 Modèle", font=self.font_section)
        Titre_frame_Modele.pack()
        Label_frame_Modele.pack(fill="both",padx=10)
        Label_frame_Modele.grid_columnconfigure(0, weight=1)
        Label_frame_Modele.grid_rowconfigure(0, weight=1)
        
        self.bouton(Label_frame_Modele, "📂 Charger Modèle", self.test,height=40).grid(row=0, column=0,padx=20,pady=20, sticky="nsew")
        self.bouton(Label_frame_Modele, "⚙️ Paramétrer Modèle", self.Parametrer_modele,height=40).grid(row=1, column=0,padx=20,pady=(0,20), sticky="nsew")


        # Section 2 : Données
        Label_frame_Donnees,Titre_frame_Donnees = self.label_frame(self.cadre, title="📊 Données", font=self.font_section)
        Titre_frame_Donnees.pack(pady=(30,0))
        Label_frame_Donnees.pack(fill="both",padx=10)
        Label_frame_Donnees.grid_columnconfigure(0, weight=1)
        Label_frame_Donnees.grid_rowconfigure(0, weight=1)

        
        # self.bouton(Label_frame_Donnees, "📁 Choix Dataset", self.Parametrer_dataset,height=40).grid(row=0, column=0,padx=20,pady=20, sticky="nsew")
        optionmenu_var = ctk.StringVar(value=Dataset)  # set initial value
        # JSON_Datasets_String=JSON_Datasets_String.replace("'", '"')
        # JSON_Datasets_Dict=json.loads(JSON_Datasets_String)


        def optionmenu_callback(choice):            
            Selected_Dataset.name=choice
            Selected_Dataset.dates=[d.split(" ")[0] for d in self.JSON_Datasets[choice]['dates']]
            Selected_Dataset.pas_temporel=self.JSON_Datasets[choice]['pas_temporel']

            Parametres_temporels.nom_dataset=Selected_Dataset.name
            Parametres_temporels.dates=Selected_Dataset.dates
            Parametres_temporels.pas_temporel=1

        combobox = ctk.CTkOptionMenu(master=Label_frame_Donnees,
                                            values=list(self.JSON_Datasets.keys()),
                                            command=optionmenu_callback,
                                            variable=optionmenu_var,
                                            )
        combobox.grid(row=0, column=0,padx=20,pady=20, sticky="nsew")
        
        
        self.bouton(Label_frame_Donnees, "📅 Paramétrer Horizon", self.Parametrer_horizon,height=40).grid(row=1, column=0,padx=20,pady=(0,20), sticky="nsew")
        
        self.bouton(self.cadre, "📈 Choix Métriques et Visualisations", self.Parametrer_metriques,height=40).pack(fill="both",padx=30,pady=(40,0))

        # Section 3 : Actions
        section_actions = ctk.CTkFrame(self.cadre,corner_radius=10,fg_color=self.cadre.cget("fg_color")) #fg_color=root.cget('fg_color')
        section_actions.pack(side="bottom",fill="both",pady=(0,10),padx=10)
        section_actions.grid_columnconfigure(0, weight=1)
        section_actions.grid_columnconfigure(1, weight=1)
        section_actions.grid_rowconfigure(0, weight=1)
        section_actions.grid_rowconfigure(1, weight=1)

        # self.bouton(section_actions, "🚀 Envoyer la configuration au serveur", self.EnvoyerConfig, bg="#d4efdf", fg="#145a32")
        # self.bouton(section_actions, "🛑 Annuler l'entraînement", self.annuler_entrainement, bg="#f9e79f", fg="#7d6608")
        # self.bouton(section_actions, "❌ Quitter", self.destroy, bg="#f5b7b1", fg="#641e16")
        
        start_btn = self.bouton(section_actions, "🚀 Start", self.EnvoyerConfig, height=60, bg="#d4efdf", fg="#145a32", font=("Roboto", 20))
        stop_btn = self.bouton(section_actions, "🛑 Stop", self.annuler_entrainement, height=60, bg="#f9e79f", fg="#7d6608", font=("Roboto", 20))
        quit_btn = self.bouton(section_actions, "❌ Quitter", self.destroy, height=60, bg="#f5b7b1", fg="#641e16", font=("Roboto", 20))

        # Alignement horizontal
        start_btn.grid(row=0, column=0, padx=10, pady=(10,0),sticky="nsew")
        stop_btn.grid(row=0, column=1, padx=10, pady=(10,0),sticky="nsew")
        quit_btn.grid(row=1, column=0, columnspan=2, padx=10, pady=(20,10),sticky="nsew")




        # self.update_idletasks()
        # self.geometry(f"520x{self.winfo_reqheight()}")

        #self.attributes('-fullscreen', True)  # Enable fullscreen
        self.after(100, lambda: self.state("zoomed"))
        self.bind("<Escape>", lambda event: self.attributes('-fullscreen', False))
        self.bind("<F11>", lambda event: self.attributes('-fullscreen', not self.attributes('-fullscreen')))

    def label_frame(self, root, title, font, width=200, height=200):
        # Crée un cadre avec un titre simulant un LabelFrame
        frame = ctk.CTkFrame(root, width=width, height=height, corner_radius=10, fg_color=root.cget('fg_color'), border_width=2, border_color="gray")
        #frame.pack_propagate(False)

        title_label = ctk.CTkLabel(root, text="  "+title+"  ", font=font)
        return frame, title_label
    
    def bouton(self, parent, texte, commande,padx=5, pady=20,bg=None, fg=None, font=None, width=None, height=None,pack=False):
        """Crée un bouton avec le style CustomTkinter par défaut"""
        btn = ctk.CTkButton(
            parent,
            text=texte,
            command=commande
        )

        if font==None:
            btn.configure(font=self.font_bouton)
        else:
            btn.configure(font=font)
        if bg!=None:
            btn.configure(fg_color=bg)
        if fg!=None:
            btn.configure(text_color=fg)
        if height!=None:
            btn.configure(height=height)
        if width!=None:
            btn.configure(width=width)
        if pack==True:
            btn.pack(fill="x",pady=pady, padx=padx)
        return btn
    
    def annuler_entrainement(self):
        """Annule l'entraînement sans fermer le programme."""
        if not self.stop_training:
            self.stop_training = True
            messagebox.showinfo("Annulation", "L'entraînement en cours a été annulé.")
        else:
            messagebox.showwarning("Info", "Aucun entraînement en cours ou déjà annulé.")

    def test(self):
        print("test")
    
    def Parametrer_modele(self):
        if self.Fenetre_Params_instance is None or not self.Fenetre_Params_instance.est_ouverte():
            self.Fenetre_Params_instance = Fenetre_Params(self)
        else:
            self.Fenetre_Params_instance.lift()  # Ramène la fenêtre secondaire au premier plan
    
    def Parametrer_horizon(self):
        if self.Fenetre_Params_horizon_instance is None or not self.Fenetre_Params_horizon_instance.est_ouverte():
            self.Fenetre_Params_horizon_instance = Fenetre_Params_horizon(self)
        else:
            self.Fenetre_Params_horizon_instance.lift()  # Ramène la fenêtre secondaire au premier plan

    def Parametrer_metriques(self):
        if self.Fenetre_Choix_metriques_instance is None or not self.Fenetre_Choix_metriques_instance.est_ouverte():
            self.Fenetre_Choix_metriques_instance = Fenetre_Choix_metriques(self)
        else:
            self.Fenetre_Choix_metriques_instance.lift()  # Ramène la fenêtre secondaire au premier plan
    
    def Formatter_JSON_global(self):
        self.config_totale={}
        # self.config_totale["Parametres_temporels"]=Parametres_temporels.__dict__
        self.config_totale["Parametres_temporels"] = {
            "horizon": Parametres_temporels.horizon,
            "portion_decoupage": Parametres_temporels.portion_decoupage
        }
        self.config_totale["Parametres_choix_reseau_neurones"]=Parametres_choix_reseau_neurones.__dict__
        #self.config_totale["Parametres_archi_reseau"]=Parametres_archi_reseau.__dict__
        self.config_totale["Parametres_choix_loss_fct"]=Parametres_choix_loss_fct.__dict__
        self.config_totale["Parametres_optimisateur"]=Parametres_optimisateur.__dict__
        self.config_totale["Parametres_entrainement"]=Parametres_entrainement.__dict__
        self.config_totale["Parametres_visualisation_suivi"]=Parametres_visualisation_suivi.__dict__
        return self.config_totale

    def Formatter_JSON_dataset(self):
        self.config_dataset={}
        self.config_dataset["name"]=Selected_Dataset.name
        self.config_dataset["dates"]=Selected_Dataset.dates
        self.config_dataset["pas_temporel"]=Selected_Dataset.pas_temporel
        return self.config_dataset
    
    def Formatter_JSON_specif(self):
        self.config_specifique={}
        if Parametres_choix_reseau_neurones.modele=="MLP":
            self.config_specifique["Parametres_archi_reseau"]=Parametres_archi_reseau_MLP.__dict__
        elif Parametres_choix_reseau_neurones.modele=="LSTM":
            self.config_specifique["Parametres_archi_reseau"]=Parametres_archi_reseau_LSTM.__dict__
        elif Parametres_choix_reseau_neurones.modele=="CNN":
            self.config_specifique["Parametres_archi_reseau"]=Parametres_archi_reseau_CNN.__dict__
        return self.config_specifique

    def obtenir_datasets(self):
        url = f"{URL}/datasets/info_all"  # IP SERVEUR_IA
        

        payload = {"message": "choix dataset"}

        try:
            r = requests.post(url, json=payload, timeout=100000)
            r.raise_for_status()
            data = r.json()
            return data
        
        except Exception as e:
            print("Erreur UI → IA :", e)
            return None

    def EnvoyerConfig(self):
        if self.Cadre_results_Entrainement.is_training==False:
            self.stop_training = False
            """Envoie la configuration au serveur et affiche l'entraînement en temps réel"""
            
            # Démarrer l'affichage de l'entraînement
            self.Cadre_results_Entrainement.start_training()
            
            # Préparer les payloads
            payload_global = self.Formatter_JSON_global()
            payload_model = self.Formatter_JSON_specif()
            payload_dataset = self.Formatter_JSON_dataset()

            # Avant d'envoyer le payload
            print("Payload envoyé au serveur :", {"payload": payload_global, "payload_model": payload_model, "payload_dataset": payload_dataset})
            
            
            ##### ENVOYER PAYLOAD DATASET SEULEMENT #####
            def run_fetch_dataset():
                """Fonction pour récupérer le dataset dans un thread séparé"""
                try:
                    r = requests.post(
                        f"{URL}/datasets/fetch_dataset",
                        json=payload_dataset,     
                        timeout=100
                    )
                    r.raise_for_status()
                    data = r.json()
                    print(f"\n")
                    print("Dataset récupéré avec succès.")
                    print("Réponse fetch_dataset :", data)

                    # Ici tu mets ce que tu veux faire avec le dataset :
                    # par ex. mettre à jour un cadre UI :
                    # self.Cadre_results_Dataset.afficher_infos(data)

                except requests.exceptions.RequestException as e:
                    print(f"Erreur de connexion lors de fetch_dataset: {e}")
                    messagebox.showerror(
                        "Erreur de connexion",
                        f"Impossible de se connecter au serveur (fetch_dataset):\n{str(e)}"
                    )


            def run_training():
                """Fonction pour exécuter l'entraînement dans un thread séparé"""
                run_fetch_dataset()
                y=[]
                yhat=[]
                try:
                    with requests.post(
                        f"{URL}/train_full", 
                        json={"payload": payload_global, "payload_model": payload_model}, 
                        stream=True,
                        timeout=None
                    ) as r:
                        r.raise_for_status()
                        print("Content-Type:", r.headers.get("content-type"))
                        
                        for line in r.iter_lines():
                            if self.stop_training:
                                requests.post(f"{URL}/stop_training")
                                break

                            if not line:
                                continue
                            
                            if line.startswith(b"data: "):
                                try:
                                    msg = json.loads(line[6:].decode("utf-8"))
                                    print("EVENT:", msg)
                                    
                                    # Traiter les différents types de messages
                                    if msg.get("type") == "epoch":
                                        # Message d'epoch avec loss
                                        epoch = msg.get("epoch")
                                        avg_loss = msg.get("avg_loss")
                                        epoch_s = msg.get("epoch_s")
                                        
                                        if epoch is not None and avg_loss is not None:
                                            # Ajouter le point au graphique
                                            self.Cadre_results_Entrainement.add_data_point(epoch, avg_loss,epoch_s)
                                    
                                    elif "epochs" in msg and "avg_loss" in msg:
                                        # Format alternatif (comme dans votre exemple)
                                        epoch = msg.get("epochs")
                                        avg_loss = msg.get("avg_loss")
                                        epoch_s = msg.get("epoch_s")
                                        
                                        if epoch is not None and avg_loss is not None:
                                            self.Cadre_results_Entrainement.add_data_point(epoch, avg_loss,epoch_s)
                                    
                                    elif msg.get("type") == "test_pair":
                                        y.append(msg.get("y"))
                                        yhat.append(msg.get("yhat"))
                                        
                                    elif msg.get("type") == "test_final":
                                        self.Cadre_results_Metrics.afficher_Metrics(msg.get("metrics"))
                                        
                                    elif msg.get("type") == "error":
                                        # Afficher les erreurs
                                        print(f"ERREUR: {msg.get('message')}")
                                        messagebox.showerror("Erreur", msg.get('message', 'Erreur inconnue'))
                                        break
                                    
                                    elif msg.get("type")=="fin_test":
                                        # Entraînement terminé
                                        self.Cadre_results_Testing.plot_predictions(y,yhat)
                                        break
                                
                                except json.JSONDecodeError as e:
                                    print(f"Erreur de décodage JSON: {e}")
                                    continue
                
                except requests.exceptions.RequestException as e:
                    print(f"Erreur de connexion: {e}")
                    messagebox.showerror("Erreur de connexion", f"Impossible de se connecter au serveur:\n{str(e)}")
                
                finally:
                    # Arrêter l'affichage de l'entraînement
                    self.Cadre_results_Entrainement.stop_training()
            
            

            # Lancer l'entraînement dans un thread séparé pour ne pas bloquer l'interface
            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()

class Cadre_Entrainement(ctk.CTkFrame):
    def __init__(self, app, master=None):
        super().__init__(master)
        self.configure(fg_color=master.cget("fg_color"))

        self.fg_color = master.winfo_rgb(master.cget("fg_color"))
        self.fg_color = '#%02x%02x%02x' % (self.fg_color[0]//256, self.fg_color[1]//256, self.fg_color[2]//256)

        # self.cadres_bg = app.cadres_bg
        # self.configure(fg_color=self.cadres_bg)
        
        # Variables pour stocker les données
        self.epochs = []
        self.losses = []
        self.data_queue = queue.Queue()
        self.is_training = False
        self.is_log=ctk.BooleanVar(value=False)
        
        # Titre
        self.titre = ctk.CTkLabel(
            self, 
            text="📊 Suivi de l'Entraînement en Temps Réel", 
            font=("Helvetica", 16, "bold"),
        )
        self.titre.pack(pady=(0, 10))

        self.progress_bar = ttk.Progressbar(self, length=800, mode='determinate')        
        
        # Frame pour les informations
        # self.info_frame = ctk.CTkFrame(self)
        self.info_frame = ctk.CTkFrame(self, fg_color=self.cget("fg_color"))
        self.info_frame.pack(fill="x", pady=(0, 10))
        
        
        # Labels d'information
        self.label_epoch = ctk.CTkLabel(
            self.info_frame,
            text="Epoch: -",
            font=("Helvetica", 12, "bold"),
        )
        self.label_epoch.pack(side="left", padx=10)

        self.label_epoch_s = ctk.CTkLabel(
            self.info_frame,
            text="Epochs/seconde: -",
            font=("Helvetica", 12, "bold"),
        )
        self.label_epoch_s.pack(side="left", padx=10)
        
        self.label_loss = ctk.CTkLabel(
            self.info_frame,
            text="Loss: -",
            font=("Helvetica", 12, "bold"),
        )
        self.label_loss.pack(side="left", padx=10)
        
        self.label_status = ctk.CTkLabel(
            self.info_frame,
            text="⏸️ En attente",
            font=("Helvetica", 12),
        )
        self.label_status.pack(side="right", padx=10)
        
        # Création du graphique matplotlib avec style moderne
        self.fig = Figure(figsize=(10, 6),facecolor=self.fg_color) #,dpi=100
        self.ax = self.fig.add_subplot(111)
        
        # Style du graphique
        self.ax.tick_params(axis='x', colors='#DCE4EE')
        self.ax.tick_params(axis='y', colors='#DCE4EE')
        for spine in self.ax.spines.values():
            spine.set_color('#DCE4EE')

        self.ax.set_facecolor(self.fg_color)
        self.ax.grid(True, linestyle='--', alpha=0.3,color='#DCE4EE') #, color='#95a5a6'
        self.ax.set_xlabel('Epoch', fontsize=16, fontweight='bold',color='#DCE4EE') #, color='#2c3e50'
        self.ax.set_ylabel('Loss', fontsize=16, fontweight='bold',color='#DCE4EE') #, color='#2c3e50'
        self.ax.set_title('Évolution de la Loss', fontsize=16, fontweight='bold', pad=20,color='#DCE4EE') #, color='#2c3e50'
        
        # Ligne de tracé (sera mise à jour)
        self.line, = self.ax.plot([], [],'o-', linewidth=2.5,
                                   color="#e74c3c", markerfacecolor="#9c9c9c")
        
        # Canvas pour afficher le graphique
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        


        ctk.CTkCheckBox(self, text="📈 Échelle Logarithmique", variable=self.is_log,
                    font=("Helvetica", 14, "bold"),
                    command=self.Log_scale).pack(side="left",pady=(10,0))
        
        # Ajustement automatique des marges
        self.fig.tight_layout()
    
    def Log_scale(self):
        if hasattr(self, 'Log_scale_possible'):
            self.ax.set_yscale('log' if self.is_log.get() else 'linear')
            if len(self.losses) > 1:
                y_min, y_max = min(self.losses), max(self.losses)
                if self.is_log.get():
                    # Marge en échelle log (multiplicative)
                    ratio = (y_max / y_min) ** 0.1
                    self.ax.set_ylim(y_min / ratio, y_max * ratio)
                else:
                    # Marge en échelle linéaire (additive)
                    y_range = y_max - y_min
                    if y_range > 0:
                        self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            # if self.is_log.get():
            #     self.ax.grid(True, which='minor', linestyle='--',alpha=0.3, color='#95a5a6')
            # self.ax.relim()
            # self.ax.autoscale_view(True, True, True)
            self.canvas.draw()
      
    def start_training(self):
        """Initialise l'affichage pour un nouvel entraînement"""
        self.is_training = True
        self.epochs = []
        self.losses = []
        self.total_epochs = Parametres_entrainement.nb_epochs
        
        self.progress_bar['value']=0
        self.progress_bar.pack(before=self.info_frame,pady=15)

        # Vider la file d'attente
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        # Réinitialiser le graphique
        self.ax.clear()
        # self.ax.set_facecolor(self.cadres_bg)
        # Style du graphique
        # self.ax.tick_params(axis='x', colors='#DCE4EE')
        # self.ax.tick_params(axis='y', colors='#DCE4EE')
        # for spine in self.ax.spines.values():
        #     spine.set_color('#DCE4EE')

        self.ax.set_facecolor(self.fg_color)
        self.ax.grid(True, linestyle='--', alpha=0.3,color='#DCE4EE') #, color='#95a5a6'
        self.ax.set_xlabel('Epoch', fontsize=16, fontweight='bold',color='#DCE4EE') #, color='#2c3e50'
        self.ax.set_ylabel('Loss', fontsize=16, fontweight='bold',color='#DCE4EE') #, color='#2c3e50'
        self.ax.set_title('Évolution de la Loss', fontsize=16, fontweight='bold', pad=20,color='#DCE4EE') #, color='#2c3e50'
        
        self.line, = self.ax.plot([], [],'o-', linewidth=2.5,
                                   color="#e74c3c", markerfacecolor="#9c9c9c",)
        
        self.label_status.configure(text="🚀 En cours...",text_color="#27ae60")
        self.canvas.draw()
        


        # Démarrer la mise à jour périodique
        self.update_plot()
    
    def add_data_point(self, epoch, loss,epoch_s):
        """Ajoute un nouveau point de données"""
        self.data_queue.put((epoch, loss,epoch_s))
    
    def update_plot(self):
        self.Log_scale_possible=True
        """Met à jour le graphique avec les nouvelles données"""
        if not self.is_training:
            return
        
        # Récupérer toutes les données disponibles dans la queue
        updated = False
        while not self.data_queue.empty():
            try:
                epoch, loss, epoch_s = self.data_queue.get_nowait()
                self.epochs.append(epoch)
                self.losses.append(loss)
                updated = True
                
                # Mettre à jour les labels
                self.label_epoch.configure(text=f"Epoch: {epoch}")
                self.label_epoch_s.configure(text=f"Epochs/seconde: {epoch_s:.2f}")
                self.label_loss.configure(text=f"Loss: {loss:.6f}")
                # Mettre à jour la barre de progression
                self.progress_bar['value'] = (epoch / self.total_epochs) * 100
            except queue.Empty:
                break
        
        # Mettre à jour le graphique si de nouvelles données sont disponibles
        if updated and len(self.epochs) > 0:
            self.line.set_data(self.epochs, self.losses)

            self.ax.set_yscale('log' if self.is_log.get() else 'linear')

            
            # Ajuster les limites des axes
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            
            if len(self.losses) > 1:
                y_min, y_max = min(self.losses), max(self.losses)
                if self.is_log.get():
                    # Marge en échelle log (multiplicative)
                    ratio = (y_max / y_min) ** 0.1
                    self.ax.set_ylim(y_min / ratio, y_max * ratio)
                else:
                    # Marge en échelle linéaire (additive)
                    y_range = y_max - y_min
                    if y_range > 0:
                        self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            self.canvas.draw()
        
        # Continuer la mise à jour si l'entraînement est en cours
        if self.is_training:
            self.after(100, self.update_plot)  # Mise à jour toutes les 100ms
    
    def stop_training(self):
        """Arrête l'entraînement et met à jour le statut"""
        # Traiter toutes les données restantes dans la queue avant d'arrêter
        while not self.data_queue.empty():
            try:
                epoch, loss = self.data_queue.get_nowait()
                self.epochs.append(epoch)
                self.losses.append(loss)
                
                # Mettre à jour les labels
                self.label_epoch.configure(text=f"Epoch: {epoch}")
                self.label_loss.configure(text=f"Loss: {loss:.6f}")
                # Mettre à jour la barre de progression
                self.progress_bar['value'] = (epoch / self.total_epochs) * 100
            except queue.Empty:
                break
        
        # Mettre à jour le graphique une dernière fois avec toutes les données
        if len(self.epochs) > 0:
            self.line.set_data(self.epochs, self.losses)
            self.ax.set_yscale('log' if self.is_log.get() else 'linear')
            
            # Ajuster les limites des axes
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            
            if len(self.losses) > 1:
                y_min, y_max = min(self.losses), max(self.losses)
                if self.is_log.get():
                    # Marge en échelle log (multiplicative)
                    ratio = (y_max / y_min) ** 0.1
                    self.ax.set_ylim(y_min / ratio, y_max * ratio)
                else:
                    # Marge en échelle linéaire (additive)
                    y_range = y_max - y_min
                    if y_range > 0:
                        self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            self.canvas.draw()
        
        self.is_training = False
        self.progress_bar.pack_forget()
        self.label_status.configure(text="✅ Terminé", text_color="#27ae60")
        
        # Afficher les statistiques finales
        if len(self.losses) > 0:
            final_loss = self.losses[-1]
            min_loss = min(self.losses)
            self.label_loss.configure(text=f"Loss finale: {final_loss:.6f} (min: {min_loss:.6f})")

class Cadre_Testing(ctk.CTkFrame):
    def __init__(self, app, master=None):
        super().__init__(master)
        self.configure(fg_color=master.cget("fg_color"))
        self.fg_color = master.winfo_rgb(master.cget("fg_color"))
        self.fg_color = '#%02x%02x%02x' % (self.fg_color[0]//256, self.fg_color[1]//256, self.fg_color[2]//256)
        

        # Titre
        self.titre = ctk.CTkLabel(
            self, 
            text="📊 Suivi de la phase de test", 
            font=("Helvetica", 16, "bold"),
        )
        self.titre.pack(pady=(0, 10))

    def save_figure(self,fig):
        file_path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("Tous les fichiers", "*.*")],
            title="Enregistrer la figure"
        )
        if file_path:
            fig.savefig(file_path)

    def plot_predictions(self, y_true_pairs, y_pred_pairs):
        for widget in self.winfo_children():
            widget.destroy()
        """
        y_true_pairs, y_pred_pairs : listes de listes (N x D)
        Affiche y vs yhat pour D dimensions (ou une seule si D=1) avec un style élégant.
        """
        if not y_true_pairs or not y_pred_pairs:
            return

        yt = np.array(y_true_pairs, dtype=float)  # (N, D)
        yp = np.array(y_pred_pairs, dtype=float)  # (N, D)

        # Squeeze si D=1
        if yt.ndim == 2 and yt.shape[1] == 1:
            yt = yt.squeeze(1)
            yp = yp.squeeze(1)

        # Création du graphique matplotlib avec style moderne
        fig = Figure(facecolor=self.fg_color)
        ax = fig.add_subplot(111)
        
        
        # Style du graphique
        ax.set_facecolor(self.fg_color)
        ax.grid(True, linestyle='--', alpha=0.3, color='#DCE4EE')
        
        if yt.ndim == 1:
            x = np.arange(len(yt))
            
            # Tracer avec un style élégant
            ax.plot(x, yt, 
                    color='#2E86AB', 
                    linewidth=2, 
                    marker='o', 
                    markersize=4, 
                    markerfacecolor='white',
                    markeredgewidth=1.5,
                    markeredgecolor='#2E86AB',
                    label='y (vraies valeurs)', 
                    alpha=0.8,
                    zorder=2)
            
            ax.plot(x, yp, 
                    color="#e74c3c", 
                    linewidth=2, 
                    marker='s', 
                    markersize=4, 
                    markerfacecolor='white',
                    markeredgewidth=1.5,
                    markeredgecolor='#A23B72',
                    label='ŷ (prédictions)', 
                    alpha=0.8,
                    zorder=2)
            
            # Remplissage entre les courbes pour montrer l'erreur
            ax.fill_between(x, yt, yp, alpha=0.2, color='gray', label='Erreur')
            
        else:
            x = np.arange(yt.shape[0])
            colors_true = ['#2E86AB', '#06A77D', '#F77F00', '#D62828']
            colors_pred = ['#A23B72', '#F18F01', '#C73E1D', '#6A4C93']
            
            for d in range(min(yt.shape[1], 4)):  # Limiter à 4 dimensions pour la lisibilité
                ax.plot(x, yt[:, d], 
                        color=colors_true[d % len(colors_true)],
                        linewidth=2,
                        marker='o',
                        markersize=3,
                        markerfacecolor='white',
                        markeredgewidth=1,
                        markeredgecolor=colors_true[d % len(colors_true)],
                        label=f'y (vrai) dim {d}',
                        alpha=0.8,
                        zorder=2)
                
                ax.plot(x, yp[:, d],
                        color=colors_pred[d % len(colors_pred)],
                        linewidth=2,
                        marker='s',
                        markersize=3,
                        markerfacecolor='white',
                        markeredgewidth=1,
                        markeredgecolor=colors_pred[d % len(colors_pred)],
                        label=f'ŷ (prédit) dim {d}',
                        alpha=0.8,
                        linestyle='--',
                        zorder=2)

        # Titre et labels
        ax.set_title('Comparaison des prédictions avec les valeurs réelles', 
                    fontsize=14, 
                    fontweight='bold',
                    pad=20)
        ax.set_xlabel('Index de l\'échantillon', fontsize=11, fontweight='bold')
        ax.set_ylabel('Valeur', fontsize=11, fontweight='bold')
        
        # Légende élégante
        legend = ax.legend(loc='best', 
                        frameon=True, 
                        fancybox=True, 
                        shadow=True,
                        fontsize=10)
        legend.get_frame().set_alpha(0.9)
        
        # Grille plus subtile
        ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
        ax.set_axisbelow(True)

        
        ax.set_facecolor(self.fg_color)
        ax.grid(True, linestyle='--', alpha=0.3, color='#95a5a6')
        
        
        # Ajuster les marges
        fig.tight_layout()
        fig.patch.set_facecolor(self.fg_color)
        


        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True,padx=(0,10))


       
        # Bouton de sauvegarde stylisé
        bouton_sauvegarde = tk.Button(
            self,
            text="💾 Enregistrer la figure",
            font=("Helvetica", 11, "bold"),
            bg="#2E86AB",           # Bleu profond pour contraster
            fg="white",             # Texte blanc lisible
            activebackground="#1B4F72",  # Survol plus foncé
            activeforeground="white",
            relief="flat",
            bd=0,
            padx=12,
            pady=6,
            command=lambda: self.save_figure(fig)
        )
        bouton_sauvegarde.pack(pady=(10, 5))



        
        # Afficher
        #plt.show()

class Cadre_Metrics(ctk.CTkFrame):
    def __init__(self, app, master=None):
        super().__init__(master)
        self.configure(fg_color=master.cget("fg_color"))
        self.fg_color = master.winfo_rgb(master.cget("fg_color"))
        self.fg_color = '#%02x%02x%02x' % (self.fg_color[0]//256, self.fg_color[1]//256, self.fg_color[2]//256)
        
        # Titre
        self.titre = tk.Label(
            self, 
            text="📊 Affichage des metrics", 
            font=("Helvetica", 16, "bold"),
        )
        self.titre.pack(pady=(0, 10))

    def afficher_Metrics(self,metrics):
        for widget in self.winfo_children():
            widget.destroy()
        for i, (metric, val) in enumerate(metrics["overall_mean"].items()):
            label = tk.Label(self, text=f"{metric}: {val:.8f}", font=("Helvetica", 16, "bold"), bg=self.fg_color)
            label.pack(anchor="w", padx=15, pady=5)

class Cadre_Prediction(ctk.CTkFrame):
    def __init__(self, app, master=None):
        super().__init__(master)
        self.configure(fg_color=master.cget("fg_color"))
        self.fg_color = master.winfo_rgb(master.cget("fg_color"))
        self.fg_color = '#%02x%02x%02x' % (self.fg_color[0]//256, self.fg_color[1]//256, self.fg_color[2]//256)
        
        # Titre
        self.titre = tk.Label(
            self, 
            text="📊 Affichage de la prédiction", 
            font=("Helvetica", 16, "bold"),
        )
        self.titre.pack(pady=(0, 10))


# Créer la fenêtre de paramétrage du modèle
class Fenetre_Params(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        # self.focus_force()       # force le focus sur la nouvelle fenêtre
        # self.grab_set()          # empêche de cliquer sur la fenêtre principale tant que le Toplevel est ouvert
        self.after(100, lambda: self.focus_force())
        self.title("⚙️ Paramètres du Modèle")
        self.geometry("700x600")
        # Polices
        self.font_titre = ("Helvetica", 18, "bold")
        self.font_section = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 12)

        
        # Frame principal avec scrollbar
        self.params_frame = ctk.CTkScrollableFrame(self)
        self.params_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.params_frame.grid_columnconfigure(0, weight=1)  # première colonne s'étire
        self.params_frame.grid_columnconfigure(1, weight=1)  # deuxième colonne s'étire aussi

        
        # # Titre simulé
        # tk.Label(self.cadre, text="Paramètres", font=self.font_titre, bg=self.fenetre_bg).pack(anchor="w", pady=(0, 10))
        
        # Polices
        self.font_titre = ("Roboto Medium", 16)
        self.font_label = ("Roboto", 12)

        # Titre
        ctk.CTkLabel(
            self.params_frame,
            text="⚙️ Configuration",
            font=("Roboto Medium", 22)
        ).grid(row=0, column=0, columnspan=2,padx=10,pady=(0,20))

        ctk.CTkLabel(
            self.params_frame,
            text="📊 Paramètres d'entraînement",
            font=("Roboto Medium", 14)
        ).grid(row=1, column=0, columnspan=2,padx=10,pady=(0,20))
        
        ctk.CTkLabel(
            self.params_frame,
            text="Nombre d'époques:",
            font=self.font_label
        ).grid(row=2, column=0, sticky="w",padx=10,pady=(0,20))
        
        self.epochs_var = ctk.StringVar(value=str(Parametres_entrainement.nb_epochs))
        ctk.CTkEntry(
            self.params_frame,
            textvariable=self.epochs_var,
            width=150
        ).grid(row=2, column=1, sticky="e",padx=10,pady=(0,20))

        ctk.CTkLabel(
            self.params_frame,
            text="Batch Size:",
            font=self.font_label
        ).grid(row=3, column=0, sticky="w",padx=10,pady=(0,20))
        
        self.batch_var = ctk.StringVar(value=str(Parametres_entrainement.batch_size))
        ctk.CTkEntry(
            self.params_frame,
            textvariable=self.batch_var,
            width=150
        ).grid(row=3, column=1, sticky="e",padx=10,pady=(0,20))

        ctk.CTkLabel(
            self.params_frame,
            text="⚙️ Configuration de l'entraînement",
            font=("Roboto Medium", 14)
        ).grid(row=4, column=0, columnspan=2,padx=10,pady=(0,20))
        
        ctk.CTkLabel(
            self.params_frame,
            text="Fonction de Perte:",
            font=self.font_label
        ).grid(row=5, column=0, sticky="w",padx=10,pady=(0,20))
        
        self.loss_var = ctk.StringVar(value=Parametres_choix_loss_fct.fonction_perte)
        ctk.CTkOptionMenu(
            self.params_frame,
            values=["MSE", "MAE", "Huber"],
            variable=self.loss_var,
            width=150
        ).grid(row=5, column=1, sticky="e",padx=10,pady=(0,20))
        
        ctk.CTkLabel(
            self.params_frame,
            text="Optimiseur:",
            font=self.font_label
        ).grid(row=6, column=0, sticky="w",padx=10,pady=(0,20))
        
        self.optim_var = ctk.StringVar(value=Parametres_optimisateur.optimisateur)
        ctk.CTkOptionMenu(
            self.params_frame,
            values=["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta"],
            variable=self.optim_var,
            width=150
        ).grid(row=6, column=1, sticky="e",padx=10,pady=(0,20))
        
        ctk.CTkLabel(
            self.params_frame,
            text="Learning Rate:",
            font=self.font_label
        ).grid(row=7, column=0, sticky="w",padx=10,pady=(0,20))
        
        self.lr_var = ctk.StringVar(value=str(Parametres_optimisateur.learning_rate))
        ctk.CTkEntry(
            self.params_frame,
            textvariable=self.lr_var,
            width=150
        ).grid(row=7, column=1, sticky="e",padx=10,pady=(0,20))

        ### Paramètres du modèle choisi ###

        self.params_model_frame = ctk.CTkFrame(self.params_frame, corner_radius=10)

        self.model_var = ctk.StringVar(value=Parametres_choix_reseau_neurones.modele)
        ctk.CTkSegmentedButton(
            self.params_frame,
            values=["MLP", "CNN", "LSTM"],
            variable=self.model_var,
            command=self.on_model_change
        ).grid(row=8, column=0, columnspan=2,padx=10,pady=(0,20),sticky="n")

        self.params_model_frame.grid(row=9, column=0, columnspan=2, padx=10, pady=(0,20), sticky="ew")
        self.params_model_frame.grid_columnconfigure(0, weight=1)  # première colonne s'étire
        self.params_model_frame.grid_columnconfigure(1, weight=1)  # deuxième colonne s'étire aussi
        
        ### Fin des paramètres du modèle choisi ###

        last_row=self.params_frame.grid_size()[1]
        ctk.CTkButton(
            self.params_frame,
            text="💾 Sauvegarder",
            font=("Roboto", 13),
            height=40,
            command=self.save_params
        ).grid(row=last_row, column=0,padx=10,pady=(50,20),sticky="ew")

        ctk.CTkButton(
            self.params_frame,
            text="❌ Annuler",
            font=("Roboto", 13),
            height=40,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
            command=self.destroy
        ).grid(row=last_row, column=1,padx=10,pady=(50,20),sticky="ew")
        self.on_model_change(self.model_var.get())

        # Afficher les paramètres du modèle sélectionné
        # self.on_model_change(self.model_var.get())

        # self.update_idletasks()
        # self.geometry(f"500x{self.winfo_reqheight()}")
    
    def est_ouverte(self):
        return self.winfo_exists()

    def save_params(self):
        Parametres_entrainement.nb_epochs = int(self.epochs_var.get())
        Parametres_entrainement.batch_size = int(self.batch_var.get())
        Parametres_choix_loss_fct.fonction_perte = self.loss_var.get()
        Parametres_optimisateur.optimisateur = self.optim_var.get()
        Parametres_optimisateur.learning_rate = float(self.lr_var.get())

        Parametres_choix_reseau_neurones.modele = self.model_var.get()
        if self.model_var.get() == "MLP":
            Parametres_archi_reseau_MLP.nb_couches = int(self.mlp_layers.get())
            Parametres_archi_reseau_MLP.hidden_size = int(self.mlp_hidden.get())
            Parametres_archi_reseau_MLP.dropout_rate = float(self.mlp_dropout.get())
            Parametres_archi_reseau_MLP.fonction_activation = self.mlp_activation.get()
        elif self.model_var.get() == "CNN":
            Parametres_archi_reseau_CNN.nb_couches = int(self.cnn_layers.get())
            Parametres_archi_reseau_CNN.hidden_size = int(self.cnn_hidden.get())
            Parametres_archi_reseau_CNN.kernel_size = int(self.cnn_kernel.get())
            Parametres_archi_reseau_CNN.stride = int(self.cnn_stride.get())
            Parametres_archi_reseau_CNN.padding = int(self.cnn_padding.get())
            Parametres_archi_reseau_CNN.fonction_activation = self.cnn_activation.get()
        elif self.model_var.get() == "LSTM":
            Parametres_archi_reseau_LSTM.nb_couches = int(self.lstm_layers.get())
            Parametres_archi_reseau_LSTM.hidden_size = int(self.lstm_hidden.get())
            Parametres_archi_reseau_LSTM.batch_first = self.lstm_batch_first.get()
            Parametres_archi_reseau_LSTM.bidirectional = self.lstm_bidirectional.get()
        self.destroy()
    
    def on_model_change(self, model_type):
        """Change les paramètres affichés selon le modèle"""
        # Effacer les widgets existants
        for widget in self.params_model_frame.winfo_children():
            widget.destroy()

        if model_type == "MLP":
            self.create_mlp_params()
        elif model_type == "CNN":
            self.create_cnn_params()
        elif model_type == "LSTM":
            self.create_lstm_params()

    def create_mlp_params(self):
        """Crée les paramètres spécifiques au MLP"""
        # ctk.CTkLabel(
        #     self.params_model_frame,
        #     text="🧠 Paramètres MLP",
        #     font=("Roboto Medium", 14)
        # ).grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # Nombre de couches
        ctk.CTkLabel(self.params_model_frame, text="Nombre de couches:", font=("Roboto", 12)).grid(row=0, column=0, sticky="w",padx=10,pady=(0,20))
        self.mlp_layers = ctk.StringVar(value=str(Parametres_archi_reseau_MLP.nb_couches))
        ctk.CTkEntry(self.params_model_frame, textvariable=self.mlp_layers, width=150).grid(row=0, column=1, sticky="e",padx=10,pady=(0,20))

        # Hidden size
        ctk.CTkLabel(self.params_model_frame, text="Hidden Size:", font=("Roboto", 12)).grid(row=1, column=0, sticky="w",padx=10,pady=(0,20))
        self.mlp_hidden = ctk.StringVar(value=str(Parametres_archi_reseau_MLP.hidden_size))
        ctk.CTkEntry(self.params_model_frame, textvariable=self.mlp_hidden, width=150).grid(row=1, column=1, sticky="e",padx=10,pady=(0,20))

        # Dropout
        ctk.CTkLabel(self.params_model_frame, text="Dropout Rate:", font=("Roboto", 12)).grid(row=2, column=0, sticky="w",padx=10,pady=(0,20))
        self.mlp_dropout = ctk.StringVar(value=str(Parametres_archi_reseau_MLP.dropout_rate))
        ctk.CTkEntry(self.params_model_frame, textvariable=self.mlp_dropout, width=150).grid(row=2, column=1, sticky="e",padx=10,pady=(0,20))

        # Activation
        ctk.CTkLabel(self.params_model_frame, text="Activation:", font=("Roboto", 12)).grid(row=3, column=0, sticky="w",padx=10,pady=(0,20))
        self.mlp_activation = ctk.StringVar(value=Parametres_archi_reseau_MLP.fonction_activation)
        ctk.CTkOptionMenu(
            self.params_model_frame,
            values=["ReLU", "GELU", "tanh", "sigmoid", "leaky_relu"],
            variable=self.mlp_activation,
            width=150
        ).grid(row=3, column=1, sticky="e",padx=10,pady=(0,20))

    def create_cnn_params(self):
        """Crée les paramètres spécifiques au CNN"""
        ctk.CTkLabel(
            self.params_model_frame,
            text="🔲 Paramètres CNN",
            font=("Roboto Medium", 14)
        ).grid(row=0, column=0, columnspan=2,padx=10, pady=(0, 20))

        params = [
            ("Nombre de couches:", Parametres_archi_reseau_CNN.nb_couches, "cnn_layers"),
            ("Hidden Size:", Parametres_archi_reseau_CNN.hidden_size, "cnn_hidden"),
            ("Kernel Size:", Parametres_archi_reseau_CNN.kernel_size, "cnn_kernel"),
            ("Stride:", Parametres_archi_reseau_CNN.stride, "cnn_stride"),
            ("Padding:", Parametres_archi_reseau_CNN.padding, "cnn_padding"),
        ]

        for index,(label, default, attr) in enumerate(params):
            ctk.CTkLabel(self.params_model_frame, text=label, font=("Roboto", 12)).grid(row=index+1, column=0, sticky="w",padx=10,pady=(5, 15))#,padx=10, pady=(0, 20)
            var = ctk.StringVar(value=str(default))
            setattr(self, attr, var)
            ctk.CTkEntry(self.params_model_frame, textvariable=var, width=150).grid(row=index+1, column=1, sticky="e",padx=10,pady=(5, 15))

        # Activation
        ctk.CTkLabel(self.params_model_frame, text="Activation:", font=("Roboto", 12)).pack(side="left")
        self.cnn_activation = ctk.StringVar(value=Parametres_archi_reseau_CNN.fonction_activation)
        ctk.CTkOptionMenu(
            self.params_model_frame,
            values=["ReLU", "GELU", "tanh", "sigmoid"],
            variable=self.cnn_activation,
            width=150
        ).grid(row=len(params)+1, column=1, sticky="e",padx=10,pady=(0,20))

    def create_lstm_params(self):
        """Crée les paramètres spécifiques au LSTM"""
        ctk.CTkLabel(
            self.params_model_frame,
            text="🔄 Paramètres LSTM",
            font=("Roboto Medium", 14)
        ).grid(row=0, column=0, columnspan=2,padx=10, pady=(0, 20))

        # Nombre de couches
        ctk.CTkLabel(self.params_model_frame, text="Nombre de couches:", font=("Roboto", 12)).grid(row=1, column=0, sticky="w",padx=10,pady=(0,20))
        self.lstm_layers = ctk.StringVar(value=str(Parametres_archi_reseau_LSTM.nb_couches))
        ctk.CTkEntry(self.params_model_frame, textvariable=self.lstm_layers, width=150).grid(row=1, column=1, sticky="e",padx=10,pady=(0,20))

        # Hidden size
        ctk.CTkLabel(self.params_model_frame, text="Hidden Size:", font=("Roboto", 12)).grid(row=2, column=0, sticky="w",padx=10,pady=(0,20))
        self.lstm_hidden = ctk.StringVar(value=str(Parametres_archi_reseau_LSTM.hidden_size))
        ctk.CTkEntry(self.params_model_frame, textvariable=self.lstm_hidden, width=150).grid(row=2, column=1, sticky="e",padx=10,pady=(0,20))

        # Bidirectional
        self.lstm_bidirectional = ctk.BooleanVar(value=Parametres_archi_reseau_LSTM.bidirectional)
        ctk.CTkCheckBox(
            self.params_model_frame,
            text="Bidirectionnel",
            variable=self.lstm_bidirectional,
            font=("Roboto", 12)
        ).grid(row=3, column=0, columnspan=2, sticky="n",padx=10,pady=(0,20))

        # Batch first
        self.lstm_batch_first = ctk.BooleanVar(value=Parametres_archi_reseau_LSTM.batch_first)
        ctk.CTkCheckBox(
            self.params_model_frame,
            text="Batch First",
            variable=self.lstm_batch_first,
            font=("Roboto", 12)
        ).grid(row=4, column=0, columnspan=2, sticky="n",padx=10,pady=(0,20))
    
    # Fonctions des fenêtres de paramétrage

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

# Créer la fenêtre de paramétrage de l'horizon des données
class Fenetre_Params_horizon(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.after(200, lambda: self.focus_force())
        self.title("🧠 Paramétrage temporels et de découpage des données")

        # Définir une police personnalisée
        self.font_titre = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 11)


        self.params_frame = ctk.CTkFrame(self)
        self.params_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        # self.params_frame.grid_columnconfigure(0, weight=1,uniform="col")  # première colonne s'étire
        # self.params_frame.grid_columnconfigure(1, weight=1,uniform="col")  # deuxième colonne s'étire aussi
        self.params_frame.grid_columnconfigure((0,1), weight=1,uniform="col")
        # self.params_frame.grid_rowconfigure((0,1,3,4,), weight=1,uniform="row")
        
        # Variables
        self.Params_temporels_horizon = ctk.IntVar(value=Parametres_temporels.horizon)
        self.date_debut_str = ctk.StringVar(value=Parametres_temporels.dates[0])
        self.date_fin_str = ctk.StringVar(value=Parametres_temporels.dates[1])
        self.Params_temporels_pas_temporel = ctk.IntVar(value=Parametres_temporels.pas_temporel)
        self.Params_temporels_portion_decoupage = ctk.IntVar(value=Parametres_temporels.portion_decoupage * 100)

        ctk.CTkLabel(self.params_frame, text="📅 Paramètres Temporels").grid(row=0, column=0, columnspan=2 , sticky="ew",padx=10,pady=20)

        # Liste des champs
        champs = [
            ("Horizon temporel (int) :", self.Params_temporels_horizon,"int"),
            (f"Pas temporel (multiple de {Selected_Dataset.pas_temporel}) :", self.Params_temporels_pas_temporel,"int"),
            ("Portion découpage (%) :", self.Params_temporels_portion_decoupage,"float"),
        ]

        for i, (label, var, type_) in enumerate(champs):
            ctk.CTkLabel(self.params_frame, text=label).grid(row=i+1, column=0, sticky="w", padx=10,pady=(0,20))
            ctk.CTkEntry(self.params_frame, textvariable=var, validate="key", validatecommand=(self.register(lambda P,t=type_: validate_fct(P, t)), "%P")).grid(row=i+1, column=1,padx=10,pady=(0,20),sticky="e")

        next_row=len(champs)+1
        # Dates
        ctk.CTkLabel(self.params_frame, text="Date de début :").grid(row=next_row, column=0, sticky="w",padx=10,pady=(0,20))
        ctk.CTkButton(self.params_frame, textvariable=self.date_debut_str, command=self.ouvrir_calendrier_debut).grid(row=next_row, column=1,padx=10,pady=(0,20),sticky="e")

        ctk.CTkLabel(self.params_frame, text="Date de fin :").grid(row=next_row+1, column=0, sticky="w",padx=10,pady=(0,20))
        ctk.CTkButton(self.params_frame, textvariable=self.date_fin_str, command=self.ouvrir_calendrier_fin).grid(row=next_row+1, column=1,padx=10,pady=(0,20),sticky="e")

        # Boutons d'action
        ctk.CTkButton(
            self.params_frame, 
            text="💾 Sauvegarder la configuration",
            font=("Roboto", 13),
            height=40,
            command=self.Save_quit
        ).grid(row=next_row+2, column=0,padx=10,pady=20,sticky="ew")

        ctk.CTkButton(
            self.params_frame, text="❌ Quitter",
            font=("Roboto", 13),
            height=40,
            command=self.destroy
        ).grid(row=next_row+2, column=1,padx=10,pady=20,sticky="ew")

        self.resizable(False, False)

    def est_ouverte(self):
        return self.winfo_exists()
    
    # Fonction locale : ouvrir calendrier debut
    def ouvrir_calendrier_debut(self):
        top = tk.Toplevel(self)
        top.title("Sélectionner la date de début")
        try:
            date_obj = datetime.strptime(self.date_debut_str.get(), "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.today()
        cal = Calendar(top, selectmode='day', date_pattern='yyyy-mm-dd',
                    year=date_obj.year, month=date_obj.month, day=date_obj.day)
        cal.pack(padx=10, pady=10)
        tk.Button(top, text="Valider", command=lambda: (self.date_debut_str.set(cal.get_date()), top.destroy())).pack(pady=10)

    # Fonction locale : ouvrir calendrier fin
    def ouvrir_calendrier_fin(self):
        top = tk.Toplevel(self)
        top.title("Sélectionner la date de fin")
        try:
            date_obj = datetime.strptime(self.date_fin_str.get(), "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.today()
        cal = Calendar(top, selectmode='day', date_pattern='yyyy-mm-dd',
                    year=date_obj.year, month=date_obj.month, day=date_obj.day)
        cal.pack(padx=10, pady=10)
        tk.Button(top, text="Valider", command=lambda: (self.date_fin_str.set(cal.get_date()), top.destroy())).pack(pady=10)

    def Save_quit(self):
        Parametres_temporels.horizon = self.Params_temporels_horizon.get()
        Parametres_temporels.pas_temporel = self.Params_temporels_pas_temporel.get()
        Parametres_temporels.portion_decoupage = self.Params_temporels_portion_decoupage.get() / 100
        Parametres_temporels.dates = [self.date_debut_str.get(), self.date_fin_str.get()]
        self.destroy()
            
    def Quit(self):
        self.Params_temporels_horizon.set(Parametres_temporels.horizon)
        self.date_debut_str.set(Parametres_temporels.dates[0])
        self.date_fin_str.set(value=Parametres_temporels.dates[1])
        self.Params_temporels_pas_temporel.set(Parametres_temporels.pas_temporel)
        self.Params_temporels_portion_decoupage.set(Parametres_temporels.portion_decoupage*100)
        self.destroy()

#Creer la fenêtre de paramètres des visualisations
class Fenetre_Choix_metriques(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        
        self.after(100, lambda: self.focus_force())
        self.title("⚙️ Paramètres des visualisations et des métriques")
        
        # Polices
        self.font_titre = ("Helvetica", 18, "bold")
        self.font_section = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 12)

        # self.geometry("500x1")  # largeur fixe, hauteur minimale

        # Frame principale
        self.params_frame = ctk.CTkFrame(self)
        self.params_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.params_frame.grid_columnconfigure(0, weight=1)  # première colonne s'étire
        self.params_frame.grid_columnconfigure(1, weight=1)  # deuxième colonne s'étire aussi

        
        # # Titre simulé
        # tk.Label(self.cadre, text="Paramètres", font=self.font_titre, bg=self.fenetre_bg).pack(anchor="w", pady=(0, 10))
        
        # Polices
        self.font_titre = ("Roboto Medium", 16)
        self.font_label = ("Roboto", 12)

        # Titre
        ctk.CTkLabel(
            self.params_frame,
            text="⚙️ Configuration des visualisations et du suivi",
            font=("Roboto Medium", 22)
        ).grid(row=0, column=0, columnspan=2,padx=10,pady=(0,20))
        # self.geometry("700x800")


        ctk.CTkLabel(self.params_frame, text="Choix des métriques (séparées par des virgules):", font=("Roboto", 12)).grid(row=1, column=0, sticky="w",padx=10,pady=(0,20))
        # self.lstm_hidden = ctk.StringVar(value=str(Parametres_visualisation_suivi.metriques))
        self.Params_visualisation_suivi_metriques = tk.StringVar(value=",".join(Parametres_visualisation_suivi.metriques))
        ctk.CTkEntry(self.params_frame, textvariable=self.Params_visualisation_suivi_metriques, width=150).grid(row=1, column=1, sticky="e",padx=10,pady=(0,20))





        last_row=self.params_frame.grid_size()[1]
        ctk.CTkButton(
            self.params_frame,
            text="💾 Sauvegarder",
            font=("Roboto", 13),
            height=40,
            command=self.save_params
        ).grid(row=last_row, column=0,padx=10,pady=(50,20),sticky="ew")

        ctk.CTkButton(
            self.params_frame,
            text="❌ Annuler",
            font=("Roboto", 13),
            height=40,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
            command=self.destroy
        ).grid(row=last_row, column=1,padx=10,pady=(50,20),sticky="ew")

        # Applique la largeur fixe et la hauteur calculée
        self.ajuster_hauteur_auto()


    def ajuster_hauteur_auto(self, largeur_fixe=700):
        self.geometry(f"{largeur_fixe}x1")
        self.update_idletasks()
        hauteur = self.winfo_height()
        self.geometry(f"{largeur_fixe}x{hauteur+40}")

    
    def est_ouverte(self):
        return self.winfo_exists()

    def save_params(self):
        Parametres_visualisation_suivi.metriques = [m.strip() for m in self.Params_visualisation_suivi_metriques.get().split(",") if m.strip()]
        self.destroy()

# Fonction utilitaires:
def validate_fct(text,type_):
    if type_ == "int":
        return text.isdigit() or text == ""
    elif type_ == "float":
        if text == "":
            return True
        try:
            float(text)
            return True
        except ValueError:
            return False

# Lancer la boucle principale
if __name__ == "__main__":
    app = Fenetre_Acceuil()
    app.mainloop()