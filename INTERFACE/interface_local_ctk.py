import tkinter as tk
from tkcalendar import Calendar
from datetime import datetime
from tkinter import filedialog, Label
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
matplotlib.use("TkAgg")
import customtkinter as ctk
import random
import json
import tkinter.font as tkfont
import pandas as pd
import os
from pydantic import BaseModel, ValidationError
from typing import List, Optional


class TimeSeriesData(BaseModel):
    timestamps: List[str]
    values: List[Optional[float]]   

class DatasetPacket(BaseModel):
    payload_name: str
    payload_dataset: TimeSeriesData

URL = "http://192.168.1.190:8000"


ctk.set_default_color_theme("INTERFACE/Themes/blue.json")
ctk.set_appearance_mode("dark")


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

class Fonts_class():
    def __init__(self):
        self.Titre_app = ("Roboto Medium", 30)
        self.sous_Titre_app = ("Roboto", 20)
        self.Section_params = ("Roboto Medium", 24,"bold")

        self.title_font = ("Roboto Medium", 28,"bold")
        self.section_font = ("Roboto Medium", 24,"bold")
        self.button_font = ("Roboto", 20)

        self.Metrics = ("Roboto Medium", 24,"bold")
        self.Tabs_title = ("Roboto", 20, "bold")

class Plot_style_class():
    def __init__(self):
        self.plot_axes = ("Roboto", 20, "bold")
        # self.plot_title = ("Roboto", 24, "bold")
        self.plot_title = {'fontname':'sans-serif','fontsize':24, 'fontweight':'bold','color':'#DCE4EE'}
        self.plot_legend = {'family':'sans-serif','size':20}

        self.text_color="#DCE4EE"
        self.primary_color="#DCE4EE"
        self.plot_background=gray_to_hex("gray17")
        self.train_line="#e74c3c"
        self.markerfacecolor="#9c9c9c"


        self.test_prediction_reel="#2E86AB"
        self.test_prediction_test="#A23B72"
        self.test_prediction_diff="#A8A8A8"

class Colors_IRMA_class():
    def __init__(self):
        # Couleurs style IRMA Conseil
        BG_DARK = "#2c3e50"
        BG_CARD = "#34495e"
        BG_INPUT = "#273747"
        TEXT_PRIMARY = "#ecf0f1"
        TEXT_SECONDARY = "#bdc3c7"
        ACCENT_PRIMARY = "#e74c3c"
        ACCENT_SECONDARY = "#3498db"
        BORDER_COLOR = "#4a5f7f"

Fonts=Fonts_class()

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


# Créer la fenêtre d'accueil
class Fenetre_Acceuil(ctk.CTk):
    def __init__(self):
        self.cadres_bg="#eaf2f8"
        self.cadres_fg="#e4eff8"
        self.fenetre_bg="#f0f4f8"

        self.JSON_Datasets={}

        self.stop_training = False  # drapeau d’annulation
        self.Payload={}
        self.Fenetre_Params_instance = None
        self.Fenetre_Params_horizon_instance = None
        self.Fenetre_Choix_datasets_instance = None
        self.Fenetre_Choix_metriques_instance = None
        self.Gestion_Datasets_instance = None
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
        self.Results_notebook.add("Représentation")
        self.Results_notebook.add("Training")
        self.Results_notebook.add("Testing")
        self.Results_notebook.add("Metrics")
        self.Results_notebook.add("Prediction")

        # Créer les cadres dans les onglets

        self.Cadre_results_Representation = Cadre_Representation(
            self, self.Results_notebook.tab("Représentation")
        )

        self.Cadre_results_Entrainement = Cadre_Entrainement(
            self, self.Results_notebook.tab("Training")
        )
        self.Cadre_results_Entrainement.pack(fill="both", expand=True)

        self.Cadre_results_Testing = Cadre_Testing(
            self, self.Results_notebook.tab("Testing")
        )
        self.Cadre_results_Testing.pack(fill="both", expand=True)

        self.Cadre_results_Metrics = Cadre_Metrics(
            self, self.Results_notebook.tab("Metrics")
        )
        self.Cadre_results_Metrics.pack(fill="both", expand=True)

        self.Cadre_results_Prediction = Cadre_Prediction(
            self, self.Results_notebook.tab("Prediction")
        )
        self.Cadre_results_Prediction.pack(fill="both", expand=True)
        
        self.Results_notebook.set("Training")
        
        # Titre
        ctk.CTkLabel(
            self.cadre, 
            text="MLApp", 
            font=Fonts.Titre_app
        ).pack(pady=(20, 10))
        # Sous-titre
        ctk.CTkLabel(
            self.cadre,
            text="Machine Learning Application",
            font=Fonts.sous_Titre_app
        ).pack(pady=(0, 40))


        # Section 1 : Modèle
        Label_frame_Modele,Titre_frame_Modele = self.label_frame(self.cadre, title="🧬 Modèle", font=Fonts.Section_params)
        Titre_frame_Modele.pack()
        Label_frame_Modele.pack(fill="both",padx=10)
        Label_frame_Modele.grid_columnconfigure(0, weight=1)
        Label_frame_Modele.grid_rowconfigure(0, weight=1)
        
        self.bouton(Label_frame_Modele, "📂 Charger Modèle", self.test,height=40,font=Fonts.button_font).grid(row=0, column=0,padx=20,pady=20, sticky="nsew")
        self.bouton(Label_frame_Modele, "⚙️ Paramétrer Modèle", self.Parametrer_modele,height=40,font=Fonts.button_font).grid(row=1, column=0,padx=20,pady=(0,20), sticky="nsew")


        # Section 2 : Données
        Label_frame_Donnees,Titre_frame_Donnees = self.label_frame(self.cadre, title="📊 Données", font=Fonts.Section_params)
        Titre_frame_Donnees.pack(pady=(30,0))
        Label_frame_Donnees.pack(fill="both",padx=10)
        Label_frame_Donnees.grid_columnconfigure(0, weight=1)
        Label_frame_Donnees.grid_rowconfigure(0, weight=1)


        
        # self.bouton(Label_frame_Donnees, "📁 Choix Dataset", self.Parametrer_dataset,height=40).grid(row=0, column=0,padx=20,pady=20, sticky="nsew")
        # JSON_Datasets_String=JSON_Datasets_String.replace("'", '"')
        # JSON_Datasets_Dict=json.loads(JSON_Datasets_String)


     
        
        self.bouton(Label_frame_Donnees, "Datasets", self.Gestion_Datasets,height=40,font=Fonts.button_font).grid(row=1, column=0,padx=20,pady=(20,20), sticky="nsew")
        
        self.bouton(Label_frame_Donnees, "📅 Paramétrer Horizon", self.Parametrer_horizon,height=40,font=Fonts.button_font).grid(row=2, column=0,padx=20,pady=(0,20), sticky="nsew")
        
        self.bouton(self.cadre, "📈 Choix Métriques et Visualisations", self.Parametrer_metriques,height=40,font=Fonts.button_font).pack(fill="both",padx=30,pady=(40,0))

        # Section 3 : Actions
        section_actions = ctk.CTkFrame(self.cadre,corner_radius=10)
        section_actions.pack(side="bottom",fill="both",pady=(0,10),padx=10)
        section_actions.grid_columnconfigure(0, weight=1)
        section_actions.grid_columnconfigure(1, weight=1)
        section_actions.grid_rowconfigure(0, weight=1)
        section_actions.grid_rowconfigure(1, weight=1)

        # self.bouton(section_actions, "🚀 Envoyer la configuration au serveur", self.EnvoyerConfig, bg="#d4efdf", fg="#145a32")
        # self.bouton(section_actions, "🛑 Annuler l'entraînement", self.annuler_entrainement, bg="#f9e79f", fg="#7d6608")
        # self.bouton(section_actions, "❌ Quitter", self.destroy, bg="#f5b7b1", fg="#641e16")
        
        start_btn = self.bouton(section_actions, "🚀 Start", self.EnvoyerConfig, height=60, background_color="#d4efdf", text_color="#145a32", font=Fonts.button_font)
        stop_btn = self.bouton(section_actions, "🛑 Stop", self.annuler_entrainement, height=60, background_color="#f9e79f", text_color="#7d6608", font=Fonts.button_font)
        quit_btn = self.bouton(section_actions, "❌ Quitter", self.destroy, height=60, background_color="#f5b7b1", text_color="#641e16", font=Fonts.button_font)

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

        #Chargement des Datasets disponibles
        #threading.Thread(target=self.obtenir_datasets, daemon=True).start()
        self.after(1000, lambda: threading.Thread(target=self.obtenir_datasets, daemon=True).start())

    def label_frame(self, root, title, font, width=200, height=200):
        # Crée un cadre avec un titre simulant un LabelFrame
        frame = ctk.CTkFrame(root, width=width, height=height, corner_radius=10, border_width=2, border_color="gray")
        #frame.pack_propagate(False)

        title_label = ctk.CTkLabel(root, text="  "+title+"  ", font=font)
        return frame, title_label
    
    def bouton(self, parent, texte, commande,padx=5, pady=20,background_color=None, text_color=None, font=None, width=None, height=None,pack=False):
        """Crée un bouton avec le style CustomTkinter par défaut"""
        btn = ctk.CTkButton(
            parent,
            text=texte,
            command=commande
        )

        if font!=None:
            btn.configure(font=font)
        # else:
        #     btn.configure(font=font)
        if background_color!=None:
            btn.configure(fg_color=background_color)
        if text_color!=None:
            btn.configure(text_color=text_color)
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
            messagebox.showinfo("Annulation", "L'entraînement en cours a été annulé.",parent=self)
        else:
            messagebox.showwarning("Info", "Aucun entraînement en cours ou déjà annulé.",parent=self)

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
    
    def Gestion_Datasets(self):
        if self.Gestion_Datasets_instance is None or not self.Gestion_Datasets_instance.est_ouverte():
            self.Gestion_Datasets_instance = Fenetre_Gestion_Datasets(self)
        else:
            self.Gestion_Datasets_instance.lift()  # Ramène la fenêtre secondaire au premier plan

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
        self.config_dataset["name"]=Parametres_temporels.nom_dataset
        self.config_dataset["dates"]=Parametres_temporels.dates
        self.config_dataset["pas_temporel"]=Parametres_temporels.pas_temporel
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
            r = requests.post(url, json=payload, timeout=1000)
            r.raise_for_status()
            data = r.json()
            self.JSON_Datasets=data
            return 1
        
        except Exception as e:
            print("Erreur UI → IA :", e)
            self.JSON_Datasets=None
            messagebox.showwarning("Erreur dans le chargement des Datasets", "Erreur lors de la récupération des datasets depuis le serveur.\nVeuillez vérifier la connexion au serveur ou le bon fonctionnement de ce dernier.",parent=self)
            return 0

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
                        timeout=10000
                    )
                    r.raise_for_status()
                    data = r.json()
                    print(f"\n")
                    print("Dataset récupéré avec succès.")
                    print(f"\n")
                    # print("Réponse fetch_dataset :", data)

                    # Ici tu mets ce que tu veux faire avec le dataset :
                    # par ex. mettre à jour un cadre UI :
                    # self.Cadre_results_Dataset.afficher_infos(data)

                except requests.exceptions.RequestException as e:
                    print(f"Erreur de connexion lors de fetch_dataset: {e}")
                    messagebox.showerror(
                        "Erreur de connexion",
                        f"Impossible de se connecter au serveur (fetch_dataset):\n{str(e)}",parent=self
                    )


            def run_training():
                """Fonction pour exécuter l'entraînement dans un thread séparé"""
                run_fetch_dataset()
                
                y=[]
                yhat=[]
                # Variables pour collecter les données du nouveau pipeline
                y_total = []
                split_info = {}
                val_predictions = []
                val_true = []
                pred_predictions = []
                pred_low = []
                pred_high = []
                pred_true = []
                val_metrics = None
                pred_metrics = None
                current_phase = "init"
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
                                    
                                    # ===== GESTION DES NOUVEAUX ÉVÉNEMENTS =====
                                    
                                    # Info de split initial
                                    if msg.get("type") == "split_info":
                                        split_info = {
                                            "n_train": msg.get("n_train"),
                                            "n_val": msg.get("n_val"),
                                            "n_test": msg.get("n_test"),
                                            "idx_val_start": msg.get("idx_val_start"),
                                            "idx_test_start": msg.get("idx_test_start"),
                                        }
                                        print(f"[SPLIT] Train: {split_info['n_train']}, "
                                            f"Val: {split_info['n_val']}, "
                                            f"Test: {split_info['n_test']}")
                                    
                                    # Série complète
                                    elif msg.get("type") == "serie_complete":
                                        y_total = msg.get("values", [])
                                        print(f"[SÉRIE] {len(y_total)} points reçus")
                                    
                                    # Changement de phase
                                    elif msg.get("type") == "phase":
                                        current_phase = msg.get("phase")
                                        status = msg.get("status")
                                        print(f"[PHASE] {current_phase} -> {status}")
                                    
                                    # ===== PHASE ENTRAÎNEMENT =====
                                    elif msg.get("type") == "epoch":
                                        epoch = msg.get("epochs") or msg.get("epoch")
                                        avg_loss = msg.get("avg_loss")
                                        epoch_s = msg.get("epoch_s")
                                        
                                        if epoch is not None and avg_loss is not None:
                                            self.Cadre_results_Entrainement.add_data_point(epoch, avg_loss, epoch_s)
                                    
                                    # ===== PHASE VALIDATION =====
                                    elif msg.get("type") == "val_start":
                                        print(f"[VAL] Début: {msg.get('n_points')} points")
                                    
                                    elif msg.get("type") == "val_pair":
                                        val_true.append(msg.get("y"))
                                        val_predictions.append(msg.get("yhat"))
                                    
                                    elif msg.get("type") == "val_end":
                                        val_metrics = msg.get("metrics")
                                        residual_std = msg.get("residual_std", 0)
                                        print(f"[VAL] Fin: MSE={val_metrics['overall_mean']['MSE']:.6f}, "
                                            f"residual_std={residual_std:.6f}")
                                    
                                    # ===== PHASE PRÉDICTION =====
                                    elif msg.get("type") == "pred_start":
                                        print(f"[PRED] Début: {msg.get('n_steps')} pas")
                                    
                                    elif msg.get("type") == "pred_point":
                                        pred_predictions.append(msg.get("yhat"))
                                        pred_low.append(msg.get("low"))
                                        pred_high.append(msg.get("high"))
                                        if msg.get("y") is not None:
                                            pred_true.append(msg.get("y"))
                                    
                                    elif msg.get("type") == "pred_end":
                                        pred_metrics = msg.get("metrics")
                                        if pred_metrics:
                                            print(f"[PRED] Fin: MSE={pred_metrics.get('MSE', 'N/A')}")
                                    
                                    # ===== DONNÉES FINALES =====
                                    elif msg.get("type") == "final_plot_data":
                                        # Utiliser directement les données envoyées par le serveur
                                        final_data = {
                                            "series_complete": msg.get("series_complete", y_total),
                                            "val_predictions": msg.get("val_predictions", val_predictions),
                                            "pred_predictions": msg.get("pred_predictions", pred_predictions),
                                            "pred_low": msg.get("pred_low", pred_low),
                                            "pred_high": msg.get("pred_high", pred_high),
                                            "idx_val_start": msg.get("idx_val_start", split_info.get("idx_val_start", 0)),
                                            "idx_test_start": msg.get("idx_test_start", split_info.get("idx_test_start", 0)),
                                        }
                                        
                                        # Afficher les métriques
                                        combined_metrics = {}
                                        if msg.get("val_metrics"):
                                            combined_metrics["validation"] = msg["val_metrics"]
                                        if msg.get("pred_metrics"):
                                            combined_metrics["prediction"] = msg["pred_metrics"]
                                        
                                        if combined_metrics:
                                            self.Cadre_results_Metrics.afficher_Metrics(combined_metrics)
                                        
                                        # Mettre à jour le graphique
                                        self.Cadre_results_Testing.update_full_plot(final_data)
                                    
                                    # ===== FIN DU PIPELINE =====
                                    elif msg.get("type") == "fin_pipeline":
                                        print("[PIPELINE] Terminé!")
                                        break
                                    
                                    # ===== ERREURS =====
                                    elif msg.get("type") == "error":
                                        print(f"ERREUR: {msg.get('message')}")
                                        messagebox.showerror("Erreur", msg.get('message', 'Erreur inconnue'))
                                        break
                                    
                                    # ===== ANCIENS FORMATS (compatibilité) =====
                                    elif msg.get("type") == "test_pair":
                                        # Ancien format - collecter comme validation
                                        val_true.append(msg.get("y"))
                                        val_predictions.append(msg.get("yhat"))
                                    
                                    elif msg.get("type") == "test_final":
                                        self.Cadre_results_Metrics.afficher_Metrics(msg.get("metrics"))
                                    
                                    elif msg.get("type") == "fin_test":
                                        # Ancien format - utiliser l'ancienne méthode
                                        if not pred_predictions:  # Si pas de nouvelles données
                                            self.Cadre_results_Testing.update_prediction_plot(y_total, val_true, val_predictions)
                                        break
                                
                                except json.JSONDecodeError as e:
                                    print(f"Erreur de décodage JSON: {e}")
                                    continue
                
                except requests.exceptions.RequestException as e:
                    print(f"Erreur de connexion: {e}")
                    messagebox.showerror("Erreur de connexion", f"Impossible de se connecter au serveur:\n{str(e)}",parent=self)
                
                finally:
                    # Arrêter l'affichage de l'entraînement
                    self.Cadre_results_Entrainement.stop_training()
            
            

            # Lancer l'entraînement dans un thread séparé pour ne pas bloquer l'interface
            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()

class Cadre_Representation(ctk.CTkFrame):
    def __init__(self, app, master=None):
        super().__init__(master)
        
        # Variables pour stocker les données
        self.epochs = []
        self.losses = []
        self.data_queue = queue.Queue()
        self.is_training = False
        self.is_log=ctk.BooleanVar(value=False)
        
        # Titre
        self.titre = ctk.CTkLabel(
            self, 
            text="Représentation du Modèle", 
            font=Fonts.Tabs_title
        )
        self.titre.pack(pady=(0, 10))

class Cadre_Entrainement(ctk.CTkFrame):
    def __init__(self, app, master=None):
        super().__init__(master)
        
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
            font=Fonts.Tabs_title
        )
        self.titre.pack(pady=(0, 10))

        #self.progress_bar = ttk.Progressbar(self, length=800, mode='determinate')        
        self.progress_bar = ctk.CTkProgressBar(self, width=800, orientation="horizontal",mode="indeterminate")
        # Frame pour les informations
        # self.info_frame = ctk.CTkFrame(self)
        self.info_frame = ctk.CTkFrame(self)
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
        self.fig = Figure(figsize=(10, 6),facecolor=Plot_style.plot_background) #,dpi=100
        self.ax = self.fig.add_subplot(111)
        


        # Style du graphique
        self.ax.tick_params(axis='both',labelsize=20,colors=Plot_style.text_color)
        self.ax.tick_params(which='minor', axis='both',labelsize=20,colors=Plot_style.text_color)
        for spine in self.ax.spines.values():
            spine.set_color(Plot_style.text_color)
          

        self.ax.set_facecolor(Plot_style.plot_background)
        self.ax.grid(True, linestyle='--', alpha=0.3,color=Plot_style.text_color)
        self.ax.grid(which='minor', linestyle=':', alpha=0.2,color=Plot_style.text_color)
        self.ax.set_xlabel('Epoch', fontsize=24, fontweight='bold',color=Plot_style.text_color) 
        self.ax.set_ylabel('Loss', fontsize=24, fontweight='bold',color=Plot_style.text_color) 
        self.ax.set_title('Évolution de la Loss', fontsize=24, fontweight='bold', pad=20,color=Plot_style.text_color) 
        
        # Ligne de tracé (sera mise à jour)
        self.line, = self.ax.plot([], [],'o-', linewidth=2.5,color=Plot_style.train_line,markerfacecolor=Plot_style.markerfacecolor)
        
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
        self.first_epoch_received = False
        self.is_training = True
        self.epochs = []
        self.losses = []
        self.total_epochs = Parametres_entrainement.nb_epochs
        
        self.progress_bar.set(0)
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start() 
        self.progress_bar.pack(before=self.info_frame,pady=15)

        # Vider la file d'attente
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break

        # Réinitialiser le graphique
        self.ax.clear()
        self.ax.set_facecolor(Plot_style.plot_background)
        
        self.ax.tick_params(axis='x', colors=Plot_style.primary_color)
        self.ax.tick_params(axis='y', colors=Plot_style.primary_color)
        for spine in self.ax.spines.values():
            spine.set_color(Plot_style.primary_color)

        # self.ax.set_facecolor(self.fg_color)
        self.ax.grid(True, linestyle='--', alpha=0.3,color=Plot_style.primary_color)
        self.ax.grid(which='minor', linestyle=':', alpha=0.2,color=Plot_style.primary_color)
        self.ax.set_xlabel('Epoch', fontsize=28, fontweight='bold',color=Plot_style.text_color)
        self.ax.set_ylabel('Loss', fontsize=28, fontweight='bold',color=Plot_style.text_color)
        self.ax.set_title('Évolution de la Loss', fontsize=28, fontweight='bold', pad=20,color=Plot_style.text_color)
        
        self.ax.tick_params(axis='both',labelsize=20,color=Plot_style.primary_color)
        self.ax.tick_params(which='minor', axis='both',labelsize=20,color=Plot_style.primary_color)

        self.line, = self.ax.plot([], [],'o-', linewidth=2.5,color=Plot_style.train_line,markerfacecolor=Plot_style.markerfacecolor)
        
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

                if not self.first_epoch_received:
                    self.first_epoch_received = True
                    self.progress_bar.stop()
                    self.progress_bar.configure(mode="determinate")
                
                # Mettre à jour les labels
                self.label_epoch.configure(text=f"Epoch: {epoch}")
                if epoch_s is not None:
                    text = f"Epochs/seconde: {epoch_s:.2f}"
                else:
                    text = "Epochs/seconde: N/A"

                self.label_epoch_s.configure(text=text)

                self.label_loss.configure(text=f"Loss: {loss:.6f}")
                # Mettre à jour la barre de progression
                self.progress_bar.set((epoch / self.total_epochs))
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
                epoch, loss, *rest  = self.data_queue.get_nowait()

                self.epochs.append(epoch)
                self.losses.append(loss)
                
                # Mettre à jour les labels
                self.label_epoch.configure(text=f"Epoch: {epoch}")
                self.label_loss.configure(text=f"Loss: {loss:.6f}")
                # Mettre à jour la barre de progression
                self.progress_bar.set(epoch / self.total_epochs)
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
    """
    Cadre d'affichage des résultats de test avec 3 zones :
    - Série complète (bleu)
    - Validation (vert)
    - Prédiction autorégressive avec halo (rouge/orange)
    """
    def __init__(self, app, master=None):
        super().__init__(master)
        

        # Titre
        self.titre = ctk.CTkLabel(
            self, 
            text="📊 Affichage de la phase de test", 
            font=Fonts.Tabs_title
        )
        self.titre.pack(pady=(0, 10))

        # Création du graphique vide au départ
        self.create_empty_prediction_plot()

        # Données stockées
        self.series_complete = []
        self.val_predictions = []
        self.pred_predictions = []
        self.pred_low = []
        self.pred_high = []
        self.idx_val_start = 0
        self.idx_test_start = 0

    def save_figure(self,fig):
        file_path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("Tous les fichiers", "*.*")],
            title="Enregistrer la figure"
        )
        if file_path:
            fig.savefig(file_path)


    def create_empty_prediction_plot(self):
        """Crée une figure vide et l'affiche dans le Frame."""
        
        # Nettoyage du frame
        # for widget in self.winfo_children():
        #     widget.destroy()

        # Création figure + axes
        self.fig = Figure(figsize=(10, 6), dpi=100,facecolor=Plot_style.plot_background)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(Plot_style.plot_background)

        # Couleurs des axes (spines)
        for spine in self.ax.spines.values():
            spine.set_color(Plot_style.primary_color)
        
        # Couleurs des labels d'axes
        self.ax.xaxis.label.set_color(Plot_style.primary_color)
        self.ax.yaxis.label.set_color(Plot_style.primary_color)

        # Couleur des ticks + taille
        self.ax.tick_params(axis="both", labelsize=20, colors=Plot_style.primary_color)#, colors=Colors.plot_axes_color

        # Grille par défaut
        # self.ax.grid(True, linestyle='--', alpha=0.3, color=Colors.plot_grid_color)
        self.ax.grid(True, which="major", linestyle=":",alpha=0.2,color=Plot_style.primary_color)
        self.ax.grid(True, which="minor", linestyle="--", linewidth=0.5,color=Plot_style.primary_color)
        # self.ax.minorticks_on()

        self.ax.set_title('Test de prédiction (vide)', fontdict=Plot_style.plot_title)

        # Canvas Matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=(0, 0))

        # Liste pour stocker les courbes (important si plusieurs tracés successifs)
        #self.true_lines = []
        #self.pred_lines = []

        # === BOUTON SAUVEGARDE ===
        bouton_sauvegarde = ctk.CTkButton(
            master=self,
            text="💾 Enregistrer la figure",
            font=Fonts.button_font,
            corner_radius=8,
            width=180,
            height=35,
            command=lambda: self.save_figure(self.fig)
        )
        bouton_sauvegarde.pack(pady=(10, 5))
    
    def update_full_plot(self, data: dict):
        """
        Met à jour le graphique avec les données finales.
        
        data attendu :
        {
            "series_complete": [...],
            "val_predictions": [...],
            "pred_predictions": [...],
            "pred_low": [...],
            "pred_high": [...],
            "idx_val_start": int,
            "idx_test_start": int,
        }
        """
        print("[Cadre_Testing] update_full_plot appelée")
        
        try:
            self.ax.clear()
            
            # Récupération des données
            series = data.get("series_complete", [])
            val_preds = data.get("val_predictions", [])
            pred_preds = data.get("pred_predictions", [])
            pred_low = data.get("pred_low", [])
            pred_high = data.get("pred_high", [])
            idx_val = data.get("idx_val_start", 0)
            idx_test = data.get("idx_test_start", 0)
            
            print(f"[Cadre_Testing] Données reçues: series={len(series)}, val={len(val_preds)}, pred={len(pred_preds)}")
            
            if not series:
                print("[Cadre_Testing] ERREUR: series_complete est vide!")
                return
            
            # Conversion en numpy
            series = np.array(series, dtype=float)
            n_total = len(series)
            x_full = np.arange(n_total)
            
            # === 1. SÉRIE COMPLÈTE (bleu) ===
            self.ax.plot(
                x_full, series,
                color='#2E86AB',
                linewidth=1.5,
                alpha=0.7,
                label='Série réelle',
                zorder=1
            )
            print(f"[Cadre_Testing] Série complète tracée: {n_total} points")
            
            # === 2. VALIDATION (vert) ===
            if val_preds and len(val_preds) > 0:
                val_preds_arr = np.array(val_preds, dtype=float)
                # Aplatir si nécessaire (liste de listes)
                if val_preds_arr.ndim > 1:
                    val_preds_arr = val_preds_arr.flatten()
                
                x_val = np.arange(idx_val, idx_val + len(val_preds_arr))
                self.ax.plot(
                    x_val, val_preds_arr,
                    color='#27AE60',
                    linewidth=2,
                    marker='o',
                    markersize=2,
                    label='Validation (teacher forcing)',
                    zorder=3
                )
                print(f"[Cadre_Testing] Validation tracée: {len(val_preds_arr)} points à partir de idx={idx_val}")
            
            # === 3. PRÉDICTION AUTORÉGRESSIVE AVEC HALO ===
            if pred_preds and len(pred_preds) > 0:
                pred_preds_arr = np.array(pred_preds, dtype=float)
                x_pred = np.arange(idx_test, idx_test + len(pred_preds_arr))
                
                # Halo (zone de confiance)
                if pred_low and pred_high and len(pred_low) > 0 and len(pred_high) > 0:
                    pred_low_arr = np.array(pred_low, dtype=float)
                    pred_high_arr = np.array(pred_high, dtype=float)
                    self.ax.fill_between(
                        x_pred, pred_low_arr, pred_high_arr,
                        color='#E74C3C',
                        alpha=0.2,
                        label='Intervalle de confiance (95%)',
                        zorder=2
                    )
                
                # Courbe centrale
                self.ax.plot(
                    x_pred, pred_preds_arr,
                    color='#E74C3C',
                    linewidth=2,
                    marker='s',
                    markersize=2,
                    label='Prédiction (one-step)',
                    zorder=4
                )
                print(f"[Cadre_Testing] Prédiction tracée: {len(pred_preds_arr)} points à partir de idx={idx_test}")
            
            # === LIGNES DE SÉPARATION ===
            if idx_val > 0:
                self.ax.axvline(idx_val, color='#27AE60', linestyle='--', linewidth=1.5, alpha=0.7)
            
            if idx_test > 0 and idx_test != idx_val:
                self.ax.axvline(idx_test, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # === STYLE ===            
            self.ax.set_facecolor(Plot_style.plot_background)
            for spine in self.ax.spines.values():
                spine.set_color(Plot_style.primary_color)
            self.ax.tick_params(axis="both", colors=Plot_style.primary_color, labelsize=20)
            self.ax.grid(True, which="major", linestyle=":",alpha=0.2,color=Plot_style.primary_color)
            self.ax.grid(True, which="minor", linestyle="--", linewidth=0.5,color=Plot_style.primary_color)

            self.ax.set_xlabel('Index', fontsize=16, fontweight='bold', color=Plot_style.primary_color)
            self.ax.set_ylabel('Valeur', fontsize=16, fontweight='bold', color=Plot_style.primary_color)
            self.ax.set_title('Test de prédiction', fontdict=Plot_style.plot_title)
            
            # Légende
            self.ax.legend(
                fancybox=True,
                labelcolor=Plot_style.text_color,
                prop=Plot_style.plot_legend,
                facecolor=Plot_style.plot_background,
                loc='best',
                framealpha=0.8
            )
            
            self.fig.tight_layout()
            self.canvas.draw()
            print("[Cadre_Testing] Graphique mis à jour avec succès!")
            
        except Exception as e:
            print(f"[Cadre_Testing] ERREUR dans update_full_plot: {e}")
            import traceback
            traceback.print_exc()
        
        #self.after(10000,app.Results_notebook.set("Testing"))
    
    def plot_strategies_comparison(self, data: dict):
        """
        Affiche la comparaison de toutes les stratégies de prédiction.
        
        data attendu:
        {
            "series_complete": [...],
            "idx_test_start": int,
            "strategies": {
                "one_step": {"predictions": [...], "low": [...], "high": [...], "metrics": {...}},
                "recalibration": {...},
                "recursive": {...},
            }
        }
        """
        print("[Cadre_Testing] plot_strategies_comparison appelée")
        
        try:
            self.ax.clear()
            
            series = np.array(data.get("series_complete", []), dtype=float)
            idx_test = data.get("idx_test_start", 0)
            strategies = data.get("strategies", {})
            
            n_total = len(series)
            x_full = np.arange(n_total)
            
            # Couleurs pour chaque stratégie
            colors = {
                "one_step": "#27AE60",      # Vert
                "recalibration": "#3498DB", # Bleu
                "recursive": "#E74C3C",     # Rouge
                "direct": "#9B59B6",        # Violet
            }
            
            # Labels
            labels = {
                "one_step": "One-Step (recalib immédiate)",
                "recalibration": "Recalibration périodique",
                "recursive": "Récursif pur",
                "direct": "Direct multi-horizon",
            }
            
            # Série réelle
            self.ax.plot(
                x_full, series,
                color='#7F8C8D',
                linewidth=1.5,
                alpha=0.7,
                label='Série réelle',
                zorder=1
            )
            
            # Tracer chaque stratégie
            for strat_name, strat_data in strategies.items():
                preds = strat_data.get("predictions", [])
                if not preds:
                    continue
                
                preds_arr = np.array(preds, dtype=float)
                x_pred = np.arange(idx_test, idx_test + len(preds_arr))
                color = colors.get(strat_name, "#333333")
                label = labels.get(strat_name, strat_name)
                
                # Courbe de prédiction
                self.ax.plot(
                    x_pred, preds_arr,
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                    label=label,
                    zorder=3
                )
                
                # Métriques dans la légende
                metrics = strat_data.get("metrics", {})
                r2 = metrics.get("R2")
                if r2 is not None:
                    self.ax.plot([], [], ' ', label=f"  R²={r2:.4f}")
            
            # Ligne de séparation
            self.ax.axvline(idx_test, color='#95A5A6', linestyle='--', linewidth=1.5, alpha=0.7)
            
            # Style
            self.ax.set_facecolor(Plot_style.plot_background)
            for spine in self.ax.spines.values():
                spine.set_color(Plot_style.primary_color)
            self.ax.tick_params(axis="both", colors=Plot_style.primary_color, labelsize=14)
            self.ax.grid(True, which="major", color=Plot_style.primary_color, linestyle=":", alpha=0.3)
            
            self.ax.set_xlabel('Index', fontsize=16, fontweight='bold', color=Plot_style.text_color)
            self.ax.set_ylabel('Valeur', fontsize=16, fontweight='bold', color=Plot_style.text_color)
            self.ax.set_title('Comparaison des Stratégies de Prédiction', fontdict=Plot_style.plot_title)
            
            self.ax.legend(
                fancybox=True,
                labelcolor=Plot_style.text_color,
                prop=Plot_style.plot_legend,
                facecolor=Plot_style.plot_background,
                loc='upper left',
                framealpha=0.8
            )
            
            self.fig.tight_layout()
            self.canvas.draw()
            print("[Cadre_Testing] Comparaison tracée avec succès!")
            
        except Exception as e:
            print(f"[Cadre_Testing] ERREUR dans plot_strategies_comparison: {e}")
            import traceback
            traceback.print_exc()

    # === Méthode de compatibilité avec l'ancien code ===
    def update_prediction_plot(self, y_total, y_true, y_pred):
        """
        Ancienne interface - conservée pour compatibilité.
        Convertit vers le nouveau format.
        """
        data = {
            "series_complete": y_total,
            "val_predictions": [],
            "pred_predictions": y_pred,
            "pred_low": [],
            "pred_high": [],
            "idx_val_start": len(y_total) - len(y_pred),
            "idx_test_start": len(y_total) - len(y_pred),
        }
        self.update_full_plot(data)

    def plot_predictions(self, y_true_pairs, y_pred_pairs):
        """Ancienne méthode conservée pour compatibilité"""
        for widget in self.winfo_children():
            if widget != self.bouton_sauvegarde:
                widget.destroy()
        
        if not y_true_pairs or not y_pred_pairs:
            return

        yt = np.array(y_true_pairs, dtype=float)
        yp = np.array(y_pred_pairs, dtype=float)

        if yt.ndim == 2 and yt.shape[1] == 1:
            yt = yt.squeeze(1)
            yp = yp.squeeze(1)

        fig = Figure(facecolor=self.fg_color)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.fg_color)
        ax.grid(True, linestyle='--', alpha=0.3, color='#DCE4EE')
        
        if yt.ndim == 1:
            x = np.arange(len(yt))
            ax.plot(x, yt, color='#2E86AB', linewidth=2, marker='o', markersize=4, 
                   markerfacecolor='white', markeredgewidth=1.5, markeredgecolor='#2E86AB',
                   label='y (vraies valeurs)', alpha=0.8, zorder=2)
            ax.plot(x, yp, color="#e74c3c", linewidth=2, marker='s', markersize=4,
                   markerfacecolor='white', markeredgewidth=1.5, markeredgecolor='#A23B72',
                   label='ŷ (prédictions)', alpha=0.8, zorder=2)
            ax.fill_between(x, yt, yp, alpha=0.2, color='gray', label='Erreur')

        ax.set_title('Comparaison des prédictions', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Valeur', fontsize=11, fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
        
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=(0, 10))

        
        # Afficher
        #plt.show()

class Cadre_Metrics(ctk.CTkFrame):
    def __init__(self, app, master=None):
        super().__init__(master)

        # Titre
        self.titre = ctk.CTkLabel(
            self, 
            text="📊 Affichage des metriques", 
            font=Fonts.Tabs_title
        )
        self.titre.pack(pady=(0, 10))

        # Frame pour les métriques
        self.metrics_frame = ctk.CTkFrame(self, fg_color=self.cget("fg_color"))
        self.metrics_frame.pack(fill="both", expand=True, padx=10, pady=10)

    def afficher_Metrics(self,metrics):
        for widget in self.winfo_children():
            if widget != self.titre:
                widget.destroy()
        for i, (metric, val) in enumerate(metrics["overall_mean"].items()):
            label = ctk.CTkLabel(self, text=f"{metric}: {val:.8f}", font=Fonts.Metrics)
            label.pack(anchor="w", padx=15, pady=5)

    def afficher_Metrics(self, metrics):
        """
        Affiche les métriques - compatible avec l'ancien et le nouveau format.
        
        Ancien format:
            {"overall_mean": {"MSE": ..., "MAE": ..., ...}, "per_dim": {...}}
        
        Nouveau format:
            {"validation": {"overall_mean": {...}}, "prediction": {"MSE": ..., ...}}
        """
        # Nettoyer l'affichage précédent
        for widget in self.metrics_frame.winfo_children():
            #if widget != self.titre:
            widget.destroy()
        
        if not metrics:
            ctk.CTkLabel(
                self.metrics_frame,
                text="Aucune métrique disponible",
                font=("Roboto", 14)
            ).pack(pady=10)
            return
        
        row = 0
        
        # ===== NOUVEAU FORMAT (validation + prediction) =====
        if "validation" in metrics or "prediction" in metrics:
            
            # --- Métriques de Validation ---
            if "validation" in metrics:
                val_metrics = metrics["validation"]
                
                # Titre section
                ctk.CTkLabel(
                    self.metrics_frame,
                    text="📗 Validation (Teacher Forcing)",
                    font=Fonts.title_font,
                    text_color="#27AE60"
                ).grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))
                row += 1
                
                # Extraire les métriques
                overall = val_metrics.get("overall_mean", val_metrics)
                self._afficher_metrics_dict(overall, row)
                row += len(overall) + 1
            
            # --- Métriques de Prédiction ---
            if "prediction" in metrics:
                pred_metrics = metrics["prediction"]
                
                # Titre section
                ctk.CTkLabel(
                    self.metrics_frame,
                    text="📕 Prédiction (Autorégressive)",
                    font=Fonts.title_font,
                    text_color="#E74C3C"
                ).grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(20, 5))
                row += 1
                
                # pred_metrics peut être directement les métriques ou avoir "overall_mean"
                if isinstance(pred_metrics, dict):
                    if "overall_mean" in pred_metrics:
                        overall = pred_metrics["overall_mean"]
                    else:
                        overall = pred_metrics
                    self._afficher_metrics_dict(overall, row)
        
        # ===== ANCIEN FORMAT (overall_mean direct) =====
        elif "overall_mean" in metrics:
            ctk.CTkLabel(
                self.metrics_frame,
                text="📊 Métriques Globales",
                font=Fonts.title_font,
                #text_color=Colors.text_color_primary
            ).grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))
            row += 1
            
            self._afficher_metrics_dict(metrics["overall_mean"], row)
        
        # ===== FORMAT SIMPLE (dict direct) =====
        else:
            ctk.CTkLabel(
                self.metrics_frame,
                text="📊 Métriques",
                font=Fonts.title_font,
            ).grid(row=row, column=0, columnspan=2, sticky="w", padx=10, pady=(10, 5))
            row += 1
            
            self._afficher_metrics_dict(metrics, row)

    def _afficher_metrics_dict(self, metrics_dict: dict, start_row: int):
        """Affiche un dictionnaire de métriques sous forme de grille"""
        
        # Couleurs pour chaque métrique
        metric_colors = {
            "MSE": "#3498DB",
            "MAE": "#9B59B6", 
            "RMSE": "#E67E22",
            "R2": "#1ABC9C"
        }
        
        for i, (metric_name, value) in enumerate(metrics_dict.items()):
            row = start_row + i
            
            # Nom de la métrique
            color = metric_colors.get(metric_name, Plot_style.text_color)
            ctk.CTkLabel(
                self.metrics_frame,
                text=f"{metric_name}:",
                text_color=color,
                font=Fonts.Metrics
            ).grid(row=row, column=0, sticky="w", padx=(20, 10), pady=3)
            
            # Valeur
            if value is None:
                val_str = "N/A"
            elif isinstance(value, float):
                if abs(value) < 0.0001:
                    val_str = f"{value:.2e}"
                elif abs(value) > 1000:
                    val_str = f"{value:.2f}"
                else:
                    val_str = f"{value:.6f}"
            else:
                val_str = str(value)
            
            ctk.CTkLabel(
                self.metrics_frame,
                text=val_str,
                font=Fonts.Metrics,
                #text_color=Colors.text_color_primary
            ).grid(row=row, column=1, sticky="e", padx=(10, 20), pady=3)







class Cadre_Prediction(ctk.CTkFrame):
    def __init__(self, app, master=None):
        super().__init__(master)

        # Titre
        self.titre = ctk.CTkLabel(
            self, 
            text="📊 Affichage de la prédiction", 
            font=Fonts.Tabs_title
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
        ).grid(row=last_row, column=0,padx=10,pady=(30,20),sticky="ew")

        ctk.CTkButton(
            self.params_frame,
            text="❌ Annuler",
            font=("Roboto", 13),
            height=40,
            command=self.destroy
        ).grid(row=last_row, column=1,padx=10,pady=(30,20),sticky="ew")
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
        ctk.CTkLabel(self.params_model_frame, text="Activation:", font=("Roboto", 12)).grid(row=len(params)+1, column=0, sticky="w",padx=10,pady=(5, 15))
        self.cnn_activation = ctk.StringVar(value=Parametres_archi_reseau_CNN.fonction_activation)
        ctk.CTkOptionMenu(
            self.params_model_frame,
            values=["ReLU", "GELU", "tanh", "sigmoid","leaky_relu"],
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
        topdebut = ctk.CTkToplevel(self)
        topdebut.title("Sélectionner la date de début")
        topdebut.geometry('400x300')
        topdebut.after(50, lambda: topdebut.focus_force())
        try:
            date_obj = datetime.strptime(self.date_debut_str.get(), "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.today()
        cal = Calendar(topdebut, selectmode='day', date_pattern='yyyy-mm-dd',
                    year=date_obj.year, month=date_obj.month, day=date_obj.day,font=("Roboto", 20))
        cal.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkButton(topdebut, text="Valider", command=lambda: (self.date_debut_str.set(cal.get_date()), topdebut.destroy())).pack(pady=10)
        
    # Fonction locale : ouvrir calendrier fin
    def ouvrir_calendrier_fin(self):
        topfin = ctk.CTkToplevel(self)
        topfin.title("Sélectionner la date de fin")
        topfin.geometry('400x300')
        topfin.after(50, lambda: topfin.focus_force())
        try:
            date_obj = datetime.strptime(self.date_fin_str.get(), "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.today()
        cal = Calendar(topfin, selectmode='day', date_pattern='yyyy-mm-dd',
                    year=date_obj.year, month=date_obj.month, day=date_obj.day,font=("Roboto", 20))
        cal.pack(fill="both", expand=True,padx=10, pady=10)
        ctk.CTkButton(topfin, text="Valider", command=lambda: (self.date_fin_str.set(cal.get_date()), topfin.destroy())).pack(pady=10)
        
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
        ).grid(row=0, column=0, columnspan=2,padx=20,pady=(20,20))
        # self.geometry("700x800")



        ctk.CTkLabel(self.params_frame, text="Choix de la métrique de visualisation",font=("Roboto Medium", 18)).grid(row=1, column=0,columnspan=2, padx=10,pady=(0,20),sticky="w")

        # Liste des champs
        self.Params_visualisation_suivi_metriques = tk.StringVar(value=",".join(Parametres_visualisation_suivi.metriques))
        champs = [
            ("Loss fonction","loss"),
            ("Métrique 2","loss2"),
            ("Métrique 3","loss3")
        ]

        for i, (label, val) in enumerate(champs):
            ctk.CTkRadioButton(self.params_frame, text=label,variable=self.Params_visualisation_suivi_metriques,value=val).grid(row=i+2, column=1, sticky="ew", padx=10,pady=(0,20))
        next_row=self.params_frame.grid_size()[1]
        ctk.CTkLabel(self.params_frame, text="Choix des métriques à calculer",font=("Roboto Medium", 18)).grid(row=next_row, column=0,columnspan=2, padx=10,pady=(0,20),sticky="w")




        # ctk.CTkLabel(self.params_frame, text="Choix des métriques (séparées par des virgules):", font=("Roboto", 12)).grid(row=1, column=0, sticky="w",padx=10,pady=(0,20))
        # # self.lstm_hidden = ctk.StringVar(value=str(Parametres_visualisation_suivi.metriques))
        # self.Params_visualisation_suivi_metriques = tk.StringVar(value=",".join(Parametres_visualisation_suivi.metriques))
        # ctk.CTkEntry(self.params_frame, textvariable=self.Params_visualisation_suivi_metriques, width=150).grid(row=1, column=1, sticky="e",padx=10,pady=(0,20))





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
            command=self.destroy
        ).grid(row=last_row, column=1,padx=10,pady=(50,20),sticky="ew")

        # Applique la largeur fixe et la hauteur calculée
        # self.ajuster_hauteur_auto()
    
    def est_ouverte(self):
        return self.winfo_exists()

    def save_params(self):
        Parametres_visualisation_suivi.metriques = [m.strip() for m in self.Params_visualisation_suivi_metriques.get().split(",") if m.strip()]
        self.destroy()

#Creer la fenêtre de gestion des datasets
class Fenetre_Gestion_Datasets(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)

        self.master_app = master
        
        self.payload_add_dataset = None
        self.after(100, lambda: self.focus_force())
        self.title("Datasets")
        
        # Polices
        self.font_titre = ("Helvetica", 18, "bold")
        self.font_section = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 12)

        # self.geometry("500x1")  # largeur fixe, hauteur minimale

        # Frame principale
        self.params_frame = ctk.CTkFrame(self)
        self.params_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.params_frame.columnconfigure((0, 1, 2), weight=1)

        self.params_frame.rowconfigure(0, weight=0)   # titre
        self.params_frame.rowconfigure(1, weight=1)   # Treeview → prend tout l'espace disponible
        self.params_frame.rowconfigure(2, weight=0)   # boutons en bas
        

        self.Selected_Dataset={}

        # # Titre simulé
        # tk.Label(self.cadre, text="Paramètres", font=self.font_titre, bg=self.fenetre_bg).pack(anchor="w", pady=(0, 10))
        
        # Polices
        self.font_titre = ("Roboto Medium", 16)
        self.font_label = ("Roboto", 12)

        # Titre
        ctk.CTkLabel(
            self.params_frame,
            text="Gestion des Datasets",
            font=("Roboto Medium", 22)
        ).grid(row=0, column=0, columnspan=2,padx=20,pady=(20,20))
        self.geometry("1000x500")

        self.gestion_datasets()

        ctk.CTkButton(
            self.params_frame,
            text="Ajouter un Dataset",
            font=("Roboto", 13),
            height=40,
            command=self.Ajouter_Dataset
        ).grid(row=2, column=0,padx=10,pady=(50,20),sticky="ew")

        ctk.CTkButton(
            self.params_frame,
            text="Supprimer un Dataset",
            font=("Roboto", 13),
            height=40,
            command=self.Supprimer_Dataset
        ).grid(row=2, column=1,padx=10,pady=(50,20),sticky="ew")

        ctk.CTkButton(
            self.params_frame,
            text="Sélectionner le Dataset",
            font=("Roboto", 13),
            height=40,
            command=self.Select_Dataset
        ).grid(row=2, column=2,padx=10,pady=(50,20),sticky="ew")

    def rafraichir_liste_datasets(self):
        """Rafraîchit l'affichage du Treeview avec les données à jour"""
        # Vider le Treeview
        for item in self.Dataset_tree.get_children():
            self.Dataset_tree.delete(item)
        
        # VÉRIFIER que JSON_Datasets existe et est un dict
        if not hasattr(app, 'JSON_Datasets') or not isinstance(app.JSON_Datasets, dict):
            print("JSON_Datasets n'est pas un dictionnaire:", app.JSON_Datasets)
            return
        
        # DEBUG : Afficher ce qu'on reçoit
        print("DEBUG JSON_Datasets:", app.JSON_Datasets)
        
        # Remplir avec les nouvelles données
        for num, (key, entry) in enumerate(app.JSON_Datasets.items()):
            # VÉRIFIER que entry est bien un dict dabs le doute
            if not isinstance(entry, dict):
                print(f"  Entry '{key}' n'est pas un dict, c'est: {type(entry)} = {entry}")
                continue
            
            nom = entry.get('nom', key)
            dates = entry.get('dates', ['?', '?'])
            dates_str = "  -  ".join([str(d.split(" ")[0]) if isinstance(d, str) else str(d) for d in dates])
            pas = entry.get('pas_temporel', '?')
            taille = 0
            
            self.Dataset_tree.insert(parent='', index=num, values=(nom, dates_str, pas, taille))

    def Supprimer_Dataset(self):
        """Supprime le dataset sélectionné après confirmation"""
        
        # Vérifier qu'un dataset est sélectionné
        if not self.Selected_Dataset or "name" not in self.Selected_Dataset:
            messagebox.showwarning(
                "Aucun Dataset sélectionné",
                "Veuillez sélectionner un dataset dans la liste avant de le supprimer.",
                parent=self
            )
            return
        
        dataset_name = self.Selected_Dataset["name"]
        
        # Demander confirmation
        confirmation = messagebox.askyesno(
            "Confirmer la suppression",
            f"⚠️ Voulez-vous vraiment supprimer le dataset '{dataset_name}' ?\n\n"
            f"Cette action est irréversible.",
            parent=self
        )
        
        if not confirmation:
            return
        
        # Préparer le payload pour le serveur
        payload_delete = {
            "name": dataset_name
        }
        
        url = f"{URL}/datasets/data_suppression_proxy"
        
        try:
            print(f"Envoi de la demande de suppression pour: {dataset_name}")
            r = requests.post(url, json=payload_delete, timeout=1000)
            r.raise_for_status()
            response_data = r.json()
            
            print("Réponse serveur:", response_data)
            
            # Recharger la liste complète des datasets depuis le serveur
            print("Rechargement de la liste des datasets...")
            app.obtenir_datasets()
            
            # Rafraîchir l'affichage
            self.rafraichir_liste_datasets()
            
            # Réinitialiser la sélection
            self.Selected_Dataset = {}
            
            messagebox.showinfo(
                "Succès",
                f"✅ Dataset '{dataset_name}' supprimé avec succès !",
                parent=self
            )
        
        except requests.exceptions.HTTPError as e:
            print("HTTPError:", e)
            error_msg = "Erreur serveur"
            if e.response is not None:
                print("Status:", e.response.status_code)
                print("Body:", e.response.text)
                try:
                    error_detail = e.response.json()
                    error_msg = error_detail.get("detail", str(e))
                except:
                    error_msg = e.response.text
            
            messagebox.showerror(
                "Erreur",
                f"❌ Impossible de supprimer le dataset:\n{error_msg}",
                parent=self
            )
        
        except requests.exceptions.RequestException as e:
            messagebox.showerror(
                "Erreur de connexion",
                f"❌ Impossible de se connecter au serveur:\n{str(e)}",
                parent=self
            )
            print("Erreur de connexion:", e)

    def Ajouter_Dataset(self):
        self.payload_add_dataset = None
        self.payload_name = None
        file_path = filedialog.askopenfilename(
            title="Sélectionner un fichier JSON",
            filetypes=[("Fichiers JSON", "*.json")]
        )

        if not file_path:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.payload_name = os.path.basename(file_path)

            payload_add_dataset = TimeSeriesData(**data)
            self.payload_add_dataset = payload_add_dataset
            
            payload_json = {
                "payload_dataset_add": (
                    self.payload_add_dataset.model_dump(mode="json")
                    if hasattr(self.payload_add_dataset, "model_dump")
                    else self.payload_add_dataset.dict()
                ),
                "payload_name": self.payload_name
            }
            print("✅ Payload validé")

        except json.JSONDecodeError as e:
            messagebox.showwarning(
                title="Erreur de lecture JSON",
                message="  Le fichier sélectionné n'est pas un JSON valide",
                parent=self
            )
            return

        except ValidationError as e:
            messagebox.showwarning(
                title="Erreur de validation",
                message="  Structure du dataset invalide",
                parent=self
            )
            print("Erreur de validation Pydantic:", e)
            return
        
        if self.payload_name in app.JSON_Datasets:
            messagebox.showwarning(
                title="Erreur de validation",
                message="  Un dataset avec ce nom existe déjà",
                parent=self
            )
            return
        
        url = f"{URL}/datasets/add_dataset"
        
        try:
            print("Envoi au serveur...")
            r = requests.post(url, json=payload_json, timeout=1000)
            r.raise_for_status()
            response_data = r.json()
            
            print("Réponse serveur:", response_data)
            
            # ✅ Vérifier que l'ajout a réussi
            if response_data.get('ok'):
                # ✅ RECHARGER la liste complète des datasets depuis le serveur
                print("Rechargement de la liste des datasets...")
                app.obtenir_datasets()
                
                # ✅ Rafraîchir l'affichage
                self.rafraichir_liste_datasets()
                
                messagebox.showinfo(
                    title="Succès",
                    message=f"✅ Dataset '{self.payload_name}' ajouté avec succès !",
                    parent=self
                )
            else:
                messagebox.showwarning(
                    title="Erreur",
                    message="  Le serveur n'a pas confirmé l'ajout",
                    parent=self
                )
        
        except requests.exceptions.RequestException as e:
            messagebox.showwarning(
                title="Erreur",
                message=f"  Erreur lors de l'ajout du dataset:\n{str(e)}",
                parent=self
            )
            print("Erreur:", e)

    def gestion_datasets(self):
        style = ttk.Style()
        style.theme_use("default")
        
        style.configure("Treeview",
                        background="#2a2d2e",
                        foreground="white",
                        fieldbackground="#343638",
                        bordercolor="#343638",
                        borderwidth=0,
                        rowheight=50,
                        font=("Arial", 22))
        style.map('Treeview', background=[('selected', '#22559b')])

        style.configure("Treeview.Heading",
                        background="#565b5e",
                        foreground="white",
                        relief="flat",
                        font=("Arial", 26, "bold"))
        style.map("Treeview.Heading",
                background=[('active', '#3484F0')])

        self.frame_datasets = ctk.CTkFrame(self.params_frame)
        self.frame_datasets.configure(fg_color="#FFFFFF")
        self.frame_datasets.grid(row=1, column=0, columnspan=3, padx=20, pady=(20,20), sticky="nsew")

        columns = ("Nom Dataset", "Dates Dataset", "Pas Temporel", "Taille Dataset")
        
        self.Dataset_tree = ttk.Treeview(self.frame_datasets, columns=columns, show="headings", selectmode="browse")
        self.Dataset_tree.heading("Nom Dataset", text="Nom Dataset")
        self.Dataset_tree.heading("Dates Dataset", text="Dates Dataset")
        self.Dataset_tree.heading("Pas Temporel", text="Pas Temporel")
        self.Dataset_tree.heading("Taille Dataset", text="Taille Dataset")

        # ✅ UTILISER la même logique que rafraichir_liste_datasets
        if hasattr(app, 'JSON_Datasets') and isinstance(app.JSON_Datasets, dict):
            for num, (key, entry) in enumerate(app.JSON_Datasets.items()):
                # ✅ VÉRIFIER que entry est bien un dict
                if not isinstance(entry, dict):
                    print(f"⚠️ Ignorer entry '{key}': {type(entry)}")
                    continue
                
                nom = entry.get('nom', key)
                dates = entry.get('dates', ['?', '?'])
                dates_str = "  -  ".join([str(d.split(" ")[0]) if isinstance(d, str) else str(d) for d in dates])
                pas = entry.get('pas_temporel', '?')
                taille = 0
                
                self.Dataset_tree.insert(parent='', index=num, values=(nom, dates_str, pas, taille))
        
        def on_select(event):
            if not self.Dataset_tree.selection():
                return
            
            item = self.Dataset_tree.selection()[0]
            nom_dataset = self.Dataset_tree.item(item, "values")[0]
            
            # ✅ Vérifier que le dataset existe et est valide
            if nom_dataset in app.JSON_Datasets and isinstance(app.JSON_Datasets[nom_dataset], dict):
                self.Selected_Dataset["name"] = nom_dataset
                self.Selected_Dataset["dates"] = app.JSON_Datasets[nom_dataset]["dates"]
                self.Selected_Dataset["dates"] = pd.to_datetime(self.Selected_Dataset["dates"]).strftime('%Y-%m-%d').tolist()
                self.Selected_Dataset["pas_temporel"] = app.JSON_Datasets[nom_dataset]["pas_temporel"]
            else:
                print(f"⚠️ Dataset '{nom_dataset}' invalide ou inexistant")

        self.Dataset_tree.bind("<<TreeviewSelect>>", on_select)
        
        scrollbar = ctk.CTkScrollbar(self.frame_datasets, command=self.Dataset_tree.yview)
        self.Dataset_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.Dataset_tree.pack(fill="both", expand=True)
    
    def Select_Dataset(self):
        if self.Selected_Dataset!={}:
            Parametres_temporels.nom_dataset=self.Selected_Dataset["name"]
            Parametres_temporels.dates=self.Selected_Dataset["dates"]

            Selected_Dataset.name=self.Selected_Dataset["name"]
            Selected_Dataset.dates=self.Selected_Dataset["dates"]
            Selected_Dataset.pas_temporel=self.Selected_Dataset["pas_temporel"]
            self.destroy()
        else:
            messagebox.showwarning("Aucun Dataset sélectionné", "Veuillez sélectionner un Dataset dans la liste.",parent=self)

    def test(self):
        pass
    
    def est_ouverte(self):
        return self.winfo_exists()

    def save_params(self):
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

def gray_to_hex(gray_code: str) -> str:
    # gray_code attendu sous forme "gray86", "gray17", etc.
    if not gray_code.startswith("gray"):
        raise ValueError("Format invalide : utilisez 'grayXX'.")

    # on récupère la valeur numérique (0–100)
    value = int(gray_code[4:])

    if not (0 <= value <= 100):
        raise ValueError("La valeur doit être entre 0 et 100.")

    # conversion en niveau 0–255
    level = round(value * 255 / 100)

    # couleur HEX en niveau de gris
    return "#{0:02x}{0:02x}{0:02x}".format(level)

Plot_style=Plot_style_class()

# Lancer la boucle principale
if __name__ == "__main__":
    app = Fenetre_Acceuil()
    app.mainloop()