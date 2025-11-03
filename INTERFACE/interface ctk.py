# import tkinter as tk
# from tkcalendar import Calendar
from datetime import datetime
import requests, json
# from tkinter import ttk
# from tkinter import messagebox
import numpy as np
import threading
import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
# from tkinter.filedialog import asksaveasfilename
matplotlib.use("TkAgg")  # backend Tkinter


import customtkinter as ctk

URL = "http://192.168.27.66:8000"

class Parametres_temporels_class():
    def __init__(self):
        self.horizon=1 # int
        self.dates=["2001-01-01", "2025-01-02"] # variable datetime
        self.pas_temporel=1 # int
        self.portion_decoupage=0.8# float entre 0 et 1
    def generate_json(self,json):
        pass
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

    def __init__(self):
        self.nb_couches=2 #None # int
        self.hidden_size=64 # int
        self.dropout_rate=0.0 # float entre 0.0 et 0.9
        #self.nb_neurones_par_couche=None # list d'int
        self.fonction_activation="ReLU" # fontion ReLU/GELU/tanh

        self.bidirectional=False # bool
        self.batch_first=False # bool

        self.kernel_size=3 # int
        self.stride=1 # int
        self.padding=0 # int
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

Parametres_temporels=Parametres_temporels_class()
Parametres_choix_reseau_neurones=Parametres_choix_reseau_neurones_class()
Parametres_archi_reseau_MLP=Parametres_archi_reseau_class.MLP_params()
Parametres_archi_reseau_CNN=Parametres_archi_reseau_class.CNN_params()
Parametres_archi_reseau_LSTM=Parametres_archi_reseau_class.LSTM_params()
Parametres_choix_loss_fct=Parametres_choix_loss_fct_class()
Parametres_optimisateur=Parametres_optimisateur_class()
Parametres_entrainement=Parametres_entrainement_class()
Parametres_visualisation_suivi=Parametres_visualisation_suivi_class()


# Créer la fenêtre d'accueil
class Fenetre_Acceuil(ctk.CTk):
    def __init__(self):
        self.cadres_bg="#eaf2f8"
        self.cadres_fg="#e4eff8"
        self.fenetre_bg="#f0f4f8"
        self.stop_training = False  # drapeau d’annulation
        self.Payload={}
        self.Fenetre_Params_instance = None
        self.Fenetre_Params_horizon_instance = None
        self.Fenetre_Choix_datasets_instance = None
        self.feur_instance = None

        ctk.CTk.__init__(self)
        self.title("🧠 Paramétrage du Réseau de Neuronnes")
        self._set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.geometry("520x1")

        # Polices
        self.font_titre = ("Helvetica", 20, "bold")
        self.font_section = ("Helvetica", 18, "bold")
        self.font_bouton = ("Helvetica", 14)

        # Cadre principal de configuration
        self.cadre = ctk.CTkFrame(self, fg_color=self.cadres_bg, corner_radius=10, border_width=2, border_color="black")
        self.cadre.pack(side="left",fill="y", padx=10, pady=20)

        # Cadre des résultats
        self.Cadre_results_global=tk.Frame(self, bg=self.cadres_bg, highlightbackground="black", highlightthickness=2)
        self.Cadre_results_global.pack(side="right",fill="both", expand=True, padx=10, pady=20)

        self.Results_notebook = ttk.Notebook(self.Cadre_results_global, style='TNotebook')
        self.Results_notebook.pack(expand=True, fill='both')

        self.Cadre_results_Entrainement = Cadre_Entrainement(self,self.Cadre_results_global)
        self.Results_notebook.add(self.Cadre_results_Entrainement, text="Training")

        self.Cadre_results_Testing = Cadre_Testing(self,self.Cadre_results_global)
        self.Results_notebook.add(self.Cadre_results_Testing, text="Testing")

        self.Cadre_results_Metrics = Cadre_Metrics(self,self.Cadre_results_global)
        self.Results_notebook.add(self.Cadre_results_Metrics, text="Metrics")

        self.Cadre_results_Prediction = Cadre_Prediction(self,self.Cadre_results_global)
        self.Results_notebook.add(self.Cadre_results_Prediction, text="Prediction")


        # Titre
        tk.Label(self.cadre, text="MLApp", font=self.font_titre, bg=self.cadres_bg, fg="#2c3e50").pack(pady=(0, 20))

        # Section 1 : Modèle
        section_modele = tk.LabelFrame(self.cadre, text="🧬 Modèle", font=self.font_section, bg=self.cadres_bg, fg="#34495e", padx=15, pady=10, bd=2, relief="groove")
        section_modele.pack(fill="x", pady=10)

        self.bouton(section_modele, "📂 Charger Modèle", self.test)
        self.bouton(section_modele, "⚙️ Paramétrer Modèle", self.Parametrer_modele)

        # Section 2 : Données
        section_data = tk.LabelFrame(self.cadre, text="📊 Données", font=self.font_section, bg=self.cadres_bg, fg="#34495e", padx=15, pady=10, bd=2, relief="groove")
        section_data.pack(fill="x", pady=10)

        self.bouton(section_data, "📁 Choix Dataset", self.Parametrer_dataset)
        self.bouton(section_data, "📅 Paramétrer Horizon", self.Parametrer_horizon)

        # Section 3 : Actions
        section_actions = tk.Frame(self.cadre, bg="#f0f4f8")
        section_actions.pack(fill="x", pady=(20, 0))

        self.bouton(section_actions, "🚀 Envoyer la configuration au serveur", self.EnvoyerConfig, bg="#d4efdf", fg="#145a32")
        self.bouton(section_actions, "🛑 Annuler l'entraînement", self.annuler_entrainement, bg="#f9e79f", fg="#7d6608")
        self.bouton(section_actions, "❌ Quitter", self.destroy, bg="#f5b7b1", fg="#641e16")

        self.update_idletasks()
        self.geometry(f"520x{self.winfo_reqheight()}")

        #self.attributes('-fullscreen', True)  # Enable fullscreen
        self.state('zoomed')
        self.bind("<Escape>", lambda event: self.attributes('-fullscreen', False))
        self.bind("<F11>", lambda event: self.attributes('-fullscreen', not self.attributes('-fullscreen')))

    def annuler_entrainement(self):
        """Annule l'entraînement sans fermer le programme."""
        if not self.stop_training:
            self.stop_training = True
            messagebox.showinfo("Annulation", "L'entraînement en cours a été annulé.")
        else:
            messagebox.showwarning("Info", "Aucun entraînement en cours ou déjà annulé.")

    def bouton(self, parent, texte, commande, bg="#ffffff", fg="#2c3e50"):
        bouton = tk.Button(
            parent, text=texte, font=self.font_bouton, command=commande,
            bg=bg, fg=fg, relief="raised", bd=2, height=2
        )
        bouton.pack(fill="x", pady=5)

        # Effet de survol
        bouton.bind("<Enter>", lambda e: bouton.config(bg="#d6eaf8"))
        bouton.bind("<Leave>", lambda e: bouton.config(bg=bg))

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

    def Parametrer_dataset(self):
        if self.Fenetre_Choix_datasets_instance is None or not self.Fenetre_Choix_datasets_instance.est_ouverte():
            self.Fenetre_Choix_datasets_instance = Fenetre_Choix_datasets(self)
        else:
            self.Fenetre_Choix_datasets_instance.lift()  # Ramène la fenêtre secondaire au premier plan
    
    def Formatter_JSON_global(self):
        self.config_totale={}
        self.config_totale["Parametres_temporels"]=Parametres_temporels.__dict__
        self.config_totale["Parametres_choix_reseau_neurones"]=Parametres_choix_reseau_neurones.__dict__
        #self.config_totale["Parametres_archi_reseau"]=Parametres_archi_reseau.__dict__
        self.config_totale["Parametres_choix_loss_fct"]=Parametres_choix_loss_fct.__dict__
        self.config_totale["Parametres_optimisateur"]=Parametres_optimisateur.__dict__
        self.config_totale["Parametres_entrainement"]=Parametres_entrainement.__dict__
        self.config_totale["Parametres_visualisation_suivi"]=Parametres_visualisation_suivi.__dict__
        return self.config_totale
    
    def Formatter_JSON_specif(self):
        self.config_specifique={}
        if Parametres_choix_reseau_neurones.modele=="MLP":
            self.config_specifique["Parametres_archi_reseau"]=Parametres_archi_reseau_MLP.__dict__
        elif Parametres_choix_reseau_neurones.modele=="LSTM":
            self.config_specifique["Parametres_archi_reseau"]=Parametres_archi_reseau_LSTM.__dict__
        elif Parametres_choix_reseau_neurones.modele=="CNN":
            self.config_specifique["Parametres_archi_reseau"]=Parametres_archi_reseau_CNN.__dict__
        return self.config_specifique

    def EnvoyerConfig(self):
        if self.Cadre_results_Entrainement.is_training==False:
            self.stop_training = False
            """Envoie la configuration au serveur et affiche l'entraînement en temps réel"""
            
            # Démarrer l'affichage de l'entraînement
            self.Cadre_results_Entrainement.start_training()
            
            # Préparer les payloads
            payload_global = self.Formatter_JSON_global()
            payload_model = self.Formatter_JSON_specif()
            
            # Avant d'envoyer le payload
            print("Payload envoyé au serveur :", {"payload": payload_global, "payload_model": payload_model})
            
            def run_training():
                """Fonction pour exécuter l'entraînement dans un thread séparé"""
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
