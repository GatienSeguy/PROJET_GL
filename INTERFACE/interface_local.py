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



#URL = "http://192.168.27.66:8000"
#URL = "http://192.168.1.180:8000"
URL = "http://138.231.149.81:8000"

# Paramètres et variables

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
class Fenetre_Acceuil(tk.Tk):
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

        tk.Tk.__init__(self)
        self.title("🧠 Paramétrage du Réseau de Neuronnes")
        self.configure(bg=self.fenetre_bg)
        self.geometry("520x1")

        #feur
        style = ttk.Style()
        style.theme_use('default')

        style.configure('TNotebook', background=self.cadres_bg, borderwidth=0)
        style.configure('TNotebook.Tab', background=self.cadres_bg, padding=[10, 5])
        style.configure('TNotebook.Tab', foreground="black")

        # Couleur du texte quand l'onglet est sélectionné
        style.map('TNotebook.Tab', foreground=[('selected', 'black')])



        # Polices
        self.font_titre = ("Helvetica", 20, "bold")
        self.font_section = ("Helvetica", 18, "bold")
        self.font_bouton = ("Helvetica", 14)

        # Cadre principal de configuration
        self.cadre = tk.Frame(self, bg=self.cadres_bg, padx=20, pady=20, width=200, highlightbackground="black", highlightthickness=2)
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


class Cadre_Entrainement(tk.Frame):
    def __init__(self, app, master=None):
        super().__init__(master)
        self.cadres_bg = app.cadres_bg
        self.configure(bg=self.cadres_bg)
        
        # Variables pour stocker les données
        self.epochs = []
        self.losses = []
        self.data_queue = queue.Queue()
        self.is_training = False
        self.is_log=tk.BooleanVar(value=False)
        
        # Titre
        self.titre = tk.Label(
            self, 
            text="📊 Suivi de l'Entraînement en Temps Réel", 
            font=("Helvetica", 16, "bold"),
            bg=self.cadres_bg,
            fg="#2c3e50"
        )
        self.titre.pack(pady=(0, 10))

        self.progress_bar = ttk.Progressbar(self, length=800, mode='determinate')        
        
        # Frame pour les informations
        self.info_frame = tk.Frame(self, bg=self.cadres_bg)
        self.info_frame.pack(fill="x", pady=(0, 10))
        
        # Labels d'information
        self.label_epoch = tk.Label(
            self.info_frame,
            text="Epoch: -",
            font=("Helvetica", 12, "bold"),
            bg=self.cadres_bg,
            fg="#34495e"
        )
        self.label_epoch.pack(side="left", padx=10)

        self.label_epoch_s = tk.Label(
            self.info_frame,
            text="Epochs/seconde: -",
            font=("Helvetica", 12, "bold"),
            bg=self.cadres_bg,
            fg="#34495e"
        )
        self.label_epoch_s.pack(side="left", padx=10)
        
        self.label_loss = tk.Label(
            self.info_frame,
            text="Loss: -",
            font=("Helvetica", 12, "bold"),
            bg=self.cadres_bg,
            fg="#e74c3c"
        )
        self.label_loss.pack(side="left", padx=10)
        
        self.label_status = tk.Label(
            self.info_frame,
            text="⏸️ En attente",
            font=("Helvetica", 12),
            bg=self.cadres_bg,
            fg="#7f8c8d"
        )
        self.label_status.pack(side="right", padx=10)
        
        # Création du graphique matplotlib avec style moderne
        self.fig = Figure(figsize=(10, 6), facecolor=self.cadres_bg)
        self.ax = self.fig.add_subplot(111)
        
        # Style du graphique
        self.ax.set_facecolor(self.cadres_bg)
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#95a5a6')
        self.ax.set_xlabel('Epoch', fontsize=12, fontweight='bold', color='#2c3e50')
        self.ax.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#2c3e50')
        self.ax.set_title('Évolution de la Loss', fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
        
        # Ligne de tracé (sera mise à jour)
        self.line, = self.ax.plot([], [], 'o-', linewidth=2.5, markersize=6, 
                                   color='#3498db', markerfacecolor='#e74c3c',
                                   markeredgewidth=2, markeredgecolor='#c0392b')
        
        # Canvas pour afficher le graphique
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        


        tk.Checkbutton(self, text="📈 Échelle Logarithmique", variable=self.is_log,
                    bg=self.cadres_bg,fg="black", font=("Helvetica", 14, "bold"), 
                    selectcolor="white",command=self.Log_scale).pack(side="left",pady=(10,0))
        
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
        self.ax.set_facecolor(self.cadres_bg)
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#95a5a6')
        self.ax.grid(True, which='minor', linestyle='--',alpha=0.3, color='#95a5a6') #Grille log
        self.ax.set_xlabel('Epoch', fontsize=12, fontweight='bold', color='#2c3e50')
        self.ax.set_ylabel('Loss', fontsize=12, fontweight='bold', color='#2c3e50')
        self.ax.set_title('Évolution de la Loss', fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
        self.line, = self.ax.plot([], [], 'o-', linewidth=2.5, markersize=6,
                                   color='#3498db', markerfacecolor='#e74c3c',
                                   markeredgewidth=2, markeredgecolor='#c0392b')
        
        self.label_status.config(text="🚀 En cours...", fg="#27ae60")
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
                self.label_epoch.config(text=f"Epoch: {epoch}")
                self.label_epoch_s.config(text=f"Epochs/seconde: {epoch_s:.2f}")
                self.label_loss.config(text=f"Loss: {loss:.6f}")
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
                self.label_epoch.config(text=f"Epoch: {epoch}")
                self.label_loss.config(text=f"Loss: {loss:.6f}")
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
        self.label_status.config(text="✅ Terminé", fg="#27ae60")
        
        # Afficher les statistiques finales
        if len(self.losses) > 0:
            final_loss = self.losses[-1]
            min_loss = min(self.losses)
            self.label_loss.config(text=f"Loss finale: {final_loss:.6f} (min: {min_loss:.6f})")

class Cadre_Testing(tk.Frame):
    def __init__(self, app, master=None):
        super().__init__(master)
        self.cadres_bg = app.cadres_bg
        self.configure(bg=self.cadres_bg)

        # Titre
        self.titre = tk.Label(
            self, 
            text="📊 Suivi de la phase de test", 
            font=("Helvetica", 16, "bold"),
            bg=self.cadres_bg,
            fg="#2c3e50"
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
        fig = Figure(facecolor=self.cadres_bg)
        ax = fig.add_subplot(111)
        
        
        # Style du graphique
        ax.set_facecolor(self.cadres_bg)
        ax.grid(True, linestyle='--', alpha=0.3, color='#95a5a6')
        
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
                    color='#A23B72', 
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

        
        ax.set_facecolor(self.cadres_bg)
        ax.grid(True, linestyle='--', alpha=0.3, color='#95a5a6')
        
        
        # Ajuster les marges
        fig.tight_layout()
        fig.patch.set_facecolor(self.cadres_bg)
        


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

class Cadre_Metrics(tk.Frame):
    def __init__(self, app, master=None):
        super().__init__(master)
        self.cadres_bg = app.cadres_bg
        self.configure(bg=self.cadres_bg)

        # Titre
        self.titre = tk.Label(
            self, 
            text="📊 Affichage des metrics", 
            font=("Helvetica", 16, "bold"),
            bg=self.cadres_bg,
            fg="#2c3e50"
        )
        self.titre.pack(pady=(0, 10))
    def afficher_Metrics(self,metrics):
        for widget in self.winfo_children():
            widget.destroy()
        for i, (metric, val) in enumerate(metrics["overall_mean"].items()):
            label = tk.Label(self, text=f"{metric}: {val:.8f}", font=("Helvetica", 16, "bold"), bg=self.cadres_bg)
            label.pack(anchor="w", padx=15, pady=5)


        pass

class Cadre_Prediction(tk.Frame):
    def __init__(self, app, master=None):
        super().__init__(master)
        self.cadres_bg = app.cadres_bg
        self.configure(bg=self.cadres_bg)

        # Titre
        self.titre = tk.Label(
            self, 
            text="📊 Affichage de la prédiction", 
            font=("Helvetica", 16, "bold"),
            bg=self.cadres_bg,
            fg="#2c3e50"
        )
        self.titre.pack(pady=(0, 10))













# Créer la fenêtre de paramétrage du modèle
class Fenetre_Params(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.cadre_bg=app.cadres_bg
        self.fenetre_bg=app.fenetre_bg
        self.title("🧠 Paramétrage du Réseau de Neuronnes")
        self.configure(bg=self.fenetre_bg)
        # Polices
        self.font_titre = ("Helvetica", 18, "bold")
        self.font_section = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 12)

        


        self.geometry("500x1")  # largeur fixe, hauteur minimale

        # Cadre principal de configuration
        self.cadre = tk.Frame(self, bg=self.fenetre_bg, padx=20, pady=20, highlightbackground="black", highlightthickness=2)
        self.cadre.pack(fill="both", expand="yes", padx=10, pady=20)
        
        # Titre simulé
        tk.Label(self.cadre, text="Paramètres", font=self.font_titre, bg=self.fenetre_bg).pack(anchor="w", pady=(0, 10))

        # Cadre des paramètres
        self.CadreParams = tk.LabelFrame(
            self.cadre, text="", font=self.font_titre,
            bg=self.cadre_bg, fg="#333333", bd=3, relief="ridge", padx=15, pady=15
        )
        self.CadreParams.pack(fill="both", expand=True, pady=(0, 20))

        # Liste des boutons
        boutons = [
            ("Choix du modèle de réseau de neurones", self.Params_choix_reseau_neurones),
            ("Paramétrage de l'architecture réseau", self.Params_archi_reseau),
            ("Choix de la fonction perte (loss)", self.Params_choix_loss_fct),
            ("Choix et paramétrage de l'optimisateur", self.Params_optimisateur),
            ("Paramètres d'entraînement", self.Params_entrainement),
            ("Paramétrage des métriques et visualisations de suivi", self.Params_visualisation_suivi),
        ]

        for texte, commande in boutons:
            self.bouton(self.CadreParams, texte, commande)


        # tk.Button(
        #     self.cadre, text="🚀 Envoyer la configuration au serveur", font=self.font_bouton,
        #     height=2, bg="#b4d9b2", fg="#0f5132", relief="raised", bd=3,
        #     command=self.EnvoyerConfig
        # ).pack(fill="x", pady=10)

        tk.Button(
            self.cadre, text="💾 Sauvegarder la configuration", font=self.font_bouton,
            height=2, bg="#b4d9b2", fg="#0f5132", relief="raised", bd=3,
            command=self.Sauvegarder_Config
        ).pack(fill="x", pady=10)

        tk.Button(
            self.cadre, text="❌ Quitter", font=self.font_bouton,
            height=2, bg="#f7b2b2", fg="#842029", relief="raised", bd=3,
            command=self.destroy
        ).pack(fill="x", pady=(0, 10))

        self.update_idletasks()
        self.geometry(f"500x{self.winfo_reqheight()}")

    def bouton(self, parent, texte, commande, bg="#ffffff", fg="#2c3e50"):
        bouton = tk.Button(
            parent, text=texte, font=self.font_bouton, command=commande,
            bg=bg, fg=fg, relief="raised", bd=2, height=2
        )
        bouton.pack(fill="x", pady=5)

        # Effet de survol
        bouton.bind("<Enter>", lambda e: bouton.config(bg="#d6eaf8"))
        bouton.bind("<Leave>", lambda e: bouton.config(bg=bg))

    def est_ouverte(self):
        return self.winfo_exists()
    
    # Fonctions des fenêtres de paramétrage
    
    def Params_choix_reseau_neurones(self):
        # Variables pour les paramètres
        Params_choix_reseau_neurones_modele = tk.StringVar(value=Parametres_choix_reseau_neurones.modele)  # str ['MLP','LSTM','GRU','CNN']

        # Dictionnaire des descriptions
        descriptions = {
            "MLP": "MLP (Multi-Layer Perceptron) : réseau de neurones dense, adapté aux données tabulaires ou vectorielles.",
            "LSTM": "LSTM (Long Short-Term Memory) : réseau récurrent conçu pour capturer les dépendances temporelles longues.",
            "GRU": "GRU (Gated Recurrent Unit) : variante plus légère du LSTM, efficace pour les séquences temporelles.",
            "CNN": "CNN (Convolutional Neural Network) : réseau spécialisé dans l'extraction de caractéristiques spatiales, souvent utilisé en vision par ordinateur."
        }

        def afficher_description():
            modele = Params_choix_reseau_neurones_modele.get()
            texte = descriptions.get(modele, "Modèle inconnu.")
            messagebox.showinfo("Description du modèle", texte)
            self.lift()  # Ramène la fenêtre secondaire au premier plan
            self.focus_force()  # Force le focus clavier
            fenetre_params_choix_reseau_neurones.lift()  # Ramène la fenêtre tertiaire au premier plan
            fenetre_params_choix_reseau_neurones.focus_force()  # Force le focus clavier


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

        # Ligne 1 : Choix du modèle + bouton "?"
        tk.Label(cadre, text="Choix du modèle :").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Combobox(cadre, values=["MLP", "LSTM", "GRU", "CNN"], textvariable=Params_choix_reseau_neurones_modele, state="readonly").grid(row=0, column=1, pady=5)
        tk.Button(cadre, text="❓", command=afficher_description, width=3).grid(row=0, column=2, padx=5)

        # Boutons
        bouton_frame = tk.Frame(fenetre_params_choix_reseau_neurones)
        bouton_frame.pack(pady=10)
        tk.Button(bouton_frame, text="Sauvegarder et quitter", command=Save_quit).grid(row=0, column=0, padx=10)
        tk.Button(bouton_frame, text="Quitter", command=Quit).grid(row=0, column=1, padx=10)

        fenetre_params_choix_reseau_neurones.mainloop()

    def Params_archi_reseau(self):
        def Save_quit():
            if( Parametres_choix_reseau_neurones.modele=="MLP"):
                Parametres_archi_reseau_MLP.nb_couches = Params_archi_reseau_nb_couches.get()
                Parametres_archi_reseau_MLP.hidden_size = Params_archi_reseau_hidden_size.get()
                Parametres_archi_reseau_MLP.dropout_rate = Params_archi_reseau_dropout_rate.get()
                Parametres_archi_reseau_MLP.fonction_activation = Params_archi_reseau_fonction_activation.get()
            elif( Parametres_choix_reseau_neurones.modele=="CNN"):
                Parametres_archi_reseau_CNN.nb_couches = Params_archi_reseau_nb_couches.get()
                Parametres_archi_reseau_CNN.hidden_size = Params_archi_reseau_hidden_size.get()
                Parametres_archi_reseau_CNN.dropout_rate = Params_archi_reseau_dropout_rate.get()
                Parametres_archi_reseau_CNN.fonction_activation = Params_archi_reseau_fonction_activation.get()
                Parametres_archi_reseau_CNN.kernel_size = Params_archi_reseau_kernel_size.get()
                Parametres_archi_reseau_CNN.stride = Params_archi_reseau_stride.get()
                Parametres_archi_reseau_CNN.padding = Params_archi_reseau_padding.get()
            elif( Parametres_choix_reseau_neurones.modele=="LSTM"):
                Parametres_archi_reseau_LSTM.nb_couches = Params_archi_reseau_nb_couches.get()
                Parametres_archi_reseau_LSTM.hidden_size = Params_archi_reseau_hidden_size.get()
                Parametres_archi_reseau_LSTM.bidirectional = Params_archi_reseau_bidirectional.get()
                Parametres_archi_reseau_LSTM.batch_first = Params_archi_reseau_batch_first.get()


            fenetre_params_archi_reseau.destroy()
        
        def Quit():
            if( Parametres_choix_reseau_neurones.modele=="MLP"):
                Params_archi_reseau_nb_couches.set(Parametres_archi_reseau_MLP.nb_couches)
                Params_archi_reseau_hidden_size.set(Parametres_archi_reseau_MLP.hidden_size)
                Params_archi_reseau_dropout_rate.set(Parametres_archi_reseau_MLP.dropout_rate)
                Params_archi_reseau_fonction_activation.set(Parametres_archi_reseau_MLP.fonction_activation)
            elif( Parametres_choix_reseau_neurones.modele=="CNN"):
                Params_archi_reseau_nb_couches.set(Parametres_archi_reseau_CNN.nb_couches)
                Params_archi_reseau_hidden_size.set(Parametres_archi_reseau_CNN.hidden_size)
                Params_archi_reseau_dropout_rate.set(Parametres_archi_reseau_CNN.dropout_rate)
                Params_archi_reseau_fonction_activation.set(Parametres_archi_reseau_CNN.fonction_activation)
                Params_archi_reseau_kernel_size.set(Parametres_archi_reseau_CNN.kernel_size)
                Params_archi_reseau_stride.set(Parametres_archi_reseau_CNN.stride)
                Params_archi_reseau_padding.set(Parametres_archi_reseau_CNN.padding)
            elif( Parametres_choix_reseau_neurones.modele=="LSTM"):
                Params_archi_reseau_nb_couches.set(Parametres_archi_reseau_LSTM.nb_couches)
                Params_archi_reseau_hidden_size.set(Parametres_archi_reseau_LSTM.hidden_size)
                Params_archi_reseau_bidirectional.set(Parametres_archi_reseau_LSTM.bidirectional)
                Params_archi_reseau_batch_first.set(Parametres_archi_reseau_LSTM.batch_first)
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

        if( Parametres_choix_reseau_neurones.modele=="MLP"):
            # if : ...Variables pour les paramètres POUR MLP
            Params_archi_reseau_nb_couches = tk.IntVar(value=Parametres_archi_reseau_MLP.nb_couches) # int
            Params_archi_reseau_hidden_size = tk.IntVar(value=Parametres_archi_reseau_MLP.hidden_size) # int
            Params_archi_reseau_dropout_rate = tk.DoubleVar(value=Parametres_archi_reseau_MLP.dropout_rate) # float entre 0.0 et 0.9
            Params_archi_reseau_fonction_activation = tk.StringVar(value=Parametres_archi_reseau_MLP.fonction_activation) # fontion ReLU/GELU/tanh
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

        elif( Parametres_choix_reseau_neurones.modele=="CNN"):
            ## if : ...Variables pour les paramètres POUR CNN
            Params_archi_reseau_nb_couches = tk.IntVar(value=Parametres_archi_reseau_CNN.nb_couches) # int
            Params_archi_reseau_hidden_size = tk.IntVar(value=Parametres_archi_reseau_CNN.hidden_size) # int
            Params_archi_reseau_dropout_rate = tk.DoubleVar(value=Parametres_archi_reseau_CNN.dropout_rate) # float entre 0.0 et 0.9
            Params_archi_reseau_fonction_activation = tk.StringVar(value=Parametres_archi_reseau_CNN.fonction_activation) # fontion ReLU/GELU/tanh
            # new CNN
            Params_archi_reseau_kernel_size = tk.IntVar(value = Parametres_archi_reseau_CNN.kernel_size)
            Params_archi_reseau_stride = tk.IntVar(value = Parametres_archi_reseau_CNN.stride)
            Params_archi_reseau_padding = tk.IntVar(value = Parametres_archi_reseau_CNN.padding)

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

            # Ligne 5 : Taille du kernel
            tk.Label(cadre, text="Taille du kernel :").grid(row=4, column=0, sticky="w", pady=5)
            tk.Entry(cadre, textvariable=Params_archi_reseau_kernel_size, validate="key", validatecommand=vcmd).grid(row=4, column=1, pady=5)

            # Ligne 6 : Stride
            tk.Label(cadre, text="Stride :").grid(row=5, column=0, sticky="w", pady=5)
            tk.Entry(cadre, textvariable=Params_archi_reseau_stride, validate="key", validatecommand=vcmd).grid(row=5, column=1, pady=5)

            # Ligne 7 : Padding
            tk.Label(cadre, text="Padding :").grid(row=6, column=0, sticky="w", pady=5)
            tk.Entry(cadre, textvariable=Params_archi_reseau_padding, validate="key", validatecommand=vcmd).grid(row=6, column=1, pady=5)

        elif( Parametres_choix_reseau_neurones.modele=="LSTM"):
            ## if : .... Variables pour les paramètres POUR LSTM
            Params_archi_reseau_nb_couches = tk.IntVar(value=Parametres_archi_reseau_LSTM.nb_couches) # int
            Params_archi_reseau_hidden_size = tk.IntVar(value=Parametres_archi_reseau_LSTM.hidden_size) # int
            Params_archi_reseau_bidirectional = tk.BooleanVar(value = Parametres_archi_reseau_LSTM.bidirectional) #bool
            Params_archi_reseau_batch_first = tk.BooleanVar(value = Parametres_archi_reseau_LSTM.batch_first) #bool

            # Ligne 1 : Nombre de couches de neurones
            tk.Label(cadre, text="Nombre de couches de neurones :").grid(row=0, column=0, sticky="w", pady=5)
            tk.Entry(cadre, textvariable=Params_archi_reseau_nb_couches, validate="key", validatecommand=vcmd).grid(row=0, column=1, pady=5)

            # Ligne 2 : Taille des couches cachées
            tk.Label(cadre, text="Taille des couches cachées :").grid(row=1, column=0, sticky="w", pady=5)
            tk.Entry(cadre, textvariable=Params_archi_reseau_hidden_size, validate="key", validatecommand=vcmd).grid(row=1, column=1, pady=5)

            # Ligne 3 : Bidirectional
            tk.Label(cadre, text="Bidirectional :").grid(row=2, column=0, sticky="w", pady=5)
            tk.Checkbutton(cadre, variable=Params_archi_reseau_bidirectional).grid(row=2, column=1, pady=5)
            # Ligne 4 : Batch first
            tk.Label(cadre, text="Batch first :").grid(row=3, column=0, sticky="w", pady=5)
            tk.Checkbutton(cadre, variable=Params_archi_reseau_batch_first).grid(row=3, column=1, pady=5)






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

    def Sauvegarder_Config(self):
        self.destroy()

# Créer la fenêtre de paramétrage de l'horizon des données
class Fenetre_Params_horizon(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        couleur_fond = "#d9d9d9"
        self.title("🧠 Paramétrage temporels et de découpage des données")

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

        # Variables
        self.Params_temporels_horizon = tk.IntVar(value=Parametres_temporels.horizon)
        self.date_debut_str = tk.StringVar(value=Parametres_temporels.dates[0])
        self.date_fin_str = tk.StringVar(value=Parametres_temporels.dates[1])
        self.Params_temporels_pas_temporel = tk.IntVar(value=Parametres_temporels.pas_temporel)
        self.Params_temporels_portion_decoupage = tk.IntVar(value=Parametres_temporels.portion_decoupage * 100)

        # Validation d'entiers
        vcmd = (self.register(self.validate_int_fct), "%P")

        # Liste des champs
        champs = [
            ("Horizon temporel (int) :", self.Params_temporels_horizon),
            ("Pas temporel (int) :", self.Params_temporels_pas_temporel),
            ("Portion découpage (%) :", self.Params_temporels_portion_decoupage),
        ]

        for i, (label, var) in enumerate(champs):
            tk.Label(self.CadreParams, text=label, bg="#ffffff").grid(row=i, column=0, sticky="w", pady=5)
            tk.Entry(self.CadreParams, textvariable=var, validate="key", validatecommand=vcmd).grid(row=i, column=1, pady=10,padx=105)

        # Dates
        tk.Label(self.CadreParams, text="Date de début :", bg="#ffffff").grid(row=3, column=0, sticky="w", pady=5)
        tk.Button(self.CadreParams, textvariable=self.date_debut_str, command=self.ouvrir_calendrier_debut).grid(row=3, column=1, pady=10)

        tk.Label(self.CadreParams, text="Date de fin :", bg="#ffffff").grid(row=4, column=0, sticky="w", pady=5)
        tk.Button(self.CadreParams, textvariable=self.date_fin_str, command=self.ouvrir_calendrier_fin).grid(row=4, column=1, pady=10)

        # Boutons d'action
        tk.Button(
            self.cadre, text="💾 Sauvegarder la configuration", font=self.font_bouton,
            height=2, bg="#b4d9b2", fg="#0f5132", relief="raised", bd=3,
            command=self.Save_quit
        ).pack(fill="x", pady=10)

        tk.Button(
            self.cadre, text="❌ Quitter", font=self.font_bouton,
            height=2, bg="#f7b2b2", fg="#842029", relief="raised", bd=3,
            command=self.destroy
        ).pack(fill="x", pady=(0, 10))

        self.update_idletasks()
        self.geometry(f"500x{self.winfo_reqheight()}")
    
    def est_ouverte(self):
        return self.winfo_exists()

    def validate_int_fct(self, text):
        return text.isdigit() or text == ""
    
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

#Creer la fenetre de choix des datasets
class Fenetre_Choix_datasets(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
        couleur_fond = "#d9d9d9"
        self.title("📂 Choix des datasets")

        # Définir une police personnalisée
        self.font_titre = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 11)

        self.geometry("500x1")  # largeur fixe, hauteur minimale

        self.cadre = tk.Frame(self, borderwidth=30)
        self.cadre.configure(bg=couleur_fond)
        self.cadre.pack(fill="both", expand="yes")

        # Titre simulé
        tk.Label(self.cadre, text="Choix des datasets", font=self.font_titre, bg=couleur_fond).pack(anchor="w", pady=(0, 10))

        # Cadre des paramètres
        self.CadreParams = tk.LabelFrame(
            self.cadre, text="", font=self.font_titre,
            bg="#ffffff", fg="#333333", bd=3, relief="ridge", padx=15, pady=15
        )
        self.CadreParams.pack(fill="both", expand=True, pady=(0, 20))

        self.Liste_datasets=["A","B","C","D","E","F","G","H","I","J","K","L","M"]  # Exemple de liste de datasets
        

        # Liste des champs
        tk.Label(self.CadreParams, text="Sélectionnez un dataset :", font=self.font_bouton, bg="#ffffff").pack(anchor="w")

        # Créer une variable pour stocker la sélection
        self.dataset_selection = tk.StringVar()

        # Créer la Listbox
        self.listbox_datasets = tk.Listbox(
            self.CadreParams,
            listvariable=self.dataset_selection,
            height=6,
            selectmode="browse",
            font=self.font_bouton,
            bg="#f0f0f0",
            activestyle="dotbox"
        )
        self.listbox_datasets.pack(fill="x", pady=(5, 10))

        # Remplir la Listbox avec les noms des datasets
        for nom in self.Liste_datasets:
            self.listbox_datasets.insert(tk.END, nom)

        # Boutons d'action
        tk.Button(
            self.cadre, text="💾 Sauvegarder la configuration", font=self.font_bouton,
            height=2, bg="#b4d9b2", fg="#0f5132", relief="raised", bd=3,
            command=self.Save_quit
        ).pack(fill="x", pady=10)
        tk.Button(
            self.cadre, text="❌ Quitter", font=self.font_bouton,
            height=2, bg="#f7b2b2", fg="#842029", relief="raised", bd=3,
            command=self.Quit
        ).pack(fill="x", pady=(0, 10))
        self.update_idletasks()
        self.geometry(f"500x{self.winfo_reqheight()}")
    
    def est_ouverte(self):
        return self.winfo_exists()

    def Save_quit(self):
        # Sauvegarder les paramètres
        self.destroy()
    
    def Quit(self):
        # Reinitialiser les paramètres
        self.destroy()

    def Récupérer_datasets(self):
        # Récupérer les datasets sélectionnés
        pass

    

# Lancer la boucle principale
if __name__ == "__main__":
    app = Fenetre_Acceuil()
    app.mainloop()