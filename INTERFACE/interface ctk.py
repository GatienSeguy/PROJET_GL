import customtkinter as ctk
from tkcalendar import Calendar
from datetime import datetime
import requests, json
from tkinter import messagebox
import numpy as np
import threading
import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
from tkinter.filedialog import asksaveasfilename
import tkinter as tk
matplotlib.use("TkAgg")  # backend Tkinter

# Configuration de CustomTkinter avec les styles par défaut (modernes)
ctk.set_appearance_mode("system")  # Mode système (auto light/dark)
ctk.set_default_color_theme("blue")  # Thème bleu par défaut

URL = "http://192.168.27.66:8000"
#URL = "http://192.168.1.180:8000"


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


# Créer la fenêtre d'accueil avec CustomTkinter
class Fenetre_Acceuil(ctk.CTk):
    def __init__(self):
        self.stop_training = False  # drapeau d'annulation
        self.Payload={}
        self.Fenetre_Params_instance = None
        self.Fenetre_Params_horizon_instance = None
        self.Fenetre_Choix_datasets_instance = None
        self.feur_instance = None

        ctk.CTk.__init__(self)
        self.title("🧠 Paramétrage du Réseau de Neuronnes")
        self.geometry("1200x700")

        # Polices (utiliser les valeurs par défaut de CustomTkinter)
        self.font_titre = ("Roboto Medium", 20)
        self.font_section = ("Roboto Medium", 16)
        self.font_bouton = ("Roboto", 13)

        # Cadre principal de configuration
        self.cadre = ctk.CTkFrame(self, corner_radius=10)
        self.cadre.pack(side="left", fill="y", padx=10, pady=10)

        # Cadre des résultats avec CTkTabview
        self.Cadre_results_global = ctk.CTkFrame(self, corner_radius=10)
        self.Cadre_results_global.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Utiliser CTkTabview
        self.Results_notebook = ctk.CTkTabview(self.Cadre_results_global)
        self.Results_notebook.pack(expand=True, fill='both', padx=5, pady=5)

        # Créer les onglets
        self.Results_notebook.add("Training")
        self.Results_notebook.add("Testing")
        self.Results_notebook.add("Metrics")
        self.Results_notebook.add("Prediction")

        # Créer les cadres dans les onglets
        self.Cadre_results_Entrainement = Cadre_Entrainement(
            self, self.Results_notebook.tab("Training")
        )
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
            font=("Roboto Medium", 24)
        ).pack(pady=(20, 10))

        # Sous-titre
        ctk.CTkLabel(
            self.cadre,
            text="Machine Learning Application",
            font=("Roboto", 12)
        ).pack(pady=(0, 20))

        # Section 1 : Modèle
        ctk.CTkLabel(
            self.cadre,
            text="🧬 Modèle",
            font=self.font_section,
            anchor="w"
        ).pack(fill="x", pady=(10, 5), padx=20)

        self.bouton(self.cadre, "📂 Charger Modèle", self.test)
        self.bouton(self.cadre, "⚙️ Paramétrer Modèle", self.Parametrer_modele)

        # Section 2 : Données
        ctk.CTkLabel(
            self.cadre,
            text="📊 Données",
            font=self.font_section,
            anchor="w"
        ).pack(fill="x", pady=(20, 5), padx=20)

        self.bouton(self.cadre, "📁 Choix Dataset", self.Parametrer_dataset)
        self.bouton(self.cadre, "📅 Paramétrer Horizon", self.Parametrer_horizon)

        # Section 3 : Actions
        ctk.CTkLabel(
            self.cadre,
            text="🚀 Actions",
            font=self.font_section,
            anchor="w"
        ).pack(fill="x", pady=(20, 5), padx=20)

        # Bouton Lancer l'entraînement
        self.bouton_lancer = ctk.CTkButton(
            self.cadre,
            text="🚀 Lancer l'entraînement",
            font=self.font_bouton,
            height=40,
            command=self.lancer_entrainement
        )
        self.bouton_lancer.pack(fill="x", pady=5, padx=20)

        # Bouton Annuler
        self.bouton_annuler = ctk.CTkButton(
            self.cadre,
            text="⛔ Annuler l'entraînement",
            font=self.font_bouton,
            height=40,
            state="disabled",
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
            command=self.annuler_entrainement
        )
        self.bouton_annuler.pack(fill="x", pady=5, padx=20)

    def bouton(self, parent, texte, commande):
        """Crée un bouton avec le style CustomTkinter par défaut"""
        btn = ctk.CTkButton(
            parent,
            text=texte,
            font=self.font_bouton,
            height=35,
            command=commande
        )
        btn.pack(fill="x", pady=5, padx=20)
        return btn

    def test(self):
        print("Test button clicked")

    def Parametrer_modele(self):
        if self.Fenetre_Params_instance is None or not self.Fenetre_Params_instance.est_ouverte():
            self.Fenetre_Params_instance = Fenetre_Params(self)
        else:
            self.Fenetre_Params_instance.lift()
            self.Fenetre_Params_instance.focus()

    def Parametrer_dataset(self):
        if self.Fenetre_Choix_datasets_instance is None or not self.Fenetre_Choix_datasets_instance.est_ouverte():
            self.Fenetre_Choix_datasets_instance = Fenetre_Choix_datasets(self)
        else:
            self.Fenetre_Choix_datasets_instance.lift()
            self.Fenetre_Choix_datasets_instance.focus()

    def Parametrer_horizon(self):
        if self.Fenetre_Params_horizon_instance is None or not self.Fenetre_Params_horizon_instance.est_ouverte():
            self.Fenetre_Params_horizon_instance = Fenetre_Params_horizon(self)
        else:
            self.Fenetre_Params_horizon_instance.lift()
            self.Fenetre_Params_horizon_instance.focus()

    def lancer_entrainement(self):
        """Lance l'entraînement en arrière-plan"""
        self.stop_training = False
        self.bouton_lancer.configure(state="disabled")
        self.bouton_annuler.configure(state="normal")
        
        # Réinitialiser les graphiques
        self.Cadre_results_Entrainement.reset_graph()
        self.Cadre_results_Testing.reset_graph()
        self.Cadre_results_Metrics.reset_display()
        
        # Lancer l'entraînement dans un thread séparé
        thread = threading.Thread(target=self.train_model_thread, daemon=True)
        thread.start()

    def annuler_entrainement(self):
        """Annule l'entraînement en cours"""
        self.stop_training = True
        self.bouton_lancer.configure(state="normal")
        self.bouton_annuler.configure(state="disabled")
        messagebox.showinfo("Annulation", "L'entraînement sera annulé à la fin de l'epoch en cours.")

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

    def train_model_thread(self):
        """Thread d'entraînement qui communique avec le serveur via SSE"""
        try:
            # Construire le payload
            payload = {
                "Parametres_temporels": {
                    "horizon": Parametres_temporels.horizon,
                    "dates": Parametres_temporels.dates,
                    "pas_temporel": Parametres_temporels.pas_temporel,
                    "portion_decoupage": Parametres_temporels.portion_decoupage
                },
                "Parametres_choix_reseau_neurones": {
                    "modele": Parametres_choix_reseau_neurones.modele
                },
                "Parametres_choix_loss_fct": {
                    "fonction_perte": Parametres_choix_loss_fct.fonction_perte,
                    "params": Parametres_choix_loss_fct.params
                },
                "Parametres_optimisateur": {
                    "optimisateur": Parametres_optimisateur.optimisateur,
                    "learning_rate": Parametres_optimisateur.learning_rate,
                    "decroissance": Parametres_optimisateur.decroissance,
                    "scheduler": Parametres_optimisateur.scheduler,
                    "patience": Parametres_optimisateur.patience
                },
                "Parametres_entrainement": {
                    "nb_epochs": Parametres_entrainement.nb_epochs,
                    "batch_size": Parametres_entrainement.batch_size,
                    "clip_gradient": Parametres_entrainement.clip_gradient
                },
                "Parametres_visualisation_suivi": {
                    "metriques": Parametres_visualisation_suivi.metriques
                }
            }

            # Payload pour le modèle spécifique
            if Parametres_choix_reseau_neurones.modele == "MLP":
                payload_model = {
                    "nb_couches": Parametres_archi_reseau_MLP.nb_couches,
                    "hidden_size": Parametres_archi_reseau_MLP.hidden_size,
                    "dropout_rate": Parametres_archi_reseau_MLP.dropout_rate,
                    "fonction_activation": Parametres_archi_reseau_MLP.fonction_activation
                }
            elif Parametres_choix_reseau_neurones.modele == "CNN":
                payload_model = {
                    "nb_couches": Parametres_archi_reseau_CNN.nb_couches,
                    "hidden_size": Parametres_archi_reseau_CNN.hidden_size,
                    "kernel_size": Parametres_archi_reseau_CNN.kernel_size,
                    "stride": Parametres_archi_reseau_CNN.stride,
                    "padding": Parametres_archi_reseau_CNN.padding,
                    "fonction_activation": Parametres_archi_reseau_CNN.fonction_activation
                }
            elif Parametres_choix_reseau_neurones.modele == "LSTM":
                payload_model = {
                    "nb_couches": Parametres_archi_reseau_LSTM.nb_couches,
                    "hidden_size": Parametres_archi_reseau_LSTM.hidden_size,
                    "bidirectional": Parametres_archi_reseau_LSTM.bidirectional,
                    "batch_first": Parametres_archi_reseau_LSTM.batch_first
                }

            # Faire la requête SSE
            url = f"{URL}/train_full"
            headers = {"Content-Type": "application/json"}
            
            with requests.post(url, json=payload, params=payload_model, headers=headers, stream=True) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if self.stop_training:
                            break
                        
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                data_str = line_str[6:]  # Remove 'data: ' prefix
                                try:
                                    data = json.loads(data_str)
                                    self.process_sse_event(data)
                                except json.JSONDecodeError:
                                    pass
                else:
                    self.after(0, lambda: messagebox.showerror(
                        "Erreur",
                        f"Erreur serveur: {response.status_code}"
                    ))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erreur", f"Erreur lors de l'entraînement:\n{str(e)}"))
        finally:
            self.after(0, self.fin_entrainement)

    def process_sse_event(self, data):
        """Traite les événements SSE reçus du serveur"""
        event_type = data.get("type")
        
        if event_type == "train":
            # Mise à jour du graphique d'entraînement
            epoch = data.get("epoch", 0)
            loss = data.get("loss", 0)
            self.after(0, lambda: self.Cadre_results_Entrainement.update_graph(epoch, loss))
            
        elif event_type == "test_prediction":
            # Mise à jour du graphique de test
            y_true = data.get("y_true", 0)
            y_pred = data.get("y_pred", 0)
            self.after(0, lambda: self.Cadre_results_Testing.add_prediction(y_true, y_pred))
            
        elif event_type == "test_metrics":
            # Affichage des métriques finales
            self.after(0, lambda: self.Cadre_results_Metrics.display_metrics(data))

    def fin_entrainement(self):
        """Appelé à la fin de l'entraînement"""
        self.bouton_lancer.configure(state="normal")
        self.bouton_annuler.configure(state="disabled")
        if not self.stop_training:
            messagebox.showinfo("Terminé", "L'entraînement est terminé!")


# Cadre pour l'entraînement
class Cadre_Entrainement(ctk.CTkFrame):
    def __init__(self, master, parent):
        super().__init__(parent)
        self.pack(fill="both", expand=True)
        self.master = master

        # Créer la figure matplotlib
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Training Loss")
        self.ax.grid(True, alpha=0.3)

        # Canvas matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Données
        self.epochs = []
        self.losses = []
        self.line, = self.ax.plot([], [], 'b-', linewidth=2)

    def reset_graph(self):
        """Réinitialise le graphique"""
        self.epochs = []
        self.losses = []
        self.line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def update_graph(self, epoch, loss):
        """Met à jour le graphique avec une nouvelle valeur"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        self.line.set_data(self.epochs, self.losses)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()


# Cadre pour les tests
class Cadre_Testing(ctk.CTkFrame):
    def __init__(self, master, parent):
        super().__init__(parent)
        self.pack(fill="both", expand=True)
        self.master = master

        # Créer la figure matplotlib
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("True Values")
        self.ax.set_ylabel("Predictions")
        self.ax.set_title("Test Predictions vs True Values")
        self.ax.grid(True, alpha=0.3)

        # Canvas matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Données
        self.y_true = []
        self.y_pred = []

    def reset_graph(self):
        """Réinitialise le graphique"""
        self.y_true = []
        self.y_pred = []
        self.ax.clear()
        self.ax.set_xlabel("True Values")
        self.ax.set_ylabel("Predictions")
        self.ax.set_title("Test Predictions vs True Values")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def add_prediction(self, y_true, y_pred):
        """Ajoute une prédiction au graphique"""
        self.y_true.append(y_true)
        self.y_pred.append(y_pred)
        
        self.ax.clear()
        self.ax.scatter(self.y_true, self.y_pred, alpha=0.6, s=50)
        
        # Ligne de référence y=x
        if self.y_true:
            min_val = min(min(self.y_true), min(self.y_pred))
            max_val = max(max(self.y_true), max(self.y_pred))
            self.ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
            self.ax.legend()
        
        self.ax.set_xlabel("True Values")
        self.ax.set_ylabel("Predictions")
        self.ax.set_title("Test Predictions vs True Values")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()


# Cadre pour les métriques
class Cadre_Metrics(ctk.CTkFrame):
    def __init__(self, master, parent):
        super().__init__(parent)
        self.pack(fill="both", expand=True, padx=20, pady=20)
        self.master = master

        # Titre
        ctk.CTkLabel(
            self,
            text="📊 Métriques de Test",
            font=("Roboto Medium", 20)
        ).pack(pady=(10, 20))

        # Zone de texte pour les métriques
        self.metrics_text = ctk.CTkTextbox(
            self,
            font=("Roboto Mono", 13),
            wrap="word"
        )
        self.metrics_text.pack(fill="both", expand=True)

    def reset_display(self):
        """Réinitialise l'affichage des métriques"""
        self.metrics_text.delete("0.0", "end")
        self.metrics_text.insert("0.0", "En attente des résultats de test...\n")

    def display_metrics(self, metrics):
        """Affiche les métriques de test"""
        self.metrics_text.delete("0.0", "end")
        
        text = "═══════════════════════════════════════\n"
        text += "          RÉSULTATS DU TEST\n"
        text += "═══════════════════════════════════════\n\n"
        
        if "mse" in metrics:
            text += f"  MSE  (Mean Squared Error)      {metrics['mse']:.6f}\n\n"
        if "mae" in metrics:
            text += f"  MAE  (Mean Absolute Error)     {metrics['mae']:.6f}\n\n"
        if "rmse" in metrics:
            text += f"  RMSE (Root Mean Squared Error) {metrics['rmse']:.6f}\n\n"
        if "mape" in metrics:
            text += f"  MAPE (Mean Abs % Error)        {metrics['mape']:.2f}%\n\n"
        if "r2" in metrics:
            text += f"  R²   (R-Squared Score)         {metrics['r2']:.6f}\n\n"
        
        text += "═══════════════════════════════════════\n"
        
        self.metrics_text.insert("0.0", text)


# Cadre pour les prédictions
class Cadre_Prediction(ctk.CTkFrame):
    def __init__(self, master, parent):
        super().__init__(parent)
        self.pack(fill="both", expand=True)
        self.master = master

        # Label d'information
        ctk.CTkLabel(
            self,
            text="🔮 Fonctionnalité de prédiction",
            font=("Roboto Medium", 18)
        ).pack(pady=(50, 10))
        
        ctk.CTkLabel(
            self,
            text="Cette fonctionnalité sera bientôt disponible.\nVous pourrez utiliser le modèle entraîné pour\nfaire des prédictions sur de nouvelles données.",
            font=("Roboto", 13),
            justify="center"
        ).pack(pady=20)


# Fenêtre de paramètres du modèle
class Fenetre_Params(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("⚙️ Paramètres du Modèle")
        self.geometry("700x800")

        # Polices
        self.font_titre = ("Roboto Medium", 16)
        self.font_label = ("Roboto", 12)

        # Frame principal avec scrollbar
        main_frame = ctk.CTkScrollableFrame(self)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Titre
        ctk.CTkLabel(
            main_frame,
            text="⚙️ Configuration du Modèle",
            font=("Roboto Medium", 22)
        ).pack(pady=(10, 30))

        # Choix du modèle
        model_frame = ctk.CTkFrame(main_frame)
        model_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(
            model_frame,
            text="Type de Modèle:",
            font=("Roboto Medium", 14)
        ).pack(pady=(15, 5))
        
        self.model_var = ctk.StringVar(value=Parametres_choix_reseau_neurones.modele)
        model_menu = ctk.CTkSegmentedButton(
            model_frame,
            values=["MLP", "CNN", "LSTM"],
            variable=self.model_var,
            command=self.on_model_change
        )
        model_menu.pack(pady=(5, 15), padx=20)

        # Frame pour les paramètres spécifiques
        self.params_frame = ctk.CTkFrame(main_frame)
        self.params_frame.pack(fill="both", expand=True, pady=10, padx=10)

        # Frame pour loss et optimizer
        loss_optim_frame = ctk.CTkFrame(main_frame)
        loss_optim_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(
            loss_optim_frame,
            text="⚙️ Configuration de l'entraînement",
            font=("Roboto Medium", 14)
        ).pack(pady=(15, 10))

        # Loss function
        loss_frame = ctk.CTkFrame(loss_optim_frame, fg_color="transparent")
        loss_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            loss_frame,
            text="Fonction de Perte:",
            font=self.font_label
        ).pack(side="left", padx=(0, 10))
        
        self.loss_var = ctk.StringVar(value=Parametres_choix_loss_fct.fonction_perte)
        ctk.CTkOptionMenu(
            loss_frame,
            values=["MSE", "MAE", "Huber"],
            variable=self.loss_var,
            width=150
        ).pack(side="right")

        # Optimizer
        optim_frame = ctk.CTkFrame(loss_optim_frame, fg_color="transparent")
        optim_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            optim_frame,
            text="Optimiseur:",
            font=self.font_label
        ).pack(side="left", padx=(0, 10))
        
        self.optim_var = ctk.StringVar(value=Parametres_optimisateur.optimisateur)
        ctk.CTkOptionMenu(
            optim_frame,
            values=["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta"],
            variable=self.optim_var,
            width=150
        ).pack(side="right")

        # Learning rate
        lr_frame = ctk.CTkFrame(loss_optim_frame, fg_color="transparent")
        lr_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            lr_frame,
            text="Learning Rate:",
            font=self.font_label
        ).pack(side="left", padx=(0, 10))
        
        self.lr_var = ctk.StringVar(value=str(Parametres_optimisateur.learning_rate))
        ctk.CTkEntry(
            lr_frame,
            textvariable=self.lr_var,
            width=150
        ).pack(side="right")

        # Training parameters
        training_frame = ctk.CTkFrame(main_frame)
        training_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(
            training_frame,
            text="📊 Paramètres d'entraînement",
            font=("Roboto Medium", 14)
        ).pack(pady=(15, 10))

        epochs_frame = ctk.CTkFrame(training_frame, fg_color="transparent")
        epochs_frame.pack(fill="x", padx=20, pady=5)
        
        ctk.CTkLabel(
            epochs_frame,
            text="Nombre d'époques:",
            font=self.font_label
        ).pack(side="left", padx=(0, 10))
        
        self.epochs_var = ctk.StringVar(value=str(Parametres_entrainement.nb_epochs))
        ctk.CTkEntry(
            epochs_frame,
            textvariable=self.epochs_var,
            width=150
        ).pack(side="right")

        batch_frame = ctk.CTkFrame(training_frame, fg_color="transparent")
        batch_frame.pack(fill="x", padx=20, pady=(5, 15))
        
        ctk.CTkLabel(
            batch_frame,
            text="Batch Size:",
            font=self.font_label
        ).pack(side="left", padx=(0, 10))
        
        self.batch_var = ctk.StringVar(value=str(Parametres_entrainement.batch_size))
        ctk.CTkEntry(
            batch_frame,
            textvariable=self.batch_var,
            width=150
        ).pack(side="right")

        # Boutons
        button_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        button_frame.pack(fill="x", pady=(30, 10))

        ctk.CTkButton(
            button_frame,
            text="💾 Sauvegarder",
            font=("Roboto", 13),
            height=40,
            command=self.save_params
        ).pack(side="left", expand=True, padx=5)

        ctk.CTkButton(
            button_frame,
            text="❌ Annuler",
            font=("Roboto", 13),
            height=40,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
            command=self.destroy
        ).pack(side="right", expand=True, padx=5)

        # Afficher les paramètres du modèle sélectionné
        self.on_model_change(self.model_var.get())

    def on_model_change(self, model_type):
        """Change les paramètres affichés selon le modèle"""
        # Effacer les widgets existants
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        if model_type == "MLP":
            self.create_mlp_params()
        elif model_type == "CNN":
            self.create_cnn_params()
        elif model_type == "LSTM":
            self.create_lstm_params()

    def create_mlp_params(self):
        """Crée les paramètres spécifiques au MLP"""
        ctk.CTkLabel(
            self.params_frame,
            text="🧠 Paramètres MLP",
            font=("Roboto Medium", 14)
        ).pack(pady=(15, 10))

        # Nombre de couches
        frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(frame, text="Nombre de couches:", font=("Roboto", 12)).pack(side="left")
        self.mlp_layers = ctk.StringVar(value=str(Parametres_archi_reseau_MLP.nb_couches))
        ctk.CTkEntry(frame, textvariable=self.mlp_layers, width=150).pack(side="right")

        # Hidden size
        frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(frame, text="Hidden Size:", font=("Roboto", 12)).pack(side="left")
        self.mlp_hidden = ctk.StringVar(value=str(Parametres_archi_reseau_MLP.hidden_size))
        ctk.CTkEntry(frame, textvariable=self.mlp_hidden, width=150).pack(side="right")

        # Dropout
        frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(frame, text="Dropout Rate:", font=("Roboto", 12)).pack(side="left")
        self.mlp_dropout = ctk.StringVar(value=str(Parametres_archi_reseau_MLP.dropout_rate))
        ctk.CTkEntry(frame, textvariable=self.mlp_dropout, width=150).pack(side="right")

        # Activation
        frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=(5, 15))
        ctk.CTkLabel(frame, text="Activation:", font=("Roboto", 12)).pack(side="left")
        self.mlp_activation = ctk.StringVar(value=Parametres_archi_reseau_MLP.fonction_activation)
        ctk.CTkOptionMenu(
            frame,
            values=["ReLU", "GELU", "tanh", "sigmoid", "leaky_relu"],
            variable=self.mlp_activation,
            width=150
        ).pack(side="right")

    def create_cnn_params(self):
        """Crée les paramètres spécifiques au CNN"""
        ctk.CTkLabel(
            self.params_frame,
            text="🔲 Paramètres CNN",
            font=("Roboto Medium", 14)
        ).pack(pady=(15, 10))

        params = [
            ("Nombre de couches:", Parametres_archi_reseau_CNN.nb_couches, "cnn_layers"),
            ("Hidden Size:", Parametres_archi_reseau_CNN.hidden_size, "cnn_hidden"),
            ("Kernel Size:", Parametres_archi_reseau_CNN.kernel_size, "cnn_kernel"),
            ("Stride:", Parametres_archi_reseau_CNN.stride, "cnn_stride"),
            ("Padding:", Parametres_archi_reseau_CNN.padding, "cnn_padding"),
        ]

        for label, default, attr in params:
            frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
            frame.pack(fill="x", padx=20, pady=5)
            ctk.CTkLabel(frame, text=label, font=("Roboto", 12)).pack(side="left")
            var = ctk.StringVar(value=str(default))
            setattr(self, attr, var)
            ctk.CTkEntry(frame, textvariable=var, width=150).pack(side="right")

        # Activation
        frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=(5, 15))
        ctk.CTkLabel(frame, text="Activation:", font=("Roboto", 12)).pack(side="left")
        self.cnn_activation = ctk.StringVar(value=Parametres_archi_reseau_CNN.fonction_activation)
        ctk.CTkOptionMenu(
            frame,
            values=["ReLU", "GELU", "tanh", "sigmoid"],
            variable=self.cnn_activation,
            width=150
        ).pack(side="right")

    def create_lstm_params(self):
        """Crée les paramètres spécifiques au LSTM"""
        ctk.CTkLabel(
            self.params_frame,
            text="🔄 Paramètres LSTM",
            font=("Roboto Medium", 14)
        ).pack(pady=(15, 10))

        # Nombre de couches
        frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(frame, text="Nombre de couches:", font=("Roboto", 12)).pack(side="left")
        self.lstm_layers = ctk.StringVar(value=str(Parametres_archi_reseau_LSTM.nb_couches))
        ctk.CTkEntry(frame, textvariable=self.lstm_layers, width=150).pack(side="right")

        # Hidden size
        frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=5)
        ctk.CTkLabel(frame, text="Hidden Size:", font=("Roboto", 12)).pack(side="left")
        self.lstm_hidden = ctk.StringVar(value=str(Parametres_archi_reseau_LSTM.hidden_size))
        ctk.CTkEntry(frame, textvariable=self.lstm_hidden, width=150).pack(side="right")

        # Bidirectional
        frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=10)
        self.lstm_bidirectional = ctk.BooleanVar(value=Parametres_archi_reseau_LSTM.bidirectional)
        ctk.CTkCheckBox(
            frame,
            text="Bidirectionnel",
            variable=self.lstm_bidirectional,
            font=("Roboto", 12)
        ).pack(anchor="w")

        # Batch first
        frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=(5, 15))
        self.lstm_batch_first = ctk.BooleanVar(value=Parametres_archi_reseau_LSTM.batch_first)
        ctk.CTkCheckBox(
            frame,
            text="Batch First",
            variable=self.lstm_batch_first,
            font=("Roboto", 12)
        ).pack(anchor="w")

    def save_params(self):
        """Sauvegarde les paramètres"""
        try:
            # Sauvegarder le type de modèle
            Parametres_choix_reseau_neurones.modele = self.model_var.get()

            # Sauvegarder selon le type de modèle
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
                Parametres_archi_reseau_LSTM.bidirectional = self.lstm_bidirectional.get()
                Parametres_archi_reseau_LSTM.batch_first = self.lstm_batch_first.get()

            # Sauvegarder loss et optimizer
            Parametres_choix_loss_fct.fonction_perte = self.loss_var.get()
            Parametres_optimisateur.optimisateur = self.optim_var.get()
            Parametres_optimisateur.learning_rate = float(self.lr_var.get())

            # Sauvegarder paramètres d'entraînement
            Parametres_entrainement.nb_epochs = int(self.epochs_var.get())
            Parametres_entrainement.batch_size = int(self.batch_var.get())

            messagebox.showinfo("Succès", "Paramètres sauvegardés avec succès!")
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Erreur", f"Valeur invalide: {str(e)}")

    def est_ouverte(self):
        return self.winfo_exists()


# Fenêtre de paramètres de l'horizon
class Fenetre_Params_horizon(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("📅 Paramètres Temporels")
        self.geometry("550x550")

        # Polices
        self.font_titre = ("Roboto Medium", 16)
        self.font_bouton = ("Roboto", 13)

        self.cadre = ctk.CTkFrame(self, corner_radius=15)
        self.cadre.pack(fill="both", expand=True, padx=30, pady=30)

        # Titre
        ctk.CTkLabel(
            self.cadre,
            text="📅 Paramètres Temporels",
            font=("Roboto Medium", 20)
        ).pack(pady=(20, 30))

        # Variables
        self.Params_temporels_horizon = ctk.IntVar(value=Parametres_temporels.horizon)
        self.date_debut_str = ctk.StringVar(value=Parametres_temporels.dates[0])
        self.date_fin_str = ctk.StringVar(value=Parametres_temporels.dates[1])
        self.Params_temporels_pas_temporel = ctk.IntVar(value=Parametres_temporels.pas_temporel)
        self.Params_temporels_portion_decoupage = ctk.IntVar(value=int(Parametres_temporels.portion_decoupage * 100))

        # Liste des champs
        champs = [
            ("Horizon temporel (int):", self.Params_temporels_horizon),
            ("Pas temporel (int):", self.Params_temporels_pas_temporel),
            ("Portion découpage (%):", self.Params_temporels_portion_decoupage),
        ]

        for i, (label, var) in enumerate(champs):
            frame = ctk.CTkFrame(self.cadre, fg_color="transparent")
            frame.pack(fill="x", padx=30, pady=8)
            
            ctk.CTkLabel(frame, text=label, font=("Roboto", 13)).pack(side="left")
            ctk.CTkEntry(frame, textvariable=var, width=100, height=35).pack(side="right")

        # Dates
        date_frame = ctk.CTkFrame(self.cadre, fg_color="transparent")
        date_frame.pack(fill="x", padx=30, pady=8)
        ctk.CTkLabel(date_frame, text="Date de début:", font=("Roboto", 13)).pack(side="left")
        ctk.CTkButton(
            date_frame,
            textvariable=self.date_debut_str,
            command=self.ouvrir_calendrier_debut,
            width=150,
            height=35
        ).pack(side="right")

        date_frame2 = ctk.CTkFrame(self.cadre, fg_color="transparent")
        date_frame2.pack(fill="x", padx=30, pady=8)
        ctk.CTkLabel(date_frame2, text="Date de fin:", font=("Roboto", 13)).pack(side="left")
        ctk.CTkButton(
            date_frame2,
            textvariable=self.date_fin_str,
            command=self.ouvrir_calendrier_fin,
            width=150,
            height=35
        ).pack(side="right")

        # Boutons d'action
        button_frame = ctk.CTkFrame(self.cadre, fg_color="transparent")
        button_frame.pack(fill="x", pady=(30, 20), padx=30)

        ctk.CTkButton(
            button_frame,
            text="💾 Sauvegarder",
            font=self.font_bouton,
            height=40,
            command=self.Save_quit
        ).pack(side="left", expand=True, padx=5)

        ctk.CTkButton(
            button_frame,
            text="❌ Annuler",
            font=self.font_bouton,
            height=40,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
            command=self.destroy
        ).pack(side="right", expand=True, padx=5)

    def est_ouverte(self):
        return self.winfo_exists()

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
        messagebox.showinfo("Succès", "Paramètres temporels sauvegardés!")
        self.destroy()


# Créer la fenêtre de choix des datasets
class Fenetre_Choix_datasets(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        self.title("📂 Choix des datasets")
        self.geometry("550x450")

        # Polices
        self.font_titre = ("Roboto Medium", 16)
        self.font_bouton = ("Roboto", 13)

        self.cadre = ctk.CTkFrame(self, corner_radius=15)
        self.cadre.pack(fill="both", expand=True, padx=30, pady=30)

        # Titre
        ctk.CTkLabel(
            self.cadre,
            text="📂 Choix des datasets",
            font=("Roboto Medium", 20)
        ).pack(pady=(20, 30))

        self.Liste_datasets = ["Dataset A", "Dataset B", "Dataset C", "Dataset D", "Dataset E", 
                               "Dataset F", "Dataset G", "Dataset H", "Dataset I", "Dataset J"]

        # Label
        ctk.CTkLabel(
            self.cadre,
            text="Sélectionnez un dataset:",
            font=("Roboto", 13)
        ).pack(padx=30, pady=(0, 10), anchor="w")

        # Variable pour la sélection
        self.dataset_selection = ctk.StringVar(value=self.Liste_datasets[0])

        # Créer un OptionMenu
        self.option_menu = ctk.CTkOptionMenu(
            self.cadre,
            values=self.Liste_datasets,
            variable=self.dataset_selection,
            width=450,
            height=40,
            font=("Roboto", 13),
            dropdown_font=("Roboto", 12)
        )
        self.option_menu.pack(fill="x", pady=(5, 20), padx=30)

        # Info supplémentaire
        info_frame = ctk.CTkFrame(self.cadre)
        info_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        
        ctk.CTkLabel(
            info_frame,
            text="ℹ️ Informations sur le dataset",
            font=("Roboto Medium", 14)
        ).pack(pady=(15, 10))
        
        ctk.CTkLabel(
            info_frame,
            text="Le dataset sélectionné sera utilisé\npour l'entraînement du modèle.",
            font=("Roboto", 12),
            justify="center"
        ).pack(pady=(0, 15))

        # Boutons d'action
        button_frame = ctk.CTkFrame(self.cadre, fg_color="transparent")
        button_frame.pack(fill="x", pady=(10, 20), padx=30)

        ctk.CTkButton(
            button_frame,
            text="💾 Sauvegarder",
            font=self.font_bouton,
            height=40,
            command=self.Save_quit
        ).pack(side="left", expand=True, padx=5)

        ctk.CTkButton(
            button_frame,
            text="❌ Annuler",
            font=self.font_bouton,
            height=40,
            fg_color="transparent",
            border_width=2,
            text_color=("gray10", "gray90"),
            command=self.destroy
        ).pack(side="right", expand=True, padx=5)

    def est_ouverte(self):
        return self.winfo_exists()

    def Save_quit(self):
        selected = self.dataset_selection.get()
        messagebox.showinfo("Succès", f"Dataset '{selected}' sélectionné!")
        self.destroy()


# Lancer la boucle principale
if __name__ == "__main__":
    app = Fenetre_Acceuil()
    app.mainloop()
