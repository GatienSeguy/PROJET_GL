import customtkinter as ctk
from tkcalendar import Calendar
from datetime import datetime
import requests, json
import tkinter as tk
from tkinter import messagebox
import numpy as np
import threading
import queue
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
from tkinter.filedialog import asksaveasfilename
matplotlib.use("TkAgg")  # backend Tkinter



# URL = "http://192.168.1.94:8000" 
#URL = "http://192.168.27.66:8000"
URL = "http://192.168.1.180:8000"
# URL = "http://138.231.149.81:8000"

# URL = "http://192.168.27.66:8000"


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
class Fenetre_Acceuil(ctk.CTk):
    def __init__(self):
        self.cadres_bg="#eaf2f8"
        self.cadres_fg="#e4eff8"
        self.fenetre_bg="#f0f4f8"
        self.stop_training = False  # drapeau d'annulation
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
        self.Cadre_results_global=ctk.CTkFrame(self, fg_color=self.cadres_bg, corner_radius=10, border_width=2, border_color="black")
        self.Cadre_results_global.pack(side="right",fill="both", expand=True, padx=10, pady=20)

        self.Results_notebook = ctk.CTkTabview(self.Cadre_results_global)
        self.Results_notebook.pack(expand=True, fill='both', padx=10, pady=10)

        self.Results_notebook.add("Training")
        self.Cadre_results_Entrainement = Cadre_Entrainement(self,self.Results_notebook.tab("Training"))

        self.Results_notebook.add("Testing")
        self.Cadre_results_Testing = Cadre_Testing(self,self.Results_notebook.tab("Testing"))

        self.Results_notebook.add("Metrics")
        self.Cadre_results_Metrics = Cadre_Metrics(self,self.Results_notebook.tab("Metrics"))

        self.Results_notebook.add("Prediction")
        self.Cadre_results_Prediction = Cadre_Prediction(self,self.Results_notebook.tab("Prediction"))


        # Titre
        ctk.CTkLabel(self.cadre, text="MLApp", font=self.font_titre, text_color="#2c3e50").pack(pady=(20, 20))

        # Section 1 : Modèle
        section_modele = ctk.CTkFrame(self.cadre, fg_color=self.cadres_bg, corner_radius=5)
        section_modele.pack(fill="x", pady=10, padx=20)
        ctk.CTkLabel(section_modele, text="🧬 Modèle", font=self.font_section, text_color="#34495e").pack(anchor="w", padx=15, pady=(10,5))

        self.bouton(section_modele, "📂 Charger Modèle", self.test)
        self.bouton(section_modele, "⚙️ Paramétrer Modèle", self.Parametrer_modele)

        # Section 2 : Données
        section_data = ctk.CTkFrame(self.cadre, fg_color=self.cadres_bg, corner_radius=5)
        section_data.pack(fill="x", pady=10, padx=20)
        ctk.CTkLabel(section_data, text="📊 Données", font=self.font_section, text_color="#34495e").pack(anchor="w", padx=15, pady=(10,5))

        self.bouton(section_data, "📁 Choix Dataset", self.Parametrer_dataset)
        self.bouton(section_data, "📅 Paramétrer Horizon", self.Parametrer_horizon)

        # Section 3 : Actions
        section_actions = ctk.CTkFrame(self.cadre, fg_color=self.cadres_bg, corner_radius=5)
        section_actions.pack(fill="x", pady=10, padx=20)
        ctk.CTkLabel(section_actions, text="🚀 Actions", font=self.font_section, text_color="#34495e").pack(anchor="w", padx=15, pady=(10,5))

        self.bouton(section_actions, "🎯 Entraîner le modèle", self.Entrainer_modele)
        self.bouton(section_actions, "🧪 Tester le modèle", self.Tester_modele)
        self.bouton(section_actions, "🔮 Prédire", self.Predire_modele)

        # Section 4 : Sauvegarde et Fermeture
        section_save = ctk.CTkFrame(self.cadre, fg_color=self.cadres_bg, corner_radius=5)
        section_save.pack(fill="x", pady=10, padx=20)
        ctk.CTkLabel(section_save, text="💾 Sauvegarde", font=self.font_section, text_color="#34495e").pack(anchor="w", padx=15, pady=(10,5))

        self.bouton(section_save, "📥 Sauvegarder le modèle", self.test)
        self.bouton(section_save, "❌ Fermer", self.destroy)


    # Fonction pour créer un bouton avec style et commande
    def bouton(self, parent, text, command, bg="#d0e8f1", fg="#0f5132"):
        ctk.CTkButton(
            parent, text=text, font=self.font_bouton, command=command,
            fg_color=bg, text_color=fg, hover_color="#b8dce8",
            corner_radius=5, height=40
        ).pack(fill="x", pady=5, padx=15)

    # Fonction vide 
    def test(self):
        pass

    # Fonction de paramétrage du modèle
    def Parametrer_modele(self):
        # Vérifier si la fenêtre existe déjà
        if self.Fenetre_Params_instance is None or not self.Fenetre_Params_instance.est_ouverte():
            self.Fenetre_Params_instance = Fenetre_Params(self)
        else:
            self.Fenetre_Params_instance.lift()  # Mettre au premier plan si elle est déjà ouverte


    # Fonction de paramétrage du dataset
    def Parametrer_dataset(self):
        # Vérifier si la fenêtre existe déjà
        if self.Fenetre_Choix_datasets_instance is None or not self.Fenetre_Choix_datasets_instance.est_ouverte():
            self.Fenetre_Choix_datasets_instance = Fenetre_Choix_datasets(self)
        else:
            self.Fenetre_Choix_datasets_instance.lift()  # Mettre au premier plan si elle est déjà ouverte


    # Fonction de paramétrage de l'horizon 
    def Parametrer_horizon(self):
        # Vérifier si la fenêtre existe déjà
        if self.Fenetre_Params_horizon_instance is None or not self.Fenetre_Params_horizon_instance.est_ouverte():
            self.Fenetre_Params_horizon_instance = Fenetre_Params_horizon(self)
        else:
            self.Fenetre_Params_horizon_instance.lift()  # Mettre au premier plan si elle est déjà ouverte

    # Fonction d'entraînement
    def Entrainer_modele(self):
        self.Cadre_results_Entrainement.start_training()

    # Fonction de test
    def Tester_modele(self):
        self.Cadre_results_Testing.start_testing()

    # Fonction de prédiction
    def Predire_modele(self):
        self.Cadre_results_Prediction.start_prediction()

class Cadre_Entrainement(ctk.CTkFrame):
    def __init__(self, master, parent):
        self.master_window = master
        ctk.CTkFrame.__init__(self, parent, fg_color=master.cadres_fg)
        self.pack(fill="both", expand=True)

        # Titre
        ctk.CTkLabel(self, text="🎯 Entraînement du modèle", font=master.font_section, text_color="#2c3e50").pack(pady=10)

        # Cadre pour la zone de texte
        text_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=5)
        text_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Zone de texte avec scrollbar (utilisation de tk.Text car CTk n'a pas de widget texte natif)
        self.text_widget = tk.Text(text_frame, wrap="word", height=10, font=("Courier", 10), bg="white", fg="black")
        self.text_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        scrollbar = ctk.CTkScrollbar(text_frame, command=self.text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        self.text_widget.config(yscrollcommand=scrollbar.set)

        # Cadre pour les graphiques
        self.graph_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=5)
        self.graph_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Figure matplotlib
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Évolution de la Loss")
        self.ax.set_xlabel("Époques")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Données pour le graphique
        self.epochs_data = []
        self.loss_data = []

        # Bouton Stop avec état
        self.stop_button = ctk.CTkButton(
            self, text="⏹ Stop Training", font=master.font_bouton,
            fg_color="#f7b2b2", text_color="#842029", hover_color="#e89b9b",
            corner_radius=5, height=40, command=self.stop_training, state="disabled"
        )
        self.stop_button.pack(fill="x", padx=20, pady=(0, 20))

    def log_message(self, message):
        self.text_widget.insert("end", message + "\n")
        self.text_widget.see("end")

    def update_graph(self, epoch, loss):
        self.epochs_data.append(epoch)
        self.loss_data.append(loss)

        self.ax.clear()
        self.ax.plot(self.epochs_data, self.loss_data, 'b-', linewidth=2)
        self.ax.set_title("Évolution de la Loss")
        self.ax.set_xlabel("Époques")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def stop_training(self):
        self.master_window.stop_training = True
        self.log_message("⚠️ Arrêt de l'entraînement demandé...")

    def start_training(self):
        self.master_window.stop_training = False
        self.text_widget.delete(1.0, "end")
        self.epochs_data = []
        self.loss_data = []
        self.ax.clear()
        self.ax.set_title("Évolution de la Loss")
        self.ax.set_xlabel("Époques")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

        self.log_message("🚀 Démarrage de l'entraînement...")
        self.stop_button.configure(state="normal")

        thread = threading.Thread(target=self.train_model_thread, daemon=True)
        thread.start()

    def train_model_thread(self):
        try:
            Payload = {
                "modele": Parametres_choix_reseau_neurones.modele,
                "hidden_size": Parametres_archi_reseau_MLP.hidden_size,
                "nb_couches": Parametres_archi_reseau_MLP.nb_couches,
                "dropout_rate": Parametres_archi_reseau_MLP.dropout_rate,
                "fonction_activation": Parametres_archi_reseau_MLP.fonction_activation,
                "fonction_perte": Parametres_choix_loss_fct.fonction_perte,
                "optimisateur": Parametres_optimisateur.optimisateur,
                "learning_rate": Parametres_optimisateur.learning_rate,
                "nb_epochs": Parametres_entrainement.nb_epochs,
                "batch_size": Parametres_entrainement.batch_size,
                "horizon": Parametres_temporels.horizon,
                "dates": Parametres_temporels.dates,
                "pas_temporel": Parametres_temporels.pas_temporel,
                "portion_decoupage": Parametres_temporels.portion_decoupage,
                "metriques": Parametres_visualisation_suivi.metriques,
            }

            self.master_window.Payload = Payload

            response = requests.post(f"{URL}/train-stream/", json=Payload, stream=True, timeout=300)

            if response.status_code == 200:
                for line in response.iter_lines():
                    if self.master_window.stop_training:
                        self.log_message("❌ Entraînement arrêté par l'utilisateur.")
                        break

                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if data["type"] == "epoch":
                                epoch = data["epoch"]
                                loss = data["loss"]
                                self.log_message(f"Époque {epoch}: Loss = {loss:.6f}")
                                self.update_graph(epoch, loss)
                            elif data["type"] == "complete":
                                self.log_message("✅ Entraînement terminé avec succès!")
                        except json.JSONDecodeError:
                            pass
            else:
                self.log_message(f"❌ Erreur {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            self.log_message(f"❌ Erreur de connexion: {str(e)}")
        finally:
            self.stop_button.configure(state="disabled")

class Cadre_Testing(ctk.CTkFrame):
    def __init__(self, master, parent):
        self.master_window = master
        ctk.CTkFrame.__init__(self, parent, fg_color=master.cadres_fg)
        self.pack(fill="both", expand=True)

        # Titre
        ctk.CTkLabel(self, text="🧪 Test du modèle", font=master.font_section, text_color="#2c3e50").pack(pady=10)

        # Cadre pour la zone de texte
        text_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=5)
        text_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Zone de texte avec scrollbar
        self.text_widget = tk.Text(text_frame, wrap="word", height=10, font=("Courier", 10), bg="white", fg="black")
        self.text_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        scrollbar = ctk.CTkScrollbar(text_frame, command=self.text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        self.text_widget.config(yscrollcommand=scrollbar.set)

        # Cadre pour les graphiques
        self.graph_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=5)
        self.graph_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Figure matplotlib
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Prédictions vs Valeurs Réelles")
        self.ax.set_xlabel("Temps")
        self.ax.set_ylabel("Valeur")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

    def log_message(self, message):
        self.text_widget.insert("end", message + "\n")
        self.text_widget.see("end")

    def update_graph(self, y_true, y_pred):
        self.ax.clear()
        self.ax.plot(y_true, 'b-', label='Valeurs Réelles', linewidth=2)
        self.ax.plot(y_pred, 'r--', label='Prédictions', linewidth=2)
        self.ax.set_title("Prédictions vs Valeurs Réelles")
        self.ax.set_xlabel("Temps")
        self.ax.set_ylabel("Valeur")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def start_testing(self):
        self.text_widget.delete(1.0, "end")
        self.log_message("🧪 Démarrage du test...")

        thread = threading.Thread(target=self.test_model_thread, daemon=True)
        thread.start()

    def test_model_thread(self):
        try:
            response = requests.post(f"{URL}/test/", json=self.master_window.Payload, timeout=300)

            if response.status_code == 200:
                data = response.json()
                self.log_message(f"✅ Test terminé!")
                self.log_message(f"Loss: {data.get('loss', 'N/A')}")

                y_true = data.get('y_true', [])
                y_pred = data.get('y_pred', [])

                if y_true and y_pred:
                    self.update_graph(y_true, y_pred)
            else:
                self.log_message(f"❌ Erreur {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            self.log_message(f"❌ Erreur de connexion: {str(e)}")

class Cadre_Metrics(ctk.CTkFrame):
    def __init__(self, master, parent):
        self.master_window = master
        ctk.CTkFrame.__init__(self, parent, fg_color=master.cadres_fg)
        self.pack(fill="both", expand=True)

        # Titre
        ctk.CTkLabel(self, text="📊 Métriques", font=master.font_section, text_color="#2c3e50").pack(pady=10)

        # Cadre pour afficher les métriques
        metrics_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=5)
        metrics_frame.pack(fill="both", expand=True, padx=20, pady=10)

        ctk.CTkLabel(metrics_frame, text="Les métriques seront affichées ici", font=("Helvetica", 12)).pack(pady=20)

class Cadre_Prediction(ctk.CTkFrame):
    def __init__(self, master, parent):
        self.master_window = master
        ctk.CTkFrame.__init__(self, parent, fg_color=master.cadres_fg)
        self.pack(fill="both", expand=True)

        # Titre
        ctk.CTkLabel(self, text="🔮 Prédiction", font=master.font_section, text_color="#2c3e50").pack(pady=10)

        # Cadre pour la zone de texte
        text_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=5)
        text_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Zone de texte avec scrollbar
        self.text_widget = tk.Text(text_frame, wrap="word", height=10, font=("Courier", 10), bg="white", fg="black")
        self.text_widget.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        scrollbar = ctk.CTkScrollbar(text_frame, command=self.text_widget.yview)
        scrollbar.pack(side="right", fill="y")
        self.text_widget.config(yscrollcommand=scrollbar.set)

        # Cadre pour les graphiques
        self.graph_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=5)
        self.graph_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Figure matplotlib
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Prédictions")
        self.ax.set_xlabel("Temps")
        self.ax.set_ylabel("Valeur")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        # Bouton pour sauvegarder les prédictions
        self.save_button = ctk.CTkButton(
            self, text="💾 Sauvegarder les prédictions", font=master.font_bouton,
            fg_color="#b4d9b2", text_color="#0f5132", hover_color="#a2c7a0",
            corner_radius=5, height=40, command=self.save_predictions
        )
        self.save_button.pack(fill="x", padx=20, pady=(0, 20))

        self.predictions_data = None

    def log_message(self, message):
        self.text_widget.insert("end", message + "\n")
        self.text_widget.see("end")

    def update_graph(self, predictions):
        self.ax.clear()
        self.ax.plot(predictions, 'g-', label='Prédictions', linewidth=2)
        self.ax.set_title("Prédictions")
        self.ax.set_xlabel("Temps")
        self.ax.set_ylabel("Valeur")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def save_predictions(self):
        if self.predictions_data is not None:
            filename = asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
            if filename:
                with open(filename, 'w') as f:
                    json.dump(self.predictions_data, f, indent=4)
                self.log_message(f"✅ Prédictions sauvegardées: {filename}")
        else:
            messagebox.showwarning("Aucune donnée", "Aucune prédiction à sauvegarder.")

    def start_prediction(self):
        self.text_widget.delete(1.0, "end")
        self.log_message("🔮 Démarrage de la prédiction...")

        thread = threading.Thread(target=self.predict_model_thread, daemon=True)
        thread.start()

    def predict_model_thread(self):
        try:
            response = requests.post(f"{URL}/predict/", json=self.master_window.Payload, timeout=300)

            if response.status_code == 200:
                data = response.json()
                self.predictions_data = data
                self.log_message(f"✅ Prédiction terminée!")

                predictions = data.get('predictions', [])

                if predictions:
                    self.update_graph(predictions)
                    self.log_message(f"Nombre de prédictions: {len(predictions)}")
            else:
                self.log_message(f"❌ Erreur {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            self.log_message(f"❌ Erreur de connexion: {str(e)}")

# Créer la fenêtre de paramétrage
class Fenetre_Params(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        couleur_fond = "#d9d9d9"
        self.title("⚙️ Paramétrage du modèle")

        # Définir une police personnalisée
        self.font_titre = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 11)

        self.geometry("700x1")  # largeur fixe, hauteur minimale

        self.cadre = ctk.CTkFrame(self, fg_color=couleur_fond)
        self.cadre.pack(fill="both", expand=True, padx=30, pady=30)

        # Titre simulé
        ctk.CTkLabel(self.cadre, text="Paramètres", font=self.font_titre, text_color="black").pack(anchor="w", pady=(0, 10))

        # Cadre des paramètres
        self.CadreParams = ctk.CTkFrame(self.cadre, fg_color="#ffffff", corner_radius=10, border_width=3, border_color="#cccccc")
        self.CadreParams.pack(fill="both", expand=True, pady=(0, 20))

        # Création du notebook avec CTkTabview
        self.notebook = ctk.CTkTabview(self.CadreParams)
        self.notebook.pack(expand=True, fill='both', padx=15, pady=15)

        # Onglet Architecture
        self.notebook.add("Architecture")
        self.tab_architecture = self.notebook.tab("Architecture")

        # Onglet Fonction de Perte
        self.notebook.add("Fonction de Perte")
        self.tab_loss = self.notebook.tab("Fonction de Perte")

        # Onglet Optimisateur
        self.notebook.add("Optimisateur")
        self.tab_optimiseur = self.notebook.tab("Optimisateur")

        # Onglet Entraînement
        self.notebook.add("Entraînement")
        self.tab_entrainement = self.notebook.tab("Entraînement")

        # Onglet Visualisation
        self.notebook.add("Visualisation")
        self.tab_visualisation = self.notebook.tab("Visualisation")

        # Remplir les onglets
        self.create_architecture_tab()
        self.create_loss_tab()
        self.create_optimiseur_tab()
        self.create_entrainement_tab()
        self.create_visualisation_tab()

        # Boutons d'action
        ctk.CTkButton(
            self.cadre, text="💾 Sauvegarder la configuration", font=self.font_bouton,
            height=40, fg_color="#b4d9b2", text_color="#0f5132", hover_color="#a2c7a0",
            corner_radius=5, command=self.Save_quit
        ).pack(fill="x", pady=10)

        ctk.CTkButton(
            self.cadre, text="❌ Quitter", font=self.font_bouton,
            height=40, fg_color="#f7b2b2", text_color="#842029", hover_color="#e89b9b",
            corner_radius=5, command=self.Quit
        ).pack(fill="x", pady=(0, 10))

        self.update_idletasks()
        self.geometry(f"700x{self.winfo_reqheight()}")

    def create_architecture_tab(self):
        frame = ctk.CTkScrollableFrame(self.tab_architecture, fg_color="#ffffff")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Choix du modèle
        self.Params_modele = tk.StringVar(value=Parametres_choix_reseau_neurones.modele)
        ctk.CTkLabel(frame, text="Type de Modèle :", font=self.font_bouton).grid(row=0, column=0, sticky="w", pady=5, padx=5)
        modele_menu = ctk.CTkOptionMenu(frame, variable=self.Params_modele, values=["MLP", "LSTM", "GRU", "CNN"], command=self.update_architecture_fields)
        modele_menu.grid(row=0, column=1, pady=5, padx=5, sticky="ew")

        # Frame pour les paramètres spécifiques
        self.params_frame = ctk.CTkFrame(frame, fg_color="#f0f0f0", corner_radius=5)
        self.params_frame.grid(row=1, column=0, columnspan=2, pady=10, padx=5, sticky="ew")

        frame.grid_columnconfigure(1, weight=1)

        self.update_architecture_fields(self.Params_modele.get())

    def update_architecture_fields(self, choice):
        # Effacer les widgets existants
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        vcmd = (self.register(self.validate_int_fct), "%P")

        if choice == "MLP":
            self.Params_archi_nb_couches = tk.IntVar(value=Parametres_archi_reseau_MLP.nb_couches)
            self.Params_archi_hidden_size = tk.IntVar(value=Parametres_archi_reseau_MLP.hidden_size)
            self.Params_archi_dropout_rate = tk.DoubleVar(value=Parametres_archi_reseau_MLP.dropout_rate)
            self.Params_archi_fonction_activation = tk.StringVar(value=Parametres_archi_reseau_MLP.fonction_activation)

            fields = [
                ("Nombre de couches :", self.Params_archi_nb_couches, "entry"),
                ("Taille cachée :", self.Params_archi_hidden_size, "entry"),
                ("Taux de dropout :", self.Params_archi_dropout_rate, "entry"),
                ("Fonction d'activation :", self.Params_archi_fonction_activation, "menu", ["ReLU", "GELU", "tanh"]),
            ]

        elif choice == "CNN":
            self.Params_archi_nb_couches = tk.IntVar(value=Parametres_archi_reseau_CNN.nb_couches)
            self.Params_archi_hidden_size = tk.IntVar(value=Parametres_archi_reseau_CNN.hidden_size)
            self.Params_archi_dropout_rate = tk.DoubleVar(value=Parametres_archi_reseau_CNN.dropout_rate)
            self.Params_archi_fonction_activation = tk.StringVar(value=Parametres_archi_reseau_CNN.fonction_activation)
            self.Params_archi_kernel_size = tk.IntVar(value=Parametres_archi_reseau_CNN.kernel_size)
            self.Params_archi_stride = tk.IntVar(value=Parametres_archi_reseau_CNN.stride)
            self.Params_archi_padding = tk.IntVar(value=Parametres_archi_reseau_CNN.padding)

            fields = [
                ("Nombre de couches :", self.Params_archi_nb_couches, "entry"),
                ("Taille cachée :", self.Params_archi_hidden_size, "entry"),
                ("Taux de dropout :", self.Params_archi_dropout_rate, "entry"),
                ("Fonction d'activation :", self.Params_archi_fonction_activation, "menu", ["ReLU", "GELU", "tanh"]),
                ("Taille du noyau :", self.Params_archi_kernel_size, "entry"),
                ("Stride :", self.Params_archi_stride, "entry"),
                ("Padding :", self.Params_archi_padding, "entry"),
            ]

        elif choice == "LSTM" or choice == "GRU":
            self.Params_archi_nb_couches = tk.IntVar(value=Parametres_archi_reseau_LSTM.nb_couches)
            self.Params_archi_hidden_size = tk.IntVar(value=Parametres_archi_reseau_LSTM.hidden_size)
            self.Params_archi_bidirectional = tk.BooleanVar(value=Parametres_archi_reseau_LSTM.bidirectional)
            self.Params_archi_batch_first = tk.BooleanVar(value=Parametres_archi_reseau_LSTM.batch_first)

            fields = [
                ("Nombre de couches :", self.Params_archi_nb_couches, "entry"),
                ("Taille cachée :", self.Params_archi_hidden_size, "entry"),
                ("Bidirectionnel :", self.Params_archi_bidirectional, "checkbox"),
                ("Batch first :", self.Params_archi_batch_first, "checkbox"),
            ]

        for i, field_info in enumerate(fields):
            if field_info[2] == "entry":
                ctk.CTkLabel(self.params_frame, text=field_info[0], font=self.font_bouton).grid(row=i, column=0, sticky="w", pady=5, padx=5)
                ctk.CTkEntry(self.params_frame, textvariable=field_info[1]).grid(row=i, column=1, pady=5, padx=5, sticky="ew")
            elif field_info[2] == "menu":
                ctk.CTkLabel(self.params_frame, text=field_info[0], font=self.font_bouton).grid(row=i, column=0, sticky="w", pady=5, padx=5)
                ctk.CTkOptionMenu(self.params_frame, variable=field_info[1], values=field_info[3]).grid(row=i, column=1, pady=5, padx=5, sticky="ew")
            elif field_info[2] == "checkbox":
                ctk.CTkCheckBox(self.params_frame, text=field_info[0], variable=field_info[1], font=self.font_bouton).grid(row=i, column=0, columnspan=2, sticky="w", pady=5, padx=5)

        self.params_frame.grid_columnconfigure(1, weight=1)

    def create_loss_tab(self):
        frame = ctk.CTkFrame(self.tab_loss, fg_color="#ffffff")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.Params_loss_fonction_perte = tk.StringVar(value=Parametres_choix_loss_fct.fonction_perte)

        ctk.CTkLabel(frame, text="Fonction de perte :", font=self.font_bouton).grid(row=0, column=0, sticky="w", pady=5, padx=5)
        ctk.CTkOptionMenu(frame, variable=self.Params_loss_fonction_perte, values=["MSE", "MAE", "Huber"]).grid(row=0, column=1, pady=5, padx=5, sticky="ew")

        frame.grid_columnconfigure(1, weight=1)

    def create_optimiseur_tab(self):
        frame = ctk.CTkFrame(self.tab_optimiseur, fg_color="#ffffff")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.Params_opt_optimisateur = tk.StringVar(value=Parametres_optimisateur.optimisateur)
        self.Params_opt_learning_rate = tk.DoubleVar(value=Parametres_optimisateur.learning_rate)
        self.Params_opt_decroissance = tk.DoubleVar(value=Parametres_optimisateur.decroissance)
        self.Params_opt_scheduler = tk.StringVar(value=Parametres_optimisateur.scheduler)
        self.Params_opt_patience = tk.IntVar(value=Parametres_optimisateur.patience)

        fields = [
            ("Optimisateur :", self.Params_opt_optimisateur, "menu", ["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta"]),
            ("Learning rate :", self.Params_opt_learning_rate, "entry"),
            ("Décroissance :", self.Params_opt_decroissance, "entry"),
            ("Scheduler :", self.Params_opt_scheduler, "menu", ["None", "Plateau", "Cosine", "OneCycle"]),
            ("Patience :", self.Params_opt_patience, "entry"),
        ]

        for i, field_info in enumerate(fields):
            if field_info[2] == "entry":
                ctk.CTkLabel(frame, text=field_info[0], font=self.font_bouton).grid(row=i, column=0, sticky="w", pady=5, padx=5)
                ctk.CTkEntry(frame, textvariable=field_info[1]).grid(row=i, column=1, pady=5, padx=5, sticky="ew")
            elif field_info[2] == "menu":
                ctk.CTkLabel(frame, text=field_info[0], font=self.font_bouton).grid(row=i, column=0, sticky="w", pady=5, padx=5)
                ctk.CTkOptionMenu(frame, variable=field_info[1], values=field_info[3]).grid(row=i, column=1, pady=5, padx=5, sticky="ew")

        frame.grid_columnconfigure(1, weight=1)

    def create_entrainement_tab(self):
        frame = ctk.CTkFrame(self.tab_entrainement, fg_color="#ffffff")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.Params_train_nb_epochs = tk.IntVar(value=Parametres_entrainement.nb_epochs)
        self.Params_train_batch_size = tk.IntVar(value=Parametres_entrainement.batch_size)

        fields = [
            ("Nombre d'époques :", self.Params_train_nb_epochs),
            ("Taille du batch :", self.Params_train_batch_size),
        ]

        for i, (label, var) in enumerate(fields):
            ctk.CTkLabel(frame, text=label, font=self.font_bouton).grid(row=i, column=0, sticky="w", pady=5, padx=5)
            ctk.CTkEntry(frame, textvariable=var).grid(row=i, column=1, pady=5, padx=5, sticky="ew")

        frame.grid_columnconfigure(1, weight=1)

    def create_visualisation_tab(self):
        frame = ctk.CTkFrame(self.tab_visualisation, fg_color="#ffffff")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(frame, text="Métriques à suivre :", font=self.font_bouton).pack(anchor="w", padx=5, pady=5)

        self.metriques_vars = {}
        for metric in ["loss", "MSE", "MAE"]:
            var = tk.BooleanVar(value=(metric in Parametres_visualisation_suivi.metriques))
            self.metriques_vars[metric] = var
            ctk.CTkCheckBox(frame, text=metric, variable=var, font=self.font_bouton).pack(anchor="w", padx=10, pady=2)

    def validate_int_fct(self, text):
        return text.isdigit() or text == ""

    def est_ouverte(self):
        return self.winfo_exists()

    def Save_quit(self):
        # Sauvegarder les paramètres d'architecture
        Parametres_choix_reseau_neurones.modele = self.Params_modele.get()

        if self.Params_modele.get() == "MLP":
            Parametres_archi_reseau_MLP.nb_couches = self.Params_archi_nb_couches.get()
            Parametres_archi_reseau_MLP.hidden_size = self.Params_archi_hidden_size.get()
            Parametres_archi_reseau_MLP.dropout_rate = self.Params_archi_dropout_rate.get()
            Parametres_archi_reseau_MLP.fonction_activation = self.Params_archi_fonction_activation.get()

        elif self.Params_modele.get() == "CNN":
            Parametres_archi_reseau_CNN.nb_couches = self.Params_archi_nb_couches.get()
            Parametres_archi_reseau_CNN.hidden_size = self.Params_archi_hidden_size.get()
            Parametres_archi_reseau_CNN.dropout_rate = self.Params_archi_dropout_rate.get()
            Parametres_archi_reseau_CNN.fonction_activation = self.Params_archi_fonction_activation.get()
            Parametres_archi_reseau_CNN.kernel_size = self.Params_archi_kernel_size.get()
            Parametres_archi_reseau_CNN.stride = self.Params_archi_stride.get()
            Parametres_archi_reseau_CNN.padding = self.Params_archi_padding.get()

        elif self.Params_modele.get() in ["LSTM", "GRU"]:
            Parametres_archi_reseau_LSTM.nb_couches = self.Params_archi_nb_couches.get()
            Parametres_archi_reseau_LSTM.hidden_size = self.Params_archi_hidden_size.get()
            Parametres_archi_reseau_LSTM.bidirectional = self.Params_archi_bidirectional.get()
            Parametres_archi_reseau_LSTM.batch_first = self.Params_archi_batch_first.get()

        # Sauvegarder les paramètres de fonction de perte
        Parametres_choix_loss_fct.fonction_perte = self.Params_loss_fonction_perte.get()

        # Sauvegarder les paramètres d'optimisateur
        Parametres_optimisateur.optimisateur = self.Params_opt_optimisateur.get()
        Parametres_optimisateur.learning_rate = self.Params_opt_learning_rate.get()
        Parametres_optimisateur.decroissance = self.Params_opt_decroissance.get()
        Parametres_optimisateur.scheduler = self.Params_opt_scheduler.get()
        Parametres_optimisateur.patience = self.Params_opt_patience.get()

        # Sauvegarder les paramètres d'entraînement
        Parametres_entrainement.nb_epochs = self.Params_train_nb_epochs.get()
        Parametres_entrainement.batch_size = self.Params_train_batch_size.get()

        # Sauvegarder les métriques
        Parametres_visualisation_suivi.metriques = [metric for metric, var in self.metriques_vars.items() if var.get()]

        self.destroy()

    def Quit(self):
        self.destroy()

# Créer la fenêtre de paramétrage de l'horizon
class Fenetre_Params_horizon(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        couleur_fond = "#d9d9d9"
        self.title("📅 Paramétrage de l'horizon")

        # Définir une police personnalisée
        self.font_titre = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 11)

        self.geometry("500x1")  # largeur fixe, hauteur minimale

        self.cadre = ctk.CTkFrame(self, fg_color=couleur_fond)
        self.cadre.pack(fill="both", expand=True, padx=30, pady=30)

        # Titre simulé
        ctk.CTkLabel(self.cadre, text="Paramètres", font=self.font_titre, text_color="black").pack(anchor="w", pady=(0, 10))

        # Cadre des paramètres
        self.CadreParams = ctk.CTkFrame(self.cadre, fg_color="#ffffff", corner_radius=10, border_width=3, border_color="#cccccc")
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
            ctk.CTkLabel(self.CadreParams, text=label, text_color="black").grid(row=i, column=0, sticky="w", pady=5, padx=15)
            ctk.CTkEntry(self.CadreParams, textvariable=var).grid(row=i, column=1, pady=10, padx=15, sticky="ew")

        # Dates
        ctk.CTkLabel(self.CadreParams, text="Date de début :", text_color="black").grid(row=3, column=0, sticky="w", pady=5, padx=15)
        ctk.CTkButton(self.CadreParams, textvariable=self.date_debut_str, command=self.ouvrir_calendrier_debut).grid(row=3, column=1, pady=10, padx=15, sticky="ew")

        ctk.CTkLabel(self.CadreParams, text="Date de fin :", text_color="black").grid(row=4, column=0, sticky="w", pady=5, padx=15)
        ctk.CTkButton(self.CadreParams, textvariable=self.date_fin_str, command=self.ouvrir_calendrier_fin).grid(row=4, column=1, pady=10, padx=15, sticky="ew")

        self.CadreParams.grid_columnconfigure(1, weight=1)

        # Boutons d'action
        ctk.CTkButton(
            self.cadre, text="💾 Sauvegarder la configuration", font=self.font_bouton,
            height=40, fg_color="#b4d9b2", text_color="#0f5132", hover_color="#a2c7a0",
            corner_radius=5, command=self.Save_quit
        ).pack(fill="x", pady=10)

        ctk.CTkButton(
            self.cadre, text="❌ Quitter", font=self.font_bouton,
            height=40, fg_color="#f7b2b2", text_color="#842029", hover_color="#e89b9b",
            corner_radius=5, command=self.destroy
        ).pack(fill="x", pady=(0, 10))

        self.update_idletasks()
        self.geometry(f"500x{self.winfo_reqheight()}")
    
    def est_ouverte(self):
        return self.winfo_exists()

    def validate_int_fct(self, text):
        return text.isdigit() or text == ""
    
    # Fonction locale : ouvrir calendrier debut
    def ouvrir_calendrier_debut(self):
        top = ctk.CTkToplevel(self)
        top.title("Sélectionner la date de début")
        try:
            date_obj = datetime.strptime(self.date_debut_str.get(), "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.today()
        cal = Calendar(top, selectmode='day', date_pattern='yyyy-mm-dd',
                    year=date_obj.year, month=date_obj.month, day=date_obj.day)
        cal.pack(padx=10, pady=10)
        ctk.CTkButton(top, text="Valider", command=lambda: (self.date_debut_str.set(cal.get_date()), top.destroy())).pack(pady=10)

    # Fonction locale : ouvrir calendrier fin
    def ouvrir_calendrier_fin(self):
        top = ctk.CTkToplevel(self)
        top.title("Sélectionner la date de fin")
        try:
            date_obj = datetime.strptime(self.date_fin_str.get(), "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.today()
        cal = Calendar(top, selectmode='day', date_pattern='yyyy-mm-dd',
                    year=date_obj.year, month=date_obj.month, day=date_obj.day)
        cal.pack(padx=10, pady=10)
        ctk.CTkButton(top, text="Valider", command=lambda: (self.date_fin_str.set(cal.get_date()), top.destroy())).pack(pady=10)

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
class Fenetre_Choix_datasets(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        couleur_fond = "#d9d9d9"
        self.title("📂 Choix des datasets")

        # Définir une police personnalisée
        self.font_titre = ("Helvetica", 14, "bold")
        self.font_bouton = ("Helvetica", 11)

        self.geometry("500x1")  # largeur fixe, hauteur minimale

        self.cadre = ctk.CTkFrame(self, fg_color=couleur_fond)
        self.cadre.pack(fill="both", expand=True, padx=30, pady=30)

        # Titre simulé
        ctk.CTkLabel(self.cadre, text="Choix des datasets", font=self.font_titre, text_color="black").pack(anchor="w", pady=(0, 10))

        # Cadre des paramètres
        self.CadreParams = ctk.CTkFrame(self.cadre, fg_color="#ffffff", corner_radius=10, border_width=3, border_color="#cccccc")
        self.CadreParams.pack(fill="both", expand=True, pady=(0, 20))

        self.Liste_datasets=["A","B","C","D","E","F","G","H","I","J","K","L","M"]  # Exemple de liste de datasets
        

        # Liste des champs
        ctk.CTkLabel(self.CadreParams, text="Sélectionnez un dataset :", font=self.font_bouton, text_color="black").pack(anchor="w", padx=15, pady=(15,5))

        # Créer une variable pour stocker la sélection
        self.dataset_selection = tk.StringVar()

        # Créer la Listbox (utilisation de tk.Listbox car CTk n'a pas de widget Listbox natif)
        listbox_frame = ctk.CTkFrame(self.CadreParams, fg_color="#f0f0f0", corner_radius=5)
        listbox_frame.pack(fill="both", expand=True, padx=15, pady=(5, 15))

        self.listbox_datasets = tk.Listbox(
            listbox_frame,
            listvariable=self.dataset_selection,
            height=6,
            selectmode="browse",
            font=self.font_bouton,
            bg="#f0f0f0",
            activestyle="dotbox"
        )
        self.listbox_datasets.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        scrollbar = ctk.CTkScrollbar(listbox_frame, command=self.listbox_datasets.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox_datasets.config(yscrollcommand=scrollbar.set)

        # Remplir la Listbox avec les noms des datasets
        for nom in self.Liste_datasets:
            self.listbox_datasets.insert(tk.END, nom)

        # Boutons d'action
        ctk.CTkButton(
            self.cadre, text="💾 Sauvegarder la configuration", font=self.font_bouton,
            height=40, fg_color="#b4d9b2", text_color="#0f5132", hover_color="#a2c7a0",
            corner_radius=5, command=self.Save_quit
        ).pack(fill="x", pady=10)
        ctk.CTkButton(
            self.cadre, text="❌ Quitter", font=self.font_bouton,
            height=40, fg_color="#f7b2b2", text_color="#842029", hover_color="#e89b9b",
            corner_radius=5, command=self.Quit
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
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    app = Fenetre_Acceuil()
    app.mainloop()
