import tkinter as tk
from tkcalendar import Calendar
from datetime import datetime
import requests, json
from tkinter import ttk
from tkinter import messagebox


import matplotlib
matplotlib.use("TkAgg")  # backend Tkinter
import matplotlib.pyplot as plt
import numpy as np
 
# URL = "http://192.168.1.94:8000" 
# URL = "http://192.168.27.66:8000"
# URL = "http://192.168.1.169:8000"
URL = "http://138.231.149.81:8000"


# Paramètres et variables

class Parametres_temporels_class():
    def __init__(self):
        self.horizon=1 # int
        self.dates=["2002-01-01", "2025-01-02"] # variable datetime
        self.pas_temporel=1 # int
        self.portion_decoupage=0.8# float entre 0 et 1
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
            self.batch_first=True # bool

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
        self.padding=1 # int
class Parametres_choix_loss_fct_class():
    def __init__(self):
        self.fonction_perte="MSE" # fonction MSE/MAE/Huber
        self.params=None # paramètres de la fonction perte (dépend de la fonction)
class Parametres_optimisateur_class():
    def __init__(self):
        self.optimisateur="Adam" # fonction Adam/SGD/RMSprop/Adagrad/Adadelta
        self.learning_rate=0.0001 # float
        self.decroissance=0.0 # float
        self.scheduler=None # fonction Plateau/Cosine/OneCycle/None
        self.patience=5 # int
class Parametres_entrainement_class():
    def __init__(self):
        self.nb_epochs=200 # int
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
        self.Payload={}
        self.Fenetre_Params_instance = None
        self.Fenetre_Params_horizon_instance = None
        self.Fenetre_Choix_datasets_instance = None
        self.feur_instance = None

        tk.Tk.__init__(self)
        self.title("🧠 Paramétrage du Réseau de Neuronnes")
        self.configure(bg="#f0f4f8")
        self.geometry("520x1")

        # Polices
        self.font_titre = ("Helvetica", 20, "bold")
        self.font_section = ("Helvetica", 18, "bold")
        self.font_bouton = ("Helvetica", 14)

        # Cadre principal
        self.cadre = tk.Frame(self, bg="#f0f4f8", padx=20, pady=20)
        self.cadre.pack(fill="both", expand=True)

        # Titre
        tk.Label(self.cadre, text="MLApp", font=self.font_titre, bg="#f0f4f8", fg="#2c3e50").pack(pady=(0, 20))

        # Section 1 : Modèle
        section_modele = tk.LabelFrame(self.cadre, text="🧬 Modèle", font=self.font_section, bg="#eaf2f8", fg="#34495e", padx=15, pady=10, bd=2, relief="groove")
        section_modele.pack(fill="x", pady=10)

        self.bouton(section_modele, "📂 Charger Modèle", self.test)
        self.bouton(section_modele, "⚙️ Paramétrer Modèle", self.Parametrer_modele)

        # Section 2 : Données
        section_data = tk.LabelFrame(self.cadre, text="📊 Données", font=self.font_section, bg="#eaf2f8", fg="#34495e", padx=15, pady=10, bd=2, relief="groove")
        section_data.pack(fill="x", pady=10)

        self.bouton(section_data, "📁 Choix Dataset", self.Parametrer_dataset)
        self.bouton(section_data, "📅 Paramétrer Horizon", self.Parametrer_horizon)

        # Section 3 : Actions
        section_actions = tk.Frame(self.cadre, bg="#f0f4f8")
        section_actions.pack(fill="x", pady=(20, 0))

        self.bouton(section_actions, "🚀 Envoyer la configuration au serveur", self.EnvoyerConfig, bg="#d4efdf", fg="#145a32")
        self.bouton(section_actions, "❌ Quitter", self.destroy, bg="#f5b7b1", fg="#641e16")

        self.update_idletasks()
        self.geometry(f"520x{self.winfo_reqheight()}")

        self.attributes('-fullscreen', True)  # Enable fullscreen
        self.bind("<Escape>", lambda event: self.attributes('-fullscreen', False))
        self.bind("<F11>", lambda event: self.attributes('-fullscreen', not self.attributes('-fullscreen')))


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
        self.config_specifique = {}
    
        if Parametres_choix_reseau_neurones.modele == "MLP":
            self.config_specifique = Parametres_archi_reseau_MLP.__dict__
        elif Parametres_choix_reseau_neurones.modele == "LSTM":
            self.config_specifique = Parametres_archi_reseau_LSTM.__dict__
        elif Parametres_choix_reseau_neurones.modele == "CNN":
            self.config_specifique = Parametres_archi_reseau_CNN.__dict__
        
        return self.config_specifique
    
    def plot_predictions(self, y_true_pairs, y_pred_pairs):
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

        # Configuration du style
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(12, 6))
        
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
        
        # Améliorer l'apparence générale
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        
        # Ajuster les marges
        fig.tight_layout()
        
        # Afficher
        plt.show()



    def EnvoyerConfig(self):
        payload_global = self.Formatter_JSON_global()
        payload_model = self.Formatter_JSON_specif()

        print("Payload envoyé au serveur :", {"payload": payload_global, "payload_model": payload_model})

        # --- buffers pour le plot et métriques ---
        y_true_pairs, y_pred_pairs = [], []
        test_metrics = None
        train_losses = []
        train_epochs = []
        
        # --- Fenêtre de progression avec graphique dynamique ---
        fenetre_progress = tk.Toplevel(self)
        fenetre_progress.title("📈 Entraînement en cours...")
        fenetre_progress.geometry("900x650")
        fenetre_progress.configure(bg="#f5f7fa")
        
        cadre_progress = tk.Frame(fenetre_progress, padx=20, pady=20, bg="#f5f7fa")
        cadre_progress.pack(fill="both", expand=True)
        
        # Labels de statut
        label_status = tk.Label(
            cadre_progress, 
            text="Initialisation...", 
            font=("Helvetica", 14, "bold"),
            bg="#f5f7fa",
            fg="#2c3e50"
        )
        label_status.pack(pady=10)
        
        cadre_info = tk.Frame(cadre_progress, bg="#f5f7fa")
        cadre_info.pack(pady=5)
        
        label_epoch = tk.Label(
            cadre_info, 
            text="", 
            font=("Helvetica", 11),
            bg="#f5f7fa",
            fg="#34495e"
        )
        label_epoch.grid(row=0, column=0, padx=20)
        
        label_loss = tk.Label(
            cadre_info, 
            text="", 
            font=("Helvetica", 11),
            bg="#f5f7fa",
            fg="#34495e"
        )
        label_loss.grid(row=0, column=1, padx=20)
        
        progress_bar = ttk.Progressbar(cadre_progress, length=800, mode='determinate')
        progress_bar.pack(pady=15)
        
        # === GRAPHIQUE MATPLOTLIB INTÉGRÉ AVEC STYLE ===
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        
        # Créer la figure
        fig = Figure(figsize=(8, 4), dpi=100, facecolor='#f5f7fa')
        ax = fig.add_subplot(111)
        
        # Style du graphique
        ax.set_facecolor('#ffffff')
        ax.set_title('Évolution de la Loss pendant l\'entraînement', 
                    fontsize=13, 
                    fontweight='bold',
                    color='#2c3e50',
                    pad=15)
        ax.set_xlabel('Epoch', fontsize=10, fontweight='bold', color='#34495e')
        ax.set_ylabel('Loss', fontsize=10, fontweight='bold', color='#34495e')
        ax.grid(True, linestyle='--', alpha=0.3, color='#bdc3c7')
        ax.set_axisbelow(True)
        
        # Supprimer les bordures supérieure et droite
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#95a5a6')
        ax.spines['bottom'].set_color('#95a5a6')
        
        # Ligne de loss (vide au début)
        line_loss, = ax.plot([], [], 
                            color='#e74c3c', 
                            linewidth=2.5, 
                            marker='o', 
                            markersize=5,
                            markerfacecolor='#ffffff',
                            markeredgewidth=2,
                            markeredgecolor='#e74c3c',
                            label='Loss',
                            alpha=0.9)
        
        # Légende
        legend = ax.legend(loc='upper right', 
                        frameon=True, 
                        fancybox=True, 
                        shadow=True,
                        fontsize=9)
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.9)
        
        fig.tight_layout()
        
        # Intégrer le graphique dans Tkinter
        canvas = FigureCanvasTkAgg(fig, master=cadre_progress)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10)
        
        # Log textuel
        text_log = tk.Text(
            cadre_progress, 
            height=6, 
            width=100, 
            font=("Courier", 9),
            bg="#f8f9fa",
            fg="#2c3e50",
            relief="flat",
            padx=10,
            pady=10
        )
        text_log.pack(pady=10)
        
        scrollbar = tk.Scrollbar(cadre_progress, command=text_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_log.config(yscrollcommand=scrollbar.set)

        def log_message(msg, color="black"):
            text_log.insert(tk.END, msg + "\n", color)
            text_log.see(tk.END)
            fenetre_progress.update()
        
        # Configuration des tags de couleur
        text_log.tag_config("black", foreground="#2c3e50")
        text_log.tag_config("blue", foreground="#3498db")
        text_log.tag_config("green", foreground="#27ae60")
        text_log.tag_config("red", foreground="#e74c3c")
        text_log.tag_config("orange", foreground="#e67e22")
        
        # Variable pour stocker le nombre total d'epochs
        total_epochs = Parametres_entrainement.nb_epochs
        
        def update_loss_plot():
            """Mise à jour DYNAMIQUE du graphique de loss"""
            if train_epochs and train_losses:
                # Exclure la première valeur pour le zoom
                epochs_to_plot = train_epochs[1:] if len(train_epochs) > 1 else train_epochs
                losses_to_plot = train_losses[1:] if len(train_losses) > 1 else train_losses
                
                if not epochs_to_plot or not losses_to_plot:
                    return
                
                # Mettre à jour les données de la ligne
                line_loss.set_data(epochs_to_plot, losses_to_plot)
                
                # Ajuster les limites des axes
                ax.relim()
                ax.autoscale_view()
                
                # Ajuster les limites Y pour bien voir la courbe
                if len(losses_to_plot) > 0:
                    y_min = min(losses_to_plot)
                    y_max = max(losses_to_plot)
                    y_range = y_max - y_min
                    if y_range > 0:
                        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                    else:
                        # Si toutes les valeurs sont identiques
                        ax.set_ylim(y_min - 0.001, y_max + 0.001)
                
                # Redessiner le canvas
                canvas.draw()
                canvas.flush_events()
                fenetre_progress.update()

        try:
            with requests.post(
                f"{URL}/train_full", 
                json={"payload": payload_global, "payload_model": payload_model}, 
                stream=True, 
                timeout=300
            ) as r:
                r.raise_for_status()
                
                for line in r.iter_lines():
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        msg = json.loads(line[6:].decode("utf-8"))
                        msg_type = msg.get("type")
                        
                        print(f"EVENT: {msg}")  # Debug
                        
                        # ========== PHASE SPLIT ==========
                        if msg_type == "split":
                            label_status.config(text="📊 Découpage des données...", fg="#3498db")
                            log_message(f"✓ Train: {msg['n_train']} échantillons | Test: {msg['n_test']} échantillons", "blue")
                        
                        # ========== PHASE ENTRAINEMENT ==========
                        elif msg_type == "train_start":
                            label_status.config(text="🚀 Entraînement démarré", fg="#27ae60")
                            total_epochs = msg.get('epochs', Parametres_entrainement.nb_epochs)  # Priorité au serveur
                            log_message(f"🚀 Début de l'entraînement sur {msg.get('n_samples', '?')} échantillons pour {total_epochs} epochs", "green")
                            fenetre_progress.update()
                        
                        # ========== GÉRER VOS ÉVÉNEMENTS ACTUELS ==========
                        # Vos événements ont 'epochs' et 'avg_loss'
                        elif 'epochs' in msg and 'avg_loss' in msg:
                            epoch = msg.get("epochs", 0)
                            loss = msg.get("avg_loss", 0.0)
                            
                            # 🔥 AJOUTER LE POINT À LA COURBE 🔥
                            train_epochs.append(epoch)
                            train_losses.append(loss)
                            
                            # Mise à jour des labels
                            label_epoch.config(text=f"📊 Epoch: {epoch}/{total_epochs}", fg="#3498db")
                            
                            # Couleur dynamique selon la loss
                            loss_color = "#27ae60" if loss < 0.001 else "#e67e22" if loss < 0.01 else "#e74c3c"
                            label_loss.config(text=f"📉 Loss: {loss:.6f}", fg=loss_color)
                            
                            progress_bar['value'] = (epoch / total_epochs) * 100
                            
                            log_message(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.6f}", "black")
                            
                            # 🔥 MISE À JOUR DYNAMIQUE DU GRAPHIQUE 🔥
                            update_loss_plot()
                        
                        # ========== FORMAT STANDARD (si vous corrigez le serveur) ==========
                        elif msg_type == "train_epoch":
                            epoch = msg.get("epoch", 0)
                            total_epochs_msg = msg.get("total_epochs", total_epochs)
                            loss = msg.get("loss", 0.0)
                            
                            train_epochs.append(epoch)
                            train_losses.append(loss)
                            
                            label_epoch.config(text=f"📊 Epoch: {epoch}/{total_epochs_msg}", fg="#3498db")
                            loss_color = "#27ae60" if loss < 0.001 else "#e67e22" if loss < 0.01 else "#e74c3c"
                            label_loss.config(text=f"📉 Loss: {loss:.6f}", fg=loss_color)
                            progress_bar['value'] = (epoch / total_epochs_msg) * 100
                            
                            log_message(f"Epoch {epoch}/{total_epochs_msg} - Loss: {loss:.6f}", "black")
                            update_loss_plot()
                        
                        elif msg_type == "train_progress":
                            progress = msg.get("progress", 0)
                            progress_bar['value'] = progress
                            fenetre_progress.update()
                        
                        elif msg_type == "train_done":
                            label_status.config(text="✅ Entraînement terminé", fg="#27ae60")
                            log_message("✅ Entraînement terminé avec succès!", "green")
                            progress_bar['value'] = 100
                            fenetre_progress.update()
                        
                        # ========== PHASE TEST ==========
                        elif msg_type == "test_start":
                            label_status.config(text="🧪 Test en cours...", fg="#3498db")
                            log_message(f"🧪 Début du test sur {msg['n_test']} échantillons", "blue")
                            progress_bar['value'] = 0
                            label_epoch.config(text="")
                            label_loss.config(text="")
                            fenetre_progress.update()
                        
                        elif msg_type == "test_pair":
                            y_true_pairs.append(msg["y"])
                            y_pred_pairs.append(msg["yhat"])
                        
                        elif msg_type == "test_progress":
                            done = msg.get("done", 0)
                            total = msg.get("total", 1)
                            progress = (done / total) * 100
                            progress_bar['value'] = progress
                            label_status.config(text=f"🧪 Test: {done}/{total}", fg="#3498db")
                            fenetre_progress.update()
                        
                        elif msg_type == "test_final":
                            test_metrics = msg["metrics"]
                            label_status.config(text="✅ Test terminé", fg="#27ae60")
                            log_message("✅ Test terminé avec succès!", "green")
                            
                            # Afficher les métriques dans le log
                            log_message("\n=== MÉTRIQUES FINALES ===", "green")
                            for metric, value in test_metrics["overall_mean"].items():
                                if value is not None:
                                    log_message(f"  {metric}: {value:.6f}", "blue")
                            
                            progress_bar['value'] = 100
                            fenetre_progress.update()
                        
                        # ========== ERREURS ==========
                        elif msg_type == "error":
                            error_msg = msg.get('message', 'Erreur inconnue')
                            label_status.config(text="❌ Erreur", fg="#e74c3c")
                            log_message(f"❌ ERREUR: {error_msg}", "red")
                            messagebox.showerror("Erreur", error_msg)
                            break
                        
                        elif msg_type == "warn":
                            log_message(f"⚠️ ATTENTION: {msg.get('message', '')}", "orange")
                            fenetre_progress.update()

        except requests.exceptions.Timeout:
            log_message("❌ Timeout: Le serveur ne répond pas", "red")
            messagebox.showerror("Timeout", "Le serveur met trop de temps à répondre")
        except Exception as e:
            log_message(f"❌ Exception: {str(e)}", "red")
            messagebox.showerror("Erreur de connexion", f"Impossible de communiquer avec le serveur:\n{str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Bouton pour fermer la fenêtre
            btn_fermer = tk.Button(
                cadre_progress,
                text="Fermer",
                command=fenetre_progress.destroy,
                font=("Helvetica", 11),
                bg="#3498db",
                fg="white",
                padx=20,
                pady=5
            )
            btn_fermer.pack(pady=10)
            
            # Afficher le graphique final des prédictions
            if y_true_pairs and y_pred_pairs:
                fenetre_progress.after(1000, lambda: self.plot_predictions(y_true_pairs, y_pred_pairs))
            
            # Afficher le tableau récapitulatif des métriques
            if test_metrics:
                fenetre_progress.after(1500, lambda: self.afficher_tableau_metriques(test_metrics, train_losses))

    def afficher_tableau_metriques(self, metrics, train_losses):
        """
        Affiche un tableau récapitulatif des métriques dans une nouvelle fenêtre Tkinter
        """
        fenetre_metrics = tk.Toplevel(self)
        fenetre_metrics.title("📊 Résultats Finaux")
        fenetre_metrics.geometry("700x500")
        
        # Cadre principal
        cadre = tk.Frame(fenetre_metrics, padx=20, pady=20, bg="#f0f4f8")
        cadre.pack(fill="both", expand=True)
        
        # Titre
        tk.Label(
            cadre, 
            text="📈 Métriques de Performance", 
            font=("Helvetica", 16, "bold"),
            bg="#f0f4f8",
            fg="#2c3e50"
        ).pack(pady=(0, 20))
        
        # Cadre pour les métriques globales
        cadre_global = tk.LabelFrame(
            cadre, 
            text="Métriques Globales", 
            font=("Helvetica", 12, "bold"),
            bg="#ffffff",
            padx=15,
            pady=15
        )
        cadre_global.pack(fill="x", pady=10)
        
        # Afficher les métriques globales
        overall = metrics["overall_mean"]
        row = 0
        for metric, value in overall.items():
            tk.Label(
                cadre_global, 
                text=f"{metric}:", 
                font=("Helvetica", 11, "bold"),
                bg="#ffffff",
                anchor="w"
            ).grid(row=row, column=0, sticky="w", pady=5, padx=5)
            
            value_text = f"{value:.6f}" if value is not None else "N/A"
            tk.Label(
                cadre_global, 
                text=value_text, 
                font=("Helvetica", 11),
                bg="#ffffff",
                fg="#27ae60" if value is not None else "#e74c3c"
            ).grid(row=row, column=1, sticky="e", pady=5, padx=5)
            row += 1
        
        # Cadre pour les métriques par dimension
        if metrics["per_dim"]["MSE"] and len(metrics["per_dim"]["MSE"]) > 1:
            cadre_dims = tk.LabelFrame(
                cadre, 
                text="Métriques par Dimension", 
                font=("Helvetica", 12, "bold"),
                bg="#ffffff",
                padx=15,
                pady=15
            )
            cadre_dims.pack(fill="both", expand=True, pady=10)
            
            # Créer un tableau
            text_dims = tk.Text(cadre_dims, height=6, width=60, font=("Courier", 10))
            text_dims.pack(fill="both", expand=True)
            
            # En-têtes
            text_dims.insert(tk.END, f"{'Dim':<5} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'R²':<12}\n")
            text_dims.insert(tk.END, "-" * 60 + "\n")
            
            # Données
            for i in range(len(metrics["per_dim"]["MSE"])):
                mse_val = metrics["per_dim"]["MSE"][i]
                mae_val = metrics["per_dim"]["MAE"][i]
                rmse_val = metrics["per_dim"]["RMSE"][i]
                r2_val = metrics["per_dim"]["R2"][i]
                
                r2_str = f"{r2_val:.6f}" if r2_val is not None else "N/A"
                text_dims.insert(tk.END, f"{i:<5} {mse_val:<12.6f} {mae_val:<12.6f} {rmse_val:<12.6f} {r2_str:<12}\n")
            
            text_dims.config(state=tk.DISABLED)
        
        # Informations supplémentaires
        info_text = f"Nombre d'échantillons de test: {metrics.get('n_test', 'N/A')}\n"
        info_text += f"Dimensions: {metrics.get('dims', 'N/A')}\n"
        if train_losses:
            info_text += f"Loss finale (entraînement): {train_losses[-1]:.6f}"
        
        tk.Label(
            cadre,
            text=info_text,
            font=("Helvetica", 10),
            bg="#f0f4f8",
            fg="#7f8c8d",
            justify="left"
        ).pack(pady=10)
        
        # Bouton pour fermer
        tk.Button(
            cadre,
            text="Fermer",
            font=("Helvetica", 11),
            command=fenetre_metrics.destroy,
            bg="#e74c3c",
            fg="white",
            padx=20,
            pady=5
        ).pack(pady=10)


##############
# Créer la fenêtre de paramétrage du modèle
class Fenetre_Params(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
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

    def est_ouverte(self):
        return self.winfo_exists()
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

#######################
############################ Faut modif ici modele par modele c.f. le fichier classes.py
############################
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