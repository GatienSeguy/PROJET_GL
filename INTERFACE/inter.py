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
from tkinter import ttk

matplotlib.use("TkAgg")

# Configuration du thème sobre
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Couleurs style IRMA Conseil
BG_DARK = "#2c3e50"
BG_CARD = "#34495e"
BG_INPUT = "#273747"
TEXT_PRIMARY = "#ecf0f1"
TEXT_SECONDARY = "#bdc3c7"
ACCENT_PRIMARY = "#e74c3c"
ACCENT_SECONDARY = "#3498db"
BORDER_COLOR = "#4a5f7f"

URL = "http://138.231.149.81:8000"

# Classes de paramètres (identiques à l'original)
class Parametres_temporels_class():
    def __init__(self):
        self.horizon = 1
        self.dates = ["2001-01-01", "2025-01-02"]
        self.pas_temporel = 1
        self.portion_decoupage = 0.8
    
    def generate_json(self, json):
        pass

class Parametres_choix_reseau_neurones_class():
    def __init__(self):
        self.modele = "MLP"

class Parametres_archi_reseau_class():
    class MLP_params():
        def __init__(self):
            self.nb_couches = 2
            self.hidden_size = 64
            self.dropout_rate = 0.0
            self.fonction_activation = "ReLU"
    
    class CNN_params():
        def __init__(self):
            self.nb_couches = 2
            self.hidden_size = 64
            self.dropout_rate = 0.0
            self.fonction_activation = "ReLU"
            self.kernel_size = 3
            self.stride = 1
            self.padding = 0
    
    class LSTM_params():
        def __init__(self):
            self.nb_couches = 2
            self.hidden_size = 64
            self.bidirectional = False
            self.batch_first = False

    def __init__(self):
        self.nb_couches = 2
        self.hidden_size = 64
        self.dropout_rate = 0.0
        self.fonction_activation = "ReLU"
        self.bidirectional = False
        self.batch_first = False
        self.kernel_size = 3
        self.stride = 1
        self.padding = 0

class Parametres_choix_loss_fct_class():
    def __init__(self):
        self.fonction_perte = "MSE"
        self.params = None

class Parametres_optimisateur_class():
    def __init__(self):
        self.optimisateur = "Adam"
        self.learning_rate = 0.001
        self.decroissance = 0.0
        self.scheduler = "None"
        self.patience = 5

class Parametres_entrainement_class():
    def __init__(self):
        self.nb_epochs = 1000
        self.batch_size = 4
        self.clip_gradient = None

class Parametres_visualisation_suivi_class():
    def __init__(self):
        self.metriques = ["loss"]

# Instances globales
Parametres_temporels = Parametres_temporels_class()
Parametres_choix_reseau_neurones = Parametres_choix_reseau_neurones_class()
Parametres_archi_reseau_MLP = Parametres_archi_reseau_class.MLP_params()
Parametres_archi_reseau_CNN = Parametres_archi_reseau_class.CNN_params()
Parametres_archi_reseau_LSTM = Parametres_archi_reseau_class.LSTM_params()
Parametres_choix_loss_fct = Parametres_choix_loss_fct_class()
Parametres_optimisateur = Parametres_optimisateur_class()
Parametres_entrainement = Parametres_entrainement_class()
Parametres_visualisation_suivi = Parametres_visualisation_suivi_class()

# Cadre d'entraînement avec toutes les fonctionnalités de l'original
class Cadre_Entrainement(ctk.CTkFrame):
    def __init__(self, app, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.master_app = app
        
        # Variables identiques à l'original
        self.epochs = []
        self.losses = []
        self.data_queue = queue.Queue()
        self.is_training = False
        self.is_log = tk.BooleanVar(value=False)
        
        # Container
        container = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Titre
        title = ctk.CTkLabel(
            container,
            text="Training Progress",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        title.pack(pady=(15, 10))
        
        # Barre de progression
        self.progress_bar = ctk.CTkProgressBar(
            container,
            height=8,
            corner_radius=4,
            progress_color=ACCENT_PRIMARY
        )
        
        # Info frame
        info_frame = ctk.CTkFrame(container, fg_color="transparent")
        info_frame.pack(fill="x", padx=20, pady=10)
        
        self.label_epoch = ctk.CTkLabel(
            info_frame,
            text="Epoch: -",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY
        )
        self.label_epoch.pack(side="left", padx=10)
        
        self.label_loss = ctk.CTkLabel(
            info_frame,
            text="Loss: -",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY
        )
        self.label_loss.pack(side="left", padx=10)
        
        self.label_status = ctk.CTkLabel(
            info_frame,
            text="⏸️ Ready",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY
        )
        self.label_status.pack(side="right", padx=10)
        
        # Graphique matplotlib
        graph_container = ctk.CTkFrame(container, fg_color=BG_INPUT, corner_radius=4)
        graph_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.fig = Figure(figsize=(10, 5), facecolor=BG_INPUT)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(BG_INPUT)
        self.ax.grid(True, linestyle='--', alpha=0.2, color='#7f8c8d')
        self.ax.set_xlabel('Epoch', fontsize=10, color=TEXT_SECONDARY)
        self.ax.set_ylabel('Loss', fontsize=10, color=TEXT_SECONDARY)
        self.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        self.ax.spines['bottom'].set_color(BORDER_COLOR)
        self.ax.spines['top'].set_color(BORDER_COLOR)
        self.ax.spines['right'].set_color(BORDER_COLOR)
        self.ax.spines['left'].set_color(BORDER_COLOR)
        
        self.line, = self.ax.plot([], [], '-', linewidth=2, color=ACCENT_PRIMARY)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Checkbox log scale
        self.log_check = ctk.CTkCheckBox(
            container,
            text="Log Scale",
            variable=self.is_log,
            command=self.Log_scale,
            fg_color=ACCENT_PRIMARY,
            hover_color=ACCENT_SECONDARY,
            text_color=TEXT_SECONDARY
        )
        self.log_check.pack(pady=10)
        
        self.fig.tight_layout()
    
    def Log_scale(self):
        if hasattr(self, 'Log_scale_possible'):
            self.ax.set_yscale('log' if self.is_log.get() else 'linear')
            if len(self.losses) > 1:
                y_min, y_max = min(self.losses), max(self.losses)
                if self.is_log.get():
                    ratio = (y_max / y_min) ** 0.1
                    self.ax.set_ylim(y_min / ratio, y_max * ratio)
                else:
                    y_range = y_max - y_min
                    if y_range > 0:
                        self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            self.canvas.draw()
    
    def start_training(self):
        self.is_training = True
        self.epochs = []
        self.losses = []
        self.total_epochs = Parametres_entrainement.nb_epochs
        
        self.progress_bar.set(0)
        self.progress_bar.pack(before=self.label_epoch.master, padx=20, pady=10)
        
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        self.ax.clear()
        self.ax.set_facecolor(BG_INPUT)
        self.ax.grid(True, linestyle='--', alpha=0.2, color='#7f8c8d')
        self.ax.set_xlabel('Epoch', fontsize=10, color=TEXT_SECONDARY)
        self.ax.set_ylabel('Loss', fontsize=10, color=TEXT_SECONDARY)
        self.ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        self.line, = self.ax.plot([], [], '-', linewidth=2, color=ACCENT_PRIMARY)
        
        self.label_status.configure(text="🚀 Training...", text_color="#27ae60")
        self.canvas.draw()
        
        self.update_plot()
    
    def add_data_point(self, epoch, loss):
        self.data_queue.put((epoch, loss))
    
    def update_plot(self):
        self.Log_scale_possible = True
        if not self.is_training:
            return
        
        updated = False
        while not self.data_queue.empty():
            try:
                epoch, loss = self.data_queue.get_nowait()
                self.epochs.append(epoch)
                self.losses.append(loss)
                updated = True
                
                self.label_epoch.configure(text=f"Epoch: {epoch}")
                self.label_loss.configure(text=f"Loss: {loss:.6f}")
                self.progress_bar.set((epoch / self.total_epochs))
            except queue.Empty:
                break
        
        if updated and len(self.epochs) > 0:
            self.line.set_data(self.epochs, self.losses)
            self.ax.set_yscale('log' if self.is_log.get() else 'linear')
            
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            
            if len(self.losses) > 1:
                y_min, y_max = min(self.losses), max(self.losses)
                if self.is_log.get():
                    ratio = (y_max / y_min) ** 0.1
                    self.ax.set_ylim(y_min / ratio, y_max * ratio)
                else:
                    y_range = y_max - y_min
                    if y_range > 0:
                        self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            self.canvas.draw()
        
        if self.is_training:
            self.after(100, self.update_plot)
    
    def stop_training(self):
        while not self.data_queue.empty():
            try:
                epoch, loss = self.data_queue.get_nowait()
                self.epochs.append(epoch)
                self.losses.append(loss)
                self.label_epoch.configure(text=f"Epoch: {epoch}")
                self.label_loss.configure(text=f"Loss: {loss:.6f}")
                self.progress_bar.set((epoch / self.total_epochs))
            except queue.Empty:
                break
        
        if len(self.epochs) > 0:
            self.line.set_data(self.epochs, self.losses)
            self.ax.set_yscale('log' if self.is_log.get() else 'linear')
            self.ax.relim()
            self.ax.autoscale_view(True, True, True)
            
            if len(self.losses) > 1:
                y_min, y_max = min(self.losses), max(self.losses)
                if self.is_log.get():
                    ratio = (y_max / y_min) ** 0.1
                    self.ax.set_ylim(y_min / ratio, y_max * ratio)
                else:
                    y_range = y_max - y_min
                    if y_range > 0:
                        self.ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            self.canvas.draw()
        
        self.is_training = False
        self.progress_bar.pack_forget()
        self.label_status.configure(text="✅ Completed", text_color="#27ae60")
        
        if len(self.losses) > 0:
            final_loss = self.losses[-1]
            min_loss = min(self.losses)
            self.label_loss.configure(text=f"Final Loss: {final_loss:.6f} (min: {min_loss:.6f})")

# Cadre Testing complet
class Cadre_Testing(ctk.CTkFrame):
    def __init__(self, app, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.master_app = app
        
        container = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        container.pack(fill="both", expand=True, padx=15, pady=15)
        
        title = ctk.CTkLabel(
            container,
            text="Testing Results",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        title.pack(pady=(15, 10))
        
        self.content_frame = ctk.CTkFrame(container, fg_color="transparent")
        self.content_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    def save_figure(self, fig):
        file_path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All files", "*.*")],
            title="Save Figure"
        )
        if file_path:
            fig.savefig(file_path)
    
    def plot_predictions(self, y_true_pairs, y_pred_pairs):
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        if not y_true_pairs or not y_pred_pairs:
            return
        
        yt = np.array(y_true_pairs, dtype=float)
        yp = np.array(y_pred_pairs, dtype=float)
        
        if yt.ndim == 2 and yt.shape[1] == 1:
            yt = yt.squeeze(1)
            yp = yp.squeeze(1)
        
        graph_container = ctk.CTkFrame(self.content_frame, fg_color=BG_INPUT, corner_radius=4)
        graph_container.pack(fill="both", expand=True)
        
        fig = Figure(figsize=(10, 5), facecolor=BG_INPUT)
        ax = fig.add_subplot(111)
        ax.set_facecolor(BG_INPUT)
        ax.grid(True, linestyle='--', alpha=0.2, color='#7f8c8d')
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
        ax.spines['bottom'].set_color(BORDER_COLOR)
        ax.spines['top'].set_color(BORDER_COLOR)
        ax.spines['right'].set_color(BORDER_COLOR)
        ax.spines['left'].set_color(BORDER_COLOR)
        
        if yt.ndim == 1:
            x = np.arange(len(yt))
            ax.plot(x, yt, color='#3498db', linewidth=2, marker='o', markersize=4, 
                   label='True', alpha=0.8)
            ax.plot(x, yp, color=ACCENT_PRIMARY, linewidth=2, marker='s', markersize=4, 
                   label='Predicted', alpha=0.8)
            ax.set_xlabel('Sample', fontsize=10, color=TEXT_SECONDARY)
            ax.set_ylabel('Value', fontsize=10, color=TEXT_SECONDARY)
            ax.legend(facecolor=BG_INPUT, edgecolor=BORDER_COLOR, 
                     labelcolor=TEXT_SECONDARY, framealpha=0.9)
        else:
            D = yt.shape[1]
            cols = min(3, D)
            rows = (D + cols - 1) // cols
            
            for d in range(D):
                ax_sub = fig.add_subplot(rows, cols, d + 1)
                ax_sub.set_facecolor(BG_INPUT)
                ax_sub.grid(True, linestyle='--', alpha=0.2)
                x = np.arange(len(yt))
                ax_sub.plot(x, yt[:, d], color='#3498db', linewidth=1.5, label='True')
                ax_sub.plot(x, yp[:, d], color=ACCENT_PRIMARY, linewidth=1.5, label='Pred')
                ax_sub.set_title(f'Dim {d+1}', fontsize=9, color=TEXT_SECONDARY)
                ax_sub.tick_params(colors=TEXT_SECONDARY, labelsize=8)
                if d == 0:
                    ax_sub.legend(fontsize=8, facecolor=BG_INPUT, edgecolor=BORDER_COLOR)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=graph_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        # Bouton de sauvegarde
        save_btn = ctk.CTkButton(
            self.content_frame,
            text="Save Figure",
            command=lambda: self.save_figure(fig),
            fg_color=ACCENT_PRIMARY,
            hover_color="#c0392b",
            height=32
        )
        save_btn.pack(pady=10)

# Cadre Metrics
class Cadre_Metrics(ctk.CTkFrame):
    def __init__(self, app, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.master_app = app
        
        container = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        container.pack(fill="both", expand=True, padx=15, pady=15)
        
        title = ctk.CTkLabel(
            container,
            text="Metrics",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        title.pack(pady=(15, 10))
        
        self.metrics_container = ctk.CTkFrame(container, fg_color="transparent")
        self.metrics_container.pack(fill="both", expand=True, padx=20, pady=10)
    
    def afficher_Metrics(self, metrics_dict):
        for widget in self.metrics_container.winfo_children():
            widget.destroy()
        
        row = 0
        col = 0
        for key, value in metrics_dict.items():
            metric_card = ctk.CTkFrame(self.metrics_container, fg_color=BG_INPUT, corner_radius=8)
            metric_card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            name_label = ctk.CTkLabel(
                metric_card,
                text=key,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=TEXT_SECONDARY
            )
            name_label.pack(pady=(15, 5))
            
            value_label = ctk.CTkLabel(
                metric_card,
                text=f"{value:.6f}" if isinstance(value, (int, float)) else str(value),
                font=ctk.CTkFont(size=20, weight="bold"),
                text_color=ACCENT_PRIMARY
            )
            value_label.pack(pady=(5, 15))
            
            col += 1
            if col >= 3:
                col = 0
                row += 1
        
        for i in range(3):
            self.metrics_container.grid_columnconfigure(i, weight=1)

# Cadre Prediction
class Cadre_Prediction(ctk.CTkFrame):
    def __init__(self, app, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.master_app = app
        
        container = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=8)
        container.pack(fill="both", expand=True, padx=15, pady=15)
        
        title = ctk.CTkLabel(
            container,
            text="Predictions",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        title.pack(pady=(15, 10))

# Fenêtre principale style IRMA Conseil
class Fenetre_Acceuil(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.stop_training = False
        self.Payload = {}
        self.Fenetre_Params_instance = None
        self.Fenetre_Params_horizon_instance = None
        self.Fenetre_Choix_datasets_instance = None
        
        self.title("IRMA Conseil - Model Training Dashboard")
        self.geometry("1400x900")
        
        # Barre de navigation
        nav_bar = ctk.CTkFrame(self, height=60, fg_color=BG_DARK, corner_radius=0)
        nav_bar.pack(fill="x", side="top")
        
        logo_label = ctk.CTkLabel(
            nav_bar,
            text="🔴 IRMA Conseil",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        logo_label.pack(side="left", padx=20, pady=15)
        
        # Menu navigation
        nav_items = ["Dashboard", "Models", "Simulation"]
        for item in nav_items:
            btn = ctk.CTkButton(
                nav_bar,
                text=item,
                fg_color="transparent",
                text_color=TEXT_SECONDARY,
                hover_color=BG_CARD,
                width=100,
                height=30
            )
            btn.pack(side="right", padx=5, pady=15)
        
        # Container principal
        main_container = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=0)
        main_container.pack(fill="both", expand=True)
        
        # Titre de la page
        title_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        title_frame.pack(fill="x", padx=30, pady=(20, 10))
        
        page_title = ctk.CTkLabel(
            title_frame,
            text="Model Training Dashboard",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=TEXT_PRIMARY,
            anchor="w"
        )
        page_title.pack(side="left")
        
        subtitle = ctk.CTkLabel(
            title_frame,
            text="Configure, train, and monitor AI prediction models for temporal data analysis",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_SECONDARY,
            anchor="w"
        )
        subtitle.pack(side="left", padx=(20, 0))
        
        # Layout en grille
        content_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=30, pady=(0, 20))
        
        content_frame.grid_columnconfigure(0, weight=2, minsize=400)
        content_frame.grid_columnconfigure(1, weight=3)
        content_frame.grid_rowconfigure(0, weight=1)
        
        # Panneau gauche - Configuration
        left_panel = self.create_config_panel(content_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        
        # Panneau droit - Résultats
        right_panel = self.create_results_panel(content_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")
    
    def create_config_panel(self, parent):
        panel = ctk.CTkScrollableFrame(parent, fg_color=BG_CARD, corner_radius=8)
        
        # Model Configuration
        config_title = ctk.CTkLabel(
            panel,
            text="Model Configuration",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        config_title.pack(anchor="w", padx=20, pady=(20, 5))
        
        load_btn = ctk.CTkButton(
            panel,
            text="Load Existing",
            fg_color="transparent",
            border_width=1,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            hover_color=BG_INPUT,
            height=36,
            command=self.test
        )
        load_btn.pack(fill="x", padx=20, pady=(0, 15))
        
        # Model Name
        self.create_input_field(panel, "Model Name", "Enter model name")
        
        # Model Type
        model_label = ctk.CTkLabel(panel, text="Model Type", text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=12))
        model_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        self.model_menu = ctk.CTkOptionMenu(
            panel,
            values=["MLP", "LSTM", "GRU", "CNN"],
            fg_color=BG_INPUT,
            button_color=BG_INPUT,
            button_hover_color=BORDER_COLOR,
            dropdown_fg_color=BG_INPUT,
            text_color=TEXT_PRIMARY,
            height=36,
            command=self.on_model_change
        )
        self.model_menu.set(Parametres_choix_reseau_neurones.modele)
        self.model_menu.pack(fill="x", padx=20, pady=(0, 10))
        
        # Loss Function & Optimizer en ligne
        row_frame = ctk.CTkFrame(panel, fg_color="transparent")
        row_frame.pack(fill="x", padx=20, pady=10)
        
        # Loss Function
        loss_container = ctk.CTkFrame(row_frame, fg_color="transparent")
        loss_container.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        loss_label = ctk.CTkLabel(loss_container, text="Loss Function", text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=12))
        loss_label.pack(anchor="w")
        
        self.loss_menu = ctk.CTkOptionMenu(
            loss_container,
            values=["MSE", "MAE", "Huber"],
            fg_color=BG_INPUT,
            button_color=BG_INPUT,
            button_hover_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            height=36
        )
        self.loss_menu.set(Parametres_choix_loss_fct.fonction_perte)
        self.loss_menu.pack(fill="x")
        
        # Optimizer
        optim_container = ctk.CTkFrame(row_frame, fg_color="transparent")
        optim_container.pack(side="left", fill="x", expand=True)
        
        optim_label = ctk.CTkLabel(optim_container, text="Optimizer", text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=12))
        optim_label.pack(anchor="w")
        
        self.optim_menu = ctk.CTkOptionMenu(
            optim_container,
            values=["Adam", "SGD", "RMSprop"],
            fg_color=BG_INPUT,
            button_color=BG_INPUT,
            button_hover_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            height=36
        )
        self.optim_menu.set(Parametres_optimisateur.optimisateur)
        self.optim_menu.pack(fill="x")
        
        # Learning Rate, Batch Size, Epochs en ligne
        params_frame = ctk.CTkFrame(panel, fg_color="transparent")
        params_frame.pack(fill="x", padx=20, pady=10)
        
        self.create_small_input(params_frame, "Learning Rate", "0.001", 0)
        self.create_small_input(params_frame, "Batch Size", "32", 1)
        self.create_small_input(params_frame, "Epochs", "100", 2)
        
        # Boutons d'action
        btn_config = ctk.CTkButton(
            panel,
            text="Configure Model",
            fg_color=ACCENT_PRIMARY,
            hover_color="#c0392b",
            height=40,
            command=self.Parametrer_modele
        )
        btn_config.pack(fill="x", padx=20, pady=(15, 5))
        
        btn_save = ctk.CTkButton(
            panel,
            text="Save Configuration",
            fg_color="transparent",
            border_width=1,
            border_color=ACCENT_PRIMARY,
            text_color=ACCENT_PRIMARY,
            hover_color=BG_INPUT,
            height=40
        )
        btn_save.pack(fill="x", padx=20, pady=(5, 20))
        
        # Data Management
        data_title = ctk.CTkLabel(
            panel,
            text="Data Management",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        data_title.pack(anchor="w", padx=20, pady=(20, 15))
        
        upload_label = ctk.CTkLabel(panel, text="Upload Dataset", text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=12))
        upload_label.pack(anchor="w", padx=20, pady=(0, 5))
        
        upload_btn = ctk.CTkButton(
            panel,
            text="Click to upload a file",
            fg_color=BG_INPUT,
            border_width=1,
            border_color=BORDER_COLOR,
            text_color=TEXT_SECONDARY,
            hover_color=BORDER_COLOR,
            height=36,
            command=self.Parametrer_dataset
        )
        upload_btn.pack(fill="x", padx=20, pady=(0, 10))
        
        horizon_label = ctk.CTkLabel(panel, text="Temporal Horizon", text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=12))
        horizon_label.pack(anchor="w", padx=20, pady=(10, 5))
        
        self.horizon_entry = ctk.CTkEntry(
            panel,
            placeholder_text="24",
            fg_color=BG_INPUT,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            height=36
        )
        self.horizon_entry.insert(0, str(Parametres_temporels.horizon))
        self.horizon_entry.pack(fill="x", padx=20, pady=(0, 10))
        
        btn_load = ctk.CTkButton(
            panel,
            text="Load Dataset",
            fg_color=ACCENT_PRIMARY,
            hover_color="#c0392b",
            height=40,
            command=self.Parametrer_horizon
        )
        btn_load.pack(fill="x", padx=20, pady=(5, 20))
        
        # Boutons d'entraînement
        train_btn = ctk.CTkButton(
            panel,
            text="▶ Start Training",
            fg_color="#27ae60",
            hover_color="#229954",
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.EnvoyerConfig
        )
        train_btn.pack(fill="x", padx=20, pady=(20, 5))
        
        stop_btn = ctk.CTkButton(
            panel,
            text="⏹ Stop Training",
            fg_color=ACCENT_PRIMARY,
            hover_color="#c0392b",
            height=45,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self.annuler_entrainement
        )
        stop_btn.pack(fill="x", padx=20, pady=(5, 20))
        
        return panel
    
    def create_input_field(self, parent, label, placeholder):
        label_widget = ctk.CTkLabel(parent, text=label, text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=12))
        label_widget.pack(anchor="w", padx=20, pady=(10, 5))
        
        entry = ctk.CTkEntry(
            parent,
            placeholder_text=placeholder,
            fg_color=BG_INPUT,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            height=36
        )
        entry.pack(fill="x", padx=20, pady=(0, 10))
        return entry
    
    def create_small_input(self, parent, label, value, col):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.grid(row=0, column=col, padx=5, sticky="ew")
        parent.grid_columnconfigure(col, weight=1)
        
        label_widget = ctk.CTkLabel(container, text=label, text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=11))
        label_widget.pack(anchor="w")
        
        entry = ctk.CTkEntry(
            container,
            placeholder_text=value,
            fg_color=BG_INPUT,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY,
            height=36
        )
        entry.insert(0, value)
        entry.pack(fill="x")
        return entry
    
    def create_results_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8)
        
        # Tabs
        self.tabview = ctk.CTkTabview(
            panel,
            fg_color=BG_CARD,
            segmented_button_fg_color=BG_INPUT,
            segmented_button_selected_color=ACCENT_PRIMARY,
            segmented_button_selected_hover_color="#c0392b",
            segmented_button_unselected_color=BG_INPUT,
            segmented_button_unselected_hover_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY
        )
        self.tabview.pack(fill="both", expand=True, padx=0, pady=0)
        
        self.tabview.add("Training")
        self.tabview.add("Testing")
        self.tabview.add("Metrics")
        self.tabview.add("Prediction")
        
        # Cadres de résultats
        self.Cadre_results_Entrainement = Cadre_Entrainement(self, self.tabview.tab("Training"))
        self.Cadre_results_Entrainement.pack(fill="both", expand=True)
        
        self.Cadre_results_Testing = Cadre_Testing(self, self.tabview.tab("Testing"))
        self.Cadre_results_Testing.pack(fill="both", expand=True)
        
        self.Cadre_results_Metrics = Cadre_Metrics(self, self.tabview.tab("Metrics"))
        self.Cadre_results_Metrics.pack(fill="both", expand=True)
        
        self.Cadre_results_Prediction = Cadre_Prediction(self, self.tabview.tab("Prediction"))
        self.Cadre_results_Prediction.pack(fill="both", expand=True)
        
        return panel
    
    def on_model_change(self, value):
        Parametres_choix_reseau_neurones.modele = value
    
    def test(self):
        messagebox.showinfo("Info", "Feature to be implemented")
    
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
    
    def Formatter_JSON_global(self):
        self.config_totale = {}
        self.config_totale["Parametres_temporels"] = Parametres_temporels.__dict__
        self.config_totale["Parametres_choix_reseau_neurones"] = Parametres_choix_reseau_neurones.__dict__
        self.config_totale["Parametres_choix_loss_fct"] = Parametres_choix_loss_fct.__dict__
        self.config_totale["Parametres_optimisateur"] = Parametres_optimisateur.__dict__
        self.config_totale["Parametres_entrainement"] = Parametres_entrainement.__dict__
        self.config_totale["Parametres_visualisation_suivi"] = Parametres_visualisation_suivi.__dict__
        return self.config_totale
    
    def Formatter_JSON_specif(self):
        self.config_specifique = {}
        if Parametres_choix_reseau_neurones.modele == "MLP":
            self.config_specifique["Parametres_archi_reseau"] = Parametres_archi_reseau_MLP.__dict__
        elif Parametres_choix_reseau_neurones.modele == "LSTM":
            self.config_specifique["Parametres_archi_reseau"] = Parametres_archi_reseau_LSTM.__dict__
        elif Parametres_choix_reseau_neurones.modele == "CNN":
            self.config_specifique["Parametres_archi_reseau"] = Parametres_archi_reseau_CNN.__dict__
        return self.config_specifique
    
    def annuler_entrainement(self):
        if not self.stop_training:
            self.stop_training = True
            messagebox.showinfo("Cancelled", "Training has been cancelled.")
        else:
            messagebox.showwarning("Info", "No training in progress.")
    
    def EnvoyerConfig(self):
        if self.Cadre_results_Entrainement.is_training == False:
            self.stop_training = False
            self.Cadre_results_Entrainement.start_training()
            
            payload_global = self.Formatter_JSON_global()
            payload_model = self.Formatter_JSON_specif()
            
            print("Payload sent:", {"payload": payload_global, "payload_model": payload_model})
            
            def run_training():
                y = []
                yhat = []
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
                                    
                                    if msg.get("type") == "epoch":
                                        epoch = msg.get("epoch")
                                        avg_loss = msg.get("avg_loss")
                                        
                                        if epoch is not None and avg_loss is not None:
                                            self.Cadre_results_Entrainement.add_data_point(epoch, avg_loss)
                                    
                                    elif "epochs" in msg and "avg_loss" in msg:
                                        epoch = msg.get("epochs")
                                        avg_loss = msg.get("avg_loss")
                                        
                                        if epoch is not None and avg_loss is not None:
                                            self.Cadre_results_Entrainement.add_data_point(epoch, avg_loss)
                                    
                                    elif msg.get("type") == "test_pair":
                                        y.append(msg.get("y"))
                                        yhat.append(msg.get("yhat"))
                                    
                                    elif msg.get("type") == "test_final":
                                        self.Cadre_results_Metrics.afficher_Metrics(msg.get("metrics"))
                                    
                                    elif msg.get("type") == "error":
                                        print(f"ERROR: {msg.get('message')}")
                                        messagebox.showerror("Error", msg.get('message', 'Unknown error'))
                                        break
                                    
                                    elif msg.get("type") == "fin_test":
                                        self.Cadre_results_Testing.plot_predictions(y, yhat)
                                        break
                                
                                except json.JSONDecodeError as e:
                                    print(f"JSON decode error: {e}")
                                    continue
                
                except requests.exceptions.RequestException as e:
                    print(f"Connection error: {e}")
                    messagebox.showerror("Connection Error", f"Unable to connect to server:\n{str(e)}")
                
                finally:
                    self.Cadre_results_Entrainement.stop_training()
            
            training_thread = threading.Thread(target=run_training, daemon=True)
            training_thread.start()

# Fenêtres modales
class Fenetre_Params(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        
        self.title("Model Configuration")
        self.geometry("700x800")
        self.configure(fg_color=BG_DARK)
        
        scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        title = ctk.CTkLabel(
            scroll_frame,
            text="Advanced Model Parameters",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        title.pack(pady=(0, 20))
        
        # Model Type
        model_section = self.create_section(scroll_frame, "Model Type")
        self.model_var = tk.StringVar(value=Parametres_choix_reseau_neurones.modele)
        
        for model in ["MLP", "LSTM", "GRU", "CNN"]:
            radio = ctk.CTkRadioButton(
                model_section,
                text=model,
                variable=self.model_var,
                value=model,
                fg_color=ACCENT_PRIMARY,
                hover_color="#c0392b",
                text_color=TEXT_PRIMARY
            )
            radio.pack(anchor="w", pady=5, padx=20)
        
        # MLP Parameters
        mlp_section = self.create_section(scroll_frame, "MLP Parameters")
        self.add_param_entry(mlp_section, "Layers:", Parametres_archi_reseau_MLP.nb_couches)
        self.add_param_entry(mlp_section, "Hidden Size:", Parametres_archi_reseau_MLP.hidden_size)
        self.add_param_entry(mlp_section, "Dropout:", Parametres_archi_reseau_MLP.dropout_rate)
        
        # Loss Function
        loss_section = self.create_section(scroll_frame, "Loss Function")
        self.loss_var = tk.StringVar(value=Parametres_choix_loss_fct.fonction_perte)
        
        for loss in ["MSE", "MAE", "Huber"]:
            radio = ctk.CTkRadioButton(
                loss_section,
                text=loss,
                variable=self.loss_var,
                value=loss,
                fg_color=ACCENT_PRIMARY,
                hover_color="#c0392b",
                text_color=TEXT_PRIMARY
            )
            radio.pack(anchor="w", pady=5, padx=20)
        
        # Optimizer
        optim_section = self.create_section(scroll_frame, "Optimizer")
        self.add_param_entry(optim_section, "Learning Rate:", Parametres_optimisateur.learning_rate)
        
        # Boutons
        btn_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=20)
        
        save_btn = ctk.CTkButton(
            btn_frame,
            text="Save Configuration",
            command=self.Save_quit,
            fg_color=ACCENT_PRIMARY,
            hover_color="#c0392b",
            height=45
        )
        save_btn.pack(fill="x", pady=(0, 10))
        
        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self.destroy,
            fg_color="transparent",
            border_width=1,
            border_color=BORDER_COLOR,
            hover_color=BG_INPUT,
            height=45
        )
        cancel_btn.pack(fill="x")
    
    def create_section(self, parent, title):
        section = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8)
        section.pack(fill="x", pady=(0, 15))
        
        title_label = ctk.CTkLabel(
            section,
            text=title,
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        title_label.pack(anchor="w", padx=20, pady=(15, 10))
        
        return section
    
    def add_param_entry(self, parent, label_text, default_value):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", padx=20, pady=5)
        
        label = ctk.CTkLabel(
            container,
            text=label_text,
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=12)
        )
        label.pack(side="left", padx=(0, 10))
        
        entry = ctk.CTkEntry(
            container,
            fg_color=BG_INPUT,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY
        )
        entry.insert(0, str(default_value))
        entry.pack(side="left", fill="x", expand=True)
    
    def est_ouverte(self):
        return self.winfo_exists()
    
    def Save_quit(self):
        Parametres_choix_reseau_neurones.modele = self.model_var.get()
        Parametres_choix_loss_fct.fonction_perte = self.loss_var.get()
        self.destroy()

class Fenetre_Params_horizon(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        
        self.title("Temporal Parameters")
        self.geometry("600x700")
        self.configure(fg_color=BG_DARK)
        
        scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        title = ctk.CTkLabel(
            scroll_frame,
            text="Temporal Configuration",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        title.pack(pady=(0, 20))
        
        params_frame = ctk.CTkFrame(scroll_frame, fg_color=BG_CARD, corner_radius=8)
        params_frame.pack(fill="x", pady=(0, 15))
        
        self.horizon_var = tk.IntVar(value=Parametres_temporels.horizon)
        self.pas_var = tk.IntVar(value=Parametres_temporels.pas_temporel)
        self.portion_var = tk.IntVar(value=int(Parametres_temporels.portion_decoupage * 100))
        self.date_debut_str = tk.StringVar(value=Parametres_temporels.dates[0])
        self.date_fin_str = tk.StringVar(value=Parametres_temporels.dates[1])
        
        self.add_entry_field(params_frame, "Horizon:", self.horizon_var)
        self.add_entry_field(params_frame, "Time Step:", self.pas_var)
        self.add_entry_field(params_frame, "Split Ratio (%):", self.portion_var)
        
        # Dates
        self.add_date_field(params_frame, "Start Date:", self.date_debut_str, self.ouvrir_calendrier_debut)
        self.add_date_field(params_frame, "End Date:", self.date_fin_str, self.ouvrir_calendrier_fin)
        
        # Boutons
        btn_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        btn_frame.pack(fill="x", pady=20)
        
        save_btn = ctk.CTkButton(
            btn_frame,
            text="Save Configuration",
            command=self.Save_quit,
            fg_color=ACCENT_PRIMARY,
            hover_color="#c0392b",
            height=45
        )
        save_btn.pack(fill="x", pady=(0, 10))
        
        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self.destroy,
            fg_color="transparent",
            border_width=1,
            border_color=BORDER_COLOR,
            hover_color=BG_INPUT,
            height=45
        )
        cancel_btn.pack(fill="x")
    
    def add_entry_field(self, parent, label_text, variable):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", padx=20, pady=10)
        
        label = ctk.CTkLabel(
            container,
            text=label_text,
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=12)
        )
        label.pack(side="left", padx=(0, 10))
        
        entry = ctk.CTkEntry(
            container,
            textvariable=variable,
            fg_color=BG_INPUT,
            border_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY
        )
        entry.pack(side="left", fill="x", expand=True)
    
    def add_date_field(self, parent, label_text, variable, command):
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.pack(fill="x", padx=20, pady=10)
        
        label = ctk.CTkLabel(
            container,
            text=label_text,
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=12)
        )
        label.pack(side="left", padx=(0, 10))
        
        btn = ctk.CTkButton(
            container,
            textvariable=variable,
            command=command,
            fg_color=BG_INPUT,
            hover_color=BORDER_COLOR,
            text_color=TEXT_PRIMARY
        )
        btn.pack(side="left", fill="x", expand=True)
    
    def est_ouverte(self):
        return self.winfo_exists()
    
    def ouvrir_calendrier_debut(self):
        top = tk.Toplevel(self)
        top.title("Select Start Date")
        try:
            date_obj = datetime.strptime(self.date_debut_str.get(), "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.today()
        cal = Calendar(top, selectmode='day', date_pattern='yyyy-mm-dd',
                      year=date_obj.year, month=date_obj.month, day=date_obj.day)
        cal.pack(padx=10, pady=10)
        tk.Button(top, text="OK", command=lambda: (self.date_debut_str.set(cal.get_date()), top.destroy())).pack(pady=10)
    
    def ouvrir_calendrier_fin(self):
        top = tk.Toplevel(self)
        top.title("Select End Date")
        try:
            date_obj = datetime.strptime(self.date_fin_str.get(), "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.today()
        cal = Calendar(top, selectmode='day', date_pattern='yyyy-mm-dd',
                      year=date_obj.year, month=date_obj.month, day=date_obj.day)
        cal.pack(padx=10, pady=10)
        tk.Button(top, text="OK", command=lambda: (self.date_fin_str.set(cal.get_date()), top.destroy())).pack(pady=10)
    
    def Save_quit(self):
        Parametres_temporels.horizon = self.horizon_var.get()
        Parametres_temporels.pas_temporel = self.pas_var.get()
        Parametres_temporels.portion_decoupage = self.portion_var.get() / 100
        Parametres_temporels.dates = [self.date_debut_str.get(), self.date_fin_str.get()]
        self.destroy()

class Fenetre_Choix_datasets(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
        
        self.title("Dataset Selection")
        self.geometry("600x600")
        self.configure(fg_color=BG_DARK)
        
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        title = ctk.CTkLabel(
            main_frame,
            text="Choose Dataset",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT_PRIMARY
        )
        title.pack(pady=(0, 20))
        
        list_frame = ctk.CTkFrame(main_frame, fg_color=BG_CARD, corner_radius=8)
        list_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        scroll_frame = ctk.CTkScrollableFrame(list_frame, fg_color=BG_INPUT)
        scroll_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        self.Liste_datasets = ["Dataset A", "Dataset B", "Dataset C", "Dataset D"]
        self.selected_dataset = tk.StringVar()
        
        for dataset in self.Liste_datasets:
            radio = ctk.CTkRadioButton(
                scroll_frame,
                text=dataset,
                variable=self.selected_dataset,
                value=dataset,
                fg_color=ACCENT_PRIMARY,
                hover_color="#c0392b",
                text_color=TEXT_PRIMARY
            )
            radio.pack(anchor="w", pady=8, padx=10)
        
        btn_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        btn_frame.pack(fill="x")
        
        save_btn = ctk.CTkButton(
            btn_frame,
            text="Select Dataset",
            command=self.Save_quit,
            fg_color=ACCENT_PRIMARY,
            hover_color="#c0392b",
            height=45
        )
        save_btn.pack(fill="x", pady=(0, 10))
        
        cancel_btn = ctk.CTkButton(
            btn_frame,
            text="Cancel",
            command=self.destroy,
            fg_color="transparent",
            border_width=1,
            border_color=BORDER_COLOR,
            hover_color=BG_INPUT,
            height=45
        )
        cancel_btn.pack(fill="x")
    
    def est_ouverte(self):
        return self.winfo_exists()
    
    def Save_quit(self):
        selected = self.selected_dataset.get()
        if selected:
            messagebox.showinfo("Selected", f"You selected: {selected}")
        self.destroy()

if __name__ == "__main__":
    app = Fenetre_Acceuil()
    app.mainloop()