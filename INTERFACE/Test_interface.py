import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import numpy as np

class MonApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Grille logarithmique enrichie")

        self.fig = Figure(figsize=(6, 4))
        self.ax = self.fig.add_subplot(111)

        # Données en log
        x = np.logspace(0.1, 2, 100)
        y = np.log10(x)
        self.ax.plot(x, y)

        # Échelle log
        self.ax.set_xscale('log')

        # Grille principale
        self.ax.grid(True, which='major', linestyle='-', color='gray')

        # Grille secondaire (petites divisions)
        self.ax.grid(True, which='minor', linestyle=':', color='lightgray')

        # Locator pour les ticks mineurs
        self.ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=12))
        self.ax.xaxis.set_minor_formatter(ticker.NullFormatter())  # Optionnel : ne pas afficher les labels mineurs

        # Canvas Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

app = MonApp()
app.mainloop()