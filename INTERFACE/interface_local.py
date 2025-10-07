import tkinter as tk
from tkinter import ttk

# Créer la fenêtre principale
fenetre = tk.Tk()
fenetre.title("Ma première interface")
fenetre.geometry("300x200")  # largeur x hauteur

# Ajouter un label
label = tk.Label(fenetre, text="Bonjour, Maxime !")
label.pack()

# Ajouter un bouton
def clic():
    label.config(text="Tu as cliqué fdp de feuj!")

bouton = tk.Button(fenetre, text="Clique-moi", command=clic)
bouton.pack()

# Lancer la boucle principale
fenetre.mainloop()
