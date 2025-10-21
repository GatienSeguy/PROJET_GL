import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title("Onglets imbriqués")
root.geometry("500x400")

# Notebook principal
main_notebook = ttk.Notebook(root)
main_notebook.pack(expand=True, fill='both')

# Onglet principal "Paramètres"
frame_parametres = ttk.Frame(main_notebook)
main_notebook.add(frame_parametres, text="Paramètres")

# Onglet principal "À propos"
frame_apropos = ttk.Frame(main_notebook)
main_notebook.add(frame_apropos, text="À propos")

# Notebook secondaire dans "Paramètres"
sub_notebook = ttk.Notebook(frame_parametres)
sub_notebook.pack(expand=True, fill='both', padx=10, pady=10)

# Sous-onglets
sub_affichage = ttk.Frame(sub_notebook)
sub_reseau = ttk.Frame(sub_notebook)

sub_notebook.add(sub_affichage, text="Affichage")
sub_notebook.add(sub_reseau, text="Réseau")

# Contenu des sous-onglets
ttk.Label(sub_affichage, text="Réglages d'affichage ici").pack(pady=20)
ttk.Label(sub_reseau, text="Paramètres réseau ici").pack(pady=20)

# Contenu de l'onglet "À propos"
ttk.Label(frame_apropos, text="Application version 1.0").pack(pady=20)

root.mainloop()