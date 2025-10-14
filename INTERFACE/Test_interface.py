import tkinter as tk

import tkinter as tk


# Créer la fenêtre principale
fenetre = tk.Tk()
fenetre.title("Fenêtre Tkinter")
fenetre.geometry("300x200")  # largeur x hauteur

# Ajouter un label
label = tk.Label(fenetre, text="Bonjour, Maxime !")
label.pack(pady=20)

# Ajouter un bouton pour fermer
bouton_quitter = tk.Button(fenetre, text="Quitter", command=fenetre.quit)
bouton_quitter.pack()

# Lancer la boucle principale
fenetre.mainloop()
