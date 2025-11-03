import customtkinter as ctk

# Configuration du thème
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Fenêtre principale
root = ctk.CTk()
root.title("Exemple CTkTabview")
root.geometry("400x300")

# Création du Tabview
tabview = ctk.CTkTabview(root)
tabview.pack(fill="both", expand=True, padx=20, pady=20)

# Ajout des onglets
tabview.add("Onglet 1")
tabview.add("Onglet 2")

# Accès aux frames internes
tab1 = tabview.tab("Onglet 1")
tab2 = tabview.tab("Onglet 2")

# Contenu de l'onglet 1
ctk.CTkLabel(tab1, text="Bienvenue dans l'onglet 1").pack(pady=20)
ctk.CTkButton(tab1, text="Bouton 1").pack()

# Contenu de l'onglet 2
ctk.CTkLabel(tab2, text="Ceci est l'onglet 2").pack(pady=20)
ctk.CTkButton(tab2, text="Bouton 2").pack()

root.mainloop()
