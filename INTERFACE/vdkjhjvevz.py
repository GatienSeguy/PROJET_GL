import customtkinter as ctk


# Créer la fenêtre principale
app = ctk.CTk()
app.title("Fenêtre CTk par défaut")
app.geometry("400x200")

# Ajouter un label
label = ctk.CTkLabel(app, text="👋 Bonjour depuis CustomTkinter !", font=("Roboto", 18))
label.pack(pady=40)
couleur_texte = label.cget("text_color")
print(couleur_texte)

# Lancer la boucle principale
app.mainloop()