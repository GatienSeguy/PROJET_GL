"""
Interface très simple avec CustomTkinter (ctk) : une fenêtre et un seul bouton.
Exécutez : python interface_ctk_un_bouton.py
"""

import customtkinter as ctk
from tkinter import messagebox

# Apparence (optionnel)
ctk.set_appearance_mode("System")  # "System", "Dark", "Light"
ctk.set_default_color_theme("blue")  # thèmes intégrés : "blue", "green", "dark-blue"

class SimpleApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Interface CTk - 1 bouton")
        self.geometry("320x140")
        self.resizable(False, False)

        # Création du bouton centré
        self.button = ctk.CTkButton(self, text="Cliquez-moi", command=self.on_click)
        self.button.pack(padx=20, pady=30, expand=True)
        print(self.button.cget("fg_color"))

    def on_click(self):
        # Action exécutée au clic : simple boîte de dialogue
        messagebox.showinfo("Info", "Bouton cliqué !")


if __name__ == "__main__":
    app = SimpleApp()
    app.mainloop()
