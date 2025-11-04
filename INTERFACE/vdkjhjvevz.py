import customtkinter as ctk

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Fenêtre Zoomée")
        self.after(100, lambda: self.state("zoomed"))

        self.bind("<F11>", lambda event: self.attributes("-fullscreen", True))
        self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))

app = App()
app.mainloop()