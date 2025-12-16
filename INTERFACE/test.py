import customtkinter as ctk
import time
import threading

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Demo CTkProgressBar")
        self.geometry("400x200")

        # --- Progress Bar ---
        self.progress = ctk.CTkProgressBar(self, width=300)
        self.progress.pack(pady=20)

        # Mode déterminé (entre 0 et 1)
        self.progress.set(0)

        # --- Buttons ---
        btn_start = ctk.CTkButton(self, text="Lancer progression", command=self.start_progress)
        btn_start.pack()

        btn_indeterminate = ctk.CTkButton(self, text="Indeterminate", command=self.start_indeterminate)
        btn_indeterminate.pack(pady=5)

        btn_stop = ctk.CTkButton(self, text="Stop", command=self.stop_progress)
        btn_stop.pack()

        self.running = False

    # --- Determinate progression ---
    def start_progress(self):
        if self.running:
            return
        self.running = True
        self.progress.set(0)

        def run():
            for i in range(101):
                if not self.running:
                    break
                self.progress.set(i / 100)
                time.sleep(0.03)

        threading.Thread(target=run, daemon=True).start()

    # --- Indeterminate mode ---
    def start_indeterminate(self):
        self.running = False
        self.progress.configure(mode="indeterminate")
        self.progress.start()  # animation

    def stop_progress(self):
        self.running = False
        self.progress.stop()
        self.progress.configure(mode="determinate")

app = App()
app.mainloop()
