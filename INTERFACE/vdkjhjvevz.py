import customtkinter as ctk

def test_focus(delay_ms):
    top = ctk.CTkToplevel(app)
    top.geometry("200x150")
    top.title(f"Toplevel delay {delay_ms}ms")

    # Force le focus après delay_ms millisecondes
    top.after(delay_ms, lambda: top.focus_force())

app = ctk.CTk()
app.geometry("400x300")

# Liste de délais à tester (en ms)
delays = [0, 5, 10, 20, 50, 100]

for d in delays:
    btn = ctk.CTkButton(app, text=f"Ouvrir {d}ms", command=lambda d=d: test_focus(d))
    btn.pack(pady=5)

app.mainloop()
