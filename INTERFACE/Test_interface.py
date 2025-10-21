import tkinter as tk

root = tk.Tk()
root.geometry("400x300")

# Cadre gauche avec marge
frame_gauche = tk.Frame(root, bg="lightblue", width=100, height=200)
frame_gauche.pack(side="left", fill="y", padx=10, pady=20)

root.mainloop()
