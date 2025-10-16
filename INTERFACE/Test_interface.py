import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar

class CalendarPicker(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Datepicker with tkcalendar")
        self.geometry("300x300")

        # Create calendar widget
        self.cal = Calendar(self, selectmode='day', date_pattern='dd/mm/yyyy')
        self.cal.pack(pady=20)

        # Button to get selected date
        ttk.Button(self, text="Get Date", command=self.get_date).pack(pady=10)

    def get_date(self):
        selected_date = self.cal.get_date()
        messagebox.showinfo("Selected Date", f"You selected: {selected_date}")

if __name__ == "__main__":
    app = CalendarPicker()
    app.mainloop()