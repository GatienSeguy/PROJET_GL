from CTkDataVisualizingWidgets import *
import customtkinter as ctk

window = ctk.CTk()
window.title("Calendar Widget")
ctk.set_appearance_mode("dark")

# init calendar
calendar_widget = CTkCalendar(window, width=300, height=210, border_width=3, border_color="white",
                              fg_color="#020317", title_bar_border_width=3, title_bar_border_color="white",
                              title_bar_fg_color="#020F43", calendar_fg_color="#020F43", corner_radius=30,
                              title_bar_corner_radius=10, calendar_corner_radius=10, calendar_border_color="white",
                              calendar_border_width=3, calendar_label_pad=5,
                              today_fg_color="white", today_text_color="black")
calendar_widget.pack(side="left", padx=20)

window.mainloop()