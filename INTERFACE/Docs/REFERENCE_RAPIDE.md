# 📋 Guide de Référence Rapide - Tkinter → CustomTkinter

## 🔄 Table de correspondance des widgets

| Tkinter | CustomTkinter | Notes |
|---------|---------------|-------|
| `tk.Tk()` | `ctk.CTk()` | Fenêtre principale |
| `tk.Toplevel()` | `ctk.CTkToplevel()` | Fenêtre secondaire |
| `tk.Frame()` | `ctk.CTkFrame()` | Frame/Cadre |
| `tk.Label()` | `ctk.CTkLabel()` | Label/Étiquette |
| `tk.Button()` | `ctk.CTkButton()` | Bouton |
| `tk.Entry()` | `ctk.CTkEntry()` | Champ de saisie |
| `tk.Text()` | `ctk.CTkTextbox()` | Zone de texte multiligne |
| `tk.Checkbutton()` | `ctk.CTkCheckBox()` | Case à cocher |
| `tk.Radiobutton()` | `ctk.CTkRadioButton()` | Bouton radio |
| `tk.Scale()` | `ctk.CTkSlider()` | Curseur |
| `tk.OptionMenu()` | `ctk.CTkOptionMenu()` | Menu déroulant |
| `tk.Progressbar()` | `ctk.CTkProgressBar()` | Barre de progression |
| `tk.Switch()` | `ctk.CTkSwitch()` | Interrupteur |
| `ttk.Notebook()` | `ctk.CTkTabview()` | Onglets (API différente!) |
| `tk.Canvas()` | `tk.Canvas()` | Utiliser Tkinter standard |
| `tk.Listbox()` | `tk.Listbox()` ou alternatives | Pas de widget natif CTk |
| `tk.Scrollbar()` | `ctk.CTkScrollableFrame()` | Intégré dans le frame |

---

## 🎨 Propriétés des widgets

### Couleurs

| Tkinter | CustomTkinter | Description |
|---------|---------------|-------------|
| `bg` ou `background` | `fg_color` | Couleur de fond |
| `fg` ou `foreground` | `text_color` | Couleur du texte |
| `activebackground` | `hover_color` | Couleur au survol |
| `activeforeground` | - | Pas d'équivalent direct |
| `highlightbackground` | `border_color` | Couleur de la bordure |
| `highlightcolor` | - | Pas d'équivalent |
| `disabledforeground` | - | Géré automatiquement |

### Dimensions

| Tkinter | CustomTkinter | Description |
|---------|---------------|-------------|
| `width` | `width` | Largeur (pixels) |
| `height` | `height` | Hauteur (pixels) |
| `padx` | - | Utiliser `.pack(padx=...)` |
| `pady` | - | Utiliser `.pack(pady=...)` |
| - | `corner_radius` | Arrondi des coins (nouveau) |
| `borderwidth` | `border_width` | Épaisseur de bordure |

### Police

| Tkinter | CustomTkinter | Description |
|---------|---------------|-------------|
| `font=("Arial", 12)` | `font=("Arial", 12)` | Même syntaxe |
| `font=("Arial", 12, "bold")` | `font=("Arial", 12, "bold")` | Même syntaxe |

---

## 📦 Gestionnaires de mise en page (inchangés)

Les gestionnaires de mise en page fonctionnent exactement pareil:

```python
# Pack
widget.pack(side="left", fill="both", expand=True, padx=10, pady=5)

# Grid
widget.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

# Place
widget.place(x=50, y=100, width=200, height=30)
```

---

## 🎯 Exemples de conversion rapide

### 1. Frame simple

**Tkinter:**
```python
frame = tk.Frame(parent, bg="#f0f0f0", bd=2, relief="groove")
frame.pack(fill="x", padx=10, pady=5)
```

**CustomTkinter:**
```python
frame = ctk.CTkFrame(parent, fg_color="#f0f0f0", corner_radius=5)
frame.pack(fill="x", padx=10, pady=5)
```

### 2. Label

**Tkinter:**
```python
label = tk.Label(parent, text="Hello", bg="#ffffff", fg="#000000", font=("Arial", 14))
label.pack(pady=10)
```

**CustomTkinter:**
```python
label = ctk.CTkLabel(parent, text="Hello", text_color="#000000", font=("Arial", 14))
label.pack(pady=10)
```

### 3. Bouton

**Tkinter:**
```python
btn = tk.Button(
    parent,
    text="Cliquez",
    bg="#4CAF50",
    fg="white",
    activebackground="#45a049",
    font=("Arial", 12),
    command=ma_fonction
)
btn.pack(pady=5)
```

**CustomTkinter:**
```python
btn = ctk.CTkButton(
    parent,
    text="Cliquez",
    fg_color="#4CAF50",
    text_color="white",
    hover_color="#45a049",
    font=("Arial", 12),
    command=ma_fonction
)
btn.pack(pady=5)
```

### 4. Entry (champ de saisie)

**Tkinter:**
```python
var = tk.StringVar()
entry = tk.Entry(parent, textvariable=var, bg="white", fg="black")
entry.pack(pady=5)
```

**CustomTkinter:**
```python
var = ctk.StringVar()
entry = ctk.CTkEntry(parent, textvariable=var)
entry.pack(pady=5)
```

### 5. OptionMenu (menu déroulant)

**Tkinter:**
```python
var = tk.StringVar(value="Option1")
menu = tk.OptionMenu(parent, var, "Option1", "Option2", "Option3")
menu.pack(pady=5)
```

**CustomTkinter:**
```python
var = ctk.StringVar(value="Option1")
menu = ctk.CTkOptionMenu(parent, variable=var, values=["Option1", "Option2", "Option3"])
menu.pack(pady=5)
```

### 6. Checkbutton

**Tkinter:**
```python
var = tk.BooleanVar(value=True)
check = tk.Checkbutton(parent, text="Activer", variable=var, bg="#f0f0f0")
check.pack(pady=5)
```

**CustomTkinter:**
```python
var = ctk.BooleanVar(value=True)
check = ctk.CTkCheckBox(parent, text="Activer", variable=var)
check.pack(pady=5)
```

### 7. Notebook → Tabview

**Tkinter:**
```python
notebook = ttk.Notebook(parent)
tab1 = tk.Frame(notebook)
tab2 = tk.Frame(notebook)
notebook.add(tab1, text="Onglet 1")
notebook.add(tab2, text="Onglet 2")
notebook.pack(fill="both", expand=True)
```

**CustomTkinter:**
```python
tabview = ctk.CTkTabview(parent)
tabview.add("Onglet 1")
tabview.add("Onglet 2")
tabview.pack(fill="both", expand=True)

# Accéder au contenu d'un onglet
tab1_content = tabview.tab("Onglet 1")
label = ctk.CTkLabel(tab1_content, text="Contenu")
label.pack()
```

---

## ⚙️ Configuration globale

### Apparence

```python
import customtkinter as ctk

# Mode d'apparence
ctk.set_appearance_mode("light")   # Clair (défaut Tkinter)
ctk.set_appearance_mode("dark")    # Sombre
ctk.set_appearance_mode("system")  # Selon le système

# Thème de couleur
ctk.set_default_color_theme("blue")      # Bleu
ctk.set_default_color_theme("green")     # Vert
ctk.set_default_color_theme("dark-blue") # Bleu foncé
```

### Scaling (mise à l'échelle)

```python
# Ajuster la taille de tous les widgets
ctk.set_widget_scaling(1.0)  # 100% (défaut)
ctk.set_widget_scaling(1.5)  # 150% (plus grand)

# Ajuster la taille des fenêtres
ctk.set_window_scaling(1.0)  # 100% (défaut)
```

---

## 🔧 Cas spéciaux

### 1. ScrollableFrame

Au lieu de créer un Canvas + Scrollbar:

**CustomTkinter:**
```python
# Frame scrollable intégré
scrollable = ctk.CTkScrollableFrame(parent, width=400, height=300)
scrollable.pack(fill="both", expand=True)

# Ajouter du contenu directement
for i in range(50):
    ctk.CTkLabel(scrollable, text=f"Item {i}").pack(pady=2)
```

### 2. Textbox (zone de texte)

**Tkinter:**
```python
text = tk.Text(parent, width=40, height=10, bg="white", fg="black")
text.pack()
text.insert("1.0", "Contenu initial")
```

**CustomTkinter:**
```python
textbox = ctk.CTkTextbox(parent, width=400, height=200)
textbox.pack()
textbox.insert("0.0", "Contenu initial")  # Note: "0.0" au lieu de "1.0"
```

### 3. Slider (curseur)

**Tkinter:**
```python
var = tk.DoubleVar(value=50)
slider = tk.Scale(parent, from_=0, to=100, variable=var, orient="horizontal")
slider.pack()
```

**CustomTkinter:**
```python
var = ctk.DoubleVar(value=50)
slider = ctk.CTkSlider(parent, from_=0, to=100, variable=var)
slider.pack()
```

### 4. Switch (interrupteur)

**CustomTkinter uniquement:**
```python
var = ctk.BooleanVar(value=False)
switch = ctk.CTkSwitch(parent, text="Activer", variable=var, command=callback)
switch.pack()
```

---

## 🚨 Pièges courants

### ❌ Erreur 1: Utiliser `bg` au lieu de `fg_color`

```python
# ❌ Ne fonctionne pas
frame = ctk.CTkFrame(parent, bg="#f0f0f0")

# ✅ Correct
frame = ctk.CTkFrame(parent, fg_color="#f0f0f0")
```

### ❌ Erreur 2: Utiliser `fg` au lieu de `text_color`

```python
# ❌ Ne fonctionne pas
label = ctk.CTkLabel(parent, text="Hello", fg="black")

# ✅ Correct
label = ctk.CTkLabel(parent, text="Hello", text_color="black")
```

### ❌ Erreur 3: Mauvaise API pour TabView

```python
# ❌ Ne fonctionne pas
tabview = ctk.CTkTabview(parent)
tab = ctk.CTkFrame(tabview)
tabview.add(tab, text="Onglet")

# ✅ Correct
tabview = ctk.CTkTabview(parent)
tabview.add("Onglet")
content = ctk.CTkFrame(tabview.tab("Onglet"))
```

### ❌ Erreur 4: Oublier les variables CTk

```python
# ❌ Peut ne pas fonctionner correctement
var = tk.StringVar()
entry = ctk.CTkEntry(parent, textvariable=var)

# ✅ Recommandé (mais tk.StringVar fonctionne aussi)
var = ctk.StringVar()
entry = ctk.CTkEntry(parent, textvariable=var)
```

---

## 📝 Checklist de migration

- [ ] Remplacer `import tkinter as tk` → garder + ajouter `import customtkinter as ctk`
- [ ] Remplacer `tk.Tk()` → `ctk.CTk()`
- [ ] Remplacer `tk.Toplevel()` → `ctk.CTkToplevel()`
- [ ] Remplacer tous les `tk.Frame()` → `ctk.CTkFrame()`
- [ ] Remplacer tous les `tk.Label()` → `ctk.CTkLabel()`
- [ ] Remplacer tous les `tk.Button()` → `ctk.CTkButton()`
- [ ] Remplacer tous les `tk.Entry()` → `ctk.CTkEntry()`
- [ ] Remplacer `bg=` → `fg_color=`
- [ ] Remplacer `fg=` → `text_color=`
- [ ] Remplacer `activebackground=` → `hover_color=`
- [ ] Remplacer `ttk.Notebook` → `ctk.CTkTabview` (attention à l'API!)
- [ ] Configurer l'apparence: `ctk.set_appearance_mode("light")`
- [ ] Tester toutes les fonctionnalités

---

## 🎨 Palette de couleurs style Tkinter

```python
# Couleurs pour imiter Tkinter
COLORS = {
    "bg_default": "#f0f0f0",        # Gris clair standard
    "bg_frame": "#e0e0e0",          # Gris pour frames
    "bg_white": "#ffffff",          # Blanc
    "text_default": "#000000",      # Noir
    "text_disabled": "#a3a3a3",     # Gris désactivé
    "button_success": "#4CAF50",    # Vert succès
    "button_danger": "#f44336",     # Rouge danger
    "button_primary": "#2196F3",    # Bleu primaire
    "border": "#a3a3a3",            # Bordure grise
}
```

---

## 🔗 Ressources utiles

- **Documentation officielle:** https://customtkinter.tomschimansky.com/
- **GitHub:** https://github.com/TomSchimansky/CustomTkinter
- **Exemples:** https://github.com/TomSchimansky/CustomTkinter/tree/master/examples
- **Wiki:** https://github.com/TomSchimansky/CustomTkinter/wiki

---

**Version:** 1.0  
**Date:** Novembre 2025  
**Auteur:** Migration Tkinter → CustomTkinter
