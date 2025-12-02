# 🔄 Migration de Tkinter vers CustomTkinter

## 📋 Résumé des changements

Ce document décrit la migration du code `interface_local.py` vers `interface_customtkinter.py` en utilisant la bibliothèque CustomTkinter tout en conservant les styles visuels par défaut de Tkinter.

---

## 🎯 Objectifs de la migration

1. ✅ Utiliser **CustomTkinter** pour une interface moderne
2. ✅ Conserver l'**apparence et les couleurs** de Tkinter par défaut
3. ✅ Maintenir la **compatibilité fonctionnelle** complète
4. ✅ Améliorer la **maintenabilité** du code

---

## 📦 Dépendances

### Installation de CustomTkinter

```bash
pip install customtkinter
```

### Bibliothèques requises

```python
customtkinter  # Version >= 5.0.0
tkcalendar     # Pour les sélecteurs de dates
requests       # Pour les appels HTTP/SSE
matplotlib     # Pour les graphiques
numpy          # Pour les calculs
```

---

## 🔄 Principales modifications

### 1. **Imports**

**Avant (Tkinter):**
```python
import tkinter as tk
from tkinter import ttk
```

**Après (CustomTkinter):**
```python
import customtkinter as ctk
import tkinter as tk  # Toujours nécessaire pour certains widgets
```

### 2. **Configuration de l'apparence**

```python
# Configuration pour ressembler à Tkinter par défaut
ctk.set_appearance_mode("light")  # Mode clair
ctk.set_default_color_theme("blue")  # Thème bleu

# Couleurs style Tkinter
self.cadres_bg = "#f0f0f0"  # Gris clair standard
self.fenetre_bg = "#f0f0f0"
```

### 3. **Classe principale**

**Avant:**
```python
class Fenetre_Acceuil(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
```

**Après:**
```python
class Fenetre_Acceuil(ctk.CTk):
    def __init__(self):
        ctk.CTk.__init__(self)
```

### 4. **Widgets remplacés**

| Tkinter | CustomTkinter | Propriétés principales |
|---------|---------------|----------------------|
| `tk.Frame` | `ctk.CTkFrame` | `fg_color` au lieu de `bg` |
| `tk.Label` | `ctk.CTkLabel` | `text_color` au lieu de `fg` |
| `tk.Button` | `ctk.CTkButton` | `fg_color`, `hover_color` |
| `tk.Entry` | `ctk.CTkEntry` | Style moderne automatique |
| `ttk.Notebook` | `ctk.CTkTabview` | API différente pour les onglets |
| `tk.OptionMenu` | `ctk.CTkOptionMenu` | Plus moderne visuellement |
| `tk.Checkbutton` | `ctk.CTkCheckBox` | Orthographe différente |
| `tk.Scrollbar` | `ctk.CTkScrollableFrame` | Intégré dans le frame |

### 5. **Gestion des onglets (Notebook → TabView)**

**Avant (ttk.Notebook):**
```python
self.Results_notebook = ttk.Notebook(parent)
cadre = tk.Frame(parent)
self.Results_notebook.add(cadre, text="Training")
```

**Après (CTkTabview):**
```python
self.Results_notebook = ctk.CTkTabview(parent)
self.Results_notebook.add("Training")
cadre = ctk.CTkFrame(self.Results_notebook.tab("Training"))
```

### 6. **Propriétés des couleurs**

**CustomTkinter utilise des noms différents:**

| Tkinter | CustomTkinter |
|---------|---------------|
| `bg` | `fg_color` |
| `fg` | `text_color` |
| `activebackground` | `hover_color` |
| `highlightbackground` | `border_color` |

### 7. **Polices**

**Style Tkinter par défaut conservé:**
```python
# Utilisation des polices système par défaut
self.font_titre = ("TkDefaultFont", 20, "bold")
self.font_section = ("TkDefaultFont", 18, "bold")
self.font_bouton = ("TkDefaultFont", 14)
```

### 8. **Fenêtres Toplevel**

**Avant:**
```python
class Fenetre_Params(tk.Toplevel):
    def __init__(self, master=None):
        super().__init__(master)
```

**Après:**
```python
class Fenetre_Params(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)
```

---

## 🎨 Conservation du style Tkinter

### Couleurs utilisées

```python
# Couleurs Tkinter par défaut
cadres_bg = "#f0f0f0"      # Gris clair (fond des frames)
cadres_fg = "#e0e0e0"      # Gris plus clair
fenetre_bg = "#f0f0f0"     # Fond de fenêtre
button_green = "#4CAF50"   # Bouton sauvegarder
button_red = "#f44336"     # Bouton annuler
```

### Personnalisation des boutons

```python
# Bouton avec style Tkinter
btn = ctk.CTkButton(
    parent,
    text="Texte",
    fg_color="#e0e0e0",        # Fond gris clair
    hover_color="#d0d0d0",     # Gris plus foncé au survol
    text_color="black",        # Texte noir
    corner_radius=3            # Coins peu arrondis
)
```

---

## ⚠️ Points d'attention

### 1. **Matplotlib reste inchangé**

Les graphiques Matplotlib fonctionnent exactement de la même manière:

```python
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Utilisation identique
self.fig = Figure(figsize=(8, 6), dpi=100)
self.canvas = FigureCanvasTkAgg(self.fig, self)
```

### 2. **tkcalendar reste en Tkinter**

Le widget Calendar utilise toujours Tkinter standard:

```python
from tkcalendar import Calendar

# Dans une fenêtre Toplevel Tkinter classique
top = tk.Toplevel(self)  # Pas ctk.CTkToplevel
cal = Calendar(top, selectmode='day', ...)
```

### 3. **Validation des entrées**

CustomTkinter ne supporte pas directement `validate` et `validatecommand`. 
Pour la validation, utilisez des callbacks sur les variables:

```python
var = ctk.StringVar()
var.trace_add("write", callback_function)
```

### 4. **Listbox non disponible**

CustomTkinter n'a pas de widget `CTkListbox` natif. Utilisez:
- `CTkOptionMenu` pour une liste déroulante
- `CTkScrollableFrame` avec des `CTkRadioButton` pour une liste sélectionnable
- Ou gardez `tk.Listbox` si nécessaire

---

## 🚀 Utilisation du nouveau code

### Lancement de l'application

```python
python interface_customtkinter.py
```

### Structure du code

```
interface_customtkinter.py
├── Classes de paramètres (inchangées)
│   ├── Parametres_temporels_class
│   ├── Parametres_choix_reseau_neurones_class
│   ├── Parametres_archi_reseau_class
│   ├── Parametres_choix_loss_fct_class
│   ├── Parametres_optimisateur_class
│   ├── Parametres_entrainement_class
│   └── Parametres_visualisation_suivi_class
│
├── Fenêtres principales (CustomTkinter)
│   ├── Fenetre_Acceuil (CTk)
│   ├── Cadre_Entrainement (CTkFrame)
│   ├── Cadre_Testing (CTkFrame)
│   ├── Cadre_Metrics (CTkFrame)
│   └── Cadre_Prediction (CTkFrame)
│
└── Fenêtres de configuration (CTkToplevel)
    ├── Fenetre_Params
    ├── Fenetre_Params_horizon
    └── Fenetre_Choix_datasets
```

---

## 📊 Comparaison visuelle

### Avant (Tkinter)
- Style Windows/Mac natif
- Boutons plats standard
- Couleurs système par défaut
- Apparence classique

### Après (CustomTkinter avec style Tkinter)
- Même palette de couleurs
- Légère amélioration des coins arrondis
- Transitions douces au survol
- Look moderne mais familier

---

## ✅ Fonctionnalités conservées

Toutes les fonctionnalités de l'interface originale sont conservées:

- ✅ Configuration du modèle (MLP/CNN/LSTM)
- ✅ Sélection des datasets
- ✅ Paramétrage de l'horizon temporel
- ✅ Lancement de l'entraînement
- ✅ Streaming SSE en temps réel
- ✅ Graphiques d'entraînement
- ✅ Graphiques de test
- ✅ Affichage des métriques
- ✅ Annulation de l'entraînement

---

## 🔧 Personnalisation avancée

### Changer le thème

```python
# Thèmes disponibles
ctk.set_default_color_theme("blue")    # Bleu (défaut)
ctk.set_default_color_theme("green")   # Vert
ctk.set_default_color_theme("dark-blue")  # Bleu foncé

# Mode d'apparence
ctk.set_appearance_mode("light")  # Clair
ctk.set_appearance_mode("dark")   # Sombre
ctk.set_appearance_mode("system") # Selon le système
```

### Personnaliser un bouton

```python
bouton = ctk.CTkButton(
    parent,
    text="Mon Bouton",
    command=ma_fonction,
    
    # Couleurs
    fg_color="#3498db",           # Couleur de fond
    hover_color="#2980b9",        # Couleur au survol
    text_color="white",           # Couleur du texte
    border_color="#2c3e50",       # Couleur de la bordure
    
    # Dimensions
    width=200,
    height=40,
    
    # Style
    corner_radius=10,             # Arrondi des coins
    border_width=2,               # Épaisseur de la bordure
    
    # Police
    font=("Arial", 14, "bold")
)
```

---

## 🐛 Débogage

### Problème : Widget ne s'affiche pas

**Solution:** Vérifiez que vous utilisez `fg_color` au lieu de `bg`:
```python
# ❌ Incorrect
frame = ctk.CTkFrame(parent, bg="#ffffff")

# ✅ Correct
frame = ctk.CTkFrame(parent, fg_color="#ffffff")
```

### Problème : Couleur du texte ne change pas

**Solution:** Utilisez `text_color` au lieu de `fg`:
```python
# ❌ Incorrect
label = ctk.CTkLabel(parent, fg="black")

# ✅ Correct
label = ctk.CTkLabel(parent, text_color="black")
```

### Problème : Les onglets ne fonctionnent pas

**Solution:** TabView a une API différente:
```python
# ✅ Correct
tabview = ctk.CTkTabview(parent)
tabview.add("Nom Onglet")
contenu = ctk.CTkFrame(tabview.tab("Nom Onglet"))
```

---

## 📚 Ressources

- **Documentation CustomTkinter:** https://customtkinter.tomschimansky.com/
- **GitHub CustomTkinter:** https://github.com/TomSchimansky/CustomTkinter
- **Exemples:** https://github.com/TomSchimansky/CustomTkinter/tree/master/examples

---

## 🎯 Conclusion

Cette migration vers CustomTkinter apporte:

1. **Modernité** - Interface plus actuelle
2. **Compatibilité** - Fonctionne sur Windows, Mac, Linux
3. **Maintenabilité** - Code plus propre et structuré
4. **Flexibilité** - Facile à personnaliser
5. **Performance** - Meilleure gestion du rendu

Tout en conservant:
- L'apparence familière de Tkinter
- Toutes les fonctionnalités existantes
- La logique métier intacte

---

**Date de migration:** 3 Novembre 2025  
**Version CustomTkinter:** 5.2.0+  
**Compatibilité Python:** 3.8+
