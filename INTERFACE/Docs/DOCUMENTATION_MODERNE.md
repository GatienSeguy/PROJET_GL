# 🎨 Interface MLApp avec CustomTkinter (Style Moderne)

## 📋 Vue d'ensemble

Ce document présente l'interface **MLApp** complètement réécrite avec **CustomTkinter**, utilisant tous les **styles par défaut modernes** de la bibliothèque pour une interface élégante et contemporaine.

---

## ✨ Caractéristiques du design moderne

### 🎨 Thème visuel

- **Mode d'apparence** : Automatique (suit le système : clair/sombre)
- **Palette de couleurs** : Thème bleu par défaut de CustomTkinter
- **Polices** : Roboto (moderne et lisible)
- **Coins arrondis** : Sur tous les frames et boutons
- **Transitions** : Animations douces au survol

### 🖌️ Éléments de design

```python
# Configuration globale
ctk.set_appearance_mode("system")  # Auto light/dark
ctk.set_default_color_theme("blue")  # Thème bleu

# Polices modernes
font_titre = ("Roboto Medium", 20)
font_section = ("Roboto Medium", 16)
font_bouton = ("Roboto", 13)
```

---

## 🏗️ Structure de l'interface

### Fenêtre principale

```
┌─────────────────────────────────────────────────────────────┐
│  MLApp - Machine Learning Application                       │
├───────────────┬─────────────────────────────────────────────┤
│               │  ┌───────────────────────────────────────┐  │
│  🧬 Modèle    │  │                                       │  │
│  - Charger    │  │          Training / Testing           │  │
│  - Paramétrer │  │          Metrics / Prediction         │  │
│               │  │                                       │  │
│  📊 Données   │  │         (Onglets avec TabView)        │  │
│  - Dataset    │  │                                       │  │
│  - Horizon    │  │                                       │  │
│               │  │                                       │  │
│  🚀 Actions   │  │                                       │  │
│  - Lancer     │  │                                       │  │
│  - Annuler    │  └───────────────────────────────────────┘  │
│               │                                             │
└───────────────┴─────────────────────────────────────────────┘
```

---

## 🎯 Composants principaux

### 1. **Fenêtre_Acceuil** (Fenêtre principale)

**Caractéristiques:**
- Geometry: 1200x700 (plus large que l'original)
- Layout: Sidebar à gauche + Zone de contenu à droite
- Thème: Adaptatif (clair/sombre selon le système)

**Sections:**
```python
# Section Modèle
- 📂 Charger Modèle
- ⚙️ Paramétrer Modèle

# Section Données  
- 📁 Choix Dataset
- 📅 Paramétrer Horizon

# Section Actions
- 🚀 Lancer l'entraînement
- ⛔ Annuler l'entraînement (désactivé par défaut)
```

### 2. **CTkTabview** (Onglets de résultats)

Remplace `ttk.Notebook` avec un design moderne:

**Onglets:**
- **Training** : Graphique de loss en temps réel
- **Testing** : Scatter plot prédictions vs vraies valeurs
- **Metrics** : Métriques de performance (MSE, MAE, RMSE, MAPE, R²)
- **Prediction** : Fonctionnalité à venir

### 3. **Fenetre_Params** (Configuration du modèle)

**Design moderne:**
- ScrollableFrame pour navigation fluide
- CTkSegmentedButton pour choix du modèle (MLP/CNN/LSTM)
- Sections bien délimitées avec titres emoji
- Boutons avec style transparent pour annulation

**Sections:**
```
⚙️ Configuration du Modèle
├── Type de Modèle (Segmented Button)
├── 🧠/🔲/🔄 Paramètres spécifiques
├── ⚙️ Configuration de l'entraînement
│   ├── Fonction de Perte
│   ├── Optimiseur
│   └── Learning Rate
└── 📊 Paramètres d'entraînement
    ├── Nombre d'époques
    └── Batch Size
```

### 4. **Fenetre_Params_horizon** (Paramètres temporels)

**Interface épurée:**
- Entrées alignées proprement
- Boutons calendrier pour sélection de dates
- Design cohérent avec le reste de l'application

### 5. **Fenetre_Choix_datasets** (Sélection de dataset)

**Améliorations:**
- OptionMenu moderne pour sélection
- Section d'informations contextuelle
- Layout plus spacieux

---

## 🎨 Styles CustomTkinter appliqués

### Boutons

```python
# Bouton principal (action positive)
ctk.CTkButton(
    text="🚀 Lancer",
    height=40,
    # Couleurs par défaut CustomTkinter (bleu)
)

# Bouton secondaire (action neutre)
ctk.CTkButton(
    text="❌ Annuler",
    fg_color="transparent",
    border_width=2,
    text_color=("gray10", "gray90")
)
```

### Frames

```python
# Frame avec coins arrondis (défaut)
ctk.CTkFrame(
    parent,
    corner_radius=10  # Valeur par défaut
)

# Frame transparent (pour layout)
ctk.CTkFrame(
    parent,
    fg_color="transparent"
)
```

### Labels

```python
# Label titre
ctk.CTkLabel(
    text="MLApp",
    font=("Roboto Medium", 24)
)

# Label section
ctk.CTkLabel(
    text="🧬 Modèle",
    font=("Roboto Medium", 16)
)
```

### Entrées de texte

```python
# Entry moderne avec coins arrondis
ctk.CTkEntry(
    textvariable=var,
    width=150,
    height=35
)
```

### OptionMenu (menus déroulants)

```python
# Menu déroulant stylé
ctk.CTkOptionMenu(
    values=["Option 1", "Option 2"],
    variable=var,
    width=150,
    dropdown_font=("Roboto", 12)
)
```

### SegmentedButton (choix exclusifs)

```python
# Bouton segmenté moderne
ctk.CTkSegmentedButton(
    values=["MLP", "CNN", "LSTM"],
    variable=model_var
)
```

### CheckBox

```python
# Case à cocher moderne
ctk.CTkCheckBox(
    text="Option",
    variable=bool_var,
    font=("Roboto", 12)
)
```

### Textbox (zone de texte)

```python
# Zone de texte scrollable
ctk.CTkTextbox(
    font=("Roboto Mono", 13),
    wrap="word"
)
```

---

## 🌓 Mode clair/sombre automatique

CustomTkinter détecte automatiquement le thème du système:

### Mode clair
- Arrière-plan: Blanc/Gris clair
- Texte: Noir/Gris foncé
- Boutons: Bleu vibrant

### Mode sombre
- Arrière-plan: Gris foncé/Noir
- Texte: Blanc/Gris clair
- Boutons: Bleu adouci

**Activation:**
```python
ctk.set_appearance_mode("system")  # Auto
# ou
ctk.set_appearance_mode("light")   # Forcé clair
ctk.set_appearance_mode("dark")    # Forcé sombre
```

---

## 📊 Graphiques Matplotlib

Les graphiques restent identiques mais s'intègrent parfaitement:

### Training Graph
```python
# Ligne bleue avec grille légère
self.ax.plot(epochs, losses, 'b-', linewidth=2)
self.ax.grid(True, alpha=0.3)
```

### Testing Graph
```python
# Scatter plot avec ligne de référence
self.ax.scatter(y_true, y_pred, alpha=0.6, s=50)
self.ax.plot([min, max], [min, max], 'r--', linewidth=2)
```

### Metrics Display
```python
# Affichage formaté avec bordures
═══════════════════════════════════════
          RÉSULTATS DU TEST
═══════════════════════════════════════
  MSE  (Mean Squared Error)      0.052143
  MAE  (Mean Absolute Error)     0.183254
  ...
```

---

## 🎯 Comparaison Tkinter vs CustomTkinter

| Aspect | Tkinter Original | CustomTkinter Moderne |
|--------|-----------------|----------------------|
| Apparence | Native système | Moderne, cohérente |
| Thèmes | Limités | Clair/Sombre auto |
| Coins | Carrés | Arrondis |
| Polices | TkDefaultFont | Roboto |
| Couleurs | Fixes | Adaptatives |
| Transitions | Aucune | Douces |
| Widgets | Basiques | Améliorés |

---

## 🚀 Utilisation

### Installation

```bash
pip install customtkinter
```

### Lancement

```bash
python interface_customtkinter.py
```

### Configuration système

Le mode d'apparence s'adapte automatiquement:
- **macOS**: Préférences Système > Général > Apparence
- **Windows 10/11**: Paramètres > Personnalisation > Couleurs
- **Linux**: Selon le gestionnaire de thème

---

## 🎨 Personnalisation avancée

### Changer le thème de couleur

```python
# Thèmes disponibles
ctk.set_default_color_theme("blue")      # Bleu (défaut)
ctk.set_default_color_theme("green")     # Vert
ctk.set_default_color_theme("dark-blue") # Bleu foncé

# Ou créer un thème personnalisé (fichier JSON)
ctk.set_default_color_theme("mon_theme.json")
```

### Ajuster le scaling

```python
# Pour écrans haute résolution
ctk.set_widget_scaling(1.5)  # 150%
ctk.set_window_scaling(1.5)  # 150%
```

### Couleurs personnalisées

```python
# Bouton avec couleurs custom
ctk.CTkButton(
    text="Custom",
    fg_color="#FF6B6B",        # Rouge coral
    hover_color="#FF5252",     # Rouge plus foncé
    text_color="#FFFFFF"       # Blanc
)
```

---

## 📦 Structure du code

```
interface_customtkinter.py
├── Imports et configuration
│   ├── customtkinter
│   ├── matplotlib
│   └── configuration globale
│
├── Classes de paramètres (inchangées)
│   ├── Parametres_temporels_class
│   ├── Parametres_choix_reseau_neurones_class
│   └── ...
│
├── Fenetre_Acceuil (CTk)
│   ├── Sidebar gauche
│   ├── Zone de contenu (TabView)
│   └── Méthodes d'action
│
├── Cadres de visualisation
│   ├── Cadre_Entrainement (graphique loss)
│   ├── Cadre_Testing (scatter plot)
│   ├── Cadre_Metrics (textbox)
│   └── Cadre_Prediction (placeholder)
│
└── Fenêtres de configuration (CTkToplevel)
    ├── Fenetre_Params
    ├── Fenetre_Params_horizon
    └── Fenetre_Choix_datasets
```

---

## ✅ Fonctionnalités implémentées

### Configuration
- ✅ Choix du type de modèle (MLP/CNN/LSTM)
- ✅ Paramétrage complet de l'architecture
- ✅ Configuration de la loss et de l'optimiseur
- ✅ Paramètres d'entraînement
- ✅ Sélection de dataset
- ✅ Configuration temporelle (horizon, dates, etc.)

### Entraînement
- ✅ Lancement asynchrone (threading)
- ✅ Streaming SSE en temps réel
- ✅ Mise à jour dynamique du graphique
- ✅ Annulation possible
- ✅ Gestion des erreurs

### Visualisation
- ✅ Graphique de loss pendant l'entraînement
- ✅ Scatter plot des prédictions vs vraies valeurs
- ✅ Affichage des métriques de performance
- ✅ Interface responsive

---

## 🎓 Exemples de code

### Créer un bouton moderne

```python
bouton = ctk.CTkButton(
    parent,
    text="Mon Bouton",
    font=("Roboto", 13),
    height=40,
    corner_radius=8,
    command=ma_fonction
)
bouton.pack(pady=10, padx=20)
```

### Créer un frame avec contenu

```python
frame = ctk.CTkFrame(parent, corner_radius=15)
frame.pack(fill="both", expand=True, padx=20, pady=20)

# Ajouter un titre
ctk.CTkLabel(
    frame,
    text="Mon Titre",
    font=("Roboto Medium", 18)
).pack(pady=(20, 10))

# Ajouter du contenu
content = ctk.CTkFrame(frame)
content.pack(fill="both", expand=True, padx=20, pady=20)
```

### Créer un formulaire

```python
form = ctk.CTkFrame(parent)
form.pack(fill="x", padx=30, pady=10)

# Champ avec label
field_frame = ctk.CTkFrame(form, fg_color="transparent")
field_frame.pack(fill="x", pady=5)

ctk.CTkLabel(
    field_frame,
    text="Nom:",
    font=("Roboto", 12)
).pack(side="left", padx=(0, 10))

entry_var = ctk.StringVar()
ctk.CTkEntry(
    field_frame,
    textvariable=entry_var,
    width=200
).pack(side="right")
```

---

## 🐛 Debugging

### Problème: Interface trop petite/grande

**Solution:** Ajuster le scaling
```python
ctk.set_widget_scaling(1.25)  # 125%
```

### Problème: Thème ne change pas

**Solution:** Forcer le mode
```python
ctk.set_appearance_mode("light")  # ou "dark"
```

### Problème: Polices non trouvées

**Solution:** Utiliser des polices système
```python
font=("Helvetica", 14)  # Au lieu de "Roboto"
```

---

## 📚 Ressources

- **Documentation CustomTkinter:** https://customtkinter.tomschimansky.com/
- **GitHub:** https://github.com/TomSchimansky/CustomTkinter
- **Wiki:** https://github.com/TomSchimansky/CustomTkinter/wiki
- **Exemples:** https://github.com/TomSchimansky/CustomTkinter/tree/master/examples

---

## 🎉 Conclusion

Cette interface moderne avec CustomTkinter offre:

1. **✨ Design contemporain** - Interface élégante et professionnelle
2. **🌓 Thème adaptatif** - S'adapte au système (clair/sombre)
3. **🎨 Cohérence visuelle** - Tous les éléments harmonisés
4. **📱 Responsive** - S'adapte aux différentes tailles d'écran
5. **🚀 Performance** - Aussi rapide que Tkinter
6. **🔧 Maintenable** - Code propre et structuré

L'application est prête à l'emploi et peut être facilement étendue avec de nouvelles fonctionnalités !

---

**Date:** 3 Novembre 2025  
**Version CustomTkinter:** 5.2.0+  
**Python:** 3.8+
