# PROJET_GL - Application de Prediction de Series Temporelles

## Presentation du Projet

PROJET_GL est une application de machine learning dediee a la prediction de series temporelles. Le projet est structure autour d'une architecture client-serveur composee de trois elements principaux :

- **Serveur DATA** : Gestion et persistance des donnees (datasets, modeles entraines, contextes)
- **Serveur IA** : Operations de machine learning (entrainement, validation, prediction)
- **Interface utilisateur** : Disponible en plusieurs versions (JavaScript/React ou Python/Tkinter)

### Fonctionnalites

- Import et gestion de datasets de series temporelles (format JSON)
- Configuration des parametres d'entrainement (horizon, pas temporel, decoupage train/test)
- Support de plusieurs architectures de reseaux de neurones : MLP, CNN, LSTM
- Entrainement avec suivi en temps reel des metriques (loss, validation)
- Prediction avec intervalles de confiance
- Sauvegarde et chargement de modeles entraines

---

## Architecture

```
                    Interface Utilisateur
                    (React/Electron ou Python/Tkinter)
                              |
                              v
              +-------------------------------+
              |       SERVEUR_IA              |
              |       (Port 8000)             |
              |                               |
              |  - Entrainement des modeles   |
              |  - Predictions                |
              |  - Validation                 |
              +-------------------------------+
                              |
                              v
              +-------------------------------+
              |       SERVEUR_DATA            |
              |       (Port 8001)             |
              |                               |
              |  - Stockage des datasets      |
              |  - Gestion des modeles        |
              |  - Persistance des contextes  |
              +-------------------------------+
```

---

## Interfaces Disponibles

### Interface JavaScript / TypeScript / React / Electron (Recommandee)

Situee dans le dossier `INTERFACE_JS/`, c'est l'interface principale et la plus complete.

**Technologies utilisees :**
- React 18 avec TypeScript
- Vite pour le bundling et le serveur de developpement
- Electron pour l'application desktop
- Tailwind CSS pour le style
- Recharts pour les graphiques
- Zustand pour la gestion d'etat
- Axios pour les requetes HTTP

**Fonctionnalites :**
- Gestion complete des datasets (upload, visualisation, suppression)
- Configuration des parametres temporels (dates, horizon, pas temporel)
- Selection et configuration de l'architecture du reseau de neurones
- Suivi en temps reel de l'entrainement avec graphiques
- Visualisation des predictions avec intervalles de confiance
- Interface moderne et responsive

### Interfaces Python / Tkinter (Legacy)

Situees dans le dossier `INTERFACE/`, ces interfaces utilisent CustomTkinter pour une interface graphique native.

**Fichiers :**
- `interface_local_ctk.py` : Interface principale avec gestion complete
- `interface_predeiction.py` : Interface simplifiee orientee prediction

**Technologies utilisees :**
- CustomTkinter pour l'interface graphique
- Matplotlib pour les graphiques
- TkCalendar pour la selection des dates

Ces interfaces sont conservees pour la compatibilite et les cas d'usage locaux, mais l'interface JavaScript est recommandee pour une utilisation en production.

Pour plus d'information veuillez lire le cahier des charges qui reprend en détail l'entièreté (ou presque) du projet : [CAHIER DES CHARGES](assets/CDC_FINAL/Cahier_des_charges_MLApp_V2_0 (1).pdf)
---

## Structure du Projet

```
PROJET_GL/
|-- SERVEUR_IA/           # Serveur FastAPI pour le ML
|   |-- main.py           # Point d'entree et routes API
|   |-- classes.py        # Modeles Pydantic
|   |-- models/           # Definitions des reseaux de neurones
|   |-- trains/           # Logique d'entrainement
|   +-- test/             # Tests et strategies de prediction
|
|-- SERVEUR_DATA/         # Serveur FastAPI pour les donnees
|   |-- main.py           # Point d'entree et routes API
|   |-- datasets/         # Fichiers JSON des series temporelles
|   |-- models/           # Modeles entraines (.pth)
|   +-- contextes/        # Configurations sauvegardees
|
|-- INTERFACE_JS/         # Interface React/Electron
|   |-- src/              # Code source React
|   |-- package.json      # Dependances npm
|   +-- vite.config.ts    # Configuration Vite et proxy
|
|-- INTERFACE/            # Interface Python/Tkinter
|   |-- interface_local_ctk.py
|   +-- Themes/           # Themes CustomTkinter
|
|-- requirements.txt      # Dependances Python
|-- environment.yml       # Environnement Conda/Mamba
|-- Ex_IA.sh             # Script de lancement serveur IA
+-- Ex_DATA.sh           # Script de lancement serveur DATA
```

---

## Quickstart

### 1. Creation de l'environnement Python

#### Option A : Avec pip (virtualenv)

```bash
# Creer l'environnement virtuel
python -m venv venv

# Activer l'environnement
# Sur Linux/macOS :
source venv/bin/activate
# Sur Windows :
venv\Scripts\activate

# Mettre a jour pip et installer les dependances
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B : Avec Mamba (recommande pour les performances)

```bash
# Creer l'environnement
mamba create -n projetgl python=3.11

# Activer l'environnement
mamba activate projetgl

# Installer les dependances
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Option C : Avec Conda

```bash
# Creer l'environnement depuis le fichier yml
conda env create -f environment.yml

# Activer l'environnement
conda activate projetgl
```

### 2. Configuration des adresses IP (pour utilisation en local)

Pour une utilisation en local, configurez les adresses IP sur `localhost` ou `127.0.0.1`.

#### Configuration du Serveur IA

Fichier : `SERVEUR_IA/main.py` (ligne 59)

```python
# Modifier cette ligne pour pointer vers le serveur DATA
DATA_SERVER_URL = os.getenv("DATA_SERVER_URL", "http://127.0.0.1:8001")
```

#### Configuration de l'Interface JavaScript

Fichier : `INTERFACE_JS/vite.config.ts` (ligne 5)

```typescript
// Modifier cette ligne pour pointer vers le serveur IA
const URL = 'http://127.0.0.1:8000'
```

#### Configuration de l'Interface Python (si utilisee)

Fichier : `INTERFACE/interface_local_ctk.py` (ligne 34)

```python
# Modifier cette ligne pour pointer vers le serveur IA
URL = "http://127.0.0.1:8000"
```

### 3. Lancement des serveurs

Ouvrez trois terminaux distincts et placez-vous a la racine du projet dans chacun.

#### Terminal 1 : Serveur DATA (port 8001)

```bash
# Activer l'environnement Python
mamba activate projetgl  # ou source venv/bin/activate

# Se placer a la racine du projet
cd /chemin/vers/PROJET_GL

# Lancer le serveur DATA
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
python -m uvicorn SERVEUR_DATA.main:app --host 0.0.0.0 --port 8001 --reload
```

Ou utilisez le script fourni :
```bash
./Ex_DATA.sh
```

#### Terminal 2 : Serveur IA (port 8000)

```bash
# Activer l'environnement Python
mamba activate projetgl  # ou source venv/bin/activate

# Se placer a la racine du projet
cd /chemin/vers/PROJET_GL

# Lancer le serveur IA
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
python -m uvicorn SERVEUR_IA.main:app --host 0.0.0.0 --port 8000 --reload
```

Ou utilisez le script fourni :
```bash
./Ex_IA.sh
```

#### Terminal 3 : Interface JavaScript

```bash
# Se placer dans le dossier de l'interface
cd /chemin/vers/PROJET_GL/INTERFACE_JS

# Installer les dependances Node.js (premiere fois uniquement)
npm install

# Lancer le serveur de developpement
npm run dev
```

**Prerequis :** Node.js doit etre installe sur votre machine. Telechargez-le depuis [nodejs.org](https://nodejs.org/).

### 4. Acces a l'application

Une fois les trois serveurs lances :

- Ouvrez votre navigateur web
- Accedez a l'adresse : `http://localhost:5173`

L'interface de l'application devrait s'afficher et etre fonctionnelle.

---

## Modeles de Reseaux de Neurones Supportes

| Modele | Description | Cas d'usage |
|--------|-------------|-------------|
| **MLP** | Multi-Layer Perceptron | Series temporelles simples, relations non-lineaires |
| **CNN** | Convolutional Neural Network | Detection de motifs locaux dans les sequences |
| **LSTM** | Long Short-Term Memory | Series avec dependances temporelles longues |

### Parametres d'entrainement configurables

- **Optimiseurs** : Adam, SGD, RMSprop, Adagrad, Adadelta
- **Fonctions de perte** : MSE, MAE, Huber
- **Hyperparametres** : learning rate, batch size, nombre d'epoques, patience (early stopping)

---

## Datasets Inclus

Le projet inclut plusieurs datasets d'exemple dans `SERVEUR_DATA/datasets/` :

- `EURO.json` : Taux de change EUR/USD
- `CACAO.json` : Prix du cacao
- `timeseries_data.json` : Serie temporelle generique
- `marees_saint_jean_de_luz.json` : Donnees de marees

Vous pouvez ajouter vos propres datasets au format JSON avec la structure suivante :

```json
{
  "timestamps": ["2024-01-01", "2024-01-02", ...],
  "values": [1.23, 1.45, ...]
}
```

---

## Tests

Des tests d'integration sont disponibles dans plusieurs dossiers :

- `SERVEUR_DATA/TESTS/` : Tests du serveur DATA
- `SERVEUR_IA/test/` : Tests du serveur IA
- `TEST_GLOBAL_ROUTES/` : Tests end-to-end

Pour executer les tests :

```bash
# Depuis la racine du projet
python -m pytest TEST_GLOBAL_ROUTES/
```

---

## Variables d'environnement

| Variable | Description | Valeur par defaut |
|----------|-------------|-------------------|
| `DATA_SERVER_URL` | URL du serveur DATA | `http://192.168.1.94:8001` |
| `KMP_DUPLICATE_LIB_OK` | Fix pour Intel MKL | `TRUE` |
| `OMP_NUM_THREADS` | Nombre de threads OpenMP | `1` |

---

## Dependances Principales

### Python
- FastAPI / Uvicorn (serveurs web)
- PyTorch (machine learning)
- NumPy / Pandas (manipulation de donnees)
- Scikit-learn (utilitaires ML)
- Pydantic (validation de donnees)

### JavaScript
- React / TypeScript
- Vite / Electron
- Tailwind CSS
- Recharts / Axios / Zustand
