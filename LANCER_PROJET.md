Coucou Chems 

1 - Active ton venv : 
ex : mamba activate <nom_de_ton_env>

(pour le créer tu peux faire : 
mamba creat -n projetgl

Puis
mamba activate projetgl

Puis 
python -m pip install --upgrade pip
pip install -r requirements.txt
)

2 - Place toi à la racine du projet : 

cd <Chemin vers Projet_GL>
exemple : cd "C:\Users\Chems\Documents\Projet_GL"

3 - Lancer le serveur IA : 
python -m uvicorn SERVEUR_IA.main:app --host 0.0.0.0 --port 8000 --reload

4 - Dans un nouveau terminal : lancer le serveur DATA :
cd <Chemin vers Projet_GL>
python -m uvicorn SERVEUR_DATA.main:app --host 0.0.0.0 --port 8001 --reload

5 - Dans un troisième terminal : Installer le truc qui fait tourner l'application (pour l'instant ce sera une page web : je sais pas faire sur windows ....)  Mets toi dans INTERFACE_JS :
AVANT :
 Pour lancer l’interface JS, Windows doit avoir Node.js installé, sinon npm et npm run dev ne fonctionneront pas. Node.js contient le vrai moteur JavaScript et le gestionnaire de packages nécessaire au projet.

Après avoir installer Node.js depuis internet :
cd "...\Projet_GL\INTERFACE_JS"
npm install
npm run dev



Final : 
tape sur internet : 
localhost:5173

==> Logiquement tu te retrouves sur l'application en javascript qui tourne !

