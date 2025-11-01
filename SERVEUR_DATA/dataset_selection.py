from .classes import Tx_choix_dataset





@app.post("/datasets/select/")
def select_dataset(payload: Tx_choix_dataset):
    dataset_name = payload.name
    if dataset_name in "/home/sofsoflefoufou/Documents/Code/PROJET_GL/SERVEUR_DATA/Datas":
        
        return {"message": f"Dataset '{dataset_name}' sélectionné avec succès."}
