import json
class Parametres_temporels_class():
    def __init__(self):
        #self.dataset="" # str
        self.__horizon=1 # int
        self.dates=["2001-01-01", "2025-01-02"] # variable datetime
        self.pas_temporel=1 # int
        self.portion_decoupage=0.8# float entre 0 et 1
aaa=Parametres_temporels_class()
config_totale={}
config_totale["Parametres_temporels"]=aaa.__dict__
print(config_totale)