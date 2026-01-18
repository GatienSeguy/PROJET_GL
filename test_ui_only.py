import os
import sys
import importlib.util
import inspect

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(PROJECT_ROOT, "INTERFACE", "TEST_interface_local_ctk.py")

spec = importlib.util.spec_from_file_location("ui_module", UI_PATH)
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

print("\n=== FONCTIONS EXPORTÉES ===")
for name, obj in inspect.getmembers(ui, inspect.isfunction):
    print("-", name)

print("\n=== CLASSES EXPORTÉES ===")
for name, obj in inspect.getmembers(ui, inspect.isclass):
    print("-", name)

print("\n=== VARIABLES GLOBALES (suspectes) ===")
for name in dir(ui):
    if not name.startswith("_"):
        attr = getattr(ui, name)
        if not inspect.isfunction(attr) and not inspect.isclass(attr):
            print("-", name)


import os
import sys
import unittest
import importlib.util

# ============================================================
# Chargement dynamique du module UI
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(PROJECT_ROOT, "INTERFACE", "TEST_interface_local_ctk.py")

spec = importlib.util.spec_from_file_location("ui_module", UI_PATH)
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)

# ============================================================
# Tests unitaires : état UI uniquement
# ============================================================

class TestUIStateOnly(unittest.TestCase):

    def test_selected_dataset_exists(self):
        self.assertTrue(hasattr(ui, "Selected_Dataset"))
        self.assertIsInstance(ui.Selected_Dataset, ui.Selected_Dataset_class)

    def test_selected_dataset_attributes(self):
        ds = ui.Selected_Dataset
        for attr in ["name", "dates", "pas_temporel"]:
            self.assertTrue(
                hasattr(ds, attr),
                f"Selected_Dataset missing attribute {attr}"
            )

    def test_parametres_entrainement_exists(self):
        self.assertTrue(hasattr(ui, "Parametres_entrainement"))
        self.assertIsInstance(
            ui.Parametres_entrainement,
            ui.Parametres_entrainement_class
        )

    def test_parametres_archi_mlp_exists(self):
        self.assertTrue(hasattr(ui, "Parametres_archi_reseau_MLP"))

        mlp_params = ui.Parametres_archi_reseau_MLP

        # Le paramètre MLP doit être un objet non vide
        attrs = [
            a for a in dir(mlp_params)
            if not a.startswith("_") and not callable(getattr(mlp_params, a))
        ]

        self.assertTrue(
            len(attrs) > 0,
            "MLP params object has no public attributes"
        )
        
    def test_url_is_defined(self):
        self.assertTrue(hasattr(ui, "URL"))
        self.assertIsInstance(ui.URL, str)
        self.assertTrue(ui.URL.startswith("http"))

    def test_modify_state_no_crash(self):
        ds = ui.Selected_Dataset
        ds.name = "TEST_DATASET"
        ds.dates = ["2024-01-01", "2024-01-02"]
        ds.pas_temporel = 1

        self.assertEqual(ds.name, "TEST_DATASET")
        self.assertEqual(ds.pas_temporel, 1)


if __name__ == "__main__":
    unittest.main(verbosity=3)