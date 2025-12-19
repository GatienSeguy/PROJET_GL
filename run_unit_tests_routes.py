from __future__ import annotations

import unittest
from types import SimpleNamespace


class FakeResponse:
    def __init__(self, status_code: int = 200, json_data=None, headers=None):
        self.status_code = status_code
        self._json_data = json_data if json_data is not None else {}
        self.headers = headers or {"content-type": "application/json"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._json_data


class RequestsMock:
    """
    Minimal requests mock that records calls.
    """
    def __init__(self):
        self.calls = []  # list of (method, url, kwargs)
        self._routes = {}  # (method, url)->FakeResponse

    def when(self, method: str, url: str, response: FakeResponse):
        self._routes[(method.upper(), url)] = response
        return self

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self._routes.get(("POST", url), FakeResponse(200, {}))

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self._routes.get(("GET", url), FakeResponse(200, {}))


class TestUIOnly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import UI module once
        import TEST_interface_local_ctk as ui
        cls.ui = ui

    def setUp(self):
        # Fresh requests mock each test
        self.req = RequestsMock()
        # Patch requests inside the UI module
        self._old_requests = self.ui.requests
        self.ui.requests = self.req

        # Patch URL base (UI uses module-level URL)
        self._old_url = getattr(self.ui, "URL", None)
        self.ui.URL = "http://fake-ia:8000"

        # Create a "window" instance without running CTk init
        # (we only test methods that don't need the GUI to exist)
        self.win = self.ui.Fenetre_Acceuil.__new__(self.ui.Fenetre_Acceuil)

    def tearDown(self):
        # Restore patches
        self.ui.requests = self._old_requests
        if self._old_url is not None:
            self.ui.URL = self._old_url

    def test_formatter_json_dataset(self):
        # Arrange: set the global parameter singleton used by the UI
        self.ui.Parametres_temporels.nom_dataset = "TEST_DATASET"
        self.ui.Parametres_temporels.dates = ["2024-01-01 00:00:00", "2024-01-01 03:00:00"]
        self.ui.Parametres_temporels.pas_temporel = 1

        # Act
        payload = self.ui.Fenetre_Acceuil.Formatter_JSON_dataset(self.win)

        # Assert (stable schema)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["name"], "TEST_DATASET")
        self.assertEqual(payload["dates"][0], "2024-01-01 00:00:00")
        self.assertEqual(payload["pas_temporel"], 1)

    def test_formatter_json_specif_mlp(self):
        self.ui.Parametres_choix_reseau_neurones.modele = "MLP"
        # Ensure the object exists and has a dict
        self.ui.Parametres_archi_reseau_MLP.in_dim = 10
        self.ui.Parametres_archi_reseau_MLP.hidden_dims = [32, 32]
        self.ui.Parametres_archi_reseau_MLP.out_dim = 1

        payload = self.ui.Fenetre_Acceuil.Formatter_JSON_specif(self.win)
        self.assertIn("Parametres_archi_reseau", payload)
        self.assertIsInstance(payload["Parametres_archi_reseau"], dict)
        # Not all fields are mandatory, but at least these should exist in MLP config
        self.assertIn("in_dim", payload["Parametres_archi_reseau"])

    def test_obtenir_datasets_calls_expected_route(self):
        url = f"{self.ui.URL}/datasets/info_all"
        expected = {"test1": 1, "test2": 2}
        self.req.when("POST", url, FakeResponse(200, expected))

        data = self.ui.Fenetre_Acceuil.obtenir_datasets(self.win)

        # Return value is server json
        self.assertEqual(data, expected)

        # Verify call recorded
        self.assertEqual(len(self.req.calls), 1)
        method, called_url, kwargs = self.req.calls[0]
        self.assertEqual(method, "POST")
        self.assertEqual(called_url, url)
        self.assertIn("json", kwargs)
        self.assertEqual(kwargs["json"], {"message": "choix dataset"})

    def test_obtenir_datasets_handles_http_error(self):
        url = f"{self.ui.URL}/datasets/info_all"
        self.req.when("POST", url, FakeResponse(500, {"detail": "boom"}))

        data = self.ui.Fenetre_Acceuil.obtenir_datasets(self.win)
        self.assertIsNone(data)  # UI catches exceptions and returns None


if __name__ == "__main__":
    unittest.main(verbosity=2)
