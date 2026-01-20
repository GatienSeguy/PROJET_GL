#!/usr/bin/env python3
"""
test_datasets_functionnels.py

Tests fonctionnels pour les routes /datasets/* de ton serveur IA.
Utilise le dataset sofiane_est_un_dieu.json.
Affiche ✅ pour succès, ❌ pour échec.
Continue malgré les erreurs/timeouts.
"""

import requests
import json
from pathlib import Path
from typing import List

BASE_URL = "http://127.0.0.1:8000"  # serveur IA
DATASET_FILE = Path(__file__).parent / "Dataset_test_global.json"
DATASET_NAME = "sofiane_est_un_dieu"
FAILURES: List[str] = []

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def ok(msg: str):
    print(f"{GREEN}✅ {msg}{RESET}")

def nok(msg: str):
    print(f"{RED}❌ {msg}{RESET}")

def info(msg: str):
    print(f"{YELLOW}ℹ️  {msg}{RESET}")

def safe_post(path: str, payload, timeout=15):
    url = f"{BASE_URL}{path}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        return r
    except requests.exceptions.RequestException as e:
        return e

# -----------------------
# TESTS FONCTIONNELS DATASETS
# -----------------------

def test_send_dataset_first_time():
    info(f"[send_dataset_first_time] Envoi du dataset {DATASET_NAME}")
    with DATASET_FILE.open("r", encoding="utf-8") as f:
        payload = {"name": DATASET_NAME, "data": json.load(f)}
    resp = safe_post("/datasets/data_add_proxy", payload)
    if isinstance(resp, Exception):
        nok(f"send_dataset_first_time : exception request -> {resp}")
        FAILURES.append(f"send_dataset_first_time: request exception {resp}")
        return
    if resp.status_code == 200:
        ok(f"send_dataset_first_time : dataset envoyé OK")
    else:
        nok(f"send_dataset_first_time : échec, HTTP {resp.status_code} body={resp.text[:200]}")
        FAILURES.append(f"send_dataset_first_time: HTTP {resp.status_code}")

def test_send_dataset_duplicate():
    info(f"[send_dataset_duplicate] Envoi duplicate du dataset {DATASET_NAME}")
    with DATASET_FILE.open("r", encoding="utf-8") as f:
        payload = {"name": DATASET_NAME, "data": json.load(f)}
    resp = safe_post("/datasets/data_add_proxy", payload)
    if isinstance(resp, Exception):
        nok(f"send_dataset_duplicate : exception request -> {resp}")
        FAILURES.append(f"send_dataset_duplicate: request exception {resp}")
        return
    # Attendu : erreur côté DATA server / IA
    if resp.status_code != 200:
        ok(f"send_dataset_duplicate : doublon détecté (HTTP {resp.status_code})")
    else:
        try:
            body = resp.json()
            if "already exists" in str(body).lower() or body.get("status") == "error":
                ok(f"send_dataset_duplicate : doublon détecté via payload (OK)")
            else:
                nok(f"send_dataset_duplicate : doublon non détecté, HTTP200 body={body}")
                FAILURES.append(f"send_dataset_duplicate: duplicate not detected")
        except Exception:
            nok(f"send_dataset_duplicate : HTTP200 mais body non JSON")
            FAILURES.append(f"send_dataset_duplicate: duplicate check failed")

def test_fetch_dataset_valid():
    info(f"[fetch_dataset_valid] Fetch dataset {DATASET_NAME}")
    payload = {"name": DATASET_NAME, "dates": ["2000-01-01", "2025-01-01"]}
    resp = safe_post("/datasets/fetch_dataset", payload)
    if isinstance(resp, Exception):
        nok(f"fetch_dataset_valid : exception request -> {resp}")
        FAILURES.append(f"fetch_dataset_valid: request exception {resp}")
        return
    if resp.status_code != 200:
        nok(f"fetch_dataset_valid : HTTP {resp.status_code}")
        FAILURES.append(f"fetch_dataset_valid: HTTP {resp.status_code}")
        return
    try:
        body = resp.json()
        if "status" in body and body["status"] == "success":
            ok(f"fetch_dataset_valid : dataset fetch OK, {len(body['data']['timestamps'])} points")
        else:
            ok(f"fetch_dataset_valid : dataset fetch OK, keys={list(body.keys())}")
    except Exception as e:
        nok(f"fetch_dataset_valid : JSON decode error {e}")
        FAILURES.append(f"fetch_dataset_valid: JSON decode error {e}")

def test_delete_dataset_nonexistent():
    name = "dataset_inexistant_123"
    info(f"[delete_dataset_nonexistent] Supprimer dataset inexistant {name}")
    payload = {"name": name}
    resp = safe_post("/datasets/data_suppression_proxy", payload)
    if isinstance(resp, Exception):
        nok(f"delete_dataset_nonexistent : exception request -> {resp}")
        FAILURES.append(f"delete_dataset_nonexistent: request exception {resp}")
        return
    if resp.status_code != 200:
        ok(f"delete_dataset_nonexistent : suppression échoue comme attendu (HTTP {resp.status_code})")
        return
    try:
        body = resp.json()
        if body.get("status") == "error" or "not found" in str(body).lower():
            ok(f"delete_dataset_nonexistent : message d'erreur reçu comme prévu")
        else:
            nok(f"delete_dataset_nonexistent : attendu erreur mais body={body}")
            FAILURES.append(f"delete_dataset_nonexistent: expected error but got OK")
    except Exception:
        ok(f"delete_dataset_nonexistent : HTTP200 mais body non JSON, accepté")

# -----------------------
# RUNNER
# -----------------------
def run_all():
    tests = [
        test_send_dataset_first_time,
        test_send_dataset_duplicate,
        test_fetch_dataset_valid,
        test_delete_dataset_nonexistent,
    ]

    info("Démarrage des tests fonctionnels datasets ...")
    for t in tests:
        try:
            t()
        except Exception as e:
            nok(f"EXCEPTION non capturée dans {t.__name__}: {e}")
            FAILURES.append(f"{t.__name__}: uncaught exception {e}")

    print("\n" + "="*60)
    info("RÉSUMÉ FINAL")
    if not FAILURES:
        ok("Tous les tests fonctionnels datasets sont passés ✅")
    else:
        nok(f"{len(FAILURES)} test(s) ont échoué :")
        for f in FAILURES:
            print(f" - {f}")
        raise RuntimeError(f"{len(FAILURES)} test(s) failed; voir output ci-dessus")

if __name__ == "__main__":
    run_all()
