README - Scripts Git (push/pull)

But: Fournir des scripts simples pour push et pull du dépôt.

Fichiers ajoutés:
- scripts/git_push_all.sh : Ajoute, commit (si besoin) et pousse la branche courante vers le remote (par défaut `origin`).
- scripts/git_pull_all.sh : Fait `fetch` puis `rebase` de la branche courante depuis le remote (par défaut `origin`).

Usage rapide:

1) Rendre les scripts exécutables (une seule fois):

   chmod +x scripts/git_push_all.sh scripts/git_pull_all.sh

2) Pour pousser les changements:

   scripts/git_push_all.sh -m "Mon message de commit"

Options:
  -m  message de commit (défaut: "update")
  -r  remote (défaut: origin)

3) Pour récupérer les changements et rebaser:

   scripts/git_pull_all.sh

Options:
  -r  remote (défaut: origin)

Bonnes pratiques recommandées:
- Configurez une clé SSH et ajoutez le remote en SSH pour éviter de rentrer vos identifiants à chaque push.
- Faites des commits petits et atomiques avec des messages clairs.
- Utilisez des branches de fonctionnalité et ouvrez des Pull Requests si vous collaborez.
- Avant d'envoyer un push massif, vérifiez `git status` et `git log -n 5`.

Dépannage:
- Si le rebase échoue: résolvez les conflits, `git rebase --continue`, ou annulez avec `git rebase --abort`.
- Pour forcer le push (danger): `git push --force-with-lease` plutôt que `--force`.

Si vous voulez, je peux:
- Ajouter un script pour créer une nouvelle branche et la pousser.
- Ajouter un script qui effectue pull puis merge au lieu de rebase.
- Générer un petit alias git à ajouter à votre `.bashrc`.

