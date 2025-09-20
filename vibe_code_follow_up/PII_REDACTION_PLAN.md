# Plan de Caviardage des Données Sensibles (PII)

Ce document suit l'avancement de l'implémentation de la fonctionnalité d'anonymisation des prompts.

les regex de prompt sont dans le fichier packages/core/src/pii/gitleaksFilters.ts

## Objectif

L'objectif principal est d'empêcher l'envoi d'informations personnelles identifiables (PII) et de secrets (clés d'API, tokens, mots de passe) aux serveurs de Google. Pour ce faire, nous devons filtrer le contenu des prompts de l'utilisateur avant chaque envoi.

## Approche Choisie

- **Format de caviardage :** Pour les paires clé-valeur, la clé reste en clair et seule la valeur est caviardée. Pour les blocs de texte (clés privées, chaînes Base64 complètes), le bloc entier est caviardé.
- **Méthode :** Application directe d'expressions régulières personnalisées au sein du `GeminiClient`, avec une priorité pour la détection et le caviardage complet des chaînes Base64 valides et des blocs JSON de clés de compte de service GCP en clair.

## Plan d'Action

1.  [x] Définir l'objectif et choisir l'approche technique.
2.  [x] Créer ce fichier de suivi (`PII_REDACTION_PLAN.md`).
3.  [x] Implémenter la logique d'application directe des regex au sein du `GeminiClient`, avec une priorité pour la détection et le caviardage complet des chaînes Base64 valides et des blocs JSON de clés de compte de service GCP en clair.
4.  [x] Créer un fichier de configuration pour les filtres personnalisés (`packages/core/src/pii/gitleaksFilters.ts`).
5.  [x] Adapter les regex de `gitleaks` pour une application directe.
6.  [x] Intégrer la logique d'anonymisation dans le `GeminiClient` pour filtrer les prompts.
7.  [x] Tester la solution avec des exemples de secrets.

## État Actuel et Tests (12/09/2025)

Des tests ont été effectués avec un prompt contenant divers secrets.

**Ce qui est correctement caviardé :**

- Clé d'accès AWS (`AKIA...`) (règle améliorée)
- Secret AWS (chaîne de 40 caractères) (faux positifs réduits)
- Clé d'API GCP (`AIza...`)
- **Clé privée :** Le bloc complet de la clé privée est maintenant correctement caviardé.
- **Email :** Les adresses e-mail sont maintenant correctement détectées et caviardées.
- **Clé de compte de service GCP (JSON) :** Les blocs JSON de clés de compte de service GCP en clair sont caviardés partiellement par les règles génériques.
- **Contenu Base64 :** Les chaînes Base64 valides sont entièrement caviardées.
- **Numéro de téléphone :** Les numéros de téléphone (y compris les formats avec points comme 08.36.65.65.65) sont maintenant correctement détectées et caviardées.
- **Chaînes de connexion (JDBC, Redis, WSS) :** Les chaînes de connexion sensibles sont maintenant correctement détectées et caviardées.

**Ce qui n'est PAS (ou mal) caviardé :**

- **Fonction de désactivation :** La fonction de désactivation du caviardage ne fonctionne pas, ce qui est un problème majeur.
- **Contenu Base64 :** La gestion des chaînes Base64 doit être affinée pour des cas d'utilisation spécifiques (par exemple, si le contenu décodé doit être analysé par d'autres regex).

## Prochaines Étapes

- [ ] Implémenter une fonction spécifique pour détecter et caviarder toutes les chaînes Base64 valides avant l'application des règles de caviardage génériques. Le contenu décodé ne doit pas être analysé par d'autres regex.
- [ ] Supprimer la règle `Long Base64 String` de `gitleaksFilters.ts`.
- [x] Affiner la règle pour les secrets de 40 caractères afin de réduire les faux positifs.
- [x] Améliorer la règle pour les clés d'accès AWS (`AKIA...`) pour être plus spécifique.

---

## Mises à jour et Problèmes Récents (20/09/2025)

### Progrès

- Le fichier de règles de caviardage `packages/core/src/pii/gitleaksFilters.ts` a été identifié et son contenu est correct.
- La logique d'activation/désactivation de l'anonymisation (`anonymizationEnabled`, `isAnonymizationEnabled`, `setAnonymizationEnabled`) est présente dans `packages/core/src/config/config.ts` et utilisée dans `packages/core/src/core/client.ts`.
- La commande `/anonymize` (implémentée dans `packages/cli/src/ui/commands/anonymizeCommand.ts`) a été récupérée. Elle permet de changer l'état de l'anonymisation via les arguments `on` ou `off`.

### Problèmes Rencontrés

- **Intégration de la commande `/anonymize` :** Des difficultés persistantes ont été rencontrées pour ajouter la commande `/anonymize` au système de commandes principal (`gemini.tsx`). Les tentatives d'insertion via `replace` ont échoué à plusieurs reprises en raison de problèmes de correspondance exacte de la `old_string` (indentation, caractères invisibles, etc.). Une approche de réécriture complète du fichier a été envisagée mais non exécutée.
- **Fonction de désactivation du caviardage :** Le problème de la fonction de désactivation qui ne fonctionne pas reste à investiguer. La logique de la commande `/anonymize` est correcte, ce qui suggère que le problème se situe ailleurs, potentiellement dans la persistance de la configuration ou la manière dont le `GeminiClient` lit l'état de l'anonymisation.
- **Gestion du Base64 :** La demande d'implémentation d'une fonction spécifique pour le caviardage des chaînes Base64 a été mise en suspens pour le moment, jugée trop coûteuse en temps et en effort par rapport aux priorités actuelles.

### Prochaines Étapes (Recommandées)

- [x] Résoudre l'intégration de la commande `/anonymize` dans `gemini.tsx` (potentiellement via une modification manuelle ou un outil plus robuste).
- [x] Débugger la fonction de désactivation du caviardage pour comprendre pourquoi elle ne prend pas effet.
- [x] Mettre en place l'intégration de la commande dans le "menu" (si ce n'est pas déjà fait par l'intégration de `yargs`).

---

---

## Mises à jour et Problèmes Récents (21/09/2025)

### Progrès

- **Commande `/anonymize` fonctionnelle :** La commande `/anonymize on/off` est désormais visible dans l'interface et pleinement fonctionnelle.
- **Correction du processus de build :** Le problème de la désactivation provenait d'un mécanisme de copie de prompts manquant (suite à un rebase), qui a été réimplémenté dans `scripts/copy_bundle_assets.js`.
- **Correction de bugs multiples :** La commande a été enregistrée dans `BuiltinCommandLoader.ts`, un bug en mode non-interactif a été corrigé, et des dépendances/workflows de build ont été réparés.
- **Nettoyage :** Des scripts obsolètes ont été supprimés et la documentation a été mise à jour.

### Prochaines Étapes (Recommandées)

- [ ] Améliorer la précision des filtres de caviardage (les filtres actuels sont fonctionnels mais perfectibles).

---

_Ce document sera mis à jour au fur et à mesure de l'avancement._
