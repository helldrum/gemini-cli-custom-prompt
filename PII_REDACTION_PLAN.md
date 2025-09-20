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

_Ce document sera mis à jour au fur et à mesure de l'avancement._
