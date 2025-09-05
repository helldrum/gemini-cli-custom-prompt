# Plan de réalisation : Compteur de tokens et estimation de coût

Ce document décrit les étapes pour implémenter un compteur de consommation de tokens en temps réel et une estimation du coût de la session dans le pied de page du CLI Gemini.

## Étapes

- **Phase 1 : Compréhension et Identification**
  - [x] Identifier où la consommation de tokens est actuellement suivie et rapportée.
  - [x] Identifier les composants de l'interface utilisateur (UI) responsables du rendu du pied de page.
  - [x] Rechercher la structure des données de consommation de tokens.

- **Phase 2 : Implémentation du Compteur de Tokens et de l'Estimation du Prix**
  - [x] Implémenter la logique de calcul du coût estimé et des tokens consommés dans `packages/cli/src/ui/components/Composer.tsx`.
  - [x] Passer les valeurs calculées au composant `Footer`.
  - [x] Mettre à jour le composant `Footer` pour afficher les valeurs reçues.
  - [x] Mettre à jour la sortie JSON (`--json`) pour inclure le nombre de tokens et l'estimation du coût.
    - **Action**: Modifié `packages/core/src/output/types.ts` pour ajouter `totalTokens` et `costEstimation` à l'interface `JsonOutput`.
    - **Action**: Modifié `packages/core/src/output/json-formatter.ts` pour inclure ces nouvelles propriétés dans la sortie JSON.
    - **Action**: Modifié `packages/cli/src/nonInteractiveCli.ts` pour calculer et passer les nouvelles valeurs au formateur JSON.

- **Phase 3 : Tests et Validation**
  - [x] Résoudre les erreurs de test et de linting causées par les changements.
    - **Action**: Corrigé les tests dans `packages/core/src/output/json-formatter.test.ts`.
    - **Action**: Rétabli `packages/cli/src/ui/components/Header.test.tsx` à la version de `main`.
    - **Action**: Ajouté `memfs` aux `devDependencies` dans `packages/cli/package.json`.
    - **Action**: Corrigé l'erreur de linting dans `packages/core/src/utils/fileUtils.test.ts`.
  - [x] Exécuter `npm run preflight` pour valider l'ensemble des changements.

- [ ] **Phase 4 : Améliorations Futures**
  - [ ] Affiner le calcul du coût avec un modèle de tarification plus précis.
  - [ ] Écrire des tests unitaires pour la nouvelle logique de calcul.
  - [ ] Mettre à jour la documentation.
