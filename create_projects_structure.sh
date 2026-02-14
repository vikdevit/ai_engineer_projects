#!/bin/bash

# Répertoire du repo
REPO="/mnt/projects/ai_engineer_projects"

cd "$REPO" || { echo "Dossier $REPO introuvable"; exit 1; }

# Boucle pour créer P02 à P15
for P in {02..15}; do
    echo "Création des dossiers pour P$P"
    mkdir -p $P/notebooks $P/src $P/models $P/results
    touch $P/notebooks/.gitkeep
    touch $P/src/.gitkeep
    touch $P/models/.gitkeep
    touch $P/results/.gitkeep
    touch $P/README.md
done

# Ajouter tout au Git
git add .
git commit -m "Initial pro structure P02-P15 with notebooks, src, models, results"
git push origin main

echo "Structure complète créée et synchronisée avec GitHub ✅"

