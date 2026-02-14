#!/bin/bash

# Répertoire du repo
REPO="/mnt/projects/ai_engineer_projects"

cd "$REPO" || { echo "Dossier $REPO introuvable"; exit 1; }

# Boucle sur les dossiers P02 à P15 existants
for P in P{02..15}; do
    if [ -d "$P" ]; then
        echo "Création de la structure dans $P"

        # Créer les sous-dossiers s'ils n'existent pas
        mkdir -p $P/notebooks $P/src $P/models $P/results

        # Créer .gitkeep uniquement si absent
        [ -f "$P/notebooks/.gitkeep" ] || touch "$P/notebooks/.gitkeep"
        [ -f "$P/src/.gitkeep" ] || touch "$P/src/.gitkeep"
        [ -f "$P/models/.gitkeep" ] || touch "$P/models/.gitkeep"
        [ -f "$P/results/.gitkeep" ] || touch "$P/results/.gitkeep"

        # Créer README.md uniquement si absent
        [ -f "$P/README.md" ] || touch "$P/README.md"

    else
        echo "Attention : dossier $P introuvable dans le repo"
    fi
done

# Ajouter les nouveaux fichiers à Git
git add .
git commit -m "Add pro structure (notebooks/src/models/results) to existing P02-P15"
git push origin main

echo "Structure ajoutée aux dossiers existants et synchronisée avec GitHub ✅"

