#!/bin/bash

# Répertoire du repo
REPO="/mnt/projects/ai_engineer_projects"

cd "$REPO" || { echo "Dossier $REPO introuvable"; exit 1; }

echo "Création de la structure des projets P02 → P15"

# Boucle pour P02 à P15
for i in $(seq -w 2 15); do
    DIR="P$i"
    echo "  -> Création du dossier $DIR"

    # Créer le dossier principal s'il n'existe pas
    [ -d "$DIR" ] || mkdir "$DIR"

    # Créer les sous-dossiers
    mkdir -p $DIR/notebooks $DIR/src $DIR/models $DIR/results

    # Créer .gitkeep si absent
    [ -f "$DIR/notebooks/.gitkeep" ] || touch "$DIR/notebooks/.gitkeep"
    [ -f "$DIR/src/.gitkeep" ] || touch "$DIR/src/.gitkeep"
    [ -f "$DIR/models/.gitkeep" ] || touch "$DIR/models/.gitkeep"
    [ -f "$DIR/results/.gitkeep" ] || touch "$DIR/results/.gitkeep"

    # Créer README.md si absent
    [ -f "$DIR/README.md" ] || touch "$DIR/README.md"
done

# Ajouter au Git, commit et push
git add .
git commit -m "Create clean pro structure for P02-P15"
git push origin main

echo "Structure complète créée et synchronisée avec GitHub"


