#!/usr/bin/env python
# coding: utf-8

# -------------------------------------------------------------
# ## Viken KHATCHERIAN
# ## Formation AI Engineer
# ## Openclassrooms
# ## Projet 4 : classifiez automatiquement des informations
# -------------------------------------------------------------

# ## Etape préalable

# In[189]:


# Contrôle présence venv uv 
import sys
print(sys.executable)


# In[190]:


# Contrôle version python
sys.version


# In[191]:


# Contrôle présence des bibliothèques installées dans le venv uv
import importlib.metadata as md

packages = [
    "imbalanced-learn", "ipykernel", "jupyterlab",
    "matplotlib", "missingno", "numpy", "pandas", 
    "plotly", "psycopg2-binary", "pyarrow" , "scikit-learn",
    "scipy", "seaborn", "shap", "sqlalchemy", "statsmodels"
]

for p in packages:
    try:
        print(p, md.version(p))
    except:
        print(p, "missing")


# ## Etape 1 : Effectuer une analyse exploratoire des fichiers de données

# ### Etape 1-1 : EDA du dataframe df_sirh avant jointure

# In[192]:


# Chargement en dataframes pandas des 3 fichiers sources sirh, évaluation et sondage  
import pandas as pd


# In[193]:


df_sirh = pd.read_csv("../data/raw/extrait_sirh.csv")


# In[194]:


df_sirh.head(3)


# In[195]:


df_sirh.info()


# In[196]:


# Renommer certaines colonnes sans dupliquer le dataframe 
df_sirh.rename(columns={
    "id_employee": "id_salarie",
    "nombre_heures_travailless": "nombre_heures_travaillees",
    "annee_experience_totale" : "nombre_total_annees_experience",
    "annees_dans_l_entreprise" : "nombre_total_annees_dans_l_entreprise",
    "annees_dans_le_poste_actuel" : "nombre_total_annees_dans_le_poste_actuel"  
}, inplace=True)


# In[197]:


df_sirh.info()


# In[198]:


# identification des colonnes numériques
int_cols_sirh = df_sirh.select_dtypes(include="int64").columns
print(int_cols_sirh)


# In[199]:


# Vérification absence de decimales sur les valeurs au format int64
for col in int_cols_sirh:
    a_decimales = (df_sirh[col] % 1 != 0).any()
    print(col, "a décimales ?" , a_decimales)


# In[200]:


# Identification des colonnes type chaînes de caractères
string_cols_sirh = df_sirh.select_dtypes(include="str").columns
print(string_cols_sirh)


# In[201]:


# Recherche des différentes instances de chacune des colonnes type chaînes de caractères
for col in string_cols_sirh:
    print(df_sirh[col].unique()) 


# In[202]:


# Retirer les majuscules, espaces, accents et parenthèses
import unicodedata

df_sirh[string_cols_sirh] = df_sirh[string_cols_sirh].apply(
    lambda string_cols_sirh: string_cols_sirh.astype(str)
    .str.lower()
    .str.replace(r"[()]", "", regex=True)  
    .str.replace(" ", "_")
    .str.strip("_") 
    .map(lambda x: unicodedata.normalize("NFKD", x)
         .encode("ascii", errors="ignore")
         .decode("utf-8"))
)


# In[203]:


# Réafficher les colonnes types chaînes de caractères mises en forme
for col in string_cols_sirh:
    print(df_sirh[col].unique()) 


# In[204]:


# Recherche de valeurs manquantes 
df_sirh.apply(lambda col: col.isna().sum()).sort_values(ascending=False)


# In[205]:


# Contrôle présence de doublons sur la colonne id_salarie
df_sirh["id_salarie"].duplicated().any()


# In[206]:


df_sirh.info()


# In[207]:


# Statistiques descriptives du dataframe df_sirh
df_sirh.describe().T


# #### premiere analyse des statistiques descriptives
# * anomalies du nombre d'heures travaillées car tous les salariés sont à 80h soit plus de deux fois 35h => cette colonne est peu informative et pourra donc être ignorée
# *  la moyenne d'âge des salariés est de 36 ans
# *  la distribution des âges des salariés est équilibrée car moyenne = médiane
# *  le nombre d'années moyen passées dans l'entreprise est autour de **7 ans**

# In[208]:


# suppression de la colonne nombre d'heures travaillées
df_sirh.drop(columns=["nombre_heures_travaillees"], inplace=True)


# In[209]:


# distribution des âges des salariés

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))

ax = sns.histplot(df_sirh["age"], bins=20)

plt.title("Distribution des salariés par âge")
plt.xlabel("Âge")
plt.ylabel("Nombre de salariés")

# ajouter les valeurs sur les barres
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(
            p.get_x() + p.get_width() / 2,
            height,
            int(height),
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.show()


# In[210]:


# distribution des revenus mensuels
plt.figure(figsize=(8,5))

ax = sns.histplot(df_sirh["revenu_mensuel"], bins=15)

plt.title("Distribution des salariés par salaire")
plt.xlabel("Salaire")
plt.ylabel("Nombre de salariés")

# ajouter les valeurs sur les barres
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(
            p.get_x() + p.get_width() / 2,
            height,
            int(height),
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.show()


# In[211]:


#  Distribution du nombre total d'annees d'experience 
plt.figure(figsize=(8,5))

ax = sns.histplot(df_sirh["nombre_total_annees_experience"], bins=15)

plt.title("Distribution des salariés par niveau d'expérience")
plt.xlabel("nombre total d'années d'expérience")
plt.ylabel("Nombre de salariés")

# ajouter les valeurs sur les barres
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(
            p.get_x() + p.get_width() / 2,
            height,
            int(height),
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.show()


# In[212]:


# Distribution du niveau d'ancienneté dans l'entreprise
plt.figure(figsize=(8,5))

ax = sns.histplot(df_sirh["nombre_total_annees_dans_l_entreprise"], bins=15)

plt.title("Distribution des salariés par niveau d'ancienneté dans l'entreprise")
plt.xlabel("nombre total d'années dans l'entreprise")
plt.ylabel("Nombre de salariés")

# ajouter les valeurs sur les barres
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.text(
            p.get_x() + p.get_width() / 2,
            height,
            int(height),
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.show()


# #### observations
# * le salaire médian de 4 900 € représente le cœur de la population salariale
# * la majorité des salariés quittent avant 10 ans
# * peu de salariés atteignent donc les niveaux d’ancienneté élevés
# * cela peut limiter la progression salariale globale et **potentiellement influencer le turnover**

# In[213]:


# Distribution par département, poste, genre et statut marital

import matplotlib.pyplot as plt
import seaborn as sns

cols = [
    ("departement", "Effectif par département", "y"),
    ("poste", "Effectif par poste", "y"),
    ("genre", "Effectif par genre", "x"),
    ("statut_marital", "Effectif par statut marital", "x")
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (col, title, orient) in enumerate(cols):

    if orient == "y":
        sns.countplot(y=df_sirh[col], ax=axes[i])
    else:
        sns.countplot(x=df_sirh[col], ax=axes[i])

    axes[i].set_title(title)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

plt.tight_layout()
plt.show()


# #### observations
# * les postes de cadre commercial, assistant de direction et de consultant sont les plus représentés
# * il n'y a pas de déséquilibre profond entre hommes et femmes bien que ces dernières soient à 25% moins représentées que les hommes

# In[214]:


# Recherche de premieres corrélations
import matplotlib.pyplot as plt
import seaborn as sns

plots = [
    (
        "nombre_total_annees_experience",
        "nombre_total_annees_dans_l_entreprise",
        "Comparaison ancienneté entreprise vs expérience"
    ),
    (
        "nombre_total_annees_dans_l_entreprise",
        "nombre_total_annees_dans_le_poste_actuel",
        "Comparaison poste actuel vs ancienneté entreprise"
    ),
    (
        "nombre_total_annees_dans_l_entreprise",
        "revenu_mensuel",
        "Comparaison revenu mensuel vs ancienneté entreprise"
    )
]

fig, axes = plt.subplots(1, 3, figsize=(18, 8))

for i, (x, y, title) in enumerate(plots):
    sns.scatterplot(
        data=df_sirh,
        x=x,
        y=y,
        ax=axes[i]
    )
    axes[i].set_title(title, fontweight='bold')
    axes[i].set_xlabel(x, fontweight='bold')
    axes[i].set_ylabel(y, fontweight='bold')

plt.tight_layout()
plt.show()


# #### observations
# * les salariés arrivent dans l'entreprise avec de l'entreprise déjà acquise ailleurs
# * les salariés dans l'entreprise changent régulièrement de poste ce qui peut signifier soit **promotions** soit **restructurations fréquentes**

# In[215]:


# Matrice globale des corrélations de df_sirh
plt.figure(figsize=(10,6))
sns.heatmap(df_sirh.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
plt.show()


# #### observations
# Les variables suivantes sont fortement corrélées entre elles:
# * l'âge augmente avec l'expérience
# * l'expérience augmente avec le salaire 
# * l'ancienneté augmente sans nécessairement changer de poste (soit **faible mobilité interne pouvant expliquer les départs** soit **évolution lente**)
# 
# Ces variables corrélées amènent de la **multicolinéarité** qui peut rendre les **poids instables** et **l'interprétation biaisée** dans une modélisaiton de machine learning. Il faudra donc:
# * soit garder l'âge
# * soit garder l'expérience
# * ou créer une feature “ratio salaire / expérience”

# ### Etape 1-2 : EDA du dataframe df_eval avant jointure

# In[216]:


df_eval = pd.read_csv("../data/raw/extrait_eval.csv")


# In[217]:


df_eval.head(3)


# In[218]:


df_eval.info()


# In[219]:


# Renommer certaines colonnes sans dupliquer le dataframe 
df_eval.rename(columns={
    "satisfaction_employee_environnement": "satisfaction_salarie_environnement",
    "satisfaction_employee_nature_travail": "satisfaction_salarie_nature_travail",
    "satisfaction_employee_equipe": "satisfaction_salarie_equipe",
    "satisfaction_employee_equilibre_pro_perso" : "satisfaction_salarie_equilibre_pro_perso",
    "eval_number" : "numero_d_evaluation",
    "heure_supplementaires" : "heures_supplementaires",
    "augementation_salaire_precedente" : "precedent_pourcentage_d_augmentation"  
}, inplace=True)


# In[220]:


df_eval.info()


# In[221]:


int_cols_eval = df_eval.select_dtypes(include="int64").columns
print(int_cols_eval)


# In[222]:


for col in int_cols_eval:
    a_decimales = (df_eval[col] % 1 != 0).any()
    print(col, "a décimales ?" , a_decimales)


# In[223]:


# Identification des colonnes type chaînes de caractères
string_cols_eval = df_eval.select_dtypes(include="str").columns
print(string_cols_eval)


# In[224]:


# Recherche des différentes instances de chacune des colonnes type chaînes de caractères
for col in string_cols_eval:
    print(df_eval[col].unique()) 


# In[225]:


df_sirh["id_salarie"].min(), df_sirh["id_salarie"].max()


# In[226]:


# Transformer les colonnes au format int64  ou chaîne de caractères
import re

df_eval["numero_d_evaluation"] = df_eval["numero_d_evaluation"].apply(
    lambda x: int(re.sub(r"\D", "", str(x)))
)

df_eval["heures_supplementaires"] = df_eval["heures_supplementaires"].apply(
    lambda x: str(x).strip().lower()
)

df_eval["precedent_pourcentage_d_augmentation"] = df_eval["precedent_pourcentage_d_augmentation"].apply(
    lambda x: int(re.sub(r"[^\d]", "", str(x)))
)


# In[227]:


for col in string_cols_eval:
    print(df_eval[col].unique()) 


# In[228]:


df_eval.info()


# In[229]:


# Recherche de valeurs manquantes 
df_eval.apply(lambda col: col.isna().sum()).sort_values(ascending=False)


# In[230]:


# Contrôle présence de doublons sur la colonne numero_d_evaluation
df_eval["numero_d_evaluation"].duplicated().any()


# In[231]:


df_eval.info()


# In[232]:


# Statistiques descriptives du dataframe df_eval
num_cols_eval = df_eval.select_dtypes(include="int64").columns
cat_cols_eval = df_eval.select_dtypes(include="str").columns
df_eval[num_cols_eval].describe().T


# In[233]:


# Tracer des distributions univariées du dataframe df_eval

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math

hierarchical_cols = ["niveau_hierarchique_poste"]
percent_cols = ["precedent_pourcentage_d_augmentation"]
exclude_cols = ["numero_d_evaluation"] 

# retirer la colonne du calcul
cols_to_plot = [col for col in num_cols_eval if col not in exclude_cols]

n_cols = 2
n_rows = math.ceil(len(cols_to_plot) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
axes = axes.ravel()

cmap = plt.cm.viridis

for i, col in enumerate(cols_to_plot):

    # cas niveau hiérarchique
    if col in hierarchical_cols:
        levels = sorted(df_eval[col].unique())

    # cas pourcentage augmentation
    elif col in percent_cols:
        levels = sorted(df_eval[col].unique())
        axes[i].set_xlim(min(levels) - 1, max(levels) + 1)

    # autres variables numériques
    else:
        levels = sorted(df_eval[col].unique())

    # normalisation couleurs
    norm = mcolors.Normalize(vmin=min(levels), vmax=max(levels))

    counts = df_eval[col].value_counts().reindex(levels, fill_value=0)

    colors = [cmap(norm(x)) for x in counts.index]

    axes[i].bar(
        counts.index,
        counts.values,
        width=0.6,
        edgecolor="black",
        color=colors
    )

    axes[i].set_title(f"Distribution de {col}")
    axes[i].set_xticks(levels)

# supprimer axes inutilisés
for j in range(len(cols_to_plot), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[234]:


# création de la satisfaction moyenne puis tracé du diagramme en barres

satisfaction_cols = [
    "satisfaction_salarie_environnement",
    "satisfaction_salarie_nature_travail",
    "satisfaction_salarie_equipe",
    "satisfaction_salarie_equilibre_pro_perso"
]

df_eval["satisfaction_moyenne"] = df_eval[satisfaction_cols].mean(axis=1)

# création des bins
df_eval["satisfaction_bin"] = pd.cut(
    df_eval["satisfaction_moyenne"],
    bins=10
)

# comptage par bin
bar_data = df_eval["satisfaction_bin"].value_counts().sort_index()

# barplot
plt.figure(figsize=(10,5))
sns.barplot(x=bar_data.index.astype(str), y=bar_data.values)

plt.xticks(rotation=45)
plt.title("Distribution satisfaction globale")
plt.xlabel("Satisfaction moyenne")
plt.ylabel("Nombre de salariés")

plt.show()


# In[235]:


# distribution des heures supplémentaires

plt.figure(figsize=(4,6))

ax = sns.countplot(
    data=df_eval,
    x="heures_supplementaires",
    order=["non", "oui"] 
)

# titres
plt.title("Répartition des heures supplémentaires")
plt.xlabel("Heures supplémentaires")
plt.ylabel("Nombre de salariés")

# ajout des valeurs sur les barres
for container in ax.containers:
    ax.bar_label(container)

plt.show()


# In[236]:


# matrice globale des corrélations de df_eval
plt.figure(figsize=(10,6))
sns.heatmap(df_eval.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
plt.show()


# #### observations
# * satisfaction globale moyenne mais pas excellente
# * **peu de différenciation individuelle sur l'augmentation salariale**
# * perception de l'environnement de travail globalement bonne mais pas homogène (400 personnes note 3 et 400 personnes note 4) **ce qui peut cacher des différences dans les différents départements de travail**
# * le système de notation d'évaluation actuelle est peu discriminant par près de 80%  des salariés ont leur performance notée à 3. Or 400 salariés evaluent la satisfaction globale à 2 **ce qui montre que les salariés évaluent moins bien leur bien-être (pouvant causer des départs) que leur performance ce qui est synonyme de stress global**.
# * 30% de l'effectif effectue des heures supplémentaires ce qui peut être source de stress
# * **plus la note d’évaluation est élevée, plus le pourcentage d’augmentation est élevé**
# * la note d’évaluation précédente n’a pas de corrélation linéaire simple avec la note actuelle (un salarié bien noté avant n’est pas forcément bien noté maintenant (et inversement)), **ce qui peut suggérer que les signaux de départ sont plutôt dans le contexte actuel que dans l’historique**
# 

# In[237]:


df_eval.info()


# ### Etape 1-3 : EDA du dataframe df_sondage avant jointure

# In[238]:


df_sondage = pd.read_csv("../data/raw/extrait_sondage.csv")


# In[239]:


df_sondage.head(3)


# In[240]:


df_sondage.info()


# In[241]:


# Renommer certaines colonnes sans dupliquer le dataframe 
df_sondage.rename(columns={
    "nb_formations_suivies" : "nombre_de_formations_suivies",
    "nombre_employee_sous_responsabilite" : "nombre_de_salaries_sous_sa_responsabilite",
    "annees_depuis_la_derniere_promotion" : "nombre_d_annees_depuis_la_derniere_promotion",
    "annes_sous_responsable_actuel" : "nombre_d_annees_sous_le_responsable_actuel"
}, inplace=True)


# In[242]:


df_sondage.info()


# In[243]:


# Affichage des noms de colonnes numériques
int_cols_sondage = df_sondage.select_dtypes(include="int64").columns
print(int_cols_sondage)


# In[244]:


# Vérification absence de décimales pour les colonnes numériques au format int64
for col in int_cols_sondage:
    a_decimales = (df_sondage[col] % 1 != 0).any()
    print(col, "a décimales ?" , a_decimales)


# In[245]:


# Identification des colonnes type chaînes de caractères
string_cols_sondage = df_sondage.select_dtypes(include="str").columns
print(string_cols_sondage)


# In[246]:


# Recherche des différentes instances de chacune des colonnes type chaînes de caractères
for col in string_cols_sondage:
    print(df_sondage[col].unique()) 


# In[247]:


# Recherche de valeurs manquantes 
df_sondage.apply(lambda col: col.isna().sum()).sort_values(ascending=False)


# In[248]:


# Retirer les majuscules, espaces, & et remplacer Y par oui
df_sondage[string_cols_sondage] = df_sondage[string_cols_sondage].apply(
    lambda col: col.astype(str)
    .str.strip()
    .str.lower()
    .str.replace("&", "et", regex=False)
    .str.replace(" ", "_", regex=False)
    .apply(lambda x: "oui" if x == "y" else x)
)


# In[249]:


for col in string_cols_sondage:
    print(df_sondage[col].unique()) 


# In[250]:


df_sondage.info()


# In[251]:


# Recherche valeurs min et max de l'identifiant (salarié)
df_sondage["code_sondage"].min(), df_sondage["code_sondage"].max()


# In[252]:


# Contrôle présence de doublons sur la colonne code_sondage
df_sondage["code_sondage"].duplicated().any()


# In[253]:


# statistiques descriptiques du dataframe df_sondage
df_sondage.describe().T


# #### observations 
# - Le nombre de salaries sous sa responsaiblité est toujours égal à 1 donc pas de variabilité dans les données, la colonne pourra être retirée du dataframe.

# In[254]:


# Visualisations univariées

sns.set_style("whitegrid", {'axes.grid.axis': 'y'})
plt.rcParams["figure.figsize"] = (6, 4)

# Distance domicile-travail
sns.histplot(df_sondage["distance_domicile_travail"], kde=False)
ax = sns.histplot(df_sondage["distance_domicile_travail"], kde=False)
ax.grid(axis="x", visible=False)
ax.grid(axis="y", visible=True)
plt.title("Distribution distance domicile-travail")
plt.show()

# Formations suivies
sns.histplot(df_sondage["nombre_de_formations_suivies"], kde=False)
ax = sns.histplot(df_sondage["nombre_de_formations_suivies"], kde=False)
ax.grid(axis="x", visible=False)
ax.grid(axis="y", visible=True)
plt.title("Nombre de formations suivies")
plt.show()

# Années depuis dernière promotion
sns.histplot(df_sondage["nombre_d_annees_depuis_la_derniere_promotion"], kde=False)
ax = sns.histplot(df_sondage["nombre_d_annees_depuis_la_derniere_promotion"], kde=False)
ax.grid(axis="x", visible=False)
ax.grid(axis="y", visible=True)
plt.title("Années depuis dernière promotion")
plt.show()

# Fréquence des déplacements
sns.countplot(data=df_sondage, x="frequence_deplacement")
ax = sns.countplot(data=df_sondage, x="frequence_deplacement")
ax.grid(axis="x", visible=False)
ax.grid(axis="y", visible=True)
plt.title("Fréquence des déplacements")
plt.show()

# Domaine d'étude 
sns.countplot(data=df_sondage, x="domaine_etude", order=df_sondage["domaine_etude"].value_counts().index)
ax = sns.countplot(data=df_sondage, x="domaine_etude", order=df_sondage["domaine_etude"].value_counts().index)
ax.grid(axis="x", visible=False)
ax.grid(axis="y", visible=True)
plt.title("Répartition du domaine d'étude")
plt.xticks(rotation=45, ha="center", fontsize=9)
plt.show()

# Ayant enfants
sns.countplot(data=df_sondage, x="ayant_enfants")
ax = sns.countplot(data=df_sondage, x="ayant_enfants")
ax.grid(axis="x", visible=False)
ax.grid(axis="y", visible=True)
plt.title("Répartition des salariés ayant des enfants")
plt.show()

# a quitté l'entreprise (variable cible potentielle)
plt.figure()

(df_sondage["a_quitte_l_entreprise"]
 .value_counts(normalize=True)
 .sort_index()
 .plot(kind="bar"))

ax = (df_sondage["a_quitte_l_entreprise"]
      .value_counts(normalize=True)
      .sort_index()
      .plot(kind="bar"))

# ajout des labels directement
for container in ax.containers:
    labels = [f"{v*100:.1f}%" for v in container.datavalues]
    ax.bar_label(container, labels=labels, fontsize=8)

ax.grid(axis="x", visible=False)
ax.grid(axis="y", visible=True)

plt.title("Analyse du pourcentage de démissions")
plt.ylabel("Proportion")
plt.xlabel("A quitté l'entreprise")
plt.xticks(rotation=0)

plt.show()


# In[255]:


# Analyse de la variable cible a_quitte_l_entreprise
df_sondage["a_quitte_l_entreprise"].value_counts(normalize=True)


# In[256]:


# visualisaitons bivariees
import math

sns.set(style="whitegrid")

cols = [
    "nombre_participation_pee",
    "nombre_de_formations_suivies",
    "nombre_de_salaries_sous_sa_responsabilite",
    "distance_domicile_travail",
    "niveau_education",
    "domaine_etude",
    "ayant_enfants",
    "frequence_deplacement",
    "nombre_d_annees_depuis_la_derniere_promotion",
    "nombre_d_annees_sous_le_responsable_actuel"
]

n_cols = 2
n_rows = math.ceil(len(cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(cols):

    # Cas variables numériques
    if df_sondage[col].dtype == "int64":
        sns.boxplot(
            data=df_sondage,
            x="a_quitte_l_entreprise",
            y=col,
            ax=axes[i]
        )

    # Cas variables catégorielles
    else:
        sns.countplot(
            data=df_sondage,
            x=col,
            hue="a_quitte_l_entreprise",
            ax=axes[i]
        )
        axes[i].tick_params(axis='x', rotation=45)

    axes[i].set_title(f"{col} vs départ de l'entreprise")
    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

# supprimer axes inutilisés
for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# #### observations
# * **la colonne ayant enfants pourra être retirée du dataframe** car la réponse est toujours oui, donc la variable n'apportera pas d'informations
# * population plutôt proche du lieu de travail, **peu de contrainte géographique globale**
# * **politique de formation active** (1000 salaries ont suivi entre 2 et 3 formations), faible dispersion des valeurs dans la boxplot
# * promotions récentes nombreuses
# * **déplacements assez présents** ce qui peut être facteur potentiel de fatigue puis de départ de l'entreprise
# * population technique dominante
# * **le dataset présente un déséquilibre de classes significatif sur la variable cible**, ce qui nécessitera des techniques adaptées (pondération, rééchantillonnage avec undersampling) lors de la modélisation machine learning
# * **davantage de départs dans le domaine “infra & cloud”** (profils très demandés, stress, sous-payé par rapport au marché)
# * **population majoritairement technique** (domaine infra & cloud), avec une forte concentration des départs dans ce segment 
# * **les salariés présentent une ancienneté faible depuis leur dernière promotion** et une fréquence de déplacement majoritairement occasionnelle
# * **le déséquilibre de classes (16% de départs) devra être pris en compte dans les étapes de modélisation**

# In[257]:


# Retirer les colonnes "ayant_enfants" et "nombre_de_salaries_sous_sa_responsabilite"
df_sondage.drop(
    columns=[
        "ayant_enfants",
        "nombre_de_salaries_sous_sa_responsabilite"
    ],
    inplace=True
)
df_sondage.info()


# In[258]:


# Matrice globale des corrélations de df_sondage
# sélection des colonnes numériques
num_cols_sondage = df_sondage.select_dtypes(include="int64")

# matrice de corrélation
corr_matrix = num_cols_sondage.corr()

# heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Heatmap des corrélations - df_sondage")
plt.show()


# #### observations
# - Une corrélation modérée (0.51) entre le nombre d’années depuis la dernière promotion et l’ancienneté avec le responsable actuel suggère que **les salariés restés longtemps sous le même management ont tendance à connaître une progression de carrière plus lente, ce qui peut refléter une certaine stagnation professionnelle**. Ceci peut être une motivation pour quitter l'entreprise.

# ### Etape 1-4 : Réalisation de la jointure pour réunir les 3 dataframes 

# In[259]:


# Réalisation de la jointure pour réunir les trois dataframes

df_sirh = df_sirh.rename(columns={"id_salarie": "id"})
df_eval = df_eval.rename(columns={"numero_d_evaluation": "id"})
df_sondage = df_sondage.rename(columns={"code_sondage": "id"})

df_joint = (
    df_sirh
    .merge(df_eval, on="id", how="inner")
    .merge(df_sondage, on="id", how="inner")
    .sort_values(by="id")
    .reset_index(drop=True)
)


# In[260]:


# Contrôle dimensions du dataframe obtenu et de l'absence de doublons
print("Shape finale :", df_joint.shape)
print("Doublons :", df_joint.duplicated().sum())


# In[261]:


df_joint.info()


# ### Etape 1-5 : EDA sur le dataframe obtenu df_joint

# In[262]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)


# In[263]:


df_joint.describe().T


# In[264]:


# Fonctions python pour l'EDA sur le dataframe df_joint issu de la jointure

def eda_univariee(df, num_cols=None, cat_cols=None):
    """Visualisations univariées pour les colonnes numériques et catégorielles"""
    if num_cols is None:
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Numériques
    for col in num_cols:
        sns.histplot(df[col], kde=False)
        plt.title(f"Distribution de {col}")
        plt.show()

    # Catégorielles
    for col in cat_cols:
        sns.countplot(y=df[col], order=df[col].value_counts().index)
        plt.title(f"Répartition de {col}")
        plt.show()


def eda_bivariee(df, target_col, num_cols=None, cat_cols=None):
    """Visualisations bivariées entre features et la cible"""
    if num_cols is None:
        num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

    # Numériques vs cible
    for col in num_cols:
        plt.figure(figsize=(8,5))
        sns.boxplot(x=df[target_col], y=df[col])

        plt.title(f"{col} selon {target_col}", fontsize=14)
        plt.xlabel(target_col, fontsize=12)
        plt.ylabel(col, fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.show()

    # Catégorielles vs cible
    for col in cat_cols:
        plt.figure(figsize=(8,5))
        sns.countplot(x=col, hue=target_col, data=df)

        plt.title(f"{col} selon {target_col}", fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10)

        plt.legend(title=target_col, fontsize=10, title_fontsize=11)

        plt.show()

def correlation_heatmap(df, method="pearson"):
    """
    - method="pearson" : variables normalement distribuées
    - method="spearman": variables non normales ou discrètes
    """
    if method == "pearson":
        cols = [
            "age"  
        ]
    elif method == "spearman":
        cols = [
            "revenu_mensuel",
            "nombre_total_annees_experience",
            "nombre_total_annees_dans_l_entreprise",
            "nombre_total_annees_dans_le_poste_actuel",
            "satisfaction_moyenne",
            "nombre_experiences_precedentes",
            "satisfaction_salarie_environnement",
            "note_evaluation_precedente",
            "niveau_hierarchique_poste",
            "satisfaction_salarie_nature_travail",
            "satisfaction_salarie_equipe",
            "satisfaction_salarie_equilibre_pro_perso",
            "note_evaluation_actuelle",
            "precedent_pourcentage_d_augmentation",
            "nombre_participation_pee",
            "nombre_de_formations_suivies",
            "distance_domicile_travail",
            "niveau_education",
            "nombre_d_annees_depuis_la_derniere_promotion",
            "nombre_d_annees_sous_le_responsable_actuel"
        ]
    else:
        raise ValueError("Méthode doit être 'pearson' ou 'spearman'")

    plt.figure(figsize=(20,18)) 

    sns.heatmap(
    df[cols].corr(method=method),
    annot=True,           
    fmt=".2f",
    cmap="coolwarm",
    annot_kws={"size": 8},  
    linewidths=0.8          
)

    plt.title(f"Matrice de corrélation ({method})", fontsize=16)
    plt.xticks(fontsize=10, rotation=90)  
    plt.yticks(fontsize=10, rotation=0)  
    plt.tight_layout()                     
    plt.show()


def pairplot_sample(df, cols, target_col, sample_size=500):
    """Pairplot pour un sous-échantillon de lignes de df_joint"""
    # Sous-échantillon
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df

    # Style et taille globale
    sns.set_context("notebook", font_scale=0.8)  
    g = sns.pairplot(
        df_sample[cols + [target_col]],
        hue=target_col,
        diag_kind="kde",
        height=3.5,    
        aspect=1.2    
    )

    for ax in g.axes.flatten():
        if ax is not None:
            ax.xaxis.label.set_fontweight('bold')
            ax.yaxis.label.set_fontweight('bold')

    # Mettre la légende en gras
    if g._legend is not None:
        for text in g._legend.get_texts():
            text.set_fontweight('bold')

    # Ajuster l'espacement
    plt.subplots_adjust(top=0.95)
    plt.show()


# In[265]:


# Fonctions python pour préparer les features X et la target y à partir de df_joint

def prepare_features(df, target_col):
    """
    Prépare X et y pour sklearn :
    - encode la cible en binaire
    - sépare features numériques et catégorielles
    - encode les catégorielles avec OneHotEncoder
    """
    # Cible
    y = df[target_col].map({"Oui":1, "Non":0})  # ajuster selon vos valeurs

    # Features
    X = df.drop(columns=[target_col, "id"])  # on enlève id et cible

    # Colonnes catégorielles
    cat_cols = X.select_dtypes(include=["object", "category"]).columns

    # OneHotEncoding
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X_encoded, y


# In[266]:


# EDA univariée

# Sélection des colonnes numériques uniquement
num_cols = df_joint.select_dtypes(include=["int64", "float64"]).columns

# Enlever l'id des listes
num_cols = [col for col in num_cols if col not in ["id"]]

# Appel de la fonction univariée avec ces colonnes
eda_univariee(df_joint, num_cols=num_cols, cat_cols=[])


# In[267]:


# EDA bivariée

# Détecter les colonnes numériques et catégorielles automatiquement
num_cols = df_joint.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df_joint.select_dtypes(include=["object", "category", "string"]).columns.tolist()

# Enlever la cible et l'id des listes
num_cols = [col for col in num_cols if col not in ["id"]]
cat_cols = [col for col in cat_cols if col not in ["id", "a_quitte_l_entreprise"]]

# Lancer l'EDA bivariée
eda_bivariee(df_joint, target_col="a_quitte_l_entreprise", num_cols=num_cols, cat_cols=cat_cols)


# In[268]:


# Réaliser la matrice des corrélations linéaires en heatmap pour les variables normalement distribuées et non normalement distribuées

# Pearson pour variables continues/normales
correlation_heatmap(df_joint, method="pearson")

# Spearman pour variables discrètes/non normales
correlation_heatmap(df_joint, method="spearman")


# In[269]:


# Pairplot pour les variables clés

# Choisir les variables numériques les plus pertinentes
selected_cols = [
    "age", 
    "revenu_mensuel",
    "nombre_total_annees_dans_l_entreprise",
    "satisfaction_moyenne",
    "note_evaluation_actuelle"
]

pairplot_sample(df_joint, cols=selected_cols, target_col="a_quitte_l_entreprise", sample_size=500)


# In[270]:


# Visualisations de la variable cible en fonction des variables numériques
num_cols = [
    "age", "revenu_mensuel", "nombre_total_annees_experience",
    "nombre_total_annees_dans_l_entreprise",
    "nombre_total_annees_dans_le_poste_actuel",
    "satisfaction_moyenne",
    "distance_domicile_travail",
    "nombre_d_annees_depuis_la_derniere_promotion"
]

for col in num_cols:
    sns.boxplot(x="a_quitte_l_entreprise", y=col, data=df_joint)
    plt.title(f"{col} vs départ")
    plt.show()


# In[271]:


# Mesurer la dispersion des valeurs et détecter les outliers avec la statistique descriptive IQR

print(f"{'Variable':40} {'Q1':>8} {'Q3':>8} {'IQR':>8} {'Min_IQR':>10} {'Max_IQR':>10} {'Outliers':>10}")
print("-" * 100)

for col in num_cols:
    Q1 = df_joint[col].quantile(0.25)
    Q3 = df_joint[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df_joint[
        (df_joint[col] < lower) | 
        (df_joint[col] > upper)
    ]

    print(f"{col:40} {Q1:8.2f} {Q3:8.2f} {IQR:8.2f} {lower:10.2f} {upper:10.2f} {len(outliers):10}")


# In[272]:


# Visualisations univeriées des variables catégorielles en distinguant effectif qui reste et effectif qui part
cat_cols = [
    "genre", "poste", "departement",
    "heures_supplementaires",
    "frequence_deplacement",
    "domaine_etude"
]

for col in cat_cols:
    sns.countplot(data=df_joint, x=col, hue="a_quitte_l_entreprise")
    plt.xticks(rotation=45)
    plt.title(f"{col} vs départ")
    plt.show()


# In[273]:


# Afichage numérique des variables catégorielles en distinguant effectif qui reste et effectif qui part
for col in cat_cols:
    print("\n" + "="*70)
    print(f"Variable : {col}")
    print("="*70)

    # tableau croisé (effectifs)
    table = pd.crosstab(df_joint[col], df_joint["a_quitte_l_entreprise"])

    # pourcentages par ligne
    table_pct = pd.crosstab(
        df_joint[col],
        df_joint["a_quitte_l_entreprise"],
        normalize="index"
    ) * 100

    # affichage combiné
    for category in table.index:
        print(f"\n {col} = {category}")

        for target in table.columns:
            count = table.loc[category, target]
            pct = table_pct.loc[category, target]

            print(f"   {target:<10} : {count:4} ({pct:5.1f}%)")


# In[274]:


# Identification des taux de départ par catégorie

for col in cat_cols:
    taux = pd.crosstab(df_joint[col], df_joint["a_quitte_l_entreprise"], normalize="index")
    taux["oui"].sort_values(ascending=False).plot(kind="bar")
    plt.title(f"Taux de départ par {col}")
    plt.show()


# In[275]:


# Affichage numérique des taux de départs par catégorie

for col in cat_cols:
    print("\n" + "="*70)
    print(f"Taux de départ par {col}")
    print("="*70)

    # calcul taux de départ
    taux = pd.crosstab(
        df_joint[col],
        df_joint["a_quitte_l_entreprise"],
        normalize="index"
    )

    # tri décroissant sur "oui"
    taux_sorted = taux["oui"].sort_values(ascending=False)

    # affichage
    for category, value in taux_sorted.items():
        print(f"{category:<30} : {value*100:5.1f}%")


# In[276]:


# Tests de statistiques inférentielles

# si p < 0.05 alors variable significative

# Test de Student pour les variables normalement distribuées

from scipy.stats import ttest_ind

alpha = 0.05

group1 = df_joint[df_joint["a_quitte_l_entreprise"] == "oui"]
group0 = df_joint[df_joint["a_quitte_l_entreprise"] == "non"]

col = "age"

stat, p = ttest_ind(group1[col], group0[col], nan_policy='omit')

if p < alpha:
    conclusion = "Différence significative"
else:
    conclusion = "Pas de différence"

print(f"{col:50} p-value = {p} → {conclusion}")


# In[277]:


# Test de Mann Whitney pour les variables non normalement distribuées

from scipy.stats import mannwhitneyu

cols_mw = [

            "revenu_mensuel",
            "nombre_total_annees_experience",
            "nombre_total_annees_dans_l_entreprise",
            "nombre_total_annees_dans_le_poste_actuel",
            "satisfaction_moyenne",
            "nombre_experiences_precedentes",
            "satisfaction_salarie_environnement",
            "note_evaluation_precedente",
            "niveau_hierarchique_poste",
            "satisfaction_salarie_nature_travail",
            "satisfaction_salarie_equipe",
            "satisfaction_salarie_equilibre_pro_perso",
            "note_evaluation_actuelle",
            "nombre_participation_pee",
            "nombre_de_formations_suivies",
            "distance_domicile_travail",
            "niveau_education",
            "nombre_d_annees_depuis_la_derniere_promotion",
            "nombre_d_annees_sous_le_responsable_actuel",
            "precedent_pourcentage_d_augmentation"
]

alpha = 0.05

for col in cols_mw:
    stat, p = mannwhitneyu(group1[col], group0[col], alternative="two-sided")

    if p < alpha:
        conclusion = "Différence significative"
    else:
        conclusion = "Pas de différence"

    print(f"{col:50} p-value = {p:.5f} → {conclusion}")


# In[278]:


# Test du Khi2 pour les variables catégorielles

from scipy.stats import chi2_contingency

alpha = 0.05

for col in cat_cols:
    table = pd.crosstab(df_joint[col], df_joint["a_quitte_l_entreprise"])
    stat, p, _, _ = chi2_contingency(table)

    if p < alpha:
        conclusion = "Différence significative"
    else:
        conclusion = "Pas de différence"

    print(f"{col:50} p-value = {p} → {conclusion}")


# #### observations:
# 
# - Les variables les plus discriminantes sont liées à **ancienneté, expérience, revenu, satisfaction et formation** **(p-values très faibles (~0 ou 1e-10)**)
# - Les variables moins discriminantes ou non significatives : **nombre d’expériences précédentes** **(p-values ~ 0.242 > 0.05)**, **satisfaction équipe, note d’évaluation actuelle, niveau d’éducation**.
# - Les variables catégorielles les plus discriminantes : **poste** **(p-values ~ 2.75e-15)**, **heures supplémentaires** **(p-values ~ 8.16e-21)**, **fréquence de déplacement** **(p-values ~ 5.6e-6)**.
# - Les segments moins discriminants : **genre**.
# - Les postes à forte rotation : représentant commercial, consultant, RH.
# - **Les salariés effectuant des heures supplémentaires ou ayant une fréquence de déplacement élevée ont un risque de départ plus élevé**.
# - **Les salariés issus de domaines d’étude RH, entrepreneuriat ou marketing sont plus susceptibles de quitter l’entreprise**.
# - **Segments à risque:**
#   -  **Postes à forte rotation** : représentant commercial, consultant, RH → ces segments sont explicitement identifiés comme plus à risque.
#   -  **Comportements à risque** : salariés effectuant des heures supplémentaires ou avec une fréquence de déplacement élevée.
#   -  **Domaines d’étude à risque** : RH, entrepreneuriat, marketing → segment ciblé pour le risque de départ
# - Multicolinéarité probable dans :
#   - Ancienneté, expérience, poste actuel, sous responsable, revenu et niveau hiérarchique.
#   -  Si ces variables sont utilisées ensemble dans un modèle linéaire (ex : régression logistique), les coefficients peuvent devenir instables.
# - Solution / feature engineering :
#   - **Créer des features composites ou ratios pour réduire la redondance** :
#     - ratio_experience_salaire = revenu_mensuel / nombre_total_annees_experience
#     - anciennete_relative = nombre_total_annees_dans_le_poste_actuel / nombre_total_annees_dans_l_entreprise
#     - anciennete_sous_responsable = nombre_d_annees_sous_le_responsable_actuel / nombre_total_annees_dans_le_poste_actuel
#     - niveau_poste_vs_revenu = revenu_mensuel / niveau_hierarchique_poste (pour capturer la sur/sous-rémunération par niveau)
#     - éviter d’inclure directement toutes les variables corrélées si modèle linéaire.
#   - **Variables moins problématiques** :
#     - Satisfaction moyenne,
#     - distance domicile-travail,
#     - formations suivies
#     - corrélation modérée ou faible, peuvent rester telles quelles.
# 

# In[280]:


# Feature engineering et préparation pour la modélisation 

import pandas as pd
import numpy as np

# Suppression des colonnes inutiles

drop_cols = [
    "id",
    "nombre_d_annees_sous_le_responsable_actuel",
    "nombre_total_annees_experience",
    "nombre_total_annees_dans_l_entreprise",
    "nombre_total_annees_dans_le_poste_actuel",
    "revenu_mensuel"
]

# Création d'un dataframe résultant pour les étapes suivantes de feature engineering puis de modélisation

df_mod = df_joint.drop(columns=drop_cols)

# Variables numériques dérivées

df_mod["ratio_salaire_experience"] = df_joint["revenu_mensuel"] / (df_joint["nombre_total_annees_experience"] + 1)
df_mod["anciennete_ratio"] = df_joint["nombre_total_annees_dans_l_entreprise"] / (df_joint["nombre_total_annees_experience"] + 1)
df_mod["anciennete_sous_responsable_ratio"] = df_joint["nombre_d_annees_sous_le_responsable_actuel"] / \
                                              (df_joint["nombre_total_annees_dans_le_poste_actuel"] + 1)
df_mod["niveau_poste_vs_revenu"] = df_joint["revenu_mensuel"] / (df_joint["niveau_hierarchique_poste"] + 1e-6)  # éviter div par 0

# Features comportementales / à risque

df_mod["stagnation"] = (df_joint["nombre_d_annees_depuis_la_derniere_promotion"] > 3).astype(int)
df_mod["stress"] = ((df_joint["heures_supplementaires"] == "oui") & (df_joint["satisfaction_moyenne"] < 3)).astype(int)
df_mod["mobilite"] = (df_joint["frequence_deplacement"] != "aucun").astype(int)
df_mod["low_satisfaction"] = (df_joint["satisfaction_moyenne"] < 2.5).astype(int)

# Encodage des variables catégorielles

def encode_categorical(df, col, threshold_low=0.05, threshold_high=0.8):
    """
    Encode les variables catégorielles :
    - Modalités < threshold_low dans 'other'
    - Modalités > threshold_high dans binaire majoritaire
    - Sinon one-hot
    """
    total = len(df)
    counts = df[col].value_counts(normalize=True)

    # Regrouper les modalités rares
    rare = counts[counts < threshold_low].index
    df[col] = df[col].replace(rare, "other")

    # Identifier la modalité majoritaire si > threshold_high
    major = counts[counts > threshold_high].index
    if len(major) == 1:
        # Créer un binaire pour majoritaire
        df[col + "_major"] = (df[col] == major[0]).astype(int)
        # Supprimer la colonne originale
        df = df.drop(columns=[col])
    else:
        # One-hot pour les autres variables
        df = pd.get_dummies(df, columns=[col], prefix=col)

    return df

# Colonnes catégorielles

categorical_cols = [
    "poste",
    "departement",
    "domaine_etude",
    "genre",
    "heures_supplementaires",
    "frequence_deplacement",
    "niveau_education"
]

for col in categorical_cols:
    df_mod = encode_categorical(df_mod, col)

# Sélection des features finales

numeric_features = [
    "satisfaction_moyenne",
    "ratio_salaire_experience",
    "anciennete_ratio",
    "anciennete_sous_responsable_ratio",
    "niveau_poste_vs_revenu",
    "distance_domicile_travail",
    "nombre_de_formations_suivies",
    "nombre_d_annees_depuis_la_derniere_promotion"
]

behavioral_features = ["stagnation", "stress", "mobilite", "low_satisfaction"]

categorical_prefixes = [
    "poste_",
    "departement_",
    "domaine_etude_",
    "genre_",
    "heures_supplementaires_",
    "frequence_deplacement_",
    "niveau_education_"
]

categorical_features = [
    col for col in df_mod.columns
    if any(col.startswith(prefix) for prefix in categorical_prefixes)
]

df_mod.info()

# Conserver uniquement les features finales souhaitées
final_features = numeric_features + behavioral_features + categorical_features

# Créer X et y pour la modélisation
X = df_mod[final_features].copy()
y = df_mod["a_quitte_l_entreprise"].map({"oui":1, "non":0})

# Contrôle format de X et y ainsi que des features finales présentes dans X

print("Shape de X :", X.shape)
print("Shape de y :", y.shape)
print("Features finales :", X.columns.tolist())


# #### Features et target finales pour la modélisation 
# 
# Les features finales sont :
# 
# **a) Numériques dérivées / calculées**
# - satisfaction_moyenne
# - ratio_salaire_experience (revenu / expérience)
# - anciennete_ratio (ancienneté / expérience)
# - anciennete_sous_responsable_ratio (ancienneté sous responsable / ancienneté poste)
# - niveau_poste_vs_revenu (revenu / niveau hiérarchique)
# - distance_domicile_travail
# - nombre_de_formations_suivies
# - nombre_d_annees_depuis_la_derniere_promotion
# 
# **b) Features comportementales / à risque**
# - stagnation (plus de 3 ans depuis dernière promotion)
# - stress (heures sup + faible satisfaction)
# - mobilite (fréquence déplacement ≠ aucun)
# - low_satisfaction (satisfaction < 2.5)
# 
# **c) Features catégorielles encodées**
# - poste_* (one-hot ou binaire majoritaire)
# - departement_*
# - domaine_etude_*
# - genre_*
# - heures_supplementaires_*
# - frequence_deplacement_*
# - niveau_education_*
# 
# Ces features couvrent les principaux insights issus de l'EDA, ce qui permet de passer à la modélisation pour la prédiction de départs.
# 
# **d) Target utilisée pour la modélisation**
# - y = df_mod["a_quitte_l_entreprise"].map({"oui":1, "non":0})
# - C’est la variable binaire cible pour la classification (1 = départ, 0 = reste).

# ##### Analyse du risque de data leakage et de la préparation de l'encodage
# 
# **Préparation / Encodage et Data Leakage:**
# - Suppression des colonnes inutiles pour éviter les features triviales ou fortement corrélées avec la target (ex : ID, revenu brut, anciennetés totales) dans un objectif d'éviter le surapprentissage
# - Variables dérivées toutes basées sur des colonnes historiques et présentes au moment du départ donc pas de data leakage lié à des informations connues après une démission.
# - Features comportementales dérivées de données présentes au moment de l’observation avant un départ donc le modèle ne voit rien de ce qui se passe après que la personne a quitté l'entreprise (pas de target leakage). 
# - Encodage catégoriel via encode_categorical :
#   - Modalités rares regroupées pour éviter overfitting sur petites catégories.
#   - Modalité majoritaire binaire pour simplifier le modèle.
#   - One-hot si plusieurs catégories (transformation sur les données d'entrée donc pas de fuite). 
# - Conversion bool vers int pour scikit-learn et ainsi ne pas avoir d’incohérence dans les calculs. 
# - ColumnTransformer avec RobustScaler pour outliers numériques et passthrough pour bool pour aucune fuite ni sur-échelle introduite par des transformations. 

# #### Raisons de préférence de features dérivées issues du feature engineering
# - Normalisation / ratio qui permet de comparer des individus ayant des expériences ou des salaires très différents.
# - Seuils comportementaux comme stagnation, low_satisfaction et stress qui sont directement issus des analyses statistiques et montrent un signal fort pour le départ.
# - Réduction de dimension pour les features très corrélées (ex : plusieurs notes de satisfaction) sont combinées pour éviter colinéarité et overfitting.
# - Encodage correct des variables catégorielles qui transforme les features comme poste, departement, genre en format exploitable par scikit-learn.

# #### Points forts du feature engineering
# - Segmentation du risque : Les features comportementales (stagnation, stress, mobilite, low_satisfaction) ciblent directement les segments à risque identifiés.
# - Réduction de la multicolinéarité : Les ratios numériques remplacent les variables fortement corrélées (anciennete, expérience, revenu, niveau) ce qui est plus stable pour les modèles linéaires.
# - Encodage adapté des variables catégorielles : poste, domaine_etude, etc., qui permettent de capturer les risques liés aux postes à forte rotation.
# - **Adaptation à la minorité : Ces nouvelles features renforcent le signal pour la classe “Départ”, ce qui est critique vu le déséquilibre des classes.**

# #### Implication pour les modèles
# - Les résultats des 3 modèles choisis pour les étapes suivantes (**Dummy, RandomForest, LightGBM**) dépendent de ces features finales dérivées, pas directement des colonnes d’origine.
# - **Comme la préparation est cohérente avec l’EDA et les tests statistiques, on peut considérer que les performances obtenues reflètent la qualité des features dérivées et pas un biais introduit par des features brutes non pertinentes.**
# - **Des mesures ont été mises en place pour l’absence de fuite de données: aucune feature future ou fortement corrélée à la variable cible n’est utilisée directement.**

# #### Conclusion avant les étapes de modélisation 
# - Toutes les features utilisées paraissent pertinentes car cohérentes avec les résultats précédents de l'EDA et des tests statistiques effectués.
# - Pas de fuite de données (data leakage) détectable.
# - Les performances des modèles (Dummy, RandomForest, LightGBM) vont maintenant principalement dépendre de la gestion du déséquilibre entre la classe minoritaire (départ) et la classe majoritaire (reste dans l'entreprise).
# - L’encodage est correct et ne risque pas de maldimensionner les features ou de créer des erreurs dans l’entraînement.
# - Les colonnes d’origine sont transformées ou supprimées pour rendre le modèle plus robuste et pertinent.
# - Les modèles ne sont pas “dépendants” d’une seule feature brute et utilisent un ensemble de features pertinentes qui seront le cas échéant encodées et normalisées.

# ## Etape 3 : Réaliser un premier modèle de classification

# In[281]:


import lightgbm as lgb
print("LightGBM version:", lgb.__version__)


# In[282]:


# Premier essai de modélisation avec la méthode de gestion des poids des classes pour prise en compte 
# du déséquilibre entre les deux classes de la target

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, RocCurveDisplay, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, f1_score
)

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vérification des features utilisées
print("Nb features X_train :", X_train.shape[1])
print("Features utilisées :", X_train.columns.tolist())

# intégrér les features comportementales créée dans le feature engineering aux features numériques
num_features_all = numeric_features + behavioral_features

categorical_features = [
    col for col in X_train.columns
    if col not in num_features_all
]

# Nouveau contrôle des features utilisées
print("\nFeatures numériques :", num_features_all)
print("\nFeatures catégorielles :", categorical_features)

# Conversion des features catégorielles de bool à int pour scikit-learn
bool_cols = X_train.select_dtypes(include='bool').columns.tolist()

X_train[bool_cols] = X_train[bool_cols].astype(int)
X_test[bool_cols] = X_test[bool_cols].astype(int)

# Preprocessor pour définition des traitements à appliquer aux colonnes de X
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_features_all),
        ('cat', 'passthrough', categorical_features)
    ]
)

preprocessor.fit(X_train)

# vérification finale

num_cols_in_model = preprocessor.transformers_[0][2]
cat_cols_in_model = preprocessor.transformers_[1][2]

print("\nFeatures numériques utilisées :")
print(num_cols_in_model)

print("\nFeatures catégorielles utilisées :")
print(cat_cols_in_model)

all_features_used = list(num_cols_in_model) + list(cat_cols_in_model)

print(f"\nNombre total de features utilisées : {len(all_features_used)}")

# Définir les modèles avec gestion du déséquilibre par le poids des classes
models = {
    "Dummy": DummyClassifier(strategy='stratified', random_state=42),
    "LogisticRegression": LogisticRegression(
        solver='liblinear',
        class_weight='balanced',
        random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight='balanced'),
    "LightGBM": LGBMClassifier(
        n_estimators=500, random_state=42, n_jobs=-1,
        scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train))
}

# Définition d'une fonction pour le pipeline d'entraînement des modèles

def train_evaluate_model(model_name, model, X_train, X_test, y_train, y_test, preprocessor):
    print(f"\n==============================")
    print(f"Modèle : {model_name}")
    print(f"==============================")

    # Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Validation croisée (ROC-AUC)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='roc_auc',
        n_jobs=1
    )

    print("\nValidation croisée (ROC-AUC) :")
    print(f"Scores : {np.round(cv_scores, 3)}")
    print(f"Moyenne : {cv_scores.mean():.3f}")
    print(f"Écart-type : {cv_scores.std():.3f}")

    # Probabilité out-of-fold pour le seuil
    y_prob_oof = np.zeros(len(y_train))

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_prob_oof[val_idx] = pipeline.predict_proba(X_val)[:, 1]

    # Optimisation du seuil sur out-of-fold
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []

    for t in thresholds:
        y_pred_oof = (y_prob_oof >= t).astype(int)
        f1_scores.append(f1_score(y_train, y_pred_oof))

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\nSeuil optimal (CV) : {best_threshold:.3f}")
    print(f"F1 (OOF) : {f1_scores[best_idx]:.3f}")

    # Entraînement final sur tout l'ensemble de train
    pipeline.fit(X_train, y_train)

    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, y_prob_train)
    auc_test = roc_auc_score(y_test, y_prob_test)

    print(f"\nROC-AUC train : {auc_train:.3f}")
    print(f"ROC-AUC test  : {auc_test:.3f}")

    # Prédictions avec un seuil fixé
    y_pred_train = (y_prob_train >= best_threshold).astype(int)
    y_pred_test = (y_prob_test >= best_threshold).astype(int)

    # Evaluation
    print("\nClassification report (Train) :")
    print(classification_report(y_train, y_pred_train, zero_division=0))

    print("\nClassification report (Test) :")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    # Matrices de confusion sur l'ensemble de train et l'ensemble de test
    cm_train = confusion_matrix(y_train, y_pred_train, labels=[0,1])
    cm_test = confusion_matrix(y_test, y_pred_test, labels=[0,1])

    # Affichage graphique de la matrice de confusion des ensembles de train et de test
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["Non départ","Départ"])
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.grid(False)
    plt.title(f"Matrice de confusion - {model_name} (Train)")
    plt.show()

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["Non départ","Départ"])
    fig, ax = plt.subplots(figsize=(5,5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.grid(False)
    plt.title(f"Matrice de confusion - {model_name} (Test)")
    plt.show()

    # Affichage numérique des valeurs
    cm_train_df = pd.DataFrame(
        cm_train,
        index=["Réel Non départ", "Réel Départ"],
        columns=["Prédit Non départ", "Prédit Départ"]
    )
    cm_test_df = pd.DataFrame(
        cm_test,
        index=["Réel Non départ", "Réel Départ"],
        columns=["Prédit Non départ", "Prédit Départ"]
    )

    print("\nMatrice de confusion (Train) :")
    display(cm_train_df)

    print("\nMatrice de confusion (Test) :")
    display(cm_test_df)

    # Courbe F1 vs Threshold
    plt.figure()
    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title(f"F1-score vs Threshold (CV - {model_name})")
    plt.grid()
    plt.show()

    # Precision vs Recall sur l'ensemble de test
    precision, recall, _ = precision_recall_curve(y_test, y_prob_test)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Test - {model_name})")
    plt.grid()
    plt.show()

    # Courbe ROC avec AUC

    plt.figure()
    roc_disp = RocCurveDisplay.from_predictions(
        y_test, 
        y_prob_test, 
        name=f"{model_name} (AUC = {roc_auc_score(y_test, y_prob_test):.3f})"
    )
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Courbe ROC - {model_name}")

    # Légende à droite de la figure
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.tight_layout()  # ajuste la figure pour ne pas couper le contenu
    plt.show()

    return pipeline, best_threshold

# Boucler le pipeline ci-dessus sur les modèles
pipelines = {}
for name, model in models.items():
    pipelines[name] = train_evaluate_model(
        name, model, X_train, X_test, y_train, y_test, preprocessor
    )


# #### observations étape 3
# 
# **Vue d’ensemble**
# 
# | Modèle               | ROC-AUC test | Seuil F1 optimal | Recall classe 1 | Précision classe 1 | F1 classe 1 |
# |----------------------|-------------|----------------|----------------|------------------|-------------|
# | Dummy                | 0.477       | 0.000          | 1.00           | 0.16             | 0.28        |
# | LogisticRegression   | 0.746       | 0.687          | 0.40           | 0.43             | 0.42        |
# | RandomForest         | 0.736       | 0.293          | 0.36           | 0.36             | 0.36        |
# | LightGBM             | 0.674       | 0.040          | 0.30           | 0.30             | 0.30        |
# 
# **Objectifs**
# - Tester la gestion du déséquilibre via les poids des classes (`class_weight` ou `scale_pos_weight`).  
# - Comparer avec le Dummy Classifier comme baseline.  
# - Vérifier l’impact sur : recall de la classe minoritaire, précision et comportement global.
#  
# **Dummy Classifier**
# 
# - **Métriques**
# 
# | Métrique | Valeur |
# |----------|-------|
# | ROC-AUC  | 0.477 |
# | Seuil    | 0.000 |
# | Recall   | 1.00  |
# | Precision| 0.16  |
# | F1       | 0.28  |
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 0               | 247           |
# | Réel Départ       | 0               | 47            |
# 
# - **Interprétation**  
#     - Prédit systématiquement la classe minoritaire (Départ).  
#     - Aucun signal informatif.  
#     - Sert uniquement de baseline.
# 
# **Logistic Regression (class_weight='balanced')**
# 
# - **Métriques**
# 
# | Métrique         | Valeur |
# |-----------------|--------|
# | ROC-AUC test     | 0.746  |
# | ROC-AUC train    | 0.845  |
# | Seuil optimal    | 0.687  |
# | Recall classe 1  | 0.40   |
# | Precision classe 1 | 0.43 |
# | F1 classe 1      | 0.42   |
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 222             | 25            |
# | Réel Départ       | 28              | 19            |
# 
# - **Interprétation**  
#     - Points positifs : bonne séparation globale (ROC-AUC ≈ 0.75), meilleure détection des départs par rapport au Dummy.  
#     - Limites : recall encore faible (28/47 départs manqués), précision modérée (25/247 faux positifs).  
#     - Lecture métier : détecte partiellement les départs mais insuffisant pour actions concrètes.
# 
# **RandomForest (class_weight='balanced')**
# 
# - **Métriques**
# 
# | Métrique         | Valeur |
# |-----------------|--------|
# | ROC-AUC test     | 0.736  |
# | ROC-AUC train    | 1.000  |
# | Seuil optimal    | 0.293  |
# | Recall classe 1  | 0.36   |
# | Precision classe 1 | 0.36 |
# | F1 classe 1      | 0.36   |
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 225             | 22            |
# | Réel Départ       | 30              | 17            |
# 
# - **Interprétation**  
#     - Points positifs : très bon pour la classe majoritaire (~91 % correct).  
#     - Limites : overfitting sévère (train ROC-AUC = 1.0), recall faible pour les départs (30/47 manqués), seuil bas (0.293) mais toujours insuffisant.  
#     - Lecture métier : prédit bien les Non-départs mais détecte peu les départs.
# 
# **LightGBM (scale_pos_weight)**
# 
# - **Métriques**
# 
# | Métrique         | Valeur |
# |-----------------|--------|
# | ROC-AUC test     | 0.674  |
# | ROC-AUC train    | 1.000  |
# | Seuil optimal    | 0.040  |
# | Recall classe 1  | 0.30   |
# | Precision classe 1 | 0.30 |
# | F1 classe 1      | 0.30   |
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 214             | 33            |
# | Réel Départ       | 33              | 14            |
# 
# - **Interprétation**  
#     - Points positifs : modèle flexible capable de capturer des relations non linéaires.  
#     - Limites : overfitting élevé, précision faible (66/247 faux positifs), performance inférieure à RandomForest.  
#     - Lecture métier : sensible mais moins précis, moins exploitable pour la détection des départs.
# 
# **Analyse des courbes**
# 
# - **ROC**  
#     - RandomForest : bonne séparation globale mais overfitting visible.  
#     - LightGBM : courbe plus plate, moins performante.  
#     - Dummy : aucune valeur informative.
# 
# -  **Precision-Recall**  
#     - RandomForest : recall améliorable via seuil, mais encore limité.  
#     - LightGBM : quelques points de recall élevé mais précision insuffisante.  
#     - Dummy : constant et non informatif.
# 
# **Analyse des matrices de confusion (Test)**
# 
# | Modèle               | Non départ correct | Départ correct | Commentaire |
# |----------------------|-----------------|----------------|------------|
# | Logistic Regression  | 222 / 247 (~90%) | 19 / 47 (~40%) | Correct pour majoritaire, faible recall minorité |
# | RandomForest         | 225 / 247 (~91%) | 17 / 47 (~36%) | Très conservateur, minorité mal détectée |
# | LightGBM             | 214 / 247 (~86%) | 14 / 47 (~30%) | Plus de faux positifs, overfitting |
# 
# 
# **Conclusions – Étape 3**
# 
# - Gestion du déséquilibre : améliore la détection par rapport à Dummy, mais reste insuffisant pour un usage métier.  
# - RandomForest : meilleur pour la classe majoritaire, mais recall faible pour les départs.  
# - LightGBM : flexible mais exposé à overfitting et faux positifs.  
# - Logistic Regression : simple et stable, compromis intéressant mais recall limité.
# 
# **Stratégies d’amélioration** 
# - tester la méthode d'undersampling pour la gestion du déséquilibre des classes
# - Ajuster `class_weight` pour augmenter le recall minoritaire.  
# - Régularisation pour limiter overfitting  
#   - RandomForest : `max_depth`, `min_samples_leaf`  
#   - LightGBM : `reg_alpha`, `reg_lambda`  
# - Optimisation du seuil F1 via Precision-Recall pour compromis recall/précision métier.  
# - Validation croisée robuste pour évaluer stabilité.  
# - Combinaison : `class_weight` + tuning hyperparamètres + calibration des probabilités.  
# - Exploration avancée : nested CV, feature importance globale, SHAP pour compréhension locale.
# 

# ## Etape 4 : Améliorer l'approche de classification

# In[283]:


### Essai avec undersampling sans le poids des classes pour la gestion du déséquilibre des classes de la target

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

models = {
    "Dummy": DummyClassifier(strategy='stratified', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=500, random_state=42, n_jobs=-1)
}

# On garde toujours le même ensemble de train et de test c'est à dire celui défini au début de l'étape 3
# pour pouvoir objectivement et sans biais comparer les performances sur l'ensemble de test entre les différentes étapes

# Définition d'une fonction pour le pipeline d'entraînement des modèles de l'étape 4 partie 1
def train_evaluate_model_et4_1(model_name, model, X_train, X_test, y_train, y_test, preprocessor):
    print(f"\n==============================")
    print(f"Modèle : {model_name}")
    print(f"==============================")

    # Pipeline avec undersampling sans class weight
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('undersample', RandomUnderSampler(random_state=42)),
        ('classifier', model)
    ])

    # Validation croisée (ROC-AUC)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_n_jobs = 1 if model_name=="LightGBM" else -1

    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring='roc_auc',
        n_jobs=cv_n_jobs
    )

    print("\nValidation croisée (ROC-AUC) :")
    print("Scores :", np.round(cv_scores, 3))
    print("Moyenne :", round(cv_scores.mean(), 3))
    print("Écart-type :", round(cv_scores.std(), 3))

    # Probabilités out-of-fold pour optimisation du seuil
    y_prob_oof = np.zeros(len(y_train))

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        y_prob_oof[val_idx] = pipeline.predict_proba(X_val)[:, 1]

    # Optimisation du seuil (F1) sur out-of-fold
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []

    for t in thresholds:
        y_pred_oof = (y_prob_oof >= t).astype(int)
        f1_scores.append(f1_score(y_train, y_pred_oof))

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\nSeuil optimal (CV) : {best_threshold:.3f}")
    print(f"F1 (OOF) : {f1_scores[best_idx]:.3f}")

    # Entraînement final sur tout l'ensemble de train
    pipeline.fit(X_train, y_train)

    y_prob_train = pipeline.predict_proba(X_train)[:,1]
    y_prob_test  = pipeline.predict_proba(X_test)[:,1]

    auc_train = roc_auc_score(y_train, y_prob_train)
    auc_test  = roc_auc_score(y_test, y_prob_test)

    print(f"\nROC-AUC train : {auc_train:.3f}")
    print(f"ROC-AUC test  : {auc_test:.3f}")

    # Prédictions avec seuil fixé
    y_pred_train = (y_prob_train >= best_threshold).astype(int)
    y_pred_test  = (y_prob_test >= best_threshold).astype(int)

    # Evaluation
    print("\nClassification report (Train) :")
    print(classification_report(y_train, y_pred_train, zero_division=0))

    print("\nClassification report (Test) :")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    # Matrices de confusion
    cm_train = confusion_matrix(y_train, y_pred_train, labels=[0,1])
    cm_test  = confusion_matrix(y_test, y_pred_test, labels=[0,1])

    # Affichage graphique des matrices de confusion (train et test)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["Non départ","Départ"])
    fig, ax = plt.subplots(figsize=(5,5))
    disp_train.plot(ax=ax, cmap='Blues', values_format='d')
    ax.grid(False)
    plt.title(f"Matrice de confusion - {model_name} (Train)")
    plt.show()

    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["Non départ","Départ"])
    fig, ax = plt.subplots(figsize=(5,5))
    disp_test.plot(ax=ax, cmap='Blues', values_format='d')
    ax.grid(False)
    plt.title(f"Matrice de confusion - {model_name} (Test)")
    plt.show()

    # Affichage numérique des matrices de confusion (train et test)
    cm_train_df = pd.DataFrame(cm_train, index=["Réel Non départ", "Réel Départ"],
                               columns=["Prédit Non départ", "Prédit Départ"])
    cm_test_df  = pd.DataFrame(cm_test, index=["Réel Non départ", "Réel Départ"],
                               columns=["Prédit Non départ", "Prédit Départ"])

    print("\nMatrice de confusion (Train) :")
    display(cm_train_df)

    print("\nMatrice de confusion (Test) :")
    display(cm_test_df)

    # Courbe F1 vs Threshold (out-of-fold)
    plt.figure()
    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title(f"F1-score vs Threshold (CV - {model_name})")
    plt.grid()
    plt.show()

    # Precision vs Recall sur l'ensemble de test
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_test)
    plt.figure()
    plt.plot(recall_curve, precision_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (Test - {model_name})")
    plt.grid()
    plt.show()

    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    roc_auc_val = auc(fpr, tpr)

    return {
        "model": pipeline,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc_val,
        "precision": precision_curve,
        "recall": recall_curve,
        "best_threshold": best_threshold
    }

# Stocker les résultats
results = {}

# Boucle sur les modèles
for name, model in models.items():
    print(f"\n==================== Entraînement : {name} ====================")
    results[name] = train_evaluate_model_et4_1(
        model_name=name,
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor
    )

# Courbe ROC comparée
plt.figure(figsize=(8,6))
for name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['roc_auc']:.2f})")
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC - Comparaison des modèles")
plt.xticks(np.arange(0,1.1,0.2))
plt.yticks(np.arange(0,1.1,0.2))
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Courbe Precision-Recall comparée
plt.figure(figsize=(8,6))
for name, res in results.items():
    plt.plot(res["recall"], res["precision"], label=f"{name}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Courbe Precision-Recall - Comparaison des modèles")
plt.xticks(np.arange(0,1.1,0.2))
plt.yticks(np.arange(0,1.1,0.2))
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Affichage des principaux points ROC et PR
def afficher_points_principaux(results, model_name, n_points=10):
    res = results[model_name]
    # ROC
    roc_df = pd.DataFrame({"FPR": res["fpr"], "TPR": res["tpr"]})
    # PR
    pr_df = pd.DataFrame({"Recall": res["recall"], "Precision": res["precision"]})

    # Échantillonnage pour limiter le nombre de points affichés
    roc_sampled = roc_df.iloc[::max(1, len(roc_df)//n_points)]
    pr_sampled  = pr_df.iloc[::max(1, len(pr_df)//n_points)]

    print(f"\n--- Points principaux ROC : {model_name} ---")
    display(roc_sampled.reset_index(drop=True))

    print(f"\n--- Points principaux Precision-Recall : {model_name} ---")
    display(pr_sampled.reset_index(drop=True))

# Affichage pour tous les modèles
for name in models.keys():
    afficher_points_principaux(results, name, n_points=10)


# #### Observations pour l'étape 4 partie undersampling
# 
# **Objectifs de l’essai**
# - **Approche testée** : undersampling de la classe majoritaire (`RandomUnderSampler`) sans `class_weight`.  
# - **But** : gérer le déséquilibre des classes dans la target.  
# - **Comparaison** : avec l’approche précédente basée sur `class_weight`.  
# - **Focus métriques** : ROC-AUC, recall/précision de la classe minoritaire (départs), F1-score pour compromis global.
# 
# **Résultats clés**
# 
# | Modèle        | ROC-AUC test | Seuil F1 optimal | Recall classe 1 | Precision classe 1 | F1 classe 1 |
# |---------------|-------------|----------------|----------------|------------------|-------------|
# | Dummy         | 0.506       | 0.000          | 1.00           | 0.16             | 0.28        |
# | RandomForest  | 0.710       | 0.646          | 0.32           | 0.35             | 0.33        |
# | LightGBM      | 0.628       | 0.939          | 0.38           | 0.26             | 0.31        |
# 
# **Dummy CLassifier**
# 
# - **Analyse**  
#     - Aucune valeur informative, prédit tout en classe 1.  
#     - Seuil F1 = 0 > modèle “aveugle” face aux données réelles.  
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 0               | 247           |
# | Réel Départ       | 0               | 47            |
# 
# - **Insight métier**  
#     - Impossible à utiliser pour détecter les départs.
# 
# **RandomForest (undersampling)**
# 
# - **Points positifs**  
#     - ROC-AUC test correct (~0.71), léger gain sur le baseline.
# 
# - **Limites**  
#     - Overfitting massif (train ROC-AUC 0.991 vs test 0.710)  
#     - Recall faible (~32%) > 32 départs manqués sur 47  
#     - Modèle conservateur malgré l’undersampling
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 219             | 28            |
# | Réel Départ       | 32              | 15            |
# 
# - **Insight métier**  
#     - Capte peu les départs, privilégie la classe majoritaire  
#     - Peu exploitable pour actions RH
# 
# **LightGBM (undersampling)**
# 
# - **Points positifs**  
#     - Flexible, capture certaines relations non linéaires  
#     - Modélise interactions complexes entre features
# 
# - **Limites**  
#     - Overfitting important (train ROC-AUC 0.978 vs test 0.628)  
#     - Recall faible (~38%), précision faible (~26%)  
#     - Performance inférieure à RandomForest
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 196             | 51            |
# | Réel Départ       | 29              | 18            |
# 
# - **Insight métier**  
#     - Encore trop de faux négatifs pour être opérationnel
# 
# **Analyse des courbes**
# 
# - **ROC**  
#     - RandomForest : bonne séparation globale mais overfitting visible  
#     - LightGBM : moins performant globalement  
#     - Dummy : inutilisable, aucune valeur discriminante
# 
# - **Precision-Recall**  
#     - RandomForest : léger gain possible via seuil, recall reste faible  
#     - LightGBM : recall parfois élevé mais trop de faux positifs > précision faible  
#     - Dummy : constante, inutilisable
# 
# **Comparaison avec class_weight**
# 
# | Aspect                       | Undersampling seul        | Class_weight |
# |-------------------------------|-------------------------|-------------------------------|
# | Recall classe minoritaire     | Faible (~32%)           | Amélioré (~40%)              |
# | Precision classe minoritaire  | Faible (~26-35%)        | Plus stable (~36-43%)        |
# | Overfitting                   | Important (train >> test)| Contrôlé via régularisation  |
# | Signal exploitable            | Limité                  | Plus exploitable métier       |
# | Impact sur la classe majoritaire | Biais vers minorité selon seuil | Meilleur compromis |
# 
# **Conclusions**
# 
# - L’undersampling seul ne suffit pas :  
#   - Perte d’information sur la classe majoritaire  
#   - Instabilité et overfitting élevé  
#   - Recall insuffisant > risque métier RH important
# 
# - Comparé à `class_weight` :  
#   - `class_weight` + ajustement de seuil F1/PR offre un meilleur compromis recall/precision  
#   - RandomForest reste plus robuste que LightGBM sur ce dataset malgré l’undersampling
# 
# **Actions d'amélioration possibles pour l'essai suivant**
# 
# - Utiliser `class_weight` 
# - Optimiser le seuil de décision via F1-score ou Precision-Recall selon priorité métier  
# - Régulariser les modèles :  
#   - RandomForest : `max_depth`, `min_samples_leaf`  
#   - LightGBM : `num_leaves`, `max_depth`, `min_data_in_leaf`  
# - Calibration des probabilités pour un seuil décisionnel plus fiable  
# - Tester modèles boostés mieux régularisés : CatBoost, XGBoost avec `scale_pos_weight`
# 
# **Insight clé**  
# - L’objectif métier étant de détecter les départs, l’undersampling seul est insuffisant pour garantir un recall satisfaisant et limiter les faux négatifs.

# In[284]:


# Essai d'optimisation avec poids des classes pour la gestion du déséquilibre entre les classes de la target

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Définition des modèles
models = {
    "Dummy": DummyClassifier(strategy='stratified', random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=500, random_state=42,
        class_weight='balanced', max_depth=8, min_samples_leaf=5
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=500, random_state=42, n_jobs=-1,
        scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train),
        num_leaves=31, min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1
    )
}

# On garde toujours le même ensemble de train et de test c'est à dire celui défini au début de l'étape 3
# pour pouvoir objectivement et sans biais comparer les performances sur l'ensemble de test entre les différentes étapes

# Fonction d'entraînement et d'évaluation
def train_evaluate_model_et4_2(model_name, model, X_train, X_test, y_train, y_test, preprocessor):
    print(f"\n==============================\nModèle : {model_name}\n==============================")

    # Pipeline 
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Validation croisée ROC-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1)
    print("\nValidation croisée (ROC-AUC) :")
    print("Scores par fold :", np.round(cv_scores, 3))
    print("Moyenne :", round(cv_scores.mean(), 3))
    print("Écart-type :", round(cv_scores.std(), 3))

    # Out-of-fold pour seuil F1
    y_prob_oof = np.zeros(len(y_train))
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        pipeline.fit(X_tr, y_tr)
        y_prob_oof[val_idx] = pipeline.predict_proba(X_val)[:,1]

    thresholds = np.linspace(0,1,100)
    f1_scores = [f1_score(y_train, (y_prob_oof >= t).astype(int)) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"\nSeuil optimal (F1 - OOF) : {best_threshold:.3f}, F1 max : {f1_scores[np.argmax(f1_scores)]:.3f}")

    # F1 vs Threshold
    plt.figure(figsize=(8,5))
    plt.plot(thresholds, f1_scores, label="F1-score")
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Seuil optimal = {best_threshold:.2f}')
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title(f"F1-score vs Threshold - {model_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    # Entraînement final
    pipeline.fit(X_train, y_train)
    y_prob_train = pipeline.predict_proba(X_train)[:,1]
    y_prob_test  = pipeline.predict_proba(X_test)[:,1]

    y_pred_train = (y_prob_train >= best_threshold).astype(int)
    y_pred_test  = (y_prob_test >= best_threshold).astype(int)

    print(f"\nROC-AUC train : {roc_auc_score(y_train, y_prob_train):.3f}")
    print(f"ROC-AUC test  : {roc_auc_score(y_test, y_prob_test):.3f}")
    print("\nClassification report (Test) :")
    print(classification_report(y_test, y_pred_test, zero_division=0))

    # Matrices de confusion
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test  = confusion_matrix(y_test, y_pred_test)
    print("\nMatrice de confusion (Train) :")
    display(pd.DataFrame(cm_train, index=["Réel Non départ","Réel Départ"], columns=["Prédit Non départ","Prédit Départ"]))
    print("\nMatrice de confusion (Test) :")
    display(pd.DataFrame(cm_test, index=["Réel Non départ","Réel Départ"], columns=["Prédit Non départ","Prédit Départ"]))

    # ROC et PR
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_test)
    roc_auc_val = auc(fpr, tpr)
    pr_auc_val  = auc(recall_curve, precision_curve)

    return {
        "model": pipeline,
        "best_threshold": best_threshold,
        "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc_val,
        "precision": precision_curve, "recall": recall_curve, "pr_auc": pr_auc_val
    }

# Boucle sur tous les modèles
results = {}
for name, model in models.items():
    results[name] = train_evaluate_model_et4_2(name, model, X_train, X_test, y_train, y_test, preprocessor)


# Courbes comparatives

# ROC
plt.figure(figsize=(8,6))
for name, res in results.items():
    plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['roc_auc']:.2f})")
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Courbe ROC - Comparaison des modèles")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Precision-Recall
plt.figure(figsize=(8,6))
for name, res in results.items():
    plt.plot(res["recall"], res["precision"], label=f"{name} (AUPR={res['pr_auc']:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Courbe Precision-Recall - Comparaison des modèles")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# Points principaux
def afficher_points_principaux(results, model_name, n_points=10):
    res = results[model_name]
    roc_df = pd.DataFrame({"FPR": res["fpr"], "TPR": res["tpr"]})
    pr_df  = pd.DataFrame({"Recall": res["recall"], "Precision": res["precision"]})
    roc_sampled = roc_df.iloc[::max(1,len(roc_df)//n_points)]
    pr_sampled  = pr_df.iloc[::max(1,len(pr_df)//n_points)]
    print(f"\n--- Points principaux ROC : {model_name} ---")
    display(roc_sampled.reset_index(drop=True))
    print(f"\n--- Points principaux Precision-Recall : {model_name} ---")
    display(pr_sampled.reset_index(drop=True))

for name in models.keys():
    afficher_points_principaux(results, name, n_points=10)


# #### observations pour l 'étape 4 partie 2 (gestion poids des classes avec régularisation et seuil F1 optimisé)
# 
# **Résumé des performances – Étape 4 (class_weight + régularisation + seuil F1)**
# 
# | Modèle        | ROC-AUC test | AUPR test | Seuil F1 | Recall classe 1 | Précision classe 1 | F1 classe 1 |
# |---------------|-------------|-----------|----------|----------------|------------------|-------------|
# | Dummy         | 0.477       | 0.16      | 0.000    | 1.00           | 0.16             | 0.28        |
# | RandomForest  | 0.730       | 0.48      | 0.414    | 0.55           | 0.33             | 0.41        |
# | LightGBM      | 0.688       | 0.36      | 0.081    | 0.45           | 0.31             | 0.37        |
# 
# **Dummy classifier**
# 
# - **Analyse**  
#     - ROC-AUC test = 0.477 > aucun signal informatif  
#     - Recall = 1.0 mais précision = 0.16 > prédit systématiquement la classe minoritaire  
#     - F1 = 0.28 > baseline non exploitable
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 0               | 247           |
# | Réel Départ       | 0               | 47            |
# 
# - **Insight métier**  
#     - Inutile pour la détection des départs, sert uniquement de référence
# 
# **RandomForest (class_weight + régularisation + seuil F1)**
# 
# - **Performance globale**  
#     - ROC-AUC test : 0.730  
#     - ROC-AUC train : 0.964 > overfitting réduit grâce à la régularisation  
#     - Recall classe 1 : 0.55 > amélioration notable par rapport à l'undersampling (~0.32)  
#     - Precision classe 1 : 0.33  
#     - F1 classe 1 : 0.41 > meilleur compromis précision/rappel
# 
# - **Matrice de confusion (Train)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 845             | 141           |
# | Réel Départ       | 11              | 179           |
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 194             | 53            |
# | Réel Départ       | 21              | 26            |
# 
# - **Analyse métier**  
#     - Détecte 26/47 départs > 55 % recall  
#     - Modèle conservatif sur la classe majoritaire mais compromis global satisfaisant  
#     - Amélioration notable du rappel par rapport à l'undersampling, ROC stable
# 
# **LightGBM (scale_pos_weight + régularisation + seuil F1)**
# 
# - **Performance globale**  
#     - ROC-AUC test : 0.688  
#     - ROC-AUC train : 1.000 > overfitting important malgré régularisation (`reg_alpha=0.1`, `reg_lambda=0.1`)  
#     - Recall classe 1 : 0.45 > amélioration par rapport à l'undersampling (~0.38)  
#     - Precision classe 1 : 0.31  
#     - F1 classe 1 : 0.37
# 
# - **Matrice de confusion (Train)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 986             | 0             |
# | Réel Départ       | 0               | 190           |
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ   | 201             | 46            |
# | Réel Départ       | 26              | 21            |
# 
# - **Analyse métier**  
#     - Rappel amélioré mais moins bon que RandomForest pour les départs  
#     - Classe majoritaire bien prédite (201/247 corrects)  
#     - Compromis global légèrement moins favorable que RandomForest
# 
# **Comparaison avec l’undersampling précédent**
# 
# | Modèle        | ROC-AUC test      | Recall classe 1     | F1 classe 1       | Commentaire |
# |---------------|-----------------|-------------------|-----------------|-------------|
# | RandomForest  | 0.710 > 0.730    | 0.32 > 0.55       | 0.33 > 0.41     | class_weight + régularisation + seuil F1 améliore rappel et F1 |
# | LightGBM      | 0.628 > 0.688    | 0.38 > 0.45       | 0.31 > 0.37     | Légère amélioration AUC et F1, rappel augmenté |
# | Dummy         | 0.506 > 0.477    | 1.00               | 0.28             | Inchangé, baseline non exploitable |
# 
# **Conclusions**
# - **RandomForest** : meilleur modèle pour détecter les départs avec compromis précision/rappel correct  
# - **LightGBM** : amélioration visible mais moins robuste pour rappel des départs  
# - **Dummy** : inutile pour la détection, sert de baseline
# 
# **Actions possibles avant étape 5**
# - Feature engineering et sélection pour réduire l’overfitting  
# - Régularisation supplémentaire :  
#   - RandomForest : `max_depth`, `min_samples_leaf`  
#   - LightGBM : `num_leaves`, `min_child_samples`, `reg_alpha`, `reg_lambda`  
# - Ajustement du seuil métier selon coût des faux positifs et faux négatifs  
# - Calibration des probabilités pour un seuil décisionnel plus fiable

# ##  Etape 5 : optimiser et interpréter le comportement du modèle retenu avec nestedCV pour éviter fuite interne de données durant le tuning des hyperparamètres

# In[285]:


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, f1_score, roc_curve, auc
)
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# On garde toujours le même ensemble de train et de test c'est à dire celui défini au début de l'étape 3
# pour pouvoir objectivement et sans biais comparer les performances sur l'ensemble de test entre les différentes étapes

# Pipeline
pipeline_rf = ImbPipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

param_grid = {
    'classifier__n_estimators': [300, 500],
    'classifier__max_depth': [5, 8],
    'classifier__min_samples_leaf': [2, 5],
    'classifier__max_features': ['sqrt', 'log2']
}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_prob_oof = np.zeros(len(y_train))

# Nested CV pour probabilités out of fold correctes (OOF)
for train_idx, val_idx in outer_cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # GridSearch interne
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search_fold = GridSearchCV(
        clone(pipeline_rf),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=inner_cv,
        n_jobs=-1
    )
    grid_search_fold.fit(X_tr, y_tr)

    # Meilleur modèle de ce fold uniquement
    best_fold_model = clone(grid_search_fold.best_estimator_)
    best_fold_model.fit(X_tr, y_tr)

    # Probabilités OOF
    y_prob_oof[val_idx] = best_fold_model.predict_proba(X_val)[:,1]

# Fonction pour courbe F1 vs Threshold
def plot_f1_vs_threshold(model_name, y_true, y_prob_oof):
    thresholds = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_true, (y_prob_oof >= t).astype(int)) for t in thresholds]

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    plt.figure(figsize=(8,5))
    plt.plot(thresholds, f1_scores, label="F1-score")
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Seuil optimal = {best_threshold:.2f}')
    plt.xlabel("Threshold")
    plt.ylabel("F1-score")
    plt.title(f"F1-score vs Threshold - {model_name}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
    print(f"F1 max : {f1_scores[best_idx]:.3f} au seuil {best_threshold:.3f}")
    return best_threshold

best_threshold = plot_f1_vs_threshold("RandomForest (OOF)", y_train, y_prob_oof)

# Entraînement final sur tout le train avec meilleurs hyperparamètres
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search_final = GridSearchCV(
    clone(pipeline_rf),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=inner_cv,
    n_jobs=-1
)
grid_search_final.fit(X_train, y_train)
best_rf_model = clone(grid_search_final.best_estimator_)
best_rf_model.fit(X_train, y_train)

# Probabilités et prédictions avec le seuil optimal
y_prob_train = best_rf_model.predict_proba(X_train)[:,1]
y_pred_train = (y_prob_train >= best_threshold).astype(int)
y_prob_test = best_rf_model.predict_proba(X_test)[:,1]
y_pred_test = (y_prob_test >= best_threshold).astype(int)

# ROC et Precision-Recall + classification report
fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train)
roc_auc_train = auc(fpr_train, tpr_train)
fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure(figsize=(8,6))
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label=f'ROC (AUC={roc_auc_test:.3f})')
plt.plot([0,1],[0,1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Courbe ROC - RandomForest')
plt.grid(alpha=0.3)
plt.legend(loc='lower right')
plt.show()

precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob_test)
pr_auc_val = auc(recall_curve, precision_curve)
plt.figure(figsize=(8,6))
plt.plot(recall_curve, precision_curve, color='green', lw=2, label=f'Precision-Recall (AUPR={pr_auc_val:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbe Precision-Recall - RandomForest')
plt.grid(alpha=0.3)
plt.legend(loc='lower left')
plt.show()

print("\nClassification report (Train) :")
print(classification_report(y_train, y_pred_train, zero_division=0))
cm_train = confusion_matrix(y_train, y_pred_train)
display(pd.DataFrame(cm_train,
                     index=["Réel Non départ","Réel Départ"],
                     columns=["Prédit Non départ","Prédit Départ"]))

print("\nClassification report (Test) :")
print(classification_report(y_test, y_pred_test, zero_division=0))
cm_test = confusion_matrix(y_test, y_pred_test)
display(pd.DataFrame(cm_test,
                     index=["Réel Non départ","Réel Départ"],
                     columns=["Prédit Non départ","Prédit Départ"]))

# K-Fold final pour robustesse (avec seuil F1 par fold)
roc_auc_scores = []
f1_scores_cv = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model_fold = clone(best_rf_model)
    model_fold.fit(X_tr, y_tr)
    y_val_prob = model_fold.predict_proba(X_val)[:,1]

    roc_auc_scores.append(roc_auc_score(y_val, y_val_prob))

    precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)
    f1_fold = 2*(precision*recall)/(precision+recall+1e-6)
    best_thresh_fold = thresholds[np.argmax(f1_fold)]
    y_val_pred = (y_val_prob >= best_thresh_fold).astype(int)
    f1_scores_cv.append(f1_score(y_val, y_val_pred))

print(f"Validation croisée K-Fold (5 folds) :")
print(f"ROC-AUC moyen ± std : {np.mean(roc_auc_scores):.3f} ± {np.std(roc_auc_scores):.3f}")
print(f"F1 moyen ± std : {np.mean(f1_scores_cv):.3f} ± {np.std(f1_scores_cv):.3f}")


# Feature importance Native, Permutation et SHAP
X_test_transformed = best_rf_model.named_steps['preprocessor'].transform(X_test)
rf_classifier = best_rf_model.named_steps['classifier']
feature_names_transformed = best_rf_model.named_steps['preprocessor'].get_feature_names_out()

print("Nb features transformées :", len(feature_names_transformed))
print("Nb importances :", len(rf_classifier.feature_importances_))

# Native
rf_importances = pd.DataFrame({
    'feature': feature_names_transformed,
    'importance': rf_classifier.feature_importances_
}).sort_values(by='importance', ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x='importance', y='feature', data=rf_importances, palette='viridis', hue='feature')
plt.title("RandomForest Feature Importance (Native)")
plt.tight_layout()
plt.show()
display(rf_importances)

# Permutation
perm_importance = permutation_importance(
    rf_classifier, X_test_transformed, y_test, n_repeats=10, random_state=42, n_jobs=-1
)
perm_df = pd.DataFrame({
    'feature': feature_names_transformed,
    'importance': perm_importance.importances_mean
}).sort_values(by='importance', ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x='importance', y='feature', data=perm_df, palette='viridis', hue='feature')
plt.title("Permutation Feature Importance")
plt.tight_layout()
plt.show()
display(perm_df)

# SHAP avec le Beeswarm
explainer = shap.TreeExplainer(rf_classifier)
shap_values = explainer(X_test_transformed)
if shap_values.values.ndim == 3 and shap_values.values.shape[2] == 2:
    shap_values_class1 = shap.Explanation(
        values=shap_values.values[:, :, 1],
        base_values=shap_values.base_values[:,1].mean() if shap_values.base_values.ndim==2 else shap_values.base_values[1],
        data=shap_values.data,
        feature_names=feature_names_transformed
    )
else:
    shap_values_class1 = shap_values

shap.summary_plot(shap_values_class1, features=X_test_transformed, feature_names=feature_names_transformed, plot_type="dot")

shap_importance = pd.DataFrame({
    'feature': feature_names_transformed,
    'mean_abs_shap': np.abs(shap_values_class1.values).mean(axis=0)
}).sort_values(by='mean_abs_shap', ascending=False)
display(shap_importance)


# Feature importance locale avec Waterfall (3 positifs + 3 négatifs)
indices_pos = np.where(y_test == 1)[0][:3]
indices_neg = np.where(y_test == 0)[0][:3]
indices = np.concatenate([indices_pos, indices_neg])

local_shap_values_list = []

for i in indices:
    expl = shap.Explanation(
        values=shap_values_class1.values[i],
        base_values=shap_values_class1.base_values,
        data=shap_values_class1.data[i],
        feature_names=feature_names_transformed
    )
    shap.waterfall_plot(expl)

    df_local = pd.DataFrame({
        'feature': feature_names_transformed,
        'shap_value': expl.values
    }).sort_values(by='shap_value', key=np.abs, ascending=False)
    local_shap_values_list.append((i, df_local))

print("\n=== Feature Importance locale (Waterfall) pour les échantillons choisis ===")
for idx, df_local in local_shap_values_list:
    print(f"\n--- Échantillon {idx} ---")
    display(df_local)


# #### observations pour l'étape 5
# 
# **Évolution des performances du modèle Random Forest (Étapes 3 à 5)**
# 
# **Tableau synthèse des étapes**
# 
# | Étape                                | ROC-AUC test | Precision (classe 1) | Recall (classe 1) | F1 (classe 1) | Lecture |
# |-------------------------------------|-------------|--------------------|-----------------|---------------|---------|
# | Étape 3 (RF baseline)                | 0.736       | 0.36               | 0.36            | 0.36          | Modèle très overfit (AUC train = 1.0), faible détection des départs |
# | Étape 4.1 (undersampling)           | 0.710       | 0.35               | 0.32            | 0.33          | Dégradation légère > perte d’information avec undersampling |
# | Étape 4.2 (class_weight + régularisation) | 0.730 | 0.33 | 0.55 | 0.41 | Forte amélioration du recall > meilleure détection des départs |
# | Étape 5 (nested CV + seuil optimisé)| 0.732       | 0.41               | 0.45            | 0.43          | Meilleur équilibre précision / recall > modèle plus exploitable, stable et robuste |
# 
# **Lecture de la progression**
# 
# - **Étape 3 > Étape 4.1 (undersampling)**  
#   - Dégradation du F1 et du recall > perte d’information  
#   - Utilité exploratoire mais sous-optimale seule  
# 
# - **Étape 4.1 > Étape 4.2 (class_weight + régularisation)**  
#   - Recall : 0.32 > 0.55  
#   - F1 : 0.33 > 0.41  
#   - Précision légèrement sacrifiée pour mieux détecter les départs (plus de faux positifs)  
# 
# - **Étape 4.2 > Étape 5 (nested CV + seuil F1 optimisé)**  
#   - Stabilisation des métriques et meilleure robustesse  
#   - Rappel légèrement redescendu (0.45) mais précision augmente (0.41)  
#   - F1 max (Out Of Fold) = 0.508 au seuil 0.495  
#   - ROC-AUC Test stable à 0.732 > séparation correcte des classes  
# 
# **Performance globale du modèle Random Forest – Étape 5**
# 
# | Métrique                    | Valeur obtenue                 | Interprétation |
# |------------------------------|-------------------------------|----------------|
# | F1 max (OOF)                 | 0.508 au seuil 0.495          | Seuil ajusté pour équilibrer précision/recall sur la classe minoritaire |
# | ROC-AUC test                 | 0.732                         | Capacité correcte de séparation des classes |
# | Classification report train  | F1 classe 1 = 0.73            | Bon apprentissage sur la classe minoritaire, léger surapprentissage sur la classe majoritaire |
# | Classification report test   | F1 classe 1 = 0.43            | Performance réduite sur la classe minoritaire, signe de déséquilibre et complexité du problème |
# | Validation K-Fold (5 folds) | ROC-AUC moyen +/- std = 0.788 +/- 0.042, F1 moyen +/- std = 0.526 +/- 0.048 | Robuste sur différentes partitions |
# 
# - **Matrice de confusion (Test)**
# 
# |                   | Prédit Non départ | Prédit Départ |
# |-------------------|-----------------|---------------|
# | Réel Non départ (0) | 217             | 30            |
# | Réel Départ (1)     | 26              | 21            |
# 
# - Classe 0 (“reste”) : Recall = 0.88, Precision = 0.89  
# - Classe 1 (“départ”) : Recall = 0.45, Precision = 0.41  
# 
# - **Synthèse** : **le modèle prédit très bien ceux qui restent, mais détecte moins de la moitié des départs ce qui peut être utile pour actions RH ciblées avec un certain taux de faux positifs**.
# 
# **Impact des hyperparamètres**
# 
# - `max_depth=5` : limite l’overfitting  
# - `min_samples_leaf=5` : stabilise les arbres  
# - `max_features='log2'` : diversité des arbres > robustesse  
# - Seuil F1 optimal ~0.495 : compromis précision / recall sur la classe minoritaire  
# 
# **Feature importance globale – Étape 5**
# - **Méthode Random Forest (Gini)**
#     - **Top 5 features**  
#     1. `num__anciennete_sous_responsable_ratio` – 0.101  
#     2. `num__niveau_poste_vs_revenu` – 0.092  
#     3. `num__anciennete_ratio` – 0.082  
#     4. `num__ratio_salaire_experience` – 0.078  
#     5. `num__stress` – 0.067  
# 
# -**Méthode Permutation Importance**
#     - **Top 5 features**  
#     1. `num__niveau_poste_vs_revenu` – 0.0122  
#     2. `num__satisfaction_moyenne` – 0.0119  
#     3. `cat__poste_manager` – 0.0095  
#     4. `num__anciennete_ratio` – 0.0092  
#     5. `cat__frequence_deplacement_aucun` – 0.0088  
# 
# -**Méthode SHAP (mean absolute)**
#     - **Top 5 features**  
#     1. `num__anciennete_sous_responsable_ratio` – 0.042  
#     2. `num__stress` – 0.038  
#     3. `cat__heures_supplementaires_oui` – 0.035  
#     4. `cat__heures_supplementaires_non` – 0.029  
#     5. `num__satisfaction_moyenne` – 0.028  
# 
# - **Insights**
#     - **Random Forest (Gini)**
#       - Met en avant :
#         - ancienneté (anciennete_*)
#         - ratio poste / salaire
#         - un peu de stress
#       - Biais connu :
#         - favorise les variables continues / forte variance
#         - sous-estime les variables catégorielles (ex : heures sup)
#     
#     - **Permutation Importance**
#       - Met en avant :
#         - niveau_poste_vs_revenu
#         - satisfaction_moyenne
#         - variables métier (poste, déplacement)
#       - Avantage :
#         - plus fiable que Gini
#         - basé sur l’impact réel sur la performance
#     
#     - **SHAP (le plus interprétable)**
#       - Met en avant :
#         - stress
#         - heures supplémentaires
#         - satisfaction
#         - ancienneté
#       - Forces :
#         - impact local
#         - capture les interactions
#     
#     - **Validation des insights**
#     
#       - **Insight 1:**
#         - “Les features liées aux heures supplémentaires et au stress ont un impact plus fort que le Gini ne le montre”
#         - Conclusion :
#           - Gini :
#             - stress présent mais pas dominant
#             - heures sup absentes
#           - SHAP :
#             - stress = rang #2
#             - heures sup = rangs #3 et #4
#           - Donc :
#             - Gini sous-estime ces variables
#             - SHAP révèle leur vrai impact
#     
#       - **Insight 2 :**
#         - “Ancienneté et satisfaction restent critiques”
#         - Conclusion :
#           - Ancienneté :
#             - Gini :rangs  #1, #3
#             - SHAP : rang #1
#             - Permutation :rang  #4
#             - => signal robuste
#           - Satisfaction :
#             - Permutation : rang #2
#             - SHAP : rang #5
#             - => toujours importante mais moins dominante
#     
#     - **Lecture globale des insights**
#     
#       - **Variables structurelles (stables)**
#         - ancienneté
#         - salaire vs poste
#         - => drivers fondamentaux
#     
#       - **Variables comportementales (sous-estimées par Gini)**
#         - stress
#         - heures supplémentaires
#         - => drivers cachés mais puissants
#     
#       - **Variables contextuelles**
#         - satisfaction
#         - poste
#         - déplacement
#         - => impact diffus
#         - Features liées aux heures supplémentaires et au stress ont un impact plus fort sur la prédiction que le Gini ne le montre  
#         - Ancienneté et satisfaction restent critiques  
# 
# **Feature importance locale – SHAP Waterfall**
# 
# **Exemples d'échantillons**  
# - **Échantillon 4** : `stress`, `anciennete_sous_responsable_ratio`, `heures_supplementaires_oui/non` dominent la prédiction 
# - **Échantillon 20** : `stress`, `poste_representant_commercial`, `heures_supplementaires_oui` → départ probable  
# - **Échantillon 21** : `satisfaction_moyenne`, `heures_supplementaires_oui`, `anciennete_sous_responsable_ratio` → départ peu probable ou probable selon signe SHAP  
# 
# **Conclusion à l'échelle locale**    
# - SHAP Waterfall permet d’expliquer les départs à l’échelle individuelle
# - Les échantillons montrent que le stress, la satisfaction moyenne (ressenti des salariés) et les heures supplémentaires peuvent être des facteurs de départ
# 
# **Synthèse finale et recommandations métier**
# 
# - Variables clés du turnover : ancienneté sous-responsable, stress, heures supplémentaires, satisfaction moyenne  
# - Classe minoritaire difficile à prédire : rappel ~0.45, F1 ~0.43  
# - Classe majoritaire : très bien prédite > moins de fausses alertes  
# - Modèle robuste et stable grâce à nested CV et seuil optimisé  
# 
# **Recommandations** 
# - Optimisation future : si l’objectif métier est de maximiser le rappel des départs, on pourrait ajuster le seuil pour prédire plus de départs, au prix de diminuer la précision.
# - Calibration du modèle : vérifier si un recalibrage (CalibratedClassifierCV) améliore la probabilité prédictive pour la classe minoritaire.
# - Feature engineering : combiner les variables sur le stress et les heures supplémentaires pour capturer l’effet cumulatif.
# - Monitoring RH : utiliser les SHAP locaux pour identifier des patterns de départ individuels, utile pour actions préventives.
# 
# **Synthèse d'un point de vue métier**
# - **Le modèle Random Forest de l'étape 5 prédit mieux les salariés qui vont rester (classe majoritaire) que ceux qui vont partir.**
# - **Il détecte certains départs, mais pas tous ce qui est utile pour cibler des actions de rétention, mais il faut accepter un certain taux de faux positifs et de départs manqués.**
# 

# In[ ]:




