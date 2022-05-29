# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 17:36:44 2022

@author:  Achraf Antri
"""

#ex1
# Exemple de création de data frame
import pandas as pd
data = {"gov": ["Tunis", "Tunis", "Kairouan","Kairouan", "Sfax",
"Sfax", "Gafsa", "Gafsa"], "annee": [2009, 2010, 2009, 2010, 2009,
2010, 2009, 2010],
"pop": [996.4, 1000.3, 558.2, 559.7, 918.5, 931.0, 335.1,
338.1]};
stat = pd.DataFrame(data);
print(stat);
# Fixer l'ordre des colonnes
stat=pd.DataFrame(data, columns=["annee", "gov", "pop"]);
print(stat);
# Fixer l'index des lignes ainsi qu'une nouvelle colonne "debt"
ayant des valeurs manquantes (NaN)
stat2=pd.DataFrame(data, columns=["annee", "gov","pop",
"debt"],index=["un", "deux", "trois", "quatre", "cinq", "six",
"sept", "huit" ]);
print(stat2);
# Lister les noms des colonnes
print(stat.columns);
# Afficher les valeurs d’une colonnes
# première méthode
print(stat["gov"]);
# deuxième méthode
print(stat.annee);
# "imputation": traiter les valeurs manquantes
stat2["debt"] = 16.5;
print(stat2);
# Création d'une nouvelle variable
stat2["nord"] = (stat2.gov == "Tunis")|(stat2.gov == "Kairouan");
print(stat2);
# Suppression d'une variable
del stat2["nord"];
print(stat2.columns);

# Enregistrer le dataframe
# Ecriture au format texte
stat2.to_csv("exemple.txt",sep="\t",encoding="utf-8", index=False)
# on regarde ce qui a été enregistré
# ouvrir un fichier txt et afficher son contenu
with open("exemple.txt", "r", encoding="utf-8") as f : text =
f.read()
print(text)
# on enregistre au format Excel
stat2.to_excel("exemple.xlsx", index=False)
# on ouvre Excel sur ce fichier (sous Windows)
from pyquickhelper.loghelper import run_cmd
from pyquickhelper.loghelper.run_cmd import skip_run_cmd
out,err = run_cmd("exemple.xlsx", wait = False)

#EX2
import pandas
# Fixer le nombre de lignes à afficher
# max_rows=10 : les 5 premières lignes et les 5 dernières lignes
pandas.options.display.max_rows=10
# Vérifier la version de pandas
print(pandas.__version__)
df = pandas.read_table("document.txt",sep='\t',header=0)
print(type(df))
print(df)
# dropna ayant l'attribut axis=1 ou axis='columns' : Suppression des
colonnes
# Supprimer les colonnes ayant au moins une valeur vide
#df1 = df.dropna(axis='columns')
#print(df1)
# Efface les colonnes vides = toutes les valeurs sont vides :
how='all'
#df2= df.dropna(axis=1,how='all')
#print(df2)
# inplace=True : changement effectué sur le dataframe en question
# df.dropna(axis=1,inplace=True,how='all')
df= df.dropna(axis=1,how='all') # suppression des colonnes vides
print(df)
# afficher les différentes variables utilisées dans la colonne
cLT2FREQ sans doublons
print(df.cLT2FREQ.unique())
# Supprimer les lignes dont la colonne "cLT2FREQ" est vide ou =0
# df.dropna(axis=0, subset=["cLT2FREQ"], inplace=True)
# axis = 0 : pour les lignes (par défaut) / axis=1 pour les colonnes
df.dropna(subset=["cLT2FREQ"], inplace=True)
#df=df.dropna(axis=0, subset=["cLT2FREQ"])
print(df)
# Sauvegarder la matrice de donnée df dans un fichier excel
df.to_excel("CleanData.xlsx", index=False)
# Sauvegarder la matrice de donnée df dans un fichier txt
df.to_csv("DocumentDF.txt",sep="\t",encoding="utf-8", index=False)




#ex3
import pandas as pd
data = {"mat": [9.5,15,12,16,14], "inf": [14,13,11,17,7],
"phy": [7.5,14,13,12,8.5]};
stat = pd.DataFrame(data);
print(stat);
print("\n Les moyennes des étudiants sont: axix=1")
# moyenne des colonnes de chaque ligne
print(stat.mean(axis = 1))
print("\n Les moyennes des matière: axix=0")
# pour calculer les moyennes des matières
# moyenne des lignes pour chaque colonne
print(stat.mean())
print("\n Vecteur des étendues arithmétiques :")
# étendue arithmétique de chaque matière
print(stat.max() - stat.min())
print("\n Vecteur ligne des médianes arithmétiques :")
print(stat.median())
# Centrer et réduire les données du dataframe
# Soustraire la moyenne de chaque colonne à chaque valeur : centrer
les valeurs
print("\n Center les valeur:")
Y = stat.sub(stat.mean())
# Y=stat-(stat.mean()))
print(Y)

# Diviser (.div) les valeurs du dataframe Y déjà centré par l'écart-
type (.std) de chaque colonne

Z = Y.div(Y.std())
print("\n Réduire les valeur:")
print(Z)


#Réaliser une matrice de corrélation pour la dataframe
print("\n Matrice de corrélation:")
RX=stat.corr()
print(RX)
# la matrice VZ des variances et covariances de Z
# Z : la matrice déjà centrée et réduite
print("\n Matrice des variances:")
VZ=Z.var() #matrice VZ des variances
print(VZ)
print("\n Matrice des covariances:")
VZ=Z.cov() #matrice VZ des covariances de Z
print(VZ)