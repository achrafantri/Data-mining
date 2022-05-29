# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 21:13:27 2022

@author:  Achraf Antri
"""

# Créer et afficher un DataFrame avec les étiquettes de l'index
import pandas as pd
import numpy as  np

data={"name": ["Anastasia","Dima","Katherine","James","Emily","Michael","Matthew","Laura","Kevin","Jonas"],
      "score":[12.5,9.0,16.5,np.nan,9.0,20.0,14.5,np.nan,8.0,19.0],
      "attempts":[1,3,2,3,2,3,1,1,2,1],
      "qualify":["yes","no","yes","no","no","yes","yes","no","no","yes"]}
q=["a","b","c","d","e","f","g","h","i","j"]
df=pd.DataFrame(data,index=q)

# Afficher les informations de base du DataFrame ainsi que ses données

print(df.info())


# Obtenir les 3 premières lignes du DataFrame donné

print(df.head(3))
#print(df.iloc[:3])


# Sélectionner les colonnes «name» et «score»


print(df[['name','score']])

# Sélectionner les colonnes «name» et «score» des lignes 1, 3, 5 et 6

print(df.iloc[[1, 3, 5, 6], [0, 1]])



# Sélectionner les lignes où le nombre de tentatives « attempts » d'examen est supérieur
#à 2


print(df[df['attempts'] > 2])



# Compter le nombre de lignes et de colonnes d'un DataFrame


print('le nombre de ligne est:',len(df.axes[0]))
print('le nombre de colonne est:',len(df.axes[1]))


# Sélectionner les lignes où le score est manquant, c'est-à-dire NaN



print(df[df['score'].isnull()])


# Sélectionner les lignes dont le score « score » est compris entre 15 et 20 (inclus)


print(df[df['score'].between(15,20)])



# Sélectionner les lignes où le nombre de tentatives à l'examen « attempts » est inférieur
#à 2 et le score « score » supérieur à 15


print(df[(df['attempts'] < 2) & (df['score'] > 15)])


# Changer le score de la ligne «d» en 11,5


df.loc['d', 'score'] = 11.5

"print(df)"
# Calculer la somme des tentatives d'examen « attempts » des élèves



print('la somme de tentative d"exam" est:',df['attempts'].sum())


# Calculer le score moyen des élèves


print('le score moyen des élèves:',df['score'].mean())


# Ajouter une nouvelle ligne «k» au DataFrame avec des valeurs données pour chaque
#colonne

df.loc['k'] = ['achraf', 16.5, 1,'yes']
print(df)


# Supprimez la nouvelle ligne et renvoyez le bloc de données d'origine

df = df.drop('k')
print(df)


# Trier le DataFrame d'abord par «name» dans l'ordre croissant


print(df.sort_values(by=['name']))


# Trier le DataFrame par «score» dans l'ordre décroissant


print(df.sort_values(by=['score'],ascending=[False]))


# Remplacer la colonne «qualify» contenant les valeurs «yes» et «no» par True et False


df['qualify'] = df['qualify'].map({'yes': True, 'no': False})
print(df)




# Changer le nom «James» en «Suresh» dans la colonne de nom du DataFrame


df['name']=df['name'].replace(['James'], 'Suresh')
print(df)

# Supprimer la colonne «attempts» du DataFrame
del df['attempts']
print(df)

# Insérer une nouvelle colonne dans le DataFrame existant


df=df.assign(age=19)
print(df)


# Parcourir les lignes d'un DataFrame

for i in df.index: 
     print( df["name"][i]+ " a le score:"+str(df["score"][i])+" et la qualification : "+str(df["qualify"][i]))

# Remplacer toutes les valeurs NaN par des Zéro

df ['score'] = df ['score']. replace (np.nan, 0)
print(df)

# Définir une valeur donnée pour une cellule particulière dans le DataFrame à l'aide de
#la valeur d'index

df.loc['a', 'score'] = '20'
df.loc['b', 'attempts'] = '20'
df.loc['c', 'qualify'] = 'yes'


print(pd.DataFrame(df))

# Compter les valeurs NaN dans une ou plusieurs colonnes dans le DataFrame

print(df.isna().sum())


# Obtenir la liste des en-têtes des colonnes du DataFrame

print(list(df.columns))

# Renommer les colonnes d'un DataFrame

df=df.rename(columns={'name': 'nom','score': 'SC','attempts':'tentative','qualify':'qualifier'})
print(df)

# Sélectionner des lignes à partir du DataFrame en fonction des valeurs de certaines
#colonnes

new_df = df.query('attempts == 3')
print(new_df)


# Changer l'ordre des colonnes du DataFrame
print(df)
print('..............................')
df = df.reindex(columns=['score','attempts','name','qualify'])
print(df)