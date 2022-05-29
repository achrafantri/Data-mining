# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13  19:18:50 2022

@author:  Achraf Antri
"""
 #modification du dossier de travail
import os

os.chdir(r"/portfolio/afd")

# librairie pandas
import pandas
# version
print(pandas.__version__)

# chargement de la première feuille de données
X = pandas.read_excel(r"seeds.xlsx", sheet_name=0, header=0)
# Le fichier est un classeur Excel nommé « seeds.xlsx »
# Les données actives sont situées dans la première feuille (sheet_name = 0
# La première ligne correspond aux noms des variables (header = 0)
# La première colonne aux identifiants des observations (index_col = 0)

#Afficher le DataFrame
pandas.options.display.max_rows=10
#data=(X.iloc[:,1:])
print(X)
#from IPython.display import display
#display(X)

# dimension
print(X.shape)

# nombre d'observations
n = X.shape[0]
print(n)

# nombre de variables
p = X.shape[1]
print(p)


# 2.Sélectionner les colonnes nécessaires dans l’objectif d’analyser les dépendances
# entre les grains et la recherche des similarités. (Modification sur X)
X=X.iloc[:,1:]
print(X)




#Remplacer les cellules Nan par moyennes
# des valeurs manquantes?
print("\nExistance des valeurs manquantes")
print(X.isnull().values.any())
# Nombre total de valeurs manquantes par colonne
print("\nValeurs manquantes par colonne")
print(X.isnull().sum())
print("\nTotal des valeurs manquantes")
# Nombre total de valeurs manquantes
print(X.isnull().sum().sum())
# Il est fréquent de remplacer les valeurs inconnues par la moyenne de la variable
X = X.fillna(X.mean())


# Nous devons explicitement centrer et réduire les variables pour réaliser une ACP normée avec PCA.
# Nous utilisons la classe StandardScaler pour ce faire.


# 4. Standardiser les données en utilisant la classe **StandardScaler** de la bibliothèque **sklearn**.

# Nous instancions l’objet StandardScaler et nous l’appliquons sur la matrice X.
# Nous obtenons une matrice Z
# classe pour standardisation
fromsklearn.preprocessingimport StandardScaler
# instanciation
sc = StandardScaler()
# transformation
Z = sc.fit_transform(X)
print(X)

"""
# ou bien
# Importation de la bibliothèque
fromsklearn.preprocessing import StandardScaler

# Instantiation d'un objet StandardScaler
sc = StandardScaler()

# Calcul des paramètres du modèle sur nos données
model= sc.fit(X)

# Transformation des données selon le modèle construit
Z = model.transform(X)
print(Z)
"""

#Afficher la centrée réduite Z de StandardScaler() sous forme de matrice
columns=["Area","Perimetre","compactness","kLength","kWidth","AsymmetryCoef","kGrooveL"]
Z_df=pandas.DataFrame(Z,columns=columns)
print(Z_df)



# Vérifions les propriétés du nouvel ensemble de données : les moyennes et les écarts-type unitaires.
# Pour une matrice de données centrée et réduite, la moyenne de chaque colonne=0 et l'écarts_type de chaque colonne = 1
# vérification - librairie numpy

import numpy

# moyenne
print("\nmoyenne\n")
print(numpy.mean(Z, axis=0))
# Les moyennes sont maintenant nulles (aux erreurs de troncature près)

# moyenne
numpy.set_printoptions(suppress=True)
print("\nmoyenne")
print(numpy.mean(Z, axis=0))

# ecart-type
print("\necart-type")
print(numpy.std(Z, axis=0))



#Afficher et analyser la matrice de corrélation.


#avec Numpy
#display(pandas.DataFrame(np.corrcoef(Z,rowvar=False)))
#avec pandas
Corr=X.corr()
print(Corr)


#Affichage graphique avancé de la matrice de corrélation#heatmap pour identifier visuellement les corrélations fortes

#librairie graphique
import seabornas sns
#https://seaborn.pydata.org/
sns.heatmap(Corr,xticklabels=Z_df.columns,yticklabels=Z_df.columns,vmin=-1,vmax=+1,center=0,cmap="RdBu",linewidths=0.5, annot=True)
#https://seaborn.pydata.org/generated/seaborn.heatmap.html
#https://matplotlib.org/stable/tutorials/colors/colormaps.html


# makeLowertriangular values Null
print(numpy.triu(numpy.ones(Corr.shape), k=1).astype(bool))
#Corr.where(cond) remplace des valeurs dans Corr lorsque la condition est false
upper_corr_mat = Corr.where(numpy.triu(numpy.ones(Corr.shape), k=1).astype(bool))
print(upper_corr_mat)
# Convert to 1-D series and drop Null values
unique_corr_pairs = upper_corr_mat.unstack().dropna()
print(unique_corr_pairs)

# Sort correlation pairs
sorted_mat = unique_corr_pairs.sort_values(ascending=False)
print(sorted_mat)

#pairplot
sns.pairplot(Z_df,corner=True,diag_kind='hist',vars=["Area",'Perimetre'], )
sns.pairplot(Z_df,corner=True,diag_kind='hist',vars=["compactness",'AsymmetryCoef'], )


#Pour plus de détails
#https://seaborn.pydata.org/generated/seaborn.pairplot.html
#https://www.delftstack.com/fr/howto/seaborn/seaborn-pairplot-python/
#couleur de palette
#http://seaborn.pydata.org/tutorial/color_palettes.html
#https://openclassrooms.com/fr/courses/4452741-decouvrez-les-librairies-python-pour-la-data-science/5559011-realisez-de-beaux-graphiques-avec-seaborn



#Réaliser sur Z une ACP normée en utilisant la méthode pca du module sklearn.decomposition


# Importation classe PCA
fromsklearn.decompositionimport PCA

# instanciation d'un objet PCA
pca = PCA()
# Le nombre de composantes (K) n’étant pas spécifié (n_components = None),
# il est par défaut égal au nombre de variables (K = p).

# nombre de composantes calculées
print(pca.n_components)


# Calculer les paramètres du modèle
#model=pca.fit(Z)

#print(model.explained_variance_)

# Transformation de Z (centrée réduite) en coordonnées factorielles (Y) selon le modèle PCA
# Y matrice des scores

#Y = model.transform(Z)

#ou directement
Y = pca.fit_transform(Z)
# fit_transform() renvoie en sortie les matrice des scores ( des coordonnées factorielles) contenantles nouvelles données utilisant les
# colonnes : composantes principales

print("\nMatrice des Scores")
print(Y)


#Afficher Y
columns = ['pca_%i' % i for i in range(7)]
print(columns)
Y_df=pandas.DataFrame(Y,columns=columns)
print(Y_df)



numpy.set_printoptions(suppress=True)
print(Y_df.corr())
print(numpy.corrcoef(Y_df,rowvar=False))




pca1=PCA()
Y1=pca1.fit_transform(Y)
print(pandas.DataFrame(Y1,columns=Y_df.columns))


# La propriété .explained_variance_ semble faire l’affaire
# pour obtenir les variances (valeurs propres, λk) associées aux composantes principales.
# variance expliquée / inertie expliquée
print("variance expliquée / inertie expliquée")
#print(pca.explained_variance_)
val_propres= pca.explained_variance_
print(val_propres)



# Les valeurs obtenues ne sont pas exactes.
# Règle: Somme des variances(Valeurs propres) = nombre de variables
print(numpy.sum(val_propres))

# Il faut alors appliquer une correction.
# valeur corrigée
print("valeurs corrigées")
val_propres = (n -1) / n * pca.explained_variance_
print(val_propres)


# Nous aurions pu obtenir les bonnes valeurs propres en passant par les valeurs singulières
# .singular_values_ issues de la factorisation de la matrice des données centrées et réduites
# ou bien en passant par les valeurs singulières
print("valeurs propres passant par les valeurs singulières")
val_propres = pca.singular_values_ ** 2 / n
print(val_propres)

import matplotlib.pyplotas plt

#plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
plt.plot(range(pca.n_components_), val_propres )
plt.title("variance expliquée /CP ")
plt.xlabel('Composantes principales')
plt.ylabel('Valeur de variance expliquée')
plt.xticks(range(pca.n_components_))
plt.show()





plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.title("Variance expliquée / CP")
plt.xlabel('Composantes principales')
plt.ylabel('Variance expliquée')
#plt.xticks(range(pca.n_components_))
plt.show()






#plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
plt.plot(range(pca.n_components_), pca.explained_variance_ratio_ *100 )
plt.title("% de variance expliquée /CP ")
plt.xlabel('Composantes principales')
plt.ylabel('Pourcentage de variance expliquée')
plt.xticks(range(pca.n_components_))
plt.show()



# proportion de variance expliquée
print("proportion de variance expliquée")
print(pca.explained_variance_ratio_ ) # Il n’est pas nécessaire d’effectuer une correction dans ce cas


exp_var_ratio = pca.explained_variance_ratio_
print(exp_var_ratio*100)

#Somme cumulative des valeurs propres
cum_sum_eigenvalues = numpy.cumsum(exp_var_ratio)
print(cum_sum_eigenvalues)


#Préparation du plot pour l'affichage

plt.plot(range(pca.n_components_),cum_sum_eigenvalues,'o-', linewidth=2, color='blue')
plt.grid(which='both', linestyle='--')
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()



#positionnement des individus dans le premier plan
fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(-6,6) #même limites en abscisse
axes.set_ylim(-6,6) #et en ordonnée
#placement des étiquettes des observations
for i in range(n):
plt.annotate(X.index[i],(Y_df.iloc[i,0],Y_df.iloc[i,1]))
#ajouter les axes
plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)
#affichage
plt.scatter(Y_df.pca_0, Y_df.pca_1, s=50)
plt.show()




# 20.  Calculer **la matrice de corrélation des anciennes variables (Zj) et des nouvelles (Yk) du plan factoriel**.

Q=pca.components_

#𝑟(𝑍𝑗, 𝑌𝑘) = racine_carrée(𝜆𝑘)*𝑞𝑗k
p=pca.n_components_
corvar = numpy.zeros((p,p))
for k in range (7):
corvar[:,k] = Q[k,:] * numpy.sqrt(val_propres [k]) #remplit par colonne
print(corvar)

CorrVariables=pandas.DataFrame(corvar,index=Z_df.columns,columns=columns)
print(CorrVariables)


#on affiche pour les deux premiers axes
print(pandas.DataFrame({'id':Z_df.columns,'PC1':corvar[:,0],'PC2':corvar[:,1]}))


# 21.  Analyser la saturation des variables en projetant les variables (Zj) sur le cercle de corrélation C


#cercle des corrélations
fig, axes = plt.subplots(figsize=(4,4))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
#affichage des étiquettes (noms des variables)
for j in range(Z_df.shape[1]):
plt.annotate(Z_df.columns[j],(corvar[j,0],corvar[j,1]))
plt.quiver(0, 0, corvar[j,0],corvar[j,1], angles = 'xy', scale_units= 'xy', scale= 1)  # Tracé d'un vecteur
#plt.scatter(T.pca_0, T.pca_1, s=50, c=colormap[classe-1])

#ajouter un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
#ajouter les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
plt.xlabel('CP1 (71.87%)')
plt.ylabel('CP2 (17,11%)')
#for i in range(210):
#    plt.annotate(X.index[i],(Y_df.iloc[i,0],Y_df.iloc[i,1]))
#affichage
plt.show()
