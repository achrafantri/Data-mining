# -*- coding: utf-8 -*-
"""
Created on Tue Avr 16 22:19:57 2022

@author:  Achraf Antri
"""
import os
os.chdir("/portfolio/afd")

import pandas
X =pandas.read_table("fromage.txt",sep="\t",header=0,index_col=0)
print(X)


importsklearn
fromsklearn.preprocessingimportStandardScaler
sc=StandardScaler()
Z =sc.fit_transform(X)
print(Z)


fromsklearn.decompositionimportPCA
pca=PCA(n_components=2)
Y =pca.fit_transform(Z)
print(Y)

#afficher la matrice des scores sous forme de dataframe
Y_df=pandas.DataFrame(Y,columns=['CP1','CP2'],index=X.index)
print(Y_df)


#Appliquer sur Y la classe KMeans de la bibliothèque sklearn pour regrouper les individus en 4 clusters

fromsklearn.clusterimportKMeans
#from sklearn import cluster.KMeans
kmeans=KMeans(n_clusters=4).fit(Y)

#kmeans= KMeans(n_clusters=4)
#kmeans.fit(Y)
print(kmeans)


importmatplotlib.pyplotasplt

  
#Afficher les coordonnées de chaque centroïde et l’inertie associée.
centroids=kmeans.cluster_centers_
print(centroids)

plt.scatter(Y[:,0], Y[:,1], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0],centroids[:, 1], c='red', s=50)
plt.show()



# Afficher les individus et leurs groupes.
importnumpyasnp
idk=np.argsort(kmeans.labels_)
df_kmeans=pandas.DataFrame(X.index[idk],kmeans.labels_[idk])
print(df_kmeans)
#print(pandas.DataFrame(X.index[idk],kmeans.labels_[idk].count()))


print(kmeans.labels_)
print(kmeans.inertia_)


# Afficher les distances des individus aux centres des clusters
#distances aux centres de classes des observations
print(kmeans.transform(Y))

#Méthode de silhouette
fromsklearnimportmetrics
l2=list()
for i innp.arange(2,14):
    y_pred=KMeans(n_clusters=i,init='k-means++',n_init=10).fit_predict(Y_df)
    l2.append(metrics.silhouette_score(Y_df,y_pred,metric='euclidean'))
    print('La silhouette index pour {0:d} classes est {1:3f}'.format(i,metrics.silhouette_score(Y_df,y_pred,metric='euclidean')))

importmatplotlib.pyplotasplt
plt.title("Silhouette")
plt.xlabel("# of clusters")
plt.plot(np.arange(2,14,1),l2)
plt.show()


frommatplotlibimportpyplotasplt
fromscipy.cluster.hierarchyimportdendrogram, linkage

Y_CAH=linkage(Y,method='ward',metric='euclidean')
#method='ward : L'algorithme de liaison à utiliser
#metric='euclidean' : La métrique de distance à utiliser.

plt.title("CAH")
dendrogram(Y_CAH,labels=X.index,orientation='left',color_threshold=0)
plt.show()

#Faire une matérialisation en 4 classes et un découpage à une hauteur qui est égale à 5
fromscipy.cluster.hierarchyimportfcluster

plt.title('CAH avec matérialisation des classes')
dendrogram(Y_CAH,labels=X.index,orientation='top',color_threshold=5)
plt.show()

#Afficher les correspondances avec les groupes de la CAH
groupes_cah=fcluster(Y_CAH,t=5,criterion='distance') -1
print(groupes_cah)

importnumpyasnp
idg=np.argsort(groupes_cah)

print(pandas.DataFrame(X.index[idg],groupes_cah[idg]))

pandas.crosstab(groupes_cah,kmeans.labels_)

