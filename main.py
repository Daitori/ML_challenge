import pandas as pd
import numpy as np
import sklearn as sk
from  sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import time


data=pd.read_csv(r"Challenge\training_data.csv",sep=",") ##Change localisation du .csv si c'est pas pareil

##J'avais des resultats de pred faible - de 0.5, j'ai tenté de faire similaire
#https://stackoverflow.com/questions/39167586/scikit-very-low-accuracy-on-classifiersnaive-bayes-decissiontreeclassifier
#Pas le meme probleme
Y=data.iloc[:,1].astype(int) 
##Val [1 2 3 5 6 7 8] avec 1286 valeurs totals
#[1:173;2:171;3:194;5:198;6:181;7:189;8:180] +- équilibré la répartion des classes

##Plus précis de normaliser les X et les Y à part
Coordinate_X=sk.preprocessing.normalize(data.iloc[:,4:72,])
Coordinate_Y=sk.preprocessing.normalize(data.iloc[:,72:140,])
X=(np.concatenate((Coordinate_X,Coordinate_Y),axis=-1))

##Sans reduction de dimension
clf = OneVsRestClassifier(SVC(kernel='poly'))
print('score: %0.3f' % cross_val_score(clf, X, Y).mean())

"""#Inutile: mais voila ca fait de quoi écrire pour dire qu'on c trompé initialement
#C'est pas possible d'avoir des features multi dimentionnel c comme ca, il faut des strings ou des nombres, et la dim doit etre de 2
Coordinate_X=data[:,4:72,]
print(Coordinate_X.shape)
Coordinate_X_Normalized=preprocessing.normalize(Coordinate_X)
Coordinate_Y=data[:,72:140,]
Coordinate_Y_Normalized=preprocessing.normalize(Coordinate_Y)
X = []
for i in range(len(Coordinate_X_Normalized)):
    L = []
    for j in range(len(Coordinate_X_Normalized[i])):
        L.append([Coordinate_X_Normalized[i][j], Coordinate_Y_Normalized[i][j]])
    X.append(L)
#Taille obtenus (1286,68,2) comme on a 68 coordonnées avec X Y, logique mais c pour les humains pas pour les algos
#Il faut donc (1286,136) (X,Y séparé c des features "indépendentes")
"""


##Avec reduction de dimension non fonctionnel

#X_train, X_test,y_train,y_test= sk.model_selection.train_test_split(X,Y, train_size=0.70)
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#Ca depend de la taille des données, je crois
tocsv_data=pd.DataFrame({"X":[],"Methods":[],"Precision":[]}) #initialisation a vide pour mettre vers un fichier .csv0w pour les résultats 
start_time = time.time() #Temps de début pour obtenir tous les résultats
for x in range(1,np.shape(X)[1]): 
    print("Avec le nombre de features=",x)
    embedding = MDS(n_components=x)
    X_transformed_MDS = embedding.fit_transform(X)
    #https://www.researchgate.net/publication/323562545_Dimensionality_reduction_methods_The_comparison_of_speed_and_accuracy
    print("Pour MDS:")
    val_acc=sk.model_selection.cross_val_score(clf, X_transformed_MDS, Y).mean()
    print('score: %0.3f' %val_acc)
    tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"X":[x],"Methods":["MDS"],"precision":[val_acc]})])
    for y in ["auto","full","arpack","randomized"]: #svd_solver{‘auto’, ‘full’, ‘arpack’, ‘randomized’}, default=’auto’
        pca = PCA(n_components=x,svd_solver=y)
        #MDS est plus précis d'apres le doc
        X_transformed_PCA=pca.fit_transform(X)
        print("Pour PCA:"+y)
        val_acc=sk.model_selection.cross_val_score(clf, X_transformed_PCA, Y).mean()
        print('score: %0.3f' %val_acc)
        tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"X":[x],"Methods":["PCA "+y],"Precision":[val_acc]})])
        ##Pour évaluer la précision de la réduction, on utilise SVM (Support Vector Machine) sur Iris la précision la plus élevé, avec(kernel="poly")
        #https://www.geeksforgeeks.org/classifier-comparison-in-scikit-learn/

print(time.time()-start_time) #Temps pour obtenir résultat
tocsv_data.to_csv("Reduction_Result.csv", sep='\t', encoding='utf-8',index=False)
