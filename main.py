import pandas as pd
import numpy as np
import sklearn as sk
from  sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
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

##Dans evaluation_pred.py on a déterminer quelle modele de pred est le plus précis
X_train, X_test,Y_train,Y_test= sk.model_selection.train_test_split(X,Y, train_size=0.80)

##Sans reduction de dimension
clf = OneVsRestClassifier(SVC(kernel='poly')).fit(X_train, Y_train)
val_acc=sk.metrics.accuracy_score(Y_test, clf.predict(X_test))
print('score: %0.3f' % val_acc)


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

"""
##Avec reduction de dimension non fonctionnel

#X_train, X_test,y_train,y_test= sk.model_selection.train_test_split(X,Y, train_size=0.70)
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#Ca depend de la taille des données, je crois
tocsv_data=pd.DataFrame({"X":[],"Methods":[],"Precision":[]}) #initialisation a vide pour mettre vers un fichier .csv0w pour les résultats 
start_time = time.time() #Temps de début pour obtenir tous les résultats
for x in [1,np.shape(X)[1]]:
    embedding = MDS(n_components=x)
    pca = PCA(n_components=x)

    #https://www.researchgate.net/publication/323562545_Dimensionality_reduction_methods_The_comparison_of_speed_and_accuracy
    #MDS est plus précis d'apres le doc
    print("Avec le nombre de features=",x)
    X_train_transformed_MDS = embedding.fit_transform(X_train)
    X_train_transformed_PCA = pca.fit_transform(X_train)
    X_test_transformed_PCA = pca.transform(X_test)

    print("Pour MDS:")
    val_acc=sk.model_selection.cross_val_score(clf, X_train_transformed_MDS, Y_train).mean() ##Tu peux pas faire fit avec MDS
    print('score: %0.3f' %val_acc)
    tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"x":[x],"Methods":["MDS"],"precision":[val_acc]})])

    print("Pour PCA:")
    clf = OneVsRestClassifier(SVC(kernel='poly')).fit(X_train_transformed_PCA, Y_train)
    val_acc=sk.metrics.accuracy_score(Y_test, clf.predict(X_test_transformed_PCA))
    print('score: %0.3f' %val_acc)
    tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"x":[x],"Methods":["PCA"],"precision":[val_acc]})])
    val_acc=sk.model_selection.cross_val_score(clf, X_train_transformed_PCA, Y_train).mean()
    print('score: %0.3f' %val_acc)
    tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"x":[x],"Methods":["PCA"],"precision":[val_acc]})])

    ##Pour évaluer la précision de la réduction, on utilise SVM (Support Vector Machine) sur Iris la précision la plus élevé, avec(kernel="poly")
    #https://www.geeksforgeeks.org/classifier-comparison-in-scikit-learn/

print(time.time()-start_time) #Temps pour obtenir résultat
tocsv_data.to_csv("Reduction_Result.csv", sep='\t', encoding='utf-8',index=False)
"""