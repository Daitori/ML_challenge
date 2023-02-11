import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.svm import SVC
from  sklearn.manifold import MDS
import time


data=pd.read_csv(r"Challenge\training_data.csv",sep=",") ##Change localisation du .csv si c'est pas pareil

##J'avais des resultats de pred faible - de 0.5, j'ai tenté de faire similaire
#https://stackoverflow.com/questions/39167586/scikit-very-low-accuracy-on-classifiersnaive-bayes-decissiontreeclassifier

Y=data.iloc[:,1].astype(int) ##Val de 1 à 8 avec 1286 valeurs

##Plus précis de normaliser les X et les Y à part
Coordinate_X=sk.preprocessing.normalize(data.iloc[:,4:72,])
Coordinate_Y=sk.preprocessing.normalize(data.iloc[:,72:140,])
X=(np.concatenate((Coordinate_X,Coordinate_Y),axis=-1))

##Dans evaluation_pred.py on a déterminer quelle modele de pred est le plus précis
X_train, X_test,y_train,y_test= sk.model_selection.train_test_split(X,Y, train_size=0.80)

clf = SVC()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
val_acc = sk.metrics.accuracy_score(y_test, preds)
print(' score: %0.3f' % val_acc)


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
#X_train, X_test,y_train,y_test= sk.model_selection.train_test_split(X,Y, train_size=0.70)
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
#Si j'ai bien compris, ca randomise pas la selection comme test_train_split
tocsv_data=pd.DataFrame({"x":[],"precision":[]}) #initialisation a vide pour mettre vers un fichier .csv0w pour les résultats 
start_time = time.time() #Temps de début pour obtenir tous les résultats
X_train, X_test,y_train,y_test= sk.model_selection.train_test_split(X,Y, train_size=0.80)
clf = SVC()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
val_acc = sk.metrics.accuracy_score(y_test, preds)

print(np.shape(X))
print('SVM score: %0.3f' % val_acc)

for x in range(1,np.shape(X)[1]+1):
    embedding = MDS(n_components=x)
    #https://www.researchgate.net/publication/323562545_Dimensionality_reduction_methods_The_comparison_of_speed_and_accuracy
    #MDS est plus précis
    X_transformed = embedding.fit_transform(X)
    print("Avec le nombre de composants=",x)
    ##Pour évaluer la précision de la réduction, on utilise SVM (Support Vector Machine) sur Iris la précision la plus élevé
    #https://www.geeksforgeeks.org/classifier-comparison-in-scikit-learn/
    svm = SVC()
    svm_scores = sk.model_selection.cross_val_score(svm, X_transformed,Y)
    print(np.shape(X_transformed))
    print('SVM score: %0.3f' % svm_scores.mean())
    tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"x":[x],"precision":[svm_scores.mean()]})])
print(time.time()-start_time) #Temps pour obtenir résultat
tocsv_data.to_csv("Test_Result.csv", sep='\t', encoding='utf-8',index=False)
"""