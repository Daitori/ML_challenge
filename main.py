import pandas as pd
import numpy as np
import sklearn as sk
from  sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import time


<<<<<<< HEAD
data=pd.read_csv(r"training_data.csv",sep=",") ##Change localisation du .csv si c'est pas pareil
=======
data=pd.read_csv(r"Challenge\training_data.csv",sep=",") ##Change localisation du .csv si c'est pas pareil

>>>>>>> 858741b5249f3b725ca94ea5c73e079769a26dfe
Y=data.iloc[:,1].astype(int) 
##Val [1 2 3 5 6 7 8] avec 1286 valeurs totals
#[1:173;2:171;3:194;5:198;6:181;7:189;8:180] +- équilibré la répartion des classes
##Plus précis de normaliser les X et les Y à part (0,01 de plus)
Coordinate_X=sk.preprocessing.normalize(data.iloc[:,4:72,]) #Normalisation des données selon X
Coordinate_Y=sk.preprocessing.normalize(data.iloc[:,72:140,]) #Normalisation des données selon Y


##Sans reduction de dimension

X=pd.DataFrame(sk.preprocessing.normalize(data.iloc[:,4:140,])) #Normalisation des données selon X et Y

clf = OneVsRestClassifier(SVC(kernel='poly'))
print('score: %0.3f' % cross_val_score(clf, X, Y).mean())

X_train, X_test,y_train,y_test= sk.model_selection.train_test_split(X,Y, train_size=0.70,random_state=0)
clf.fit(X_train,y_train)
print("Score sur le test set: %0.3f" % clf.score(X_test,y_test))
#score: 0.761
#Score sur le test set: 0.718

X=pd.DataFrame(np.concatenate((Coordinate_X,Coordinate_Y),axis=-1)) #Concaténation des données normalisées selon X et Y séparément

#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
clf = OneVsRestClassifier(SVC(kernel='poly'))
print('score: %0.3f' % cross_val_score(clf, X, Y).mean())

X_train, X_test,y_train,y_test= sk.model_selection.train_test_split(X,Y, train_size=0.70,random_state=0)
clf.fit(X_train,y_train)
print("Score sur le test set: %0.3f" % clf.score(X_test,y_test))

#score: 0.754
#Score sur le test set: 0.746

csv_predict=pd.DataFrame({"label":[]}) #initialisation a vide pour mettre vers un fichier .csv pour les résultats 
data_predict=pd.read_csv(r"data.csv",sep=",")
X=pd.DataFrame(np.concatenate((sk.preprocessing.normalize(data_predict.loc[:,'x_0':'x_67']),sk.preprocessing.normalize(data_predict.loc[:,'y_0':'y_67'])),axis=-1))
Y=pd.DataFrame({"label":clf.predict(X).astype(int)})
csv_predict=pd.concat([csv_predict,Y])
csv_predict.to_csv("result.csv",index=False)

"""
##Avec reduction de dimension non fonctionnel

#tocsv_data_pca=pd.DataFrame({"X":[],"Methods":[],"Precision":[]}) #initialisation a vide pour mettre vers un fichier .csv pour les résultats 
tocsv_data_mds=pd.DataFrame({"X":[],"Methods":[],"Precision":[]}) #initialisation a vide pour mettre vers un fichier .csv pour les résultats 
##start_time = time.time() #Temps de début pour obtenir tous les résultats
X_train, X_test,y_train,y_test= sk.model_selection.train_test_split(X,Y, train_size=0.70)
clf = OneVsRestClassifier(SVC(kernel='poly'))

for x in range(1,50): 
    print("Avec le nombre de features=",x)
    embedding = MDS(n_components=x)
    X_train_transformed_MDS = embedding.fit_transform(X_train)
    X_test_transformed_MDS = embedding.fit_transform(X_test)
    clf.fit(X_train_transformed_MDS,y_train)
    #https://www.researchgate.net/publication/323562545_Dimensionality_reduction_methods_The_comparison_of_speed_and_accuracy
    print("Pour MDS:")
    val_acc=clf.score(X_test_transformed_MDS,y_test)
    print('score: %0.3f' %val_acc)
    tocsv_data_mds=pd.concat([tocsv_data_mds,pd.DataFrame({"X":[x],"Methods":["MDS"],"Precision":[val_acc]})])
    for y in ["auto","full","arpack","randomized"]: #svd_solver{‘auto’, ‘full’, ‘arpack’, ‘randomized’}, default=’auto’
        pca = PCA(n_components=x,svd_solver=y)
        #MDS est plus précis d'apres le doc
        X_transformed_PCA=pca.fit_transform(X)
        print("Pour PCA:"+y)
        val_acc=sk.model_selection.cross_val_score(clf, X_transformed_PCA, Y).mean()
        print('score: %0.3f' %val_acc)
<<<<<<< HEAD
        tocsv_data_pca=pd.concat([tocsv_data_pca,pd.DataFrame({"X":[x],"Methods":[y],"Precision":[val_acc]})])
        ##Pour évaluer la précision de la réduction, on utilise SVM (Support Vector Machine) sur Iris la précision la plus élevé, avec(kernel="poly")
        #https://www.geeksforgeeks.org/classifier-comparison-in-scikit-learn/
    
#print(time.time()-start_time) #Temps pour obtenir résultat
#tocsv_data_pca.to_csv("Reduction_Result_PCA.csv", sep=' ', encoding='utf-8',index=False)
tocsv_data_mds.to_csv("Reduction_Result_MDS.csv", sep=' ', encoding='utf-8',index=False)
"""
=======
        tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"X":[x],"Methods":["PCA "+y],"Precision":[val_acc]})])


print(time.time()-start_time) #Temps pour obtenir résultat
tocsv_data.to_csv("Reduction_Result.csv", sep='\t', encoding='utf-8',index=False)
>>>>>>> 858741b5249f3b725ca94ea5c73e079769a26dfe
