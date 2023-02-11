import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.svm import SVC
from  sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

data=pd.read_csv(r"Challenge\training_data.csv",sep=",") ##Change localisation du .csv si c'est pas pareil

##J'avais des resultats de pred faible - de 0.5, j'ai tenté de faire similaire
#https://stackoverflow.com/questions/39167586/scikit-very-low-accuracy-on-classifiersnaive-bayes-decissiontreeclassifier

Y=data.iloc[:,1].astype(int) ##Val de 1 à 8 avec 1286 valeurs

##Plus précis de normaliser les X et les Y à part
Coordinate_X=sk.preprocessing.normalize(data.iloc[:,4:72,])
Coordinate_Y=sk.preprocessing.normalize(data.iloc[:,72:140,])
X=(np.concatenate((Coordinate_X,Coordinate_Y),axis=-1))

##Evaluation des différents models avant features selections
X_train, X_test,y_train,y_test= sk.model_selection.train_test_split(X,Y, train_size=0.80)

#initialisation a vide pour mettre vers un fichier .csv0w pour les résultats 
tocsv_data=pd.DataFrame({"Models":[],"precision":[]}) 


clf = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) ##Si les resultats ne sont pas satisfaisante on modifie
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
cm_acc = confusion_matrix(y_test, preds)
print(cm_acc)
val_acc = accuracy_score(y_test, preds)
print('score: %0.3f' % val_acc)
tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"Models":["KNeighborsClassifier"],"precision":[val_acc]})])

clf = SVC(kernel = 'linear', random_state = 0) #SVC linear
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
cm_acc = confusion_matrix(y_test, preds)
print(cm_acc)
val_acc = accuracy_score(y_test, preds)
print(' score: %0.3f' % val_acc)
tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"Models":["SVC linear"],"precision":[val_acc]})])

clf = SVC(random_state = 0)  #SVC non linear
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
cm_acc = confusion_matrix(y_test, preds)
print(cm_acc)
val_acc = sk.metrics.accuracy_score(y_test, preds)
print(' score: %0.3f' % val_acc)
tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"Models":["SVC no linear"],"precision":[val_acc]})])

clf = SVC(kernel = 'rbf', random_state = 0) # Kernel SVM rbf
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
cm_acc = confusion_matrix(y_test, preds)
print(cm_acc)
val_acc = sk.metrics.accuracy_score(y_test, preds)
print(' score: %0.3f' % val_acc)
tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"Models":["Kernel SVM rbf"],"precision":[val_acc]})])

clf = GaussianNB()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
cm_acc = confusion_matrix(y_test, preds)
print(cm_acc)
val_acc = sk.metrics.accuracy_score(y_test, preds)
print(' score: %0.3f' % val_acc)
tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"Models":["GaussianNB"],"precision":[val_acc]})])


clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
cm_acc = confusion_matrix(y_test, preds)
print(cm_acc)
val_acc = sk.metrics.accuracy_score(y_test, preds)
print(' score: %0.3f' % val_acc)
tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"Models":["Decision Tree entropy"],"precision":[val_acc]})])


clf =RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
cm_acc = confusion_matrix(y_test, preds)
print(cm_acc)
val_acc = sk.metrics.accuracy_score(y_test, preds)
print(' score: %0.3f' % val_acc)
tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"Models":["Random Forest"],"precision":[val_acc]})])

clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
cm_acc = confusion_matrix(y_test, preds)
print(cm_acc)
val_acc = sk.metrics.accuracy_score(y_test, preds)
print(' score: %0.3f' % val_acc)
tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"Models":["AdaBoost Classifier"],"precision":[val_acc]})])


clf = QuadraticDiscriminantAnalysis()
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
cm_acc = confusion_matrix(y_test, preds)
print(cm_acc)
val_acc = sk.metrics.accuracy_score(y_test, preds)
print(' score: %0.3f' % val_acc)
tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"Models":["Quadratic Discriminant Analysis"],"precision":[val_acc]})])

clf = MLPClassifier(alpha=1, max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
cm_acc = confusion_matrix(y_test, preds)
print(cm_acc)
val_acc = sk.metrics.accuracy_score(y_test, preds)
print(' score: %0.3f' % val_acc)
tocsv_data=pd.concat([tocsv_data,pd.DataFrame({"Models":["MLP Classifier"],"precision":[val_acc]})])

tocsv_data.to_csv("Test_Result.csv", sep='\t', encoding='utf-8',index=False)

