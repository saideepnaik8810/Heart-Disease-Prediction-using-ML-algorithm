import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

heart_dataset=pd.read_csv('./datasets/heart.csv')

y=heart_dataset['target']
x=heart_dataset.drop(['target'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

knnclassifier=KNeighborsClassifier(n_neighbors=1)
knnclassifier.fit(x_train,y_train)
pickle.dump(knnclassifier,open('Knnmodel.pkl','wb'))

y_knn_pred=knnclassifier.predict(x_test)

# acc=accuracy_score(y_knn_pred,y_test)
# print("Accuracy is "+"\033[1m {:.2f}%" .format(acc*100))



# error = []
# # Calculating error for K values between 1 and 30
# for i in range(1, 30):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(x_train, y_train)
#     pred_i = knn.predict(x_test)
#     error.append(np.mean(pred_i != y_test))
# print("Minimum error:-",min(error),"at K =",error.index(min(error))+1)
 

