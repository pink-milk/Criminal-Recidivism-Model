import numpy as np
import math
from sklearn import preprocessing

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import pandas as pd


#normalize data in pd df
data = pd.read_csv('reoffense_train.csv')

test_data = pd.read_csv('reoffense_test.csv')

# norm_data=preprocessing.normalize(data)
# norm_data = np.delete(norm_data, -1, axis=1)

data['sex']=data['sex'].replace('Female',0)
data['sex']=data['sex'].replace('Male',1)

data['age_cat']=data['age_cat'].replace('Less than 25', 0)
data['age_cat']=data['age_cat'].replace('25 - 45', 1)
data['age_cat']=data['age_cat'].replace('Greater than 45', 2)

data['race']=data['race'].replace('Caucasian', 0)
data['race']=data['race'].replace('African-American', 1)
data['race']=data['race'].replace('Hispanic', 2)
data['race']=data['race'].replace('Native American', 3)
data['race']=data['race'].replace('Asian', 4)
data['race']=data['race'].replace('Other', 5)

data['c_charge_degree']=data['c_charge_degree'].replace('M', 0)
data['c_charge_degree']=data['c_charge_degree'].replace('F', 1)

label=data.iloc[:,-1]
race=data.iloc[:,3]

# print(race)

data=data.drop(columns=['c_charge_desc', 'label'])



test_data['sex']=test_data['sex'].replace('Female',0)
test_data['sex']=test_data['sex'].replace('Male',1)

test_data['age_cat']=test_data['age_cat'].replace('Less than 25', 0)
test_data['age_cat']=test_data['age_cat'].replace('25 - 45', 1)
test_data['age_cat']=test_data['age_cat'].replace('Greater than 45', 2)

test_data['race']=test_data['race'].replace('Caucasian', 0)
test_data['race']=test_data['race'].replace('African-American', 1)
test_data['race']=test_data['race'].replace('Hispanic', 2)
test_data['race']=test_data['race'].replace('Native American', 3)
test_data['race']=test_data['race'].replace('Asian', 4)
test_data['race']=test_data['race'].replace('Other', 5)

test_data['c_charge_degree']=test_data['c_charge_degree'].replace('M', 0)
test_data['c_charge_degree']=test_data['c_charge_degree'].replace('F', 1)



test_data=test_data.drop(columns=['c_charge_desc'])

#create tables with only white and AA
data_AA=data[data['race'] == 1] 
data_cauc=data[data['race'] == 0] 

# data.drop(columns=['race'])

#---------------normalize Miner: .68
norm_data=preprocessing.normalize(data)

#make table with only AA race data
norm_data_AA=norm_data
# norm_data_AA=norm_data_AA['race' == 0]

norm_AA=preprocessing.normalize(data_AA)
norm_cauc=preprocessing.normalize(data_cauc)


norm_data_test=preprocessing.normalize(test_data)
#----------------------------------------
clf=GradientBoostingClassifier()
clf=clf.fit(norm_data, label)
#predict test set
y_pred = clf.predict(norm_data_test)

np.savetxt("boost_output.txt", y_pred,fmt='%i')
#----------------------------------------

x_train,x_test,y_train,y_test=train_test_split(norm_data,label,test_size=0.2)

clf_split= GradientBoostingClassifier()
clf_split= clf_split.fit(x_train,y_train)



# print(a)
# print(f1)
#print(matrix)

#5-fold cross validation
kf5 = KFold(n_splits=5, shuffle=False)
cross_score=cross_val_score(clf_split, norm_data, label, cv=kf5)
cross_pred=cross_val_predict(clf_split, norm_data, label, cv=kf5)
# print(cross_pred.shape)
# print(cross_score)

acc = accuracy_score(label, cross_pred)
# print(acc)

m=confusion_matrix(label,cross_pred)
# print(m)

#DIY confusion matrix
aa_matrix=[0,0,0,0]
cauc_matrix=[0,0,0,0]

#race, label, cross pred, are used to make my confusion matrix
for i in range(len(label)):
    #if false positive
    #aa
    if race[i]==1:

        if cross_pred[i]==0:
            #FN
            if label[i]==1:
                aa_matrix[2]+=1
            #TN
            else:
                aa_matrix[0]+=1
        if cross_pred[i]==1:
            #FP
            if label[i]==0:
                aa_matrix[1]+=1
            #TP
            else:
                aa_matrix[3]+=1
    #cauc
    if race[i]==0:
        if cross_pred[i]==0:
            if label[i]==1:
                cauc_matrix[2]+=1
            else:
                cauc_matrix[0]+=1
        if cross_pred[i]==1:
            if label[i]==0:
                cauc_matrix[1]+=1
            else:
                cauc_matrix[3]+=1

#FPR calculation
aa_fp=aa_matrix[1]/(aa_matrix[0]+aa_matrix[1])
cauc_fp=cauc_matrix[1]/(cauc_matrix[0]+cauc_matrix[1])

#Calibration calculation
aa_calib=aa_matrix[3]/(aa_matrix[3]+aa_matrix[1])
cauc_calib=cauc_matrix[3]/(cauc_matrix[3]+cauc_matrix[1])

#Print matrix for both groups
print(aa_matrix)
print(cauc_matrix)

print(aa_fp)
print(cauc_fp)

print(aa_calib)
print(cauc_calib)











