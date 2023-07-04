# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 21:43:10 2021

@author: Ahmed Amr
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense ,Dropout
import matplotlib.pyplot as plt


def create_data(data_dir):
    dataset = pd.read_csv(data_dir)
    dataset.head();
    #dataset.info() #info of coloumns

    dataset=dataset.drop(['enrollee_id','city'],axis=1) # not important for our model
    #first handle missing values
    dataset['gender'] = dataset['gender'].fillna("Other")
    dataset['relevent_experience'] = dataset['relevent_experience'].fillna("No relevent experience")
    dataset['enrolled_university'] = dataset['enrolled_university'].fillna("no_enrollment")
    dataset['experience'] = dataset['experience'].fillna("<1")
    dataset['company_size'] = dataset['company_size'].fillna("<10")
    dataset['last_new_job'] = dataset['last_new_job'].fillna("never")
    #there company_type and major_discipline and education_level
    dataset['company_type']=dataset['company_type'].fillna("Pvt Ltd")
    dataset['major_discipline']=dataset['major_discipline'].fillna("No Major")
    dataset['education_level']=dataset['education_level'].fillna("Phd")
    # Recheck for missing values (NaN) in the dataset
    dataset.isna().any()

    # Importing required package

    encode_x = LabelEncoder()

    # Encoding columns ProductID, 
    dataset['relevent_experience'] = encode_x.fit_transform(dataset['relevent_experience'])
    dataset['gender'] = encode_x.fit_transform(dataset['gender'])
    dataset['enrolled_university'] = encode_x.fit_transform(dataset['enrolled_university'])
    dataset['education_level'] = encode_x.fit_transform(dataset['education_level'])
    dataset['major_discipline'] = encode_x.fit_transform(dataset['major_discipline'])
    dataset['company_type'] = encode_x.fit_transform(dataset['company_type'])

    #we have handle 3 object colomuns
    dataset.experience.unique()
    dataset['experience'] = dataset['experience'].replace('>20', 21)
    dataset['experience'] = dataset['experience'].replace('<0', 0)
    dataset['experience'] = dataset['experience'].replace('<1', 0)
    # Converting StayYearsCity from object to integer
    dataset['experience'] = dataset['experience'].astype(str).astype(int)

    #we will do this for the 2 objects
    dataset.company_size.unique()
    dataset['company_size'] = dataset['company_size'].replace('10000+', 7)
    dataset['company_size'] = dataset['company_size'].replace('50-99', 2)
    dataset['company_size'] = dataset['company_size'].replace('<10', 0)
    dataset['company_size'] = dataset['company_size'].replace('1000-4999', 5)
    dataset['company_size'] = dataset['company_size'].replace('100-500', 3)
    dataset['company_size'] = dataset['company_size'].replace('500-999', 4)
    dataset['company_size'] = dataset['company_size'].replace('10/49', 1)
    dataset['company_size'] = dataset['company_size'].replace('5000-9999', 6)
    dataset['company_size'] = dataset['company_size'].astype(str).astype(int)

    dataset.last_new_job.unique()
    dataset['last_new_job'] = dataset['last_new_job'].replace('>4', 5)
    dataset['last_new_job'] = dataset['last_new_job'].replace('never', 0)
    dataset['last_new_job'] = dataset['last_new_job'].astype(str).astype(int)


    dataset['city_development_index'] = dataset['city_development_index'].astype(float).astype(int)
    dataset.info()
    ################################

    #handle outlires
    print(dataset['training_hours'].quantile(0.10))
    print(dataset['training_hours'].quantile(0.90))
    Q1=dataset['training_hours'].quantile(0.25)
    Q3=dataset['training_hours'].quantile(0.75)
    IQR=Q3-Q1
    Min= Q1 - 1.5 * IQR
    Max= Q3 + 1.5 * IQR

    dataset["training_hours"] = np.where(dataset["training_hours"] <Min, 11,dataset['training_hours'])
    dataset["training_hours"] = np.where(dataset["training_hours"] >Max, 146,dataset['training_hours'])
    return dataset


dataset = create_data('aug_train.csv')
predict = create_data('aug_test.csv')
##################
# first split our data for train and test 
train_dataset , test_dataset = train_test_split(dataset,test_size=0.25)
#print(train_dataset.shape )
#print( test_dataset.shape)
X_train=train_dataset.iloc[0:,0:11]
Y_train=train_dataset.iloc[0:,11]
#print(X_train.shape )
X_test=test_dataset.iloc[0:,0:11]
Y_test=test_dataset.iloc[0:,11]

# define the keras model
model = Sequential()
model.add(Dense(11, activation='elu'))#input layer
model.add(Dense (10, activation='relu')) #hidden
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))#output

#############

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
history=model.fit(X_train, Y_train,epochs=100,verbose=1, validation_data=(X_test, Y_test))
test_loss, test_acc = model.evaluate(X_test, Y_test)
# evaluate the keras model
print("accuracy ",test_acc)
################
X_predict=predict.iloc[0:,0:11]
print(model.predict(X_predict))
###########################

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

#confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
prediction=np.round(model.predict(X_train))
cm = confusion_matrix(Y_train, prediction)

cm_df = pd.DataFrame(cm, index = [i for i in "01"], columns = [i for i in "01"])

plt.figure(figsize=(5,4))
sns.heatmap(cm_df, annot=True,fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()
