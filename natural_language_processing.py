# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import np_utils
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import re
import nltk
from sklearn.metrics import confusion_matrix
from keras.models import load_model


# Importing the dataset
dataset = pd.read_csv('Labelled_Data.txt', delimiter = ',,,', quoting = 3, header=None)
dataset.columns = ['Question', 'Category']
# Cleaning the texts

#nltk.download('stopwords')
#from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0, len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Question'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    #review = [ps.stem(word) for word in review]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 100)
X = cv.fit_transform(corpus).toarray()



# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
y= dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


labelencoder_y = LabelEncoder()
y_train= labelencoder_y.fit_transform(y_train)
onehotencoder = OneHotEncoder()
y_train=y_train.reshape(-1,1)
y_train= onehotencoder.fit_transform(y_train).toarray()




# Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(X_train, y_train)


classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = 100))
classifier.add(Dropout(p = 0.3))


# Adding the second hidden layer
classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.3))


# Adding the output layer
classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 5, epochs = 100)

#Saving the trained model
classifier.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'


# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')


# Predicting the Test set results
y_pred = model.predict(X_test)
from numpy import argmax
inverted=[]
for i in range(0,len(y_pred)):
    inverted.append(labelencoder_y.inverse_transform([argmax(y_pred[i])])[0])


# Making the Confusion Matrix

cm = confusion_matrix(y_test,inverted)
#classifier.predict(cv.transform(corpus).toarray()


print('################################################### Model is ready to Use ##################################################')

user_query=''
while not (user_query=='exit'):
    user_query=''
    user_query=input("Enter your Query or type Exit to terminate: ")
    if not user_query.lower()=='exit':
            user_query= re.sub('[^a-zA-Z]', ' ', user_query)
            user_query = user_query.lower()
            user_query = user_query.split()
            #user_query= [ps.stem(word) for word in user_query]
            user_query = ' '.join(user_query)
            input_batch=[]
            input_batch.append(user_query)
            X_samp = cv.transform(input_batch).toarray()
            y_samp=classifier.predict(X_samp)
            output = ' '.join(labelencoder_y.inverse_transform([argmax(y_samp)]))
            print('type: ' + output)

print('################################################## Thank you! ###############################################################')

    
    
    
    



