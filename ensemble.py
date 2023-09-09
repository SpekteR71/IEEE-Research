import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import io
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten,BatchNormalization,LSTM
from sklearn.preprocessing import LabelEncoder, StandardScaler

##CNN Model

from google.colab import files
upload = files.upload()
print(upload.keys())

d= pd.read_csv(io.BytesIO(upload['Static (1).csv']))
#d.head()
cnn_df = pd.DataFrame(d)
le =LabelEncoder()
cnn_df['filename']=le.fit_transform(cnn_df['filename'])

cnn_df_train = cnn_df.iloc[:60,:].sample(10)
cnn_df_test = cnn_df.iloc[61:,:].sample(10)
# cnn_df_test

cnn_df_tt = cnn_df_train.sample(5,replace = True)
cnn_X = cnn_df.iloc[:,:-1]
cnn_Y = cnn_df.iloc[:,-1]



cnn_X_train,cnn_X_test,cnn_Y_train,cnn_Y_test = train_test_split(cnn_X,cnn_Y,test_size = 0.2,random_state = 42)
cnn_X_train.shape


#CNN model
from keras.layers.pooling.max_pooling1d import MaxPooling1D
from keras.models import Sequential
from keras.layers import Dense,Conv1D,Flatten,BatchNormalization,LSTM

cnn_model = Sequential()

# Reshape 1D data into 2D data
cnn_X_train_2d = cnn_X_train.values.reshape(cnn_X_train.shape[0], cnn_X_train.shape[1], 1)
cnn_X_test_2d = cnn_X_test.values.reshape(cnn_X_test.shape[0], cnn_X_test.shape[1], 1)

# Create a 1D Convolutional model
cnn_model.add(Conv1D(32, kernel_size=(3), activation='relu', input_shape=(cnn_X_train.shape[1], 1)))

# You can add more Convolutional layers if needed here
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Flatten())
cnn_model.add(Dense(64, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))




cnn_model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

history = cnn_model.fit(cnn_X_train_2d,cnn_Y_train,validation_data = (cnn_X_test_2d,cnn_Y_test),epochs=50)

loss, accuracy = cnn_model.evaluate(cnn_X_test_2d, cnn_Y_test)
#print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

##RNN Model

from google.colab import files
uploaded = files.upload()

rnn_df= pd.read_csv(io.BytesIO(uploaded['Dynamic.csv']))

#encode the 'filename'->contains letter,numebers
le =LabelEncoder()
rnn_df['filename']=le.fit_transform(rnn_df['filename'])
rnn_X=rnn_df.iloc[:,:-1]
rnn_y=rnn_df.iloc[:,-1]
label_encoder=LabelEncoder()
rnn_y=label_encoder.fit_transform(rnn_y)

# Split the data into training and testing sets
rnn_X_train, rnn_X_test, rnn_y_train, rnn_y_test = train_test_split(rnn_X, rnn_y, test_size=0.2, random_state=42)

print(rnn_X_train.shape)

# reshape from [samples, features] into [samples, timesteps, features]
rnn_X_train1 = rnn_X_train.values.reshape((rnn_X_train.shape[0], 1, rnn_X_train.shape[1]))#use .values as u cant reshape it directly
rnn_X_test1 = rnn_X_test.values.reshape((rnn_X_test.shape[0], 1, rnn_X_test.shape[1]))



# Define the RNN model
from tensorflow.keras.layers import BatchNormalization, Dropout
rnn_model = Sequential()
rnn_model.add(LSTM(64, input_shape=(1, 22),return_sequences=True))
rnn_model.add(Dense(64,activation="relu"))
rnn_model.add(BatchNormalization())
rnn_model.add(LSTM(64,return_sequences=False))
rnn_model.add(Dense(64,activation="relu"))
rnn_model.add(BatchNormalization())
rnn_model.add(Dense(1,activation="sigmoid"))


rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#RNNf1->83


rnn_X_train1 = rnn_X_train1.astype('float32')#tf needs this specific data type or else cant convert numpy array to tensor
rnn_y_train1 = rnn_y_train.astype('float32')
rnn_X_test1 = rnn_X_test1.astype('float32')
rnn_y_test = rnn_y_test.astype('float32')

rnn_model.fit(rnn_X_train1, rnn_y_train1, epochs=35, batch_size=30, validation_data=(rnn_X_test1,rnn_y_test))


# Evaluate the model
loss, accuracy = rnn_model.evaluate(rnn_X_test1, rnn_y_test)
#print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")



Boosting

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

rnn_gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.1,max_depth=3, random_state=42)
rnn_gb_classifier.fit(rnn_X_train,rnn_y_train)
rnn_y_pred = rnn_gb_classifier.predict(rnn_X_test)

rnn_accuracy = accuracy_score(rnn_y_test, rnn_y_pred)
print(f"Accuracy: {rnn_accuracy}")

from sklearn.ensemble import GradientBoostingClassifier
cnn_gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate = 0.1,max_depth=3, random_state=42)
cnn_gb_classifier.fit(cnn_X_train,cnn_Y_train)
cnn_y_pred = cnn_gb_classifier.predict(cnn_X_test)

cnn_accuracy = accuracy_score(cnn_Y_test, cnn_y_pred)
print(f"Accuracy: {cnn_accuracy}")


Stacking

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

rnn_tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
rnn_forest_model = RandomForestClassifier(n_estimators=50, random_state=42)
# rnn_svc_model = SVC(kernel='linear', C=0.1, random_state=42)

meta_model = LogisticRegression()
rnn_tree_model.fit(rnn_X_train,rnn_y_train)
rnn_forest_model.fit(rnn_X_train,rnn_y_train)
# rnn_svc_model.fit(rnn_X_train,rnn_y_train)

rnn_pred1 = rnn_tree_model.predict(rnn_X_test)
rnn_pred2 = rnn_forest_model.predict(rnn_X_test)
# rnn_pred3 = rnn_svc_model.predict(rnn_X_test)


rnn_stacked_x = [rnn_pred1, rnn_pred2]
rnn_stacked_x = np.array(rnn_stacked_x).T

meta_model.fit(rnn_stacked_x,rnn_y_test)
stack_pred = meta_model.predict(rnn_stacked_x)

rnn_accuracy = accuracy_score(rnn_y_test, stack_pred)
print(f"Stacking Accuracy: {accuracy}")

rnn_X_train.shape



cnn_tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
cnn_forest_model = RandomForestClassifier(n_estimators=50, random_state=42)
cnn_svc_model = SVC(kernel='linear', C=0.1, random_state=42)

meta_model = LogisticRegression()

cnn_tree_model.fit(cnn_X_train,cnn_Y_train)
cnn_forest_model.fit(cnn_X_train,cnn_Y_train)
cnn_svc_model.fit(cnn_X_train,cnn_Y_train)

# cnn_pred1 = cnn_tree_model.predict(cnn_X_test)
# cnn_pred2 = cnn_forest_model.predict(cnn_X_test)
# cnn_pred3 = cnn_svc_model.predict(cnn_X_test)

# cnn_stacked_x = [cnn_pred1, cnn_pred2, cnn_pred3]
# cnn_stacked_x = np.array(cnn_stacked_x).T

# meta_model.fit(cnn_stacked_x,cnn_Y_test)
# stack_pred = meta_model.predict(cnn_stacked_x)

# cnn_accuracy = accuracy_score(cnn_Y_test, stack_pred)
# print(f"Stacking Accuracy: {accuracy}")

stacked_avg_accuracy = (rnn_accuracy+cnn_accuracy)/2

###Bagging


from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators = 100,
    max_samples = 0.8,
    oob_score = True,
    random_state=0
)

# bag_model.fit(cnn_X_train,cnn_Y_train)
# cac = bag_model.oob_score_
# cac

# bag_model.fit(rnn_X_train,rnn_y_train)
# rac = bag_model.oob_score_
# rac

# print("Ensemble accuracy:",(cac+rac)/2)

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='binary')
# recall = recall_score(y_test, y_pred, average='binary')
# f1 = f1_score(y_test, y_pred, average='binary')

# print('Accuracy:', accuracy)
# print('Precision:', precision)
# print('Recall:', recall)
# print('F1 Score:', f1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
rnn_X_test_2d = rnn_X_test.values.reshape(rnn_X_test.shape[0], -1)
# Get predictions for RNN model
bag_model.fit(rnn_X_train,rnn_y_train)
rnn_y_pred = bag_model.predict(rnn_X_test_2d)

# Compute metrics for RNN model
rnn_accuracy = accuracy_score(rnn_y_test, rnn_y_pred)
rnn_precision = precision_score(rnn_y_test, rnn_y_pred, average='weighted') # specify other options for multi-class problems
rnn_recall = recall_score(rnn_y_test, rnn_y_pred, average='weighted') # specify other options for multi-class problems
rnn_f1 = f1_score(rnn_y_test, rnn_y_pred, average='weighted') # specify other options for multi-class problems

print(f'RNN Model - Accuracy: {rnn_accuracy}, Precision: {rnn_precision}, Recall: {rnn_recall}, F1: {rnn_f1}')

# Fit the CNN model and get predictions
bag_model.fit(cnn_X_train, cnn_Y_train)
cnn_y_pred = bag_model.predict(cnn_X_test)

# Compute metrics for CNN model
cnn_accuracy = accuracy_score(cnn_Y_test, cnn_y_pred)
cnn_precision = precision_score(cnn_Y_test, cnn_y_pred, average='weighted') # specify other options for multi-class problems
cnn_recall = recall_score(cnn_Y_test, cnn_y_pred, average='weighted') # specify other options for multi-class problems
cnn_f1 = f1_score(cnn_Y_test, cnn_y_pred, average='weighted') # specify other options for multi-class problems

print(f'CNN Model - Accuracy: {cnn_accuracy}, Precision: {cnn_precision}, Recall: {cnn_recall}, F1: {cnn_f1}')




# Compute average metrics
avg_accuracy = (cnn_accuracy + rnn_accuracy) / 2
avg_precision = (cnn_precision + rnn_precision) / 2
avg_recall = (cnn_recall + rnn_recall) / 2
avg_f1 = (cnn_f1 + rnn_f1) / 2

print('\nAverage Metrics:')
print('Accuracy:', avg_accuracy)
print('Precision:', avg_precision)
print('Recall:', avg_recall)
print('F1 Score:', avg_f1)


# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# # Predict classes for test set
# cnn_y_pred = bag_model.predict(cnn_X_test)
# rnn_y_pred = bag_model.predict(rnn_X_test)

# # Compute metrics for CNN model
# cnn_accuracy = accuracy_score(cnn_Y_test, cnn_y_pred)
# cnn_precision = precision_score(cnn_Y_test, cnn_y_pred, average='binary')
# cnn_recall = recall_score(cnn_Y_test, cnn_y_pred, average='binary')
# cnn_f1 = f1_score(cnn_Y_test, cnn_y_pred, average='binary')

# print('CNN Model:')
# print('Accuracy:', cnn_accuracy)
# print('Precision:', cnn_precision)
# print('Recall:', cnn_recall)
# print('F1 Score:', cnn_f1)

# # Compute metrics for RNN model
# rnn_accuracy = accuracy_score(rnn_y_test, rnn_y_pred)
# rnn_precision = precision_score(rnn_y_test, rnn_y_pred, average='binary')
# rnn_recall = recall_score(rnn_y_test, rnn_y_pred, average='binary')
# rnn_f1 = f1_score(rnn_y_test, rnn_y_pred, average='binary')

# print('\nRNN Model:')
# print('Accuracy:', rnn_accuracy)
# print('Precision:', rnn_precision)
# print('Recall:', rnn_recall)
# print('F1 Score:', rnn_f1)

# # Compute average metrics
# avg_accuracy = (cnn_accuracy + rnn_accuracy) / 2
# avg_precision = (cnn_precision + rnn_precision) / 2
# avg_recall = (cnn_recall + rnn_recall) / 2
# avg_f1 = (cnn_f1 + rnn_f1) / 2

# print('\nAverage Metrics:')
# print('Accuracy:', avg_accuracy)
# print('Precision:', avg_precision)
# print('Recall:', avg_recall)
# print('F1 Score:', avg_f1)