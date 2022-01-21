
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

adres = "auto-mpg.data"
sutunlar = ['MPG','Cylinders','Displacament','Horsepower','Weight','Acceleration',
                'Model Year','Origin']

veriseti = pd.read_csv(adres,names=sutunlar,skipinitialspace=True,na_values="?",comment="\t",sep=" ")

veriseti.head(10)

veriseti.tail(10)

veriseti.isna().sum()

veriseti = veriseti.dropna()

veriseti.isna().sum()

veriseti["Origin"].unique()

veriseti["Origin"] = veriseti["Origin"].map({1:"USA",2:"Europe",3:"Japan"})

veriseti.tail(10)

veriseti = pd.get_dummies(veriseti,columns=["Origin"],prefix="",prefix_sep="")

veriseti.tail()

train_set = veriseti.sample(frac=0.8,random_state=0)
test_set = veriseti.drop(train_set.index)

train_set.columns


#sns.pairplot(train_set[['MPG', 'Cylinders', 'Displacament', 'Horsepower', 'Weight',
#       'Acceleration', 'Model Year']])

print(train_set.describe().transpose())

train_labels = train_set.pop("MPG")
test_labels = test_set.pop("MPG")


import tensorflow as tf
from tensorflow.keras import layers


normalizer = layers.experimental.preprocessing.Normalization(axis=-1)
normalizer.adapt(train_set)


dogrusal_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(1)])

# 25
dogrusal_model.compile(loss = "mean_absolute_error",optimizer = tf.keras.optimizers.Adam(learning_rate= 0.1))

history = dogrusal_model.fit(train_set,train_labels,epochs=100,validation_split=0.2,verbose=0)

history = history.history


plt.figure()
plt.title("dogrusal Model")
plt.plot(history["loss"],label="egitim hatasi")
plt.plot(history["val_loss"],label="dogruluk hatasi")
plt.legend()
plt.show()



dnn_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(64,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(1)])

dnn_model.compile(loss = "mean_absolute_error",optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001))


history = dnn_model.fit(train_set,train_labels,epochs=100,validation_split=0.2,verbose=0)

history = history.history


plt.figure()
plt.title("Derin Model")
plt.plot(history["loss"],label="egitim hatasi")
plt.plot(history["val_loss"],label="dogruluk hatasi")
plt.legend()
plt.show()

sonuc = dogrusal_model.evaluate(test_set,test_labels)
sonuc2 = dnn_model.evaluate(test_set,test_labels)

print(test_labels[:5])

print(dnn_model.predict(test_set[:5]))

print(dogrusal_model.predict(test_set[:5]))

