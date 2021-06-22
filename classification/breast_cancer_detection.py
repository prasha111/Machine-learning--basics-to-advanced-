import tensorflow as tf


tf.version


from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

data.type

type(data)

data.keys()

data.data.shape

#536 samples 30 features

data.target

data.data

data.filename

data.DESCR

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(data.data, data.target, test_size = 0.33)

N, D=x_train.shape

## now preprocessing of data

from sklearn.preprocessing import StandardScaler
scalar= StandardScaler()

x_train = scalar.fit_transform(x_train)

x_test = scalar.transform(x_test)

#now build the model

(data.data[1:10])

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(1,input_shape=(D,),activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=('accuracy'))

r=model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=100)

print('Train_score:', model.evaluate(x_train, y_train))


print('Test_score:', model.evaluate(x_test, y_test))




import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
## validation loss is the loss on testing data

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()

### making prediction

p=model.predict(x_test)

print (p)

