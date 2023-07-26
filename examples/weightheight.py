import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.losses import categorical_crossentropy

def plotData(data):

    male = data[data['Gender'] == 'Male']
    female = data[data['Gender'] == 'Female']
    _ = plt.scatter(male['Weight'], male['Height'], color='blue', marker='.', alpha=0.05)
    _ = plt.scatter(female['Weight'], female['Height'], color='red', marker='.', alpha=0.05)

    plt.show()

def plotHistory(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()   

def createModel():
    model= Sequential()
    model.add(Dense(30, input_shape=(2,),activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # softmax / sigmoid, sigmoid give probability between 0...1
    return model

if __name__=='__main__':
    print ("reading weight and heights")
    data=pd.read_csv("data/weight-height.csv")
    gender_dict={'Male':0,'Female':1}
    data['Gender']=data['Gender'].map(gender_dict).astype('float32')
    print(data.head(10))
    print(data.shape)
    # plotData(data) 
    
    x = data[['Height','Weight']]
    y = data[["Gender"]]
    print (x.head(), x.shape)
    print (y.head(), y.shape)

    x_train,x_validation, y_train, y_validation = train_test_split(x,y,train_size=0.8, test_size=0.2)

    model=createModel()
    # binary classification problem => use binary_crossentropy loss function 'binary_crossentropy'
    # classification CategoricalCrossentropy categorical_crossentropy
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # here show anim : why do we need batch ? Why do we need epochs ?
    history=model.fit(x_train, y_train, epochs=100, batch_size=10, validation_data=(x_validation, y_validation))
    model.save('models/weight-height.model')

    plotHistory(history)
    # evaluate the keras model
    # _, accuracy = model.evaluate(x_test, y_test)
    # print('Accuracy: %.2f' % (accuracy*100))

    # make probability predictions with the model / 
    predictions=model.predict(np.array([[80.3, 180.4],[55.3,150.4]])) # predictions are between 0 and 1, we can see them as probability for the class so just check if <0.5 or >0.5
    rounded = [ ("Female" if x[0]>=0.5 else "Male") for x in predictions]
    print (rounded)
        
