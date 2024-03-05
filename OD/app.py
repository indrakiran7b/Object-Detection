import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Load the pre-trained model
model = load_model('mnist.h5')
def mian():
    from keras import layers
    from keras import models
    from keras.datasets import mnist
    from keras.utils import to_categorical

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()

    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, batch_size=64)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    model.predict(test_images)
    print(test_acc)
def preprocess_image(image):
    # Convert the image to grayscale
    gray = image.convert('L')
    
    # Resize the image to (28, 28)
    resized = gray.resize((28, 28))
    
    # Convert the image to a numpy array and normalize
    img_array = np.array(resized) / 255.0
    
    # Reshape to (1, 28, 28, 1) as the model expects a batch of images
    img_array = img_array.reshape((1,28, 28, 1))
    
    return img_array

def predict_digit(image):
    preprocessed_image = preprocess_image(image)
    
    # Make a prediction using the model
    pred = model.predict(preprocessed_image)[0]
    
    # Get the index of the highest predicted value
    final_pred = np.argmax(pred)
    
    return final_pred

picture = st.camera_input("Take a picture")
if "myImage" not in st.session_state.keys():
    st.session_state["myImage"] = None

if picture:
    st.session_state["myImage"] = picture
    st.image(picture)
    # st.subheader(picture)
    fileName = st.session_state["myImage"].name
    save = st.button("Save")

    if save:
        with open(fileName, "wb") as imageFile:
            sa = imageFile.write(st.session_state["myImage"].getbuffer())
            if sa:
                # s_butt = st.button("Predict")
                st.header(fileName)
                
                    # Load the saved image
                mian()
                saved_image = Image.open(fileName)
                st.header("ok image")
                image = preprocess_image(saved_image)
                pred = model.predict(image)
                final_pred = np.argmax(pred)
                st.header(final_pred)
