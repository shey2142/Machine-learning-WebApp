import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import tensorflow as tf

def main():
    st.title("Cifar10 Web Classifier")
    st.write("Upload any image that you think fits into one of the classes and see if the prediction is correct.")
    
    # Allow the user to upload an image
    file = st.file_uploader("Please Upload an Image", type=['jpg', 'png'])

    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        # Resize and preprocess the image
        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image) / 255
        img_array = img_array.reshape((1, 32, 32, 3))

        # Load the pre-trained model
        model = tf.keras.models.load_model('cifar10_model.h5')

        # Make a prediction
        prediction = model.predict(img_array)
        cifar10_classes_ascii = [
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck"
        ]

        # Create a horizontal bar chart to display predictions
        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_classes_ascii))
        ax.barh(y_pos, prediction[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_classes_ascii)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title('CIFAR-10 PREDICTION')

        # Display the chart using Streamlit
        st.pyplot(fig)
    else:
        st.text("You have not uploaded an image yet.")

if __name__ == '__main__':
    main()
