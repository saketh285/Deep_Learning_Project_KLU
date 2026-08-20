import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images, labels, img_size=(48, 48)):
    processed_images = []
    for img in images:
        # Resize the image
        resized_img = cv2.resize(img, img_size)
        # Normalize pixel values
        normalized_img = resized_img / 255.0
        processed_images.append(normalized_img)
    return np.array(processed_images), labels

# Load your dataset here
# Assuming `images` is a list of images and `labels` is a list of corresponding labels
# Convert images to numpy array
images = np.array(images)
# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Preprocess the images
X_train, y_train = preprocess_images(X_train, y_train)
X_val, y_val = preprocess_images(X_val, y_val)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Build the model
model = build_model()

from tensorflow.keras.utils import to_categorical

# Convert labels to one-hot encoding
y_train_categorical = to_categorical(y_train)
y_val_categorical = to_categorical(y_val)

# Train the model
model.fit(X_train, y_train_categorical, validation_data=(X_val, y_val_categorical), epochs=10, batch_size=32)


# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val_categorical)
print(f"Validation Accuracy: {accuracy}")

# Load a new image and preprocess it
new_image = cv2.imread('path_to_your_image.jpg', cv2.IMREAD_GRAYSCALE)
new_image = cv2.resize(new_image, (48, 48))
new_image = new_image / 255.0
new_image = np.expand_dims(new_image, axis=0)

# Make a prediction
prediction = model.predict(new_image)
predicted_label = np.argmax(prediction)
print(f"Predicted Stress Level: {predicted_label}")
