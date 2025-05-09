import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emotion_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),
    
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax') 
])

emotion_model.load_weights('face_emotion\\emotion_model.weights.h5')
print("Loaded weights into model")

test_data_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_data_gen.flow_from_directory(
        'test', 
        target_size=(48, 48), 
        batch_size=64, 
        color_mode="grayscale", 
        class_mode='categorical',
        shuffle=False)

predictions = emotion_model.predict(test_generator)

print("-----------------------------------------------------------------")
c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))

cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=[emotion_dict[i] for i in range(7)])
cm_display.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

print("-----------------------------------------------------------------")
print(classification_report(test_generator.classes, predictions.argmax(axis=1), target_names=[emotion_dict[i] for i in range(7)]))
