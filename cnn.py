import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from datasets import load_from_disk
from load_data import prepare_split
import numpy as np
import matplotlib.pyplot as plt


def CNN(input_shape, classes):
    """
    I am using ReLU, Max Pool and Softmax activation
    4 conv and 2 max pool layers
    """
    model = Sequential([Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
                        Conv2D(32, kernel_size=(3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.25),
                        Conv2D(64, kernel_size=(3, 3), activation='relu'),
                        Conv2D(64, kernel_size=(3, 3), activation='relu'),
                        MaxPooling2D(pool_size=(2, 2)),
                        Dropout(0.25),
                        Flatten(),
                        Dense(256, activation='relu'),
                        Dropout(0.5),
                        Dense(classes, activation='softmax')])

    return model


# load dataset and get the splits
dataset = load_from_disk("./shapes_dataset")
X_train, t_train = prepare_split(dataset["train"])
X_val, t_val = prepare_split(dataset["validation"])
X_test, t_test = prepare_split(dataset["test"])
classes = 6

# reshape the input and convert the labels to hot encoding
X_train_cnn = X_train.reshape(-1, 28, 28, 1)
X_val_cnn = X_val.reshape(-1, 28, 28, 1)
X_test_cnn = X_test.reshape(-1, 28, 28, 1)
y_train_cnn = to_categorical(t_train, num_classes=classes)
y_val_cnn = to_categorical(t_val, num_classes=classes)
y_test_cnn = to_categorical(t_test, num_classes=classes)

# init and compile CNN model
cnn_model = CNN((28, 28, 1), classes)
optimizer = Adam(learning_rate=0.0005)
cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# train model and evaluate on test set
cnn_fit = cnn_model.fit(X_train_cnn, y_train_cnn, validation_data=(X_val_cnn, y_val_cnn), epochs=50, batch_size=256)

pred_probs = cnn_model.predict(X_test_cnn)
predictions = np.argmax(pred_probs, axis=1)
test_labels = np.argmax(y_test_cnn, axis=1)
cnn_accuracy = np.mean(predictions == test_labels)

best_epoch = 0
best_val_acc = 0
best_val_loss = float('inf')

for epoch, (val_acc, val_loss) in enumerate(zip(cnn_fit.history['val_accuracy'], cnn_fit.history['val_loss']), start=1):
    if val_acc > best_val_acc:
        best_epoch = epoch
        best_val_acc = val_acc
        best_val_loss = val_loss

print(f"Best Epoch: {best_epoch}")
print(f"Corresponding Validation Accuracy: {best_val_acc:.4f}")
print(f"CNN Test Accuracy: {cnn_accuracy:.4f}")

# Extract values from history
train_loss = cnn_fit.history['loss']
val_acc = cnn_fit.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

# Plot data (training loss and validation accuracy)
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(epochs, train_loss, label='Training Loss', color='blue')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss Curve for CNN')
axs[0].legend()
axs[0].grid(True)
axs[1].plot(epochs, val_acc, label='Validation Accuracy', color='orange')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Validation Accuracy Curve for CNN')
axs[1].legend()
axs[1].grid(True)
plt.tight_layout()
plt.show()