from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from datasets import load_from_disk
from load_data import prepare_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam


def MLP_model(input_shape, classes):
    """
    build the MLP model to use
    I am using ReLU and softmax activation functions
    """
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(2048, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.1),  # Reduced dropout
        Dense(1024, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(classes, activation='softmax')
    ])
    
    return model


# load dataset and get splits
dataset = load_from_disk("./shapes_dataset")
X_train, t_train = prepare_split(dataset["train"])
X_val, t_val = prepare_split(dataset["validation"])
X_test, t_test = prepare_split(dataset["test"])
classes = 6

# standerdize the data
X_train = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)
X_val = (X_val - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)
X_test = (X_test - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)

y_train = to_categorical(t_train, num_classes=classes)
y_val = to_categorical(t_val, num_classes=classes)
y_test = to_categorical(t_test, num_classes=classes)

# init and compile MLP model
mlp_model = MLP_model((28 * 28,), classes)
optimizer = Adam(learning_rate=0.001)
mlp_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
trained_ml_model = mlp_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=512)

# Evaluate the model on the test set
predictions = np.argmax(mlp_model.predict(X_test), axis=1)
test_labels = np.argmax(y_test, axis=1)
mlp_acc = np.mean(predictions == test_labels)

best_epoch = 0
best_val_acc = 0
best_val_loss = float('inf')

for epoch, (val_acc, val_loss) in enumerate(zip(trained_ml_model.history['val_accuracy'], trained_ml_model.history['val_loss']), start=1):
    if val_acc > best_val_acc:
        best_epoch = epoch
        best_val_acc = val_acc
        best_val_loss = val_loss

print(f"Best Epoch: {best_epoch}")
print(f"Corresponding Validation Accuracy: {best_val_acc:.4f}")
print(f"MLP classifier Test Accuracy: {mlp_acc:.4f}")

# Extract values from history
train_loss = trained_ml_model.history['loss']
val_acc = trained_ml_model.history['val_accuracy']
epochs = range(1, len(train_loss) + 1)

# plot data (training loss and validation accuracy)
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].plot(epochs, train_loss, label='Training Loss', color='blue')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training Loss Curve')
axs[0].legend()
axs[0].grid(True)
axs[1].plot(epochs, val_acc, label='Validation Accuracy', color='orange')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Validation Accuracy Curve')
axs[1].legend()
axs[1].grid(True)
plt.tight_layout()
plt.show()
