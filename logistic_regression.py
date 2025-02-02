import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from load_data import prepare_split


def get_softmax(z):
    """
    returns softmax probablities
    """
    # prevent overflow by subtracting max
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    sm_z = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return sm_z


def get_cross_entropy_loss(y, t, W, reg):
    """
    returns CE loss, provided model outputs and ground truths
    """
    M = y.shape[0]
    t_one_hot = np.zeros_like(y)
    t_one_hot[np.arange(M), t] = 1
    ce_loss = -np.sum(t_one_hot * np.log(y + 1e-10)) / M
    reg_loss = reg * np.sum(W ** 2)

    return ce_loss + reg_loss


def train_log_soft_reg(X_train, t_train, X_val, t_val, classes, alpha, epochs, batch_size, reg):
    """
    train the model
    """
    # init shape and random weights
    N, D = X_train.shape
    W = np.random.randn(D, classes) * 0.01
    train_losses = []
    valid_accs = []
    best_epoch = 0
    best_valid_acc = 0
    W_best = None

    for epoch in range(epochs):
        indices = np.arange(N)
        np.random.shuffle(indices)
        X_train, t_train = X_train[indices], t_train[indices]

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            X_batch, t_batch = X_train[start:end], t_train[start:end]

            # calculate CE loss and update weights
            z = np.dot(X_batch, W)
            y = get_softmax(z)
            t_dash = np.zeros_like(y)
            t_dash[np.arange(t_batch.size), t_batch] = 1
            grad_W = np.dot(X_batch.T, (y - t_dash)) / X_batch.shape[0]
            grad_W = grad_W + 2* reg * W
            W = W - alpha * grad_W

        # Track metrics
        _, _, train_loss, _ = predict_log_soft_reg(X_train, W, t_train, reg)
        _, _, _, valid_acc = predict_log_soft_reg(X_val, W, t_val, reg)
        
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch
            W_best = W.copy()

    return best_epoch, best_valid_acc, W_best, train_losses, valid_accs


def predict_log_soft_reg(X, W, t, reg):
    """
    use model and predict the values
    """
    z = np.dot(X, W)
    y = get_softmax(z)
    labels = np.argmax(y, axis=1)
    ce_loss = get_cross_entropy_loss(y, t, W, reg)
    acc = np.mean(labels == t)

    return y, labels, ce_loss, acc


# load the dataset, get the splits and standardize
dataset = load_from_disk("./shapes_dataset")
X_train, t_train = prepare_split(dataset["train"])
X_val, t_val = prepare_split(dataset["validation"])
X_test, t_test = prepare_split(dataset["test"])
X_train = (X_train - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)
X_val = (X_val - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)
X_test = (X_test - np.mean(X_train, axis=0)) / (np.std(X_train, axis=0) + 1e-8)

# train the logistic softmax regression model
classes = 6
alpha = 0.0001
epochs = 50
batch_size = 512
reg_term = 0.001
best_epoch, best_valid_acc, W_best, train_losses, valid_accs = train_log_soft_reg(X_train, t_train, X_val, t_val, classes, alpha, epochs, batch_size, reg_term)

# get test set accuracy
softmax_outputs, test_predtions, test_loss, test_accuracy = predict_log_soft_reg(X_test, W_best, t_test, reg_term)
print(f"Best Epoch: {best_epoch}")
print(f"Corresponding Validation Accuracy: {best_valid_acc}")
print(f"Logistic Softmax Regression Test Accuracy: {test_accuracy:.4f}")

# plot data (training loss and validation accuracy)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(train_losses, label='Training Loss', color='blue')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('CE Loss')
ax[0].legend()
ax[0].set_title('Training Loss Curve for Logistic Regression')
ax[1].plot(valid_accs, label='Validation Accuracy', color='green')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].legend()
ax[1].set_title('Validation Accuracy Curve for Logistic Regression')
plt.tight_layout()
plt.show()
