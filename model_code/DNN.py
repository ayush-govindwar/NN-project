# All the imports we will be needing
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import seaborn as sns

# We first define a class for our neural network
# Then we will define all the functions which it will need one by one
class DNN():
    # First, we have to write an init function to initialize values which the model needs
    def __init__(self, sizes: list, activation='sigmoid'):
        self.sizes = sizes
        self.num_layers = len(sizes)

        if activation == 'sigmoid':
            self.activation = self.sigmoid
        elif activation == 'relu':
            self.activation = self.relu
        else:
            raise ValueError("Activation can either be 'sigmoid' or 'relu'. Given neither")
        
        self.params = self.initialize()
        self.cache = {} # for saving the activations

    # Defines the functionality of the ReLU activation function
    def relu(self, x, grad=False):
        if grad:
            x = np.where(x<0, 0, x)
            x = np.where(x>=0, 1, x)
            return x
        return np.maximum(0,x)

    # Defines the functionality of the sigmoid activation function
    def sigmoid(self, x, grad=False):
        if grad:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))
    
    # Defines the functionality of the softmax activation function
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    # Here, we initialize the weights and biases for all the layers
    # We will be using the He initialization
    def initialize(self):
        params = {}
        for i in range(1, self.num_layers):
            # He init
            scale = np.sqrt(1./self.sizes[i-1])
            params[f"W{i}"] = np.random.randn(self.sizes[i], self.sizes[i-1]) * scale #Return a sample (or samples) from the “standard normal” distribution.
            params[f"b{i}"] = np.zeros((self.sizes[i], 1))

        return params
    
    # Defines the functionality to move forward through the neural network
    # We use the cache which we defined the init function
    def forward(self, x):
        self.cache["X"] = x
        self.cache["A0"] = x.T

        for i in range(1, self.num_layers):
            self.cache[f"Z{i}"] = self.params[f"W{i}"] @ self.cache[f"A{i-1}"] + self.params[f"b{i}"] # The same W*A[i-1] + b 
            # We apply the selected activation function to all layers other than the last layer
            # Last layer uses softmax so we get probabilities for each class
            if (i < self.num_layers-1):
                self.cache[f"A{i}"] = self.activation(self.cache[f"Z{i}"])
            else:
                self.cache[f"A{i}"] = self.softmax(self.cache[f"Z{i}"])
        
        return self.cache[f"A{self.num_layers-1}"]
    
    # Defines the functionality to move back through the neural net, calculating all the necessary gradients as we go backward
    def backprop(self, y, y_hat):
        batch_size = y.shape[0]
        self.grads = {}

        l = self.num_layers - 1
        dZ = y_hat - y.T
        self.grads[f"W{l}"] = (1./batch_size) * (dZ @ self.cache[f"A{l-1}"].T) # Same as (dZ*A[l-1])/total
        self.grads[f"b{l}"] = (1./batch_size) * np.sum(dZ, axis=1, keepdims=True) # Same as (dZ/total)
        dA = self.params[f"W{l}"].T @ dZ # Same as W*dZ

        for i in range(l-1, 0, -1):
            dZ = dA * self.activation(self.cache[f"Z{i}"], grad=True)
            self.grads[f"W{i}"] = (1./batch_size) * (dZ @ self.cache[f"A{i-1}"].T)
            self.grads[f"b{i}"] = (1./batch_size) * np.sum(dZ, axis=1, keepdims=True)
            if i>1:
                dA = self.params[f"W{i}"].T @ dZ
        
        return self.grads
    
    # This actually updates the weights and biases
    # This is where the model "learns"
    def optimize(self, lr=0.001):
        for key in self.params:
            self.params[key] = self.params[key] - lr*self.grads[key]
    
    # Defines the loss function for our neural net
    # We will be using the cross entropy loss function
    def cross_entropy_loss(self, y, y_hat):
        epsilon = 1e-15  # Small constant to prevent log(0)
        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)
        return -(1./y.shape[0])*(np.sum(y.T*np.log(y_hat)))

    # Calculates the accuracy of our model
    def acc(self, y, y_hat):
        return np.mean(np.argmax(y, axis=-1) == np.argmax(y_hat.T, axis=-1))
    
    # This function is where the actual training of our model takes place
    # We culminate all we have made so far into one function
    def train(self, X_train, y_train, X_test, y_test, epochs = 10, batch_size=64, lr=0.01):
        num_batches = -(-X_train.shape[0]//batch_size)

        start_time = time.time()
        # Information template so we know how our model is learning as the epochs progress
        template = "Epoch {}: {:.2f}s, train acc={:.2f}, train loss={:.2f}, test acc={:.2f}, test loss={:.2f}"
        stats = {"train_acc" : [],
                 "train_loss" : [],
                 "test_acc" : [],
                 "test_loss" : []}

        for epoch in range(epochs):
            for i in range(num_batches):
                # Dividing the data into small batches
                begin = i * batch_size
                end = min(begin + batch_size, X_train.shape[0]-1)
                X = X_train[begin:end]
                y = y_train[begin:end]

                # Training aka forward propagation then backward propagation and then optimize
                y_hat = self.forward(X)
                self.backprop(y, y_hat)
                self.optimize(lr)

            # Testing the current model on the testing and training data to get current accuracy
            y_hat = self.forward(X_train)
            train_acc = self.acc(y_train, y_hat)
            train_loss = self.cross_entropy_loss(y_train, y_hat)

            y_hat = self.forward(X_test)
            test_acc = self.acc(y_test, y_hat)
            test_loss = self.cross_entropy_loss(y_test, y_hat)

            print(template.format(epoch+1, time.time()-start_time, train_acc, train_loss, test_acc, test_loss))

            stats["train_acc"].append(train_acc)
            stats["train_loss"].append(train_loss)
            stats["test_acc"].append(test_acc)
            stats["test_loss"].append(test_loss)
        
        return stats
    
    # We use this function to test our model
    def test(self, img, y=None, save_path='model_code/results/test.png'):
        img_reshaped = img.reshape(1, -1)
        pred_prob = self.forward(img_reshaped)
        y_hat = np.argmax(pred_prob)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.title('Input Image')

        plt.subplot(1, 2, 2)
        probabilities = pred_prob.flatten()
        colors = ['lightcoral'] * 10
        colors[y_hat] = 'lightgreen'
        
        plt.bar(range(10), probabilities, color=colors)
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title(f'Prediction: {y_hat}' + (f'\nActual: {y}' if y is not None else ''))
        plt.xticks(range(10))
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
        return y_hat
    
    # This function evaluates our model in four different ways
    # First, we graph the training and testing accuracy and loss for each epoch
    # Then, we make the confusion matrix
    # Then, we calculate the precision, recall, and f1-score
    # We also make predictions on some testing images to show our the model is predicting
    # Finally, we print the architecture of the model, training and testing accuracy, and loss
    def eval(self, X_train, y_train, X_test, y_test, stats=None, save_path='model_code/results/evals.png'):
        sns.set_style("darkgrid")
        fig = plt.figure(figsize=(20, 20))

        # 1. Training History Plot
        if stats:
            plt.subplot(4, 2, 1)
            plt.plot(stats['train_acc'], label='Train Accuracy')
            plt.plot(stats['test_acc'], label='Test Accuracy')
            plt.title('Training History - Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(4, 2, 2)
            plt.plot(stats['train_loss'], label='Train Loss')
            plt.plot(stats['test_loss'], label='Test Loss')
            plt.title('Training History - Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

        # 2. Confusion Matrix
        y_pred_test = np.argmax(self.forward(X_test), axis=0)
        y_true_test = np.argmax(y_test, axis=1)
        
        conf_matrix = np.zeros((10, 10))
        for i in range(len(y_true_test)):
            conf_matrix[y_true_test[i]][y_pred_test[i]] += 1

        plt.subplot(4, 2, 3)
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')

        # 3. precision, recall, and F1-score
        precision = np.zeros(10)
        recall = np.zeros(10)
        f1_score = np.zeros(10)
        support = np.zeros(10)

        for i in range(10):
            tp = conf_matrix[i][i]
            fp = np.sum(conf_matrix[:, i]) - tp
            fn = np.sum(conf_matrix[i, :]) - tp
            support[i] = np.sum(conf_matrix[i, :])
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

        plt.subplot(4, 2, 4)
        plt.axis('off')
        
        header = ['Class', 'Precision', 'Recall', 'F1-score', 'Support']
        cell_text = []
        
        for i in range(10):
            cell_text.append([
                f'{i}',
                f'{precision[i]:.2f}',
                f'{recall[i]:.2f}',
                f'{f1_score[i]:.2f}',
                f'{int(support[i])}'
            ])
        
        avg_precision = np.mean(precision)
        avg_recall = np.mean(recall)
        avg_f1 = np.mean(f1_score)
        total_support = int(np.sum(support))
        
        cell_text.append([
            'avg/total',
            f'{avg_precision:.2f}',
            f'{avg_recall:.2f}',
            f'{avg_f1:.2f}',
            f'{total_support}'
        ])
        
        table = plt.table(cellText=cell_text,
                        colLabels=header,
                        loc='center',
                        cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('Classification Report', pad=20)

        # 4. Sample Predictions
        n_samples = 4
        random_indices = np.random.randint(0, len(X_test), n_samples)
        
        for idx, i in enumerate(random_indices):
            plt.subplot(4, 2, 5 + idx)
            img = X_test[i].reshape(28, 28)
            pred_prob = self.forward(X_test[i].reshape(1, -1))
            pred = np.argmax(pred_prob)
            true = np.argmax(y_test[i])
            
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f'Pred: {pred}, True: {true}')
        
        # 5. Add text information
        test_acc = self.acc(y_test, self.forward(X_test))
        train_acc = self.acc(y_train, self.forward(X_train))
        test_loss = self.cross_entropy_loss(y_test, self.forward(X_test))
        
        info_text = (
            f'Model Architecture: {self.sizes}\n'
            f'Final Train Accuracy: {train_acc:.4f}\n'
            f'Final Test Accuracy: {test_acc:.4f}\n'
            f'Final Test Loss: {test_loss:.4f}\n'
        )
        
        fig.text(0.01, 0.02, info_text, fontsize=10, family='monospace')
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        return {
            'test_accuracy': test_acc,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'support': support,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1
        }