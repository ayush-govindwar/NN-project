import numpy as np 
from DNN import DNN
import time
from utils import fetch_mnist, preprocess, show_graphs


def main():
    print("\nStarting Neural Network training process")
    
    X, y = fetch_mnist()
    X, y = preprocess(X, y)

    train_size = 60000  # total is 70000
    test_size = y.shape[0] - train_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Train set size: {train_size}, Test set size: {test_size}")

    print("Shuffling training data...")
    shuffle_index = np.random.permutation(np.arange(train_size))
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

    print("Initializing Neural Network...")
    model = DNN(sizes=[784, 64, 10], activation='relu')
    
    print("Starting training...")
    training_start = time.time()
    stats = model.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=128, lr=0.01)
    training_time = time.time() - training_start
    show_graphs(stats)
    
    print(f"Training completed in {training_time:.2f} seconds")

    idx = np.random.randint(0, 10000)
    img = X_test[idx]
    actual_label = np.argmax(y_test[idx])
    model.test(img, actual_label)
    eval_results = model.eval(X_train, y_train, X_test, y_test, stats=stats)
    print(eval_results)

if __name__ == "__main__":
    main()