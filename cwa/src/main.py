import ann
import mnist
import os
import time

import matplotlib.pyplot as plt

def main():
    input_nodes = 784
    hidden_nodes = 254
    output_nodes = 10
    learning_rate = 0.11

    num_iterations = 25

    iterations = []
    accuracy = []

    for i in range(1, num_iterations + 1):
        net = ann.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

        mnist.train("cwa/MNIST/mnist_train.csv", net, output_nodes, i)

        net.save_to_file("cwa/src/ann/weights")
    
        net.load_from_file("cwa/src/ann/weights.npz")
    
        percent = mnist.test("cwa/MNIST/mnist_test.csv", net)

        print(f"{i} performance: {percent}%")
    
        iterations.append(i)
        accuracy.append(percent / 100)  # Store as a fraction for plotting
        
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, accuracy, marker='o', linestyle='-', linewidth=2)

    # Labels and title
    plt.title("Training Accuracy vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid(True)

    # Optional: show accuracy as percentage on the y-axis
    #plt.ylim(0, 1)
    #plt.yticks([i/10 for i in range(0, 11)], [f"{i*10}%" for i in range(0, 11)])

    plt.xticks(range(1, max(iterations) + 1))

    # Try to show interactively (works when running locally)
    try:
        plt.show()
    except Exception:
        pass



if __name__ == '__main__':
    main()