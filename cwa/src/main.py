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

    iterations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
    accuracy = [0.05, 0.8415, 0.845, 0.8355, 0.841, 0.843, 0.8445, 0.8384, 0.8365, 0.8388, 0.8414, 0.8399, 0.8398, 0.8428, 0.8425, 0.8393, 0.8438, 0.8397, 0.8422, 0.8411, 0.845099999999999, 0.8393, 0.8376, 0.8397, 0.841199999999999, 0.842]
    

    '''
    for i in range(1, num_iterations + 1):
        net = ann.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

        #mnist.train("cwa/MNIST/mnist_train.csv", net, output_nodes, i)

        #net.save_to_file("cwa/src/ann/weights")
    
        #net.load_from_file("cwa/src/ann/weights.npz")
    
        percent = mnist.test("cwa/MNIST/mnist_test.csv", net)

        print(f"{i} performance: {percent}%")
    
        iterations.append(i)
        accuracy.append(percent / 100)  # Store as a fraction for plotting
    '''
        
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, accuracy, marker='o', linestyle='-', linewidth=2)

    # Labels and title
    plt.title("Training Accuracy vs. Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid(True)

    # Optional: show accuracy as percentage on the y-axis
    plt.ylim(0, 1)
    #plt.yticks([i/10 for i in range(0, 11)], [f"{i*10}%" for i in range(0, 11)])

    #plt.xticks(range(0, max(iterations) + 1))

    # Try to show interactively (works when running locally)
    try:
        plt.show()
    except Exception:
        pass



if __name__ == '__main__':
    main()