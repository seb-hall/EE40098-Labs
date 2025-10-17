import ann
import mnist

def main():
    input_nodes = 784 
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3

    net = ann.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    mnist.train("cwa/MNIST/mnist_train_100.csv", net, output_nodes)
    mnist.test("cwa/MNIST/mnist_test_10.csv", net)

if __name__ == '__main__':
    main()