import ann
import mnist

def main():
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.25

    num_iterations = 10

    net = ann.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    ##for i in range(num_iterations):
    ##    mnist.train("cwa/MNIST/mnist_train.csv", net, output_nodes, i)

    #net.save_to_file("cwa/src/ann/weights.txt")

    net.load_from_file("cwa/src/ann/weights.txt.npz")

    percent = mnist.test("cwa/MNIST/mnist_test.csv", net)

if __name__ == '__main__':
    main()