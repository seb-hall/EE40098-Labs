import ann
import mnist
import os
import time

def seek_optimal_weights():
    import matplotlib.pyplot as plt

    hidden_nodes_min = 1
    hidden_nodes_max = 1000

    learning_rate_min = 0.01
    learning_rate_max = 0.3

    num_attempts = 25000

    input_nodes = 784
    output_nodes = 10

    best_performance = 0
    best_hidden_nodes = 0
    best_learning_rate = 0

    results = []

    for i in range(num_attempts):
            start = time.perf_counter()

            # random hidden nodes and learning rate
            hidden_nodes = int(hidden_nodes_min + (hidden_nodes_max - hidden_nodes_min) * os.urandom(1)[0] / 255)
            learning_rate = learning_rate_min + (learning_rate_max - learning_rate_min) * os.urandom(1)[0] / 255

            net = ann.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

            mnist.train("cwa/MNIST/fashion_mnist_train_1000.csv", net, output_nodes, 1)
            performance = mnist.test("cwa/MNIST/fashion_mnist_test_10.csv", net)

            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0

            results.append((hidden_nodes, learning_rate, performance))

            if performance > best_performance:
                best_performance = performance
                best_hidden_nodes = hidden_nodes
                best_learning_rate = learning_rate

            print(f"{i}: hidden={hidden_nodes}, lr={learning_rate}, perf={performance}%, time={elapsed_ms:.2f} ms")

    print("Best Performance: ", best_performance, "%")
    print("Best Hidden Nodes: ", best_hidden_nodes)
    print("Best Learning Rate: ", best_learning_rate)

    # Prepare arrays for plotting
    hs = [r[0] for r in results]
    lrs = [r[1] for r in results]
    perfs = [r[2] for r in results]

    plt.figure(figsize=(9, 6))
    # color = performance, marker size scaled by performance
    scatter = plt.scatter(hs, lrs, c=perfs, cmap='viridis', s=[20], edgecolors='none', zorder=2)
    plt.colorbar(scatter, label='Performance (%)')
    plt.xlabel('Hidden Nodes')
    plt.ylabel('Learning Rate')
    plt.title('MNIST Parameter Search: Hidden Nodes vs Learning Rate')
    plt.grid(True)

    # Save plot next to this file in a "plots" folder
    out_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'parameter_search.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=200)
    print("Saved plot to", out_path)

    # Try to show interactively (works when running locally)
    try:
        plt.show()
    except Exception:
        pass


def main():
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.25

    num_iterations = 1

    seek_optimal_weights()
    #net = ann.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # $for i in range(num_iterations):
    #     mnist.train("cwa/MNIST/mnist_train.csv", net, output_nodes, i)
# 
    # net.save_to_file("cwa/src/ann/weights")
# 
    # net.load_from_file("cwa/src/ann/weights.npz")
# 
    # percent = mnist.test("cwa/MNIST/mnist_test.csv", net)

if __name__ == '__main__':
    main()