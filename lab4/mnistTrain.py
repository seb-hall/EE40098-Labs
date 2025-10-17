# Load the MNIST 100 training samples CSV file into a list
training_data_file = open("cwa/MNIST/mnist_train_100.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()
# Train the neural network on each training sample
for record in training_data_list:
    # Split the record by the commas
    all_values = record.split(",")
    # Scale and shift the inputs from 0..255 to 0.01..1
    inputs = (numpy.asarray(all_values[1:], dtype=numpy.float64) / 255.0 * 0.99) + 0.01
    # Create the target output values (all 0.01, except the desired label which is 0.99)
    targets = numpy.zeros(output_nodes) + 0.01
    # All_values[0] is the target label for this record
    targets[int(all_values[0])] = 0.99
    # Train the network
    n.train(inputs, targets)
pass