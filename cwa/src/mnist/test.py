# Import numpy for arrays and matplotlib for drawing the numbers
import numpy

def test(file, net):
    # Load the MNIST test samples CSV file into a list
    test_data_file = open(file, "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    # Scorecard list for how well the network performs, initially empty
    scorecard = []
    # Loop through all of the records in the test data set
    for record in test_data_list:
        # Split the record by the commas
        all_values = record.split(",")
        # The correct label is the first value
        correct_label = int(all_values[0])
        print(correct_label, "Correct label")
        # Scale and shift the inputs
        inputs = (numpy.asarray(all_values[1:], dtype=numpy.float64) / 255.0 * 0.99) + 0.01
        # Query the network
        outputs = net.query(inputs)
        # The index of the highest value output corresponds to the label
        label = numpy.argmax(outputs)
        print(label, "Network label")
        # Append either a 1 or a 0 to the scorecard list
        if (label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass
    # Calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    score_percent = (scorecard_array.sum() / scorecard_array.size)*100
    print("Performance = ", score_percent, "%")

    return score_percent 