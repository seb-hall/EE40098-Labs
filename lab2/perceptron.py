# import libs
import numpy

def perceptron(inputs, weights, bias):
    """
    A single perceptron function

    Args:
        inputs: Input values
        weights: Weight values
        bias: Bias value

    Returns:
        int: Output of the perceptron (0 or 1)
    """
    # Convert the inputs list into a numpy array
    inputs = numpy.array(inputs)

    # Conver the weights list into a numpy array
    weights = numpy.array(weights)

    # Calculate the dot product
    summed = numpy.dot(inputs, weights)

    # Add the bias
    summed = summed + bias

    # Calculate output
    output = 1 if summed > 0 else 0
    
    return output

# Test the perceptron
inputs = [1.0, 0.0]
weights = [1.0, 1.0]
bias = -1

print("Inputs: ", inputs)
print("Weights: ", weights)
print("Bias: ", bias)
print("Result: ", perceptron(inputs, weights, bias))
