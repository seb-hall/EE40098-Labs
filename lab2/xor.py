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
inputs = [1.0, 1.0]

print("Inputs: ", inputs)

weights_and = [1.0, 1.0]
bias_and = -1.5
result_and = perceptron(inputs, weights_and, bias_and)

print("Weights AND: ", weights_and)
print("Bias AND: ", bias_and)
print("Result AND: ", result_and)

weights_or = [1.0, 1.0]
bias_or = -0.5
result_or = perceptron(inputs, weights_or, bias_or)

print("Weights OR: ", weights_or)
print("Bias OR: ", bias_or)
print("Result OR: ", result_or)

weights_xor = [-1.0, 1.0]
bias_xor = -0.5
inputs_xor = [result_and, result_or]
result_xor = perceptron(inputs_xor, weights_xor, bias_xor)

print("Weights XOR: ", weights_xor)
print("Bias XOR: ", bias_xor)
print("Result XOR: ", result_xor)


