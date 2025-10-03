# Lists are a way of storing lots of elements
squares = [1, 4, 9, 16, 25]
# Just like strings they can be indexed (from 0) and sliced
one = squares[0]
end = squares[-1]
# Note that the slice operation returns a new list containing the elements
squaresSlice = squares[0:3]
# Lists can be concatenated
squares = squares + [36, 49, 64, 81, 100]
# Unlike strings, lists are mutable. You can change the individual elements
squares[0] = 1 ** 2
# Len() also works on lists...
squaresLength = len(squares)