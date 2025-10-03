print("Enter a number: ")
number = int(input())
factorial = 1
for i in range(1, number + 1):
    factorial = factorial * i
print("The factorial of " + str(number) + " is " + str(factorial))