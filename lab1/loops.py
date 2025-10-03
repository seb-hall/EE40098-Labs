# Loops are a great way of repeating sections of code # Can you work out what this does?
a = 0
b = 1
# Importantly, the contents of the loop is enclosed using indents (single tab)
# Not {} like you get in C, loops can be nested, you just keep indenting
while b < 10:
    print(b)
    a = b
    b = b + a
# We can also use for loops
# We need to generate a list of items to loop through
words = ["cat", "window", "defenestrate"]
for word in words:
    print(word)
# If we just want to loop a set number of times, we can make a range of integers # An easy way to this is using range()
# range(10) produces [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in range(10):
    print(i)
# We can perform a conditional check using the if statement
# the else and the elif (else if) are optional, note the indentation to group the code
if pi < 3:
    pi = 3.141
    print("Who ate the pie?")
elif pi > 4:
    print("Too much pie..")
else:
    pi = 0;