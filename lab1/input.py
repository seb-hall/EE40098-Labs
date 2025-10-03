# We have already seen how to use print() to display data
# We can use input to get some data from the console
response = input("Is Python amazing? ")
print("You said: " + response)
# If we want to convert from string to int, or vice versa
limit = int(input("Enter a limit: "))
if limit > 3:
# We could also have used "Your limit of", limit, "was too big"
    print("Your limit of " + str(limit) + " was too big")