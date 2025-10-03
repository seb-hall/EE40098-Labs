# Strings can use either single or double quotes
firstName = "Bob"
lastName = "Widlar"
# but be careful if you need to ”contain” a quote mark, use \ to escape # lastName = Mc’Gee
lastName = "Mc\'Gee"
# String can be concatenated using +
fullName = firstName + " " + lastName
# Strings can be indexed, with the first character having index [0].
# There is no character type, a character is simply a string of length 1
firstLetter = fullName[0]
# You can also index from the right, so [-1] is the last character (-0 = 0)
lastLetter = fullName[-1]
# If you just want a slice, you can give a range # Starting at 0 and ending at 2
firstName = fullName[0:3]
# You can’t change modify a string, but you can over-write it or make a new one # You can use len() to check the size..
nameLength = len(fullName)