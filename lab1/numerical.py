# This is a comment!
# Python knows that pi is a float
pi = 3.141
# and that freq is an integer
freq = 20000
# So nextFreq will also be an integer
nextFreq = freq + 1
# and radians/s will be a float, because we used a float in the calculation
rads = 2 * pi * nextFreq
# Classic division always returns a float
subRads = rads / 2.4
# Floor division returns the integer part (discards the fraction)
# Modulo division returns the fractional part (discards the integer part)
subRadsInt = rads // 2.4
remainder = rads % 2.4
# Powers can be calculated using **, here we square the frequency
f2 = freq ** 2