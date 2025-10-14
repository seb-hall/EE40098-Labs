import matplotlib.pyplot as plt

# AND
points = [(0, 0), (0, 1), (1, 0), (1, 1)]
colors = ['red', 'red', 'red', 'green']  # Each point gets a unique color

# Plot points
for (x, y), color in zip(points, colors):
    plt.scatter(x, y, color=color, s=100, label=f'({x}, {y})')

# Plot AND function line: y = A ∧ B
x_vals = [0, 1.5]
y_vals = [1.5, 0] 
plt.plot(x_vals, y_vals, color='black', linestyle='--', label='AND function')

# Styling
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.xlabel('B')
plt.ylabel('A')
plt.title('Logic States for AND')
plt.axis([-.2, 1.2, -.2, 1.2])  # Padding around edges

plt.show()
