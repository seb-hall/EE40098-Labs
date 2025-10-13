# import libs
import matplotlib.pyplot as plt

fig = plt.figure()

plt.scatter(0, 0, s=50, color="red", zorder=3)
plt.scatter(0, 1, s=50, color="red", zorder=3)
plt.scatter(1, 0, s=50, color="red", zorder=3)
plt.scatter(1, 1, s=50, color="blue", zorder=3)

plt.xlim(-2, 2)
plt.ylim(-2, 2)

plt.xlabel("Input 1")
plt.ylabel("Input 2")
plt.title("State Space of Input Vector")

plt.grid(True, linewidth=1, linestyle=":")
plt.tight_layout()

plt.show()