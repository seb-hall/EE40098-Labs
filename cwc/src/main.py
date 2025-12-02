# %%
from data import DataLoader

data = DataLoader()
data.load_from_mat("cwc/data/D1.mat")

print(data.data.shape)


# %%
import matplotlib.pyplot as plt

plt.plot(data.data[0])
plt.show()


