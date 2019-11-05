import numpy as np

data = np.array([[152, 51], [156, 53], [160, 54], [164, 55],
                 [168, 57], [172, 60], [176, 62], [180, 65],
                 [184, 69], [188, 72]])

print(data)

print(data.shape)

x, y = data[:, 0], data[:, 1]
print(x)
print(y)
