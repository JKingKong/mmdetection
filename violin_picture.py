import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(figsize=(10, 10))


all_data = [np.random.normal(0, std, 10) for std in range(9, 10)]
all_data = [np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])]
axes.violinplot(all_data,
                   showmeans=False,
                   showmedians=True
                   )
axes.set_title('violin plot')

# adding horizontal grid lines

axes.yaxis.grid(True)
t = [y + 1 for y in range(len(all_data))]
axes.set_xticks([y + 1 for y in range(len(all_data))], )


plt.setp(axes, xticks=[y + 1 for y in range(len(all_data))],
         xticklabels=['correct'],
         )

plt.show()