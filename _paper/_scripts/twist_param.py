import numpy as np
import matplotlib.pyplot as plt
N = 100
t = np.linspace(0, 1, N)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))
k = np.linspace(1, 2, 1)
B = np.linspace(1, 2, 1)
Pm = np.linspace(1, 3, 3)

for kval in k:
    for bval in B:
        for pmval in Pm:
            alpha = (kval * t * 2*np.pi / N) + bval * np.sin(N * t*2*np.pi / pmval)
            ax1.plot(t, alpha, label = f"k={kval}, B={bval}, Pm={pmval}")
            ax1.set_ylabel(r"$\alpha$")
            ax1.set_xlabel("t")
            ax1.legend(loc='upper right')

k = np.linspace(1, 4, 4)
B = np.linspace(1, 2, 1)
Pm = np.linspace(1, 2, 1)
for kval in k:
    for bval in B:
        for pmval in Pm:
            alpha = (kval * t * 2*np.pi / N) + bval * np.sin(N * t*2*np.pi / pmval)
            ax2.plot(t, alpha, label = f"k={kval}, B={bval}, Pm={pmval}")
            ax2.legend(loc='upper right')
            ax2.set_xlabel("t")

k = np.linspace(1, 2, 1)
B = np.linspace(1, 4, 4)
Pm = np.linspace(1, 2, 1)
for kval in k:
    for bval in B:
        for pmval in Pm:
            alpha = (kval * t * 2*np.pi / N) + bval * np.sin(N * t*2*np.pi / pmval)
            ax3.plot(t, alpha, label = f"k={kval}, B={bval}, Pm={pmval}")
            ax3.legend(loc='upper right')
            ax3.set_xlabel("t")
plt.tight_layout()
plt.savefig("./_paper/Twistparameter.pdf")