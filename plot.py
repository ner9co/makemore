import matplotlib.pyplot as plt

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap="Blues")
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color="grey")
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="grey")
plt.axis("off")
