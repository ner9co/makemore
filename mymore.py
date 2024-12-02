import torch
import torch.nn.functional as F


# %matplotlib inline


# Set seed
g = torch.Generator().manual_seed(2147483647)


# Read the data into memory
with open("names.txt", "r") as f:
    words = f.read().splitlines()


# Create the lookup table
chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}


# --------------------------------------------
# Bigram model
# --------------------------------------------
# Create table of counts
N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# Normalize the rows in count table N
P = N.float()
P /= P.sum(1, keepdim=True)

# Inference
for i in range(5):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))


# ----------------------------------------------
# Neural network approach
# ----------------------------------------------
## Create training set of bigrams (x,y)
xs, ys = [], []
for w in words:
    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# Randomly initialize 27 neuron's weight. each neuron recieves 27 inputs
W = torch.randn((27, 27), generator=g, requires_grad=True)

## gradient descent
for k in range(10):
    # forward pass
    xenc = F.one_hot(
        xs, num_classes=27
    ).float()  # input to the network: one-hot encoding
    logits = xenc @ W  # predict log-counts
    counts = logits.exp()  # counts equivalent to N
    probs = counts / counts.sum(1, keepdim=True)  # probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean()
    print(loss.item())

    # Backward pass
    W.grad = None  # set gradient to zero
    loss.backward()

    # update
    W.data += -0.1 * W.grad


## inference
for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print("".join(out))
