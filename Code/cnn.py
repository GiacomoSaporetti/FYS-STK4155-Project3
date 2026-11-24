import copy, random, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


# CNN class made with Pytorch 
class Net(nn.Module):
    def __init__(self, conv_layers: list[dict], fc_layers: list[int],
                 in_shape=(1, 28, 28), n_classes=10):
        super().__init__()
        self.n_classes = n_classes
        self.criterion = nn.CrossEntropyLoss()
        layers, c = [], in_shape[0]

        # Convolutional section of the network

        # 'conv_layers' should be a list of dicts in the form {"out": 8, "k": 5, "pool": 2, "stride": 1, "pad": 0, ...}
        for cfg in conv_layers:
            layers += [nn.Conv2d(c, cfg["out"], cfg["k"],
                                 stride=cfg.get("stride", 1),
                                 padding=cfg.get("pad", 0)),
                       nn.ReLU()]

            if cfg.get("pool"):
                layers.append(nn.MaxPool2d(cfg["pool"]))

            # Update last layer output channel size
            c = cfg["out"]

        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            # Get the number of elements that are going to be the input to the first fully connected layer
            n = self.features(torch.zeros(1, *in_shape)).numel()

        # Fully connected section of the network

        head, d = [nn.Flatten()], n

        # 'fc_layers' should be a list in the form [layer1_nodes, layer2_nodes, ...]
        for h in fc_layers:
            head += [nn.Linear(d, h), nn.ReLU()]
            d = h
        head.append(nn.Linear(d, n_classes))  # no ReLU/softmax on the logits
        self.head = nn.Sequential(*head)

    def forward(self, x):
        # Returns raw logits
        return self.head(self.features(x))

    def loss(self, x, y):
        # y: class indices, torch.long, shape [N] (NOT one-hot)
        return self.criterion(self(x), y)

    def predict(self, x, one_hot=False):
        # Predicted class indices, or one-hot encoded if requested
        with torch.no_grad():
            self.eval()
            pred = self(x).argmax(dim=1)
            return F.one_hot(pred, self.n_classes) if one_hot else pred


def set_seed(seed=0):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# Class for training and validation of each CNN
class CNN:
    def __init__(self, conv_layers, fc_layers, learning_rate=1e-3, optimizer="adam", weight_decay=0.0, max_epochs=100, patience=5, seed=0):
        set_seed(seed)  # before building Net, so the init is reproducible
        self.max_epochs, self.patience = max_epochs, patience
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = Net(conv_layers, fc_layers).to(self.device)
        opts = {"adam": optim.Adam, "sgd": optim.SGD, "adamw": optim.AdamW}
        self.opt = opts[optimizer](self.net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.n_params = sum(p.numel() for p in self.net.parameters())
        self.train_time = None  # seconds, set by fit()


    def fit(self, train_loader, val_loader):
        history, best, best_state, bad = [], float("inf"), None, 0
        t0 = time.time()
        
        for epoch in range(self.max_epochs):
            self.net.train()
            total = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                loss = self.net.loss(x, y)
                loss.backward()
                self.opt.step()
                total += loss.item() * len(x)
            val_loss, val_acc = self.evaluate(val_loader)

            # Update history
            history.append({"epoch": epoch,
                            "train_loss": total / len(train_loader.dataset),
                            "val_loss": val_loss, "val_acc": val_acc})

            # Early stop in case of increasing validation loss
            if val_loss < best:
                best, bad = val_loss, 0
                # Save best model
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break

        # Restore best model (in case of early stop)
        self.net.load_state_dict(best_state)  
        self.train_time = time.time() - t0
        return history


    def evaluate(self, loader):
        with torch.no_grad():
            self.net.eval()
            loss = correct = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.net(x)
                loss += self.net.criterion(logits, y).item() * len(x)
                correct += (logits.argmax(1) == y).sum().item()
            n = len(loader.dataset)
        return loss / n, correct / n


    def predict_proba(self, loader):
        # Softmax probabilities [N, n_classes] and true labels [N], in loader order
        # (use a loader with shuffle=False if the order has to match the dataset)
        self.net.eval()
        P, Y = [], []
        with torch.no_grad():
            for x, y in loader:
                P.append(torch.softmax(self.net(x.to(self.device)), 1).cpu())
                Y.append(y)
        return torch.cat(P).numpy(), torch.cat(Y).numpy()


def run_cnn(cfg, archs, train_set, val_set, max_epochs, patience, seed=0, batch_size=128):
    # Train one configuration: cfg = {"arch": name in archs, "lr": ..., "wd": ...}
    # Returns (row of metrics, trained CNN), the signature expected by tuning.grid_search
    # Fresh loaders every run: the shuffle generator is re-seeded, so runs are comparable
    g = torch.Generator().manual_seed(seed)
    tr = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
    va = DataLoader(val_set, batch_size=512)
    m = CNN(**archs[cfg["arch"]], learning_rate=cfg["lr"], weight_decay=cfg["wd"],
            max_epochs=max_epochs, patience=patience, seed=seed)
    hist = m.fit(tr, va)
    best = min(hist, key=lambda h: h["val_loss"])
    row = {**cfg, "val_acc": best["val_acc"], "val_loss": best["val_loss"],
           "best_epoch": best["epoch"], "epochs_run": len(hist),
           "n_params": m.n_params, "time_s": m.train_time}
    return row, m


def save_cnn(path):
    # Returns a callback (row, model) that stores the weights together with their config
    def _save(row, m):
        row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}  # numpy -> python
        torch.save({"row": row, "state": m.net.state_dict()}, path)
    return _save


def load_net(path, archs):
    # Rebuild a Net from a checkpoint written by save_cnn (or by the older {"cfg", "state"} format)
    ck = torch.load(path)
    row = ck.get("row", ck.get("cfg"))
    net = Net(**archs[row["arch"]])
    net.load_state_dict(ck["state"])
    return net
