"""
VoltSight — BPNN (Back-Propagation Neural Network) codé à la main en NumPy
==========================================================================

Réseau de neurones multicouche entièrement connecté, entraîné par
rétropropagation du gradient. Aucune bibliothèque de deep learning :
chaque étape (propagation avant, calcul de l'erreur, rétropropagation,
mise à jour des poids) est écrite explicitement pour être comprise.

Architecture (exemple) :  entrée (n) -> 128 -> 32 -> 1 (sigmoïde)

Tâche : classification binaire
    y = 1 : isolateur défectueux
    y = 0 : isolateur normal

Fonction de perte : entropie croisée binaire PONDÉRÉE.
    Les défauts sont rares (environ 1 sur 5). On donne plus de poids
    aux exemples défectueux pour que le réseau ne les ignore pas :
        L = -(1/N) * Σ [ w_pos * y * log(p) + (1 - y) * log(1 - p) ]

Optimisation : descente de gradient stochastique par mini-lots,
avec momentum, régularisation L2 et arrêt précoce (early stopping).
"""

import numpy as np


class BPNN:
    def __init__(self, layer_sizes, activation="relu", lr=0.01, momentum=0.9,
                 l2=1e-4, dropout=0.0, pos_weight=1.0, seed=42):
        """
        layer_sizes : ex. [4096, 128, 32, 1] (entrée, cachées..., sortie=1)
        activation  : 'relu' ou 'sigmoid' pour les couches cachées
        lr          : taux d'apprentissage (learning rate)
        momentum    : inertie de la mise à jour (0 = SGD simple)
        l2          : pénalité L2 sur les poids (évite le surapprentissage)
        dropout     : proportion de neurones cachés éteints à l'entraînement
        pos_weight  : poids des exemples défectueux dans la perte
        """
        assert layer_sizes[-1] == 1, "Sortie binaire : 1 neurone"
        self.sizes = layer_sizes
        self.activation = activation
        self.lr, self.momentum, self.l2 = lr, momentum, l2
        self.dropout, self.pos_weight = dropout, pos_weight
        self.rng = np.random.default_rng(seed)

        # Initialisation des poids :
        #   He (ReLU)     : N(0, 2/n_in)
        #   Xavier (sigm.): N(0, 1/n_in)
        self.W, self.b = [], []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            scale = np.sqrt((2.0 if activation == "relu" else 1.0) / n_in)
            self.W.append(self.rng.normal(0, scale, (n_in, n_out)))
            self.b.append(np.zeros(n_out))
        self.vW = [np.zeros_like(w) for w in self.W]   # vitesses (momentum)
        self.vb = [np.zeros_like(b) for b in self.b]
        self.history = {"train_loss": [], "val_loss": []}

    # ------------------------------------------------------------------
    # Fonctions d'activation et leurs dérivées
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _act(self, z):
        return np.maximum(0, z) if self.activation == "relu" else self._sigmoid(z)

    def _act_deriv(self, z, a):
        if self.activation == "relu":
            return (z > 0).astype(z.dtype)
        return a * (1 - a)

    # ------------------------------------------------------------------
    # 1) Propagation avant
    # ------------------------------------------------------------------
    def forward(self, X, training=False):
        """Calcule la sortie et garde les valeurs intermédiaires (cache)."""
        a = X
        cache = [(None, X, None)]            # (z, a, masque dropout) par couche
        L = len(self.W)
        for l in range(L):
            z = a @ self.W[l] + self.b[l]
            if l == L - 1:                   # couche de sortie : sigmoïde
                a = self._sigmoid(z)
                mask = None
            else:                            # couche cachée
                a = self._act(z)
                mask = None
                if training and self.dropout > 0:   # dropout « inversé »
                    mask = (self.rng.random(a.shape) >= self.dropout) / (1 - self.dropout)
                    a = a * mask
            cache.append((z, a, mask))
        return a.ravel(), cache

    # ------------------------------------------------------------------
    # 2) Perte : entropie croisée binaire pondérée + L2
    # ------------------------------------------------------------------
    def loss(self, p, y):
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        bce = -np.mean(self.pos_weight * y * np.log(p) + (1 - y) * np.log(1 - p))
        reg = 0.5 * self.l2 * sum((w ** 2).sum() for w in self.W)
        return bce + reg

    # ------------------------------------------------------------------
    # 3) Rétropropagation
    # ------------------------------------------------------------------
    def backward(self, y, cache):
        """Retourne les gradients dL/dW et dL/db pour chaque couche.

        Pour la sortie sigmoïde avec entropie croisée pondérée, la dérivée
        par rapport à z (avant la sigmoïde) se simplifie en :
            dL/dz = [ p * (w_pos * y + 1 - y) - w_pos * y ] / N
        (avec w_pos = 1 on retrouve la formule classique p - y).
        """
        N = y.shape[0]
        p = cache[-1][1].ravel()
        delta = ((p * (self.pos_weight * y + 1 - y) - self.pos_weight * y) / N)[:, None]

        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for l in reversed(range(len(self.W))):
            a_prev = cache[l][1]
            gW[l] = a_prev.T @ delta + self.l2 * self.W[l]
            gb[l] = delta.sum(axis=0)
            if l > 0:                                    # propager l'erreur
                z_prev, a_prev_act, mask = cache[l]
                delta = (delta @ self.W[l].T)
                if mask is not None:
                    delta = delta * mask
                    a_prev_act = a_prev_act / np.where(mask == 0, 1, mask)
                delta = delta * self._act_deriv(z_prev, a_prev_act)
        return gW, gb

    # ------------------------------------------------------------------
    # 4) Mise à jour des poids (SGD + momentum)
    # ------------------------------------------------------------------
    def _update(self, gW, gb):
        for l in range(len(self.W)):
            self.vW[l] = self.momentum * self.vW[l] - self.lr * gW[l]
            self.vb[l] = self.momentum * self.vb[l] - self.lr * gb[l]
            self.W[l] += self.vW[l]
            self.b[l] += self.vb[l]

    # ------------------------------------------------------------------
    # Entraînement
    # ------------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None, epochs=200, batch_size=64,
            patience=20, verbose=10):
        """Entraîne le réseau. Arrêt précoce sur la perte de validation."""
        best, best_state, wait = np.inf, None, 0
        n = X.shape[0]
        for ep in range(1, epochs + 1):
            idx = self.rng.permutation(n)
            for s in range(0, n, batch_size):
                bi = idx[s:s + batch_size]
                _, cache = self.forward(X[bi], training=True)
                gW, gb = self.backward(y[bi], cache)
                self._update(gW, gb)

            tr = self.loss(self.predict_proba(X), y)
            self.history["train_loss"].append(tr)
            if X_val is not None:
                va = self.loss(self.predict_proba(X_val), y_val)
                self.history["val_loss"].append(va)
                if va < best - 1e-5:
                    best, wait = va, 0
                    best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b], ep)
                else:
                    wait += 1
                if verbose and ep % verbose == 0:
                    print(f"  époque {ep:4d} | perte train {tr:.4f} | perte val {va:.4f}")
                if wait >= patience:
                    if verbose:
                        print(f"  arrêt précoce à l'époque {ep} (meilleure : {best_state[2]})")
                    break
        if best_state is not None:
            self.W, self.b, self.best_epoch = best_state
        return self

    def predict_proba(self, X):
        return self.forward(X, training=False)[0]

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Vérification du gradient (preuve que la rétropropagation est juste)
    # ------------------------------------------------------------------
    def gradient_check(self, X, y, n_checks=20, eps=1e-5):
        """Compare le gradient analytique au gradient numérique
        (différences finies). Une erreur relative < 1e-6 = implémentation correcte."""
        drop = self.dropout
        self.dropout = 0.0
        _, cache = self.forward(X, training=False)
        gW, _ = self.backward(y, cache)
        errs = []
        for _ in range(n_checks):
            l = self.rng.integers(len(self.W))
            i = self.rng.integers(self.W[l].shape[0])
            j = self.rng.integers(self.W[l].shape[1])
            old = self.W[l][i, j]
            self.W[l][i, j] = old + eps; lp = self.loss(self.predict_proba(X), y)
            self.W[l][i, j] = old - eps; lm = self.loss(self.predict_proba(X), y)
            self.W[l][i, j] = old
            num = (lp - lm) / (2 * eps)
            ana = gW[l][i, j]
            errs.append(abs(num - ana) / max(1e-12, abs(num) + abs(ana)))
        self.dropout = drop
        return float(np.max(errs))
