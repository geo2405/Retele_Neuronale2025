
import pickle, csv
import numpy as np


def load_train_from_tuple_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    X_list, y_list = [], []
    for sample in data:
        x_i, y_i = sample[0], sample[1]
        x_i = np.asarray(x_i).reshape(-1)      # 28×28 -> 784
        X_list.append(x_i)
        y_list.append(int(y_i))

    X = np.stack(X_list).astype(np.float32)    # (N,784)
    y = np.array(y_list, dtype=np.int64)       # (N,)
    return X, y


def load_test_from_tuple_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    X_list = []
    for sample in data:
        x_i = sample[0] if isinstance(sample, (tuple, list)) else sample
        x_i = np.asarray(x_i).reshape(-1)
        X_list.append(x_i)

    X = np.stack(X_list).astype(np.float32)    # (nr imagini,784)
    return X


def softmax(Z): #din brut--> probabilitati
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True) #aplic formula

def one_hot(y, num_classes=10):
    oh = np.zeros((y.size, num_classes), dtype=np.float32)
    oh[np.arange(y.size), y] = 1.0
    return oh



def train_minibatch(X, y, num_classes=10, lr=0.1, epochs=20, batch_size=128,
                    l2=1e-4, lr_decay=0.95, seed=42):

    rng = np.random.default_rng(seed)
    N, D = X.shape
    C = num_classes

    # inițializare mică
    W = (rng.standard_normal((D, C)) * 0.01).astype(np.float32)
    b = np.zeros(C, dtype=np.float32)

    Xn = X / 255.0
    T  = one_hot(y, C)

    for ep in range(epochs):
        idx = rng.permutation(N)
        Xs, Ts = Xn[idx], T[idx] #rearanjez imaginile si etichetele in oridinea random idx

        for start in range(0, N, batch_size):
            xb = Xs[start:start+batch_size]
            tb = Ts[start:start+batch_size]
            m  = xb.shape[0] #(128, 784), m ia 128

            # forward
            z  = xb @ W + b # fiecare rand = img, fiecare col=eticheta
            yh = softmax(z) #val prezisa

            # gradient
            dZ  = (yh - tb) / m
            dW  = xb.T @ dZ + l2 * W
            db  = dZ.sum(axis=0)

            W -= lr * dW
            b -= lr * db

        # mic feedback pe train
        pred = softmax(Xn @ W + b).argmax(axis=1)
        acc  = (pred == y).mean()
        print(f"Epoch {ep+1}/{epochs}  acc(train)={acc:.4f}  lr={lr:.4f}")

        lr *= lr_decay

    return W, b


def predict_and_write_csv(X_test, W, b, out_csv="submission.csv"):
    Xn = X_test / 255.0
    z  = Xn @ W + b
    pred = softmax(z).argmax(axis=1)


    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for i, p in enumerate(pred):
            writer.writerow([i, int(p)])

    print(f"Am scris {out_csv}")


def main():
    train_pkl = "extended_mnist_train.pkl"
    test_pkl  = "extended_mnist_test.pkl"

    print("Încarc train...")
    X_train, y_train = load_train_from_tuple_pkl(train_pkl)
    print("X_train:", X_train.shape, " y_train:", y_train.shape)

    print("Încarc test...")
    X_test = load_test_from_tuple_pkl(test_pkl)
    print("X_test:", X_test.shape)

    print("Antrenez (mini-batch, L2, lr-decay)...")
    W, b = train_minibatch(
        X_train, y_train,
        num_classes=10,
        lr=0.1, epochs=20, batch_size=128,
        l2=1e-4, lr_decay=0.95, seed=42
    )

    print("Prezic pe test și scriu CSV...")
    predict_and_write_csv(X_test, W, b, out_csv="submission.csv")

if __name__ == "_main_":
    main()