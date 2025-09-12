import numpy as np

np.random.seed(42)
N = 100  # sample size
X = np.random.randn(N, 2)  # n row and 2 colomn
y = (X[:, 0] * X[:, 1] > 0).astype(int)  # labels if the x1 and x2 are same signs

input_size = 2
hidden_size = 4
output_size = 1

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# 3. training loop
lr = 0.1
for epoch in range(1000):
    # forward propagation
    z1 = X.dot(W1) + b1          # (N × 4) twoinput times weight (each row of data time each colomn of wight one and two for four hidden layer neuron)plus bias
    a1 = relu(z1)                # activation
    z2 = a1.dot(W2) + b2         # (N × 1)
    y_hat = sigmoid(z2)          # predicted probability

    # loss function: Binary Cross-Entropy
    loss = -np.mean(y * np.log(y_hat + 1e-9) + (1 - y) * np.log(1 - y_hat + 1e-9))#

    # backward propagation
    dz2 = y_hat - y.reshape(-1, 1)       # (N × 1) dl/dy= -(y/y_hat - (1-y)/(1-y_hat)) , y= sigmoid(z)-> dy/dz2 = sigmoid(z)*(1-sigmoid(z)) = y(1-y) , dl/dz= y_hat - y
    dW2 = a1.T.dot(dz2) / N              # (4 × 1) dl/dw2= dl/dz2 * dz2/dw2, dz2/dw2 = a1 (.T just to match the dimension) (N × 1) z=wa+b, dz/dw = a1
    db2 = np.mean(dz2, axis=0, keepdims=True) # dz/db= 1, dl/db= dl/dz * dz/db = dl/dz2 find the mean of all samples

    da1 = dz2.dot(W2.T) # dl/da1= dl/dz2 * dz2/da1, dz2/da1 = w2.T (.T just to match the dimension) (N × 4) z=wa+b, dz/da = w
    dz1 = da1 * relu_derivative(z1)      # (N × 4) use to see which neuron is activated
    dW1 = X.T.dot(dz1) / N # (2 × 4) dl/dw1= dl/dz1 * dz1/dw1, dz1/dw1 = a0 (.T just to match the dimension) (N × 4) z=wa+b, dz/dw = a0
    db1 = np.mean(dz1, axis=0, keepdims=True) # dz/db= 1, dl/db= dl/dz * dz/db = dl/dz2 find the mean of all samples

    # update weights and biases
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 100 == 0:
        preds = (y_hat > 0.5).astype(int).flatten()
        acc = np.mean(preds == y)
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Acc: {acc:.2f}")
