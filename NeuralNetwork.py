import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # weights initialization
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights_hidden_output) + self.bias_output
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y_true, y_pred):
        y_true = y_true.reshape(-1, 1)
        # output layer error calc
        output_error = y_true - y_pred
        output_delta = output_error * self.sigmoid_derivative(y_pred)

        # hidden layer error calc
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        # weight and bias updates
        self.weights_hidden_output += self.learning_rate * np.dot(self.a1.T, output_delta)
        self.bias_output += self.learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += self.learning_rate * np.dot(X.T, hidden_delta)
        self.bias_hidden += self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, y_pred)
            # if epoch % 100 == 0:
            #     loss = self.compute_loss(y, y_pred)
            #     print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.round(y_pred).ravel()
