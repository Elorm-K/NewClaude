"""
Ode to Linear Regression
========================

In the garden of machine learning's art,
There lives a class both simple and smart.
No forests of trees, no layers deep,
Just a line drawn true — a promise to keep.

  y = mx + b

Where m is the slope, the weight of the clue,
And b is the bias, the starting-point view.
We feed it our data, a scattering of dots,
And it seeks out the line that connects all the spots.

It learns by the gradient, descending with care,
Each step a correction, each update laid bare.
The loss function whispers, "you're off by this much,"
And the weights shift in answer, refined to the touch.

No activation, no softmax in sight,
Just the purest of math bathed in linear light.
It won't fit a circle, or spiral, or curve,
But for problems kept simple, it never will swerve.

So raise up a toast to this humble old class,
The first that we teach and the first that we pass.
For every deep network that towers and gleams
Was built on the shoulders of linear dreams.

      -- A tribute to LinearRegression,
         the gentle giant of machine learning.
"""

import numpy as np


class LinearRegression:
    """A simple Linear Regression model trained with gradient descent."""

    def __init__(self, learning_rate: float = 0.01, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            y_pred = X @ self.weights + self.bias
            error = y_pred - y

            # Gradient descent update
            self.weights -= self.learning_rate * (2 / n_samples) * (X.T @ error)
            self.bias -= self.learning_rate * (2 / n_samples) * error.sum()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


if __name__ == "__main__":
    # A tiny demo — fitting a line to noisy data
    np.random.seed(42)
    X = np.random.rand(100, 1)
    y = 3 * X.squeeze() + 2 + np.random.randn(100) * 0.1

    model = LinearRegression(learning_rate=0.1, epochs=500).fit(X, y)
    print(f"Learned weight : {model.weights[0]:.4f}  (true: 3.0)")
    print(f"Learned bias   : {model.bias:.4f}  (true: 2.0)")
