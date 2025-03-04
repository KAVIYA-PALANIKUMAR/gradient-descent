import numpy as np
import matplotlib.pyplot as plt
import time

class LinearRegressionMSE:
    """ 
    A simple Linear Regression model with Mean Squared Error loss.
    Uses both Gradient Descent and Linear Search for slope optimization.
    """

    def __init__(self, learning_rate=0.01, tolerance=1e-4):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.m1 = np.random.uniform(-15, 15)  # Random initial slope
        self.m2 = 2  # Intercept

    def generate_data(self, num_points=50, noise_std=1.0):
        """
        Generates synthetic linear data with noise.
        :param num_points: Number of data points
        :param noise_std: Standard deviation of noise
        """
        self.x = np.linspace(-5, 5, num_points)
        self.y = 3 * self.x + self.m2 + np.random.normal(0, noise_std, num_points)

    def mse_loss(self, m1):
        """
        Computes Mean Squared Error for a given slope m1.
        :param m1: Slope parameter
        :return: MSE loss value
        """
        return np.mean((m1 * self.x + self.m2 - self.y) ** 2)

    def gradient_descent(self):
        """
        Performs Gradient Descent to optimize the slope (m1).
        """
        prev_loss = float('inf')
        self.gd_steps = []
        start_time = time.time()

        while True:
            gradient = np.mean(2 * self.x * (self.m1 * self.x + self.m2 - self.y))
            new_m1 = self.m1 - self.learning_rate * gradient
            current_loss = self.mse_loss(new_m1)

            if abs(prev_loss - current_loss) < self.tolerance:
                break

            self.gd_steps.append((self.m1, self.mse_loss(self.m1)))  # Log step
            self.m1 = new_m1
            prev_loss = current_loss

        self.gd_time = time.time() - start_time
        return self.m1

    def linear_search(self, m1_range=(-15, 15), num_points=100):
        """
        Performs brute-force Linear Search to find the best m1.
        :param m1_range: Range of m1 values to search
        :param num_points: Number of search points
        """
        m1_values = np.linspace(m1_range[0], m1_range[1], num_points)
        start_time = time.time()
        best_m1 = min(m1_values, key=self.mse_loss)
        self.ls_time = time.time() - start_time
        return best_m1

    def plot_results(self):
        """
        Plots the MSE loss function, Gradient Descent steps, and Linear Search result.
        """
        m1_values = np.linspace(-15, 15, 100)
        loss_values = [self.mse_loss(m) for m in m1_values]

        plt.figure(figsize=(10, 6))
        plt.plot(m1_values, loss_values, 'g-', label="MSE vs m1 (Loss Curve)")
        plt.scatter(*zip(*self.gd_steps), color='blue', s=20, label="Gradient Descent Steps")
        plt.scatter(self.gd_steps[-1][0], self.gd_steps[-1][1], color='red', s=100,
                    label=f"Final m1 (GD): {self.gd_steps[-1][0]:.3f}")
        plt.axvline(self.best_m1_ls, color='purple', linestyle='dashed',
                    label=f"Best m1 (Linear Search): {self.best_m1_ls:.3f}")
        plt.xlabel("m1 (Slope)")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.title("Gradient Descent and Linear Search on Loss Function")
        plt.show()

    def run(self):
        """
        Runs the entire process: data generation, optimization, and visualization.
        """
        self.generate_data()
        self.best_m1_gd = self.gradient_descent()
        self.best_m1_ls = self.linear_search()
        self.plot_results()

        print("\nComparison of Methods:")
        print(f"Gradient Descent Optimized m1: {self.best_m1_gd:.5f} (Time: {self.gd_time:.6f} sec)")
        print(f"Linear Search Best m1: {self.best_m1_ls:.5f} (Time: {self.ls_time:.6f} sec)")

# Run the optimized model
if __name__ == "__main__":
    model = LinearRegressionMSE()
    model.run()
