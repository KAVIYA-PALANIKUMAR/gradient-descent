# gradient-descent
📌 Overview

This project implements Gradient Descent and Linear Search (Brute Force) to find the slope (m1) in a linear equation( y=m1x + m2 + N(0,1) ). It compares both approaches in terms of efficiency and accuracy while visualizing the Mean Squared Error (MSE) landscape.

🎯 Objective

Generate synthetic data with a linear relationship.

Optimize the computational cost to find slope (m1).

The methods used:

1. Gradient Descent (iterative approach)

2. Linear Search (brute force approach)

Compare execution time and accuracy.

Visualize the optimization process using a loss curve.


📜 Algorithm Breakdown

🔹 1. Data Generation

x: Linearly spaced values.
m1 (true slope) and m2 (intercept) define the ground truth.

Random noise is added to simulate real-world variations.

🔹 2. Mean Squared Error (MSE)

The loss function used is:

MSE = (1/n) * Σ (m1 * x + m2 - y)²
This quantifies the error between predicted and actual values.

🔹 3. Gradient Descent Optimization

Starts with a random slope (m1).

Iteratively updates m1 using the formula:

where is the gradient and is the learning rate.

Stops when the loss change is below a threshold (tolerance).

Logged steps are plotted to visualize the descent path.

🔹 4. Linear Search (Brute Force) Optimization

Evaluates MSE at multiple slope values.

Selects the best m1 based on minimum loss.

This is exhaustive but guarantees an optimal result.

🔹 5. Performance Comparison

Execution time of both methods is measured.

The final slopes (m1) from both methods are compared.

A graph is generated to visually depict the loss curve and the Gradient descent steps

📊 Visualization & Results

The graphs provide insight into the optimization process:

Loss Curve: Shows how MSE varies with m1.

Gradient Descent Steps: Highlights the iterative updates.

Best Slope Comparison: Indicates where each method converges.

