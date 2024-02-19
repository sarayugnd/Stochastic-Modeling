
#EXERCISE 2 - PART 1
# Set up the trend function 
trend <- function(x, a0, a1, a2) {
  a0 + a1 * x + a2 * x^2
}

# Define the parameters
a0 <- 5  
a1 <- 0.5
a2 <- -0.1

# Generate observation times
n <- 100
x <- seq(-5, 5, length.out = n)

# Simulate the trend 
trend_vals <- trend(x, a0, a1, a2)

# Simulate the GP residuals
set.seed(123)
resid <- rnorm(n, 0, 1) 

#final GP
y <- trend_vals + resid

# Plot the simulated GP
plot(x, y, type = "l", col = "blue", xlab = "X", ylab = "Y", main = "Simulated GP with quadratic trend")
lines(x, trend_vals, col = "red", lwd = 2)


#EXERCISE 2 - PART 2 (using OLS)
# Simulate some data
n <- 100 
x <- seq(-5, 5, length.out = n)
K <- MASS::mvrnorm(n, rep(0, n), exp(-0.5 * dist(x)^2)) # GP covariance
y <- K %*% rnorm(n) # GP realisation

# OLS fitting
X <- matrix(1, nrow = n, ncol = 1) # No trend
a <- solve(t(X) %*% X) %*% t(X) %*% y
resid <- y - X %*% a

# Get fitted values
yhat <- X %*% a 

# Plot data and fitted GP  
plot(x, y, xlab = "X", ylab = "Y", main ="Fitted Gaussian process model (no trend)")
lines(x, yhat, col = "red")


#EXERCISE 2 - PART 3
# Leave-One-Out CV
err <- rep(NA, n)
for (i in 1:n) {
  # Leave out ith observation
  y_out <- y[-i]
  X_out <- X[-i,]
  
  # Refit on remaining data
  a_out <- solve(t(X_out) %*% X_out) %*% t(X_out) %*% y_out
  
  # Predict left-out observation
  yhat_i <- X[i,] %*% a_out
  
  # Calculate squared error
  err[i] <- (y[i] - yhat_i)^2 
}

# Estimate prediction error
cat("LOOCV Prediction Error (No Trend):", mean(err), "\n")


#EXERCISE 2 - PART 4 (Linear trend)
# Simulate data with linear trend 
y <- 0.5 * x + rnorm(n, 0, 1)

# Leave-One-Out CV
X <- cbind(1, x) # Add linear trend
err <- rep(NA, n)

for (i in 1:n) {
  # Leave out ith observation
  y_out <- y[-i]
  X_out <- X[-i,]
  
  # Refit 
  a_out <- solve(t(X_out) %*% X_out) %*% t(X_out) %*% y_out
  
  # Predict
  yhat_i <- X[i,] %*% a_out
  
  # Error
  err[i] <- (y[i] - yhat_i)^2
}

cat("LOOCV Prediction Error (Linear Trend):", mean(err), "\n")

# Plot data with linear trend
plot(x, y, xlab = "X", ylab = "Y", main = "Gaussian Process Model with linear trend")
legend("topright", legend = c("Observed", "Predicted"), col = c("blue", "red"), lwd = 2)
abline(a_out[1], a_out[2], col = "red", lwd = 2)

# Plot LOOCV errors for linear trend
plot(1:n, err, type = "p", pch = 16, col = "red",
     xlab = "Data", ylab = "LOO Prediction Error",
     main = "LOO Errors for Linear Trend")


#EXERCISE 2 - PART 4 (Quadratic trend)
# Simulate data with quadratic trend
y <- 0.5 * x + 0.1 * x^2 + rnorm(n, 0, 1) 

# Leave-One-Out CV 
X <- cbind(1, x, x^2) # Add quadratic term
err <- rep(NA, n)

for (i in 1:n) {
  # Leave out ith observation
  y_out <- y[-i]
  X_out <- X[-i,]
  
  # Refit 
  a_out <- solve(t(X_out) %*% X_out) %*% t(X_out) %*% y_out
  
  # Predict
  yhat_i <- X[i,] %*% a_out
  
  # Error
  err[i] <- (y[i] - yhat_i)^2
}

cat("LOOCV Prediction Error (Quadratic Trend):", mean(err), "\n")

# Plot data with quadratic trend
plot(x, y, xlab = "X", ylab = "Y", main = "Gaussian Process Model with quadratic trend")
lines(x, trend(x, a_out[1], a_out[2], a_out[3]), col = "red", lwd = 2)
legend("topright", legend = c("Observed", "Predicted"), col = c("blue", "red"), lwd = 2)

# Plot LOOCV errors for quadratic trend
plot(1:n, err, type = "p", pch = 16, col = "red",
     xlab = "Data", ylab = "LOO Prediction Error",
     main = "LOOCV Errors for Quadratic Trend")

