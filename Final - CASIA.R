library(leaps)

# Example 1 Subset selection -----
n <- 50
p <- 15
set.seed(1)
X <- matrix(rnorm(n*p), n, p)
beta <- rep(1, p)
mu <- scale(X %*% beta) * 7

re <- 1000
result <- matrix(0, re, p)
for(r in 1:re){
  y <- mu + rnorm(n)
  reg <- regsubsets(X, y, nvmax = p, intercept = F)
  for(i in 1:p){
    betahat <- coef(reg, i)
    yhat <- X[, which(letters %in% names(betahat)), drop = F] %*% betahat
    result[r, i] <- sum((y - mu) * yhat)
  }
  print(r)
}
plot(colMeans(result), type = 'o', ylab = 'Degrees of Freedom', xlab = 'Subset Size', main = 'Best Subset Regression')
abline(0, 1, lty = 2)
abline(h = 15, lty = 2)

re <- 1000
result <- matrix(0, re, p)
for(r in 1:re){
  y <- mu + rnorm(n)
  reg <- regsubsets(X, y, nvmax = p, intercept = F, method = 'forward')
  for(i in 1:p){
    betahat <- coef(reg, i)
    yhat <- X[, which(letters %in% names(betahat)), drop = F] %*% betahat
    result[r, i] <- sum((y - mu) * yhat)
  }
  print(r)
}
plot(colMeans(result), type = 'o', ylab = 'Degrees of Freedom', xlab = 'Subset Size', main = 'Forward Stepwise Regression')
abline(0, 1, lty = 2)
abline(h = 15, lty = 2)

re <- 1000
result <- matrix(0, re, p)
for(r in 1:re){
  y <- mu + rnorm(n)
  reg <- regsubsets(X, y, nvmax = p, intercept = F, method = 'backward')
  for(i in 1:p){
    betahat <- coef(reg, i)
    yhat <- X[, which(letters %in% names(betahat)), drop = F] %*% betahat
    result[r, i] <- sum((y - mu) * yhat)
  }
  print(r)
}
plot(colMeans(result), type = 'o', ylab = 'Degrees of Freedom', xlab = 'Subset Size', main = 'Backward Stepwise Regression')
abline(0, 1, lty = 2)
abline(h = 15, lty = 2)


# Example 2 Lasso -----
n <- 2
p <- 2
re <- 1000
X <- matrix(c(0, 2, 1, -5), n, p)
beta <- c(-6, -1)
y <- matrix(0, 2, re)
l1 <- seq(0, 6, 0.05) # l1 constraint
path <- matrix(0, p, length(l1))
yhat <- array(0, dim = c(n, length(l1), re))

set.seed(1)
for(r in 1:re){
  y[, r] <- X %*% beta + rnorm(n) * 0.03
  reg <- glmnet(X, y[, r], lambda = exp(seq(-5, 3, 0.001)), intercept = F, standardize = F)
  path[1, ] <- approx(colSums(abs(reg$beta)), reg$beta[1, ], xout = l1)$y # interpolation
  path[2, ] <- approx(colSums(abs(reg$beta)), reg$beta[2, ], xout = l1)$y
  yhat[, , r] <- X %*% path
  print(r)
}
plot(l1, path[1, ], type = 'l', ylim = c(-6, 1), xlab = 'L1 constraint', ylab = 'Coefficient', main = 'Lasso Solution path')
lines(l1, path[2, ])
abline(h = 0, lty = 2)
abline(v = l1[68], lty = 2)
abline(v = l1[69], lty = 2)

df <- matrix(0, length(l1), 10)
for(l in 1:length(l1)){
  for(i in 1:10){
  ind <- ((i-1)*100+1):(i*100)
  df[l, i] <- (1/0.03^2) * (cov(y[1, ind], yhat[1, l, ind]) + cov(y[2, ind], yhat[2, l, ind]))
  }
}
df <- rowMeans(df)
plot(l1, df, type = 'l', xlab = 'L1 constraint', ylab = 'Degrees of Freedom', main = 'Lasso Regression')
abline(v = l1[68], lty = 2)
abline(v = l1[69], lty = 2)


# Example 3 Unbounded ----
A <- 1e4
X <- diag(A, 2)
beta <- c(1, 1)
re <- 1e5
set.seed(1)
y <- A + matrix(rnorm(2*re), re, 2)
yhat <- t(apply(y, 1, function(x) x*(x > mean(x))))
df <- mean(apply(((y - A) * yhat), 1, sum))
std <- sqrt(var(apply(((y - A) * yhat), 1, sum)) / B)
c(df, std)
