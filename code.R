library(MASS)
library(np)
library(foreach)
library(doParallel)

# Silence the continuous bandwidth optimization printouts
options(np.messages = FALSE)

# ==========================================
# 1. Data Preparation
# ==========================================
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
data <- read.csv(url, header = FALSE)

colnames(data) <- c("id","diagnosis","radius_mean","texture_mean",
                    "perimeter_mean","area_mean","smoothness_mean",
                    "compactness_mean","concavity_mean",
                    "concave_points_mean","symmetry_mean",
                    "fractal_dimension_mean","radius_se","texture_se",
                    "perimeter_se","area_se","smoothness_se",
                    "compactness_se","concavity_se",
                    "concave_points_se","symmetry_se",
                    "fractal_dimension_se","radius_worst",
                    "texture_worst","perimeter_worst","area_worst",
                    "smoothness_worst","compactness_worst",
                    "concavity_worst","concave_points_worst",
                    "symmetry_worst","fractal_dimension_worst")

# Encode target variable (Malignant = 1, Benign = 0)
Y <- ifelse(data$diagnosis == "M", 1, 0)
X <- data[, 3:32]

# ==========================================
# 2. Train / Test Partitioning (80/20 Split)
# ==========================================
set.seed(496) # Fixed seed for strict reproducibility
train_idx <- sample(1:nrow(X), size = 0.8 * nrow(X))

X_train <- X[train_idx, ]
Y_train <- Y[train_idx]
X_test  <- X[-train_idx, ]
Y_test  <- Y[-train_idx]

# ==========================================
# 3. Stage 1: Parametric Screening (AIC)
# ==========================================
null_model <- glm(Y_train ~ 1, data = X_train, family = binomial)
full_model <- glm(Y_train ~ ., data = X_train, family = binomial)

# Forward-stepwise selection using AIC penalty (2k)
aic_model <- step(null_model,
                  scope = list(lower = null_model, upper = full_model),
                  direction = "forward",
                  trace = 0)

selected_features <- names(coef(aic_model))[-1]

# Regularization: Limit to k=6 to mitigate the curse of dimensionality
# and ensure sufficient local neighborhood density for kernel estimators.
selected_features <- selected_features[1:6]

cat("Optimal subset isolated after Stage 1 (k=6):\n")
print(selected_features)
X_sub_train <- X_train[, selected_features]

# ==========================================
# 4. Stage 2: Nonparametric LOOCV (Local Linear)
# ==========================================
cores <- min(parallel::detectCores() - 1, length(selected_features))
cl <- makeCluster(cores)
registerDoParallel(cl)

# Isolate predictive power of each focus variable via Kernel Smoothing
results <- foreach(focus_var = selected_features, 
                   .combine = rbind, .packages = "np") %dopar% {
                     
                     x_focus <- X_sub_train[[focus_var]]
                     
                     # Regress Y on the focus variable (Local Linear Estimator)
                     bw_out_ll <- npregbw(xdat = x_focus, ydat = Y_train, 
                                          regtype = "ll", bwmethod = "cv.ls")
                     
                     # Extract Out-of-Sample Mean Squared Error
                     mse_ll <- bw_out_ll$fval 
                     
                     data.frame(Variable = focus_var, MSE_LL = mse_ll)
                   }

stopCluster(cl)

# ==========================================
# 5. Variable Ranking & Elimination
# ==========================================
cat("\n--- Predictive Errors (Local Linear LOOCV MSE) ---\n")
print(results)

# Sort by Mean Squared Error (Descending)
ranked_ll <- results[order(-results$MSE_LL), ]

# The variable with the highest error contains the least structural info
var_to_drop_ll <- ranked_ll$Variable[1]

cat("\n--- Nonparametric Feature Elimination ---\n")
cat("Highest Error (Least Info):", var_to_drop_ll, "-> Dropped.\n")

# ==========================================
# 6. Final Validation & Negative Control
# ==========================================
final_features <- setdiff(selected_features, var_to_drop_ll)
dropped_features <- setdiff(colnames(X), final_features)

# Train Parsimonious Model (5 optimal features)
final_model <- glm(Y_train ~ ., data = X_train[, final_features, drop=FALSE], family = binomial)

# Train Negative Control Model (25 discarded features)
# Note: This model is expected to throw convergence warnings due to multicollinearity
control_model <- glm(Y_train ~ ., data = X_train[, dropped_features, drop=FALSE], family = binomial)

# Predict classifications on unseen testing data
pred_final_prob <- predict(final_model, newdata = X_test[, final_features, drop=FALSE], type = "response")
pred_final_class <- ifelse(pred_final_prob > 0.5, 1, 0)

pred_control_prob <- predict(control_model, newdata = X_test[, dropped_features, drop=FALSE], type = "response")
pred_control_class <- ifelse(pred_control_prob > 0.5, 1, 0)

# Evaluate global test accuracy
acc_final <- mean(pred_final_class == Y_test)
acc_control <- mean(pred_control_class == Y_test)

cat("\n--- Final Testing Accuracy (Hold-out Set) ---\n")
cat(sprintf("Parsimonious Model (%d Variables): %.2f%%\n", length(final_features), acc_final * 100))
cat(sprintf("Negative Control (%d Variables): %.2f%%\n", length(dropped_features), acc_control * 100))