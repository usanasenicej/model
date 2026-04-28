# Load necessary libraries
library(caret)
library(ggplot2)
library(GGally) # For pair plots
library(randomForest)

# Load the built-in iris dataset
data(iris)
cat("Dataset loaded: iris\n")

# 0. Exploratory Data Analysis (Visualizing the data)
cat("Generating EDA plots...\n")
plot_pairs <- ggpairs(iris, aes(color = Species, alpha = 0.5)) +
  theme_minimal() +
  labs(title = "Iris Dataset Pairwise Relationships")
ggsave("iris_pairs.png", plot = plot_pairs, width = 10, height = 8)

# Set seed for reproducibility
set.seed(42)

# 1. Split the data (80% training, 20% testing)
trainIndex <- createDataPartition(iris$Species, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
dataTrain <- iris[ trainIndex,]
dataTest  <- iris[-trainIndex,]

# 2. Define training control (5-fold cross-validation)
fitControl <- trainControl(method = "cv", number = 5)

# 3. Train a Random Forest model using caret
cat("Training Random Forest model...\n")
rf_model <- train(Species ~ ., 
                  data = dataTrain, 
                  method = "rf", 
                  trControl = fitControl)

# Print the model summary
print(rf_model)

# 4. Make predictions on the testing set
predictions <- predict(rf_model, newdata = dataTest)

# Evaluate the model
cm <- confusionMatrix(predictions, dataTest$Species)
print(cm)

# 5. Visualize the Confusion Matrix using ggplot2
# Convert confusion matrix table to data frame
cm_df <- as.data.frame(cm$table)

# Create a heatmap of the confusion matrix
plot_cm <- ggplot(data = cm_df, aes(x = Reference, y = Prediction)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  geom_text(aes(label = Freq), vjust = 0.5, size = 6, fontface = "bold") +
  scale_fill_gradient(low = "#f7fbff", high = "#084594") +
  theme_minimal(base_size = 14) +
  labs(title = "Confusion Matrix Heatmap",
       subtitle = "Random Forest Model Performance",
       x = "Actual Species",
       y = "Predicted Species",
       fill = "Record Count")

# Save the plot to a file
ggsave("confusion_matrix.png", plot = plot_cm, width = 7, height = 5)

# 6. Feature Importance Visualization
cat("Calculating feature importance...\n")
importance <- varImp(rf_model, scale = FALSE)
plot_imp <- ggplot(importance) +
  theme_minimal() +
  labs(title = "Feature Importance (Random Forest)",
       x = "Variable",
       y = "Importance Score") +
  geom_bar(stat = "identity", fill = "steelblue")
ggsave("feature_importance.png", plot = plot_imp, width = 7, height = 5)

cat("Success! All visualizations saved: iris_pairs.png, confusion_matrix.png, feature_importance.png.\n")
