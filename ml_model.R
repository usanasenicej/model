# Load necessary libraries
library(caret)
library(ggplot2)

# Load the built-in iris dataset
data(iris)

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
  geom_text(aes(label = Freq), vjust = 0.5, size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(title = "Confusion Matrix Heatmap",
       subtitle = "Random Forest Model Predictor",
       x = "Actual Species",
       y = "Predicted Species",
       fill = "Frequency")

# Save the plot to a file
ggsave("confusion_matrix.png", plot = plot_cm, width = 6, height = 4)
cat("Visualization saved as confusion_matrix.png in the current directory.\n")
