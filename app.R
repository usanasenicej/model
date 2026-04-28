library(shiny)
library(bslib)
library(caret)
library(ggplot2)
library(randomForest)
library(GGally)
library(DT)
library(shinycssloaders)

# Load data
data(iris)

# Set seed for reproducibility
set.seed(42)

# Pre-train the model once for the app
cat("Pre-training model for the application...\n")
trainIndex <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
dataTrain <- iris[trainIndex,]
dataTest  <- iris[-trainIndex,]
fitControl <- trainControl(method = "cv", number = 5)
rf_model <- train(Species ~ ., data = dataTrain, method = "rf", trControl = fitControl)

# UI Definition
ui <- page_navbar(
  title = "Iris Species Intelligence",
  theme = bs_theme(
    version = 5,
    bootswatch = "lux",
    primary = "#1a1a1a",
    secondary = "#6c757d",
    base_font = font_google("Inter"),
    heading_font = font_google("Outfit")
  ),
  
  nav_panel(
    title = "Prediction Lab",
    layout_sidebar(
      sidebar = sidebar(
        title = "Measurements (cm)",
        width = 300,
        sliderInput("sepal_l", "Sepal Length", min = 4.3, max = 7.9, value = 5.8, step = 0.1),
        sliderInput("sepal_w", "Sepal Width", min = 2.0, max = 4.4, value = 3.0, step = 0.1),
        sliderInput("petal_l", "Petal Length", min = 1.0, max = 6.9, value = 4.3, step = 0.1),
        sliderInput("petal_w", "Petal Width", min = 0.1, max = 2.5, value = 1.3, step = 0.1),
        hr(),
        actionButton("predict_btn", "Run Prediction", class = "btn-primary w-100")
      ),
      card(
        card_header("Species Prediction Result"),
        layout_column_wrap(
          width = 1/2,
          value_box(
            title = "Predicted Species",
            value = textOutput("pred_class"),
            showcase = bsicons::bs_icon("flower1"),
            theme = "primary"
          ),
          value_box(
            title = "Confidence Score",
            value = textOutput("pred_prob"),
            showcase = bsicons::bs_icon("percent"),
            theme = "secondary"
          )
        )
      ),
      card(
        card_header("Feature Contribution"),
        plotOutput("pred_plot") %>% withSpinner(color="#1a1a1a")
      )
    )
  ),
  
  nav_panel(
    title = "Visual Analytics",
    layout_column_wrap(
      width = 1,
      card(
        card_header("Dataset Relationships (EDA)"),
        plotOutput("pair_plot", height = "600px") %>% withSpinner(color="#1a1a1a")
      ),
      card(
        card_header("Feature Importance"),
        plotOutput("imp_plot") %>% withSpinner(color="#1a1a1a")
      )
    )
  ),
  
  nav_panel(
    title = "Model Diagnostics",
    layout_column_wrap(
      width = 1/2,
      card(
        card_header("Confusion Matrix"),
        plotOutput("cm_plot") %>% withSpinner(color="#1a1a1a")
      ),
      card(
        card_header("Accuracy Metrics"),
        verbatimTextOutput("model_summary")
      )
    )
  ),
  
  nav_panel(
    title = "Raw Data",
    card(
      card_header("Iris Dataset Explorer"),
      DTOutput("iris_table")
    )
  )
)

# Server Logic
server <- function(input, output, session) {
  
  # Reactive prediction
  prediction <- eventReactive(input$predict_btn, {
    new_data <- data.frame(
      Sepal.Length = input$sepal_l,
      Sepal.Width = input$sepal_w,
      Petal.Length = input$petal_l,
      Petal.Width = input$petal_w
    )
    
    prob <- predict(rf_model, new_data, type = "prob")
    class <- predict(rf_model, new_data)
    
    list(class = class, prob = prob)
  }, ignoreNULL = FALSE)
  
  output$pred_class <- renderText({
    as.character(prediction()$class)
  })
  
  output$pred_prob <- renderText({
    p <- prediction()$prob
    paste0(round(max(p) * 100, 1), "%")
  })
  
  output$pred_plot <- renderPlot({
    p_data <- as.data.frame(t(prediction()$prob))
    colnames(p_data) <- "Probability"
    p_data$Species <- rownames(p_data)
    
    ggplot(p_data, aes(x = Species, y = Probability, fill = Species)) +
      geom_bar(stat = "identity") +
      scale_fill_manual(values = c("#1a1a1a", "#4a4a4a", "#8a8a8a")) +
      theme_minimal() +
      ylim(0, 1) +
      labs(title = "Probability Distribution", y = "Probability", x = NULL)
  })
  
  output$pair_plot <- renderPlot({
    ggpairs(iris, aes(color = Species, alpha = 0.5)) +
      theme_minimal() +
      scale_color_manual(values = c("#1a1a1a", "#4a4a4a", "#8a8a8a")) +
      scale_fill_manual(values = c("#1a1a1a", "#4a4a4a", "#8a8a8a"))
  })
  
  output$imp_plot <- renderPlot({
    imp <- varImp(rf_model, scale = FALSE)
    ggplot(imp) +
      geom_bar(stat = "identity", fill = "#1a1a1a") +
      theme_minimal() +
      labs(title = "Relative Feature Importance", x = "Variable", y = "Score")
  })
  
  output$cm_plot <- renderPlot({
    preds <- predict(rf_model, dataTest)
    cm <- confusionMatrix(preds, dataTest$Species)
    cm_df <- as.data.frame(cm$table)
    
    ggplot(cm_df, aes(Reference, Prediction, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "white", size = 8) +
      scale_fill_gradient(low = "#6c757d", high = "#1a1a1a") +
      theme_minimal() +
      labs(title = "Testing Set Confusion Matrix")
  })
  
  output$model_summary <- renderPrint({
    rf_model
  })
  
  output$iris_table <- renderDT({
    datatable(iris, options = list(pageLength = 10))
  })
}

shinyApp(ui, server)
