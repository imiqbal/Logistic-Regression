#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
# Logistic Regression Web App for Binary and Multinomial Classification
# Developed by M.Iqbal Jeelani (SKUAST-Kashmir)
# Contact: imstat09@gmail.com
# YouTube: https://www.youtube.com/@Iqbalstat

# Load required libraries
library(shiny)
library(shinydashboard)
library(DT)
library(ggplot2)
library(dplyr)
library(nnet)
library(caret)
library(readr)
library(tidyr)
library(plotly)
library(pROC) # For ROC curve

# UI
ui <- dashboardPage(
  dashboardHeader(title = "Logistic Regression"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Data", tabName = "data", icon = icon("database")),
      menuItem("Preprocessing", tabName = "preprocessing", icon = icon("filter")),
      menuItem("Model", tabName = "model", icon = icon("cogs")),
      menuItem("Results", tabName = "results", icon = icon("chart-bar")),
      menuItem("Prediction", tabName = "prediction", icon = icon("search")),
      menuItem("About", tabName = "about", icon = icon("info-circle"))
    )
  ),
  dashboardBody(
    tabItems(
      # Data tab
      tabItem(tabName = "data",
              fluidRow(
                box(
                  title = "Data Input", status = "primary", solidHeader = TRUE, width = 12,
                  tabsetPanel(
                    tabPanel("Upload Data",
                             fileInput("file", "Choose CSV File (Max 5000 rows)", accept = c(".csv")),
                             checkboxInput("header", "Header", TRUE),
                             radioButtons("sep", "Separator", choices = c(Comma = ",", Semicolon = ";", Tab = "\t"), selected = ","),
                             actionButton("loadDemo", "Load Demo Data (Iris)")
                    ),
                    tabPanel("Data Preview",
                             DT::dataTableOutput("dataPreview")
                    ),
                    tabPanel("Data Summary",
                             verbatimTextOutput("dataSummary")
                    ),
                    tabPanel("Data Structure",
                             verbatimTextOutput("dataStructure")
                    )
                  )
                )
              )
      ),
      
      # Preprocessing tab
      tabItem(tabName = "preprocessing",
              fluidRow(
                box(
                  title = "Target Variable Selection", status = "primary", solidHeader = TRUE, width = 12,
                  selectInput("targetVar", "Select Target Variable", choices = NULL),
                  radioButtons("targetType", "Target Type", choices = c("Binary" = "binary", "Multinomial" = "multinomial"), selected = "binary"),
                  verbatimTextOutput("targetDistribution"),
                  plotlyOutput("targetDistPlot")
                )
              ),
              fluidRow(
                box(
                  title = "Feature Selection", status = "primary", solidHeader = TRUE, width = 12,
                  checkboxGroupInput("features", "Select Features", choices = NULL),
                  actionButton("selectAll", "Select All"),
                  actionButton("deselectAll", "Deselect All"),
                  actionButton("preprocess", "Apply Preprocessing")
                )
              ),
              fluidRow(
                box(
                  title = "Train/Test Split", status = "primary", solidHeader = TRUE, width = 12,
                  sliderInput("trainSplit", "Training Data Percentage", min = 50, max = 90, value = 70),
                  checkboxInput("randomSeed", "Set Random Seed", TRUE),
                  numericInput("seedValue", "Seed Value", 123, min = 1, max = 999999)
                )
              )
      ),
      
      # Model tab
      tabItem(tabName = "model",
              fluidRow(
                box(
                  title = "Model Settings", status = "primary", solidHeader = TRUE, width = 12,
                  numericInput("cvFolds", "Cross-Validation Folds", 3, min = 2, max = 5),
                  checkboxInput("standardize", "Standardize Features", TRUE),
                  actionButton("trainModel", "Train Model")
                )
              ),
              fluidRow(
                box(
                  title = "Model Formula", status = "warning", solidHeader = TRUE, width = 12,
                  verbatimTextOutput("formula")
                )
              ),
              fluidRow(
                box(
                  title = "Training Status", status = "warning", solidHeader = TRUE, width = 12,
                  verbatimTextOutput("trainingStatus")
                )
              )
      ),
      
      # Results tab
      tabItem(tabName = "results",
              fluidRow(
                tabBox(
                  title = "Model Results", width = 12,
                  tabPanel("Summary", verbatimTextOutput("modelSummary")),
                  tabPanel("Coefficients", DT::dataTableOutput("coefficients")),
                  tabPanel("Performance Metrics", verbatimTextOutput("performanceMetrics")),
                  tabPanel("Confusion Matrix", verbatimTextOutput("confusionMatrix"), plotOutput("confusionMatrixPlot")),
                  tabPanel("Class Performance", plotOutput("classPerformancePlot"))
                )
              ),
              fluidRow(
                box(
                  title = "Feature Importance", status = "primary", solidHeader = TRUE, width = 12,
                  plotOutput("featureImportance")
                )
              )
      ),
      
      # Prediction tab
      tabItem(tabName = "prediction",
              fluidRow(
                box(
                  title = "Input Values for Prediction", status = "primary", solidHeader = TRUE, width = 12,
                  uiOutput("predictionInputs"),
                  actionButton("predictButton", "Predict")
                )
              ),
              fluidRow(
                box(
                  title = "Prediction Results", status = "success", solidHeader = TRUE, width = 12,
                  verbatimTextOutput("predictionResult"),
                  plotOutput("predictionProbPlot")
                )
              )
      ),
      
      # About tab
      tabItem(tabName = "about",
              fluidRow(
                box(
                  title = "About This Application", status = "info", solidHeader = TRUE, width = 12,
                  tags$div(
                    tags$h3("Logistic Regression Shiny App"),
                    tags$p("This Shiny app provides an intuitive interface for performing logistic regression analysis on both binary and multinomial outcomes. It includes essential tools like confusion matrix, class-wise performance, feature importance, and model validation through train/test split and cross-validation. Logistic regression is especially valuable in forestry and agriculture, where it's used to predict outcomes like disease presence in trees or crop success based on environmental and management factors. By estimating probabilities rather than just yes/no answers, it supports better decision-making in real-world scenarios."),
                    tags$br(),
                    tags$h4("Developer Information:"),
                    tags$p(tags$strong("Developed by:"), "M.Iqbal Jeelani (SKUAST-Kashmir)"),
                    tags$p(tags$strong("Contact:"), tags$a("imstat09@gmail.com", href="mailto:imstat09@gmail.com")),
                    tags$p(tags$strong("YouTube Channel:"), tags$a("https://www.youtube.com/@Iqbalstat", href="https://www.youtube.com/@Iqbalstat", target="_blank")),
                    tags$br(),
                    tags$p("Follow the YouTube channel for more statistical tutorials and applications."),
                    tags$hr(),
                    tags$h4("How to use this app:"),
                    tags$ol(
                      tags$li("Upload your data or use the demo dataset"),
                      tags$li("Select your target variable and features"),
                      tags$li("Configure model settings and train your model"),
                      tags$li("Explore results and make predictions")
                    )
                  )
                )
              )
      )
    )
  ),
  tags$footer(
    tags$div(
      style = "text-align: center; padding: 10px; background-color: #f5f5f5; border-top: 1px solid #ddd;",
      "Developed by M.Iqbal Jeelani (SKUAST-Kashmir) | Contact: ",
      tags$a("imstat09@gmail.com", href="mailto:imstat09@gmail.com"),
      " | YouTube: ",
      tags$a("@Iqbalstat", href="https://www.youtube.com/@Iqbalstat", target="_blank")
    )
  )
)

# Server
server <- function(input, output, session) {
  # Reactive values
  values <- reactiveValues(
    data = NULL,
    processedData = NULL,
    train = NULL,
    test = NULL,
    model = NULL,
    targetLevels = NULL,
    predictions = NULL,
    probabilities = NULL,
    modelType = NULL
  )
  
  # Load data
  observeEvent(input$file, {
    data <- read.csv(input$file$datapath, 
                     header = input$header, 
                     sep = input$sep,
                     stringsAsFactors = TRUE)
    if(nrow(data) > 5000) {
      showNotification("Dataset exceeds 5000 rows. Sampling 5000 rows.", type = "warning")
      data <- data[sample(nrow(data), 5000), ]
    }
    values$data <- data
    updateSelectInput(session, "targetVar", choices = names(values$data))
    updateCheckboxGroupInput(session, "features", choices = names(values$data))
  })
  
  # Load demo data (Iris dataset)
  observeEvent(input$loadDemo, {
    iris_data <- iris
    colnames(iris_data)[5] <- "variety"
    values$data <- as.data.frame(iris_data)
    values$data$variety <- as.factor(values$data$variety)
    
    updateSelectInput(session, "targetVar", choices = names(values$data))
    updateCheckboxGroupInput(session, "features", choices = names(values$data))
    updateSelectInput(session, "targetVar", selected = "variety")
    updateRadioButtons(session, "targetType", selected = "multinomial")
    
    showNotification("Iris demo data loaded successfully.", type = "message")
  })
  
  # Data preview
  output$dataPreview <- DT::renderDataTable({
    req(values$data)
    DT::datatable(values$data, options = list(scrollX = TRUE, pageLength = 10))
  })
  
  # Data summary
  output$dataSummary <- renderPrint({
    req(values$data)
    summary(values$data)
  })
  
  # Data structure
  output$dataStructure <- renderPrint({
    req(values$data)
    str(values$data)
  })
  
  # Update target distribution
  observe({
    req(values$data, input$targetVar)
    
    if(input$targetVar %in% names(values$data)) {
      output$targetDistribution <- renderPrint({
        table(values$data[[input$targetVar]])
      })
      
      output$targetDistPlot <- renderPlotly({
        target_counts <- as.data.frame(table(values$data[[input$targetVar]]))
        names(target_counts) <- c("Category", "Count")
        
        p <- ggplot(target_counts, aes(x = Category, y = Count, fill = Category)) +
          geom_bar(stat = "identity") +
          theme_minimal() +
          labs(title = paste("Distribution of", input$targetVar),
               x = input$targetVar, y = "Count")
        
        ggplotly(p)
      })
      
      if(length(unique(values$data[[input$targetVar]])) == 2) {
        updateRadioButtons(session, "targetType", selected = "binary")
      } else if(length(unique(values$data[[input$targetVar]])) > 2) {
        updateRadioButtons(session, "targetType", selected = "multinomial")
      }
    }
  })
  
  # Select all features
  observeEvent(input$selectAll, {
    updateCheckboxGroupInput(session, "features", 
                             choices = names(values$data),
                             selected = setdiff(names(values$data), input$targetVar))
  })
  
  # Deselect all features
  observeEvent(input$deselectAll, {
    updateCheckboxGroupInput(session, "features", 
                             choices = names(values$data),
                             selected = NULL)
  })
  
  # Apply preprocessing
  observeEvent(input$preprocess, {
    req(values$data, input$targetVar, input$features)
    
    if(anyNA(values$data)) {
      showNotification("Dataset contains missing values. Please clean the data.", type = "error")
      return()
    }
    
    features <- setdiff(input$features, input$targetVar)
    if(length(features) == 0) {
      showNotification("Please select at least one feature.", type = "error")
      return()
    }
    
    processed_data <- values$data
    processed_data[[input$targetVar]] <- as.factor(processed_data[[input$targetVar]])
    for(feat in features) {
      if(is.character(processed_data[[feat]])) {
        processed_data[[feat]] <- as.factor(processed_data[[feat]])
      }
    }
    values$targetLevels <- levels(processed_data[[input$targetVar]])
    values$processedData <- processed_data[, c(input$targetVar, features)]
    
    set.seed(if(input$randomSeed) input$seedValue else sample.int(1000000, 1))
    splitIndex <- createDataPartition(values$processedData[[input$targetVar]], p = input$trainSplit/100, list = FALSE)
    values$train <- values$processedData[splitIndex, ]
    values$test <- values$processedData[-splitIndex, ]
    
    showNotification("Preprocessing complete!", type = "message")
    
    output$formula <- renderPrint({
      formula_text <- paste(input$targetVar, "~", paste(features, collapse = " + "))
      cat(formula_text)
    })
  })
  
  # Train model
  observeEvent(input$trainModel, {
    req(values$train, values$test, input$targetVar)
    
    if(nrow(values$train) < 10) {
      showNotification("Training set is too small. Increase training split or upload larger dataset.", type = "error")
      return()
    }
    
    withProgress(message = "Training model...", value = 0, {
      incProgress(0.1, detail = "Validating data...")
      if(anyNA(values$train) || anyNA(values$test)) {
        showNotification("Missing values in training/test data.", type = "error")
        return()
      }
      
      incProgress(0.3, detail = "Building formula...")
      features <- setdiff(names(values$train), input$targetVar)
      formula_str <- paste(input$targetVar, "~", paste(features, collapse = " + "))
      formula_obj <- as.formula(formula_str)
      
      incProgress(0.5, detail = "Setting up cross-validation...")
      train_control <- trainControl(
        method = "cv",
        number = input$cvFolds,
        classProbs = TRUE,
        summaryFunction = multiClassSummary
      )
      
      incProgress(0.7, detail = "Training model...")
      values$model <- tryCatch({
        if(input$targetType == "binary") {
          values$modelType <- "binary"
          train(
            formula_obj,
            data = values$train,
            method = "glm",
            family = "binomial",
            preProcess = if(input$standardize) c("center", "scale") else NULL,
            trControl = train_control
          )
        } else {
          values$modelType <- "multinomial"
          train(
            formula_obj,
            data = values$train,
            method = "multinom",
            preProcess = if(input$standardize) c("center", "scale") else NULL,
            trControl = train_control,
            trace = FALSE
          )
        }
      }, error = function(e) {
        showNotification(paste("Training failed:", e$message), type = "error")
        output$trainingStatus <- renderPrint({
          cat("Training failed:", e$message, "\n")
        })
        return(NULL)
      })
      
      if(is.null(values$model)) return()
      
      incProgress(0.9, detail = "Evaluating model...")
      values$predictions <- predict(values$model, newdata = values$test)
      values$probabilities <- predict(values$model, newdata = values$test, type = "prob")
      
      conf_matrix <- confusionMatrix(values$predictions, values$test[[input$targetVar]])
      
      output$modelSummary <- renderPrint({
        summary(values$model)
      })
      
      output$coefficients <- DT::renderDataTable({
        if(values$modelType == "binary") {
          coef_data <- as.data.frame(summary(values$model$finalModel)$coefficients)
          coef_data$Variable <- rownames(coef_data)
          rownames(coef_data) <- NULL
          names(coef_data) <- c("Estimate", "Std.Error", "z-value", "p-value", "Variable")
          coef_data <- coef_data[, c("Variable", "Estimate", "Std.Error", "z-value", "p-value")]
          coef_data$Significant <- ifelse(coef_data$`p-value` < 0.05, "Yes", "No")
          DT::datatable(coef_data, options = list(pageLength = 10))
        } else {
          coef_data <- coef(values$model$finalModel)
          coef_long <- data.frame()
          for(i in 1:nrow(coef_data)) {
            class_name <- rownames(coef_data)[i]
            class_coefs <- data.frame(
              Class = class_name,
              Variable = colnames(coef_data),
              Coefficient = as.numeric(coef_data[i, ])
            )
            coef_long <- rbind(coef_long, class_coefs)
          }
          DT::datatable(coef_long, options = list(pageLength = 10))
        }
      })
      
      output$performanceMetrics <- renderPrint({
        print(conf_matrix)
      })
      
      output$confusionMatrix <- renderPrint({
        print(conf_matrix$table)
      })
      
      output$confusionMatrixPlot <- renderPlot({
        conf_data <- as.data.frame(conf_matrix$table)
        names(conf_data) <- c("Reference", "Prediction", "Freq")
        ggplot(conf_data, aes(x = Reference, y = Prediction, fill = Freq)) +
          geom_tile() +
          geom_text(aes(label = Freq), color = "white", size = 5) +
          scale_fill_gradient(low = "steelblue", high = "darkblue") +
          theme_minimal() +
          labs(title = "Confusion Matrix")
      })
      
      output$classPerformancePlot <- renderPlot({
        if(values$modelType == "binary") {
          metrics <- data.frame(
            Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1"),
            Value = c(
              conf_matrix$overall["Accuracy"],
              conf_matrix$byClass["Sensitivity"],
              conf_matrix$byClass["Specificity"],
              conf_matrix$byClass["Pos Pred Value"],
              conf_matrix$byClass["F1"]
            )
          )
          ggplot(metrics, aes(x = Metric, y = Value, fill = Metric)) +
            geom_bar(stat = "identity") +
            geom_text(aes(label = sprintf("%.3f", Value)), vjust = -0.5) +
            theme_minimal() +
            labs(title = "Performance Metrics", y = "Value", x = "") +
            ylim(0, 1.1)
        } else {
          metrics <- conf_matrix$byClass
          metrics_df <- data.frame(
            Class = rownames(metrics),
            Sensitivity = metrics[, "Sensitivity"],
            Specificity = metrics[, "Specificity"],
            Precision = metrics[, "Pos Pred Value"],
            F1 = metrics[, "F1"]
          )
          metrics_long <- tidyr::pivot_longer(metrics_df, 
                                              cols = c(Sensitivity, Specificity, Precision, F1),
                                              names_to = "Metric", 
                                              values_to = "Value")
          ggplot(metrics_long, aes(x = Class, y = Value, fill = Metric)) +
            geom_bar(stat = "identity", position = "dodge") +
            theme_minimal() +
            labs(title = "Performance Metrics by Class", y = "Value", x = "Class") +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))
        }
      })
      
      output$featureImportance <- renderPlot({
        if(values$modelType == "binary") {
          coefs <- coef(values$model$finalModel)
          coef_data <- data.frame(
            Feature = names(coefs),
            Importance = abs(as.numeric(coefs))
          )
          coef_data <- coef_data[-1, ] # Remove intercept
          coef_data <- coef_data[order(coef_data$Importance, decreasing = TRUE), ]
          
          ggplot(coef_data, aes(x = reorder(Feature, Importance), y = Importance)) +
            geom_bar(stat = "identity", fill = "steelblue") +
            coord_flip() +
            theme_minimal() +
            labs(title = "Feature Importance (Absolute Coefficient Values)", x = "Features", y = "Importance")
        } else {
          coef_data <- coef(values$model$finalModel)
          avg_importance <- colMeans(abs(coef_data))
          importance_df <- data.frame(
            Feature = names(avg_importance),
            Importance = avg_importance
          )
          importance_df <- importance_df[order(importance_df$Importance, decreasing = TRUE), ]
          
          ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
            geom_bar(stat = "identity", fill = "steelblue") +
            coord_flip() +
            theme_minimal() +
            labs(title = "Feature Importance (Average Absolute Coefficient Values)", x = "Features", y = "Importance")
        }
      })
      
      output$trainingStatus <- renderPrint({
        cat("Training complete!\n")
        cat("Model type:", if(input$targetType == "binary") "Binary Logistic Regression" else "Multinomial Logistic Regression", "\n")
        cat("Number of observations in training set:", nrow(values$train), "\n")
        cat("Number of observations in test set:", nrow(values$test), "\n")
        cat("Accuracy on test set:", round(conf_matrix$overall["Accuracy"] * 100, 2), "%\n")
      })
      
      incProgress(1.0, detail = "Complete!")
    })
  })
  
  # Update prediction inputs
  output$predictionInputs <- renderUI({
    req(values$processedData, values$model)
    
    feature_list <- list()
    features <- setdiff(names(values$processedData), input$targetVar)
    
    for(feat in features) {
      if(is.factor(values$processedData[[feat]])) {
        feature_list[[feat]] <- selectInput(
          inputId = paste0("pred_", feat),
          label = feat,
          choices = levels(values$processedData[[feat]])
        )
      } else {
        min_val <- min(values$processedData[[feat]], na.rm = TRUE)
        max_val <- max(values$processedData[[feat]], na.rm = TRUE)
        mean_val <- mean(values$processedData[[feat]], na.rm = TRUE)
        
        feature_list[[feat]] <- sliderInput(
          inputId = paste0("pred_", feat),
          label = feat,
          min = min_val,
          max = max_val,
          value = mean_val,
          step = (max_val - min_val) / 100
        )
      }
    }
    
    do.call(tagList, feature_list)
  })
  
  # Make prediction
  observeEvent(input$predictButton, {
    req(values$model, values$processedData)
    
    features <- setdiff(names(values$processedData), input$targetVar)
    new_data <- data.frame(matrix(NA, nrow = 1, ncol = length(features)))
    names(new_data) <- features
    
    for(feat in features) {
      input_id <- paste0("pred_", feat)
      new_data[[feat]] <- input[[input_id]]
    }
    
    predicted_class <- predict(values$model, newdata = new_data)
    predicted_probs <- predict(values$model, newdata = new_data, type = "prob")
    
    output$predictionResult <- renderPrint({
      cat("Predicted Class:", as.character(predicted_class), "\n\n")
      cat("Prediction Probabilities:\n")
      print(predicted_probs)
    })
    
    output$predictionProbPlot <- renderPlot({
      prob_df <- data.frame(
        Class = colnames(predicted_probs),
        Probability = as.numeric(predicted_probs[1, ])
      )
      
      ggplot(prob_df, aes(x = Class, y = Probability, fill = Class)) +
        geom_bar(stat = "identity") +
        geom_text(aes(label = sprintf("%.3f", Probability)), vjust = -0.5) +
        theme_minimal() +
        labs(title = "Prediction Probabilities", y = "Probability", x = "Class") +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        ylim(0, 1)
    })
  })
}

# Run the app
shinyApp(ui = ui, server = server)