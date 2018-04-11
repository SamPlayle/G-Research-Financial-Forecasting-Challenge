#' G Research Financial Forecasting Challenge
#' 
#' Sam Playle
#' 
#' https://financialforecasting.gresearch.co.uk/

#' We have a training dataset with about half a million datapoints. The 
#' independent variable we are trying to predict (y) is the excess returns for a 
#' given stock on a given day. The independent variables are the stock, the day, 
#' the market, and eleven anonymous numeric fields. There are about three 
#' thousand distinct stocks in the dataset.
#' 
#' The test dataset and training dataset contain different days (i.e. there is 
#' no day that has records in both the training dataset and the test dataset). 
#' Therefore, any algorithm which depends on knowing market moves on a given day 
#' will not be able to generalize beyond the training dataset. To account for 
#' this, we make sure that our cross-validation folds all cover mutually 
#' disjoint sets of days. 

#' Load the libraries that we'll need:
library(readr); library(xgboost); library(dplyr)

#' Load the training and testing datasets
training_data <- "D:/G Research Data/train.csv"
train <- read_csv(training_data)
testing_data <- "D:/G Research Data/test.csv"
test <- read_csv(testing_data)

#' In order to keep track of how well our model will generalize, we will keep a 
#' validation set that will not be used to train the model, only to measure how 
#' well it performs on unseen data. A typical validation set comprises about 25%
#' or 30% of the training data, but since we don't have that much data to train
#' on anyway (258 days) we'll use a smaller hold-out, of 20%, so we select 28% 
#' of the days in the training dataset and split them off into a validation 
#' dataset.

# Use a seed for the pseudo-random number generator to ensure reproducability
set.seed(76)
validation_days <- training_days[as.logical(rbinom(n = length(training_days), 
                                                   size = 1, prob = 0.2))]

sub_train <- train %>% filter(! Day %in% validation_days)
validation <- train %>% filter( Day %in% validation_days)

#' On inspecting the training dataset, we find that all of the anonymous 
#' technical factors ("xA" etc) share the property of being strictly 
#' non-negative and heavily concentrated around 0. To make it easier to train a
#' model, let's transform them to be more spread out: with logarithmic 
#' transformations we can transform them into bell-shaped distributions (albeit
#' with fatter or skinnier tails than a normal distribution) with mean 0 and 
#' standard deviation 1. Let's create a function that will implement these
#' transformations:

transform_var <- function(df) {
  #' Define the variables we want to transform:
  num_vars <- names(sub_train)[grepl("^x[0-6][A-E]*$", names(sub_train))]
  for (var in num_vars) {
    #' Give new names to the transformed variables: P for "prime". 
    new_var <- paste(var, "P", sep = "")
    #' Add a small 'epsilon', namely 10^(-100), to avoid log(0). 
    df[[new_var]] <- (log(df[[var]] + 1e-100) 
                      - mean(log(sub_train[[var]] + 1e-100),
                             na.rm = TRUE))/sd(log(sub_train[[var]] + 1e-100), 
                                               na.rm = TRUE)
    #' In case of any missing values, use 0 (which ought to be the mean value). 
    df[[new_var]] <- ifelse(is.na(df[[new_var]]), 0, df[[new_var]])
  }
  #' y is already in a bell-shaped distribution, but it can still be 
  #' normalized to have mean 0 and standard deviation 1.
  df[["yP"]] <- (df[["y"]] - mean(sub_train[["y"]]))/sd(sub_train[["y"]])
  #' Use 1-hot encoding for the markets.
  df[["Mar2"]] <- as.numeric(df[["Market"]] == 2)
  df[["Mar3"]] <- as.numeric(df[["Market"]] == 3)
  df[["Mar4"]] <- as.numeric(df[["Market"]] == 4)
  df
}

#' Our model will require the use of the stock-specific mean returns and 
#' volatilities. Let's compute these from the training dataset, so they can be
#' appended to the testing dataset at the prediction stage.  
stock_data <- sub_train %>% transform_var %>% group_by(Stock) %>% 
                      summarise(stockMean = mean(yP), 
                                stockSD = ifelse(is.na( sd(yP)), 1, sd(yP))) 

#' We want to encode information about the stock but without just listing a 
#' label from 1 to ~3000. We can do this with "target encoding" AKA "mean 
#' encoding". The idea is to include information about a given record's stock 
#' by listing the mean output for that stock. 
#' 
#' There is a problem though: We can't simply list the mean for a stock 
#' calculated with the whole training set. This would be a form of "data
#' leakage" because it would mean that there is information about the 
#' dependent variable (y) contained in one of the independent variables (the 
#' mean that is being used to encode the stock). To avoid this, we split the 
#' training dataset into four cross-validation folds, determined by randomly 
#' splitting into four parts the days that appear in the training dataset. 
#' Within each fold, we encode a stock as the mean return of that stock within
#' the *other* three folds.

set.seed(136)

#' Create a data frame defining a random group (from 1 to 4) for each day 
#' appearing in the training dataset. 
train_days <- unique(sub_train$Day)
day_folds <- sample(1:4, length(train_days), replace = TRUE)
fold_df <- tibble(Day = train_days, fold = day_folds)

#' Transform the training data frame.
train_df <- sub_train %>% transform_var %>% left_join(fold_df, by = "Day")
#' Sometimes a plyr function overrides a dplyr function at this point so make 
#' sure that doesn't happen...
library("plyr")
detach(package:plyr)

#' Create a dataset showing the mean returns and standard deviation for each 
#' stock *outside* of a given fold:
replacement_df <- tibble()
for (i in 1:4) {
  df <- train_df %>% filter(fold != i) %>% group_by(Stock) %>% 
    summarise(cvMean = mean(yP), cvSd = sd(yP)) %>% ungroup %>% mutate(fold = i)
  replacement_df <- replacement_df %>% bind_rows(df)
}

#' Sometimes a stock does not exist outside of a given fold. In this case, use
#' the overall mean and volatility outside of that fold.
replacement_df2 <- tibble()
for (i in 1:4) {
  df <- train_df %>% filter(fold != i) %>% summarise(foldMean = mean(yP), 
                                                     foldSd = sd(yP)) %>% 
    ungroup %>% mutate(fold = i)
  replacement_df2 <- replacement_df2 %>% bind_rows(df)
}

#' Use these out-of-fold means and volatilities to encode the stocks in the 
#' training dataset. 
train_df <- train_df %>% left_join(replacement_df, by = c("fold", "Stock")) %>% 
                         left_join(replacement_df2, "fold") %>% 
                         mutate(cvMean = ifelse(is.na(cvMean), 
                                                foldMean, cvMean), 
                                cvSd = ifelse(is.na(cvSd), foldSd, cvSd)) %>% 
                         select( - foldSd, - foldMean)

#' Scale the stock mean column by some noise to avoid getting a model that just
#' "memorizes" all the stock means (because they will not be the same when we
#' generalize). 

n <- nrow(train_df)
noise1 <-  exp(rnorm(n, sd = 0.01))
noise2 <- exp(rnorm(n, sd = 0.01))

train_df <- train_df %>% mutate(stockSD = noise1 * cvSd, 
                                stockMean = noise2 * cvMean) %>% 
                select(yP, Weight, Day, Mar2, Mar3, Mar4, stockMean, stockSD, 
                       x0P, x1P, x2P, x3AP, x3BP, x3CP, x3DP, x3EP, x4P, x5P, 
                       x6P, fold) 

train_fold <- train_df[["fold"]]
train_y <- train_df[["yP"]]
train_weight <- train_df[["Weight"]]
train_matrix <- train_df %>% select( - yP, - Weight, - fold) %>% as.matrix()

#' We want to use the same folds for cross-validation as we used to define the
#' stock means, otherwise the cross-validation process will not accurately 
#' resemble the generalization process. 
custom_folds <- vector("list", 4)
for (i in 1:4) {
  custom_folds[[i]] <- which(train_fold == i)
}

#' Create a grid of plausible values to test for cross-validation.
xgbGrid <- expand.grid(
  nrounds = c(2, 6, 18, 54), #' Best is 18
  max_depth = c(2, 6, 12, 20), #' Best is 6
  eta =  0.3,
  gamma = c(2, 6, 14, 20), #' Best is 14.
  colsample_bytree =  c(1,0.5,0.25), #' Best is 1.
  min_child_weight = c(1,2,3,4), #' Best is 2.
  subsample = 0.7
)

#' Use cross-validation with the custom folds that we've defined. 
xgbTrControl <- trainControl(
  method = "repeatedcv",
  index = custom_folds,
  verboseIter = TRUE,
  returnData = FALSE,
  allowParallel = TRUE
)

#' Train a gradient-boosted tree model using xgboost.
xgbTrain <- train(
  x = train_matrix, 
  y = train_y,
  #' The metric depends on weights so make sure our model trains on them too. 
  weights = train_weight,
  #' Use the sum of squared differences metric. 
  objective = "reg:linear",
  trControl = xgbTrControl,
  tuneGrid = xgbGrid,
  method = "xgbTree"
)

#' Check how well we do on our held-out validation set before we decide to 
#' upload:
val_matrix <- validation %>% transform_var %>% left_join(stock_data, 
                                                         by = "Stock") %>% 
              select(Day, Mar2, Mar3, Mar4, stockMean, stockSD, x0P, x1P, x2P, 
                    x3AP, x3BP, x3CP, x3DP, x3EP, x4P, x5P, x6P) %>% as.matrix()
#' Predict the transformed "y" from the validation dataset. 
yP_hat <- predict(xgbTrain$finalModel, val_matrix) 
#' Transform these predictions by the inverse of the normalization 
#' that we carried out for the training data.
y_hat <- mean(sub_train[["y"]]) + sd(sub_train[["y"]]) * yP_hat
y_hat <- ifelse(is.na(y_hat), mean(sub_train[["y"]]), y_hat)

#' Compute the weighted sum of squared differences:
sum(validation$Weight * (validation$y - y_hat)^2)

#' Once we are satisfied with the quality of the model, we can use it to make
#' predictions on the test dataset which we can then upload. We need to modify 
#' the transformation function that we used above, so that it doesn't throw an
#' error when it tries to transform y. 
transform_test <- function (df) 
{
  num_vars <- names(sub_train)[grepl("^x[0-6][A-E]*$", names(sub_train))]
  for (var in num_vars) {
    new_var <- paste(var, "P", sep = "")
    df[[new_var]] <- (log(df[[var]] + 1e-100) - mean(log(sub_train[[var]] + 
                               1e-100), na.rm = TRUE))/sd(log(sub_train[[var]] + 
                                                          1e-100), na.rm = TRUE)
    df[[new_var]] <- ifelse(is.na(df[[new_var]]), 0, df[[new_var]])
  }
  df[["Mar2"]] <- as.numeric(df[["Market"]] == 2)
  df[["Mar3"]] <- as.numeric(df[["Market"]] == 3)
  df[["Mar4"]] <- as.numeric(df[["Market"]] == 4)
  df
}

#' Put the test data in the correct format for the model to use. 
test_matrix <- test %>% transform_test %>% 
                  left_join(stock_data, by = "Stock") %>% 
                  select(Day, Mar2, Mar3, Mar4, stockMean, stockSD, x0P, x1P, 
                         x2P, x3AP, x3BP, x3CP, x3DP, x3EP, x4P, x5P, x6P) %>% 
                  as.matrix()

#' Get the predictions for the "normalized" y:
yP_hat <- predict(xgbTrain$finalModel, test_matrix) 
#' Transform the predictions onto the correct scale:
y_hat <- mean(sub_train[["y"]]) + sd(sub_train[["y"]]) * yP_hat
y_hat <- ifelse(is.na(y_hat), mean(sub_train[["y"]]), y_hat)

#' Output the predictions as a CSV file ready for upload.
write_csv(data_frame(index = test$Index, y = y_hat), 
          "D:/G Research Data/model3pred.csv")
