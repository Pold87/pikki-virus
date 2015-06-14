
library(h2o)

########
localH2O <- h2o.init(nthreads = -1, max_mem_size = '7g')

train <- read.csv("unique_train.csv", header = T)
test <- read.csv("new_test_priors.csv", header = T)
test.X <- NULL

# train <- head(train, 100)
# test <- head(test, 100)

train.data <- as.h2o(localH2O, train)
test.data <- as.h2o(localH2O, test)


########Execute deeplearning
model <- h2o.deeplearning(x = c(1:2, 4:78, 80:91),  # column numbers for predictors
               y = 92,   # column number for label
               data = train.data, # data in H2O format
               activation = "TanhWithDropout", # or 'Tanh'
               input_dropout_ratio = 0.2, # % of inputs dropout
               hidden_dropout_ratios = c(0.5, 0.5, 0.5), # % for nodes dropout
               balance_classes = TRUE,
               hidden = c(50, 50, 50), # three layers of 50 nodes
               epochs = 100) # max. no. of epochs

########
y_hat <- h2o.predict(model, test.data)

df_y_hat <- as.data.frame(y_hat)

WnvPresent = df_y_hat[, 3]

summary(WnvPresent)
Id <- 1:nrow(WnvPresent)

write.csv(cbind(Id, WnvPresent), file = "result.csv", row.names  = FALSE)
