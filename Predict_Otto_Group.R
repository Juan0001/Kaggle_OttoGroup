## Import libraries
require(xgboost)
require(methods)
require(data.table)
require(magrittr)

## import data for manipulate
train <- fread('train.csv', header = T, stringsAsFactors = F)
test <- fread('test.csv', header=TRUE, stringsAsFactors = F)

## data copy
Tn <- fread('train.csv', header = T, stringsAsFactors = F)
Tt <- fread('test.csv', header=TRUE, stringsAsFactors = F)

## Use XGBoost to do prediction
# Delete ID column in training dataset
train[, id := NULL]

## Delete ID column in testing dataset
test[, id := NULL]

## Check the content of the last column
train[1:6, ncol(train), with  = F]

## Save the name of the last column
nameLastCol <- names(train)[ncol(train)]

## Convert from classes to numbers
y <- train[, nameLastCol, with = F][[1]] %>% gsub('Class_','',.) %>% {as.integer(.) -1}

## Display the first 5 levels
y[1:5]

## Remove label column from train dataset
train[, nameLastCol:=NULL, with = F]

## Convert data sets to numeric Matrix format for XGBoost
trainMatrix <- train[,lapply(.SD,as.numeric)] %>% as.matrix
testMatrix <- test[,lapply(.SD,as.numeric)] %>% as.matrix

## Cross validation
numberOfClasses <- max(y) + 1

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = numberOfClasses)

cv.nround <- 5
cv.nfold <- 3

bst.cv = xgb.cv(param=param, data = trainMatrix, label = y, 
                nfold = cv.nfold, nrounds = cv.nround)

## Model training
nround = 50
bst = xgboost(param=param, data = trainMatrix, label = y, nrounds=nround)

## Prediction
pred <- predict(bst, testMatrix)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

## Data output for submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submissionXGBoost.csv', quote=FALSE,row.names=FALSE)