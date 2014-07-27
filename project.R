library(doParallel)
library(caret)
library(kernlab)

cl <- makeCluster(detectCores())
registerDoParallel(cl)

initial.data <- read.csv("pml-training.csv")

str(initial.data)
initial.data[,7] <- NULL
initial.data[,6] <- NULL
initial.data[,5] <- NULL
initial.data[,4] <- NULL
initial.data[,3] <- NULL
initial.data[,2] <- NULL
initial.data[,1] <- NULL
initial.data <- initial.data[,-grep(x = names(initial.data),pattern = "var|avg|max|min|stddev|amplitude|skewness|kurtosis")]
sum(is.na(initial.data))

set.seed(14832)
index <- createDataPartition(y=initial.data$classe, p=0.75, list=FALSE)
training.data <- initial.data[index,]
testing.data <- initial.data[-index,]

preProc <- preProcess(training.data[,-dim(training.data)[2]], method="pca")
trainPC <- predict(preProc, training.data[,-dim(training.data)[2]])
testPC  <- predict(preProc, testing.data[,-dim(testing.data)[2]])

modelFit.rf <- train(training.data$classe ~ ., method="rf", data=trainPC, trControl = trainControl(method = 'cv'))
confusionMatrix(training.data$classe,predict(modelFit.rf,trainPC))
oos.cm <- confusionMatrix(testing.data$classe,predict(modelFit.rf,testPC))
error.table <- rbind(oos.cm$table, colSums(oos.cm$table))
error.table <- apply(error.table, MARGIN = 2, FUN = function (x) 100*x/x[length(x)])
error.table <- error.table[-dim(error.table)[1],]

submission.data <- read.csv("./predmachlearn-003/project/pml-testing.csv")
str(submission.data)
submission.data[,7] <- NULL
submission.data[,6] <- NULL
submission.data[,5] <- NULL
submission.data[,4] <- NULL
submission.data[,3] <- NULL
submission.data[,2] <- NULL
submission.data[,1] <- NULL
submission.data <- submission.data[,-grep(x = names(submission.data),pattern = "var|avg|max|min|stddev|amplitude|skewness|kurtosis")]
submitPC <- predict(preProc, submission.data[,-dim(submission.data)[2]])
predict(modelFit.rf,submitPC)