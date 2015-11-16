#Load required packages
library(caret)
library(randomForest)

#Set working directory correctly
setwd("~/Data Science/Practical Machine Learning/Course_project")

#Set seed for reproducibility
set.seed(39)

#Load the data
training_data <- read.csv("pml-training.csv")
test_data <- read.csv("pml-testing.csv")

#Calculate percentage of NA values per column
nof_rows <- nrow(training_data)
perc_NA <- colSums(is.na(training_data))/nof_rows

#Keep the columns which have less than 10% NA values
keep_cols <- perc_NA < 0.1
training_data <- training_data[,keep_cols]

#Find near zero variance variables and delete them
nzv_train <- nearZeroVar(training_data, saveMetrics=TRUE)
training_data <- training_data[, nzv_train$nzv==FALSE]

#Check for complete cases in remainder of data
sum(!complete.cases(training_data))
head(training_data)

#Delete the first six columns, since they do not contain valuable information
training_data <- training_data[,-(1:6)]

#Partition the data into training and cross validation set
inTrain <- createDataPartition(training_data$classe, p=0.70, list=F)
train_data <- training_data[inTrain, ]
CV_data <- training_data[-inTrain, ]

#Determine the model using a random forest
model <- randomForest(classe ~. , data=train_data)

#Predict our model using the cross validation data and show outcome
CV_prediction <- predict(model, CV_data)
confusionMatrix(CV_data$classe, CV_prediction)

#Determine expected accuracy and out of sample error of our 
#model based on cross validation set
accuracy <- postResample(CV_prediction, CV_data$classe)
sample_error <- 1 - as.numeric(confusionMatrix(CV_data$classe, CV_prediction)$overall[1])

#Predict our model using the test_data (which of course we did not preprocess)
prediction <- predict(model, test_data)

#Write files to desired output for automatic grading
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(prediction)