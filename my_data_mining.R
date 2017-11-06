


#install and import all the required packages
install.packages("RWeka", dependencies = TRUE)
install.packages("e1071", dependencies = TRUE)
install.packages("caret", dependencies = TRUE)
install.packages("dplyr", dependencies = TRUE)
install.packages("class", dependencies = TRUE)
install.packages("readxl", dependencies = TRUE)
library(readxl)
library("e1071")
library(class)
library(caret)
library(rJava)
library(RWeka)


#Dataset division into training and test by sampling

divideDataset<-function(data,seed){
  ## 80% of the sample size
  smp_size <- floor(0.80 * nrow(data))
  ## set the seed to make your partition reproductible
  set.seed(seed)
  train_ind <- sample(seq_len(nrow(data)), size = smp_size)
  colums<-c("Overall Life","Male Life", "Female Life", "Continent")
  data_train <- data[train_ind,colums ]
  data_test <- data[-train_ind,colums ]
  t_train <- data$Continent[train_ind]
  t_test <- data$Continent[-train_ind]
  output<-list(data_train,data_test,t_train,t_test)
  return(output)
}

#SVM

mySvm<-function(data,seed){
#print("SVM")
train_test<-divideDataset(data,seed)
let_train<-train_test[[1]]
let_test<-train_test[[2]]
#Fit model
svmfit <- svm(Continent ~., data = let_train, kernel = "linear", cost = 1, scale = FALSE)
svmfit
#Tune to check best performance
tuned <- tune(svm, Continent ~., data = let_train, kernel = "linear", ranges = list(cost=c(0.001,0.01,.1,1,10,100)))
summary(tuned)
#Make predictions
p <- predict(svmfit, let_test, type="class")
length(let_test$Continent)
table(p, let_test$Continent)
#Analyse results
#Confusion matrix
confusionMatrix(p, let_test$Continent )
#Accuracy
#print(mean(p== let_test$Continent))
svmoutput<-mean(p== let_test$Continent)
return(svmoutput)
}

#KNN

myKnn<-function(data,seed){
  #print("KNN")
  #Slicing
  knn_train_test<-divideDataset(data,seed)
  let_train<-knn_train_test[[1]]
  let_test<-knn_train_test[[2]]
  #Preprocessing and training
  trainX <- let_train[,names(let_train) != "Continent"]
  preProcValues <- preProcess(x = trainX,method = c("center", "scale"))
  preProcValues
  #Fit Model- Using Caret's train model to find best k
  ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
  knnFit <- train(Continent~., data = let_train, method = "knn", trControl = ctrl,preProcess = c("center","scale"), tuneLength = 20)
  #knnFit
  plot(knnFit)
  #Make predictions
  knnPredict <- predict(knnFit,newdata = let_test )
  knnPredict
  #Summarize accuracy
  #Confusion Matrix
  confusionMatrix(knnPredict, let_test$Continent )
  #Accuracy
  #print(mean(knnPredict == let_test$Continent))
  knnoutput<-mean(knnPredict== let_test$Continent)
  return(knnoutput)
  
 }

#RIPPER

myRipper<-function(data,seed){
  #print("RIPPER")
  train_test<-divideDataset(data,seed)
  let_train<-train_test[[1]]
  let_test<-train_test[[2]]
  # fit model-Using Weka Control function of JRip to tune
  fit <- JRip(Continent~., data=let_train,  control = Weka_control( F =50))
  # summarize the fit
  summary(fit)
  # make predictions
  predictions <- predict(fit, let_test)
  # summarize accuracy
  tb<-table(predictions, let_test$Continent)
  #Confusion Matrix
  confusionMatrix(predictions, let_test$Continent )
  #Accuracy
  #print(mean(predictions== let_test$Continent))
  ripsoutput<-mean(predictions== let_test$Continent)
  return(ripsoutput)
}

#c45

myc45<-function(data,seed){
  #print("C45")
  train_test<-divideDataset(data,seed)
  let_train<-train_test[[1]]
  let_test<-train_test[[2]]
  # fit model-Using Weka Control function of J48 to tune
  fit <- J48(Continent~., data=let_train, control = Weka_control(R = TRUE, M = 9))
  # summarize the fit
  summary(fit)
  # make predictions
  c45predictions <- predict(fit, let_test)
  # summarize accuracy
  tb<-table(c45predictions, let_test$Continent)
  #Confusion Matrix
  confusionMatrix(c45predictions, let_test$Continent )
  #Accuracy
  #print(mean(c45predictions== let_test$Continent))
  c45output<-mean(c45predictions== let_test$Continent)
  return(c45output)
  }

#MAIN
main<-function(){
#reading from dataset address location
Life_expectancy_dataset <- read_excel("Life_expectancy_dataset.xlsx")
View(Life_expectancy_dataset)
#Converting Continent to factor
Life_expectancy_dataset[c("Continent")]<- lapply(Life_expectancy_dataset[c("Continent")], factor)
data=Life_expectancy_dataset
randomSeeds<-list(2018,2166,2289,2322,2408)
lp<-length(randomSeeds)
results <- list(kknn=c(), kc45=c(), krip=c(), ksvn=c())
#Reporting results for 5 groups for each algorithm with help of seeds
for (i in 1:lp)
{
  for(j in 1:4){
    res <-   myKnn(data,randomSeeds[[i]])
    
    if(j==2){
      res <- myc45(data,randomSeeds[[i]])
    }
    
    if(j==3){
      res <-  myRipper(data,randomSeeds[[i]])
    }
    
    if(j==4){
      res <- mySvm(data,randomSeeds[[i]])
    }
    
   
    results[[j]] <- c(results[[j]], res)

  
  }
}
#Calculating average accuracy and average standard deviation
avg <- c()
msd <- c()
for (i in 1:4){
  temp <- c()
 
  for (each in results[i]){
    temp <- append(temp, each)
  }
  avg <- append(avg, mean(temp))
  msd <- append(msd, sd(temp))
}
  #present the results
  for(i in 1:4){
    umethod <- "KNN"
    
    if(i==2){
      umethod <- "C4.5"
    }
    
    if(i==3){
      umethod <- "RIPPER"
    }
    
    if(i==4){
      umethod <- "SVM"
    }
    
    print(sprintf("method name: %s; averaged accuracy: %.2f; accuracy standard deviation: %.3f", umethod, avg[i], msd[i]))
    }
}



#Calling main executes the entire program
main()



