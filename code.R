## 1.load packages ####
library(xgboost)
library(shapviz)
library(ggplot2)
library(readxl)
library(tidyverse)
library(Matrix)  

## 2.read data ####
dt <- read_excel("features.examples.xlsx")
str(dt)
dt <- as.data.frame(dt)
dt[,2:11] <- lapply(dt[,2:11], factor)

## 2.1 divide dataset, trainset(700) and testset(299).
set.seed(0)  
s = sample(999,700)  
trainset = dt[s,]  
testset = dt[-s,]

trainset.Y <- trainset[,1]
trainset.overhead <- trainset[,2:6]
trainset.horizontal <- trainset[,7:11]
trainset.final <- trainset[,c(2,3:5)]

testset.Y <- testset[,1]
testset.overhead <- testset[,2:6]
testset.horizontal <- testset[,7:11]
testset.final <- testset[,c(2,3:5)]

## 3 overhead perspective model    ####
dtrain.overhead <- xgb.DMatrix(data.matrix(trainset.overhead), 
                               label=trainset.Y)
dtest.overhead <- xgb.DMatrix(data.matrix(testset.overhead), 
                               label=testset.Y)

fit.overhead <- xgb.train(
  params = list(learning_rate = 0.1, objective = "binary:logistic"),
  data = dtrain.overhead,
  nrounds = 65L)

pre_xgb = round(predict(fit.overhead,newdata = dtest.overhead))  
(test.overhead <- table(testset.Y,pre_xgb,dnn=c("true","pre")))
(Accuracy.test.overhead = sum(diag(test.overhead))/nrow(testset.overhead))
(Recall.test.overhead = test.overhead[1,1]/(test.overhead[1,1]+
                                              test.overhead[1,2]))
(Precision.test.overhead = test.overhead[1,1]/(test.overhead[1,1]+
                                                 test.overhead[2,1]))
(F1 <- 2*Precision.test.overhead*Recall.test.overhead/(Precision.test.overhead+
                                                         Recall.test.overhead))

shp <- shapviz(fit.overhead, 
               X_pred = data.matrix(testset.overhead),
               X = testset.overhead)

sv_importance(shp,show_numbers = TRUE)+
  theme_bw()

sv_importance(shp, kind = "beeswarm")+
  theme_bw()





## 4 horizontal perspective model    ####
dtrain.horizontal <- xgb.DMatrix(data.matrix(trainset.horizontal), 
                                 label=trainset.Y)
dtest.horizontal <- xgb.DMatrix(data.matrix(testset.horizontal), 
                                label=testset.Y)

fit.horizontal <- xgb.train(
  params = list(learning_rate = 0.1, objective = "binary:logistic"),
  data = dtrain.horizontal,
  nrounds = 65L)

pre_xgb = round(predict(fit.horizontal,newdata = dtest.horizontal))  
(test.horizontal <- table(testset.Y,pre_xgb,dnn=c("true","pre")))
(Accuracy.test.horizontal = sum(diag(test.horizontal))/nrow(testset.horizontal))
(Recall.test.horizontal = test.horizontal[1,1]/(test.horizontal[1,1]+
                                                  test.horizontal[1,2]))
(Precision.test.horizontal = test.horizontal[1,1]/(test.horizontal[1,1]+
                                                     test.horizontal[2,1]))
(F1 <- 2*Precision.test.horizontal*Recall.test.horizontal/(Precision.test.horizontal+
                                                             Recall.test.horizontal))

shp <- shapviz(fit.horizontal, 
               X_pred = data.matrix(testset.horizontal),
               X = testset.horizontal)

sv_importance(shp,show_numbers = TRUE)+
  theme_bw()

sv_importance(shp, kind = "beeswarm")+
  theme_bw()
## 5 final perspective model    ####
dtrain.final <- xgb.DMatrix(data.matrix(trainset.final), 
                            label=trainset.Y)
dtest.final <- xgb.DMatrix(data.matrix(testset.final), 
                           label=testset.Y)

fit.final <- xgb.train(
  params = list(learning_rate = 0.1, objective = "binary:logistic"),
  data = dtrain.final,
  nrounds = 65L)

pre_xgb = round(predict(fit.final,newdata = dtest.final))  
(test.final <- table(testset.Y,pre_xgb,dnn=c("true","pre")))
(Accuracy.test.final = sum(diag(test.final))/nrow(testset.final))
(Recall.test.final = test.final[1,1]/(test.final[1,1]+
                                        test.final[1,2]))
(Precision.test.final = test.final[1,1]/(test.final[1,1]+
                                           test.final[2,1]))
(F1 <- 2*Precision.test.final*Recall.test.final/(Precision.test.final+
                                                   Recall.test.final))

shp <- shapviz(fit.final, 
               X_pred = data.matrix(testset.final),
               X = testset.final)

sv_importance(shp,show_numbers = TRUE)+
  theme_bw()

sv_importance(shp, kind = "beeswarm")+
  theme_bw()