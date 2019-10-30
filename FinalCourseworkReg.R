##REGRESSION

#LOAD ALL LIBRARIES
rm(list = ls())
install.packages('car')
install.packages('caret')
install.packages('gbm')


library('MASS')
library('ISLR')
library('caret')
library('randomForest')
library ('gbm')
library('leaps')
library(car)


#LOAD DATASET
attach(Boston)
dt = data.table::as.data.table(Boston)

#SPLIT DATASET INTO TRAIN AND TEST SET
set.seed(123)
rows = sample(1:nrow(Boston),nrow(Boston)/2)
train = dt[rows]
test = dt[-rows]


#FEATURE SELECTION
#USING regsubsets() BASED ON BIC, R^2, ADJ R^2, Cp
set.seed(123)
sub = regsubsets(medv~., data = train)
plot(sub ,scale ="r2")
plot(sub ,scale ="adjr2")
plot(sub ,scale ="Cp")
plot(sub ,scale ="bic")

#CHECK VIF
vif(lm(medv~., data = train))

#USING regsubsets() FOR FORWARD AND BACKWARD SELECTION
set.seed(123)
regfit.fwd=regsubsets (medv~., data = train ,nvmax =13,method ="forward")
summary(regfit.fwd)
regfit.bwd=regsubsets (medv~., data = train,nvmax =13,method ="backward")
summary (regfit.bwd)

#USING stepAIC() FOR FORWARD AND BACKWARD SELECTION WITH BIC
set.seed(123)
lr1 = lm(formula = medv ~., data = train)   # CREATE LINEAR MODEL WITH ALL VARIABLES
lr.back=stepAIC(lr1,trace = FALSE,direction = 'backward',k=log(nrow(train)))
lr.back$anova
lr.for=stepAIC(lr1,trace = FALSE,direction = 'forward',k=log(nrow(train)))
lr.for$anova

#USING stepAIC() FOR FORWARD AND BACKWARD SELECTION WITH AIC
lr.back.aic=stepAIC(lr1,trace = FALSE,direction = 'backward')
lr.back.aic$anova
lr.for.aic=stepAIC(lr1,trace = FALSE,direction = 'forward')
lr.for.aic$anova

#USING randomForest() FOR IMPORTANT FEATURES
set.seed(123)
rf_medv = randomForest(medv ~ ., data = train, mtry = 5,importance = TRUE)
rf_medv$importance
varImpPlot(rf_medv) #CREATE PLOT OF IMPORTANT FEATURES
vif(lr1)  #CHECK VIF

#BEST THREE MODEL
#RANDOM FOREST
#TRAIN MODEL
set.seed(123)
rf_boston = randomForest(medv ~ ., data = train, mtry =4,importance = TRUE)
pred_rf = predict(rf_boston, newdata = test)  #PREDICT ON TEST DATA
rmse_rf = sqrt(mean((pred_rf - test$medv)^2))  #CALCULATE RMSE
mae_rf = mean(abs(pred_rf - test$medv))   #CALCULATE MAE
rmse_rf
mae_rf

#BOOSTING
#TRAIN MODEL
set.seed(123)
boost_medv=gbm(medv~.,data=train,distribution="gaussian",n.trees = 10000, interaction.depth = 8,shrinkage = 0.001,verbose = F,cv.folds = 5)
pred_boost=predict(boost_medv ,newdata=test,n.trees=10000) #PREDICT ON TEST DATA
rmse_boost = sqrt(mean((pred_boost - test$medv)^2))   #CALCULATE RMSE
mae_boost = mean(abs(pred_boost - test$medv))   #CALCULATE MAE
rmse_boost
mae_boost

#BAGGING
#TRAIN MODEL
set.seed(123)
bag_medv = randomForest(medv ~ ., data = train, mtry = 13, ntree = 10000, importance = TRUE)
pred_bag = predict(bag_medv, newdata = test)     #PREDICT ON TEST DATA
rmse_bag = sqrt(mean((pred_bag - test$medv)^2))  #CALCULATE RMSE
mae_bag = mean(abs(pred_bag - test$medv))    #CALCULATE MAE
rmse_bag
mae_bag