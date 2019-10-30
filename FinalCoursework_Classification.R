##CLASSIFICATION

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
library(reshape2)
library(ggplot2)
library(corrplot)
library('bestglm')



#LOAD DATASET
attach(Boston)
dt = data.table::as.data.table(Boston)

#CREATE NEW VARIABLE 'crate'
Boston$crate = "No"
Boston$crate[crim > median(crim)] = 'Yes'
Boston$crate = factor(Boston$crate)
table(Boston$crate)
dt = Boston[-drop(1)] #DROP OLD VARIABLE 'crim'


#SPLIT DATASET INTO TRAIN AND TEST SET
set.seed(123)
rows = sample(1:nrow(dt),nrow(dt)/2)
train = dt[rows,]
test = dt[-rows,]


#CREATE CORRELATION MATRIX
mydata=Boston[-drop(15)]
mydata.cor = cor(mydata)
corrplot.mixed(mydata.cor)

#FEATURE SELECTION
#USING stepAIC() FOR FORWARD AND BACKWARD SELECTION WITH BIC
set.seed(123)
logit = glm(crate~.,data = train, family = binomial) # CREATE LOGISTIC MODEL WITH ALL VARIABLES
logit.back=stepAIC(logit,trace = FALSE,direction = 'backward',k=log(nrow(train)))
logit.back$anova
logit.for=stepAIC(logit,trace = FALSE,direction = 'forward',k=log(nrow(train)))
logit.for$anova

#CHECK VIF
vif(logit)

#USING stepAIC() FOR FORWARD AND BACKWARD SELECTION WITH AIC
logit.back.aic=stepAIC(logit,trace = FALSE,direction = 'backward')
logit.back.aic$anova
logit.for.aic=stepAIC(logit,trace = FALSE,direction = 'forward')
logit.for.aic$anova

#USING bestglm() WITH AIC
bglm = bestglm(train,IC ='AIC',family = binomial)
bglm$BestModel
bglm$Subsets

#USING bestglm() WITH BIC
bglm.bic = bestglm(train,IC ='BIC',family = binomial)
bglm.bic$BestModel
bglm.bic$Subsets

#USING randomForest() FOR IMPORTANT FEATURES
set.seed(123)
rf_crate = randomForest(crate~.,data = train, importance =TRUE,mtry=4)
rf_crate$importance
varImpPlot(rf_crate) #CREATE PLOT OF IMPORTANT FEATURES


#BEST THREE MODEL
#RANDOM FOREST
set.seed(123)
###LOOCV
control <- trainControl(method='LOOCV')
set.seed(123)
#TRAIN MODEL
tunegrid <- expand.grid(.mtry=4)
rf_default <- train(crate~., 
                    data=train, 
                    method='rf', 
                    metric='Accuracy', 
                    tuneGrid=tunegrid, 
                    trControl=control)

rf.predict <- predict(rf_default, newdata = test )
confusionMatrix(rf.predict, test$crate )


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
control <- trainControl(method='CV', number = 25)
set.seed(123)
#mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=13)
bag_default <- train(crate~., 
                     data=train, 
                     method='treebag', 
                     metric='Accuracy', 
                     # tuneGrid=tunegrid, 
                     trControl=control)

bag.predict <- predict(bag_default, newdata = test )
confusionMatrix(bag.predict, test$crate )

#BOOSTING
set.seed(123)
# CHANGE THE VALUE 0 AND 1 INSTEAD OF 'YES' , 'NO'
Boston$crate = "0"
Boston$crate[crim > median(crim)] = '1'
Boston$crate = factor(Boston$crate)
table(Boston$crate)

# Drop old crim variable
dt = data.table::as.data.table(Boston[-drop(1)])
dt$crate=as.character(dt$crate)

#TRAIN MODEL
set.seed (123)
boost_boston=gbm(crate~.,data=train,distribution ='bernoulli',cv.folds = 15,n.trees=2000,interaction.depth = 10,shrinkage =0.001,verbose =F)
pred_boost=predict(boost_boston ,newdata=test,n.trees=2000,na.action=na.pass)
gbm.class<-ifelse(pred_boost<0.5,'0','1')
table(gbm.class, test$crate)
mean(gbm.class ==test$crate)