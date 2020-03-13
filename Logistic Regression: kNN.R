library(quantmod)
library(lubridate)
library(plyr)
library(glmnet)
library(PerformanceAnalytics)
library(TTR)
library(parallel)

rm(list = ls())

#get the monthly data of s&p500 adjusted value and the volume value
t1 <- "1900-01-01"
sp <- lapply("SPY", function(sym) get(getSymbols(sym,from = t1, periodicity = "monthly")) )
head(sp, 5)
vol <- lapply(sp, function(x) x[,5])
vol <- na.omit(Reduce(function(...) merge(...),vol ))
sp <- lapply(sp, function(x) x[,6])
sp <- na.omit(Reduce(function(...) merge(...),sp ))

#get the target variable which equals to 1 when return is positive and 0 when return is negative
dif <- diff(sp)
head(dif, 5)

for (i in 1:length(dif)){
  dif[i] <- if(is.na(dif[i]) || dif[i]>=0) 1 else 0
}
target <- tail(dif, 500)

#get the volume feature 
vol_tr <- tail(vol, 500)

#calculate the technical indicator data of s&p500 adjusted value
sma <-SMA(sp,n=20)
sma_tr <- tail(sma, 500)

ema <-EMA(sp,n=20)
ema_tr <- tail(ema, 500)

macd <- MACD(sp, nFast=12, nSlow=26, nSig=9, maType=SMA)
macd_tr <- tail(macd[,1],500)
signal_tr <- tail(macd[,2],500)

mon <- momentum(sp, n=2)
mon_tr <- tail(mon, 500)

roc <- ROC(sp,n=2)
roc_tr <- tail(roc, 500)

rsi <- RSI(sp, n=14)
rsi_tr <- tail(rsi, 500)

#get the data of cpi and hpi from FRED for the same time period
v <- c("CSUSHPISA", "CPIAUCSL") 
ind <- new.env()
getSymbols(v, src="FRED", env = ind, adjust = TRUE)
cpi = ind$CPIAUCSL
cpi <- tail(cpi, 500)
hpi <- ind$CSUSHPISA
hpi <- tail(hpi, 500)

#combine all the features
data_tr <- cbind(cpi, hpi, sma_tr, ema_tr, macd_tr, signal_tr, mon_tr, roc_tr, rsi_tr, vol_tr, target)
names(data_tr) <- c("cpi", "hpi", "sma", "ema", "macd", "signal", "mon", "roc", "rsi", "vol", "target")

#drop the first three rows as too many NAs
data_tr <- data_tr[4:nrow(data_tr),]
data_tr = na.omit(data_tr)

x <- model.matrix(target~.,data_tr)[,-1]
head(x)
y <- data_tr$target
head(y)

#divide the train dataseta and test dataset
set.seed(100)
train <- sample(1:nrow(x),nrow(x)*0.8)
test <- (-train)

x.test <- x[test,]
x.train <- x[train,]
y.test <- y[test]
y.train <- y[train]

#the mse calculation function
my.mse <- function(pred,act){
  mse <- mean((pred-act)^2)
  return(mse)
  }

#the set of lambda to be tested
grid <- 10^seq(10,-2,length=500)

#find the optimal lambda for ridge model
MSE.ridge <- c(NA)
ridge.mse <- glmnet(x.train,y.train,alpha=0,lambda=grid,thresh=1e-12)
ridge.mse

for(i in 1:length(grid)){
  ridge.pred.tmp <- predict(ridge.mse,s=grid[i],newx <- x.test)
  MSE.ridge[i] <- my.mse(ridge.pred.tmp,y.test)
}
plot(MSE.ridge)

#get the optimal lambda
lambda.ridge <- grid[which.min(MSE.ridge)]
sprintf("Optimal value of lambda for ridge is %1f",lambda.ridge)

#get the coefficients of features based on optimal lambda
ridge.mod = glmnet(x,y,alpha = 0,lambda = lambda.ridge)
str(ridge.mod)
print(ridge.mod$beta)

#get the mse of ridge model
min(MSE.ridge)

#find the optimal lambda for lasso model
MSE.lasso <- c(NA)
lasso.mse <- glmnet(x.train,y.train,alpha=1,lambda=grid,thresh=1e-12)

for(i in 1:length(grid)){
  lasso.pred.tmp <- predict(lasso.mse,s=grid[i],newx <- x.test)
  MSE.lasso[i] <- my.mse(lasso.pred.tmp,y.test)
  }
plot(MSE.lasso)

lambda.lasso <- grid[which.min(MSE.lasso)]
sprintf("Optimal value of lambda is %10f",lambda.lasso)

#get the coefficients of lasso based on optimal lambda
lasso.mod = glmnet(x,y,alpha = 1,lambda = lambda.lasso)
str(lasso.mod)
print(lasso.mod$beta)

#get the mse of lasso model
min(MSE.lasso)

#normalize the features
myNorm <- function(x) {
  minX <- matrix(rep(apply(x,2,min),dim(x)[1]),byrow=T,ncol=dim(x)[2])
  maxX <- matrix(rep(apply(x,2,max),dim(x)[1]),byrow=T,ncol=dim(x)[2]) 
  num <- x - minX
  denom <- maxX - minX
  normdat <- num/denom
  return (normdat) 
}
data.norm <- myNorm(x)

#set train and test data
set.seed(99)
ind <- sample(2,nrow(data.norm), replace=T, prob = c(0.8,0.2))

knn.train <- data.norm[which(ind==1),]
knn.test <- data.norm[which(ind==2),]

knn.train.lab <- y[which(ind==1),]
knn.test.lab <- y[which(ind==2),]

library("caret")

#combine the dataset for train and test
knn.data.train <- data.frame(knn.train,knn.train.lab) 
dim(knn.data.train)
names(knn.data.train)[11] <- "Movement"
head(knn.data.train)

knn.data.test <- data.frame(knn.test,knn.test.lab) 
names(knn.data.test)[11] <- "Movement"

#a set of k to be test 
k <- c(1,2,3,5,10,15,25,40,50,100)
accuracy <- c()
for (a in 1:length(k)){
  knn.model <- knn3(Movement~., data=knn.data.train, k = k[a],use.all=FALSE)
  #make the prediction using knn model
  dis <- predict(knn.model, newdata = data.frame(knn.test)) 
  head(dis,20)
  y.pre <- c()
  for (i in 1:length(dis[,1])){
    y.pre[i] <- if (dis[i,1] <= dis[i,2]) 1 else 0 
  }
  #caculate the accuracy of the prediction of knn model
  correct <- 0
  for (i in 1: length(y.pre)){
    if (y.pre[i] == knn.test.lab[i]) {
      correct <- correct +1}
    correct
  }
  accuracy[a] = correct / length(y.pre)
}
accuracy

