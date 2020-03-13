library(quantmod)
library(lubridate)
library(plyr)
library(glmnet)
library(PerformanceAnalytics)
library(TTR)
library(parallel)

rm(list = ls())

#get the data of s&p500 adjusted value and the volume value
t1 <- "2000-01-01"
sp <- lapply("SPY", function(sym) get(getSymbols(sym,from = t1, periodicity = "monthly")) )
head(sp, 5)
vol <- lapply(sp, function(x) x[,5])
vol <- na.omit(Reduce(function(...) merge(...),vol ))
sp <- lapply(sp, function(x) x[,6])
sp <- na.omit(Reduce(function(...) merge(...),sp ))

#get the return of s&p500
spr <- Return.calculate(sp)
sp_tr <- tail(sp, 100)
spr_tr <- tail(spr, 100)
vol_tr <- tail(vol, 100)

#calculate the technical indicator data of s&p500 adjusted value
sma <-SMA(sp,n=20)
sma_tr <- tail(sma, 100)

ema <-EMA(sp,n=20)
ema_tr <- tail(ema, 100)

macd <- MACD(sp, nFast=12, nSlow=26, nSig=9, maType=SMA)
macd_tr <- tail(macd[,1],100)
signal_tr <- tail(macd[,2],100)

mon <- momentum(sp, n=2)
mon_tr <- tail(mon, 100)

roc <- ROC(sp,n=2)
roc_tr <- tail(roc, 100)

rsi <- RSI(sp, n=14)
rsi_tr <- tail(rsi, 100)

#get the data of cpi and hpi from FRED for the same time period
v <- c("CSUSHPISA", "CPIAUCSL") 
ind <- new.env()
getSymbols(v, src="FRED", env = ind, adjust = TRUE)
cpi = ind$CPIAUCSL
cpi <- tail(cpi, 100)
hpi <- ind$CSUSHPISA
hpi <- tail(hpi, 100)

#combine all the features
data_tr <- cbind(cpi, hpi, sma_tr, ema_tr, macd_tr, signal_tr, mon_tr, roc_tr, rsi_tr, vol_tr, spr_tr)
names(data_tr) <- c("cpi", "hpi", "sma", "ema", "macd", "signal", "mon", "roc", "rsi", "vol", "spr")

#drop the first three rows as too many NAs
data_tr <- data_tr[4:nrow(data_tr),]
data_tr
data_tr = na.omit(data_tr)

#standardize the dataset
data_std <- apply(data_tr,2,sd,na.rm=T)
std.vars <- matrix(rep(data_std,nrow(data_tr)),ncol=ncol(data_tr),byrow = T)
data_std_tr <- data_tr / std.vars
#transfer the date type from DMatrix to matrix for future predict() function
data_std_tr <- as.matrix(data_tr / std.vars)

x <- data_std_tr[,1:10]
head(x)
y <- data_std_tr[, 11]
head(y)

#divide the train dataseta and test dataset
set.seed(1)
train <- sample(1:nrow(x),nrow(x)/2)
test <- (-train)

x.test <- x[test,]
x.train <- x[train,]
y.test <- y[test]
y.train <- y[train]

#the mse calculation function
my.mse <- function(pred,act){
  +     mse <- mean((pred-act)^2)
  +     return(mse)
  + }

grid <- 10^seq(10,-2,length=100)

#find the optimal lambda for ridge model
MSE.ridge <- c(NA)
ridge.mse <- glmnet(x.train,y.train,alpha=0,lambda=grid,thresh=1e-12)

for(i in 1:length(grid)){
  +     ridge.pred.tmp <- predict(ridge.mse,s=grid[i],newx <- x.test)
  +     MSE[i] <- my.mse(ridge.pred.tmp,y.test)
  + }
plot(MSE.ridge)

lambda.ridge <- grid[which.min(MSE.ridge)]
sprintf("Optimal value of lambda for ridge is %.1f",lambda.ridge)

#get the coefficients of features based on optimal lambda
ridge.mod = glmnet(x,y,alpha = 0,lambda = lambda.opt)
str(ridge.mod)
print(ridge.mod$beta)
#get the mse of ridge model
MSE.ridge [100]

#find the optimal lambda for lasso model
MSE.lasso <- c(NA)
lasso.mse <- glmnet(x.train,y.train,alpha=0.5,lambda=grid,thresh=1e-12)

for(i in 1:length(grid)){
  +     lasso.pred.tmp <- predict(lasso.mse,s=grid[i],newx <- x.test)
  +     MSE.lasso[i] <- my.mse(lasso.pred.tmp,y.test)
  + }
plot(MSE.lasso)

lambda.lasso <- grid[which.min(MSE.lasso)]
sprintf("Optimal value of lambda is %.1f",lambda.lasso)

#get the coefficients of lasso based on optimal lambda
lasso.mod = glmnet(x,y,alpha = 1,lambda = lambda.lasso)
str(lasso.mod)
print(lasso.mod$beta)

#get the mse of lasso model
MSE.lasso[100]
