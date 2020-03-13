rm(list=ls()) 
setwd("/Users/zihan/Desktop/2020\ spring/AI\ for\ finance/") 
getwd()
library("ggplot2")
library("reshape")
library("plm")
library("rpart") 
library("zoo") 
library("plyr") 
library("dplyr")
library("stringr") 
library("reshape2") 
library("ggplot2") 
library("pander") 
library("DataCombine") 
library("plm") 
library("quantmod")

# Import the mortgage data:
load("Motrgage_Annual.Rda")

# Rename the data (matter of preference):
df <- p.mort.dat.annual
rm(p.mort.dat.annual)

df <- pdata.frame(df, index=c("LOAN_ID","year"),
                  stringsAsFactors = F)

# Print the class of the variable:
class(df) 

# Generate Variables we want:
# Default 1/0 indicator (180+ DPD):
df$def <- 0
# Save the indices (rows) of 
tmp <- which(df$F180_DTE == df$date) 
df$def[tmp] <- 1

# Replace NUM_UNIT with MULTI_UNIT dummy:
table(df$NUM_UNIT)
df$MULTI_UN <- 0
tmp <- which(df$NUM_UNIT > 1)
df$MULTI_UN[tmp] <- 1

# Count the number of loans:
print(length(unique(df$LOAN_ID)))

# Compress the data to single loans:
df.annual <-df %>% 
  group_by(LOAN_ID) %>%
  mutate(def.max = max(def)) %>%
  mutate(n = row_number()) %>%
  ungroup()

# Print the variable names in df.annual
names(df.annual)

# keep one obs per loan:
tmp <- which(df.annual$n == 1)
df.annual <- df.annual[tmp,]
dim(df.annual)

# Keep only relevant variables for default analysis:
my.vars <- c("ORIG_RT","ORIG_AMT","ORIG_TRM","OLTV",
             "DTI","LAST_RT","CSCORE_B","Loan.Age",
             "MULTI_UN","CSCORE_MN","n.obs","n.year",
             "ORIG_VAL","VinYr","def.max")
df.model <- subset(df.annual,select=my.vars)
names(df.model)

# Print the number of defaults/non-defaults
table(df.model$def.max)
tmp <- table(df.model$def.max)
df.rate <- tmp[2]/sum(tmp)*100
message(sprintf("The default rate is: %4.2f%%",df.rate))

# Print the objects in memory:
ls()

# Remove all but df.model
rm(list=setdiff(ls(), "df.model"))
ls()

library("caret")
head(df.model)

df.model.noNA <- df.model[complete.cases(df.model),]

# Select all except def.max
x <- subset(df.model.noNA, select=c(-def.max,-VinYr))
y <- as.factor(df.model.noNA$def.max)

# Up and down sampling examples:
down_train <- downSample(x = x[, -ncol(x)],
                         y = y)
table(down_train$Class)
names(down_train)
down_train$Class <- as.numeric(as.character(down_train$Class))

library(ISLR)
library(leaps)

#validation set approach
set.seed(1)
train<-sample(c(T,F),nrow(down_train),rep=T)
test<-!train

train.mat<-model.matrix(Class~.,data=down_train[train,])
test.mat<-model.matrix(Class~.,data=down_train[test,])

# best subset selection
regfit.best<-regsubsets(Class~.,data=down_train[train,], nvmax=10)

# calculate MSE of test dataset
val.errors<-rep(NA,10)
for(i in 1:10){
  coefi<-coef(regfit.best,i)
  pred<-test.mat[,names(coefi)]%*%coefi
  val.errors[i]<-mean((down_train$Class[test]-pred)^2)
}

which.min(val.errors)

regfit.best<-regsubsets(Class~.,data=down_train,nvmax=10)
coef(regfit.best,10)

min(val.errors)

# forward stepwise selection
regfit.fwd<-regsubsets(Class~.,data = down_train[train,],nvmax=10,method="forward")

# calculate the MSE
fwd.errors<-rep(NA,10)
for(i in 1:10){
  coefi<-coef(regfit.fwd,i)
  pred<-test.mat[,names(coefi)]%*%coefi
  fwd.errors[i]<-mean((down_train$Class[test]-pred)^2)
}

which.min(fwd.errors)
regfit.fwd<-regsubsets(Class~.,data=down_train,nvmax=10)
coef(regfit.fwd,10)
min(val.errors)

#back stepwise selection
regfit.bwd<-regsubsets(Class~.,data = down_train[train,],nvmax=10,method="backward")

# calculate the MSE
bwd.errors<-rep(NA,10)
for(i in 1:10){
  coefi<-coef(regfit.bwd,i)
  pred<-test.mat[,names(coefi)]%*%coefi
  bwd.errors[i]<-mean((down_train$Class[test]-pred)^2)
}

which.min(bwd.errors)
regfit.bwd<-regsubsets(Class~.,data=down_train,nvmax=10)
coef(regfit.bwd,10)
min(bwd.errors)

# draw the plots to show pattern of change of MSEs
plot(val.errors,type = "o", col = "green", xlab = "models", ylab = "best subset errors",
     main = "best subset errors plot")
plot(bwd.errors,type = "o", col = "red", xlab = "models", ylab = "backward errors",
     main = "backword errors plot")
plot(fwd.errors,type = "o", col = "blue", xlab = "models", ylab = "forward errors",
     main = "forward errors plot")

# Principal Components Regression 
train_x <- subset(down_train, select = -Class)
library(psych)
fa.parallel(train_x[,], fa="pc", n.iter=100, show.legend=FALSE, main="Scree plot with parallel analysis")
pc <- principal(train_x[,], nfactors=5)
pc
