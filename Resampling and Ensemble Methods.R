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
# 1. Default 1/0 indicator (180+ DPD):
df$def <- 0
# Save the indices (rows) of 
tmp <- which(df$F180_DTE == df$date) 
df$def[tmp] <- 1

# 2. Replace NUM_UNIT with MULTI_UNIT dummy:
table(df$NUM_UNIT)
df$MULTI_UN <- 0
tmp <- which(df$NUM_UNIT > 1)
df$MULTI_UN[tmp] <- 1

# 3. Count the number of loans:
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
my.vars <- c("ORIG_CHN","ORIG_RT",
             "ORIG_AMT","ORIG_TRM","OLTV",
             "DTI","OCC_STAT",
             "MULTI_UN",
             "CSCORE_MN",
             "ORIG_VAL",
             "VinYr","def.max")
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

#generate dummy variables
library("fastDummies")
train_x <- subset(down_train, select = -Class)
train_y <- as.numeric(down_train$Class)
train_dummy <- dummy_cols(train_x, select_columns = c("ORIG_CHN", "OCC_STAT"), remove_first_dummy = TRUE, remove_selected_columns = TRUE)
train_dummy <- data.frame(cbind(train_dummy,train_y))
names(train_dummy)[12] <- "Class"

#The validation set approach
N <- 10
lm.fit <- list(NA)
MSE <- rep(NA,length=N) 
for (i in 1:N){
  set.seed(100)
  lm.fit[[i]] <- lm(Class~poly(ORIG_AMT,i),data=train_dummy,subset=train) 
  MSE[i] <- mean((Class-predict(lm.fit[[i]],train_dummy))[-train]^2)
} 
plot(y=MSE,x=c(1:N),type="b")

attach(train_dummy)

S <- c(1:10)
N <- 7
MSE <- matrix(NA,nrow=N,ncol=10) 
for(j in 1:10){
  set.seed(j)
  train <- sample(4458,2229) 
  for (i in 1:N){
    lm.fit <- lm(Class~poly(ORIG_AMT,i),data=train_dummy,subset=train)
    MSE[i,j] <- mean((Class-predict(lm.fit,train_dummy))[-train]^2) }
}
library("reshape2") 
library("ggplot2")

df <- as.data.frame(MSE)
df$k <- c(1:N)
df.melt <- melt(df,id="k")
p <- ggplot(df.melt,aes(x=k,y=value,col=variable)) +
  geom_line() + theme(legend.position = "none") +
  labs(x="Polynomial Order", y = "MSE",title= "Validation Set Approach")
print(p)

library("glmnet")
library("boot") 
attach(train_dummy)

#LOOCV approach
S <- c(1:10)
N <- 10
cv.err <- matrix(NA,nrow=N,ncol=1) 
for(j in 1:1){
  set.seed(j)
  train <- sample(4458,2229) 
  for (i in 1:N){
    glm.fit <- glm(Class~poly(ORIG_AMT,i),data=train_dummy)
    cv.err[i,j] <- cv.glm(train_dummy,glm.fit)$delta[1] }
}
library("reshape2")
library("ggplot2")

df <- as.data.frame(cv.err)
df$k <- c(1:N)
df.melt <- melt(df,id="k")
p <- ggplot(df.melt,aes(x=k,y=value,col=variable)) +
  geom_line() + theme(legend.position = "none") +
  labs(x="Polynomial Order", y = "CV Error",title= "LOOCV Approach")
print(p)

#K-fold approach

attach(train_dummy)
S <- c(1:10)
N <- 10
cv.err.k <- matrix(NA,nrow=N,ncol=10)
for(j in 1:10){ 
  set.seed(1)
  for (i in 1:N){
    glm.fit <- glm(Class~poly(ORIG_AMT,i),data=train_dummy)
    cv.err.k[i,j] <- cv.glm(train_dummy,glm.fit,K=j+1)$delta[1] }
}
df <- as.data.frame(cv.err.k)
df$k <- c(1:N)
df.melt <- melt(df,id="k")
p <- ggplot(df.melt,aes(x=k,y=value,col=variable)) +
  geom_line() +
  labs(x="Polynomial Order", y = "CV Error",title= "k-Fold CV Approach")
print(p)

#Ensemble method

#Bagging
library("ipred")

library("plyr") 
library("dplyr") 
library("e1071") 
library("ranger")

x <- subset(train_dummy, select=c(-Class))
y <- as.factor(down_train$Class)

# 9 fold cross validation:
fitControl <- trainControl(method = "cv",number=9)
# Set the tuning parameters:
grid <- expand.grid(.vars=ncol(x))

grid <- expand.grid(.mtry=ncol(x), .splitrule="gini",
                    .min.node.size=9)
bag.treebag <- train(x=x,y=y, method="ranger",
                     trControl = fitControl,
                     metric="Accuracy",
                     tuneGrid=grid,
                     num.trees=15)
print(bag.treebag)

print(bag.treebag$finalModel)

b <- c(1,seq(10,500,by=10)) 
oob.error <- c(NA) 
bag.treebag <- list(NA)

for (i in 1:length(b)){ 
  bag.treebag[[i]] <- train(x=x,y=y,
                            method="ranger",
                            trControl = fitControl,
                            metric="Accuracy",
                            tuneGrid=grid,
                            num.trees=b[i])
  oob.error[i] <- bag.treebag[[i]]$finalModel$prediction.error
  print(i) 
}
plot(oob.error)

which.min(oob.error) #28
#b= 290
oob.error[28]

#Random forest
b <- c(1,seq(50,200,by=10))
m <- c(2:ncol(x))
oob.error.rf <- matrix(NA,ncol=length(m),
                       nrow=length(b))
rf.treebag <- list(NA) 
nn <- 1
for (j in 1:length(m)){
  grid <- expand.grid(.mtry=m[j], 
                      .splitrule="gini",
                      .min.node.size=5) 
  for (i in 1:length(b)){
                        rf.treebag[[nn]] <- train(x=x,y=y, 
                                                  method="ranger",
                                                  trControl = fitControl,
                                                  metric="Accuracy",
                                                  tuneGrid=grid,
                                                  num.trees=b[i])
                        oob.error.rf[i,j] <- rf.treebag[[nn]]$finalModel$prediction.error 
                        print(i)
                        nn <- nn + 1
                        print(nn) 
                        }
}

library("reshape")
library("ggplot2")
df.plot <- data.frame(b,oob.error.rf) 
names.tmp <- c("b")
for (i in 1:length(m)){
  names.tmp[i+1] <- paste0("mtry.",m[i]) 
  }
names(df.plot) <- names.tmp
melt.df.plot <- melt(df.plot,id="b")
p <- ggplot(melt.df.plot,aes(x=b,y=value,color=variable)) + geom_line()
print(p)

which.min(oob.error.rf) #16
b[16] #190
oob.error.rf[16]

#Boosting

library("gbm")
library("caret")
fitControl <- trainControl(method = "cv",number=9)
# Convert character vars to factors
for (i in 1:ncol(x)){ 
  print(class(x[,i]))
}

# Set the tuning parameters:
grid <- expand.grid(.n.trees=c(100,1000), .interaction.depth=c(2:5),
                    .shrinkage=c(0.001,0.005,0.1), .n.minobsinnode=5)
boost.gbm <- train(x=x,y=y, 
                   method="gbm",
                   trControl = fitControl,
                   metric="Accuracy",
                   tuneGrid=grid)
summary(boost.gbm) 
print(boost.gbm$modelInfo) 
results.gbm <- boost.gbm$results 

#view the gbm results
library("knitr")
kable(as.data.frame(results.gbm),digits=c(4,0,0,0,4,4,4,4),format.args = list(big.mark=","))



                          