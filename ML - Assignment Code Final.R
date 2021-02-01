##################### MACHINE LEARNING ASSIGNMENT 1 #####################
########################### DIABETES DATA SET  ##########################

        ############## IMPORTING PACKAGES AND DATA SET ############

#install.packages('pls')
library(faraway)
library(ggplot2)
library(leaps)
library(glmnet)
library(pls)

df = diabetes


####################### EXPLORATORY DATA ANALYSIS ############################


              ######### DATA FRAME CHECK AND CLEAN  ###########

dim(df)
head(df)
help(diabetes) 

duplicated(df) #No rows are duplicated

sum(is.na(df)) #Lots of rows are missing values

head(df$bp.2s)#  bp.2s, bp.2d have a lot of missing values
head(df$bp.2d)

df<- na.omit(df) #Omiting missing values to calculate correlations below

cor(df$bp.2s, df$bp.1s)#They also have high correlations

df = diabetes #Resetting df to original

df <- df[,-c(1,15,16)] #Removing bp.2s, bp.2d, and id from dataframe

df<- na.omit(df) #Removing any remaining rows with missing data


        ###### CONTINUOUS VARIABLE INSPECTION AND EXPLORATION   #########


### chol, stab.glu, hdl, ratio, age, height,weight, 
### bp.1s, bp.1d, waist, hip, time.ppn


#Separate df for continuous predictors

continuous_predictors <- df[,-c(5,6,8,11)] #Remove response and discrete variables

# Descriptive statistics

summary(continuous_predictors) #M, min, max
sapply(continuous_predictors, sd, na.rm = TRUE) #SD

cor_matrix = cor(continuous_predictors)
round(cor_matrix, 2) # Inter Correlations

#Histogram with Density Overlay

ggplot(df, aes(x = hdl)) +
  geom_histogram(aes(y = ..density..), binwidth = 7.5,fill = "cornflowerblue", 
                 col = "black") +
  geom_density() +
  xlab("High Density Lipoprotein") + ylab("Density")+
  theme_minimal()


      ###########   DISCRETE VARIABLE INSPECTION AND EXPLORATION    ##########

### location, gender, frame

xtabs(~frame, data=df) # Count for each category/factor level

#Bar Charts for Discrete Variables

ggplot(df, mapping = aes(frame, fill = glyhb))+  
  geom_bar(col = 'black', fill = 'cornflowerblue')+   
  xlab('Frame') + ylab('Frequency')+
  geom_text(stat="count",aes(label=..count..),
            position=position_stack(0.5), color="white", size=5)



########################### REGRESSION ANALYSIS ##############################



      ######## METHOD 1 Part 1 - BSS Model Selection Criteria  ########

#BSS Models

bss = regsubsets(glyhb~., df, nvmax = 15)
results = summary(bss)
names(results)

#Store Results

RSS = results$rss
r2 = results$rsq
Cp = results$cp
BIC = results$bic
Adj_r2 = results$adjr2
cbind(RSS, r2, Cp, BIC, Adj_r2)

#Plot RSS and r2

par(mfrow = c(1, 2))
plot(RSS, xlab = "Number of Predictors", ylab = "RSS", type = "l", lwd = 2)
plot(r2, xlab = "Number of Predictors", ylab = "R-square", type = "l", lwd = 2)

#Find optimal model and plot under various criteria

which.min(Cp)
which.min(BIC)
which.max(Adj_r2)

par(mfrow = c(1, 3))
plot(Cp, xlab = "Number of Predictors", ylab = "Cp", type = 'l', lwd = 2)
points(6, Cp[6], col = "red", cex = 2, pch = 8, lwd = 2)
plot(BIC, xlab = "Number of Predictors", ylab = "BIC", type = 'l', lwd = 2)
points(3, BIC[3], col = "red", cex = 2, pch = 8, lwd = 2)
plot(Adj_r2, xlab = "Number of Predictors", ylab = "Adjusted RSq", type = "l", lwd = 2)
points(8, Adj_r2[8], col = "red", cex = 2, pch = 8, lwd = 2)



# Models through Model Selection Criteria

cp_bss = lm(glyhb~chol + stab.glu + ratio + 
                         location + age + time.ppn, df)

bic_bss = lm(glyhb~ stab.glu + ratio + age, df)

r2_bss = lm(glyhb~chol + stab.glu + ratio + 
              location + age + time.ppn + waist + frame, df)

#Choose BIC Model

coef(bic_bss,3)

# Final BSS Model through Model Selection Criteria

bic_bss = lm(glyhb~ stab.glu + ratio + age, df)

summary(bic_bss)



      ######## METHOD 1 Part 2 - BSS through Validation and CV  ########


#Function for making predictions

predict.regsubsets = function(object, newdata, id, ...){
  form = as.formula(object$call[[2]])
  mat = model.matrix(form, newdata)
  coefi = coef(object, id = id)
  xvars = names(coefi)
  mat[, xvars]%*%coefi
}

#Split Data into Train and Test

set.seed(10) 
training.obs = sample(1:366, 244)
df.train = df[training.obs, ]
df.test = df[-training.obs, ]

# Train Model

bss = regsubsets(glyhb~., data = df.train, nvmax = 16)


# Calculate MSE for Predictions based on each Model

val.error<-c()
for(i in 1:16){
  pred = predict.regsubsets(bss, df.train, i)
  val.error[i] = mean((df.train$glyhb - pred)^2)
}
val.error

which.min(val.error)

#CV returns model with 16 predictors, no reduction, for seed = 10

cv.bss = regsubsets(glyhb~., data = df, nvmax = 16)
coef(bss, 16)

### Changing seed and running 100 times - Run Lines 186 to 240 to compare CV vs Kfold

min.valid = c()
for(n in 1:100){
  set.seed(n)
  training.obs = sample(1:366, 244)
  df.train = df[training.obs, ]
  df.test = df[-training.obs, ]
  
  best = regsubsets(glyhb~., data = df.train, nvmax = 16)
  val.error<-c()
  for(i in 1:16){
    pred = predict.regsubsets(best, df.test, i)
    val.error[i] = mean((df.test$glyhb - pred)^2)
  }
  val.error
  min.valid[n] = which.min(val.error)
}

### BSS - K fold Cross Validation Approach

min.valid1 = c()

for(n in 1:100){
  set.seed(n)
  k = 10
  set.seed(n)
  folds = sample(1:k, 366, replace = TRUE)
  folds
  cv.errors = matrix(NA, k, 16, dimnames = list(NULL, paste(1:16)))
  for(j in 1:k){
    best.fit = regsubsets(glyhb~., data = df[folds!=j, ], nvmax = 16)
    for(i in 1:16){
      pred = predict(best.fit, df[folds==j, ], id = i)
      cv.errors[j,i] = mean( (df$glyhb[folds==j]-pred)^2)
    }
  }
  mean.cv.errors = apply(cv.errors, 2, mean)
  mean.cv.errors
  min.valid1[n] = which.min(mean.cv.errors)
}


#Plot both!

par(mfrow=c(1, 2))

hist(min.valid, col = "blue", nclass = 50, xlab = 'Number of Predictors', main = 'BSS with Validation')
abline(v = mean(min.valid), col = 2, lwd = 4)
legend('topright', legend=c('Average selection'),bty = 'n', lty = 1, lwd = 4, col = 2)

hist(min.valid1, col = "blue", nclass = 50, xlab = 'Number of Predictors', main = 'BSS with K Fold cross-validation')
abline(v = mean(min.valid1), col = 2, lwd = 4)
legend('topright', legend=c('Average selection'),bty = 'n', lty = 1, lwd = 4, col = 2)


### Final BSS model shown thorugh K fold with 8 Predictors as most stable

# Factors need to be made factors with I()

k_fold_bss = lm(glyhb~chol + stab.glu + ratio + I(location) + age + I(frame) + waist+
            time.ppn, data = df)

summary(k_fold_bss)




        ###############         METHOD 2 - LASSO           ###########




#Define Y  as the response variable

y = df$glyhb

#Define x as a model containin the predictors

x = model.matrix(glyhb~., df)[,-5] #Remove Response

lasso_model = glmnet(x, y)

lasso_model$lambda #Values of lambda 

dim(lasso_model$beta)

lasso_model$beta[,1:3] #Predictor Coefficients as lambda changes
coef(lasso_model)[,1:3] # Intercept + Predictors Coefficients as lambda changes

#Regularisation Paths

par(mfrow=c(1, 2))
plot(lasso_model, xvar = 'lambda')
plot(lasso_model) 

### Lasso with Cross Validation (CV)

#Make Model

set.seed(35)
lasso.cv = cv.glmnet(x, y)

#Find min and 1se lambda - min is much lower as expected

lasso.cv$lambda.min # 0.022 when seed = 35
lasso.cv$lambda.1se # 0.449 when seed = 35

round(cbind(
  coef(lasso.cv, s = 'lambda.min'),
  coef(lasso.cv, s = 'lambda.1se')),4)

par(mfrow=c(1,2))
plot(lasso.cv)
plot(lasso_model, xvar = 'lambda')
abline(v = log(lasso.cv$lambda.min), lty = 3)
abline(v = log(lasso.cv$lambda.1se), lty = 3)

#Which Lasso Model performs better in terms of prediction?

par(mfrow=c(1,1))

repetitions = 100
cor.1 = c()
cor.2 = c()
for(i in 1:repetitions){
  
  # Split Data
  
  set.seed(i)
  
  training.obs = sample(1:366, 244)
  y.train = df$glyhb[training.obs]
  x.train = model.matrix(glyhb~., df[training.obs, ])[,-1]
  y.test = df$glyhb[-training.obs]
  x.test = model.matrix(glyhb~., df[-training.obs, ])[,-1]
  
  # Train Model
  
  lasso.train = cv.glmnet(x.train, y.train)
  
  # Predict based on lambda min and 1se
  
  predict.1 = predict(lasso.train, x.test, s = 'lambda.min')
  predict.2 = predict(lasso.train, x.test, s = 'lambda.1se')
  
  #Look at Predictive Performance
  
  cor.1[i] = cor(y.test, predict.1)
  cor.2[i] = cor(y.test, predict.2)
}

#Plot

boxplot(cor.1, cor.2, names = c('min-CV lasso','1-se lasso'), ylab = 'Test correlation', col = 5)

#Coefficient Estimates at this point

coef(lasso.cv, s = 'lambda.min')




          #############         METHOD 3 - PCR           #############



set.seed(35)
pcr.fit=pcr(glyhb~., data=df, scale=TRUE, validation = 'CV')
summary(pcr.fit)

#Validate
dev.off()
validationplot(pcr.fit, val.type = 'MSEP', main = "") 

#Which Number of Components has lowest MSE

min.pcr = which.min(MSEP(pcr.fit)$val[1,1, ] ) - 1
min.pcr

coef(pcr.fit, ncomp = min.pcr) 

#Regularisation Paths 

coef.mat = matrix(NA, 16, 16)
for(i in 1:16){
  coef.mat[,i] = pcr.fit$coefficients[,,17-i]
}
plot(coef.mat[1,], type = 'l', ylab = 'Coefficients', 
     xlab = 'Number of components', ylim = c(min(coef.mat), max(coef.mat)))
for(i in 2:16){
  lines(coef.mat[i,], col = i)
}
abline(v = min.pcr, lty = 3)

#Coefficients for components = 10

coef(pcr.fit, ncomp = min.pcr) 

########################## PREDICTIVE PERFORMANCE ###########################

                    ########## COMPARE MODELS  ###########

#Predict Function Previously Defined in Lines X-Y

#Empty Vectors to store Correlation and MSE Values

cor.bss1 = c()
cor.bss2 = c()
cor.lasso = c()
cor.pcr = c()

mse.bss1 = c()
mse.bss2 = c()
mse.lasso = c()
mse.pcr = c()

#For Loop: This Runs 100 repetitions of varying seed to create 100 comparisons 
        #  between predictions and actual values

repetitions = 100
for(i in 1:repetitions){
  
  # Set Seed:
  
  set.seed(i)
  
  ### SPLIT DATA
  
  training.obs = sample(1:366, 244)
  
  # Split Data for Method 1 and 3 (BSS and PCR)
  
  df.train = df[training.obs, ]
  df.test = df[-training.obs, ]
  
  # Split Data for Method 2 (Lasso)
  
  x.train = model.matrix(glyhb~., df[training.obs,])[,-5]
  y.train = df$glyhb[training.obs]
  x.test = model.matrix(glyhb~., df[-training.obs,])[,-5]
  y.test = df$glyhb[-training.obs]
  
  ### BSS
  # Train BSS
  
  bss1 = regsubsets(glyhb~., data = df.train, nvmax = 15)
  bss2 = regsubsets(glyhb~., data = df.train, nvmax = 8)
  results1 = summary(bss1)
  results2 = summary(bss1)
  min.bic = which.min(results$bic)
  
  # Predict BSS
  
  predict.bss1 = predict.regsubsets(bss1, df.test, min.bic)
  predict.bss2 = predict.regsubsets(bss2, df.test, 8)
  
  # Evaluate BSS
  
  cor.bss1[i] = cor(df.test$glyhb, predict.bss1)
  cor.bss2[i] = cor(df.test$glyhb, predict.bss2)
  mse.bss1[i] = mean((df.test$glyhb - predict.bss1)^2)
  mse.bss2[i] = mean((df.test$glyhb - predict.bss2)^2)
  
  ### LASSO
  # Train Lasso
  lasso.cv = cv.glmnet(x.train, y.train)
  min.cv.lasso = lasso.cv$lambda.min
  
  # Predict Lasso
  
  predict.lasso = predict(lasso.cv, x.test, s = min.cv.lasso)
  
  # Evaluate Lasso
  
  cor.lasso[i] = cor(y.test, predict.lasso)
  mse.lasso[i] = mean((y.test - predict.lasso)^2)
  
  ### PCR
  
  # Train PCR
  pcr.fit=pcr(glyhb~., data=df, scale=TRUE, validation = 'CV')
  min.pcr = which.min(MSEP(pcr.fit)$val[1,1, ] ) - 1
  
  #Predict PCR
  predict.pcr = predict(pcr.fit, df.test, ncomp = min.pcr)
  
  #Evaluate PCR
  cor.pcr[i] = cor(y.test, predict.pcr)
  mse.pcr[i] = mean((df.test$glyhb - predict.pcr)^2)
}

#Plot


boxplot(cor.bss1, cor.bss2, cor.lasso, cor.pcr, names = c('BSS1','BSS2','Lasso', 'PCR'), 
  ylab = 'Test correlation', col = 2)

boxplot(mse.bss1, mse.bss2, mse.lasso, mse.pcr, names = c('BSS1', 'BSS2', 'Lasso', 'PCR'), 
        ylab = 'MSE', col = 2)


      ########## PCR MODEL USED FOR CLASSIFICATION   ##########

repetitions = 100

error_rate_vec = c()

for (i in 1:repetitions){
  
  #Set Seed
  set.seed(i)

  #Split Data
  training.obs = sample(1:366, 244)
  df.train = df[training.obs, ]
  df.test = df[-training.obs, ]
  
  # Actual Results
  y = rep(0, 122)
  y[df.test$glyhb > 7] = 1
  table(y)

  # Train Model
  pcr.fit.logit=pcr(glyhb~., data=df.train, scale=TRUE, validation = 'CV')
  
  #Make Predictions and turn into binary response
  gly = predict(pcr.fit.logit, df.test, ncomp = 10)
  pred <- ifelse(gly>7, "1", "0")
  
  #Make Confusion Matrix
  confusion_matrix = table(pred, y) 
  
  #Calculate and Store error rate in empty Vec
  error_rate_vec[i] = (confusion_matrix[1,2] + confusion_matrix[2,1])/122

}

boxplot(error_rate_vec, xlab ="PCR Model used for Classification", 
        ylab = 'Test Classification Error Rate', col = 2)

mean(error_rate_vec)

(1-mean(error_rate_vec))*100
