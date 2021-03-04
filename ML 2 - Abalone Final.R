####################### MACHINE LEARNING ASSIGNMENT 2 ##########################
###################### PREDICITING THE AGE OF ABALONES #########################


################## INSTALL PACKAGES AND IMPORT LIBRARIES #######################


#install.packages("readr")
#install.packages("knitr")
#install.packages("data.table")
#install.packages("tm")
#install.packages("stringr")
#install.packages("skimr")
#install.packages("ggplot2")
#install.packages("psych")
#install.packages("scales")
#install.packages("tree")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("randomForest")
#install.packages("gbm")
#install.packages("neuralnet")
#install.packages("keras")

library(readr)
library(knitr)
library(data.table)
library(tm) 
library(stringr)
library(skimr)
library(ggplot2)
library(psych)
library(scales)
library(tree)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(neuralnet)
library(keras)


########################### IMPORT AND CLEAN DATA ##############################


abaloneDataURL <- 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'
data <- fread(abaloneDataURL)

df = data.frame(data)

# Change Column Names

colnames(df) 

names(df)[names(df) == "V1"] <- "sex"
names(df)[names(df) == "V2"] <- "length"
names(df)[names(df) == "V3"] <- "diameter"
names(df)[names(df) == "V4"] <- "height"
names(df)[names(df) == "V5"] <- "whole_weight"
names(df)[names(df) == "V6"] <- "schucked_weight"
names(df)[names(df) == "V7"] <- "viscera_weight"
names(df)[names(df) == "V8"] <- "shell_weight"
names(df)[names(df) == "V9"] <- "rings"

# Check Structure and change sex to factor and rings to numeric

str(df)

df$sex <- as.factor(df$sex)
df$rings <- as.numeric(df$rings)

### Scale all numeric predictors back to original numbers (x*200)

df[,2:8] <- df[,2:8]*200


######################### EXPLORATORY DATA ANALYSIS ###########################


# Check if any rows are missing data points

sum(is.na(df)) # 0

### Check for outliers and null observations

cont_variables = df[,-c(1)]

colours = c(rep("black", nrow(df)))

pairs.panels(cont_variables, col = colours)

pairs(cont_variables, col = colours, pch=16) #Two points appear to be outliers looking at height

which(df$height > 100) # 1418 and 2052 are outliers

colours[c(1418, 2052)] <- "red"

pairs.panels(cont_variables, col = colours)

pairs(cont_variables, col = colours, pch=16) 

colSums(df == 0) # Height also has two observations where height == 0

which(df$height == 0) #1258, 3997

### Remove the outliers and height == 0 rows, reassign cont_variables

df = df[-c(1258, 1418, 2052, 3997),]

cont_variables = df[,-c(1)]

pairs.panels(cont_variables, col = colours)

### Continuous Variables

summary(cont_variables) #M, min, max
sapply(cont_variables, sd, na.rm = TRUE) #SD

cor_matrix = cor(cont_variables)
round(cor_matrix, 2)

### Boxplots for dimension and weight groupings

boxplot(cont_variables[1:3], 
        col = c("cornflowerblue","lightgreen","orange"),
        main = "Dimension Variables for Abalones",
        ylab = "Measurement (mm)")

boxplot(cont_variables[4:7], main = "Weight Variables for Abalones", 
        col = c("cornflowerblue","lightgreen","orange", "pink"),
        ylab = "Measurement (grams)")

# Boxplot for response variable

boxplot(df$rings, col = "cornflowerblue",
        xlab = "Rings", ylab = "Number of Rings")

# Discrete Variables (just "sex")

xtabs(~sex, df)

# Bar Chart

ggplot(df, mapping = aes(x = sex))+  
  geom_bar(col = 'black', fill = 'cornflowerblue')+   
  xlab('Sex') + ylab('Frequency')+
  geom_text(stat="count",aes(label=..count..),
            position=position_stack(0.5), color="white", size=5)


########################## MODELLLING PROCESS ##################################


### Create Training and Test at 70% split

set.seed(50) 
index = sample(1:4173, 4173*0.7)
df_train = df[index, ]
df_test = df[-index, ]


########################## REGRESSION TREES ####################################

##### Standard Tree

tree_model = tree(rings ~., data = df_train)
summary(tree_model)

# Plot using R part

plot_fit = rpart(rings ~., data = df_train)
rpart.plot(plot_fit, uniform=TRUE, main = "Regression Tree for Rings")

# Predict and test

predictions = predict(tree_model, df_test)
mean((df_test$rings-predictions)^2)

### Pruned Tree

prune_model = cv.tree(tree_model,FUN=prune.tree)
plot(prune_model)
title("Deviance as Size of Pruned Tree Increases", line = +3)

prune_model = prune.tree(tree_model, best = 7)
summary(prune_model)

plot(prune_model)
text(prune_model,pretty=0)

### Bagged Tree

# Check which number of bagged trees returns best model

mse_vec=c()

for (i in 1:100){

  bag_model = randomForest(rings~., df_train, mtry=8, ntree=(i*5), importance =TRUE)
  
  pred = predict(bag_model, df_test)
  
  mse = mean((pred - df_test$rings)^2)
  
  mse_vec[i] = mse

}

mse_vec[which.min(mse_vec)]

tree_number = seq(from = 5, to = 500, by = 5)

colours = c(rep("black", 100))
colours[c(20, 61)] <- c("green", "red")

# Performance does not increase after approx 100 trees
plot(tree_number, mse_vec, main = "MSE as Number of Trees Increases", 
     xlab = "Number of Trees", ylab = "MSE", pch = 16,
     col = colours) 

bag_model = randomForest(rings~.,df_train, mtry=8, ntree=100, importance =TRUE)


### Random Forest Tree

mse_values = double(8)

for (i in 1:8) {

rf_model = randomForest(rings~., data=df_train, mtry=i, ntree=100, importance = TRUE)

pred = predict(rf_model, df_test)

mse = mean((pred - df_test$rings)^2)

mse_values[i] = mse

}

colours = c(rep("black", 8))
colours[5] = "green"

plot(mse_values,main = "MSE as Number of Predictors Increases", # Lowest MSE with 5 predictors
     xlab = "Number of Predictors", ylab = "MSE", pch = 16,
     col = colours) 

rf_model = randomForest(rings~.,data=df_train, mtry=5, ntree=100, importance = TRUE)

varImpPlot(rf_model)
importance(rf_model)

### Boosted Tree

# Finding best interaction depth

mse_depth = double(10)

for (i in 1:10) {
    
    boost_model =gbm(rings~.,data=df_train, distribution="gaussian",
                     n.trees =100, interaction.depth =i)
    
    pred = predict(boost_model, df_test)
    
    mse = mean((pred - df_test$rings)^2)
    
    mse_depth[i] = mse
    
}

colours = c(rep("black", 10))
colours[c(4,10)] = c("green", "red")

plot(mse_depth, main = "MSE as Depth of Trees Increases", 
     xlab = "Depth", ylab = "MSE", pch = 16,
     col = colours)

# Finding best shrinkage parameter

shrink_index = seq(from = 0.002, to = 1, by = 0.002)

mse_shrinkage = c()

for (i in 1:500){
  boost_model =gbm(rings~.,data=df_train, distribution="gaussian",
                   n.trees =100, interaction.depth =4, shrinkage = (i/500))

  pred = predict(boost_model, df_test)
  
  mse = mean((pred - df_test$rings)^2)
  
  mse_shrinkage[i] = mse
  
}

colours = c(rep("black", 500))
colours[86] = c("red")
  
plot(shrink_index, mse_shrinkage, main = "MSE as Shrinkage Parameter Increases",
     xlab = "Lambda (Shrinkage Parameter)",ylab = "MSE", pch = 16,
     col = alpha(colours,0.8))

shrink_index[which.min(mse_shrinkage)] #0.172

boost_model =gbm(rings~.,data=df_train, distribution="gaussian",
                 n.trees = 100, interaction.depth = 4, shrinkage = 0.172)

summary(boost_model)


######################## REGRESSION TREE COMPARISON ########################


##### Comparing Performance Between all trees #####

repetitions = 100

mse_tree = c()
mse_prune = c()
mse_bag = c()
mse_rf = c()
mse_boost = c()

cor_tree = c()
cor_prune = c()
cor_bag = c()
cor_rf = c()
cor_boost = c()

for (i in 1:repetitions){
  
  #Set Seed
  set.seed(i)
  
  #Split Data
  training.obs = sample(1:4173, 4173*0.7 )
  df_train = df[training.obs, ]
  df_test = df[-training.obs, ]
  
  ##Train Models
  
  tree_model = tree(rings ~., data = df_train)
  prune_model = prune.tree(tree_model, best = 8)
  bag_model = randomForest(rings~.,df_train, mtry=8, ntree=100, importance =TRUE)
  rf_model = randomForest(rings~.,data=df_train, mtry=5, ntree=100, importance = TRUE)
  boost_model =gbm(rings~.,data=df_train, distribution="gaussian",
                   n.trees = 100, interaction.depth = 4, shrinkage = 0.148)
  
  #Make Predictions 
  
  pred_tree = predict(tree_model, df_test)
  pred_prune = predict(prune_model, df_test)
  pred_bag = predict(bag_model, df_test)
  pred_rf = predict(rf_model, df_test)
  pred_bag = predict(boost_model, df_test)
  
  # Compute and Store MSE
  
  mse_tree[i] = mean((pred_tree - df_test$rings)^2)
  mse_prune[i] = mean((pred_prune - df_test$rings)^2)
  mse_bag[i] = mean((pred_bag - df_test$rings)^2)
  mse_rf[i] = mean((pred_rf - df_test$rings)^2)
  mse_boost[i] = mean((pred_bag - df_test$rings)^2)
  
  cor_tree[i] = cor(pred_tree, df_test$rings)
  cor_prune[i] = cor(pred_prune, df_test$rings)
  cor_bag[i] = cor(pred_bag, df_test$rings)
  cor_rf[i] = cor(pred_rf, df_test$rings)
  cor_boost[i] = cor(pred_bag, df_test$rings)
  
}


boxplot(mse_tree, mse_prune, mse_bag, mse_rf, mse_boost,
        names = c('Standard Tree','Pruned Tree','Bagged Tree', 
                  'Random Forest Tree', "Boosted Tree"),
        main = "Test MSE for each Regression Tree Model over 100 Individual Tests",
        xlab = "Regression Tree Model", ylab = 'Test MSE',
        col = c("red","light green", "cornflowerblue", "violet", "orange"))

boxplot(cor_tree, cor_prune, cor_bag, cor_rf, cor_boost,
        names = c('Standard Tree','Pruned Tree','Bagged Tree', 
                  'Random Forest Tree', "Boosted Tree"), 
        main = "Test Correlation for each Regression Tree Model over 100 Individual Tests",
        xlab = "Regression Tree Model", ylab = 'Test Correlation', 
        col = c("red","dark green", "cornflowerblue", "violet", "orange"))
violet
mean(mse_rf) #4.67
mse_rf[which.min(mse_rf)]


########################### NEURAL NETWORKS ###################################

data = df

data$sex = as.numeric(data$sex)

data = as.matrix(data)

dimnames(data) <- NULL

set.seed(1234)
index = sample(1:4173, 4173*0.7 )
training = data[index,1:8]       
test = data[-index,1:8]       
training_y = data[index,9]   
test_y = data[-index,9]    

### Normalize Predictors

m = colMeans(training)
s = apply(training, 2, sd)
training = scale(training, center = m, scale = s)
test = scale(test, center = m, scale = s)

#Create

model <- keras_model_sequential()
model %>%
  layer_dense(units = 8, activation = "relu",
              input_shape = c(8)) %>%
  layer_dense(units = 1)
model

# Compile

model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_rmsprop(),
  metrics = c("mae")
)

#Fit 

model %>% fit(
  training, training_y,
  epochs = 100, batch_size = 32,
  validation_split = 0.2
)

# Evaluate

model %>% evaluate(test, test_y)
pred = predict(deep.net, test)

mean((test_y-pred)^2)
plot(test_y, pred)

# Fine Tune

# 4.74

model <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = "relu",
              input_shape = c(8)) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 1)
model

model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_rmsprop(),
  metrics = c("mae")
)
model %>% fit(
  training, training_y,
  epochs = 100, batch_size = 32,
  validation_split = 0.2
)

model %>% evaluate(test, test_y)

### Visualise!!!

data2 = df

data2$sex = as.numeric(data2$sex)
str(data2)

data2 = as.matrix(data2)


n <- neuralnet(rings~., data = data2,
               hidden = c(4),
               linear.output = F,
               lifesign = "full",
               rep=1)

plot(n,
     col.hidden = "darkblue",
     col.hidden.synapse = "darkblue",
     show.weights = F,
     information = F,
     bias = F,
     fill = "cornflowerblue")

############################### COMPARE MODELS ################################


#### RF

# Create data sets

set.seed(1)

training.obs = sample(1:4173, 4173*0.7 )
df_train = df[training.obs, ]
df_test = df[-training.obs, ]

# Train RF and make predictions and then rounded predictions

rf_model = randomForest(rings~.,data=df_train, mtry=5, ntree=100, importance = TRUE)

pred_rf = predict(rf_model, df_test)
mse_rf = mean((pred_rf-df_test$rings)^2)

round_pred_rf = round(pred_rf, digits = 0)

### RF classification rate

classification_vec_rf=c()

for (i in 1:1252){
  
  predicted = round_pred_rf[i]
  actual = df_test$rings[i]
  
  if (predicted == actual) {
    classification_vec_rf[i] <-1
  } else if (predicted == actual +1) {
    classification_vec_rf[i] <-1
  }else if (predicted == actual -1) {
    classification_vec_rf[i] <-1
  } else if (predicted == actual +2) {
    classification_vec_rf[i] <-1
  }else if (predicted == actual -2) {
    classification_vec_rf[i] <-1
  } else if (predicted == actual +3) {
    classification_vec_rf[i] <-1
  }else if (predicted == actual -3){
    classification_vec_rf[i] <-1
  }else {
    classification_vec_rf[i] <-0
  }
}

mean(classification_vec_rf)


# Train NN and make predictions

# Correct data for NN usage

data = df

data$sex = as.numeric(data$sex)

data = as.matrix(data)

dimnames(data) <- NULL

set.seed(1)
index = sample(1:4173, 4173*0.7 )
training = data[index,1:8]       
test = data[-index,1:8]       
training_y = data[index,9]   
test_y = data[-index,9]    

### Normalize Predictors

m = colMeans(training)
s = apply(training, 2, sd)
training = scale(training, center = m, scale = s)
test = scale(test, center = m, scale = s)

nn_model <- keras_model_sequential() %>%
  layer_dense(units = 8, activation = "relu",
              input_shape = c(8)) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 1)
nn_model

nn_model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_rmsprop(),
  metrics = c("mae")
)
nn_model %>% fit(
  training, training_y,
  epochs = 100, batch_size = 32,
  validation_split = 0.2
)

# Predictions

pred_nn = predict(nn_model, test)
mse_nn = mean((pred_nn-test_y)^2)

round_pred_nn = round(pred_nn, digits = 0)


## NN Classification Rate

classification_vec_nn=c()

for (i in 1:1252){
  
  predicted = round_pred_nn[i]
  actual = test_y[i]
  
  if (predicted == actual) {
    classification_vec_nn[i] <-1
  } else if (predicted == actual +1) {
    classification_vec_nn[i] <-1
  }else if (predicted == actual -1) {
    classification_vec_nn[i] <-1
  } else if (predicted == actual +2) {
    classification_vec_nn[i] <-1
  }else if (predicted == actual -2) {
    classification_vec_nn[i] <-1
  } else if (predicted == actual +3) {
    classification_vec_nn[i] <-1
  }else if (predicted == actual -3){
    classification_vec_nn[i] <-1
  }else {
    classification_vec_nn[i] <-0
  }
}

mean(classification_vec_nn)