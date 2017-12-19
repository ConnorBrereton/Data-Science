# Set up environment ------------------------------------------------------
setwd("/Users/angold/Downloads")
list.files(pattern="*.csv")
unsw=read.csv("unsw.train.csv",header=T)
options(digits=2)
library (xgboost)
library(caret)

# Data Inspection ---------------------------------------------------------
head(unsw) #preview first 6 rows
str(unsw) #get variable list with types
dim(unsw) #dimensions
unique(levels(unsw$attack_cat)) #labels of attack classes

#counts and proportions
counts=table(unsw$attack_cat) #get counts of labeled attacks
prop.table(counts)*100 # proportion by attack category--note unbalanced classes

# Data Prep ---------------------------------------------------------------
attack_labels=as.factor(unsw$attack_cat) #save list for for grid search 

#binary indicators
proto_ind=model.matrix( ~proto - 1, data=unsw) #convert proto
service_ind=model.matrix( ~service - 1, data=unsw) #convert service
state_ind=model.matrix( ~state - 1, data=unsw) #convert proto

#recode outcome
unsw$attack_ind=as.numeric(unsw$attack_cat)-1
unsw[222:300, c("attack_cat", "attack_ind")]
table(unsw$attack_cat,unsw$attack_ind)

#drop categorical variables
outcome=unsw$attack_ind #separate outcome labels for sparse matrix
drops=c("proto", "service", "state", "attack_cat", "attack_ind")
unsw=unsw[ , !(names(unsw) %in% drops)]
dim(unsw)

#Create Design Matrix
design.matrix=as.matrix(cbind(unsw, proto_ind, service_ind, state_ind))
dim(design.matrix) #new dimensions

# Create sparse matrix
unsw_sparse = xgb.DMatrix(design.matrix, label=outcome)

# Grid Search for Parameters ----------------------------------------------

# set up the cross-validated hyper-parameter search
xgb.grid = expand.grid(
  nrounds = c(4,6,8),
  eta = c(0.01, 0.001, 0.0001), #step size / learning rate & is inversely related to number of rounds 
  max_depth = c(4, 10, 20, 30, 40), #too high can cause to overfitting
  gamma = 1,  #Regularization shrinkage parameter
  colsample_bytree = .6,# adds randomness to avoid overfitting 
  min_child_weight = .5, #linked to overfitting
  subsample = .8 #adds randomness to avoid overfitting 

)

# establish the training control parameters
xgb.controls = trainControl(
  method = "cv",
  number = 3,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",  # save losses across all models
  classProbs = TRUE,    # set to TRUE for AUC to be computed
  summaryFunction = multiClassSummary, #for multiclass prediction
  allowParallel = TRUE
)

# train the model for each parameter combination in the grid & using CV to evaluate
xgb.train = train(
  x = design.matrix,
  y = attack_labels,
  trControl = xgb.controls,
  tuneGrid = xgb.grid,
  method = "xgbTree"
)

# scatter plot of the prediction accuracy against max_depth and eta
ggplot(xgb.train$results, aes(x = as.factor(eta), y = max_depth, size = Accuracy, color = Accuracy)) +
  geom_point() +
  theme_bw() +
  scale_size_continuous(guide = "none")

grid.out=as.data.frame(summary(xgb.train))
str(xgb.train)
xgb.train$bestTune
xgb.train$finalModel

grid.results=as.data.frame(xgb.train$results)
head(grid.results)
grid.results$Accuracy

print(c("The best parameter values are", xgb.train$bestTune))
# Cross Validation for Parameters  ----------------------------------------
param=list("objective" = "multi:softmax",
          "eval_metric" = "merror", 
          "eta"=1, "num_class" =10,
          "max_depth" = 40) # will be [1, 191]we would have to run a real 
                            # iterative routine to optimize this param
num_class = 10 #uses 0 indexing with label=[0,9]
bst.cv=xgb.cv(param=param, data=unsw_sparse, Label=outcome, nfold=10,
              num_class = 10, nrounds = 10, prediction = TRUE) #bump to 400 
cvlog=as.data.frame(bst.cv$evaluation_log)
plot(cvlog$test_merror_mean, type="l", lty=6, lwd=2,col="red",
     xlab="Number of Rounds",
     ylab="Multiclass Error")

# Model Training ----------------------------------------------------------
num_class = 10 #uses 0 indexing with label=[0,9]
bstSparse = xgboost(data = unsw_sparse, Label = outcome,
                    objective = "multi:softmax", num_class = 10, nrounds = 20,
                    max_depth=10, colsample_bytree=.6) # add in parameters for imbalanced classes

# Testing Model -----------------------------------------------------------
test.unsw=read.csv("unsw.train.csv",header=T)
counts=table(test.unsw$attack_cat)
props=prop.table(counts)
100*props # check for imbalanced classes 

#binary indicators
proto_ind.t=model.matrix( ~proto - 1, data=test.unsw) #convert proto
service_ind.t=model.matrix( ~service - 1, data=test.unsw) #convert service 
state_ind.t=model.matrix( ~state - 1, data=test.unsw) #convert proto

#recode outcome into numeric index in one variable
test.unsw$attack_ind=as.numeric(test.unsw$attack_cat)-1
test.unsw[222:300, c("attack_cat", "attack_ind")]
table(test.unsw$attack_cat,test.unsw$attack_ind)
test.unsw[222:300, c("attack_cat", "attack_ind")]
test_label=test.unsw$attack_ind #create list of test labels

#drop categorical variables
test.outcome=test.unsw$attack_ind #separate outcome labels for sparse matrix
drops=c("proto", "service", "state", "attack_cat", "attack_ind")
test.unsw=test.unsw[ , !(names(test.unsw) %in% drops)]
dim(test.unsw)

#Create Design Matrix
design.matrix=as.matrix(cbind(test.unsw, proto_ind.t, service_ind.t, state_ind.t))

# Create sparse matrix
test.unsw_sparse = xgb.DMatrix(design.matrix)
#label=outcome)
str(test.unsw_sparse)
dim(test.unsw_sparse)


################
###Prediction###
################

pred <- predict(bstSparse,test.unsw_sparse) #prediction

#Snapshot of observations/prediction
test_label[1:20]
pred[1:20] #Snapshot of observations/prediction

matrix.counts=table(test_label, pred)
prop.table(matrix.counts, 2)
match=sum(as.numeric(pred) == test_label) #counts number of correct predictions
match/nrow(test.unsw_sparse) #gives precitions accuracy

prop=prop.table(matrix.counts, 2)
prop.df=as.data.frame((prop))


# Visualization -----------------------------------------------------------

#build confusion matrix
confusion = as.data.frame(table(test_label, pred)) #counts
names(confusion) = c("actual","predicted","freq") #name columns in count table
names(prop.df) = c("actual","predicted","percent") # names for proportion tablee
as.data.frame(table(test_label, pred)) #

# first we draw tiles and fill color based on percentage of test cases
tile <- ggplot() +
  geom_tile(aes(x=actual, y=predicted,fill=percent),data=prop.df, color="black",size=0.1) +
  labs(x="Actual",y="Predicted")
tile.c = tile + scale_fill_gradient(low="white", high="green") 
tile.l= tile.c + geom_text(aes(x=actual, y=predicted,label=round(percent,2)), data= prop.df)
#label=sprintf("%.1f", Percent)),

  
#Variable Importance
train.matrix= xgb.DMatrix(design.matrix, label=test.outcome)  #remove "label"
head(design.matrix)
train.model <- xgb.train(params = param, 
                         data = train.matrix,
                         nrounds = 10)  
names=names <-  colnames(design.matrix)
importance_matrix = xgb.importance(feature_names = names, model = train.model)
head(importance_matrix)
gp = xgb.plot.importance(importance_matrix[2:16], xlab="Gain", ylab="Feature Name")

  
# Extra -------------------------------------------------------------------
# lastly we draw diagonal tiles. We use alpha = 0 so as not to hide previous layers but use size=0.3 to highlight border



# Recode to add labels ----------------------------------------------------
#Observation recoding
prop.df$observed[prop.df$actual==0]="Analysis"
prop.df$observed[prop.df$actual==1]="Backdoor"
prop.df$observed[prop.df$actual==2]="DoS"
prop.df$observed[prop.df$actual==3]="Exploits"
prop.df$observed[prop.df$actual==4]="Fuzzers"
prop.df$observed[prop.df$actual==5]="Normal"
prop.df$observed[prop.df$actual==6]="Reconnaissance"
prop.df$observed[prop.df$actual==7]="Shellcode" 
prop.df$observed[prop.df$actual==8]="Worms"
prop.df$observed[prop.df$actual==9]="Generic"

#Prediction recoding
prop.df$predict[prop.df$predicted==0]="Analysis"
prop.df$predict[prop.df$predicted==1]="Backdoor"
prop.df$predict[prop.df$predicted==2]="DoS"
prop.df$predict[prop.df$predicted==3]="Exploits"
prop.df$predict[prop.df$predicted==4]="Fuzzers"
prop.df$predict[prop.df$predicted==5]="Normal"
prop.df$predict[prop.df$predicted==6]="Reconnaissance"
prop.df$predict[prop.df$predicted==7]="Shellcode" 
prop.df$predict[prop.df$predicted==8]="Worms"
prop.df$predict[prop.df$predicted==9]="Generic"

head(prop.df)


tile