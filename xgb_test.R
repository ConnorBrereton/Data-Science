setwd("/Users/connor/Desktop/dev/xgboost/xgboost")
list.files(pattern = "*.csv")
unsw_t = read.csv("UNSW_NB15_testing-set.csv", header=T, nrows=50000)
install.packages("xgboost")
require(xgboost)
library(xgboost)

# Data Inspection ---------------------------------------------------------
head(unsw_t) #preview first 6 rows
str(unsw_t) #get variable list with types
dim(unsw_t) #dimensions

#unique levels within 'proto'
levels(unsw_t$proto)
outcome=table(unsw_t$attack_cat) #get counts of labeled attacks 
prop.table(outcome)

# Data Prep ---------------------------------------------------------------

#binary indicators
proto_ind=model.matrix( ~proto - 1, data=unsw_t) #convert proto
service_ind=model.matrix( ~service - 1, data=unsw_t) #convert 
state_ind=model.matrix( ~state - 1, data=unsw_t) #convert proto

#recode outcome into numeric index in one variable
unsw_t$attack_ind[unsw_t$attack_cat=="Analysis"]=0
unsw_t$attack_ind[unsw_t$attack_cat=="Backdoor"]=1
unsw_t$attack_ind[unsw_t$attack_cat=="DoS"]=2
unsw_t$attack_ind[unsw_t$attack_cat=="Exploits" ]=3
unsw_t$attack_ind[unsw_t$attack_cat=="Fuzzers"]=4
unsw_t$attack_ind[unsw_t$attack_cat=="Normal" ]=5
unsw_t$attack_ind[unsw_t$attack_cat== "Reconnaissance" ]=6
unsw_t$attack_ind[unsw_t$attack_cat=="Shellcode" ]=7
unsw_t$attack_ind[unsw_t$attack_cat=="Worms" ]=8

unsw_t[222:300, c("attack_cat", "attack_ind")]


#drop categorical variables 
outcome=unsw_t$attack_ind #separate outcome labels for sparse matrix
drops=c("proto", "service", "state", "attack_cat", "attack_ind")
unsw_t=unsw_t[ , !(names(unsw_t) %in% drops)]
dim(unsw_t)

#Create Design Matrix
design.matrix=as.matrix(cbind(unsw_t, proto_ind, service_ind, state_ind))
dim(design.matrix) #new dimensions
head(design.matrix)
str(design.matrix)

# Create sparse matrix
unsw_t_sparse = xgb.DMatrix(design.matrix, label=outcome)
dim(unsw_t_sparse)

head(unsw_t_sparse)


bstSparse = xgboost(data = unsw_t_sparse, label = outcome,
                   objective = "multi:softmax", num_class = 10, nrounds = 20)

#### EXTRA

set.seed(101) # Set Seed so that same sample can be reproduced in future also

pred <- predict(bstSparse, unsw_t_sparse$data)
print(length(pred))

print(head(pred))

prediction <- as.numeric(pred > 0.5)
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error", err))







