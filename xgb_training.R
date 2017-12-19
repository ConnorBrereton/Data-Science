setwd('/Users/connor/Desktop/dev/xgboost/xgboost')
library(xgboost)

list.files(pattern = "*.csv")
unsw = read.csv('UNSW_NB15_training-set.csv', header=T, nrows=50000)
require(xgboost)

# Data Inspection ---------------------------------------------------------
head(unsw) #preview first 6 rows
str(unsw) #get variable list with types
dim(unsw) #dimensions

#unique levels within 'proto'
levels(unsw$proto)
outcome=table(unsw$attack_cat) #get counts of labeled attacks 
prop.table(outcome)

# Data Prep ---------------------------------------------------------------

#binary indicators

#recode outcome into numeric index in one variable
unsw$attack_ind[unsw$attack_cat=="Analysis"]=0
unsw$attack_ind[unsw$attack_cat=="Backdoor"]=1
unsw$attack_ind[unsw$attack_cat=="DoS"]=2
unsw$attack_ind[unsw$attack_cat=="Exploits" ]=3
unsw$attack_ind[unsw$attack_cat=="Fuzzers"]=4
unsw$attack_ind[unsw$attack_cat=="Normal" ]=5
unsw$attack_ind[unsw$attack_cat== "Reconnaissance" ]=6
unsw$attack_ind[unsw$attack_cat=="Shellcode" ]=7
unsw$attack_ind[unsw$attack_cat=="Worms" ]=8

unsw[222:300, c("attack_cat", "attack_ind")]


#drop categorical variables 
outcome=unsw$attack_ind #separate outcome labels for sparse matrix
drops=c("proto", "service", "state", "attack_cat", "attack_ind")
unsw=unsw[ , !(names(unsw) %in% drops)]
dim(unsw)

#Create Design Matrix
design.matrix=as.matrix(cbind(unsw, proto_ind, service_ind, state_ind))
dim(design.matrix) #new dimensions
head(design.matrix)
str(design.matrix)

# Create sparse matrix
unsw_sparse = xgb.DMatrix(design.matrix, label=outcome)
dim(unsw_sparse)





bstSparse = xgboost(data = unsw_sparse, label = outcome,
                    objective = "multi:softmax", num_class = 10, nrounds = 20)

#### EXTRA

set.seed(101) # Set Seed so that same sample can be reproduced in future also

# Select 70% of data as sample from total 'n' rows of the data  
sample = sample.int(n = nrow(unsw), size = floor(.70*nrow(unsw)), replace = FALSE, prob = NULL)
train = unsw[sample, ]