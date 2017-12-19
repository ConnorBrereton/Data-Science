setwd("/Users/connor/Desktop/cisco/xgboost")
list.files(pattern="*.csv")
unsw=read.csv("train.csv", header=T, nrows=70000)
require (xgboost)
library (xgboost)
library(dplyr)

# Data Inspection ---------------------------------------------------------
head(unsw) #preview first 6 rows
str(unsw) #get variable list with types
dim(unsw) #dimensions

#unique levels within 'proto'
levels(unsw$proto)
counts=table(unsw$attack_cat) #get counts of labeled attacks
prop.table(counts)*100

# Data Prep ---------------------------------------------------------------
#binary indicators
proto_ind=model.matrix( ~proto - 1, data=unsw) #convert proto
service_ind=model.matrix( ~service - 1, data=unsw) #convert
state_ind=model.matrix( ~state - 1, data=unsw) #convert proto

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
unsw$attack_ind[test.unsw$attack_cat=="Generic" ]=9

length(unique(unsw$attack_ind))

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
str(unsw_sparse)
dim(unsw_sparse)

head(unsw_sparse)

# Model Training ----------------------------------------------------------

num_class = 10 #uses 0 indexing with label=[0,8]
bstSparse = xgboost(data = unsw_sparse, Label = outcome,
                    objective = "multi:softmax", num_class = 10, nrounds = 20)


# Testing Model -----------------------------------------------------------
test.unsw=read.csv("test.csv",header=T, nrows=50000)
str(test.unsw)
counts=table(test.unsw$attack_cat)
props=prop.table(counts)
100*props

#binary indicators
proto_ind=model.matrix( ~proto - 1, data=test.unsw) #convert proto
service_ind=model.matrix( ~service - 1, data=test.unsw) #convert
state_ind=model.matrix( ~state - 1, data=test.unsw) #convert proto

#recode outcome into numeric index in one variable
test.unsw$attack_ind[test.unsw$attack_cat=="Analysis"]=0
test.unsw$attack_ind[test.unsw$attack_cat=="Backdoor"]=1
test.unsw$attack_ind[test.unsw$attack_cat=="DoS"]=2
test.unsw$attack_ind[test.unsw$attack_cat=="Exploits" ]=3
test.unsw$attack_ind[test.unsw$attack_cat=="Fuzzers"]=4
test.unsw$attack_ind[test.unsw$attack_cat=="Normal" ]=5
test.unsw$attack_ind[test.unsw$attack_cat== "Reconnaissance" ]=6
test.unsw$attack_ind[test.unsw$attack_cat=="Shellcode" ]=7
test.unsw$attack_ind[test.unsw$attack_cat=="Worms" ]=8
test.unsw$attack_ind[test.unsw$attack_cat=="Generic" ]=9

#recode outcome into numeric index in one variable
#recode outcome
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
design.matrix=as.matrix(cbind(test.unsw, proto_ind, service_ind, state_ind))
dim(design.matrix) #new dimensions
head(design.matrix)
str(design.matrix)

# Create sparse matrix
test.unsw_sparse = xgb.DMatrix(design.matrix)
str(test.unsw_sparse)
dim(test.unsw_sparse)


################
###Prediction###
################

pred <- predict(bstSparse, test.unsw_sparse) #prediction

length(pred)
length(test_label)

test_label[1:20]
pred[1:20]
matrix.counts=table(test_label, pred)

options(digits = 2)
prop=prop.table(matrix.counts, 2)
prop.df=as.data.frame((prop))
match=sum(as.numeric(pred) == test_label)
match
match/50000



#build confusion matrix
confusion = as.data.frame(table(test_label, pred)) #counts
names(confusion) = c("actual","predicted","freq") #name columns in count table
names(prop.df) = c("actual","predicted","percent") # names for proportion tablee
as.data.frame(table(test_label, pred)) #

#calculate percentage of test cases based on actual frequency
# confusion = merge(confusion, actual, by=c(&quot;Actual&quot;))
#confusion$Percent = confusion$Freq/confusion$ActualFreq*100

#render plot
# we use three different layers
# first we draw tiles and fill color based on percentage of test cases
tile <- ggplot() +
  geom_tile(aes(x=actual, y=predicted, fill=percent), data=prop.df, color="black",size=0.1) +
  labs(x="Actual",y="Predicted")

tile.c = tile + scale_fill_gradient(low="grey",high="green")
#label=sprintf("%.1f", Percent)),

# lastly we draw diagonal tiles. We use alpha = 0 so as not to hide previous layers but use size=0.3 to highlight border
tile.f = tile.c + 
  geom_tile(aes(x=actual,y=pred),data=subset(prop.df, as.character(actual)==as.character(pred)), color="black",size=0.3, fill="black", alpha=0) 

#Convert number back to text


# first we draw tiles and fill color based on percentage of test cases
tile <- ggplot() +
  geom_tile(aes(x=actual, y=predicted,fill=percent),data=prop.df, color="black",size=0.1) +
  labs(x="Actual",y="Predicted")
tile.c = tile + scale_fill_gradient(low="grey", high="green")
#label=sprintf("%.1f", Percent)),


prop.df=as.data.frame((props))
head(prop.df)

#recode outcome into numeric index in one variable
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

# Added in 07/25/17 to fix data

unsw$attack_ind=as.numeric(unsw$attack_cat)-1
unsw[222:300, c("attack_cat", "attack_ind")]
table(unsw$attack_cat,unsw$attack_ind)

head(prop.df)

#render
tile