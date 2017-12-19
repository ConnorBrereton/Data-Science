# install.packages("naivebayes")
setwd("/Users/connor/Desktop/cisco/xgboost")
list.files(pattern="*.csv")
unsw=read.csv("train.csv",header=T)
# install.packages("naivebayes")
library(naivebayes)
library(dplyr)
install.packages("unbalanced")
library(unbalanced)

# Data Inspection ---------------------------------------------------------
head(unsw) #preview first 6 rows
str(unsw) #get variable list with types
dim(unsw) #dimensions
unique(levels(unsw$attack_cat)) #labels of attack classes
length(levels(unsw$attack_cat))

#counts and proportions
counts=table(unsw$attack_cat) #get counts of labeled attacks
prop.table(counts)*100 # proportion by attack category--note unbalanced classes

# Data Prep ---------------------------------------------------------------
#attack_labels=as.factor(unsw$attack_cat) #save list for for grid search 
#binary indicators
#proto_ind=model.matrix( ~proto - 1, data=unsw) #convert proto
#service_ind=model.matrix( ~service - 1, data=unsw) #convert service
#state_ind=model.matrix( ~state - 1, data=unsw) #convert proto

#recode outcome
#unsw$attack_ind=as.numeric(unsw$attack_cat)-1
#unsw[222:300, c("attack_cat", "attack_ind")]
#table(unsw$attack_cat,unsw$attack_ind)

#drop categorical variables
#outcome=unsw$attack_ind #separate outcome labels for sparse matrix
drops=c( "label" 
         # "proto", "service", "state","attack_cat", 
)
unsw=unsw[ , !(names(unsw) %in% drops)]
dim(unsw)
str(unsw)

# Load Test Data ----------------------------------------------------------
unsw.test=read.csv("test.csv" ,header=T)

# Data Inspection ---------------------------------------------------------
str(unsw.test) #get variable list with types
dim(unsw.test) #dimensions
unique(levels(unsw.test$attack_cat)) #labels of attack classes
length(levels(unsw.test$attack_cat))

#counts and proportions
counts=table(unsw.test$attack_cat) #get counts of labeled attacks
prop.table(counts)*100 # proportion by attack category--note unbalanced classes

# Data Prep ---------------------------------------------------------------
#drop un-needed variables
outcome=unsw.test$attack_ind #separate outcome labels for sparse matrix
drop.test=c( "label", "attack_cat")
unsw.test=unsw.test[ , !(names(unsw.test) %in% drop.test)]
dim(unsw.test)
str(unsw.test)   

# -------- Naive Bayes --------
## Categorical data only:
#data(unsw, package = "naivebayes")
model <- naive_bayes(attack_cat ~ ., data = unsw)
pred=predict(model, unsw, type = "class")


confusion.counts=table(unsw$attack_cat,pred)
confusion.props=prop.table(confusion.counts,1)
options(digits=1)
confusion.props*100


# --------- ggplot ---------

confusion_table = as.data.frame(table(confusion.props))
# names(confusion.counts) = c("actual","predicted","freq") #name columns in count table
names(confusion.props) = c("actual","predicted","percent") # names for proportion tablee
as.data.frame(table(unsw$attack_cat, pred))

# draw the tile
tilee <- ggplot() +
  geom_tile(aes(x=actual, y=predicted, fill=percent), data=confusion_table, color="black",size=0.1) +
  labs(x="Actual",y="Predicted")
tile.b = tilee + scale_fill_gradient(low="white", high="green") 
tile.k= tile.b + geom_text(aes(x=actual, y=predicted,label=round(percent,2)), data=confusion_table)
#label=sprintf("%.1f", Percent))









