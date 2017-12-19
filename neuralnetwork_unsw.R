setwd("/Users/connor/Desktop/cisco/neuralnetwork")
list.files(pattern="*.csv")
unsw=read.csv("train.csv", header=T, nrows=70000)
# install.packages('h2o')
library('h2o')

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

# ------------------------------------ Neural Network ------------------------------------

# input matrix
X=matrix(c(1,0,1,0,1,0,1,1,0,1,0,1),nrow = 3, ncol=4,byrow = TRUE)

# output matrix
Y=matrix(c(1,1,0),byrow=FALSE)

#sigmoid function
sigmoid<-function(x){
  1/(1+exp(-x))
}

# derivative of sigmoid function
derivatives_sigmoid<-function(x){
  x*(1-x)
}

# variable initialization
epoch=5000
lr=0.1
inputlayer_neurons=ncol(X)
hiddenlayer_neurons=3
output_neurons=1

#weight and bias initialization
wh=matrix( rnorm(inputlayer_neurons*hiddenlayer_neurons,mean=0,sd=1), inputlayer_neurons, hiddenlayer_neurons)
bias_in=runif(hiddenlayer_neurons)
bias_in_temp=rep(bias_in, nrow(X))
bh=matrix(bias_in_temp, nrow = nrow(X), byrow = FALSE)
wout=matrix( rnorm(hiddenlayer_neurons*output_neurons,mean=0,sd=1), hiddenlayer_neurons, output_neurons)

bias_out=runif(output_neurons)
bias_out_temp=rep(bias_out,nrow(X))
bout=matrix(bias_out_temp,nrow = nrow(X),byrow = FALSE)

# forward propagation
for(i in 1:epoch){
  
  hidden_layer_input1= X%*%wh
  hidden_layer_input=hidden_layer_input1+bh
  hidden_layer_activations=sigmoid(hidden_layer_input)
  output_layer_input1=hidden_layer_activations%*%wout
  output_layer_input=output_layer_input1+bout
  output= sigmoid(output_layer_input)
  
  # Back Propagation
  
  E=Y-output
  slope_output_layer=derivatives_sigmoid(output)
  slope_hidden_layer=derivatives_sigmoid(hidden_layer_activations)
  d_output=E*slope_output_layer
  Error_at_hidden_layer=d_output%*%t(wout)
  d_hiddenlayer=Error_at_hidden_layer*slope_hidden_layer
  wout= wout + (t(hidden_layer_activations)%*%d_output)*lr
  bout= bout+rowSums(d_output)*lr
  wh = wh +(t(X)%*%d_hiddenlayer)*lr
  bh = bh + rowSums(d_hiddenlayer)*lr
  
}
output