#Kathal Aditya Rajendra
library(ggplot2)
library(dplyr)
library(caret)
library(reshape2)
library(boot)
library(moments)
library(broom)
set.seed(7008)
#Summary:
#Model Used : MultiVariate Linear Regression
#Data preprocessing : 

data <- read.csv("mlr.csv")
str(data)
colnames(data) <- c("RD" , "Admin", "MS" , "State" ,"Profit")
#EDA

#First we see the basic things like if na is present
#number of rows and columns
#mean and median of all numerical columns
#number of states present
nrow(data)
ncol(data)
sum(is.na(data))
summary(data)

#First we try to look that the distribution of data in each numerical column
#We try to see visually if any skewness is present.
ggplot(data, aes(x=RD)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white" , bins = 15)+
  geom_density(alpha=.2, fill="#FF6666")+
  geom_vline(aes(xintercept=mean(RD)),color="blue", linetype="dashed", size=1)
#From the plot we see that the data is normally distributed. So we do not need to 
#check the numerical value of the skewness. So now we need to see that the distribution 
#is in different states.
ggplot(data, aes(x=RD , color = State)) + 
  geom_histogram(aes(y=..density..), fill = "white" , bins = 15 , position = "identity")+
  geom_density(alpha=.2, fill="#FF6666")

data %>% 
  group_by(State) %>% 
  summarise(mean_value = mean(RD)) %>% 
  arrange(mean_value) %>%
  ggplot(aes(x=State, y= mean_value, color= State)) +
  geom_bar(stat="identity", fill="white")
#The distribution of two cities that is New York and Florida is normal. But in California 
#the distribution is leaning towards left. Thus the startups in that state invest less 
#in RD and that to consciously. Maybe its some government policy or any other external
#reason that leads to left shift.

#Now we see will see Administration 
ggplot(data, aes(x=Admin)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white" , bins = 10)+
  geom_density(alpha=.2, fill="#FF6666")+
  geom_vline(aes(xintercept=mean(Admin)),color="blue", linetype="dashed", size=1)

#Again the data has a normal distribution. So no need to check the skewness value 
#numerically Now again we see its distribution in different states
ggplot(data, aes(x=Admin , color = State)) + 
  geom_histogram(aes(y=..density..), fill = "white" , bins = 10 , position = "identity")+
  geom_density(alpha=.2, fill="#FF6666")

data %>% 
  group_by(State) %>% 
  summarise(mean_value = mean(Admin)) %>% 
  arrange(mean_value) %>%
  ggplot(aes(x=State, y= mean_value, color= State)) +
  geom_bar(stat="identity", fill="white")
#New York are little bit left inclined. As per mean value all States have equal spending
#in Adminstration

#Marketing
ggplot(data, aes(x=MS)) + 
  geom_histogram(aes(y=..density..), colour="black", fill="white" , bins = 10)+
  geom_density(alpha=.2, fill="#FF6666")+
  geom_vline(aes(xintercept=mean(MS)),color="blue", linetype="dashed", size=1)
#The visual inspection shows that the predictor is skewed. So we check the numerical 
#value of the skewness
skewness(data$MS)
#The numerical value shows that the skewness is not one of concern. So we can move forward

#Now we see state wise distribution
ggplot(data, aes(x= MS , color = State)) + 
  geom_histogram(aes(y=..density..), fill = "white" , bins = 10 , position = "identity")+
  geom_density(alpha=.2, fill="#FF6666")

data %>% 
  group_by(State) %>% 
  summarise(mean_value = mean(MS)) %>% 
  arrange(mean_value) %>%
  ggplot(aes(x=State, y= mean_value, color= State)) +
  geom_bar(stat="identity", fill="white")
#Now we can understand that the little skewness that was present in the data was majorily 
#due to new york and florida.




#Now we have seen all the numerical predicators singular. Wee would like to now undestand 
#how each one of them affeat the profits.
#RD
ggplot(data, aes(x=RD, y=Profit, shape=State, color=State)) +
  geom_point()
#The points show a proper linear regression between RD and profit.Also the data
#across state is uniformly distributed.
ggplot(data, aes(x=Admin, y=Profit, shape=State, color=State)) +
  geom_point()
#The points are randomly distributed.So this predicator is not a good
#variable for linear regression related model.Data distribution across
#states is again uniform
ggplot(data, aes(x=Admin, y=Profit, shape=State, color=State)) +
  geom_point()
#Again we see no proper line forming. But a majority part of the data is forming a 
#line. So it is a good fit.


#Now we see for outliners using the box plot method
boxplot(data$RD)
ggplot(data, aes(x= State , y= RD, color=State)) +
  geom_boxplot()
#We see no outliners by this method for this variable. The same statement stands for
#state wise distribution

boxplot(data$Admin)
ggplot(data, aes(x= State , y= Admin, color=State)) +
  geom_boxplot()
#For the total distribution we see no outliners. But for state wise distribution we see 
#two points as the outliners. If these points come as outliner in our second test also
# we will remove them.
boxplot(data$MS)
ggplot(data, aes(x= State , y= MS, color=State)) +
  geom_boxplot()
#Again no outliners in the total distribution. But a single outliner so again we will 
#check again in the second method.

#The whole dataset visualized in a single graph
pairs(data %>% select(-State))


#Now we check the outliners using the second method that is WINSORIZATION METHOD
WM <- function(x){
  q1 <- quantile(x , c(.01))
  q3 <- quantile(x , c(.99))
  count <- 1
  for (i in x){
    if (i < q1 || i > q3){
      print(paste(c(i,count)))
    }
    count <- count+1
  }
} 
WM(data$RD)
WM(data$Admin)
WM(data$MS)

#From the second method we can see that exactly the same amount of outliners are present
#in the dataset. The row one is outliner in two variable. SO we need to remove it. Since 
#Admin is not a good predicator for our model so we are going to check its importance 
#if high we will remove those rows else keep them.
data <- data[2:nrow(data), ]


#We have only 3 different states. So we can do one hot encoding.
dummy <- dummyVars(" ~ .", data=data)
newdata <- data.frame(predict(dummy, newdata = data)) 

#Check the importance of variables in respect to the profit output first we use
#backwards selection aka recursive feature selection method and 
#secondly we will use the correlation method

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=100)
# run the RFE algorithm
results <- rfe(newdata[,1:6], newdata[,7], size = c(1:6) , rfeControl=control)
# summarize the results
print(results)
# list the chosen features
predictors(results)
# plot the results
plot(results, type=c("g", "o"))

#Now we look at the correaltion method
cormat <- round(cor(newdata),2)
melted_cormat <- melt(cormat)
ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
  geom_tile() + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) 


#From using both the analysis above we can clearly see that marketing is highly correlated
#with profit.So we can drop it. Also from the result of RFE we can see that RD is the 
#important variable. Thus
newdata <- newdata %>% select(-MS , -StateNew.York , -StateFlorida , -StateCalifornia)


#0-1 normalization 
newdata$RD <- (newdata$RD - min(newdata$RD))/(max(newdata$RD)-min(newdata$RD))
newdata$Admin <- (newdata$Admin-min(newdata$Admin))/(max(newdata$Admin)-min(newdata$Admin))

## 70% of the sample size
smp_size <- floor(0.7 * nrow(newdata))
train_ind <- sample(seq_len(nrow(newdata)), size = smp_size)
train <- newdata[train_ind, ]
test <- newdata[-train_ind, ]

model <- lm(formula = Profit~. , data = train)
#ploting the model 
plot(model)
test$pred <- predict(model , test[,1:2] ,type = "response")
test %>% 
  summarise(
    R2 = cor(Profit, pred)^2,
    MSE = mean((Profit - pred)^2),
    RMSE = sqrt(MSE),
    MAE = mean(abs(Profit - pred)))


# We have acheived a good accuracy from our model. Thus linear regression was a 
#good fit to this data.