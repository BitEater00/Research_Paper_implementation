##Kathal Aditya Rajendra 18BCD7008
#Paper doi : https://doi.org/10.1016/j.eswa.2008.01.039
#K-Medoids Paper Implementation
library(dplyr)
library(nlme)
library(factoextra)
library(caret)
set.seed(12) #Setting seeds so that results are re producible.

df <- iris #importing dataset
trainIndex <- createDataPartition(df$Species, p = .7,list = FALSE,times = 1)
train_f <- df[ trainIndex,] #training data split
test_f <- df[-trainIndex,] #testing data split
train <- train_f %>% select(-Species)
test <- test_f %>% select(-Species) 
k <- 3 #number of clusters
implementation(train,test,train_f,test_f,k) #calling our function

implementation <- function(train,test,train_f,test_f,k){
  
  #step1
  #There we are creating the initial distance matrix 
  #This matrix will be used in future to reduce the time 
  #required to calcualte distance and also reduce 
  #time for inclusion of new data points
  n = nrow(train)
  #The matrix that will hold the data
  dist_mat <- matrix(data = NA , nrow = n , ncol = n)
  for(i in 1:n){
    for(j in i:n){
      if(is.na(dist_mat[i,j])){
        dist_mat[i,j] <- dist(rbind(train[i,] , train[j,]) , method = "euclidean")
        dist_mat[j,i] <- dist_mat[i,j]
      }
    }
  }
  
  #calculation of vj. This value will help us in selecting 
  #proper initial clusters and thus the algorithm will converge faster
  train$vj <- c(0)
  train$cluster <- c(-1)
  train$seq <- seq(from = 1 , to = n , by = 1)
  for(j in 1:n){
    vj = 0
    for(i in 1:n){
      num <- dist_mat[i,j]
      denom <- sum(dist_mat[i,])
      vj = vj + (num/denom)
    }
    train[j,"vj"] <- vj
  }
  #sorting the data based on vj value and select top k clusters as the initial clusters
  centroid_select <- train %>% arrange(vj) %>% head(k)
  #matrix to store the centroid of each cluster
  c_dist <- matrix(data = NA , nrow = 1 , ncol = k)
  
  #Assigning the initial clusters to all the rows using the initial centroids
  for(i in 1:n){
    csin_dist <- matrix(data = NA , nrow = 1 , ncol = k)
    for(j in 1:k){
      #here for distance we can directly refer to the dist matrix
      csin_dist[1,j] <- dist_mat[i,centroid_select[j,"seq"]] 
    }
    
    #assigning the cluster to a data point
    train[i,"cluster"] <- which.min(csin_dist[1,])
    index <- which.min(csin_dist[1,])
    #storing that cluster information in the c_dist for future reference
    c_dist[1,index] <- ifelse(is.na(c_dist[1,index]) ,min(csin_dist[1,]), c_dist[1,index]+ min(csin_dist[1,]))
  }
  
  #While to run until the sum of dist from centroids is almost same
  while(TRUE){
    
    #storing prev_sum for future use
    prev_sum <- sum(c_dist[1,])
    
    #step2
    #Now we check for new centroid in the data set 
    #Here we look for a new centroids in the present cluster only
    #Thus we are decreasing the number of iteration required
    for(i in 1:k){
      new_cen_dist <- matrix(data = NA , nrow = 1 , ncol = n)
      #points to check for.
      points_to_consider <- train %>% filter(cluster == i) %>% select(seq)
      for(j in 1:n){
        for(z in points_to_consider){
          #storing distance from each point in the point 
          #again distance calcualtion is avoided due to distance matrix
          new_cen_dist[1,j] <- sum(dist_mat[z,j])
        }
      }
      
      #finding the point which provided the min sum of distances of all
      new_min <- min(new_cen_dist[1,])
      pos <- which.min(new_cen_dist[1,])
      
      #checking the the sum of distances is  minimum or not
      #If less than prev min then update
      if(new_min < c_dist[1,i]){
        centroid_select[i,] <- train[pos,]
      }
      
      #step3
      #Assigning each row to its nearest mediod
      c_dist <- matrix(data = 0 , nrow = 1 , ncol = k)
      for(i in 1:n){
        csin_dist <- matrix(data = NA , nrow = 1 , ncol = k)
        for(j in 1:k){
          csin_dist[1,j] <- dist_mat[i,centroid_select[j,"seq"]]
        }
        train[i,"cluster"] <- which.min(csin_dist[1,])
        index <- which.min(csin_dist[1,])
        #update sum for all rows
        c_dist[1,index] <- c_dist[1,index] + min(csin_dist[1,])
      }
      
    }
    
    #break condition if no change in sum
    if(sum(c_dist) == prev_sum){
      break
      
    }
  }
  
  #now for test data using the previous mediods
  n <- nrow(test)
  cluster <- rep.int(0 , n)
  centroid_select_test <- centroid_select %>% select(-c("vj","cluster" , "seq")) 
  c_dist <- matrix(data = 0 , nrow = 1 , ncol = k)
  #Assignment of cluster to data points based on previous mediods
  for(i in 1:n){
    csin_dist <- matrix(data = 0 , nrow = 1 , ncol = k)
    for(j in 1:k){
      csin_dist[1,j] <- dist(rbind(test[i,] , centroid_select_test[j,]) , method = "euclidean")
    }
    cluster[i] <- which.min(csin_dist[1,])
  }
  
  
  #plotting of all the data
  test$cluster <- as.factor(cluster)
  train$cluster <- as.factor(train$cluster)
  train$Species <- train_f$Species
  test$Species <- test_f$Species
  test$seq <- seq(1,n,1)
  plot_data_train <-  train %>% group_by(cluster)
  plot_data_test <-  test %>% group_by(cluster)
  
  #plot of train data and our prediction class
  ggplot(plot_data_train, aes(x= seq , y= Petal.Length , shape= cluster , color=cluster)) + geom_point()
  #plot of train data and true class
  ggplot(plot_data_train, aes(x= seq , y= Petal.Length , shape= Species , color=Species)) + geom_point()
  #plot of test data and our prediction class
  ggplot(plot_data_test, aes(x= seq , y= Petal.Length , shape= cluster , color=cluster)) + geom_point()
  #plot of test data and true class
  ggplot(plot_data_test, aes(x= seq , y= Petal.Length , shape= Species , color=Species)) + geom_point()
}
