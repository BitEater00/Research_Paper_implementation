#Kathal Aditya Rajendra
#Paper Link : https://globaljournals.org/item/5412-a-modified-version-of-the-k-means-clustering-algorithm
#K-Means - Reducing the number of iterations and also improving the accuracy
library(dplyr)
library(nlme)
library(factoextra)
library(caret)
set.seed(7008) #setting seed so that results are reproducible
df <- rbind(iris,iris)
df <- rbind(df,df) #importing dataset
trainIndex <- createDataPartition(df$Species, p = .7,list = FALSE,times = 1)
train_f <- df[ trainIndex,] #training data
test_f <- df[-trainIndex,] #testing data
train <- train_f %>% select(-Species)
test <- test_f %>% select(-Species)
our_fun(train,test,train_f,test_f) #running the proposed K-Means Algorithm
pre_fun(train,test,train_f,test_f) #running the standard K-Means Algorithm

our_fun <- function(train,test,train_f,test_f){
  
  #variable to keep tab of how many iteration ran in total
  proof_count <- 0
  
  #This is the step one of the algorithm. Here we are finding the diatance from the center
  dump_data <- train %>% mutate(distance = sqrt(Sepal.Length^2 + Sepal.Width^2 + Petal.Length^2 + Petal.Width^2))
  #Now we are arranging in increasing order
  dump_data <- dump_data %>% arrange(distance)
  #We do not need the distance column in future so we are droping it.
  dump_data <- dump_data %>% select(-distance)
  #choosing the initial centroids. Three partition are to be done. So division  by 3
  v1 <- (nrow(train) %/% 3)%/%2
  v2 <- v1 + nrow(train) %/% 3
  v3 <- v2 + nrow(train) %/% 3
  #storing the intial centroids
  center1 <- dump_data[v1,]
  center2 <- dump_data[v2,]
  center3 <- dump_data[v3,]
  
  #creating the two datastructure required fot the implementation of the proposed 
  #K means algorithm
  clusters <- rep(-1,nrow(train)) #Will store the last cluster number
  distance <- rep(-1,nrow(train)) #Will store the distance from that cluster
  
  #This is the first iteration. Here we are running this to assign
  #All the data structures in the program to have their initial values
  #There one should observe no IF condition is given.
  for (i in 1:nrow(train)){
    
    #calcualtin distance from the centroids
    dist1 <-  dist(rbind(center1 , train[i,]) , method = "euclidean") 
    dist2 <-  dist(rbind(center2 , train[i,]) , method = "euclidean")
    dist3 <-  dist(rbind(center3 , train[i,]) , method = "euclidean")
    distance_from_centers <- c(dist1 , dist2 , dist3)
    #finding the minimum distance and saving it in distance[]
    distance[i] <- min(distance_from_centers)
    #finding which cluster that min distance correspondos to and saving it.
    clusters[i] <- which.min(distance_from_centers)
    proof_count = proof_count + 1
  }
  
  #binging the cluster and distance data with train dataframe
  train <- cbind(train , cbind(clusters,distance))
  
  #the main loop where we will apply the conditions of the proposed algorithm
  while(TRUE){
    
    #saving old centroid location to facilate error calculation in the future
    old_center1 <- center1
    old_center2 <- center2
    old_center3 <- center3
    
    #spliting the data into grps as per their cluster allocation in i-1th iteration
    grp_data <- train %>% group_split(clusters)
    center1 <- colMeans(grp_data[[1]])  #new centroid 1
    center2 <- colMeans(grp_data[[2]]) #new centroid 2
    center3 <- colMeans(grp_data[[3]]) #new centroid 3
    centers <- rbind(center1 , center2 , center3)
    err1 <- abs(sum(old_center1 - center1)) #calculating the diff in prev and new value
    err2 <- abs(sum(old_center2 - center2)) #calculating the diff in prev and new value
    err3 <- abs(sum(old_center3 - center3)) #calculating the diff in prev and new value
    #if the difference is less than 0.0001 we will stop the while loop
    if(err1 < 0.0001 && err2 < 0.0001 && err3 < 0.0001)
    {
      break
    }
    
    #the for loop to check every row in the dataset.
    for (i in 1:nrow(train)){
      
      #the proposed condition that leads to less number of iterations
      if(dist(rbind(centers[train[i,"clusters"] , ] , train[i,]) , method = "euclidean") > train[i,"distance"]){
        dist1 <-  dist(rbind(center1 , train[i,]) , method = "euclidean")
        dist2 <-  dist(rbind(center2 , train[i,]) , method = "euclidean")
        dist3 <-  dist(rbind(center3 , train[i,]) , method = "euclidean")
        distance_from_centers <- c(dist1 , dist2 , dist3)
        train$distance[i] <- min(distance_from_centers)
        train$clusters[i] <- which.min(distance_from_centers)
        proof_count = proof_count + 1
      }
    }
  }
  
  #naming the clusters so that visually we can see the difference
  train$clusters[train$clusters == 1] <- "setosa"
  train$clusters[train$clusters == 2] <- "versicolor"
  train$clusters[train$clusters == 3] <- "virginica"
  
  #dropping distance as it is not required anymore
  train <- train %>% select(-distance)
  Species <- train_f$Species
  train <- cbind(train , Species)
  
  #initialation of the variable that will keep count of how many we were able to 
  #classify correctly
  count <- 0
  for (i in 1:nrow(train)){
    if(train[i,"Species"] == train[i,"clusters"]){
      count <- count+1 
    }
  }
  #printing accuracy
  print((count/nrow(train))*100)
  
  
  #We take the centroids from the train data and see they classify the test data
  #for loop to assign the test data their nearest cluster
  for (i in 1:nrow(test)){
    dist1 <-  dist(rbind(center1 , test[i,]) , method = "euclidean")
    dist2 <-  dist(rbind(center2 , test[i,]) , method = "euclidean")
    dist3 <-  dist(rbind(center3 , test[i,]) , method = "euclidean")
    distance_from_centers <- c(dist1 , dist2 , dist3)
    test$distance[i] <- min(distance_from_centers)
    test$clusters[i] <- which.min(distance_from_centers)
  }
  
  #again renaming the clusters
  test$clusters[test$clusters == 1] <- "setosa"
  test$clusters[test$clusters == 2] <- "versicolor"
  test$clusters[test$clusters == 3] <- "virginica"
  
  #distance column not required in future
  test <- test %>% select(-distance)
  Species <- test_f$Species
  test <- cbind(test , Species)
  
  #initialation of the variable that will keep count of how many we were able to 
  #classify correctly
  count <- 0
  for (i in 1:nrow(test)){
    if(test[i,"Species"] == test[i,"clusters"]){
      count <- count+1 
    }
  }
  #printing accuracy
  print((count/nrow(test))*100)
  
  #printing the number of times the iteration ran
  print(paste(c("The number of iterations were " , proof_count)) , sep = " : ")
}
pre_fun <- function(train,test,train_f,test_f){
  
  #variable to keep count of the total number of iterations
  proof_count <- 0
  
  #random initialization of centroids
  center1 <- train[1,]
  center2 <- train[2,]
  center3 <- train[3,]
  clusters <- rep(-1,nrow(train))
  
  #initial cluster allocation for loop
  for (i in 1:nrow(train)){
    dist1 <-  dist(rbind(center1 , train[i,]) , method = "euclidean")
    dist2 <-  dist(rbind(center2 , train[i,]) , method = "euclidean")
    dist3 <-  dist(rbind(center3 , train[i,]) , method = "euclidean")
    distance_from_centers <- c(dist1 , dist2 , dist3)
    clusters[i] <- which.min(distance_from_centers)
    proof_count = proof_count + 1
  }
  
  train <- cbind(train , cbind(clusters))
  
  #the while loop to continously assign clusters until error is reduced to given range
  while(TRUE){
    
    #increment of variable every time the loop is running 
    
    
    #saving old centoids for calculating diff later
    old_center1 <- center1
    old_center2 <- center2
    old_center3 <- center3
    
    #finding new centroids
    grp_data <- train %>% group_split(clusters)
    center1 <- colMeans(grp_data[[1]])  #new center 1
    center2 <- colMeans(grp_data[[2]])  #new center 2
    center3 <- colMeans(grp_data[[3]])  #new center 3
    centers <- rbind(center1 , center2 , center3)
    err1 <- abs(sum(old_center1 - center1)) #calculating diff for center 1
    err2 <- abs(sum(old_center2 - center2)) #calculating diff for center 2
    err3 <- abs(sum(old_center3 - center3)) #calculating diff for center 3
    if(err1 < 0.0001 && err2 < 0.0001 && err3 < 0.0001)
    {
      break
    }
    
    #assigning and looking for any changes in cluster for every row
    for (i in 1:nrow(train)){
      
      proof_count = proof_count + 1
      dist1 <-  dist(rbind(center1 , train[i,]) , method = "euclidean")
      dist2 <-  dist(rbind(center2 , train[i,]) , method = "euclidean")
      dist3 <-  dist(rbind(center3 , train[i,]) , method = "euclidean")
      distance_from_centers <- c(dist1 , dist2 , dist3)
      train$clusters[i] <- which.min(distance_from_centers)
    }
  }
  
  #renaming the clusters
  train$clusters[train$clusters == 1] <- "setosa"
  train$clusters[train$clusters == 2] <- "versicolor"
  train$clusters[train$clusters == 3] <- "virginica"
  Species <- train_f$Species
  
  #training accuracy
  train <- cbind(train , Species)
  count <- 0
  for (i in 1:nrow(train)){
    if(train[i,"Species"] == train[i,"clusters"]){
      count <- count+1 
    }
  }
  
  #printing accuracy
  print((count/nrow(train))*100)
  
  
  #now we see the accuracy for test data.
  #so first we assign them to clusters using the previous centroids
  for (i in 1:nrow(test)){
    dist1 <-  dist(rbind(center1 , test[i,]) , method = "euclidean")
    dist2 <-  dist(rbind(center2 , test[i,]) , method = "euclidean")
    dist3 <-  dist(rbind(center3 , test[i,]) , method = "euclidean")
    distance_from_centers <- c(dist1 , dist2 , dist3)
    test$clusters[i] <- which.min(distance_from_centers)
  }
  
  #renaming the clusters
  test$clusters[test$clusters == 1] <- "setosa"
  test$clusters[test$clusters == 2] <- "versicolor"
  test$clusters[test$clusters == 3] <- "virginica"
  Species <- test_f$Species
  test <- cbind(test , Species)
  #calculating accuracy
  count <- 0
  for (i in 1:nrow(test)){
    if(test[i,"Species"] == test[i,"clusters"]){
      count <- count+1 
    }
  }
  #printing accuracy
  print((count/nrow(test))*100)
  
  #printing the number of times the iteration ran
  print(paste(c("The number of iteratoins were " , proof_count)) , sep = " : ")
}
