
#====================================================================================== -
#====================================================================================== -
#====================================================================================== -

# RFE Ranger ======================= ----

dir <- "O:/Projects/KP0011/3/"
setwd(dir)

data <- readRDS("Analysis/SI_dynamics/preds_rf_regr_allctrls.rds")

#create multifolds for repeated n-fold cross validation
index <- caret::createDataPartition(data$sev, p = 0.8, times = 30, groups = 9)


# The candidate set of the number of predictors to evaluate
subsets <- c(length(data), 200, 150, 120, 105, 90, 75, 
             60, 50, 40, 35, 30, 25, 20, 17, 14, 12:1)

#define trait to analyze
trait <- "sev"

# function to calculate accuracy
get_acc <- function(model, testdata) {
  preds_class <- caret::predict.train(rf_ranger,newdata = test[ , names(test) != "trt"])
  true_class <- test$trt
  res <- cbind(preds_class, true_class) %>% data.frame()
  match <- ifelse(res$preds_class == res$true_class, 1, 0) %>% sum()
  acc <- match/nrow(test)
}

# function to calculate RMSE  
rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

####################################### -

## PERFORM RERUSIVE FEATURE ELIMINATION 

start.time <- Sys.time()  

#outer resampling
#CV of feature selection
out <- list()
for(i in 1:length(index)){
  
  #Verbose
  print(paste("resample ", i, "/", length(index), sep = ""))
  
  #use indices to create train and test data sets for the resample
  ind <- as.numeric(index[[i]])
  train <- data[ind,]
  test <- data[-ind, ]
  
  #for each subset of decreasing size
  #tune/train rf and select variables to retain
  keep_vars <- drop_vars <- test_perf <- train_perf <- npred <- train_perf_adj <- NULL
  for(j in 1:length(subsets)){
    
    #define new training data
    #except for first iteration, where the full data set ist used
    if(exists("newtrain")) {train = newtrain}
    
    #Verbose
    print(paste("==> subset size = ", length(train)-1, sep = ""))
    
    #adjust mtry parameter to decreasing predictor set
    #maximum mtry at 200
    mtry <- ceiling(seq(1, length(train[-1]), len = 17)) %>% unique()
    if(any(mtry > 250)){
      mtry <- mtry[-which(mtry >= 250)]
    }
    
    min.node.size <- 3
    
    #specify model tuning parameters
    tune_grid <- expand.grid(mtry = mtry,
                             splitrule = "variance",
                             min.node.size = min.node.size) 
    
    #define inner resampling procedure
    ctrl <- caret::trainControl(method = "repeatedcv",
                                number = 10,
                                rep = 1,
                                verbose = FALSE,
                                allowParallel = TRUE,
                                savePredictions = TRUE)
    
    #define model to fit
    formula <- as.formula(paste(trait, " ~ .", sep = ""))
    
    #tune/train random forest
    rf_ranger <- caret::train(formula,
                              data = train,
                              preProc = c("center", "scale"),
                              method = "ranger",
                              tuneGrid = tune_grid,
                              importance = "permutation",
                              num.trees = 2500,
                              trControl = ctrl)
    
    #extract predobs of each cv fold
    predobs_cv <- plyr::match_df(rf_ranger$pred, rf_ranger$bestTune, on = c("mtry", "min.node.size"))
    
    #Average predictions of the held out samples;
    predobs <- predobs_cv %>% 
      group_by(rowIndex) %>% 
      summarize(obs = mean(obs), 
                mean_pred = mean(pred))
    
    #get train performance
    train_perf[j] <- caret::getTrainPerf(rf_ranger)$TrainRMSE
    
    # #calculate adjusted train performance
    # train_perf_adj[j] <- sqrt(sum((predobs$mean_pred-predobs$obs)^2)/nrow(predobs))
    
    # #calculate test performance
    # test_perf[j] <- rmse(test$sev, predict(rf_ranger, test)) 
    
    #number of preds used
    npred[[j]] <- length(train)-1
    
    #extract retained variables
    #assign ranks
    #define reduced training data set
    if(j < length(subsets)){
      #extract top variables to keep for next iteration
      keep_vars[[j]] <- varImp(rf_ranger)$importance %>% 
        tibble::rownames_to_column() %>% 
        tibble::as_tibble() %>% rename(var = rowname) %>%
        arrange(desc(Overall)) %>% slice(1:subsets[j+1]) %>% pull(var)
      #extract variables dropped from dataset
      drop_vars[[j]] <- names(train)[!names(train) %in% c(keep_vars[[j]], "sev")] %>% 
        tibble::enframe() %>% mutate(rank = length(subsets)-j+1) %>% 
        dplyr::select(value, rank) %>% rename(var = value)
      #define new training data
      newtrain <- dplyr::select(train, sev, keep_vars[[j]])
      #last iteration
    } else {
      drop_vars[[j]] <- names(train)[names(train) != "sev"] %>% 
        tibble::enframe() %>% mutate(rank = length(subsets)-j+1) %>% 
        dplyr::select(value, rank) %>% rename(var = value)
    }
    
  } #END OF FEATURE ELIMINATION ON RESAMPLE i
  
  #clean environment 
  rm("newtrain")
  
  #gather results for resample i
  ranks <- drop_vars %>% do.call("rbind", .)
  out[[i]] <- list(ranks, train_perf, train_perf_adj, test_perf, npred)
  
} #END OF OUTER RESAMPLING

saveRDS(out, "O:/Projects/KP0011/3/Analysis/RFE/rfe_regr_dyn_rf.rds")

end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

#====================================================================================== -

# RFE Cubist ======================= ----

dir <- "O:/Projects/KP0011/3/"
setwd(dir)

data <- readRDS("Analysis/SI_dynamics/preds_rf_regr_allctrls.rds")

#create multifolds for repeated n-fold cross validation
index <- caret::createDataPartition(data$sev, p = 0.8, times = 30, groups = 9)

# The candidate set of the number of predictors to evaluate
subsets <- c(length(data), 100, 50, 40, 30,
             25, 20, 17, 14, 12, 10:1)

#define trait to analyze
trait <- "sev"

# function to calculate accuracy
get_acc <- function(model, testdata) {
  preds_class <- caret::predict.train(rf_ranger,newdata = test[ , names(test) != "trt"])
  true_class <- test$trt
  res <- cbind(preds_class, true_class) %>% data.frame()
  match <- ifelse(res$preds_class == res$true_class, 1, 0) %>% sum()
  acc <- match/nrow(test)
}

# function to calculate RMSE  
rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

#set up a cluster
library(parallel)
library(doParallel)
cluster <- makeCluster(11, outfile = "")
registerDoParallel(cluster)
clusterEvalQ(cluster, library(doParallel))
clusterEvalQ(cluster, .libPaths("T:/R3UserLibs_rds"))
clusterEvalQ(cluster, library(doParallel))

####################################### -

## PERFORM RERUSIVE FEATURE ELIMINATION 

start.time <- Sys.time()  

#outer resampling
#CV of feature selection
out <- list()
for(i in 1:length(index)){
  
  #Verbose
  print(paste("resample ", i, "/", length(index), sep = ""))
  
  #use indices to create train and test data sets for the resample
  ind <- as.numeric(index[[i]])
  train <- data[ind,]
  test <- data[-ind, ]
  
  #for each subset of decreasing size
  #tune/train rf and select variables to retain
  keep_vars <- drop_vars <- test_perf <- train_perf <- npred <- train_perf_adj <- NULL
  for(j in 1:length(subsets)){
    
    #define new training data
    #except for first iteration, where the full data set ist used
    if(exists("newtrain")) {train = newtrain}
    
    #Verbose
    print(paste("==> subset size = ", length(train)-1, sep = ""))
    
    #define tuning parameter grid
    train_grid <-   train_grid <- expand.grid(committees = c(1, 2, 3, 4, 5, 7, 10),
                                              neighbors = 0)
    
    #define inner resampling procedure
    ctrl <- caret::trainControl(method = "repeatedcv",
                                number = 10,
                                repeats = 5, #no repeats
                                savePredictions = TRUE,
                                selectionFunction = "oneSE",
                                verboseIter = FALSE,
                                allowParallel = TRUE)
    
    #define model to fit
    formula <- as.formula(paste(trait, " ~ .", sep = ""))
    
    #tune/train a cubist regression model
    cubist <- train(
      formula,
      data = train,
      preProcess = c("center", "scale"),
      method = "cubist",
      tuneGrid = train_grid,
      trControl = ctrl
    )
    
    #extract predobs of each cv fold
    predobs_cv <- plyr::match_df(cubist$pred, cubist$bestTune, on = c("committees", "neighbors"))
    
    #Average predictions of the held out samples;
    predobs <- predobs_cv %>% 
      group_by(rowIndex) %>% 
      summarize(obs = mean(obs), 
                mean_pred = mean(pred))
    
    #get train performance
    train_perf[j] <- caret::getTrainPerf(cubist)$TrainRMSE
    #calculate adjusted train performance
    train_perf_adj[j] <- sqrt(sum((predobs$mean_pred-predobs$obs)^2)/nrow(predobs))
    
    #calculate test performance
    test_perf[j] <- rmse(test$sev, predict(cubist, test)) 
    
    #number of preds used
    npred[[j]] <- length(train)-1
    
    #extract retained variables
    #assign ranks
    #define reduced training data set
    if(j < length(subsets)){
      #extract top variables to keep for next iteration
      keep_vars[[j]] <- varImp(cubist)$importance %>% 
        tibble::rownames_to_column() %>% 
        tibble::as_tibble() %>% rename(var = rowname) %>%
        arrange(desc(Overall)) %>% slice(1:subsets[j+1]) %>% pull(var)
      #extract variables dropped from dataset
      drop_vars[[j]] <- names(train)[!names(train) %in% c(keep_vars[[j]], "sev")] %>% 
        tibble::enframe() %>% mutate(rank = length(subsets)-j+1) %>% 
        dplyr::select(value, rank) %>% rename(var = value)
      #define new training data
      newtrain <- dplyr::select(train, sev, keep_vars[[j]])
      #last iteration
    } else {
      drop_vars[[j]] <- names(train)[names(train) != "sev"] %>% 
        tibble::enframe() %>% mutate(rank = length(subsets)-j+1) %>% 
        dplyr::select(value, rank) %>% rename(var = value)
    }
    
  } #END OF FEATURE ELIMINATION ON RESAMPLE i
  
  #clean environment 
  rm("newtrain")
  
  #gather results for resample i
  ranks <- drop_vars %>% do.call("rbind", .)
  out[[i]] <- list(ranks, train_perf, train_perf_adj, test_perf, npred)
  
} #END OF OUTER RESAMPLING

saveRDS(out, "O:/Projects/KP0011/3/Analysis/RFE/rfe_regr_dyn_cubist.rds")


end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken

#END========================== ----

#====================================================================================== -
#====================================================================================== -


#====================================================================================== -

#HEADER ----

# AUTHOR: Jonas Anderegg

# Tidy up rfe outputs
# Get robust feature ranks
# create performance profile plots

#====================================================================================== -

d <- readRDS("O:/Projects/KP0011/3/Analysis/RFE/rfe_regr_dyn_cubist.rds")
d <- readRDS("O:/Projects/KP0011/3/Analysis/RFE/rfe_regr_dyn_rf.rds")

varnames <- readRDS("O:/Projects/KP0011/3/Analysis/SI_dynamics/preds.rds") %>% 
  dplyr::select(contains("SI_")) %>% names() %>% data.frame() %>% 
  tibble::add_column(paste("V", 1:278, sep = "")) %>% 
  tibble::as_tibble() %>% rename("varname" = 1, 
                                 "abbr" = 2) %>% 
  dplyr::select(2, 1)


subsets <- d[[1]][[5]]

allranks <- lapply(d, "[[", 1)

colnames <- c("var", paste("Resample", 1:30, sep = ""))

#create ranks table
ranks <- allranks %>%
  Reduce(function(dtf1,dtf2) full_join(dtf1,dtf2,by="var"), .)
names(ranks) <- colnames

colnames <- c("subset_size", paste("Resample", 1:30, sep = ""))

RMSEtrain <- lapply(d, "[[", 2) %>% lapply(., cbind, subsets) %>%
  lapply(., as_tibble) %>% Reduce(function(dtf1,dtf2) full_join(dtf1,dtf2,by="subsets"), .) %>% 
  dplyr::select(subsets, everything())
names(RMSEtrain) <- colnames

RMSEtrain_adj <- lapply(d, "[[", 3) %>% lapply(., cbind, subsets) %>%
  lapply(., as_tibble) %>% Reduce(function(dtf1,dtf2) full_join(dtf1,dtf2,by="subsets"), .) %>% 
  dplyr::select(subsets, everything())
names(RMSEtrain_adj) <- colnames

RMSEtest <- lapply(d, "[[", 4) %>% lapply(., cbind, subsets) %>%
  lapply(., as_tibble) %>% Reduce(function(dtf1,dtf2) full_join(dtf1,dtf2,by="subsets"), .) %>% 
  dplyr::select(subsets, everything())
names(RMSEtest) <- colnames

#average over resamples, get sd of means
Trainperf <- RMSEtrain %>% 
  tibble::as_tibble() %>%
  tidyr::gather(resample, RMSE, 2:31) %>%
  group_by(subset_size) %>%
  arrange(subset_size) %>%
  summarise_at(vars(RMSE), funs(mean, sd), na.rm = TRUE) %>%
  mutate(set = "Train") %>% 
  as.data.frame()
Trainperf_adj <- RMSEtrain_adj %>% 
  tibble::as_tibble() %>%
  tidyr::gather(resample, RMSE, 2:31) %>%
  group_by(subset_size) %>%
  arrange(subset_size) %>%
  summarise_at(vars(RMSE), funs(mean, sd), na.rm = TRUE) %>%
  mutate(set = "Train_adj") %>% 
  as.data.frame()
Testperf <- RMSEtest %>% 
  tibble::as_tibble() %>%
  tidyr::gather(resample, RMSE, 2:31) %>%
  group_by(subset_size) %>%
  arrange(subset_size) %>%
  summarise_at(vars(RMSE), funs(mean, sd), na.rm = TRUE) %>%
  mutate(set = "Test") %>% 
  as.data.frame()

#mean ranks and sd
robranks <- ranks %>% 
  tibble::as_tibble() %>%
  tidyr::gather(resample, rank, 2:31) %>%
  group_by(var) %>%
  summarise_at(vars(rank), funs(mean, sd), na.rm = TRUE) %>%
  arrange(mean) %>% 
  as.data.frame()

# write.csv(robranks, "O:/Projects/KP0011/3/Analysis/RFE/robranks_cubist.csv")

Perf_cubist <- rbind(Trainperf, Trainperf_adj, Testperf) %>% mutate(algorithm = "cubist")
Perf_rf <- rbind(Trainperf, Trainperf_adj, Testperf) %>% mutate(algorithm = "ranger")

Perf <- rbind(Perf_cubist, Perf_rf) %>% 
  dplyr::filter(set != "Train_adj") %>% 
  mutate(algorithm = ifelse(algorithm == "cubist", "Cubist", "Random Forest"))


pd <- position_dodge(0.5) # move them .05 to the left and right

pdf("O:/Projects/KP0011/3/Figures/perfprof.pdf", width = 7, height = 3.5)

#plot performance profiles
ggplot(Perf, aes(x = subset_size, y = mean, group = set, colour = set)) +
  geom_point(position = pd) + geom_line() +
  geom_errorbar(position = pd, aes(ymin=mean-sd, ymax=mean+sd), width=1, alpha = 0.5) + 
  xlab("#Features") + ylab("RMSE") +
  scale_x_continuous(limits = c(-2.5, 32.5)) +
  facet_wrap(~algorithm) +
  theme_bw() +
  theme(legend.title=element_blank(),
        plot.title = element_text(size=15, face="bold"),
        strip.text = element_text(face = "bold"))

dev.off()


