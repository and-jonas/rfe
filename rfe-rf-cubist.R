#====================================================================================== -

#HEADER ----

# AUTHOR: Jonas Anderegg

# Train regression models for disease severity prediction
# Test performance on FPWW022
# Perform recurisve feature elimination with rf and cubist as base learner

#====================================================================================== -

# .libPaths("T:/R3UserLibs_rds")
.libPaths("T:/R3UserLibs")

library(dplyr)
library(tidyr)
library(stringr)
library(tibble)
library(caret)
library(ranger)
library(desplot)
library(Cubist)
library(elasticnet)
library(pls)

dir <- "O:/Projects/KP0011/3/"
setwd(dir)

source("O:/Projects/KP0011/3/Project/Septoria-Project/R/Utils/004_plot_predobs.R")

#====================================================================================== -

#Prepare data ----

#get predictor data
spc0 <- readRDS("Analysis/SI_dynamics/preds.rds")

#all controls
spc <- spc0

# #only treated plots
# spc <- spc0 %>% dplyr::filter(Exp == "FPWW023")

# limited controls
# spc <- spc0 %>% dplyr::filter(Exp == "FPWW023" | Gen_Name %in% c("KWS AURUM LP 819.4.04", "ARINA"))

#extract variable names
var_names <- spc %>% dplyr::select(contains("SI_")) %>% names() %>% data.frame()

#get disease severity data
dis_sev <- readRDS("RefData/dis_sev.rds") %>% dplyr::select(Plot_ID, sev)
DATA <- left_join(spc, dis_sev)

plot_seq_inf<- left_join(spc, dis_sev) %>% dplyr::select(-contains("SI_"), -trt, -sev)

#prepare data for modelling
data <- DATA %>% 
  select_if(function(x) !any(is.na(x))) %>% 
  mutate(trt = as.factor(trt)) %>% tidyr::drop_na() %>% 
  dplyr::select(sev, contains("SI_"))
#transform varnames to "legal" names (?)
names(data) <- c("sev", paste("V", 1:(length(data)-1), sep = ""))

saveRDS(data, "Analysis/SI_dynamics/preds_rf_regr_allctrls.rds")

#====================================================================================== -

# FULL MODELS =============== ----

#> PLSR ----

indx <- createMultiFolds(data$sev, k = 10, times = 10)

ctrl <- caret::trainControl(method = "repeatedcv", 
                            index = indx,
                            savePredictions = TRUE,
                            verboseIter = TRUE,
                            selectionFunction = "oneSE")

maxcomp = 15

plsr <- caret::train(sev ~ .,
                     data = data,
                     preProcess = c("center", "scale"),
                     method = "pls",
                     tuneLength = maxcomp, 
                     trControl = ctrl,
                     importance = TRUE)

plot(plsr)

saveRDS(plsr, "O:/Projects/KP0011/3/Analysis/regression/Models_dyn/plsr_limctrls.rds")

#> Cubist ----

#tuning parameter values
train_grid <- expand.grid(committees = c(1, 2, 3, 5, 10, 15, 20, 50),
                          neighbors = c(0, 1, 2, 5))

cubist <- caret::train(sev ~ .,
                       data = data,
                       preProcess = c("center", "scale"),
                       method = "cubist",
                       tuneGrid = train_grid,
                       trControl = ctrl,
                       importance = TRUE)

plot(cubist)

saveRDS(cubist, "O:/Projects/KP0011/3/Analysis/regression/Models_dyn/cubist_limctrls.rds")

#> rf-ranger ----

#tuning parameter values
mtry <- c(1, 2, 5, 9, 14, 20, 30, 45, 70, 100, 200)
min_nodes <- c(1, 2, 5, 10)

tune_grid <- expand.grid(mtry = mtry,
                         splitrule = "variance", #default
                         min.node.size = min_nodes)

rf_ranger <- caret::train(sev ~ .,
                          data = data,
                          preProc = c("center", "scale"),
                          method = "ranger",
                          tuneGrid = tune_grid,
                          importance = "permutation",
                          num.trees = 2000,
                          num.threads = 10, 
                          trControl = ctrl)

plot(rf_ranger)

saveRDS(rf_ranger, "O:/Projects/KP0011/3/Analysis/regression/Models_dyn/rf_limctrls.rds")

#> Ridge ----

#tuning parameter values
ridgeGrid <- data.frame(lambda = seq(0, .15, length = 15))

ridge <- caret::train(sev ~ .,
                      data = data,
                      # preProcess = NULL,
                      preProcess = c("center", "scale"),
                      method = "ridge",
                      tuneGrid = ridgeGrid,
                      trControl = ctrl,
                      importance = TRUE)

plot(ridge)

saveRDS(ridge, "O:/Projects/KP0011/3/Analysis/regression/Models_dyn/ridge_limctrls.rds")


#====================================================================================== -


#Create predobs plots ----

dir <- "O:/Projects/KP0011/3/"
setwd(dir)

#load required data
spcdat <- spc0 <- readRDS("Analysis/SI_dynamics/preds.rds")
dis_sev <- readRDS("O:/Projects/KP0011/3/RefData/dis_sev.rds")

dir <- "O:/Projects/KP0011/3/Analysis/regression/Models_dyn/"
setwd(dir)

#load models
mod_names <- as.list(list.files(pattern = "ctrls"))
mod_names_adj <- as.list(list.files(pattern = "allctrls"))
mods <- lapply(mod_names, readRDS)
mods_adj <- lapply(mod_names_adj, readRDS)

#plot performance profiles
lapply(mods, plot)

#create predicted vs. observed plots
predobsplots <- lapply(mods, plot_predobs, adjust = FALSE, spcdat = spcdat)

pdf("O:/Projects/KP0011/3/Figures/predobs_adj_DYN.pdf", 8.5, 8.5)
gridExtra::grid.arrange(grobs = predobsplots,
                        ncol = 2)
dev.off()

# plot to pdf
pdf("O:/Projects/KP0011/3/Figures/predobs_DYN.pdf", 8.5, 8.5, onefile = TRUE)
for (i in seq(1, length(predobsplots), 3)) {
  gridExtra::grid.arrange(grobs=predobsplots[i:(i+2)], 
                          ncol=2)
}
dev.off()

#====================================================================================== -

#Model generalization ----

#plots included in the training set;
#never include these, to garantee same testing set for all approaches
train_plots <- spc0 %>% dplyr::filter(grepl("FPWW022", Plot_ID)) %>% pull(Plot_ID) %>% unique()

#prepare FPWW023 data
newdata <- readRDS("O:/Projects/KP0011/3/Analysis/SI_dynamics/preds_FPWW022.rds") %>% 
  #drop plots used for training
  dplyr::filter(!Plot_ID %in% train_plots) %>% 
  dplyr::filter(complete.cases(.)) %>% 
  dplyr::select(contains("SI_"))
names(newdata) <- paste("V", 1:(length(newdata)), sep = "")

plot_seq <- readRDS("O:/Projects/KP0011/3/Analysis/SI_dynamics/preds_FPWW022.rds") %>% 
  #drop plots used for training
  dplyr::filter(!Plot_ID %in% train_plots) %>% 
  dplyr::filter(complete.cases(.)) %>% 
  pull(Plot_ID)

#get plot information for desplot
plot_inf <- readRDS("O:/Projects/KP0011/3/Analysis/SI_dynamics/Main_Exp/preds_FPWW022.rds") %>% 
  #drop plots used for training
  mutate(trt = as.factor(trt)) %>% 
  dplyr::select(-contains("SI_")) %>%
  dplyr::filter(Plot_ID %in% plot_seq)

#get severity predictons
preds <- lapply(mods, predict.train, newdata = newdata[ , names(newdata) != "sev"]) %>% 
  lapply(., as.numeric) %>% 
  lapply(., expss::na_if, expss::gt(0.5))

plot_distr <- list()
pred_data <- list()
for(i in 1:length(preds)){
  pred_data[[i]] <- preds[i] %>% as.data.frame(col.names = "V1") %>% 
    mutate(model = paste(mod_names[i]) %>% gsub(".rds", "", .))
  plot_distr[[i]] <- ggplot(pred_data[[i]]) +
    geom_bar(aes(x = V1), stat = "bin", binwidth = 0.01) +
    facet_wrap(~model) +
    scale_x_continuous(limits = c(-0.5, 0.5)) +
    scale_y_continuous(limits = c(0, 50)) +
    geom_vline(xintercept = 0, col = "red") +
    geom_vline(xintercept = 0.05, col = "red", lty = 2) +
    xlab("Predicted STB severity") + ylab("Count") +
    theme_bw()
}

# plot to pdf
pdf("O:/Projects/KP0011/3/Figures/preddist_DYN.pdf", 8.5, 8.5, onefile = TRUE)
for (i in seq(1, length(plot_distr), 3)) {
  gridExtra::grid.arrange(grobs=plot_distr[i:(i+2)], 
                          ncol=2)
}
dev.off()

#====================================================================================== -

#Spatial Patterns ----

pred_des <- lapply(pred_data, function(x) cbind(x, plot_seq) %>% 
                     full_join(., plot_inf, by = c("plot_seq" = "Plot_ID")) %>% 
                     as_tibble() %>% 
                     rename(pred = 1, 
                            algorithm = 2) %>% 
                     mutate(pred = ifelse(pred < 0, 0, pred)))


plot <- list()
for(i in 1:length(pred_des)){
  data <- pred_des[[i]]
  plot[[i]] <- desplot(pred ~ RangeLot + RowLot | Lot,
                       data = data, cex = 0.6, ticks = TRUE, show.key = TRUE,
                       midpoint = "midrange",
                       col.regions = RedGrayBlue,
                       main = paste(unique(pred_des[[i]]$algorithm)))
}

# plot to pdf
pdf("O:/Projects/KP0011/3/Figures/pred_spatial_DYN.pdf", 8.5, 8.5, onefile = TRUE)
for (i in seq(1, length(plot), 3)) {
  gridExtra::grid.arrange(grobs=plot[i:(i+2)], 
                          ncol=2)
}
dev.off()

#END========================== ----

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
    
    #Verbose iter
    print(paste("==> subset size = ", length(train)-1, sep = ""))
    
    #adjust mtry parameter to decreasing predictor set
    #maximum mtry at 200
    mtry <- ceiling(seq(1, length(train[-1]), len = 17)) %>% unique()
    if(any(mtry > 250)){
      mtry <- mtry[-which(mtry >= 250)]
    }
    
    min.node.size <- c(1, 2, 5, 10)

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
                                              neighbors = c(0, 1, 2, 5, ))
    
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

