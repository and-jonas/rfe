
#====================================================================================== -

#HEADER ----

# AUTHOR: Jonas Anderegg

# Perform recursive feature elimination using different base-learners
# Get robust feature ranks
# create performance profile plots

#====================================================================================== -

.libPaths("T:/R3UserLibs_rds")
# .libPaths("T:/R3UserLibs")
library(tidyverse)
library(caret)
library(ranger)
library(Cubist)
library(parallel)
library(doParallel)

source("O:/Projects/KP0011/3/rfe/rfe_utils.R")

dir <- "O:/Projects/KP0011/3/"
setwd(dir)

#====================================================================================== -

data <- readRDS("Analysis/SI_dynamics/preds_rf_regr_allctrls.rds")

# The candidate set of the number of predictors to evaluate
subsets <- c(length(data), 200, 150, 120, 105, 90, 75, 
             60, 50, 40, 35, 30, 25, 20, 17, 14, 12:1)

rfe <- perform_rfe(response = "sev", base_learner = "ranger",
                   p = 0.7, times = 3, groups = 9, 
                   subsets = subsets, data = data,
                   runParallel = FALSE,
                   importance = "permutation",
                   num.trees = 1000)

cubist <- readRDS("O:/Projects/KP0011/3/Analysis/RFE/rfe_regr_dyn_cubist.rds")
ranger <- readRDS("O:/Projects/KP0011/3/Analysis/RFE/rfe_regr_dyn_rf.rds")

OUT_cubist <- tidy_rfe_output(cubist, "cubist")
OUT_ranger <- tidy_rfe_output(ranger, "ranger")

OUT <- bind_rows(OUT_cubist[[1]], OUT_ranger[[1]])

PROF <- plot_perf_profile(OUT)
