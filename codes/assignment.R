##################
# Setup
##################

library(tidyverse)
library(datasets)
library(MASS)
library(ISLR)
library(caret)
library(janitor)
library(skimr)
library(GGally)
library(RColorBrewer)
library(kableExtra)
library(factoextra)
library(NbClust)

color <- c(brewer.pal( 3, "Set2" )[1], brewer.pal( 3, "Set2" )[2], brewer.pal( 3, "Set2" )[3], brewer.pal( 3, "Set2" )[5])

#########################################################
# 1. Supervised learning with penalized models and PCA
#########################################################

# goal is to predict the logarithm of the property value: logTotalValue

# import data
data <- readRDS(url('http://www.jaredlander.com/data/manhattan_Train.rds')) %>%
  mutate(logTotalValue = log(TotalValue)) %>%
  drop_na()

# change variable names to lowercase 
data <- clean_names(data)

# a, data exploration -----------------------------------------------------

# most numerical variables have a skewed distribution
skim(data)

# convert some variables to factor which are stored as numeric
data <- data %>% mutate(council = factor(council),
                        police_prct = factor(police_prct),
                        health_area = factor(health_area))

# loess graphs between target variable and numeric variables
numericals <- c("built_far", "num_floors", "res_area", "bldg_area", "facil_far", "comm_far", "resid_far", "bldg_depth", "bldg_front", "lot_front", "lot_depth", "units_res", "units_total", "com_area", "lot_area", "num_bldgs", "other_area", "strge_area", "garage_area", "retail_area", "office_area", "easements", "factry_area")

plist = sapply(numericals, function(col) {
  ggplot(data[1:5000,], aes_string(x = col, y = "log_total_value")) + geom_smooth(method="loess", colour = color[1]) + geom_point() +
    labs( x = paste0("\n", col), y = "ln(total value) \n", title = paste0("Pattern of association between ln(total value) and ", col)) +
    theme( panel.grid.minor.x = element_blank(), 
           plot.title = element_text( size = 12, face = "bold", hjust = 0.5 ) ) +
    theme_bw()
}, simplify=FALSE)

# filter extreme values and add logs of skewed numerical variables

data <- data %>% dplyr::filter(units_total<2000 & lot_front<2000 & built_far<90) %>% 
  mutate( log_built_far = ifelse(built_far == 0, 0, log(built_far)),
                         log_bldg_area = ifelse(bldg_area == 0, 0, log(bldg_area)),
                         log_res_area = ifelse(res_area == 0, 0, log(res_area)),
                         log_lot_area = ifelse(lot_area == 0, 0, log(lot_area)),
                         log_com_area = ifelse(com_area == 0, 0, log(com_area)),
                         log_units_total = ifelse(units_total == 0, 0, log(units_total)),
                         log_units_res = ifelse(units_res == 0, 0, log(units_res)),
                         log_lot_depth = ifelse(lot_depth == 0, 0, log(lot_depth)),
                         log_lot_front = ifelse(lot_front == 0, 0, log(lot_front)),
                         log_bldg_front = ifelse(bldg_front == 0, 0, log(bldg_front)),
                         log_bldg_depth = ifelse(bldg_depth == 0, 0, log(bldg_depth)),
                         log_easements = ifelse(easements == 0, 0, log(easements)),
                         log_office_area = ifelse(office_area == 0, 0, log(office_area)),
                         log_retail_area = ifelse(retail_area == 0, 0, log(retail_area)),
                         log_garage_area = ifelse(garage_area == 0, 0, log(garage_area)),
                         log_strge_area = ifelse(strge_area == 0, 0, log(strge_area)),    
                         log_factry_area = ifelse(factry_area == 0, 0, log(factry_area)),   
                         log_other_area = ifelse(other_area == 0, 0, log(other_area)),  
                         log_num_bldgs = ifelse(num_bldgs == 0, 0, log(num_bldgs)),
                         log_num_floors = ifelse(num_floors == 0, 0, log(num_floors)),
                         log_comm_far = ifelse(comm_far == 0, 0, log(comm_far)))

# check correlation between numerical outcome variables and target variable
ggcorr(data[,3:47])

# create list of numerical predictors with high correlation
num_high <- c("built_far", "num_floors", "res_area", "bldg_area")

# check correlation and distribution for highly correlated numerical predictors
ggpairs(data, columns = c("log_total_value", "built_far", "num_floors", "res_area", "bldg_area"))

# create list of numerical predictors with medium correlation
num_med <- c("facil_far", "comm_far", "resid_far", "bldg_front", "lot_front", "units_res", "units_total", "com_area", "lot_area")

# create list of numerical predictors with low correlation
num_low <- c("num_bldgs", "other_area", "strge_area", "garage_area", "retail_area", "office_area", "easements", "factry_area", "lot_depth", "bldg_depth")

# check conditional means of target variable for categorical variables
categoricals <- c("school_district", "fire_service", "zone_dist1", "zone_dist2", "zone_dist3",  "zone_dist4", "class", "land_use", "owner_type", "extension", "proximity", "irregular_lot", "lot_type", "basement_type", "landmark", "built", "historic_district", "high", "police_prct", "council", "health_area")          

for (i in 1:length(categoricals)) {
  data %>%
    group_by(get(categoricals[i])) %>%
    summarise(mean_log_total_value = mean(log_total_value),  n=n()) %>%
    print
}                          

# create a list of categoricals where the means vary more between categories
cat_high <- c("school_district", "zone_dist1",  "class", "owner_type", "extension", "proximity", "lot_type", "basement_type", "landmark", "built", "high", "police_prct", "council", "health_area")

# create a list of categoricals where the means vary little between categories
cat_low <- c("fire_service", "irregular_lot", "historic_district", "land_use", "zone_dist2", "zone_dist3",  "zone_dist4")


# b, separate training and test sets --------------------------------------

set.seed(3)
training_ratio <- 0.3
train_indices <- createDataPartition(
  y = data[["log_total_value"]],
  times = 1,
  p = training_ratio,
  list = FALSE
) %>% as.vector()
data_train <- data[train_indices, ]
data_test <- data[-train_indices, ]


# c, linear regression with 10-fold cross validation ----------------------

# define predictor sets
# level variables
level_1 <- c(num_high, cat_high)
level_2 <- c(num_high, num_med, cat_high)
level_3 <- c(num_high, num_med, cat_high, cat_low)
level_4 <- c(num_high, num_med, num_low, cat_high)
level_5 <- c(num_high, num_med, num_low, cat_high, cat_low)

# ln variables
log_1 <- c("log_built_far", "log_num_floors", "log_res_area", "log_bldg_area", cat_high)
log_2 <- c(log_1, "facil_far", "comm_far", "resid_far", "log_bldg_front", "log_lot_front", "log_units_res", "log_units_total", "log_com_area", "log_lot_area")
log_3 <- c(log_2, cat_low)
log_4 <- c(log_2, "log_num_bldgs", "log_other_area", "log_strge_area", "log_garage_area", "log_retail_area", "log_office_area", "log_easements", "log_factry_area", "log_lot_depth", "log_bldg_depth")
log_5 <- c(log_3, "log_num_bldgs", "log_other_area", "log_strge_area", "log_garage_area", "log_retail_area", "log_office_area", "log_easements", "log_factry_area", "log_lot_depth", "log_bldg_depth")

# set cv parameters
train_control <- trainControl(method = "cv", number = 10)

# run model
# model with level_1
set.seed(3)
system.time({
  ols_model1 <- train(
    formula(paste0("log_total_value ~", paste0(level_1, collapse = " + "))),
    data = data_train,
    method = "lm",
    trControl = train_control
  )
})

# model with level_2
set.seed(3)
system.time({
  ols_model2 <- train(
    formula(paste0("log_total_value ~", paste0(level_2, collapse = " + "))),
    data = data_train,
    method = "lm",
    trControl = train_control
  )
})

# model with log_1
set.seed(3)
system.time({
  ols_model3 <- train(
    formula(paste0("log_total_value ~", paste0(log_1, collapse = " + "))),
    data = data_train,
    method = "lm",
    trControl = train_control
  )
})

# model with log_2
set.seed(3)
system.time({
  ols_model4 <- train(
    formula(paste0("log_total_value ~", paste0(log_2, collapse = " + "))),
    data = data_train,
    method = "lm",
    trControl = train_control
  )
})

# CV RMSE table
temp_models <-
  list("OLS 1 (level)" = ols_model1,
       "OLS 2 (level)" = ols_model2,
       "OLS 3 (log)" = ols_model3,
       "OLS 4 (log)" = ols_model4)

result_temp <- resamples(temp_models) %>% summary()

# get test RMSE
result_rmse <- imap(temp_models, ~{
  mean(result_temp$values[[paste0(.y,"~RMSE")]])
}) %>% unlist() %>% as.data.frame() %>%
  rename("CV RMSE" = ".")

knitr::kable( result_rmse, caption = "Performance comparison of OLS models", digits = 2 ) %>% kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))


# d, penalized linear models ----------------------------------------------

# ridge model
ridge_tune_grid <- expand.grid(
  "alpha" = c(0),
  "lambda" = seq(0.05, 0.5, by = 0.025)
)

set.seed(3)
ridge_model <- train(
  formula(paste0("log_total_value ~", paste0(log_2, collapse = " + "))),
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = ridge_tune_grid,
  trControl = train_control
)

# lasso model
tenpowers <- 10^seq(-1, -5, by = -1)

lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = c(tenpowers, tenpowers / 2) 
)

set.seed(3)
lasso_model <- train(
  formula(paste0("log_total_value ~", paste0(log_3, collapse = " + "))),
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = lasso_tune_grid,
  trControl = train_control
)

# elastic net model
enet_tune_grid <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = union(lasso_tune_grid[["lambda"]], ridge_tune_grid[["lambda"]])
)

set.seed(3)
enet_model <- train(
  formula(paste0("log_total_value ~", paste0(log_3, collapse = " + "))),
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = enet_tune_grid,
  trControl = train_control
)

# save model to rds
saveRDS(enet_model, 'C:/CEU/Winter_Term/Data_Science_1/Assignment_1/enet_model.rds')
enet_model <- read_rds('C:/CEU/Winter_Term/Data_Science_1/Assignment_1/enet_model.rds')

# model comparison
temp_models <-
  list("OLS" = ols_model4,
       "Ridge" = ridge_model,
       "LASSO" = lasso_model,
       "Elastic Net" = enet_model)

result_temp <- resamples(temp_models) %>% summary()

# get test RMSE
result_rmse <- imap(temp_models, ~{
  mean(result_temp$values[[paste0(.y,"~RMSE")]])
}) %>% unlist() %>% as.data.frame() %>%
  rename("CV RMSE" = ".")

knitr::kable( result_rmse, caption = "Performance comparison of OLS and penalized models", digits = 2 ) %>% kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))

comp_plot <- bwplot(resamples(temp_models))
diff <- summary(diff(resamples(temp_models)))

# e, the simplest model which is still good enough ------------------------

# set cv parameters
train_control <- trainControl(method = "cv", number = 10, selectionFunction = "oneSE")

# same predictor set but choose best model based on one SE
# the lambda parameter is twice as big as before meaning that more coefficients are shrunk to
# zero which results in a more simple model
# RMSE is still 0.52
set.seed(3)
lasso_model_onese <- train(
  formula(paste0("log_total_value ~", paste0(log_3, collapse = " + "))),
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = lasso_tune_grid,
  trControl = train_control
)

# get the parameters
l1 <- rbind(lasso_model$bestTune, lasso_model_onese$bestTune)
rownames(l1) <- c('LASSO', 'LASSO 1 SE')

# calculate the number of non-null coefficients
lasso_model_coef <- coef(
  lasso_model$finalModel,
  lasso_model$bestTune$lambda)
lasso_non_null <- length(lasso_model_coef[lasso_model_coef != 0])

lasso_model_onese_coef <- coef(
  lasso_model_onese$finalModel,
  lasso_model_onese$bestTune$lambda)
lasso_onese_non_null <- length(lasso_model_onese_coef[lasso_model_onese_coef != 0])
num_coef <- c(lasso_non_null, lasso_onese_non_null)

# add them to the table
l1 <- cbind(l1, num_coef)

knitr::kable( l1, caption = "Difference between original and 1 SE LASSO models", digits = 2 ) %>% kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))

# same predictor set but choose best model based on one SE
# alpha parameter significantly decreases
# the penalty for the sum of squared coefficients is smaller
# but at the same time lamdba increases a lot which means that more coefficients are shrunk to zero
# RMSE is still 0.52
set.seed(3)
enet_model_onese <- train(
  formula(paste0("log_total_value ~", paste0(log_3, collapse = " + "))),
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale"),
  tuneGrid = enet_tune_grid,
  trControl = train_control
)

# save model to rds
saveRDS(enet_model_onese, 'C:/CEU/Winter_Term/Data_Science_1/Assignment_1/enet_model_onese.rds')
enet_model_onese <- read_rds('C:/CEU/Winter_Term/Data_Science_1/Assignment_1/enet_model_onese.rds')

# get the parameters
e1 <- rbind(enet_model$bestTune, enet_model_onese$bestTune)
rownames(e1) <- c('Elastic Net', 'Elastic Net 1 SE')

# calculate the number of non-null coefficients
enet_model_coef <- coef(
  enet_model$finalModel,
  enet_model$bestTune$lambda)
enet_non_null <- length(enet_model_coef[enet_model_coef != 0])

enet_model_onese_coef <- coef(
  enet_model_onese$finalModel,
  enet_model_onese$bestTune$lambda)
enet_onese_non_null <- length(enet_model_onese_coef[enet_model_onese_coef != 0])
num_coef <- c(enet_non_null, enet_onese_non_null)

# add them to the table
e1 <- cbind(e1, num_coef)

knitr::kable( e1, caption = "Difference between original and 1 SE Elastic Net models", digits = 3 ) %>% kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))


# f, PCA for the linear model ---------------------------------------------

# set cv parameters
train_control <- trainControl(method = "cv", number = 10)

tune_grid <- data.frame(ncomp = 1:60)

set.seed(3)
pcr_fit <- train(
  formula(paste0("log_total_value ~", paste0(log_2, collapse = " + "))),
  data = data_train,
  method = "pcr",
  trControl = train_control,
  tuneGrid = tune_grid,
  preProcess = c("center", "scale")
)
pcr_fit

# optimal number of principal components is 55 but RMSE is higher than before: 0.648
# the individual components only account for a small share of the total variance
pcr_fit$finalModel$loadings

# g, PCA for penalized models  --------------------------------------------

train_control <- trainControl(
                    method = "cv",
                    number = 10,
                    preProcOptions = list(thres = 0.95))

# ridge model
ridge_tune_grid <- expand.grid(
  "alpha" = c(0),
  "lambda" = seq(0.05, 0.5, by = 0.025)
)

set.seed(3)
ridge_model_pca <- train(
  formula(paste0("log_total_value ~", paste0(log_2, collapse = " + "))),
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale", "pca", "nzv"),
  tuneGrid = ridge_tune_grid,
  trControl = train_control
)

# save model
saveRDS(ridge_model_pca, 'C:/CEU/Winter_Term/Data_Science_1/Assignment_1/ridge_model_pca.rds')
ridge_model_pca <- read_rds('C:/CEU/Winter_Term/Data_Science_1/Assignment_1/ridge_model_pca.rds')

# lasso model
tenpowers <- 10^seq(-1, -5, by = -1)

lasso_tune_grid <- expand.grid(
  "alpha" = c(1),
  "lambda" = c(tenpowers, tenpowers / 2) 
)

set.seed(3)
lasso_model_pca <- train(
  formula(paste0("log_total_value ~", paste0(log_3, collapse = " + "))),
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale", "pca", "nzv"),
  tuneGrid = lasso_tune_grid,
  trControl = train_control
)

# save model
saveRDS(lasso_model_pca, 'C:/CEU/Winter_Term/Data_Science_1/Assignment_1/lasso_model_pca.rds')
lasso_model_pca <- read_rds('C:/CEU/Winter_Term/Data_Science_1/Assignment_1/lasso_model_pca.rds')

# elastic net model
enet_tune_grid <- expand.grid(
  "alpha" = seq(0, 1, by = 0.1),
  "lambda" = union(lasso_tune_grid[["lambda"]], ridge_tune_grid[["lambda"]])
)

set.seed(3)
enet_model_pca <- train(
  formula(paste0("log_total_value ~", paste0(log_3, collapse = " + "))),
  data = data_train,
  method = "glmnet",
  preProcess = c("center", "scale", "pca", "nzv"),
  tuneGrid = enet_tune_grid,
  trControl = train_control
)

# save model
saveRDS(enet_model_pca, 'C:/CEU/Winter_Term/Data_Science_1/Assignment_1/enet_model_pca.rds')
enet_model_pca <- read_rds('C:/CEU/Winter_Term/Data_Science_1/Assignment_1/enet_model_pca.rds')

# comparison table
temp_models <-
  list(
       "Ridge with PCA" = ridge_model_pca,
       "LASSO with PCA" = lasso_model_pca,
       "Elastic Net with PCA" = enet_model_pca)

result_temp <- resamples(temp_models) %>% summary()

# get test RMSE
result_rmse <- imap(temp_models, ~{
  mean(result_temp$values[[paste0(.y,"~RMSE")]])
}) %>% unlist() %>% as.data.frame() %>%
  rename("CV RMSE" = ".")

knitr::kable( result_rmse, caption = "Performance comparison of penalized models with PCA", digits = 2 ) %>% kable_styling( position = "center", latex_options = 'hold_position', bootstrap_options = c("striped", "hover"))


# h, evaluate best model on test set --------------------------------------

# RMSE is 0.536 which is close to that of the train data
RMSE(predict(lasso_model_onese, newdata = data_test), data_test[["log_total_value"]])

#########################################################
# 2. Clustering on the USArrests dataset
#########################################################

# import data
data <- USArrests

# a, data pre-processing steps before clustering --------------------------

# observations need to be standardized because otherwise the distances between observations
# would also be influenced by the units of measurements

# standardize observations
data_stand <- as.data.frame(scale(data))

# b, determine K ----------------------------------------------------------

# look at the elbow point
fviz_nbclust(data_stand, kmeans, method = "wss")

# use NbCLust to determine K
nb <- NbClust(data_stand, method = "kmeans", min.nc = 2, max.nc = 10, index = "all")
# the best number of clusters is 2


# c, cluster observations and plot them -----------------------------------

# create 2 clusters
km_2 <- kmeans(data_stand, centers = 2, nstart = 20)

# add cluster labels to df
data_w_clusters_2 <- mutate(data_stand, cluster = factor(km_2$cluster))

# plot observations coloured by clusters
ggplot(data_w_clusters_2, aes(x = UrbanPop, y = Murder, color = cluster)) +
  geom_point() +
  theme_bw() +
  labs( x='\n Urban Population', y='Murder \n', title = 'Data split into two clusters') +
  theme( panel.grid.minor.x = element_blank(), 
         plot.title = element_text( size = 12, face = "bold", hjust = 0.5 ) )

# same with 3 clusters
km_3 <- kmeans(data_stand, centers = 3, nstart = 20)

# add cluster labels to df
data_w_clusters_3 <- mutate(data_stand, cluster = factor(km_3$cluster))

# plot observations coloured by clusters
ggplot(data_w_clusters_3, aes(x = UrbanPop, y = Murder, color = cluster)) +
  geom_point() +
  theme_bw() +
  labs( x='\n Urban Population', y='Murder \n', title = 'Data split into three clusters') +
  theme( panel.grid.minor.x = element_blank(), 
         plot.title = element_text( size = 12, face = "bold", hjust = 0.5 ) )

# d, perform and plot PCA -------------------------------------------------

# perform PCA
pca_result <- prcomp(data, scale = TRUE)
first_two_pc <- as_tibble(pca_result$x[, 1:2])

# add cluster labels from previous exercise
data_w_clusters_pca_1 <- mutate(first_two_pc, cluster = factor(km_2$cluster))

# plot clusters
ggplot(data_w_clusters_pca_1, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point() +
  theme_bw() +
  labs( x='\n PC1', y='PC2 \n', title = 'Data split into two clusters (PCA)') +
  theme( panel.grid.minor.x = element_blank(), 
         plot.title = element_text( size = 12, face = "bold", hjust = 0.5 ) )

# create clusters for PCA
km_pca <- kmeans(first_two_pc, centers = 2, nstart = 20)

# add cluster labels to df
data_w_clusters_pca_2 <- mutate(first_two_pc, cluster = factor(km_pca$cluster))

# plot clusters
ggplot(data_w_clusters_pca_2, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point() +
  theme_bw() +
  labs( x='\n PC1', y='PC2 \n', title = 'Data split into two clusters (PCA)') +
  theme( panel.grid.minor.x = element_blank(), 
         plot.title = element_text( size = 12, face = "bold", hjust = 0.5 ) )

#########################################################
# 3. PCA of high-dimensional data
#########################################################

# import data
genes <- read_csv("https://www.statlearning.com/s/Ch10Ex11.csv", col_names = FALSE) %>%
  t() %>% as_tibble()  # the original dataset is of dimension 1000x40 so we transpose it
dim(data)


# a, Perform PCA on this data with scaling features -----------------------

pca_result <- prcomp(genes, scale. = TRUE)
print(pca_result)

# b, Visualize data points in the space of the first two pc-s --------

fviz_pca_ind(pca_result, axes = c(1,2))


# c, Which individual features can matter the most in separating --------

# check loadings of features for PC1
pc1_loadings <- pca_result$rotation[, "PC1"]
pc1_df <- as.data.frame(pc1_loadings)

# get the two features with the largest coordinates
pc1_df %>% arrange(abs(pc1_loadings)) %>% top_n(2)
# V589 and V502

# plot observations by these two features
# add healthy column to genes
genes <- genes %>% mutate(healthy = factor(c(rep(1, 20), rep(0, 20))))

# plot
ggplot(data = genes, aes( x = V589, y = V502, color = healthy )) +
  geom_point() +
  theme_bw() +
  labs( x='\n V589', y='V502 \n', title = 'Separating healthy and non-healthy individuals') +
  theme( panel.grid.minor.x = element_blank(), 
         plot.title = element_text( size = 12, face = "bold", hjust = 0.5 ) )
