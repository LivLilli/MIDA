library(mice)
library(pracma)
library(BBmisc)
library(Metrics)
set.seed(987)
'
mice_imputation(df)

Inputs:
  - corrupted df;

Outputs:
  - vector of 3 elements: mean, max and min of RMSEsum
'
mice_imputation = function(input_df){
  # initialize vector of 5 rmse (5 imputations)
  imp_5 = c()
  # normalize input
  input_df = normalize(input_df, method = 'range')
  # mice
  imp = mice(input_df, m = 5, method = 'pmm')
  # na to zero
  input_df[is.na(input_df)] = 0
  for (j in 1:5){
    # imputed df of j imputation
    result_df = complete(imp,j)
    all_rmse = c()
    result_df = normalize(result_df, method = 'range')
    for (i in names(result_df)){
      pred_vector = result_df[[i]]
      actual_vector = input_df[[i]]
      # rmse of i-feature
      single_rmse = rmse(actual_vector, pred_vector)
      all_rmse[i] = single_rmse
    }
    # sum of rmse among features
    rmse_sum = sum(all_rmse)
    # append rmse sum of j-imputation to vector
    imp_5[j] = rmse_sum
  }
  # compute mean,amx,min of 5 rmse sum
  mean_rmse = mean(imp_5)
  max_rmse = max(imp_5)
  min_rmse = min(imp_5)
  result = c(mean_rmse, max_rmse, min_rmse)
  return(result)
}


'
save_to_csv(df, name)

Inputs:
  - df to save;
  - name of the csv file.

Outputs:
  - saves csv file.
'
save_to_csv = function(df, name){
  if (file.exists(name)) 
    #Delete file if it exists
    file.remove(name)
  
  write.csv(df, name)
}

# vector of datasets names
dataset = c('BH', 'BC', 'DN', 'GL', 'HV', 'IS',
            'ON', 'SL', 'SR', 'ST', 'SN', 'SB', 'VC', 'VW', 'ZO')
# vectors with names of 4 missingness types applied
methods = c('mcarU', 'mcarR', 'mnarU', 'mnarR')
# list of df names
rows = c()
i = 1
for  (df in dataset){
  for (missingness in methods){
    name = paste(df,missingness)
    rows[i] = name
    i = i+1
  }
}

# initialize df of results
result = data.frame(0,0,0)
colnames(result) = c('mean', 'max', 'min')
for (df_name in rows){
  current_dataset = dataset[i]
  # opne file
  file = paste('corrupted_datasets/',df_name, '.csv', sep = '')
  df = read.csv(file, header= TRUE, row.names = 1)
  # apply mice
  rmse_vector = mice_imputation(df)
  # add results to df
  r = data.frame(rmse_vector[1],rmse_vector[2],rmse_vector[3])
  names(r) = c('mean', 'max','min')
  result = rbind(result,r)
}

# drop first row
result = result[-c(1),]
# rename rows
row.names(result) = rows
# save
save_to_csv(result, 'results/mice_results.csv')




