library(ITSMe)


results_file <- "QSM_results.csv"

# Creating results file if it doesn't already exist
if (!file.exists(results_file)) {
  results <- data.frame(reference = character(), total_length = numeric(),
                        total_volume = numeric(), mean_circ = numeric(),
                        mean_circ_inv = numeric(), length_ord1 = numeric(),
                        volume_ord_1 = numeric(), num_orders = numeric())
  
  write.table(results, file = results_file, sep = ";", row.names = FALSE,
              col.names = TRUE)
}

qsm_analysis <- function(file) {
  
  # Reference = filename without extension
  reference <- tools::file_path_sans_ext(basename(file))
  
  qsm <- read_tree_qsm(path = file)
  
  # Get cumulated length of all detected branches
  total_length <- qsm$treedata$TotalLength[1]
  
  # Get total volume
  total_volume <- (qsm$treedata$TotalVolume[1])*0.001 # converting L into m^3
  
  # Get circumferences from diameters
  diameters <- qsm$branch$diameter
  circumferences <- (diameters*pi)*100 # converting m into cm
  mean_circ <- mean(circumferences)
  
  # Get circumferences of branches that can be inventoried
  inv_circumferences <- circumferences[circumferences >= 20]
  mean_circ_inv <- mean(inv_circumferences)
  
  # Get length and volume of 1st order branch
  length_ord1 <- qsm$treedata$LenBranchOrd[1]
  volume_ord1 <- (qsm$treedata$VolBranchOrd[1])*0.001
  
  # Get number of branch orders
  num_orders <- qsm$treedata$MaxBranchOrder[1]
  
  results_data <- read.csv2(results_file)
  
  # Write results
  if (!(reference %in% results_data$reference)) { # to avoid duplication

    row <- data.frame(reference, total_length, total_volume, mean_circ,
                      mean_circ_inv, length_ord1, volume_ord1, num_orders)
    
    write.table(row, file=results_file, sep=";", append = TRUE, quote = FALSE,
                col.names = FALSE, row.names = FALSE)
  }
}


# List .mat files
qsm_files <- list.files(path = "results/", pattern = "\\.mat$",
                        full.names = TRUE)

# Perform QSM analysis on all .mat files
for (qsm_file in qsm_files) {
  qsm_analysis(file=qsm_file)
}