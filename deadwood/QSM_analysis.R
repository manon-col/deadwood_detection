library(ITSMe)


results_file <- "QSM_results.csv"

# Creating results file if it doesn't already exist
if (!file.exists(results_file)) {
  results <- data.frame(reference = character(), total_length = numeric(),
                        total_volume = numeric(), mean_circ = numeric(),
                        num_orders = numeric(), length_ord01 = numeric(),
                        volume_ord01 = numeric(), circ_ord01 = numeric(),
                        inv_circ_ord01 = numeric())
  
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
  
  # Get circumferences from radius
  radius <- qsm$cylinder$radius
  circumferences <- radius*2*pi*100 # converting m into cm
  mean_circ <- mean(circumferences)
  
  # Get number of branch orders
  num_orders <- qsm$treedata$MaxBranchOrder[1]
  
  # Get length and volume of branches of order 0 (trunk) and 1
  length_ord01 <- qsm$treedata$LenBranchOrd[1]+qsm$treedata$LenBranchOrd[2]
  volume_ord01 <- (qsm$treedata$VolBranchOrd[1]+qsm$treedata$VolBranchOrd[2])*0.001
  
  # Get cylinders of order 0 and 1
  cyl_orders <- qsm$cylinder$BranchOrder
  cylinders <- which((cyl_orders)<=1)
  
  # Get circumferences of orders 0 and 1
  circs <- c()
  
  # Circumferences in inventory range (>= 20cm)
  inv_circs <- c()
  
  for (cyl in cylinders)  {
    
    circ <- qsm$cylinder$radius[cyl]*2*pi*100 # metres into centimetres
    circs <-c(circs, circ)
    
    if (circ >= 20) {
      inv_circs <- c(inv_circs, circ)
    }
  }
  
  # Get mean circumference for orders 0 and 1
  circ_ord01 <- mean(circs)
  inv_circ_ord01 <- mean(inv_circs)

  results_data <- read.csv2(results_file)
  
  # Write results
  if (!(reference %in% results_data$reference)) { # to avoid duplication

    row <- data.frame(reference, total_length, total_volume, mean_circ,
                      num_orders, length_ord01, volume_ord01, circ_ord01,
                      inv_circ_ord01)
    
    write.table(row, file=results_file, sep=";", append = TRUE, quote = FALSE,
                col.names = FALSE, row.names = FALSE)
  }
}


# List .mat files
qsm_files <- list.files(path = "QSMs/", pattern = "\\.mat$",
                        full.names = TRUE)

# Perform QSM analysis on all .mat files
for (qsm_file in qsm_files) {
  qsm_analysis(file=qsm_file)
}