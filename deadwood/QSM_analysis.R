library(ITSMe)


# Creating results file if it doesn't already exist
if (!file.exists("QSM_results.csv")) {
  results <- data.frame(reference = character(), total_length = numeric(),
                        total_volume = numeric(), )
  write.table(results, file = "results.csv", sep = ";", row.names = FALSE,
              col.names = TRUE)
}

qsm_analysis <- function(file) {
  
  # Reference = filename without extension
  # reference <- tools::file_path_sans_ext(basename(file))
  
  qsm <- read_tree_qsm(path = file)
  
  # Get cumulated length of all detected branches
  total_length <- qsm$treedata$TotalLength[1]
  
  # Get total volume
  total_volume <- (qsm$treedata$TotalVolume[1])*0.001 # converting L into m^3
  
  # Get circumferences from diameters
  diameters <- qsm$branch$diameter
  circumferences <- (diameters*pi)*100 # converting m into cm
  
  inv_circumferences <- circumferences[circumferences >= 20]
  
  mean_circ <- mean(circumferences)
  mean_circ_inv <- mean(inv_circumferences)
  
  # Get length, volume and circumference of 1st order branch
  length_ord1 <- qsm$treedata$LenBranchOrd[1]
  print(length_ord1)
  
  
  # results_data <- read.csv2(results_file)
  # 
  # if (!(reference %in% results_data$reference)) { # to avoid duplication
  #   
  #   row <- data.frame(reference, total_length, mid_circ, volume)
  #   write.table(row, file=results_file, sep=";", append = TRUE, quote = FALSE,
  #               col.names = FALSE, row.names = FALSE)
}


# List .mat files
qsm_files <- list.files(path = "results/", pattern = "\\.mat$", full.names = TRUE)

# Clipping all las files
for (qsm_file in qsm_files) {
  
  qsm_analysis(file=qsm_file)
  
}