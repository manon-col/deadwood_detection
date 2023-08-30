library(readr)
library(dplyr)
library(Metrics)

file <- "field_inventory.csv"

df <- read.csv2(file)
df <- df %>%
  filter(plot != "")

# Calculate detection level, detectability, model accuracy

stats <- df %>%
  group_by(plot) %>%
  summarise(detection_level = (n() - sum(detected == "no")) / n(),
            detectability = (n() - sum(detectable == "no")) / n(),
            model_accuracy = sum(detected == "yes" & detectable == "yes")/
              sum(detectable == "yes"))

stats <- rbind(stats, c("Mean",
                        mean(stats$detection_level),
                        mean(stats$detectability),
                        mean(stats$model_accuracy)))

stats <- rbind(stats, c("Min",
                        min(stats$detection_level),
                        min(stats$detectability),
                        min(stats$model_accuracy)))

stats <- rbind(stats, c("Max",
                        max(stats$detection_level),
                        max(stats$detectability),
                        max(stats$model_accuracy)))

print(stats)

# Retrieve volume data and write a csv file

result_df <- data.frame(plot = character(0), element = character(0),
                       reference_volume = numeric(0),
                       volume_detected = numeric(0),
                       volume_detectable = numeric(0))

for (plot in unique(df$plot)) {
  
  path_volumes <- file.path("..", "deadwood", paste0(plot, "_cl_volumes.csv"))
  volumes <- read.csv2(path_volumes)
  
  for (element in unique(df$element[df$plot==plot])) {
    
    subset_df <- df[df$plot == plot & df$element == element, ]
    id_cluster_detected <- strsplit(subset_df$id_cluster_dw, "; ")[[1]]
    id_cluster_detectable <- strsplit(subset_df$id_cluster_np, "; ")[[1]]
    
    result_df <- data.frame()
    detected_volume <- 0
    detectable_volume <- 0
    
    for (id in id_cluster_detected) {
      detected_volume <- detected_volume +
        volumes[volumes$cluster == id, "volume"]
      }
    
    for (id in id_cluster_detectable) {
      detectable_volume <- detectable_volume +
        volumes[volumes$cluster == id, "volume"]
    }
    
    if (length(id_cluster_detected)+length(id_cluster_detectable)>0) {
      
      new_row <- data.frame(plot = plot,
                            element = element,
                            reference_volume = subset_df$total_volume,
                            detected_volume = detected_volume,
                            detectable_volume = detectable_volume)
      
      result_df <- rbind(result_df, new_row)
    }
  }
}

colnames(result_df) <- c("plot", "element", "reference_volume",
                         "detected_volume", "detectable_volume")

write.csv2(result_df, "corresp_volume.csv", row.names = FALSE)

# Calculate total volumes

df_vol <- read.csv2("corresp_volume.csv")

tot_vol <- df_vol %>%
  group_by(plot) %>%
  summarise(total_volume_ref = sum(reference_volume),
            total_volume_detected = sum(detected_volume),
            total_volume_detectable = sum(detectable_volume))

# Calculate bias and RMSE

rmse_detected <- rmse(tot_vol$total_volume_ref, tot_vol$total_volume_detected)
rmse_detectable <- rmse(tot_vol$total_volume_ref, tot_vol$total_volume_detectable)
bias_detected <- bias(tot_vol$total_volume_ref, tot_vol$total_volume_detected)
bias_detectable <- bias(tot_vol$total_volume_ref, tot_vol$total_volume_detectable)

bias_rmse <- data.frame(colnames("total_volume_detectable",
                                 "total_volume_detected"),
                        rownames("bias", "rmse"))

bias_rmse <- rbind(bias_rmse,
                   c(bias_detectable, bias_detected),
                   c(rmse_detectable, rmse_detected))
