library(readr)
library(dplyr)
library(lmtest)
library(Metrics)
library(viridis)
library(ggplot2)
library(ggpmisc)
library(ggpubr)
library(car)

file <- "field_inventory.csv"

df <- read.csv2(file)
df <- df %>%
  filter(plot != "")

# Calculate detection level, detectability, model accuracy

stats <- df %>%
  group_by(plot) %>%
  summarise(nb_reference = n(),
            nb_detectable = sum(detectable == "yes"),
            nb_detected = sum(detected == "yes"),
            detectability = 100*(n() - sum(detectable == "no")) / n(),
            detection_level = 100*(n() - sum(detected == "no")) / n(),
            model_accuracy = 100*sum(detected == "yes" & detectable == "yes")/
              sum(detectable == "yes"))

stats <- rbind(stats, c("Mean",
                        mean(stats$nb_reference),
                        mean(stats$nb_detectable),
                        mean(stats$nb_detected),
                        mean(stats$detectability),
                        mean(stats$detection_level),
                        mean(stats$model_accuracy)))

stats <- rbind(stats, c("Min",
                        min(stats$nb_reference),
                        min(stats$nb_detectable),
                        min(stats$nb_detected),
                        min(stats$detectability),
                        min(stats$detection_level),
                        min(stats$model_accuracy)))

stats <- rbind(stats, c("Max",
                        max(stats$nb_reference),
                        max(stats$nb_detectable),
                        max(stats$nb_detected),
                        max(stats$detectability),
                        max(stats$detection_level),
                        max(stats$model_accuracy)))

# Round with 2 digits
stats <- stats %>%
  mutate_at(vars(-all_of("plot")), as.numeric) %>%
  mutate_if(is.numeric, round, digits = 2)

print(stats)
write.csv2(stats, "plot_count_accuracy.csv", row.names = FALSE) # save file

# Retrieve volume data at the element level, calculate total volume at the plot
# level


vol_df <- data.frame(plot = character(0), element = character(0),
                       reference_volume = numeric(0),
                       detectable_volume = numeric(0),
                       detected_volume = numeric(0))

plot_vol_df <- data.frame(plot = character(0),
                          total_reference_volume = numeric(0),
                          total_detectable_volume = numeric(0),
                          percent_detectable = numeric(0),
                          total_detected_volume = numeric(0),
                          percent_detected = numeric(0),
                          model_acc_percent = numeric(0))

for (plot in unique(df$plot)) {

  path_volumes <- file.path("..", "deadwood", paste0(plot, "_cl_volumes.csv"))
  volumes <- read.csv(path_volumes)
  
  plot_detected_volume <- 0
  all_id_detected <- c()
  plot_detectable_volume <- 0
  all_id_detectable <- c()
  
  for (element in unique(df$element[df$plot==plot])) {
    
    subset_df <- df[df$plot == plot & df$element == element, ]
    id_cluster_detected <- strsplit(subset_df$id_cluster_detected, "; ")[[1]]
    id_cluster_detectable <- strsplit(subset_df$id_cluster_detectable, "; ")[[1]]
    detected_volume <- 0
    detectable_volume <- 0

    for (id in id_cluster_detected) {
      volume <- volumes[volumes$cluster == id, "volume"]
      detected_volume <- detected_volume + volume
      
      if (!(id %in% all_id_detected)) {
        plot_detected_volume <- plot_detected_volume + volume
      }
      all_id_detected <- c(all_id_detected, id)
    }
    
    for (id in id_cluster_detectable) {
      volume <- volumes[volumes$cluster == id, "volume"]
      detectable_volume <- detectable_volume + volume
      
      if (!(id %in% all_id_detectable)) {
        plot_detectable_volume <- plot_detectable_volume + volume
      }
      all_id_detectable <- c(all_id_detectable, id)
    }
    
    if (length(id_cluster_detected)+length(id_cluster_detectable)>0) {
      
      new_row <- data.frame(plot = plot,
                            element = element,
                            reference_volume = subset_df$total_volume,
                            detectable_volume = detectable_volume,
                            detected_volume = detected_volume)
      
      vol_df <- rbind(vol_df, new_row)
    }
  }
  
  new_row <- data.frame(plot=plot,
                        total_reference_volume =
                          sum(df[df$plot==plot,]$total_volume),
                        total_detectable_volume = plot_detectable_volume,
                        percent_detectable=100*plot_detectable_volume/
                          sum(df[df$plot==plot,]$total_volume),
                        total_detected_volume = plot_detected_volume,
                        percent_detected=100*plot_detected_volume/
                          sum(df[df$plot==plot,]$total_volume),
                        model_acc_percent=100*plot_detected_volume/
                          plot_detectable_volume)

  plot_vol_df <- rbind(plot_vol_df, new_row)
}

# Save csv for volume data at element level

colnames(vol_df) <- c("plot", "element", "reference_volume",
                      "detectable_volume", "detected_volume")
write.csv2(vol_df, "element_volumes.csv", row.names = FALSE)

# Save csv for volume data at plot level

plot_vol_df <- rbind(plot_vol_df,
                     c("Mean",
                       mean(plot_vol_df$total_reference_volume),
                       mean(plot_vol_df$total_detectable_volume),
                       mean(plot_vol_df$percent_detectable),
                       mean(plot_vol_df$total_detected_volume),
                       mean(plot_vol_df$percent_detected),
                       mean(plot_vol_df$model_acc_percent)),
                     c("Min",
                       min(plot_vol_df$total_reference_volume),
                       min(plot_vol_df$total_detectable_volume),
                       min(plot_vol_df$percent_detectable),
                       min(plot_vol_df$total_detected_volume),
                       min(plot_vol_df$percent_detected),
                       min(plot_vol_df$model_acc_percent)),
                     c("Max",
                       max(plot_vol_df$total_reference_volume),
                       max(plot_vol_df$total_detectable_volume),
                       max(plot_vol_df$percent_detectable),
                       max(plot_vol_df$total_detected_volume),
                       max(plot_vol_df$percent_detected),
                       max(plot_vol_df$model_acc_percent)))

# Round with 3 digits
plot_vol_df <- plot_vol_df %>%
  mutate_at(vars(-all_of("plot")), as.numeric) %>%
  mutate_if(is.numeric, round, digits = 3)
colnames(plot_vol_df) <- c("plot", "total_reference_volume",
                         "total_detectable_volume", "percent_detectable",
                         "total_detected_volume", "percent_detected",
                         "model_acc")
write.csv2(plot_vol_df, "plot_volumes.csv", row.names = FALSE)

# Calculate bias and RMSE for matched volumes

df_plot <- read.csv2("plot_volumes.csv")
df_plot <- head(df_plot, -3)


ref_vol <- as.numeric(df_plot$total_reference_volume)
detected_vol <- as.numeric(df_plot$total_detected_volume)
detectable_vol <- as.numeric(df_plot$total_detectable_volume)

rmse_detected <- rmse(ref_vol, detected_vol)
rmse_detectable <- rmse(ref_vol, detectable_vol)
bias_detected <- bias(ref_vol, detected_vol)
bias_detectable <- bias(ref_vol, detectable_vol)

bias_rmse <- data.frame(bias=numeric(0), rmse=numeric(0))

bias_rmse <- rbind(bias_rmse,
                   data.frame(bias=bias_detectable,
                              rmse=rmse_detectable),
                   data.frame(bias=bias_detected,
                              rmse=rmse_detected))

# Calculate bias and RMSE for TOTAL deadwood within inventory radius
# (matched + other)

inventory_volumes <- data.frame(plot=character(0), inventory_volume=numeric(0))

for (plot in unique(df$plot)) {
  
  path_inventory_volumes <- file.path("..", "deadwood",
                                      paste0(plot, "_cl_volumes_inventory.csv"))
  
  volumes <- read.csv(path_inventory_volumes)
  sum_volumes <- sum(as.numeric(volumes$volume), na.rm = TRUE)
  
  new_row <- data.frame(plot = plot, inventory_volume = sum_volumes)
  inventory_volumes <- rbind(inventory_volumes, new_row)
}

rmse_inv <- rmse(actual=ref_vol, predicted=inventory_volumes$inventory_volume)
bias_inv <- bias(actual=ref_vol, predicted=inventory_volumes$inventory_volume)

print(inventory_volumes)
print(rmse_inv)
print(bias_inv)

bias_rmse <- rbind(bias_rmse,
                   data.frame(bias=bias_inv,
                              rmse=rmse_inv))
# Round with 2 digits

bias_rmse <- bias_rmse %>%
  mutate_if(is.numeric, round, digits = 2)

# Save all bias and rmse

colnames(bias_rmse) <- c("bias", "rmse")
rownames(bias_rmse) <- c("detectable", "detected", "estimated_inventory")

print(bias_rmse)
write.csv2(bias_rmse, "bias_rmse.csv")

# Linear regression

df_vol <- read.csv2("element_volumes.csv")

df_vol <- df_vol[-c(22,72),]
df_vol$plot <- as.factor(df_vol$plot)

lm <- lm(detectable_volume~reference_volume - 1, data=df_vol,
         na.action=na.exclude)

summary(lm)

# Tests

res = lm$residuals
raintest(lm) # p-value>0.05: linearity hypothesis verified
Box.test(res, type = "Ljung") # p<0.05: autocorrelation
dwtest(lm) # p<0.05: autocorelation of 1st order
bptest(lm) # p<0.05: homoscedasticity is present
shapiro.test(res)  # < p<0.05: normality hypothesis rejected


df_vol$log_x <- log(df_vol$reference_volume)
df_vol$log_y <- log(df_vol$detectable_volume)


ggplot(data=df_vol, aes(x=reference_volume, y=detectable_volume, col=plot)) +
  # geom_smooth(method = "lm", formula = y~x-1, color = "#69b3a2") +
  geom_point() +
  geom_abline(slope=1, intercept = 0, linetype = "dashed") +
  labs(x = "Reference volume", y = "Detectable volume") +
  annotate("text", x = 0.65, y = 0.55, label = "y=x", fontface = "italic") +
  scale_color_manual(values = viridis_pal()(length(unique(df_vol$plot)))) +
  theme_minimal() +
  theme(
    axis.line = element_line(size = 0.5, linetype = 1)) +
  scale_x_continuous(expand=c(0,0), limits = c(0, 2)) +
  scale_y_continuous(expand=c(0,0), limits = c(0, 0.6))

ggsave(filename = "reg_volume.png",
       height = 1500,
       width = 2000,
       units = "px",
       bg = "white")