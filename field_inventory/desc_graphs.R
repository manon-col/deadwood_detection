library(ggplot2)
library(scales)
library(ggpubr)
library(dplyr)


# Load data
data <- read.csv2("field_inventory.csv", dec=",")

# Species (unused)
plot_species <- ggplot(data, aes(x=species)) +
  geom_bar(fill="#69b3a2", alpha=0.9, show.legend=TRUE) +
  labs(x = "Species", y = "Number of elements") +
  scale_y_continuous(expand = c(0,0,0,2),
                     breaks = scales::pretty_breaks(n = 10)) +
  theme_minimal() +
  theme(axis.line = element_line(size = 0.5, linetype=1),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.text.x = element_text(hjust=1, vjust=1, angle=45)
  )

# Saproxylation stage, barplot (unused)
plot_saprox <- ggplot(data, aes(x=stage_saproxylation)) +
  geom_bar(fill="#69b3a2", alpha=0.9, show.legend=FALSE) +
  labs(x = "Saproxylation stage", y = "Number of elements") +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 6)) +
  scale_y_continuous(expand = c(0,0,0,2),
                     breaks = scales::pretty_breaks(n = 10)) +
  theme_minimal() +
  theme(axis.line = element_line(size = 0.5, linetype=1),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank()
        )

# Convert the variables to factors
data$species <- factor(data$species)
data$stage_saproxylation <- factor(data$stage_saproxylation)

# Species and saproxylation stage, stacked barplot
plot_spec_saprox <- ggplot(data, aes(x = species, fill = stage_saproxylation)) +
  geom_bar(position = "stack", alpha = 0.9) +
  labs(x = "Species", y = "Number of elements", ) +
  scale_y_continuous(expand = c(0, 0, 0, 2),
                     breaks = scales::pretty_breaks(n = 10)) +
  scale_fill_viridis_d("Saproxylation
stage") +
  theme_minimal() +
  theme(
    axis.line = element_line(size = 0.5, linetype = 1),
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    axis.text.x = element_text(hjust = 1, vjust = 1, angle = 45,
                               face = ifelse(
                                 levels(data$species)!='unidentified',
                                 "italic", "plain"))
  )

# Volume, histogram (unused)
plot_hist_vol <- ggplot(data, aes(x=total_volume)) +
  geom_histogram(bins=100, fill="#69b3a2",
                 alpha=0.9, show.legend=FALSE) +
  labs(x = "Volume (m³)", y = "Number of elements") +
  scale_x_continuous(breaks = scales::pretty_breaks(n = 20)) +
  scale_y_continuous(expand = c(0,0,0,2),
                     breaks = scales::pretty_breaks(n = 10)) +
  theme_minimal() +
  theme(axis.line = element_line(size = 0.5, linetype=1),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank()
  )

# Volume, boxplot
plot_box_vol <- ggplot(data, aes(species, total_volume)) +
  geom_boxplot(aes(col=species), show.legend=FALSE) +
  labs(x = "Species", y = "log10(volume)") +
  scale_y_continuous(trans = 'log10',
                     breaks = trans_breaks("log10", function(x) 10^x,
                                           n = 6, only.loose = TRUE),
                     labels = label_number(scale = 10 ^ (1 / 3))) +
  scale_color_viridis_d() +
  theme_minimal() +
  theme(axis.line = element_line(size = 0.5, linetype=1),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.text.x = element_text(hjust=1, vjust=1, angle=45,
                                   face = ifelse(
                                     levels(data$species)!='unidentified',
                                     "italic", "plain"))
  )

# Mid-length circumference, boxplot
plot_box_circ <- ggplot(data, aes(species, mean_circ)) +
  geom_boxplot(aes(col=species), show.legend=FALSE) +
  labs(x = "Species", y = "Mid-length circumference") +
  theme_minimal() +
  theme(axis.line = element_line(size = 0.5, linetype=1),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        axis.text.x = element_text(hjust=1, vjust=1, angle=45)
  ) +
  scale_y_continuous(expand = c(0,0,0,1),
                     breaks = scales::pretty_breaks(n = 2)) +
  scale_color_viridis_d()

# Arrange 2 graphs together
des <- ggarrange(
  plot_spec_saprox,
  plot_box_vol,
  nrow = 2
)

# Save plot
ggsave(plot = des,
       filename = "data_description.png",
       height = 2000,
       width = 1400,
       units = "px",
       bg = "white")