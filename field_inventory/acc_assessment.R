library(readr)
library(dplyr)

file <- "field_inventory.csv"

df <- read.csv2(file)
df <- df %>%
  filter(plot != "")

# Calculate correctness

correctness <- df %>%
  group_by(plot) %>%
  summarise(desired_value = (n() - sum(detected == "no")) / n())

print(correctness)