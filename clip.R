library(lidR)


# Loading spheres coordinates
spheres_file <- 'spheres_coordinates.csv'
spheres_data <- read.csv2(spheres_file)


clip_las <- function(las_file, radius) {
  #'Create a circular subset of a given radius around the centre with known
  #'coordinates within a point cloud.

  # Reference = filename without extension
  filename <- tools::file_path_sans_ext(basename(las_file))

  # Retrieving coordinates of the centre sphere
  line <- spheres_data[spheres_data$reference == filename, ]
  centre_x <- as.numeric(line$Xcentre)
  centre_y <- as.numeric(line$Ycentre)
  
  las_data <- readLAS(las_file)
  
  # Clipping
  plot <- clip_circle(las_data, x = centre_x, y = centre_y, r = radius)
  
  # Out file path
  out <- paste("scans/", filename, "_clip.las", sep = "")
  
  # Saving out file
  writeLAS(plot, out)
}


# Listing las files
las_files <- list.files(path = "scans/", pattern = "\\.las$", full.names = TRUE)

# Clipping all las files
for (las_file in las_files) {
  
  # Checking if the file isn't already clipped
  expected_file <- paste(tools::file_path_sans_ext(basename(las_file)), "_clip.las", sep = "")
  
  if (!file.exists(expected_file)) {
    clip_las(las_file, radius = 22)
  }
}