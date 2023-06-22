library(lidR)


# Loading spheres coordinates
spheres_file <- 'spheres_coordinates.csv'
spheres_data <- read.csv2(spheres_file)


# Function which creates a circular subset of a (las) point cloud of a given
# radius
clip_las <- function(las_file, radius) {

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
  out <- paste(filename, "_clip.las")
  
  # Saving out file
  writeLAS(plot, out)
}


# Listing all las files
las_files <- list.files(pattern = "\\.las$", full.names = TRUE)

# Cliping las files
for (las_file in las_files) {
  
  # Checking if the file isn't already clipped
  expected_file <- paste(tools::file_path_sans_ext(basename(las_file)),
                         "_clip.las")
  
  if (!file.exists(expected_file)) {
    clip_las(las_file, radius = 22)
  }
}