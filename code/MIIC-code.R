library(miicsdg)

# Load diabetes dataset
diabetes_data <- read.csv("/mnt/c/Users/diabetes.csv")

# Factors
diabetes_data$StudyWeek <- as.factor(diabetes_data$StudyWeek)
diabetes_data$ShortId <- as.factor(diabetes_data$ShortId)
diabetes_data$IntensityGoalAchieved <- as.factor(diabetes_data$IntensityGoalAchieved)
diabetes_data$Hypoglycemia <- as.factor(diabetes_data$Hypoglycemia)
diabetes_data$Hyperglycemia <- as.factor(diabetes_data$Hyperglycemia)
diabetes_data$IPAQ.SF.Category <- as.factor(diabetes_data$IPAQ.SF.Category)

# Run MIIC-SDG
miicsdg_results <- miicsdg::miicsdg(diabetes_data)


# Verify the conversion
str(diabetes_data$StudyWeek)

# Run the MIIC-SDG algorithm on diabetes data
miicsdg_results <- miicsdg::miicsdg(diabetes_data)

# Extract the 5 output objects
synthetic_diabetes <- miicsdg_results[['synthetic_data']]
adjacency_matrix <- miicsdg_results[['adjacency_matrix_DAG']]
data_types <- miicsdg_results[['data_types']]
edges_server <- miicsdg_results[['edges_miic_server']]
miic_output <- miicsdg_results[['miic']]

# Save the DAG network files
output_folder <- "/mnt/c/Users/diabetes_miic_sdg_output"
miicsdg::writeToFile(miicsdg_results, output_folder)

# Save synthetic dataset
write.csv(synthetic_diabetes, "/mnt/c/Users/synthetic_diabetes_4.csv", row.names = FALSE)
