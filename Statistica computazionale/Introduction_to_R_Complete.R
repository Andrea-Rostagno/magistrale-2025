# =============================================================================
# INTRODUCTION TO R - COMPLETE COURSE SCRIPT
# =============================================================================
# Author: Prof. Gianluca Mastrantonio
# Institution: Politecnico di Torino
# Course: Introduction to R Programming and Statistical Analysis
# 
# This script contains all R commands from the complete R course notebook
# organized by topic with explanations and comments.
# =============================================================================

# =============================================================================
# 1. FIRST STEPS IN R
# =============================================================================

# Welcome to R! This is a comment (lines starting with # are ignored)

# R as a calculator
print("Hello, R!")
2 + 3
10 - 4
6 * 7
15 / 3
2^3  # Exponentiation

# Check R version and basic information
R.version.string
sessionInfo()

# =============================================================================
# 2. R HELP SYSTEM
# =============================================================================

# Getting help in R
help(mean)        # Full help documentation
?mean            # Shortcut for help
??mean           # Search for functions containing "mean"
example(mean)    # Run examples from help file

# Help for operators (need quotes)
help("+")
help("if")

# Apropos: find functions containing a string
apropos("mean")
apropos("plot")

# Finding help for packages
help(package = "stats")

# Search in help files
help.search("linear model")

# =============================================================================
# 3. DATA TYPES AND BASIC OPERATIONS
# =============================================================================

# Numeric data type
x <- 5.7
class(x)
typeof(x)
is.numeric(x)

# Integer data type (needs L suffix or as.integer)
y <- 5L
class(y)
typeof(y)

# Convert between types
as.integer(x)
as.numeric(y)

# Character data type
name <- "Alice"
class(name)
is.character(name)

# Logical data type
is_student <- TRUE
class(is_student)
is.logical(is_student)

# Complex numbers
z <- 3 + 4i
class(z)
is.complex(z)

# Type checking functions
is.numeric(x)
is.character(name)
is.logical(is_student)

# =============================================================================
# 4. VECTORS - THE FOUNDATION OF R
# =============================================================================

# Creating vectors
numbers <- c(1, 2, 3, 4, 5)
names <- c("Alice", "Bob", "Charlie")
logical_vec <- c(TRUE, FALSE, TRUE, FALSE)

# Vector properties
length(numbers)
class(numbers)
str(numbers)

# Sequence generation
seq1 <- 1:10                    # Simple sequence
seq2 <- seq(1, 10, by = 2)     # Custom sequence
seq3 <- seq(0, 1, length.out = 11)  # Specified length

# Repetition
rep(1, 5)                      # Repeat single value
rep(c(1, 2), 3)               # Repeat vector
rep(c(1, 2), each = 3)        # Repeat each element

# Vector indexing
numbers[1]                     # First element
numbers[c(1, 3, 5)]           # Multiple elements
numbers[-1]                    # All except first
numbers[numbers > 3]          # Conditional indexing

# Named vectors
ages <- c("Alice" = 25, "Bob" = 30, "Charlie" = 28)
ages["Alice"]                 # Access by name
names(ages)                   # Get names

# Vector arithmetic (element-wise operations)
v1 <- c(1, 2, 3)
v2 <- c(4, 5, 6)
v1 + v2                       # Addition
v1 * v2                       # Multiplication
v1^2                          # Power

# Vector recycling example
c(1, 2, 3) + c(10, 20)       # Shorter vector recycled

# Useful vector functions
numbers <- c(1, 5, 3, 9, 2)
sum(numbers)                  # Sum
mean(numbers)                 # Mean
median(numbers)               # Median
max(numbers)                  # Maximum
min(numbers)                  # Minimum
sort(numbers)                 # Sort ascending
sort(numbers, decreasing = TRUE)  # Sort descending
length(numbers)               # Length
unique(c(1, 1, 2, 2, 3))     # Unique values

# =============================================================================
# 5. MATRICES AND ARRAYS
# =============================================================================

# Create matrices
matrix1 <- matrix(1:12, nrow = 3, ncol = 4)
matrix2 <- matrix(1:12, nrow = 3, ncol = 4, byrow = TRUE)

# Matrix from vectors
v1 <- c(1, 2, 3)
v2 <- c(4, 5, 6)
matrix3 <- rbind(v1, v2)      # Row bind
matrix4 <- cbind(v1, v2)      # Column bind

# Matrix properties
dim(matrix1)                  # Dimensions
nrow(matrix1)                 # Number of rows
ncol(matrix1)                 # Number of columns
length(matrix1)               # Total elements

# Matrix indexing
matrix1[1, 2]                 # Element at row 1, column 2
matrix1[1, ]                  # First row
matrix1[, 2]                  # Second column
matrix1[1:2, 1:3]            # Submatrix

# Matrix operations
A <- matrix(1:4, nrow = 2)
B <- matrix(5:8, nrow = 2)
A + B                         # Element-wise addition
A * B                         # Element-wise multiplication
A %*% B                       # Matrix multiplication
t(A)                          # Transpose

# Matrix functions
rowSums(A)                    # Row sums
colSums(A)                    # Column sums
rowMeans(A)                   # Row means
colMeans(A)                   # Column means

# =============================================================================
# 6. DATA FRAMES - WORKING WITH DATASETS
# =============================================================================

# Create data frame
students <- data.frame(
  name = c("Alice", "Bob", "Charlie", "Diana"),
  age = c(20, 22, 21, 23),
  grade = c("A", "B", "A", "C"),
  passed = c(TRUE, TRUE, TRUE, FALSE),
  stringsAsFactors = FALSE
)

# Data frame properties
str(students)                 # Structure
summary(students)             # Summary statistics
head(students)                # First few rows
tail(students)                # Last few rows
nrow(students)                # Number of rows
ncol(students)                # Number of columns
names(students)               # Column names

# Indexing data frames
students[1, ]                 # First row
students[, 2]                 # Second column
students$name                 # Column by name
students[["name"]]            # Another way to access columns
students["age"]               # Returns data frame
students[students$age > 21, ] # Conditional indexing

# Adding new columns
students$gpa <- c(3.8, 3.2, 3.9, 2.8)
students$year <- "Sophomore"

# Removing columns
students$year <- NULL

# =============================================================================
# 7. LISTS - COMPLEX DATA STRUCTURES
# =============================================================================

# Create lists (can contain different data types)
my_list <- list(
  numbers = c(1, 2, 3, 4, 5),
  names = c("Alice", "Bob", "Charlie"),
  matrix = matrix(1:6, nrow = 2),
  logical = TRUE
)

# List indexing
my_list[[1]]                  # First element (returns vector)
my_list[1]                    # First element (returns list)
my_list$numbers               # Access by name
my_list[["numbers"]]          # Another way to access by name

# List properties
length(my_list)               # Number of elements
names(my_list)                # Element names
str(my_list)                  # Structure

# Adding to lists
my_list$new_item <- "Hello"
my_list[[5]] <- c(10, 20, 30)

# =============================================================================
# 8. PROBABILITY DISTRIBUTIONS IN R
# =============================================================================

# Normal Distribution
# dnorm: probability density function
dnorm(0)                      # Standard normal at x=0
dnorm(1.96)                   # At x=1.96
dnorm(0, mean=5, sd=2)        # Non-standard normal

# pnorm: cumulative distribution function
pnorm(0)                      # P(Z <= 0) for standard normal
pnorm(1.96)                   # P(Z <= 1.96) ≈ 0.975
pnorm(10, mean=5, sd=2)       # P(X <= 10) for N(5,4)

# qnorm: quantile function (inverse CDF)
qnorm(0.5)                    # 50th percentile (median)
qnorm(0.975)                  # 97.5th percentile ≈ 1.96
qnorm(0.95, mean=5, sd=2)     # 95th percentile of N(5,4)

# rnorm: random number generation
rnorm(5)                      # 5 random numbers from standard normal
rnorm(10, mean=5, sd=2)       # 10 random numbers from N(5,4)

# Binomial Distribution
dbinom(3, size=10, prob=0.3)  # P(X=3) for Binomial(10, 0.3)
pbinom(3, size=10, prob=0.3)  # P(X<=3) for Binomial(10, 0.3)
qbinom(0.5, size=10, prob=0.3) # Median of Binomial(10, 0.3)
rbinom(5, size=10, prob=0.3)  # 5 random samples

# Poisson Distribution
dpois(2, lambda=3)            # P(X=2) for Poisson(3)
ppois(2, lambda=3)            # P(X<=2) for Poisson(3)
qpois(0.5, lambda=3)          # Median of Poisson(3)
rpois(5, lambda=3)            # 5 random samples

# =============================================================================
# 9. FUNCTIONS AND PACKAGE MANAGEMENT
# =============================================================================

# Basic function example
calculate_circle_area <- function(radius) {
  area <- pi * radius^2
  return(area)
}

# Test the function
calculate_circle_area(5)
calculate_circle_area(c(1, 2, 3, 4, 5))  # Works with vectors too

# Function with default arguments
greet_person <- function(name, greeting = "Hello", punctuation = "!") {
  message <- paste(greeting, name, punctuation, sep = "")
  return(message)
}

# Test with different argument combinations
greet_person("Alice")
greet_person("Bob", "Hi")
greet_person("Charlie", "Good morning", ".")
greet_person("Diana", punctuation = "!!!")  # Named argument

# Function returning multiple values (as a list)
calculate_statistics <- function(numbers) {
  n <- length(numbers)
  mean_val <- mean(numbers)
  median_val <- median(numbers)
  sd_val <- sd(numbers)
  
  # Return multiple values as a named list
  return(list(
    count = n,
    mean = mean_val,
    median = median_val,
    std_dev = sd_val,
    summary = summary(numbers)
  ))
}

# Test the function
test_data <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
stats_result <- calculate_statistics(test_data)
print(stats_result)

# Access individual components
cat("Mean:", stats_result$mean, "\n")
cat("Standard deviation:", stats_result$std_dev, "\n")

# Function with variable arguments
print_all <- function(...) {
  args <- list(...)
  cat("Number of arguments:", length(args), "\n")
  for (i in 1:length(args)) {
    cat("Argument", i, ":", args[[i]], "\n")
  }
}

print_all("hello", 42, TRUE, "world")

# Package management functions
is_package_installed <- function(package_name) {
  package_name %in% rownames(installed.packages())
}

# Check if a package is installed
cat("Is ggplot2 installed?", is_package_installed("ggplot2"), "\n")

# Safe library loading
safe_library <- function(package_name) {
  if (is_package_installed(package_name)) {
    library(package_name, character.only = TRUE)
    cat("Package", package_name, "loaded successfully.\n")
    return(TRUE)
  } else {
    cat("Package", package_name, "is not installed.\n")
    return(FALSE)
  }
}

# Get information about installed packages
installed_pkgs <- installed.packages()[, c("Package", "Version", "Priority")]
head(installed_pkgs, 10)  # Show first 10 packages

# Count installed packages
cat("Total installed packages:", nrow(installed.packages()), "\n")

# =============================================================================
# 10. DATA VISUALIZATION
# =============================================================================

# Generate sample data for plotting
set.seed(123)  # For reproducible results
x <- 1:20
y <- x + rnorm(20, 0, 3)
categories <- factor(sample(c("A", "B", "C"), 20, replace = TRUE))

# Basic scatter plot
plot(x, y)
plot(x, y, main = "Scatter Plot", xlab = "X values", ylab = "Y values")

# Customized scatter plot
plot(x, y, 
     main = "Enhanced Scatter Plot",
     xlab = "X values", 
     ylab = "Y values",
     col = "blue", 
     pch = 16,
     cex = 1.2)

# Add a line of best fit
abline(lm(y ~ x), col = "red", lwd = 2)

# Basic histogram
data <- rnorm(1000, mean = 50, sd = 10)
hist(data)
hist(data, 
     main = "Histogram of Normal Data",
     xlab = "Values",
     ylab = "Frequency",
     col = "lightblue",
     breaks = 20)

# Basic boxplot
group1 <- rnorm(100, mean = 50, sd = 10)
group2 <- rnorm(100, mean = 55, sd = 12)
group3 <- rnorm(100, mean = 45, sd = 8)
boxplot(group1, group2, group3,
        names = c("Group 1", "Group 2", "Group 3"),
        main = "Boxplot Comparison",
        ylab = "Values",
        col = c("red", "green", "blue"))

# Basic bar plot
counts <- table(categories)
barplot(counts,
        main = "Bar Plot of Categories",
        xlab = "Categories",
        ylab = "Frequency",
        col = rainbow(length(counts)))

# Line plot
time <- 1:12
temperature <- c(5, 7, 12, 18, 23, 28, 31, 29, 24, 17, 10, 6)
plot(time, temperature, 
     type = "l",
     main = "Temperature Over Time",
     xlab = "Month",
     ylab = "Temperature (°C)",
     col = "red",
     lwd = 2)

# Add points to the line plot
points(time, temperature, col = "blue", pch = 16)

# Multiple plots in one figure
par(mfrow = c(2, 2))  # 2x2 grid

# Plot 1: Scatter plot
plot(x, y, main = "Scatter Plot")

# Plot 2: Histogram
hist(data, main = "Histogram")

# Plot 3: Boxplot
boxplot(group1, group2, group3, main = "Boxplot")

# Plot 4: Bar plot
barplot(counts, main = "Bar Plot")

# Reset to single plot
par(mfrow = c(1, 1))

# =============================================================================
# 11. ADVANCED PLOTTING TECHNIQUES
# =============================================================================

# Create more complex visualization
# Correlation plot
x1 <- rnorm(50)
y1 <- x1 + rnorm(50, 0, 0.5)
plot(x1, y1,
     main = "Correlation Example",
     xlab = "X variable",
     ylab = "Y variable",
     col = "darkblue",
     pch = 19)

# Add correlation coefficient to plot
correlation <- cor(x1, y1)
text(max(x1) * 0.7, max(y1) * 0.9, 
     paste("r =", round(correlation, 3)),
     col = "red", cex = 1.2)

# Density plot
plot(density(data),
     main = "Density Plot",
     xlab = "Values",
     ylab = "Density",
     col = "purple",
     lwd = 2)

# Add normal curve for comparison
curve(dnorm(x, mean = mean(data), sd = sd(data)),
      add = TRUE, col = "red", lwd = 2, lty = 2)

# Legend
legend("topright", 
       c("Sample Data", "Normal Curve"),
       col = c("purple", "red"),
       lwd = 2,
       lty = c(1, 2))

# =============================================================================
# 12. BASIC STATISTICAL OPERATIONS
# =============================================================================

# Summary statistics
data_sample <- rnorm(100, mean = 50, sd = 10)

# Central tendency
mean(data_sample)
median(data_sample)

# Variability
var(data_sample)              # Variance
sd(data_sample)               # Standard deviation
range(data_sample)            # Range
IQR(data_sample)              # Interquartile range

# Quantiles
quantile(data_sample)         # Default quantiles
quantile(data_sample, probs = c(0.1, 0.9))  # Custom quantiles

# Complete summary
summary(data_sample)

# Correlation and covariance
x_var <- rnorm(50)
y_var <- 2 * x_var + rnorm(50)
cor(x_var, y_var)             # Correlation
cov(x_var, y_var)             # Covariance

# Simple linear regression
model <- lm(y_var ~ x_var)
summary(model)

# Basic hypothesis testing
# One-sample t-test
t.test(data_sample, mu = 50)

# Two-sample t-test
group_a <- rnorm(30, mean = 50, sd = 10)
group_b <- rnorm(30, mean = 55, sd = 10)
t.test(group_a, group_b)

# =============================================================================
# 13. WORKSPACE MANAGEMENT
# =============================================================================

# List objects in workspace
ls()

# Remove specific objects
# rm(object_name)

# Remove all objects (use with caution!)
# rm(list = ls())

# Get working directory
getwd()

# Set working directory (example - don't run unless needed)
# setwd("/path/to/your/directory")

# Save workspace
# save.image("my_workspace.RData")

# Load workspace
# load("my_workspace.RData")

# Save specific objects
# save(data_sample, students, file = "my_data.RData")

# =============================================================================
# END OF SCRIPT
# =============================================================================

# This script provides a comprehensive introduction to R programming
# covering all major topics from the complete R course notebook.
# 
# To use this script:
# 1. Run sections individually to understand each concept
# 2. Modify examples to experiment with different parameters
# 3. Add your own data and analyses
# 4. Use as a reference for R syntax and functions
#
# For more advanced topics, consider exploring:
# - Data manipulation with dplyr
# - Advanced graphics with ggplot2
# - Statistical modeling and machine learning
# - Time series analysis
# - Spatial data analysis
