#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

// Function to calculate the mean
double computeMean(const std::vector<double>& data) {
    double sum = std::accumulate(data.begin(), data.end(), 0.0); // Sum all elements
    return sum / data.size(); // Divide by total count
}

// Function to calculate the median
double computeMedian(std::vector<double> data) {
    std::sort(data.begin(), data.end()); // Sort the dataset
    size_t size = data.size();
    
    if (size % 2 == 0) {
        return (data[size / 2 - 1] + data[size / 2]) / 2.0; // Average of middle elements
    } else {
        return data[size / 2]; // Middle element
    }
}

// Function to calculate the variance
double computeVariance(const std::vector<double>& data, double mean) {
    double varianceSum = std::accumulate(data.begin(), data.end(), 0.0, 
        [mean](double acc, double value) { return acc + (value - mean) * (value - mean); });
    
    return varianceSum / data.size(); // Divide by total count
}

// Function to calculate the standard deviation
double computeStandardDeviation(double variance) {
    return std::sqrt(variance); // Square root of variance
}

int main() {
    // Example dataset
    std::vector<double> dataset = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Compute statistical values
    double mean = computeMean(dataset);
    double median = computeMedian(dataset);
    double variance = computeVariance(dataset, mean);
    double standardDeviation = computeStandardDeviation(variance);

    // Display results
    std::cout << "Mean: " << mean << std::endl;
    std::cout << "Median: " << median << std::endl;
    std::cout << "Variance: " << variance << std::endl;
    std::cout << "Standard Deviation: " << standardDeviation << std::endl;

    return 0;
}


