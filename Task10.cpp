#include <iostream>
#include <random>

// Function to estimate pi using Monte Carlo method
double estimate_pi(int samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int inside_circle = 0;
    for (int i = 0; i < samples; ++i) {
        double x = dis(gen);
        double y = dis(gen);
        if (x*x + y*y <= 1.0) inside_circle++;  // Inside unit circle
    }

    return (4.0 * inside_circle) / samples;
}

int main() {
    int n = 1000000; // Number of samples
    std::cout << "Estimated Pi: " << estimate_pi(n) << std::endl;
    return 0;
}

//-----output-----
Estimated Pi: 3.14101
