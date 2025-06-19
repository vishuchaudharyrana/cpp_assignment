/*Task 8: Numerical Integration
Can you implement the Trapezoidal Rule and Simpson’s Rule for numerical integration, 
clearly demonstrate their working by comparing the numerical results with the analytical (exact) solution of an integral?
*/

#include <iostream>
#include <cmath>

// Function to integrate
double f(double x) {
    return sin(x);  // Change to any other function
}

// Trapezoidal rule implementation
double trapezoidal(double a, double b, int n) {
    double h = (b - a) / n;
    double result = (f(a) + f(b)) / 2.0;
    for (int i = 1; i < n; ++i)
        result += f(a + i * h);
    return result * h;
}

// Simpson's rule implementation
double simpson(double a, double b, int n) {
    if (n % 2 != 0) ++n; // Make n even
    double h = (b - a) / n;
    double result = f(a) + f(b);
    for (int i = 1; i < n; i++)
        result += (i % 2 == 0 ? 2 : 4) * f(a + i * h);
    return result * h / 3.0;
}

int main() {
    std::cout << "Trapezoidal: " << trapezoidal(0, M_PI, 100) << std::endl;
    std::cout << "Simpson: " << simpson(0, M_PI, 100) << std::endl;
    return 0;
}

//------output-------
Trapezoidal: 1.99984
Simpson: 2
