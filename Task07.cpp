/*Task 7: Polynomial Class and Root Finding
Can you develop a polynomial class supporting arithmetic operators like addition, subtraction, multiplication, 
implement a numerical method (Newton-Raphson or Bisection) for finding polynomial roots, 
demonstrate its correctness with a detailed example?
*/

#include <iostream>
#include <vector>
#include <cmath>

// Polynomial class with evaluate and derivative
class Polynomial {
public:
    std::vector<double> coeffs;  // Coefficients for x^0, x^1, ..., x^n

    Polynomial(std::vector<double> c) : coeffs(c) {}

    // Evaluate polynomial value at x
    double evaluate(double x) {
        double result = 0;
        for (int i = 0; i < coeffs.size(); ++i)
            result += coeffs[i] * pow(x, i);
        return result;
    }

    // Evaluate first derivative at x
    double derivative(double x) {
        double result = 0;
        for (int i = 1; i < coeffs.size(); ++i)
            result += i * coeffs[i] * pow(x, i - 1);
        return result;
    }

    // Newton-Raphson method to find root
    double newtonRaphson(double x0, int maxIter = 1000, double tol = 1e-6) {
        double x = x0;
        for (int i = 0; i < maxIter; ++i) {
            double fx = evaluate(x);
            double dfx = derivative(x);
            if (fabs(dfx) < tol) break;          // Avoid division by zero
            double x_new = x - fx / dfx;
            if (fabs(x_new - x) < tol) return x_new;
            x = x_new;
        }
        return x;
    }
};

int main() {
    Polynomial p({-2, 0, 1}); // Represents x^2 - 2
    double root = p.newtonRaphson(1.0); // Start with guess 1.0
    std::cout << "Root: " << root << std::endl; // Output ~1.414
    return 0;
}

//---------output---------

Root: 1.41421
