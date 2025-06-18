/*Task 1: Matrix Multiplication Class 
 
 Implement a templated matrix class supporting efficient multiplication, addition, subtraction, and I/O operations.
 Overload relevant operators (+, -, *, <<).
 Clearly demonstrate correctness through comprehensive test cases, including edge cases.
*/

#include <iostream>
#include <vector>

using namespace std;

template<typename T>
class Matrix {
private:
    vector<vector<T>> data;
    int rows, cols;

public:
    // Constructor
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows, vector<T>(cols, 0));
    }

    // Input Matrix
    void input() {
        cout << "Enter elements (" << rows << "x" << cols << "):\n";
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                cin >> data[i][j];
    }

    // Output Matrix
    void print() const {
        for (const auto& row : data) {
            for (const auto& val : row)
                cout << val << " ";
            cout << "\n";
        }
    }

    // Operator Overloads
    Matrix operator+(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }

    Matrix operator-(const Matrix& other) const {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }

    Matrix operator*(const Matrix& other) const {
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < other.cols; ++j)
                for (int k = 0; k < cols; ++k)
                    result.data[i][j] += data[i][k] * other.data[k][j];
        return result;
    }

    // For printing using <<
    friend ostream& operator<<(ostream& os, const Matrix& m) {
        for (int i = 0; i < m.rows; ++i) {
            for (int j = 0; j < m.cols; ++j)
                os << m.data[i][j] << " ";
            os << "\n";
        }
        return os;
    }
};

// ---------- Main Function ----------
int main() {
    int r1, c1, r2, c2;

    cout << "Enter rows and columns of Matrix A: ";
    cin >> r1 >> c1;
    Matrix<int> A(r1, c1);
    A.input();

    cout << "Enter rows and columns of Matrix B: ";
    cin >> r2 >> c2;
    Matrix<int> B(r2, c2);
    B.input();

    if (r1 == r2 && c1 == c2) {
        cout << "\nA + B =\n" << (A + B);
        cout << "\nA - B =\n" << (A - B);
    } else {
        cout << "\nAddition/Subtraction not possible (size mismatch).\n";
    }

    if (c1 == r2) {
        cout << "\nA * B =\n" << (A * B);
    } else {
        cout << "\nMultiplication not possible (A's cols != B's rows).\n";
    }

    return 0;
}


//-------output------
Enter rows and columns of Matrix A: 2
2
Enter elements (2x2):
1
2
3
4
Enter rows and columns of Matrix B: 2
2
Enter elements (2x2):
55
7
8
9

A + B =
56 9 
11 13 

A - B =
-54 -5 
-5 -5 

A * B =
71 25 
197 57 

