#include <iostream>
#include <vector>
#include <cmath>

// Templated 3D vector class
template <typename T>
class Vector3 {
public:
    T x, y, z;

    Vector3(T x=0, T y=0, T z=0) : x(x), y(y), z(z) {}

    // Vector addition
    Vector3 operator+(const Vector3& v) const {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }

    // Scalar multiplication
    Vector3 operator*(T scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }

    // Dot product
    T dot(const Vector3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    // Cross product
    Vector3 cross(const Vector3& v) const {
        return Vector3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    // Normalize the vector
    Vector3 normalize() const {
        double mag = sqrt(dot(*this));
        return Vector3(x / mag, y / mag, z / mag);
    }

    // Print vector
    void print() const {
        std::cout << "(" << x << ", " << y << ", " << z << ")\n";
    }
};

int main() {
    Vector3<double> v1(1, 2, 3), v2(4, 5, 6);
    Vector3<double> sum = v1 + v2;
    Vector3<double> cross = v1.cross(v2);

    std::cout << "Dot product: " << v1.dot(v2) << std::endl;
    std::cout << "Cross product: "; cross.print();
    std::cout << "Normalized v1: "; v1.normalize().print();

    return 0;
}

//-----output------
Dot product: 32
Cross product: (-3, 6, -3)
Normalized v1: (0.267261, 0.534522, 0.801784)

