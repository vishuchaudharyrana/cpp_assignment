#  C++ Assignment Tasks Overview

This repository includes 11 C++ programs designed to build core programming concepts and modern C++ (C++17) practices including data structures, algorithms, concurrency, smart pointers, and numerical methods.

Each file is self-contained and prints results to the console. These programs are educational and demonstrate real-world problem-solving patterns.

---

## 🔧 How to Compile & Run

Open terminal or command prompt, then compile each file as:

```bash
g++ TaskXX.cpp -std=c++17 -pthread -o TaskXX
./TaskXX

🔹 Task01.cpp — Matrix Operations
Concepts Covered:

Templates

Operator overloading

Matrix arithmetic

What it does:
Implements a generic Matrix<T> class that supports:

Input/output via >>/<<

Matrix addition, subtraction, multiplication

Checks for dimension compatibility

Learning Outcome:
You learn how to use templates to generalize matrix operations for different data types (int, float, etc.).

🔹 Task02.cpp — Infix Expression Evaluator
Concepts Covered:

Stack data structure

Operator precedence

Expression parsing

What it does:
Evaluates infix expressions like 100 * (2 + 12) / 14 using:

Operator and operand stacks

Precedence rules

Parentheses handling

Learning Outcome:
Builds understanding of expression parsing, precedence, and evaluation logic using standard C++ libraries.

🔹 Task03.cpp — STL List Operations
Concepts Covered:

std::list usage

Iterators and reverse traversal

What it does:
Demonstrates list operations:

push_front, push_back

pop_front, pop_back

insert, remove, reverse print

Learning Outcome:
Explains how STL containers simplify linked list operations and how iterators work.

🔹 Task04.cpp — Parallel Merge Sort
Concepts Covered:

Multithreading using std::async

Merge sort algorithm

What it does:
Performs merge sort on a vector of integers:

Sorts in parallel for large chunks

Falls back to sequential for small ones

Learning Outcome:
Introduces parallel algorithms, thread-futures, and efficient divide-and-conquer strategy.

🔹 Task05.cpp — Atomic Shared Pointer
Concepts Covered:

Smart pointers

Thread safety

Reference counting

What it does:
Creates a custom smart pointer (AtomicSharedPtr) with:

Reference counting via std::atomic

Thread-safe memory management

Learning Outcome:
Teaches how shared resources are managed safely in multi-threaded environments.

🔹 Task06.cpp — Work-Stealing Thread Pool
Concepts Covered:

Producer-consumer problem

Mutexes, condition variables

Work-stealing algorithm

What it does:
Simulates:

3 producer queues with tasks

3 consumers that steal work from any queue

Learning Outcome:
Illustrates advanced concurrency models and coordination among threads.

🔹 Task07.cpp — Polynomial Root Finder
Concepts Covered:

Newton-Raphson method

Polynomial evaluation

Numerical methods

What it does:
Defines a Polynomial class with:

evaluate() and derivative()

Newton-Raphson implementation to find root

Learning Outcome:
Explains how to solve nonlinear equations numerically and implement math logic in code.

🔹 Task08.cpp — Trapezoidal and Simpson’s Rule
Concepts Covered:

Numerical integration

Function approximation

What it does:
Estimates area under the curve (∫sin(x)dx) using:

Trapezoidal rule

Simpson's rule

Learning Outcome:
Shows how integration can be done algorithmically using finite sums.

🔹 Task09.cpp — 3D Vector Class
Concepts Covered:

Templated classes

Vector operations (math)

What it does:
Implements Vector3<T> with:

Addition, scalar multiplication

Dot product, cross product

Normalization

Learning Outcome:
Builds geometric algebra foundation and shows how vector operations are implemented from scratch.

🔹 Task10.cpp — Monte Carlo Simulation for Pi
Concepts Covered:

Random number generation

Monte Carlo estimation

Unit circle sampling

What it does:
Estimates the value of π by:

Randomly sampling points in a square

Counting how many fall inside the unit circle

Learning Outcome:
Demonstrates how probability can be used for estimation using statistical sampling.

Task15.cpp
This C++ program performs basic statistical analysis on a dataset using the Standard Template Library (<vector>, <algorithm>, <numeric>, and <cmath>). It includes the implementation of functions to calculate:

Mean: The average value of the dataset.

Median: The middle value when the dataset is sorted.

Variance: A measure of data spread around the mean.

Standard Deviation: The square root of the variance, representing data dispersion.

Example Dataset:
cpp
Copy
Edit
std::vector<double> dataset = {1.0, 2.0, 3.0, 4.0, 5.0};
Sample Output:
yaml
Copy
Edit
Mean: 3  
Median: 3  
Variance: 2  
Standard Deviation: 1.41421
This is a simple and modular program useful for statistical computation and data analysis demonstrations in C++.
