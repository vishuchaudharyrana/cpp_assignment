#include <iostream>
#include <stack>
#include <string>
#include <cctype>
using namespace std;

// Function to return precedence of operators
int precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/') return 2;
    return 0;
}

// Function to apply an operator to two numbers
int applyOp(int a, int b, char op) {
    switch (op) {
        case '+': return a + b;
        case '-': return a - b;
        case '*': return a * b;
        case '/': return (b != 0) ? a / b : 0; // avoid division by zero
    }
    return 0;
}

// Main evaluator function
int evaluate(const string& expr) {
    stack<int> values;     // stack to store numbers
    stack<char> ops;       // stack to store operators
    int i = 0;

    while (i < expr.length()) {
        // Skip whitespace
        if (isspace(expr[i])) {
            i++;
            continue;
        }

        // If current character is a digit, parse the number
        if (isdigit(expr[i])) {
            int val = 0;
            while (i < expr.length() && isdigit(expr[i]))
                val = val * 10 + (expr[i++] - '0');
            values.push(val);
        }

        // If opening parenthesis, push to ops stack
        else if (expr[i] == '(') {
            ops.push(expr[i]);
            i++;
        }

        // If closing parenthesis, solve entire parenthesis expression
        else if (expr[i] == ')') {
            while (!ops.empty() && ops.top() != '(') {
                int b = values.top(); values.pop();
                int a = values.top(); values.pop();
                char op = ops.top(); ops.pop();
                values.push(applyOp(a, b, op));
            }
            ops.pop(); // remove '('
            i++;
        }

        // If operator
        else {
            // Process all higher or equal precedence operators
            while (!ops.empty() && precedence(ops.top()) >= precedence(expr[i])) {
                int b = values.top(); values.pop();
                int a = values.top(); values.pop();
                char op = ops.top(); ops.pop();
                values.push(applyOp(a, b, op));
            }
            ops.push(expr[i]);
            i++;
        }
    }

    // Process any remaining operators
    while (!ops.empty()) {
        int b = values.top(); values.pop();
        int a = values.top(); values.pop();
        char op = ops.top(); ops.pop();
        values.push(applyOp(a, b, op));
    }

    return values.top(); // final result
}

// ----------- Main Function with Test Cases -------------
int main() {
    string expressions[] = {
        "3 + 5",                      // Simple addition
        "10 + 2 * 6",                 // Operator precedence
        "100 * 2 + 12",               // Mix of * and +
        "100 * (2 + 12)",             // Parentheses
        "100 * (2 + 12) / 14",        // Full expression
        "(3 + 5) * (2 + (4 - 1))"     // Nested parentheses
    };

    // Evaluate and print result of each expression
    for (const auto& expr : expressions) {
        cout << "Expression: " << expr << "\nResult: " << evaluate(expr) << "\n\n";
    }

    return 0;
}
