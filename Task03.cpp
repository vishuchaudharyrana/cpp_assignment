#include <iostream>
#include <list>
using namespace std;

// Function to demonstrate basic STL list operations
void demoListOperations() {
    // Create a list with initial values
    list<int> myList = {10, 20, 30};
    cout << "Initial list: ";
    for (int val : myList) cout << val << " ";

    // Add elements to the end and beginning
    myList.push_back(40);     // adds 40 at the end
    myList.push_front(5);     // adds 5 at the beginning
    cout << "\nAfter push_back(40) and push_front(5): ";
    for (int val : myList) cout << val << " ";

    // Remove elements from the end and beginning
    myList.pop_back();        // removes 40
    myList.pop_front();       // removes 5
    cout << "\nAfter pop_back() and pop_front(): ";
    for (int val : myList) cout << val << " ";

    // Insert 15 at 2nd position (after first element)
    auto it = next(myList.begin(), 1);
    myList.insert(it, 15);
    cout << "\nAfter insert(15) at 2nd position: ";
    for (int val : myList) cout << val << " ";

    // Remove all elements with value 20
    myList.remove(20);
    cout << "\nAfter remove(20): ";
    for (int val : myList) cout << val << " ";

    // Print list in reverse order
    cout << "\nList in reverse: ";
    for (auto rit = myList.rbegin(); rit != myList.rend(); ++rit)
        cout << *rit << " ";

    cout << endl;
}

// Main function
int main() {
    demoListOperations();
    return 0;
}
