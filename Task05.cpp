#include <iostream>
#include <atomic>
#include <thread>

template <typename T>
class AtomicSharedPtr {
private:
    T* ptr;
    std::atomic<int>* ref_count;

public:
    // Constructor
    explicit AtomicSharedPtr(T* p = nullptr) : ptr(p), ref_count(new std::atomic<int>(1)) {}

    // Copy constructor
    AtomicSharedPtr(const AtomicSharedPtr& other) {
        ptr = other.ptr;
        ref_count = other.ref_count;
        ref_count->fetch_add(1);
    }

    // Destructor
    ~AtomicSharedPtr() {
        if (ref_count->fetch_sub(1) == 1) {
            delete ptr;
            delete ref_count;
        }
    }

    // Assignment operator
    AtomicSharedPtr& operator=(const AtomicSharedPtr& other) {
        if (this != &other) {
            if (ref_count->fetch_sub(1) == 1) {
                delete ptr;
                delete ref_count;
            }
            ptr = other.ptr;
            ref_count = other.ref_count;
            ref_count->fetch_add(1);
        }
        return *this;
    }

    T* get() const { return ptr; }
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
};

// Test the smart pointer
void test_atomic_ptr() {
    AtomicSharedPtr<int> p1(new int(42));
    {
        AtomicSharedPtr<int> p2 = p1;
        std::cout << "Value: " << *p2 << std::endl;
    } // p2 destroyed
    std::cout << "Still accessible: " << *p1 << std::endl;
}

int main() {
    std::thread t1(test_atomic_ptr);
    std::thread t2(test_atomic_ptr);
    t1.join();
    t2.join();
    return 0;
}
 
//------output------

Value: Value: 4242
Still accessible: 42

Still accessible: 42

