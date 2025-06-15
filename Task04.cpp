#include <iostream>
#include <vector>
#include <future>
#include <algorithm>

// Threshold below which sorting will be sequential
const size_t PARALLEL_THRESHOLD = 1000;

// Merge two sorted halves
template<typename T>
void merge(std::vector<T>& data, int left, int mid, int right) {
    std::vector<T> temp(right - left + 1);
    int i = left, j = mid + 1, k = 0;

    while (i <= mid && j <= right) {
        temp[k++] = (data[i] < data[j]) ? data[i++] : data[j++];
    }

    while (i <= mid) temp[k++] = data[i++];
    while (j <= right) temp[k++] = data[j++];

    for (int m = 0; m < temp.size(); ++m)
        data[left + m] = temp[m];
}

// Concurrent merge sort
template<typename T>
void merge_sort(std::vector<T>& data, int left, int right) {
    if (left >= right) return;

    int mid = (left + right) / 2;

    if ((right - left) > PARALLEL_THRESHOLD) {
        // Parallel sort
        auto left_future = std::async(std::launch::async, merge_sort<T>, std::ref(data), left, mid);
        merge_sort(data, mid + 1, right);
        left_future.get();
    } else {
        // Sequential sort
        merge_sort(data, left, mid);
        merge_sort(data, mid + 1, right);
    }

    merge(data, left, mid, right);
}

int main() {
    std::vector<int> data = {8, 4, 5, 3, 7, 2, 6, 1};

    merge_sort(data, 0, data.size() - 1);

    std::cout << "Sorted: ";
    for (auto val : data) std::cout << val << " ";
    return 0;
}
