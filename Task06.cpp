/*Task 6: Concurrent Multi-Queue Producer-Consumer with Work-Stealing

  Develop a producer-consumer system featuring multi-queue and efficient work-stealing.
  Implement advanced synchronization mechanisms like:std::mutex ,std::condition_variable and Efficient queue handling
  Provide a benchmark and demonstrate Balanced task distribution Improved throughput
*/

#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>

const int QUEUE_COUNT = 3;         // Number of producer queues
const int TASKS_PER_QUEUE = 5;     // Tasks per producer queue

std::vector<std::queue<int>> task_queues(QUEUE_COUNT);   // Task queues
std::vector<std::mutex> queue_mutexes(QUEUE_COUNT);      // Mutex for each queue
std::condition_variable_any cv;                          // Condition variable for notification
std::atomic<bool> done{false};                           // Flag to stop consumers

// Producer thread function
void producer(int qid) {
    for (int i = 0; i < TASKS_PER_QUEUE; ++i) {
        std::lock_guard<std::mutex> lock(queue_mutexes[qid]);  // Lock specific queue
        task_queues[qid].push(i + qid * 10);                   // Push unique task
        cv.notify_all();                                       // Notify consumers
    }
}

// Consumer thread with work-stealing
void consumer(int id) {
    while (!done) {
        for (int q = 0; q < QUEUE_COUNT; ++q) {                // Try all queues
            std::unique_lock<std::mutex> lock(queue_mutexes[q]);
            if (!task_queues[q].empty()) {
                int task = task_queues[q].front();
                task_queues[q].pop();
                lock.unlock();  // Unlock before processing
                std::cout << "Consumer " << id << " got task " << task << " from queue " << q << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100)); // simulate processing
                break;
            }
        }
    }
}

int main() {
    std::vector<std::thread> producers, consumers;

    for (int i = 0; i < QUEUE_COUNT; ++i)
        producers.emplace_back(producer, i);  // Start producers

    for (int i = 0; i < 3; ++i)
        consumers.emplace_back(consumer, i);  // Start consumers

    for (auto& t : producers) t.join();      // Wait for all producers to finish

    done = true;                             // Signal consumers to stop
    cv.notify_all();

    for (auto& t : consumers) t.join();      // Wait for all consumers to finish

    return 0;
}

//-------output-------

Consumer 0 got task 0 from queue 0

