#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>

std::vector<int> merge_vectors(const std::vector<int>& left, const std::vector<int>& right) {
    std::vector<int> result;
    result.reserve(left.size() + right.size());
    std::merge(left.begin(), left.end(), right.begin(), right.end(), std::back_inserter(result));
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int total_n = 16;
    int local_n = total_n / world_size;

    std::vector<int> global_data;
    std::vector<int> local_data(local_n);

    if (rank == 0) {
        global_data.resize(total_n);
        for (int i = 0; i < total_n; ++i) {
            global_data[i] = rand() % 100;
        }
        std::cout << "Original Array: ";
        for (int val : global_data) std::cout << val << " ";
        std::cout << std::endl;
    }

    // Scatter chunks to processes
    MPI_Scatter(global_data.data(), local_n, MPI_INT,
                local_data.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);

    std::sort(local_data.begin(), local_data.end());

    // Binary tree reduction
    int step = 1;
    while (step < world_size) {
        if (rank % (2 * step) == 0) {
            if (rank + step < world_size) {
                int recv_size = local_n * step;
                std::vector<int> neighbor_data(recv_size);
                MPI_Recv(neighbor_data.data(), recv_size, MPI_INT, rank + step, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_data = merge_vectors(local_data, neighbor_data);
            }
        } else {
            int target = rank - step;
            MPI_Send(local_data.data(), (int)local_data.size(), MPI_INT, target, 0, MPI_COMM_WORLD);
            break;
        }
        step *= 2;
    }

    // Prints the final sorted array
    if (rank == 0) {
        std::cout << "Sorted Array:   ";
        for (int val : local_data) std::cout << val << " ";
        std::cout << std::endl;
    }

    MPI_Finalize();
    return 0;
}