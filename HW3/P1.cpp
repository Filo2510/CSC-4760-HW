#include <mpi.h>
#include <iostream>
#include <cstdlib>
using namespace std;

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 3) {
        if (world_rank == 0) cerr << "Usage: " << argv[0] << " <P> <Q>\n";
        MPI_Finalize(); return 1;
    }

    int P = atoi(argv[1]), Q = atoi(argv[2]);

    if (P < 1 || Q < 1 || P * Q != world_size) {
        if (world_rank == 0)
            cerr << "Error: P*Q must equal world size (" << world_size << ").\n";
        MPI_Finalize(); return 1;
    }

    int color1 = world_rank / Q;
    int key1   = world_rank % Q;   // determines rank order within group

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color1, key1, &row_comm);

    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);

    int rank_sum = 0;
    MPI_Reduce(&world_rank, &rank_sum, 1, MPI_INT, MPI_SUM, 0, row_comm);

    if (row_rank == 0)
        cout << "[Split1] Row " << color1 << ": sum of world ranks = " << rank_sum << "\n";

    MPI_Barrier(MPI_COMM_WORLD);

    int color2 = world_rank % Q;
    int key2   = world_rank / Q;

    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color2, key2, &col_comm);

    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);

    // Root holds its own world rank
    int bcast_val = (col_rank == 0) ? world_rank : -1;
    MPI_Bcast(&bcast_val, 1, MPI_INT, 0, col_comm);

    cout << "[Split2] World rank " << world_rank
         << " (col=" << color2 << ", local=" << col_rank
         << ") received bcast value = " << bcast_val << "\n";

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
