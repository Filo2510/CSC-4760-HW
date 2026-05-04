#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

int lsize(int M, int N, int r)  { return M/N + (r < M%N ? 1 : 0); }
int lstart(int M, int N, int r) { return r*(M/N) + (r < M%N ? r : M%N); }

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int wrank, wsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    MPI_Comm_size(MPI_COMM_WORLD, &wsize);
    if (argc != 4) {
        if (wrank == 0) cerr << "Usage: " << argv[0] << " <P> <Q> <M>\n";
        MPI_Finalize(); return 1;
    }
    int P = atoi(argv[1]), Q = atoi(argv[2]), M = atoi(argv[3]);
    if (P < 1 || Q < 1 || M < 1 || P*Q != wsize) {
        if (wrank == 0) cerr << "Error: need P*Q == world_size, P,Q,M >= 1\n";
        MPI_Finalize(); return 1;
    }
    int row = wrank/Q, col = wrank%Q;
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    int xs = lsize(M, P, row), xst = lstart(M, P, row);
    vector<double> x_local(xs, 0.0);
    if (col == 0) {
        vector<int> sc(P), sd(P);
        for (int r = 0; r < P; r++) { sc[r] = lsize(M,P,r); sd[r] = lstart(M,P,r); }
        vector<double> x_full(wrank == 0 ? M : 0);
        if (wrank == 0) for (int i = 0; i < M; i++) x_full[i] = (double)i;
        MPI_Scatterv(x_full.data(), sc.data(), sd.data(), MPI_DOUBLE,
                     x_local.data(), xs, MPI_DOUBLE, 0, col_comm);
    }
    // Broadcast x_local across each row
    MPI_Bcast(x_local.data(), xs, MPI_DOUBLE, 0, row_comm);

    int ys = lsize(M, Q, col), yst = lstart(M, Q, col);
    vector<double> y_local(ys, 0.0);
    if (row == 0) {
        vector<int> sc(Q), sd(Q);
        for (int c = 0; c < Q; c++) { sc[c] = lsize(M,Q,c); sd[c] = lstart(M,Q,c); }
        vector<double> y_full(wrank == 0 ? M : 0);
        if (wrank == 0) for (int i = 0; i < M; i++) y_full[i] = (double)(i * 2);
        MPI_Scatterv(y_full.data(), sc.data(), sd.data(), MPI_DOUBLE,
                     y_local.data(), ys, MPI_DOUBLE, 0, row_comm);
    }
    MPI_Bcast(y_local.data(), ys, MPI_DOUBLE, 0, col_comm);

    double local_dot = 0.0;
    int i_start = max(xst, yst);
    int i_end   = min(xst + xs, yst + ys);
    for (int i = i_start; i < i_end; i++)
        local_dot += x_local[i - xst] * y_local[i - yst];

    // Allreduce to get global dot
    double global_dot = 0.0;
    MPI_Allreduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    if (wrank == 0)
        cout << "Dot product x . y = " << global_dot << "\n";

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
