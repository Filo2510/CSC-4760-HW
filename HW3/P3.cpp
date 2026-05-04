#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

int lsize(int M, int N, int r) { return M/N + (r < M%N ? 1 : 0); }
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
        if (wrank == 0) cerr << "Error: need P*Q == world_size, and P,Q,M >= 1\n";
        MPI_Finalize(); return 1;
    }
    int row = wrank/Q, col = wrank%Q;
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);
    int xs = lsize(M, P, row);
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
    vector<int> rc(P), rd(P);
    for (int r = 0; r < P; r++) { rc[r] = lsize(M,P,r); rd[r] = lstart(M,P,r); }
    vector<double> x_all(M);
    MPI_Allgatherv(x_local.data(), xs, MPI_DOUBLE,
                   x_all.data(), rc.data(), rd.data(), MPI_DOUBLE, col_comm);
    int ys = lsize(M, Q, col);
    vector<double> y_local(ys, 0.0);
    for (int j = 0; j < ys; j++) { // Changed for problem 3
        int J = col + j * Q;
        y_local[j] = x_all[J];
    }
    for (int r = 0; r < wsize; r++) {
        if (wrank == r) {
            cout << "P" << wrank << "(row=" << row << ",col=" << col << ") x=[";
            for (int i = 0; i < xs; i++) cout << (i?",":"") << (int)x_local[i];
            cout << "]  y=[";
            for (int i = 0; i < ys; i++) cout << (i?",":"") << (int)y_local[i];
            cout << "]\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
