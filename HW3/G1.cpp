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
    if (argc != 5) {
        if (wrank==0) cerr << "Usage: " << argv[0] << " <P> <Q> <M> <N>\n";
        MPI_Finalize(); return 1;
    }
    int P=atoi(argv[1]), Q=atoi(argv[2]), M=atoi(argv[3]), N=atoi(argv[4]);
    if (P<1||Q<1||M<1||N<1||P*Q!=wsize) {
        if (wrank==0) cerr << "Error: need P*Q==world_size, all>=1\n";
        MPI_Finalize(); return 1;
    }
    int row=wrank/Q, col=wrank%Q;
    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);

    int xs=lsize(M,P,row), xst=lstart(M,P,row);
    vector<double> x_local(xs,0.0);
    if (col==0) {
        vector<int> sc(P),sd(P);
        for(int r=0;r<P;r++){sc[r]=lsize(M,P,r);sd[r]=lstart(M,P,r);}
        vector<double> xf(wrank==0?M:0);
        if(wrank==0) for(int i=0;i<M;i++) xf[i]=(double)i;
        MPI_Scatterv(xf.data(),sc.data(),sd.data(),MPI_DOUBLE,x_local.data(),xs,MPI_DOUBLE,0,col_comm);
    }
    MPI_Bcast(x_local.data(),xs,MPI_DOUBLE,0,row_comm);

    // Horizontal scatter distribution
    int ys=lsize(M,Q,col);
    vector<double> y_local(ys,0.0);
    if (row==0) {
        vector<int> sc(Q),sd(Q);
        for(int c=0;c<Q;c++){sc[c]=lsize(M,Q,c);sd[c]=lstart(M,Q,c);}
        vector<double> yf(wrank==0?M:0);
        if(wrank==0) for(int i=0;i<M;i++) yf[i]=(double)(i*2);
        MPI_Scatterv(yf.data(),sc.data(),sd.data(),MPI_DOUBLE,y_local.data(),ys,MPI_DOUBLE,0,row_comm);
    }
    MPI_Bcast(y_local.data(),ys,MPI_DOUBLE,0,col_comm);

    vector<int> rc(P),rd(P);
    for(int r=0;r<P;r++){rc[r]=lsize(M,P,r);rd[r]=lstart(M,P,r);}
    vector<double> x_all(M);
    MPI_Allgatherv(x_local.data(),xs,MPI_DOUBLE,x_all.data(),rc.data(),rd.data(),MPI_DOUBLE,col_comm);

    double local_dot=0.0;
    if (row==0) for(int j=0;j<ys;j++) local_dot += x_all[col+j*Q]*y_local[j];
    double global_dot=0.0;
    MPI_Allreduce(&local_dot,&global_dot,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    if (wrank==0) cout << "Part 1 - Dot product x.y = " << global_dot << "\n";

    // Matrix-vector y = Av
    int vs=lsize(N,Q,col), vst=lstart(N,Q,col);
    vector<double> v_local(vs,0.0);
    if (row==0) {
        vector<int> sc(Q),sd(Q);
        for(int c=0;c<Q;c++){sc[c]=lsize(N,Q,c);sd[c]=lstart(N,Q,c);}
        vector<double> vf(wrank==0?N:0);
        if(wrank==0) for(int j=0;j<N;j++) vf[j]=(double)j;
        MPI_Scatterv(vf.data(),sc.data(),sd.data(),MPI_DOUBLE,v_local.data(),vs,MPI_DOUBLE,0,row_comm);
    }
    MPI_Bcast(v_local.data(),vs,MPI_DOUBLE,0,col_comm);

    vector<double> A_local(xs*vs);
    for(int i=0;i<xs;i++) for(int j=0;j<vs;j++) A_local[i*vs+j]=(double)((xst+i)*N+(vst+j));

    vector<double> y_mv(xs,0.0);
    for(int i=0;i<xs;i++) for(int j=0;j<vs;j++) y_mv[i]+=A_local[i*vs+j]*v_local[j];
    MPI_Allreduce(MPI_IN_PLACE,y_mv.data(),xs,MPI_DOUBLE,MPI_SUM,row_comm);

    // Print y=Av
    if(col==0) {
        cout<<"P"<<wrank<<"(row="<<row<<") y=Av: [";
        for(int i=0;i<xs;i++) cout<<(i?",":"")<<(long long)y_mv[i];
        cout<<"]\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}
