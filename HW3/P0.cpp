#include <iostream>
#include <stdexcept>
using namespace std;

int linear_size(int M, int P, int p) {
    return M / P + (p < M % P ? 1 : 0);
}

int linear_start(int M, int P, int p) {
    return p * (M / P) + (p < M % P ? p : M % P);
}

int linear_l2g(int M, int P, int p, int i) {
    return linear_start(M, P, p) + i;
}

void linear_g2l(int M, int P, int I, int &p, int &i) {
    int base = M / P, extra = M % P, boundary = extra * (base + 1);
    if (I < boundary) { p = I / (base + 1); i = I % (base + 1); }
    else { int I2 = I - boundary; p = extra + I2 / base; i = I2 % base; }
}

void scatter_g2l(int P, int I, int &p, int &i) {
    p = I % P;
    i = I / P;
}

int scatter_l2g(int P, int p, int i) {
    return i * P + p;
}

// Prints the linear distribution layout for all processes
void print_linear(int M, int P) {
    cout << "\nLinear layout (M=" << M << ", P=" << P << "):\n";
    for (int p = 0; p < P; p++) {
        int start = linear_start(M, P, p), size = linear_size(M, P, p);
        cout << "  p=" << p << "  start=" << start << "  size=" << size << "  indices=[" << start;
        for (int k = 1; k < size; k++) cout << ", " << start + k;
        cout << "]\n";
    }
}

// Prints the scatter distribution layout for all processes
void print_scatter(int M, int P) {
    cout << "\nScatter layout (M=" << M << ", P=" << P << "):\n";
    for (int p = 0; p < P; p++) {
        cout << "  p=" << p << "  indices=[";
        bool first = true;
        for (int I = p; I < M; I += P) { if (!first) cout << ", "; cout << I; first = false; }
        cout << "]\n";
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        cerr << "Usage: " << argv[0] << " <M> <P> <p> <i>\n";
        cerr << "  M = vector length, P = num processes, p = rank, i = local index\n";
        return 1;
    }

    int M = stoi(argv[1]), P = stoi(argv[2]), p = stoi(argv[3]), i = stoi(argv[4]);

    if (M <= 0 || P <= 0)                   { cerr << "Error: M and P must be positive.\n"; return 1; }
    if (p < 0 || p >= P)                    { cerr << "Error: p must be in [0, P-1].\n";   return 1; }
    if (i < 0 || i >= linear_size(M, P, p)) {
        cerr << "Error: i=" << i << " out of range for p=" << p
             << " (local size=" << linear_size(M, P, p) << ").\n";
        return 1;
    }

    cout << "=== Index Mapping: M=" << M << ", P=" << P << ", p=" << p << ", i=" << i << " ===\n";

    print_linear(M, P);
    print_scatter(M, P);

    // linear local to global
    int I = linear_l2g(M, P, p, i);
    cout << "\n--- Linear local-to-global ---\n";
    cout << "  linear(p=" << p << ", i=" << i << ") --> I = " << I << "\n";

    int pv, iv;
    linear_g2l(M, P, I, pv, iv);
    cout << "  verify: I=" << I << " --> (p=" << pv << ", i=" << iv << ") "
         << ((pv == p && iv == i) ? "[OK]" : "[FAIL]") << "\n";

    // scatter process and local index
    int pp, ip;
    scatter_g2l(P, I, pp, ip);
    cout << "\n--- Global I to scatter mapping ---\n";
    cout << "  scatter(I=" << I << ") --> p'=" << pp << ", i'=" << ip << "\n";

    int I2 = scatter_l2g(P, pp, ip);
    cout << "  verify: (p'=" << pp << ", i'=" << ip << ") --> I=" << I2 << " "
         << ((I2 == I) ? "[OK]" : "[FAIL]") << "\n";

    cout << "\n=== Summary ===\n";
    cout << "  Linear  (p=" << p << ", i=" << i << ") --> Global I=" << I << "\n";
    cout << "  Scatter (I=" << I << ")       --> p'=" << pp << ", i'=" << ip << "\n";

    return 0;
}
