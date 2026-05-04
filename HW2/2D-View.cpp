#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

void addVectorToRows(Kokkos::View<double**> A, Kokkos::View<double*> B, Kokkos::View<double**> C) {
    if (A.extent(1) != B.extent(0)) {
        throw std::invalid_argument(
            "Dimension mismatch: Matrix columns (" + std::to_string(A.extent(1)) +
            ") must match Vector length (" + std::to_string(B.extent(0)) + ")."
        );
    }
    int rows = A.extent(0);
    int cols = A.extent(1);
    Kokkos::parallel_for("AddBroadcast",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {rows, cols}),
        KOKKOS_LAMBDA(const int i, const int j) {
            C(i, j) = A(i, j) + B(j);
        }
    );
}

// Prints a 2D View
void printMatrix(const std::string& name, const Kokkos::View<double**>& M) {
    auto M_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), M);
    std::cout << name << ":\n";
    for (size_t i = 0; i < M_h.extent(0); i++) {
        std::cout << "  [ ";
        for (size_t j = 0; j < M_h.extent(1); j++)
            std::cout << M_h(i, j) << " ";
        std::cout << "]\n";
    }
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int rows = 3, cols = 3;

        Kokkos::View<double**> A("A", rows, cols);
        Kokkos::View<double*>  B("B", cols);
        Kokkos::View<double**> C("C", rows, cols);

        auto A_h = Kokkos::create_mirror_view(A);
        auto B_h = Kokkos::create_mirror_view(B);

        double init_A[3][3] = {{130, 147, 115},
                               {224, 158, 187},
                               { 54, 158, 120}};
        double init_B[3] = {221, 12, 157};

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) A_h(i, j) = init_A[i][j];
            B_h(i) = init_B[i];
        }
        Kokkos::deep_copy(A, A_h);
        Kokkos::deep_copy(B, B_h);

        try {
            addVectorToRows(A, B, C);
            Kokkos::fence();
            printMatrix("A", A);
            std::cout << "B: [ 221 12 157 ]\n\n";
            printMatrix("Soln (A + B row-wise)", C);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    Kokkos::finalize();
    return 0;
}