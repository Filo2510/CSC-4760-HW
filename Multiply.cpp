#include <Kokkos_Core.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

void matVecMultiply(Kokkos::View<double**> A, Kokkos::View<double*> B, Kokkos::View<double*> C) {
    if (A.extent(1) != B.extent(0)) {
        throw std::invalid_argument(
            "Dimension mismatch: A columns (" + std::to_string(A.extent(1)) +
            ") must match B length (" + std::to_string(B.extent(0)) + ")."
        );
    }
    int rows = A.extent(0);
    int cols = A.extent(1);

    Kokkos::parallel_for("MatVecMultiply", rows,
        KOKKOS_LAMBDA(int i) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += A(i, j) * B(j);
            }
            C(i) = sum;
        }
    );
}

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Using Problem 7
        const int rows = 3, cols = 3;

        Kokkos::View<double**> A("A", rows, cols);
        Kokkos::View<double*>  B("B", cols);
        Kokkos::View<double*>  C("C", rows);

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
            matVecMultiply(A, B, C);
            Kokkos::fence();

            auto C_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C);
            std::cout << "A x B result: [ ";
            for (int i = 0; i < rows; i++) std::cout << C_h(i) << " ";
            std::cout << "]\n";
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }
    Kokkos::finalize();
    return 0;
}