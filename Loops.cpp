#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int ROWS = 1000, COLS = 1000;

        // Create and fill a 2D View
        Kokkos::View<double**> A("Matrix", ROWS, COLS);
        Kokkos::parallel_for("Fill",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {ROWS, COLS}),
            KOKKOS_LAMBDA(int i, int j) {
                A(i, j) = static_cast<double>(i * COLS + j + 1);
            }
        );
        Kokkos::fence();

        Kokkos::View<double*> rowSumParallel("RowSumParallel", ROWS);
        Kokkos::Timer timerParallel;

        Kokkos::parallel_for("ParallelRowSum", ROWS,
            KOKKOS_LAMBDA(int i) {
                double sum = 0.0;
                for (int j = 0; j < COLS; j++) sum += A(i, j);
                rowSumParallel(i) = sum;
            }
        );
        Kokkos::fence();
        double timeParallel = timerParallel.seconds();

        // Serial row sums with a standard for loop
        auto A_h = Kokkos::create_mirror_view(A);
        Kokkos::deep_copy(A_h, A);

        std::vector<double> rowSumSerial(ROWS, 0.0);
        Kokkos::Timer timerSerial;

        for (int i = 0; i < ROWS; i++) {
            double sum = 0.0;
            for (int j = 0; j < COLS; j++) sum += A_h(i, j);
            rowSumSerial[i] = sum;
        }
        double timeSerial = timerSerial.seconds();

        // Verify correctness
        auto rsp_h = Kokkos::create_mirror_view(rowSumParallel);
        Kokkos::deep_copy(rsp_h, rowSumParallel);
        std::cout << "First 3 row sums (parallel): ";
        for (int i = 0; i < 3; i++) std::cout << rsp_h(i) << "  ";
        std::cout << "\nFirst 3 row sums (serial):   ";
        for (int i = 0; i < 3; i++) std::cout << rowSumSerial[i] << "  ";
        std::cout << "\n\n";

        std::cout << "Time (parallel_for): " << timeParallel << " seconds\n";
        std::cout << "Time (serial for):   " << timeSerial   << " seconds\n";
        std::cout << "Speedup (serial/parallel): "
                  << timeSerial / timeParallel << "x\n";
    }
    Kokkos::finalize();
    return 0;
}