#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        const int N = 10;
        Kokkos::View<double*> input("input", N);
        Kokkos::View<double*> output("output", N);

        // Fill input
        Kokkos::parallel_for("Fill", N, KOKKOS_LAMBDA(int i) {
            input(i) = 1.0;
        });
        Kokkos::fence();

        const int runs = 5;
        for (int run = 0; run < runs; run++) {
            Kokkos::Timer timer;

            // Parallel scan
            Kokkos::parallel_scan("PrefixSum", N,
                KOKKOS_LAMBDA(int i, double& update, bool final) {
                    update += input(i);
                    if (final) output(i) = update;
                }
            );
            Kokkos::fence();

            double elapsed = timer.seconds();

            // First run
            if (run == 0) {
                auto output_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output);
                std::cout << "Partial sums: ";
                for (int i = 0; i < N; i++) std::cout << output_h(i) << " ";
                std::cout << "\n\n";
            }

            std::cout << "Run " << run + 1 << " time: " << elapsed << " seconds\n";
        }
    }
    Kokkos::finalize();
    return 0;
}