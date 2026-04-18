#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int N = 10;
        Kokkos::View<double*> myView("myView", N);

        Kokkos::parallel_for("Fill", N, KOKKOS_LAMBDA(int i) {
            myView(i) = (double)(i * i);  // 0, 1, 4, 9, 16, ...
        });

        double maxVal = 0.0;

        Kokkos::parallel_reduce("FindMax", N,
            KOKKOS_LAMBDA(int i, double& localMax) {
                if (myView(i) > localMax) {
                    localMax = myView(i);
                }
            },
            Kokkos::Max<double>(maxVal)
        );

        std::cout << "Maximum value: " << maxVal << std::endl;
    }
    Kokkos::finalize();
    return 0;
}