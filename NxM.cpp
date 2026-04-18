#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int N = 4, M = 5;

        Kokkos::View<double**> myView("myView", N, M);

        Kokkos::parallel_for("FillView",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N, M}),
            KOKKOS_LAMBDA(int i, int j) {
                myView(i, j) = 1000.0 * i * j;
            }
        );

        auto hostView = Kokkos::create_mirror_view(myView);
        Kokkos::deep_copy(hostView, myView);

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                std::cout << hostView(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
    Kokkos::finalize();
    return 0;
}