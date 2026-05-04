#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int N = 3; // pick any N

        Kokkos::View<double****> myView("4DView", 5, 7, 12, N);

        std::cout << "Created 4D View: " << myView.label() << std::endl;
        std::cout << "Dimensions: "
                  << myView.extent(0) << " x "
                  << myView.extent(1) << " x "
                  << myView.extent(2) << " x "
                  << myView.extent(3) << std::endl;
    }
    Kokkos::finalize();
    return 0;
}