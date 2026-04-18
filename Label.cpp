#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // Create a 1D View with a label
        Kokkos::View<double*> myView("MyKokkosView", 10);

        // Print the label using .label()
        std::cout << "View label: " << myView.label() << std::endl;
        std::cout << "View size:  " << myView.size() << std::endl;
    }
    Kokkos::finalize();
    return 0;
}