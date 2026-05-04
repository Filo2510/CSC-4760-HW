#include <iostream>
#include <iomanip>
#include <cstring>

int main() {
    const int N = 100000;
    const double delta = 1e-18;

    double sum1 = 1.0;
    for (int i = 0; i < N; i++) sum1 += delta;

    double sum2 = 0.0;
    for (int i = 0; i < N; i++) sum2 += delta;
    sum2 += 1.0;

    double diff = sum1 - sum2;

    // Print results
    std::cout << std::setprecision(20) << std::fixed;
    std::cout << "Sum 1 (start 1.0, add 1e-18 x100000):       " << sum1 << "\n";
    std::cout << "Sum 2 (start 0, add 1e-18 x100000, then +1):" << sum2 << "\n";
    std::cout << "Difference (sum1 - sum2):                    " << diff << "\n\n";

    // Print hex bit patterns
    unsigned long long hex1, hex2, hexd;
    std::memcpy(&hex1, &sum1, sizeof(double));
    std::memcpy(&hex2, &sum2, sizeof(double));
    std::memcpy(&hexd, &diff, sizeof(double));

    std::cout << std::hex << std::uppercase;
    std::cout << "Sum 1 in hex: 0x" << hex1 << "\n";
    std::cout << "Sum 2 in hex: 0x" << hex2 << "\n";
    std::cout << "Diff  in hex: 0x" << hexd << "\n";

    return 0;
}
