#include <iostream>
#include <memory>

extern "C" void func(const double*, std::size_t);

int main()
{
    std::cout << "Hello, World! [from C++]" << std::endl;

    //constexpr : const定数式
    constexpr auto N = std::size_t(10);
    auto a = std::make_unique<double[]>(N);
    a[0] = 0.12345;
    func(a.release(), N);   //need to unique_ptr.release()

    return 0;
}
