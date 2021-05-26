#define TEST

#include "main.cpp"

#include <fstream>


int main(int argc, char *argv[]) {
    assert(argc == 2);
    std::ifstream fin(argv[1]);
    input::getHV(fin);
    std::cerr << std::fixed;
    std::cerr << std::setprecision(10);
    solve();
    fin.close();
}
