static constexpr int N = 1; // # of dimensions

#include <cmath> //for trig
#include <numbers> //for pi
#include "../../include/ButcherTableau.hpp"


int main(int argc, char * const argv[])
{
    // --------------------------------------------------------------------
    // Define temporal parameters.
    // --------------------------------------------------------------------
    int step = 0;
    double t = 0;
    double dtRK = 0;

    // pp_hed::ButcherTableau RKtable = pp_hed::get_butcher_tableau("midpoint");
    pp_hed::ButcherTableau RKtable = pp_hed::get_butcher_tableau("heun_euler");

    if (RKtable.isEmbedded()) {
      std::cout << RKtable.name() << " is using error coefficients = ";
      for (int k = 0; k < RKtable.nStages(); ++k) {
        std::cout << (k > 0 ? " " : "") << RKtable.e(k);
      }
      std::cout << std::endl;
    }

    return 0;
}