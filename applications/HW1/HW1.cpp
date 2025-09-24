static constexpr int N = 1; // # of dimensions
static constexpr int Nfields = 2; // # of fields (position & velocity)

#include <numbers> //for pi
#include "../../include/Common.hpp"
#include "../../include/ButcherTableau.hpp"

int main(int argc, char * const argv[])
{
  // --------------------------------------------------------------------
  // Define constants
  // --------------------------------------------------------------------
  double π = std::numbers::pi;
  double μ0 = 4 * π * 1e-7; // H/m (vacuum permeability)
  double max_ratio = 200;  // maximum compression ratio
  double max_time = 3100 * 1e-9; // 3100 ns, maximum simulated time

  // --------------------------------------------------------------------
  // Define temporal parameters.
  // --------------------------------------------------------------------
  const int num_steps = 5000;
  double t = 0;
  double Δt = 0.5*1e-10;
  pp_hed::ButcherTableau RKtable = pp_hed::get_butcher_tableau("rk4");

  // --------------------------------------------------------------------
  // Define problem-specific information.
  // --------------------------------------------------------------------
  double τpeak = 100 * 1e-9; // 100 ns rise time
  double h = 7.5 * 1e-3; // 7.5 mm height of liner

  // These are the values for part (b)
  // double ρ = 1850; // 1850 kg/m^3 density of Beryllium
  // double Ipeak = 18 * 1e6; // 18 MA peak current
  // double r_lo = 2.79 * 1e-3; // 2.79 mm initial radius of liner (outer)
  // double r_go = 2.325 * 1e-3; // 2.325 mm initial radius of liner (inner)

  // These are the values for part (c)
  // double ρ = 1850; // 1850 kg/m^3 density of Beryllium
  // double Ipeak = 1 * 1e6; // 1 MA peak current
  // double r_lo = 2.79 * 1e-3; // 2.79 mm initial radius of liner (outer)
  // double r_go = 2.78868 * 1e-3; // 2.78868 mm initial radius of liner (inner)

  // These are the values for part (d)
  // double ρ = 1850; // 1850 kg/m^3 density of Beryllium
  // double Ipeak = 1 * 1e6; // 1 MA peak current
  // double r_lo = 2 * 2.79 * 1e-3; // 2 * 2.79 mm initial radius of liner (outer)
  // double r_go = 5.57984 * 1e-3; // 5.57984 mm initial radius of liner (inner)

  // These are the values for part (e)
  // double ρ = 1850; // Π = 3
  // double ρ = 1387.5; // Π = 4
  double ρ = 1110; // Π = 5
  double Ipeak = 18 * 1e6; // 18 MA peak current
  double r_lo = 2.79 * 1e-3; // 2.79 mm initial radius of liner (outer)
  double r_go = 2.325 * 1e-3; // 2.325 mm initial radius of liner (inner)

  double volume = π * (r_lo * r_lo - r_go * r_go) * h;
  double mass = ρ * volume;

  // --------------------------------------------------------------------
  // Define field information.
  // --------------------------------------------------------------------
  std::array<double, Nfields> fields;
  // initial position
  fields[0] = r_lo;
  // initial velocity
  fields[1] = 0.0;

  auto rhs = [&](std::array<double, Nfields> fields, double t) {
    std::array<double, Nfields> RHS_data;

    // r_dot RHS
    RHS_data[0] = fields[1];

    // v_dot RHS
    // double current = PFL_current(Ipeak, τpeak, t);
    double current = LTD_current(Ipeak, τpeak, t);

    double numerator = - μ0 * h * current * current;
    double denominator = 4.0 * π * mass * fields[0];
    RHS_data[1] = numerator / denominator;

    return RHS_data;
  };

  // --------------------------------------------------------------------
  // Generic Runge-Kutta explicit time integrator
  // --------------------------------------------------------------------
  auto RKstep = [&](std::array<double, Nfields>& fields, double t, double Δt) {
    const int nStages = RKtable.nStages();

    // Storage for stage values
    std::vector<std::array<double, Nfields>> k(nStages);

    // Compute stage values
    for (int stage = 0; stage < nStages; ++stage) {
      // Create temporary field values for this stage
      std::array<double, Nfields> temp_fields = fields;

      // Add contribution from previous stages according to Butcher tableau
      const auto a_coeffs = RKtable(stage);
      for (int prev_stage = 0; prev_stage < stage; ++prev_stage) {
        for (int field_idx = 0; field_idx < Nfields; ++field_idx) {
          temp_fields[field_idx] += Δt * a_coeffs[prev_stage] * k[prev_stage][field_idx];
        }
      }

      // Compute RHS at this stage
      double stage_time = t + RKtable.c(stage) * Δt;
      k[stage] = rhs(temp_fields, stage_time);
    }

    // Update solution using weighted sum of stage values
    std::array<double, Nfields> fields_new = fields;
    for (int stage = 0; stage < nStages; ++stage) {
      for (int field_idx = 0; field_idx < Nfields; ++field_idx) {
        fields_new[field_idx] += Δt * RKtable.bH(stage) * k[stage][field_idx];
      }
    }

    return fields_new;
  };

  // --------------------------------------------------------------------
  // Run the simulation
  // --------------------------------------------------------------------
  std::cout << "Time-marching with " << RKtable.name() << " integrator" << std::endl;
  std::cout << "Step " << 0 << ": t = " << t << ", r_l = " << fields[0]
            << ", v_l = " << fields[1] << ", I = " << LTD_current(Ipeak, τpeak, t)
            << ", C_r = " << 1.0 << ", KE = " << 0.0 << std::endl;

  for (int step = 1; step < num_steps; ++step) {
    // Advance solution one step
    fields = RKstep(fields, t, Δt);
    t += Δt;

    // Calculate metrics
    double ratio = r_lo / fields[0];
    double KE = 0.5 * mass * fields[1] * fields[1]; // Kinetic energy

    // Print current state
    std::cout << "Step " << step << ": t = " << t << ", r_l = " << fields[0]
              << ", v_l = " << fields[1] << ", I = " << LTD_current(Ipeak, τpeak, t)
              << ", C_r = " << ratio << ", KE = " << KE << std::endl;

    // Check for possible early exit
    if (fields[0] < 0.0) {
      std::cout << "Liner position went negative, r_l = " << fields[0]
                << ". Exiting simulation" << std::endl;
      break; // Exit the loop
    }
    if (ratio > max_ratio) {
      std::cout << "Maximum compression ratio achieved, C_r = " << ratio
                << ". Exiting simulation" << std::endl;
      break; // Exit the loop
    }
    if (t > max_time) {
      std::cout << "Maximum simulation time reached, t = " << t
                << ". Exiting simulation" << std::endl;
      break; // Exit the loop
    }
  }

  return 0;
}