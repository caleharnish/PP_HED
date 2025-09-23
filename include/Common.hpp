#pragma once

#include <cmath> //for trig
#include <numbers> //for pi

double PFL_current(double Ipeak, double τpeak, double t) {
  double π = std::numbers::pi;
  double arg = π * t / (2.0 * τpeak);

  return Ipeak * sin(arg) * sin(arg);
}

double LTD_current(double Ipeak, double τpeak, double t) {
  double π = std::numbers::pi;
  double arg = π * t / (2.0 * τpeak);

  return Ipeak * sin(arg);
}
