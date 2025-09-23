#pragma once

#include <span>
#include <string_view>
#include <vector>
#include <iostream>

namespace pp_hed {
struct ButcherTableau
{
  std::string_view     name_;
  std::vector<double>     a_; // [K*(K+1)/2]
  std::vector<double>     b_; // [K]
  std::vector<double> bstar_; // [K]
  std::vector<double>     c_; // [K + 1]  // <-- contains an "extra" time == 1
  std::vector<double>     e_; // [K]
  int              n_stages_;
  int                 order_;
  bool          is_embedded_;

  struct named_ctor_args {
    std::string_view name;
    int K;
    int order;
    bool embedded;
    std::vector<double> a;
    std::vector<double> b;
    std::vector<double> bstar;
    std::vector<double> c;
  };

  ButcherTableau(const ButcherTableau&) = default;
  ButcherTableau(ButcherTableau&&) = default;

  explicit ButcherTableau(named_ctor_args&& args)
      : ButcherTableau(args.name, args.K, args.order, args.embedded,
                       std::move(args.a),
                       std::move(args.b),
                       std::move(args.bstar),
                       std::move(args.c))
  {
  }

  explicit ButcherTableau(std::string_view name, int K, int order, bool embedded,
                 std::vector<double> a,
                 std::vector<double> b,
                 std::vector<double> bstar,
                 std::vector<double> c)
      :        name_(name)
      ,           a_(std::move(a))
      ,           b_(std::move(b))
      ,       bstar_(std::move(bstar))
      ,           c_(std::move(c))
      ,           e_(K)
      ,    n_stages_(K)
      ,       order_(order)
      , is_embedded_(embedded)
  {
    for (int k = 0; k < n_stages_; ++k) {
      e_[k] = b_[k] - bstar_[k];
    }
  }

  std::string_view name() const {
    return name_;
  }

  int nStages() const {
    return n_stages_;
  }

  bool isEmbedded() const {
    return is_embedded_;
  }

  int order() const {
    return order_;
  }

  double operator()(int i, int j) const {
    return a(i, j);
  }

  std::span<const double> operator()(int i) const {
    return std::span(a_.begin() + (i*(i-1)) / 2, i);
  }

  double a(int i, int j) const {
    return a_[(i*(i-1))/2 + j];
  }

  double bL(int i) const {
    return bstar_[i];
  }

  double bH(int i) const {
    return b_[i];
  }

  double c(int i) const {
    return c_[i];
  }

  double e(int i) const {
    return e_[i];
  }
};

/// Create a ButcherTableau.
///
/// Valid string identifiers are:
///
///  "Midpoint"
///  "heun"
///  "ralston"
///  "rk4"
///  "rk4_38"
///  "heun_euler"
///  "rkf1_2"
///  "bogacki_shampine"
///  "rkf4_5_A"
///  "rkf4_5_B"
///  "cash_karp"
///  "domand_prince"
///
ButcherTableau get_butcher_tableau(std::string_view = "rkf4_5_A");
}
