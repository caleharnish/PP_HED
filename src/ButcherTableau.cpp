#include "ButcherTableau.hpp"
#include <map>

static const pp_hed::ButcherTableau midpoint{{
    .name = "Midpoint",
    .K = 2,
    .order = 2,
    .embedded = false,
    .a = {
      0.5,
      0, 1 //b
    },
    .b = { 0, 1 },
    .bstar = { 0, 1 },
    .c = {
      0,
      0.5,
      1  // <-- extra parameter needed
    }
  }};

static const pp_hed::ButcherTableau heun{{
    .name = "Heun",
    .K = 2,
    .order = 2,
    .embedded = false,
    .a = {
      1,
      0.5, 0.5 //b
    },
    .b = { 0.5, 0.5 },
    .bstar = { 0.5, 0.5 },
    .c = {
      0,
      1,
      1  // <-- extra parameter needed
    }
  }};

static const pp_hed::ButcherTableau ralston{{
    .name = "Ralston",
    .K = 2,
    .order = 2,
    .embedded = false,
    .a = {
      2./3,
      1./4, 3./4 //b
    },
    .b = { 1./4, 3./4 },
    .bstar = { 1./4, 3./4 },
    .c = {
      0,
      2./3,
      1  // <-- extra parameter needed
    }
  }};

static const pp_hed::ButcherTableau rk3{{
    .name = "RK3",
    .K = 3,
    .order = 3,
    .embedded = false,
    .a = {
      0.5,
      -1, 2,
      1./6, 2./3, 1./6 //b
    },
    .b = { 1./6, 2./3, 1./6 },
    .bstar = { 1./6, 2./3, 1./6 },
    .c = {
      0,
      0.5,
      1,
      1 // <-- extra parameter needed
    }
  }};

static const pp_hed::ButcherTableau heun3{{
    .name = "heun3",
    .K = 3,
    .order = 3,
    .embedded = false,
    .a = {
      1./3,
      0, 2./3,
      1./4, 0, 3./4 //b
    },
    .b = { 1./4, 0, 3./4 },
    .bstar = { 1./4, 0, 3./4 },
    .c = {
      0,
      1./3,
      2./3,
      1 // <-- extra parameter needed
    }
  }};

static const pp_hed::ButcherTableau ralston3{{
    .name = "ralston3",
    .K = 3,
    .order = 3,
    .embedded = false,
    .a = {
      1./2,
      0, 3./4,
      2./9, 1./3, 4./9 //b
    },
    .b = { 2./9, 1./3, 4./9 },
    .bstar = { 2./9, 1./3, 4./9 },
    .c = {
      0,
      1./2,
      3./4,
      1 // <-- extra parameter needed
    }
  }};

static const pp_hed::ButcherTableau ssprk3{{
    .name = "SSPRK3",
    .K = 3,
    .order = 3,
    .embedded = false,
    .a = {
      1,
      1./4, 1./4,
      1./6, 1./6, 2./3 //b
    },
    .b = { 1./6, 1./6, 2./3 },
    .bstar = { 1./6, 1./6, 2./3 },
    .c = {
      0,
      1,
      1./2,
      1 // <-- extra parameter needed
    }
  }};

static const pp_hed::ButcherTableau rk4{{
    .name = "RK4",
    .K = 4,
    .order = 4,
    .embedded = false,
    .a = {
      0.5,
      0, 0.5,
      0, 0, 1,
      1./6, 1./3, 1./3, 1./6 //b
    },
    .b = { 1./6, 1./3, 1./3, 1./6 },
    .bstar = { 1./6, 1./3, 1./3, 1./6 },
    .c = {
      0,
      0.5,
      0.5,
      1,
      1 // <-- extra parameter needed
    }
  }};

static const pp_hed::ButcherTableau rk4_38{{
    .name = "RK4 three-eighths",
    .K = 4,
    .order = 4,
    .embedded = false,
    .a = {
      1./3,
      -1./3, 1,
      1, -1, 1,
      1./8, 3./8, 3./8, 1./8 //b
    },
    .b = { 1./8, 3./8, 3./8, 1./8 },
    .bstar = { 1./8, 3./8, 3./8, 1./8 },
    .c = {
      0,
      1./3,
      2./3,
      1,
      1 // <-- extra parameter needed
    }
  }};

// ------------------------------------------------------------------------
// EMBEDDED METHODS
// ------------------------------------------------------------------------

//uses orders 1 & 2
static const pp_hed::ButcherTableau heun_euler{{
    .name = "Heun-Euler",
    .K = 2,
    .order = 2,
    .embedded = true,
    .a = {
      1,
      1,  0 //bL
    },
    .b = { 0.5, 0.5 },
    .bstar = { 1, 0 },
    .c = {
      0,
      1,
      1 // <-- extra parameter needed
    }
  }};

//uses orders 1 & 2
//NOTE: this one might not work based on the structure of the a & b lists
static const pp_hed::ButcherTableau rkf1_2{{
    .name = "Fehlberg 1(2)",
    .K = 3,
    .order = 2,
    .embedded = true,
    .a = {
      0.5   ,
      1./256, 255./256, 0 //bL
    },
    .b = { 1./512, 255./256, 1./512},
    .bstar = { 1./256, 255./256, 0},
    .c = {
      0,
      0.5,
      1,
      1 // <-- extra parameter needed
    },
  }};

//uses orders 2 & 3
static const pp_hed::ButcherTableau bogacki_shampine{{
    .name = "Bogacki-Shampine",
    .K = 4,
    .order = 3,
    .embedded = true,
    .a = {
      0.5 ,
      0  , 3./4,
      2./9 , 1./3, 4./9,
      7./24, 1./4, 1./3, 1./8//bL
    },
    .b = {
      2./9 , 1./3, 4./9,   0
    },
    .bstar = {
      7./24, 1./4, 1./3, 1./8
    },
    .c = {
      0,
      0.5,
      3./4,
      1,
      1 // <-- extra parameter needed
    }
  }};

//uses orders 4 & 5
static const pp_hed::ButcherTableau rkf4_5_A{{
    .name = "Fehlberg 4(5) α2 = 1/3",
    .K = 6,
    .order = 5,
    .embedded = true,
    .a = {
      2./9  ,
      1./12 ,    1./4  ,
      69./128, -243./128,  135./64   ,
      -17./12,   27./4  ,  -27./5    ,    16./15   ,
      65./432,   -5./16 ,   13./16   ,     4./27   ,  5./144,
      1./9  ,      0   ,     9./20   ,    16./45   ,  1./12 ,   0 //bL
    },
    .b = {
      47./450,     0   ,    12./25   ,    32./225  ,  1./30 , 6./25
    },
    .bstar = {
      1./9  ,      0   ,     9./20   ,    16./45   ,  1./12 ,   0
    },
    .c = {
      0,
      2./9,
      1./3,
      3./4,
      1,
      5./6,
      1 // <-- extra parameter needed
    }
  }};

static const pp_hed::ButcherTableau rkf4_5_B{{
    .name = "Fehlberg 4(5) α2 = 3/8",
    .K = 6,
    .order = 5,
    .embedded = true,
    .a = {
      1./4    ,
      3./32   ,       9./32  ,
      1932./2197, -7200./2197,   7296./2197   ,
      439./216,        -8    ,  -3680./513    ,  -845./4104 ,
      -8./27  ,         2    ,  -3544./2565   ,  1859./4104 , -11./40,
      25./216 ,         0    ,   1408./2565   ,  2197./4104 , -1./5  ,   0 //bL
    },
    .b = {
      16./135,      0   , 6656./12825, 28561./56430, -9./50 , 2./55
    },
    .bstar = {
      25./216,      0   , 1408./2565 ,  2197./4104 , -1./5  ,   0
    },
    .c = {
      0,
      1./4,
      3./8,
      12./13,
      1,
      1./2,
      1 // <-- extra parameter needed
    }
  }};

//uses orders 4 & 5
static const pp_hed::ButcherTableau cash_karp{{
    .name = "Cash-Karp",
    .K = 6,
    .order = 5,
    .embedded = true,
    .a = {
      1./5    ,
      3./40   ,   9./40 ,
      3./10   ,  -9./10 ,     6./5    ,
      -11./54   ,   5./2  ,   -70./27   ,    35./27    ,
      1631./55296, 175./512,   575./13824, 44275./110592, 253./4096,
      2825./27648,     0   , 18575./48384, 13525./55296 , 277./14336,   1./4    //bL
    },
    .b = {
      37./378  ,     0   ,   250./621  ,   125./594   ,     0     , 512./1771
    },
    .bstar = {
      2825./27648,     0   , 18575./48384, 13525./55296 , 277./14336,   1./4
    },
    .c = {
      0,
      1./5,
      3./10,
      3./5,
      1,
      7./8,
      1 // <-- extra parameter needed
    }
  }};

//uses orders 4 & 5
static const pp_hed::ButcherTableau dormand_prince{{
    .name = "Dormand-Prince",
    .K = 7,
    .order = 5,
    .embedded = true,
    .a = {
      1./5    ,
      3./40   ,      9./40  ,
      44./45   ,    -56./15  ,    32./9    ,
      19372./6561 , -25360./2187, 64448./6561 , -212./729,
      9017./3168 ,   -355./33  , 46732./5247 ,   49./176,  -5103./18656 ,
      35./384  ,        0    ,   500./1113 ,  125./192,  -2187./6784  ,  11./84,
      5179./57600,        0    ,  7571./16695,  393./640, -92097./339200, 187./2100, 1./40 //bL
    },
    .b = {
      35./378  ,        0    ,   500./1113 ,  125./192,  -2187./6784  ,  11./84  , 0
    },
    .bstar = {
      5179./57600,        0    ,  7571./16695,  393./640, -92097./339200, 187./2100, 1./40
    },
    .c = {
      0,
      1./5,
      4./5,
      8./9,
      1,
      1,
      1 // <-- extra parameter needed
    }
  }};

// Using a map because it's easy and performance here doesn't matter.
static const std::map<std::string_view, pp_hed::ButcherTableau> tables_ = {
  {"midpoint", midpoint},
  {"heun", heun},
  {"heun3", heun3},
  {"ralston", ralston},
  {"ralston3", ralston3},
  {"ssprk3", ssprk3},
  {"rk3", rk3},
  {"rk4", rk4},
  {"rk4_38", rk4_38},
  {"heun_euler", heun_euler},
  {"rkf1_2", rkf1_2},
  {"bogacki_shampine", bogacki_shampine},
  {"rkf4_5_A", rkf4_5_A},
  {"rkf4_5_B", rkf4_5_B},
  {"cash_karp", cash_karp},
  {"dormand_prince", dormand_prince}
};

pp_hed::ButcherTableau pp_hed::get_butcher_tableau(std::string_view id) {
  if (auto i = tables_.find(id); i != tables_.end()) {
    return i->second;
  }
  return rkf4_5_A;
}
