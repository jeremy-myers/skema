#include "AlgParams.h"
#include <algorithm>
#include <climits>
#include <cstring>
#include <ios>
#include <limits>
#include <sstream>

AlgParams::AlgParams()
    : inputfilename(""),
      outputfilename_prefix(""),
      issparse(false),
      issymmetric(false),
      isspd(false),
      matrix_m(0),
      matrix_n(0),
      rank(1),
      solver("fd"),
      print_level(0),
      debug_level(0),
      save_U(false),
      save_S(true),
      save_V(false),
      init_U(""),
      init_S(""),
      init_V(""),
      window(1),
      dense_svd_solver(false),
      primme_matvec("kokkos"),
      primme_eps(1e-4),
      primme_convtest_eps(1e-4),
      primme_convtest_skipitn(1),
      primme_initSize(0),
      primme_maxBasisSize(0),
      primme_minRestartSize(0),
      primme_maxBlockSize(0),
      primme_printLevel(0),
      primme_maxMatvecs(0),
      primme_maxIter(0),
      primme_locking(-1),
      primme_method("PRIMME_DEFAULT_METHOD"),
      primme_methodStage2("PRIMME_DEFAULT_METHOD"),
      reduce_rank_alpha(0),
      dynamic_tol_factor(1.0),
      dynamic_tol_iters(0),
      model(""),
      range(1),
      core(1),
      seed(0),
      seeds({0, 1, 2, 3}),
      kernel(""),
      gamma(1.0),
      sampler(""),
      samples(0),
      compute_resnorms(true),
      estimate_resnorms(true),
      compute_resnorms_only(false) {}

void error(std::string s) {
  std::cerr << "FATAL ERROR: " << s << std::endl;
  throw std::runtime_error(s);
}

void AlgParams::print(std::ostream& out) const {
  if (issparse) {
    out << "  sparse matrix";
  } else {
    out << "  dense matrix";
    out << ", size = " << matrix_m << " x " << matrix_n;
  }
  if (isspd) {
    out << "  (symmetric positive definite)\n";
  } else if (!isspd && issymmetric) {
    out << "  (symmetric)\n";
  } else if (matrix_m == matrix_n) {
    out << "  (square)\n";
  } else {
    out << "  (rectangular)\n";
  }
  out << "  rank = " << rank << std::endl;
  out << "  window size = " << window << std::endl;
  out << "  solver = " << solver << std::endl;

  if (!init_U.empty()) {
    out << "  initial guess U = " << init_U << std::endl;
  }
  if (!init_V.empty()) {
    out << "  initial guess V = " << init_V << std::endl;
  }

  out << "  save S = " << std::boolalpha << save_S << std::endl;
  out << "  save U = " << std::boolalpha << save_U << std::endl;
  out << "  save V = " << std::boolalpha << save_V << std::endl;

  if (solver == "fd") {
    out << "  svd solver = ";
    if (!dense_svd_solver) {
      out << "iterative" << std::endl;
    } else {
      out << "direct" << std::endl;
    }
    out << "  alpha = " << reduce_rank_alpha << std::endl;
  }

  if (solver == "sketchy") {
    out << "  range = " << range << std::endl;
    if (!isspd) {
      out << "  core = " << core << std::endl;
    }
    out << "  eta = " << eta << std::endl;
    out << "  nu = " << nu << std::endl;
    out << "  model = " << model << std::endl;
    if (isspd || !kernel.empty()) {
      out << "  seed = " << seed << std::endl;
    } else {
      out << "  seeds = ";
      for (auto i : seeds) {
        std::cout << " " << i;
      }
      std::cout << std::endl;
    }
  }

  if (!dense_svd_solver && solver == "fd") {
    out << "  iterative solver = PRIMME" << std::endl;
    out << "    PRIMME printLevel = " << primme_printLevel << std::endl;
    out << "    PRIMME tolerance = " << primme_eps << std::endl;
    if (samples > 0) {
      out << "    PRIMME conv test func eps = " << primme_convtest_eps
          << std::endl;
      out << "    PRIMME conv test skip = " << primme_convtest_skipitn
          << std::endl;
    }
    if (primme_maxIter > 0) {
      out << "    PRIMME maxOuterIters = " << primme_maxIter << std::endl;
      ;
    }
    if (primme_maxMatvecs > 0) {
      out << "    PRIMME maxMatvecs = " << primme_maxMatvecs << std::endl;
      ;
    }
    if (primme_maxBasisSize > 0) {
      out << "    PRIMME maxBasisSize = " << primme_maxBasisSize << std::endl;
    }
    if (primme_maxBlockSize > 0) {
      out << "    PRIMME maxBlockSize = " << primme_maxBasisSize << std::endl;
    }
    if (primme_minRestartSize > 0) {
      out << "    PRIMME minRestartSize = " << primme_minRestartSize
          << std::endl;
      ;
    }
    if (dynamic_tol_iters > 0) {
      out << "    Dynamic tolerance factor = " << dynamic_tol_factor
          << std::endl;
      out << "    Dynamic tolerance iterations = " << dynamic_tol_iters
          << std::endl;
    }
  }

  if (!kernel.empty()) {
    out << "  kernel = " << kernel << std::endl;
    out << "  gamma = " << gamma << std::endl;
  }

  out << "  calculate exact resnorms = " << std::boolalpha << compute_resnorms
      << std::endl;

  if (samples > 0) {
    out << "  estimate resnorms = " << std::boolalpha << estimate_resnorms
        << std::endl;
    out << "  sampler = " << sampler << std::endl;
    out << "  num samples = " << samples << std::endl;
  }
}

void AlgParams::parse(std::vector<std::string>& args) {
  // Parse options from command-line, using default values set above as defaults

  // Generic options
  inputfilename = parse_string(args, "--input", "");
  outputfilename_prefix = parse_string(args, "--output", "");
  issparse = parse_bool(args, "--sparse", "--dense", false);
  issymmetric = parse_bool(args, "--symmetric", "--rectangular", false);
  isspd = parse_bool(args, "--spd", "--not-spd", false);
  matrix_m = parse_int(args, "--m", matrix_m, 0, INT_MAX);
  matrix_n = parse_int(args, "--n", matrix_n, 0, INT_MAX);
  rank = parse_int(args, "--rank", rank, 1, INT_MAX);
  solver = parse_string(args, "--solver", "primme");
  print_level = parse_int(args, "--print-level", print_level, 0, 5);
  debug_level = parse_int(args, "--debug-level", debug_level, 0, 5);
  save_U = parse_bool(args, "--save-U", "--save-U-off", false);
  save_S = parse_bool(args, "--save-S", "--save-S-off", true);
  save_V = parse_bool(args, "--save-V", "--save-V-off", false);
  init_U = parse_string(args, "--init-U", "");
  init_S = parse_string(args, "--init-S", "");
  init_V = parse_string(args, "--init-V", "");
  init_dr_map_transpose =
      parse_bool(args, "--init-dr-trans", "--init-dr-trans-off", true);

  // Streaming options
  window = parse_int(args, "--window", window, 1, INT_MAX);

  // FDISVD options
  reduce_rank_alpha = parse_real(args, "--reduce-rank-alpha", 0.0, 0.0, 1.0);

  // SketchySVD options
  model = parse_string(args, "--model", "gauss");
  range = parse_int(args, "--range", range, 0, INT_MAX);
  core = parse_int(args, "--core", core, 0, INT_MAX);
  seed = parse_int(args, "--seed", seed, 0, INT_MAX);
  seeds = parse_int_array(args, "--seeds", seeds, 0, INT_MAX);
  eta = parse_real(args, "--eta", 1.0, 0.0, std::numeric_limits<double>::max());
  nu = parse_real(args, "--nu", 1.0, 0.0, std::numeric_limits<double>::max());

  // PRIMME &  solver options
  dense_svd_solver = parse_bool(args, "--svd", "--svds", false);
  primme_matvec = parse_string(args, "--primme_matvec", "default");
  primme_printLevel = parse_int(args, "--primme_printLevel", 0, 0, 5);
  primme_eps = parse_real(args, "--primme_eps", primme_eps,
                          std::numeric_limits<double>::epsilon(), 1.0);
  primme_convtest_eps =
      parse_real(args, "--primme_convtest_eps", primme_convtest_eps,
                 std::numeric_limits<double>::epsilon(), 1.0);
  primme_convtest_skipitn =
      parse_int(args, "--primme_convtest_skip", 1, 1, INT_MAX);
  primme_initSize = parse_int(args, "--primme_initSize", 0, 0, INT_MAX);
  primme_maxBasisSize = parse_int(args, "--primme_maxBasisSize", 0, 0, INT_MAX);
  primme_minRestartSize =
      parse_int(args, "--primme_minRestartSize", 0, 0, INT_MAX);
  primme_maxBlockSize = parse_int(args, "--primme_maxBlockSize", 0, 0, INT_MAX);
  primme_maxMatvecs = parse_int(args, "--primme_maxMatvecs", 0, 0, INT_MAX);
  primme_maxIter = parse_int(args, "--primme_maxIters", 0, 0, INT_MAX);
  primme_locking = parse_int(args, "--primme_locking", -1, -1, 1);
  primme_method =
      parse_string(args, "--primme_method", "PRIMME_DEFAULT_METHOD");
  primme_methodStage2 =
      parse_string(args, "--prime_methodStage2", "PRIMME_DEFAULT_METHOD");
  dynamic_tol_factor = parse_real(args, "--dynamic-tol-factor", 1.0, 0.0,
                                  std::numeric_limits<double>::max());
  dynamic_tol_iters = parse_int(args, "--dynamic-tol-iters", 0, 0, INT_MAX);

  // Kernel options
  kernel = parse_string(args, "--kernel", "");
  gamma = parse_real(args, "--gamma", gamma, 0.0,
                     std::numeric_limits<double>::max());

  // Sampling options
  sampler = parse_string(args, "--sampler", "");
  samples = parse_int(args, "--num-samples", 0, 0, INT_MAX);

  // Resnorms options
  compute_resnorms =
      parse_bool(args, "--compute-resnorms", "--compute-resnorms-off", true);
  estimate_resnorms =
      parse_bool(args, "--estimate-resnorms", "--estimate-resnorms-off", true);
  compute_resnorms_only = parse_bool(args, "--compute-resnorms-only",
                                     "--compute-resnorms-only-off", false);
}

double parse_real(std::vector<std::string>& args,
                  const std::string& cl_arg,
                  double default_value,
                  double min,
                  double max) {
  double tmp = default_value;
  auto it = std::find(args.begin(), args.end(), cl_arg);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg.size() > 2) && (cl_arg[0] == '-') &&
      (cl_arg[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg.substr(2));
  }
  if (it != args.end()) {
    auto arg_it = it;
    // get next cl_arg
    ++it;
    if (it == args.end()) {
      args.erase(arg_it);
      return tmp;
    }
    // convert to double
    char* cend = 0;
    tmp = std::strtod(it->c_str(), &cend);
    // check if cl_arg is actually a double
    if (it->c_str() == cend) {
      std::ostringstream error_string;
      error_string << "Unparseable input: " << cl_arg << " " << *it
                   << ", must be a double" << std::endl;
      error(error_string.str());
      exit(1);
    }
    // Remove argument from list
    args.erase(arg_it, ++it);
  }
  // check if double is within bounds
  if (tmp < min || tmp > max) {
    std::ostringstream error_string;
    error_string << "Bad input: " << cl_arg << " " << tmp
                 << ",  must be in the range (" << min << ", " << max << ")"
                 << std::endl;
    error(error_string.str());
    exit(1);
  }
  return tmp;
}

int parse_int(std::vector<std::string>& args,
              const std::string& cl_arg,
              int default_value,
              int min,
              int max) {
  int tmp = default_value;
  auto it = std::find(args.begin(), args.end(), cl_arg);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg.size() > 2) && (cl_arg[0] == '-') &&
      (cl_arg[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg.substr(2));
  }
  if (it != args.end()) {
    auto arg_it = it;
    // get next cl_arg
    ++it;
    if (it == args.end()) {
      args.erase(arg_it);
      return tmp;
    }
    // convert to ttb_indx
    if (*it == "inf" || *it == "Inf")
      tmp = INT_MAX;
    else {
      char* cend = 0;
      tmp = std::strtol(it->c_str(), &cend, 10);
      // check if cl_arg is actually a ttb_indx
      if (it->c_str() == cend) {
        std::ostringstream error_string;
        error_string << "Unparseable input: " << cl_arg << " " << *it
                     << ", must be an integer" << std::endl;
        error(error_string.str());
        exit(1);
      }
    }
    // Remove argument from list
    args.erase(arg_it, ++it);
  }
  // check if double is within bounds
  if (tmp < min || tmp > max) {
    std::ostringstream error_string;
    error_string << "Bad input: " << cl_arg << " " << tmp
                 << ",  must be in the range (" << min << ", " << max << ")"
                 << std::endl;
    error(error_string.str());
    exit(1);
  }
  return tmp;
}

bool parse_bool(std::vector<std::string>& args,
                const std::string& cl_arg_on,
                const std::string& cl_arg_off,
                bool default_value) {
  // return true if arg_on is found
  auto it = std::find(args.begin(), args.end(), cl_arg_on);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg_on.size() > 2) && (cl_arg_on[0] == '-') &&
      (cl_arg_on[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg_on.substr(2));
  }
  if (it != args.end()) {
    args.erase(it);
    return true;
  }

  // return false if arg_off is found
  it = std::find(args.begin(), args.end(), cl_arg_off);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg_off.size() > 2) && (cl_arg_off[0] == '-') &&
      (cl_arg_off[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg_off.substr(2));
  }
  if (it != args.end()) {
    args.erase(it);
    return false;
  }

  // return default value if not specified on command line
  return default_value;
}

std::string parse_string(std::vector<std::string>& args,
                         const std::string& cl_arg,
                         const std::string& default_value) {
  std::string tmp = default_value;
  auto it = std::find(args.begin(), args.end(), cl_arg);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg.size() > 2) && (cl_arg[0] == '-') &&
      (cl_arg[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg.substr(2));
  }
  if (it != args.end()) {
    auto arg_it = it;
    // get next cl_arg
    ++it;
    if (it == args.end()) {
      args.erase(arg_it);
      return tmp;
    }
    // get argument
    tmp = *it;
    // Remove argument from list
    args.erase(arg_it, ++it);
  }
  return tmp;
}

std::vector<int> parse_int_array(std::vector<std::string>& args,
                                 const std::string& cl_arg,
                                 const std::vector<int>& default_value,
                                 int min,
                                 int max) {
  char* cend = 0;
  int tmp;
  std::vector<int> vals;
  auto it = std::find(args.begin(), args.end(), cl_arg);
  // If not found, try removing the '--'
  if ((it == args.end()) && (cl_arg.size() > 2) && (cl_arg[0] == '-') &&
      (cl_arg[1] == '-')) {
    it = std::find(args.begin(), args.end(), cl_arg.substr(2));
  }
  if (it != args.end()) {
    auto arg_it = it;
    // get next cl_arg
    ++it;
    if (it == args.end()) {
      args.erase(arg_it);
      return default_value;
    }
    const char* arg_val = it->c_str();
    if (arg_val[0] != '[') {
      std::ostringstream error_string;
      error_string << "Unparseable input: " << cl_arg << " " << arg_val
                   << ", must be of the form [int,...,int] with no spaces"
                   << std::endl;
      error(error_string.str());
      exit(1);
    }
    while (strlen(arg_val) > 0 && arg_val[0] != ']') {
      ++arg_val;  // Move past ,
      // convert to int
      tmp = std::strtol(arg_val, &cend, 10);
      // check if cl_arg is actually a int
      if (arg_val == cend) {
        std::ostringstream error_string;
        error_string << "Unparseable input: " << cl_arg << " " << arg_val
                     << ", must be of the form [int,...,int] with no spaces"
                     << std::endl;
        error(error_string.str());
        exit(1);
      }
      // check if int is within bounds
      if (tmp < min || tmp > max) {
        std::ostringstream error_string;
        error_string << "Bad input: " << cl_arg << " " << arg_val
                     << ",  must be in the range (" << min << ", " << max << ")"
                     << std::endl;
        error(error_string.str());
        exit(1);
      }
      vals.push_back(tmp);
      arg_val = cend;
    }
    // Remove argument from list
    args.erase(arg_it, ++it);
    // return index array if everything is OK
    return vals;
  }
  // return default value if not specified on command line
  return default_value;
}

std::vector<std::string> build_arg_list(int argc, char** argv) {
  std::vector<std::string> arg_list(argc - 1);
  for (int i = 1; i < argc; ++i)
    arg_list[i - 1] = argv[i];
  return arg_list;
}

bool check_and_print_unused_args(const std::vector<std::string>& args,
                                 std::ostream& out) {
  if (args.size() == 0)
    return false;

  for (auto arg : args)
    out << arg << " ";
  out << std::endl << std::endl;

  return true;
}
