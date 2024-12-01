#include "Skema_AlgParams.hpp"
#include <algorithm>
#include <climits>
#include <cstring>
#include <ios>
#include <limits>
#include <sstream>
#include "Skema_Utils.hpp"

AlgParams::AlgParams()
    : solver(Skema::Solver_Method::default_type),
      decomposition_type(Skema::Decomposition_Type::default_type),
      inputfilename(""),
      outputfilename(""),
      debug_filename(""),
      issparse(false),
      issymmetric(false),
      matrix_m(0),
      matrix_n(0),
      rank(1),
      print_level(0),
      debug(false),
      window(1),
      hist(true),
      isvd_dense_solver(false),
      isvd_sampler(Skema::Sampler_Type::default_type),
      isvd_num_samples(0),
      isvd_sampling(false),
      isvd_convtest_eps(1e-4),
      isvd_convtest_skip(0),
      isvd_initial_guess(false),
      primme_eps(1e-4),
      primme_initSize(0),
      primme_maxBasisSize(0),
      primme_minRestartSize(0),
      primme_maxBlockSize(0),
      primme_printLevel(0),
      primme_maxMatvecs(0),
      primme_maxIter(0),
      primme_locking(false),
      primme_method("PRIMME_DEFAULT_METHOD"),
      primme_methodStage2("PRIMME_DEFAULT_METHOD"),
      sketch_range(1),
      sketch_core(1),
      dim_redux(Skema::DimRedux_Map::default_type),
      seeds({0, 1, 2, 3}),
      kernel_func(Skema::Kernel_Map::default_type),
      kernel_gamma(1.0) {}

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
  out << "\n";
  out << "  rank = " << rank << std::endl;
  out << "  window size = " << window << std::endl;
  out << "  solver = " << Skema::Solver_Method::names[solver] << std::endl;

  switch (Skema::Solver_Method::types[solver]) {
    case Skema::Solver_Method::ISVD:
      if (isvd_num_samples > 0) {
        out << "  iSVD sampler = " << isvd_sampler << std::endl;
        out << "  iSVD num samples = " << isvd_num_samples << std::endl;
        out << "  iSVD conv test func eps = " << isvd_convtest_eps << std::endl;
        out << "  iSVD conv test skip = " << isvd_convtest_skip << std::endl;
      }
      std::cout << "  iSVD initial guess = " << std::boolalpha
                << isvd_initial_guess << std::endl;
      break;

    case Skema::Solver_Method::SKETCH:
      out << "  range = " << sketch_range << std::endl;
      out << "  core = " << sketch_core << std::endl;
      out << "  eta = " << sketch_eta << std::endl;
      out << "  nu = " << sketch_nu << std::endl;
      out << "  model = " << Skema::DimRedux_Map::names[dim_redux] << std::endl;
      break;

    case Skema::Solver_Method::PRIMME:
      out << "    PRIMME method = " << primme_method << std::endl;
      out << "    PRIMME methodStage2 = " << primme_methodStage2 << std::endl;
      out << "    PRIMME printLevel = " << primme_printLevel << std::endl;
      out << "    PRIMME tolerance = " << primme_eps << std::endl;
      out << "    PRIMME locking = " << std::boolalpha << primme_locking
          << std::endl;
      if (isvd_num_samples > 0) {
        out << "    PRIMME conv test func eps = " << isvd_convtest_eps
            << std::endl;
        out << "    PRIMME conv test skip = " << isvd_convtest_skip
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
        out << "    PRIMME maxBlockSize = " << primme_maxBlockSize << std::endl;
      }
      if (primme_minRestartSize > 0) {
        out << "    PRIMME minRestartSize = " << primme_minRestartSize
            << std::endl;
        ;
      }
      break;
  }

  if (Skema::Kernel_Map::types[kernel_func] != Skema::Kernel_Map::type::NONE) {
    out << "  kernel = " << Skema::Kernel_Map::types[kernel_func] << std::endl;
    out << "  gamma = " << kernel_gamma << std::endl;
  }
}

void AlgParams::parse(std::vector<std::string>& args) {
  /* Parse options from command-line, using default values set above as defaults
   */
  // Generic options
  inputfilename = parse_filepath(args, "--input", "");
  outputfilename = parse_filepath(args, "--output", "");
  debug_filename = parse_filepath(args, "--debug-file", "");
  issparse = parse_bool(args, "--sparse", "--dense", false);
  issymmetric = parse_bool(args, "--symmetric", "--asymmetric", false);
  matrix_m = parse_int(args, "--m", matrix_m, 0, INT_MAX);
  matrix_n = parse_int(args, "--n", matrix_n, 0, INT_MAX);
  rank = parse_int(args, "--rank", rank, 1, INT_MAX);
  solver = parse_enum(args, "--solver", Skema::Solver_Method::default_type,
                      Skema::Solver_Method::num_types,
                      Skema::Solver_Method::types, Skema::Solver_Method::names);
  print_level = parse_int(args, "--print-level", print_level, 0, 5);
  debug = parse_bool(args, "--debug", "--debug-off", false);

  // Streaming options
  window = parse_int(args, "--window", window, 1, INT_MAX);

  // SketchySVD options
  dim_redux =
      parse_enum(args, "--model", Skema::DimRedux_Map::default_type,
                 Skema::DimRedux_Map::num_types, Skema::DimRedux_Map::types,
                 Skema::DimRedux_Map::names);
  sketch_range = parse_int(args, "--range", sketch_range, 0, INT_MAX);
  sketch_core = parse_int(args, "--core", sketch_core, 0, INT_MAX);
  seeds = parse_int_array(args, "--seeds", seeds, 0, INT_MAX);
  sketch_eta =
      parse_real(args, "--eta", 1.0, 0.0, std::numeric_limits<double>::max());
  sketch_nu =
      parse_real(args, "--nu", 1.0, 0.0, std::numeric_limits<double>::max());

  // iSVD solver options
  isvd_dense_solver = parse_bool(args, "--svd", "--svds", false);
  isvd_sampler =
      parse_enum(args, "--isvd-sampler", Skema::Sampler_Type::default_type,
                 Skema::Sampler_Type::num_types, Skema::Sampler_Type::types,
                 Skema::Sampler_Type::names);
  isvd_num_samples = parse_int(args, "--isvd-num-samples", 0, 0, INT_MAX);
  isvd_convtest_eps = parse_real(args, "--isvd-convtest-eps", isvd_convtest_eps,
                                 std::numeric_limits<double>::epsilon(), 1.0);
  isvd_convtest_skip = parse_int(args, "--isvd-convtest-skip", 0, 0, INT_MAX);
  isvd_initial_guess = parse_bool(args, "--isvd-initial-guess",
                                  "--isvd-initial-guess-off", false);

  // PRIMME solver options
  primme_printLevel = parse_int(args, "--primme_printLevel", 0, 0, 5);
  primme_eps = parse_real(args, "--primme_eps", primme_eps,
                          std::numeric_limits<double>::epsilon(), 1.0);
  primme_initSize = parse_int(args, "--primme_initSize", 0, 0, INT_MAX);
  primme_maxBasisSize = parse_int(args, "--primme_maxBasisSize", 0, 0, INT_MAX);
  primme_minRestartSize =
      parse_int(args, "--primme_minRestartSize", 0, 0, INT_MAX);
  primme_maxBlockSize = parse_int(args, "--primme_maxBlockSize", 0, 0, INT_MAX);
  primme_maxMatvecs = parse_int(args, "--primme_maxMatvecs", 0, 0, INT_MAX);
  primme_maxIter = parse_int(args, "--primme_maxIters", 0, 0, INT_MAX);
  primme_locking =
      parse_bool(args, "--primme_locking", "--primme_locking-off", false);
  primme_method =
      parse_string(args, "--primme_method", "PRIMME_DEFAULT_METHOD");
  primme_methodStage2 =
      parse_string(args, "--prime_methodStage2", "PRIMME_DEFAULT_METHOD");

  // Kernel options
  kernel_func = parse_enum(args, "--kernel", Skema::Kernel_Map::default_type,
                           Skema::Kernel_Map::num_types,
                           Skema::Kernel_Map::types, Skema::Kernel_Map::names);
  kernel_gamma = parse_real(args, "--gamma", kernel_gamma, 0.0,
                            std::numeric_limits<double>::max());

  if (isvd_num_samples > 0)
    isvd_sampling = true;
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

template <typename T>
T parse_enum(std::vector<std::string>& args,
             const std::string& cl_arg,
             T default_value,
             unsigned num_values,
             const T* values,
             const char* const* names) {
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
    // convert to string
    std::string arg_val = *it;
    // Remove argument from list
    args.erase(arg_it, ++it);
    // find name in list of names
    for (unsigned i = 0; i < num_values; ++i) {
      if (arg_val == names[i])
        return values[i];
    }
    // if we got here, name wasn't found
    std::ostringstream error_string;
    error_string << "Bad input: " << cl_arg << " " << arg_val
                 << ",  must be one of the values: ";
    for (unsigned i = 0; i < num_values; ++i) {
      error_string << names[i];
      if (i != num_values - 1)
        error_string << ", ";
    }
    error_string << "." << std::endl;
    error(error_string.str());
    exit(1);
  }
  // return default value if not specified on command line
  return default_value;
}

template <typename T>
typename T::type parse_enum_helper(const std::string& name) {
  for (unsigned i = 0; i < T::num_types; ++i) {
    if (name == T::names[i])
      return T::types[i];
  }

  std::ostringstream error_string;
  error_string << "Invalid enum choice " << name
               << ",  must be one of the values: ";
  for (unsigned i = 0; i < T::num_types; ++i) {
    error_string << T::names[i];
    if (i != T::num_types - 1)
      error_string << ", ";
  }
  error_string << "." << std::endl;
  error(error_string.str());
  return T::default_type;
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

std::filesystem::path parse_filepath(std::vector<std::string>& args,
                                     const std::string& cl_arg,
                                     const std::string& default_value) {
  std::string path_str = parse_string(args, cl_arg, default_value);
  return std::filesystem::path(path_str);
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
