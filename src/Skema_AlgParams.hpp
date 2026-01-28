#pragma once
#include "Skema_Utils.hpp"
#include <climits>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace Skema {

struct AlgParams {
  size_t matrix_m;
  size_t matrix_n;
  size_t matrix_nnz;
  size_t rank;
  size_t window;
  double normalize_matrix;
  size_t isvd_rank_add_factor;
  double primme_eps;
  double isvd_convtest_eps;
  size_t isvd_convtest_skip;
  size_t isvd_num_samples;
  size_t sketch_range;
  size_t sketch_core;
  double sketch_eta;
  double sketch_nu;
  double kernel_gamma;
  std::vector<int> seeds;
  std::string primme_method;
  std::string primme_methodStage2;
  std::filesystem::path inputfilename;
  std::filesystem::path outputfilename;
  std::filesystem::path history_filename;
  std::filesystem::path debug_filename;
  std::filesystem::path primme_outputFile;
  Solver_Method::type solver;
  Decomposition_Type::type decomposition_type;
  int num_passes;
  int print_level;
  int primme_initSize;
  int primme_maxBasisSize;
  int primme_minRestartSize;
  int primme_maxBlockSize;
  int primme_printLevel;
  int primme_maxMatvecs;
  int primme_maxIter;
  double primme_aNorm;
  DimRedux_Map::type dim_redux;
  Kernel_Map::type kernel_func;
  Sampler_Type::type isvd_sampler;
  bool issparse;
  bool issymmetric;
  bool debug;
  bool hist;
  bool rayleigh_ritz_pass;
  bool isvd_dense_solver;
  bool isvd_initial_guess;
  bool isvd_compute_residual_iters;
  bool primme_locking;
  bool force_three_sketch;
  bool sketch_compute_svals_iters;
  bool isvd_sampling;
  Decomposition_Type::type norm2_solver;

  // Constructor initializing values to defaults
  AlgParams();
  AlgParams(const AlgParams& other) = default;
  AlgParams(AlgParams& other)       = default;

  // Parse options
  void parse(std::vector<std::string>& args);

  // Print options
  void print(std::ostream& out) const;

  static void print_help(std::ostream& out);
};

void error(std::string s);

bool parse_bool(std::vector<std::string>& args, const std::string& cl_arg_on,
                const std::string& cl_off_off, bool default_value);

template <typename T>
T parse_enum(std::vector<std::string>& args, const std::string& cl_arg,
             T default_value, unsigned num_values, const T* values,
             const char* const* names);

template <typename T>
typename T::type parse_enum_helper(const std::string& name);

int parse_int(std::vector<std::string>& args, const std::string& cl_arg,
              int default_value, int min = 0, int max = 100);

std::filesystem::path parse_filepath(std::vector<std::string>& args,
                                     const std::string& cl_arg,
                                     const std::string& default_value);

double parse_real(std::vector<std::string>& args, const std::string& cl_arg,
                  double default_value, double min = 0.0, double max = 1.0);

std::string parse_string(std::vector<std::string>& args,
                         const std::string& cl_arg,
                         const std::string& default_value);

std::vector<int> parse_int_array(std::vector<std::string>& args,
                                 const std::string& cl_arg,
                                 const std::vector<int>& default_value,
                                 int min = 1, int max = INT_MAX);

// Convert (argc,argv) to list of strings
std::vector<std::string> build_arg_list(int argc, char** argv);

// Print out unrecognized command line arguments.  Returns true if there
// are any, false otherwise
bool check_and_print_unused_args(const std::vector<std::string>& args,
                                 std::ostream& out);

}  // namespace Skema
