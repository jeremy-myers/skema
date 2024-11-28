#pragma once
#include <climits>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include "Skema_Utils.hpp"

struct AlgParams {
  // General options
  Skema::Solver_Method::type solver;
  Skema::Decomposition_Type::type decomposition_type;
  // std::string inputfilename;
  // std::string outputfilename;
  std::filesystem::path inputfilename;
  std::filesystem::path outputfilename;
  bool issparse;
  bool issymmetric;
  size_t matrix_m;
  size_t matrix_n;
  size_t matrix_nnz;
  size_t rank;
  int print_level;
  bool debug;
  size_t window;
  bool hist;

  /* ISVD options */
  bool isvd_dense_solver;
  bool isvd_initial_guess;

  // ISVD: PRIMME options
  double primme_eps;
  double isvd_convtest_eps;
  int isvd_convtest_skip;
  int primme_initSize;
  int primme_maxBasisSize;
  int primme_minRestartSize;
  int primme_maxBlockSize;
  int primme_printLevel;
  int primme_maxMatvecs;
  int primme_maxIter;
  bool primme_locking;
  std::string primme_method;
  std::string primme_methodStage2;

  // ISVD: Sampling options
  Skema::Sampler_Type::type isvd_sampler;
  bool isvd_sampling;
  size_t isvd_num_samples;

  /* Sketch options */
  size_t sketch_range;
  size_t sketch_core;
  double sketch_eta;
  double sketch_nu;
  std::vector<int> seeds;

  /* DimRedux */
  Skema::DimRedux_Map::type dim_redux;

  /* Kernel options */
  Skema::Kernel_Map::type kernel_func;
  double kernel_gamma;

  // Constructor initializing values to defaults
  AlgParams();

  // Parse options
  void parse(std::vector<std::string>& args);

  // Print options
  void print(std::ostream& out) const;
};

void error(std::string s);

bool parse_bool(std::vector<std::string>& args,
                const std::string& cl_arg_on,
                const std::string& cl_off_off,
                bool default_value);

template <typename T>
T parse_enum(std::vector<std::string>& args,
             const std::string& cl_arg,
             T default_value,
             unsigned num_values,
             const T* values,
             const char* const* names);

template <typename T>
typename T::type parse_enum_helper(const std::string& name);

int parse_int(std::vector<std::string>& args,
              const std::string& cl_arg,
              int default_value,
              int min = 0,
              int max = 100);

std::filesystem::path parse_filepath(std::vector<std::string>& args,
                                     const std::string& cl_arg,
                                     const std::string& default_value);

double parse_real(std::vector<std::string>& args,
                  const std::string& cl_arg,
                  double default_value,
                  double min = 0.0,
                  double max = 1.0);

std::string parse_string(std::vector<std::string>& args,
                         const std::string& cl_arg,
                         const std::string& default_value);

std::vector<int> parse_int_array(std::vector<std::string>& args,
                                 const std::string& cl_arg,
                                 const std::vector<int>& default_value,
                                 int min = 1,
                                 int max = INT_MAX);

// Convert (argc,argv) to list of strings
std::vector<std::string> build_arg_list(int argc, char** argv);

// Print out unrecognized command line arguments.  Returns true if there
// are any, false otherwise
bool check_and_print_unused_args(const std::vector<std::string>& args,
                                 std::ostream& out);