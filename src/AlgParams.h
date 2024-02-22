#pragma once
#include <cstddef>
#ifndef ALG_PARAMS_H
#define ALG_PARAMS_H

#include <climits>
#include <iostream>
#include <string>
#include <vector>

struct AlgParams {
  // General options
  std::string inputfilename;
  std::string outputfilename_prefix;
  bool issparse;
  bool issymmetric;
  bool isspd;
  size_t matrix_m;
  size_t matrix_n;
  size_t rank;
  std::string solver;
  int print_level;
  int debug_level;
  bool save_U;
  bool save_S;
  bool save_V;
  std::string init_U;
  std::string init_S;
  std::string init_V;
  bool init_dr_map_transpose;

  // Streaming options
  size_t window;

  // FDISVD options
  bool dense_svd_solver;
  std::string primme_matvec;
  double reduce_rank_alpha;
  double dynamic_tol_factor;
  int dynamic_tol_iters;

  // SketchySVD options
  std::string model;
  size_t range;
  size_t core;
  double eta;
  double nu;
  int seed;
  std::vector<int> seeds;

  // PRIMME options
  double primme_eps;
  double primme_convtest_eps;
  int primme_convtest_skipitn;
  int primme_initSize;
  int primme_maxBasisSize;
  int primme_minRestartSize;
  int primme_maxBlockSize;
  int primme_printLevel;
  int primme_maxMatvecs;
  int primme_maxIter;
  int primme_locking;
  std::string primme_method;
  std::string primme_methodStage2;

  // Kernel options
  std::string kernel;
  double gamma;

  // Sampling options
  std::string sampler;
  size_t samples;

  // Resnorm options  
  bool compute_resnorms;
  bool estimate_resnorms;
  bool compute_resnorms_only;

  // Constructor initializing values to defaults
  AlgParams();

  // Parse options
  void parse(std::vector<std::string>& args);

  // Print options
  void print(std::ostream& out) const;
};

void error(std::string s);

int parse_int(std::vector<std::string>& args,
              const std::string& cl_arg,
              int default_value,
              int min = 0,
              int max = 100);

double parse_real(std::vector<std::string>& args,
                  const std::string& cl_arg,
                  double default_value,
                  double min = 0.0,
                  double max = 1.0);

bool parse_bool(std::vector<std::string>& args,
                const std::string& cl_arg_on,
                const std::string& cl_off_off,
                bool default_value);

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

#endif /* ALG_PARAMS_H */