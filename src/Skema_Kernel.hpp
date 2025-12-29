#pragma once
#include <cassert>
#include <cstddef>
#include "Skema_Utils.hpp"

namespace Skema {

struct Kernel_stats {
  scalar_type time{0.0};
  scalar_type elapsed_time{0.0};
};

template <typename MatrixType>
class Kernel {
 public:
  Kernel() : stats_(std::make_shared<Kernel_stats>()) {};
  virtual ~Kernel() {}
  virtual matrix_type compute(const MatrixType&, const size_type,
                              const size_type, const MatrixType&,
                              const size_type, const size_type, const size_type,
                              const range_type) = 0;

  virtual std::shared_ptr<Kernel_stats> stats() { return stats_; };

 protected:
  std::shared_ptr<Kernel_stats> stats_;
};

template <typename MatrixType>
class GaussRBF : public Kernel<MatrixType> {
 public:
  GaussRBF(const size_type nrow_, const size_type ncol_,
           const scalar_type gamma_)
      : nrow(nrow_), ncol(ncol_), gamma(gamma_) {
    data = matrix_type("Gauss RBF kernel matrix", nrow, ncol);
  };
  ~GaussRBF() {};

  matrix_type compute(const MatrixType&, const size_type, const size_type,
                      const MatrixType&, const size_type, const size_type,
                      const size_type, const range_type) override;

 protected:
  matrix_type data;
  size_type nrow;
  size_type ncol;
  scalar_type gamma;

 private:
  void enforce_unit_diagonal(const size_type, const size_type, const size_type);
};

template class GaussRBF<matrix_type>;
template class GaussRBF<crs_matrix_type>;
}  // namespace Skema