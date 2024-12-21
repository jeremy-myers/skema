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
  virtual MatrixType compute(const MatrixType&,
                             const size_type,
                             const size_type,
                             const MatrixType&,
                             const size_type,
                             const size_type,
                             const size_type,
                             const range_type) = 0;

  virtual std::shared_ptr<Kernel_stats> stats() { return stats_; };

 protected:
  std::shared_ptr<Kernel_stats> stats_;
};

template <typename MatrixType>
class GaussRBF : public Kernel<MatrixType> {
 public:
  GaussRBF() : gamma(0.0) {};
  GaussRBF(const scalar_type gamma_) : gamma(gamma_) {};
  virtual ~GaussRBF() {};

  MatrixType compute(const MatrixType&,
                     const size_type,
                     const size_type,
                     const MatrixType&,
                     const size_type,
                     const size_type,
                     const size_type,
                     const range_type) override;

  inline void set_gamma(const scalar_type gamma_) { gamma = gamma_; };

 protected:
  MatrixType data;
  scalar_type gamma;
};

template class GaussRBF<matrix_type>;
template class GaussRBF<crs_matrix_type>;
}  // namespace Skema