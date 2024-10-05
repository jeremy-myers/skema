#pragma once
#ifndef SKEMA_KERNEL_H
#define SKEMA_KERNEL_H
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_Utils.hpp"

namespace Skema {
template <typename MatrixType>
class Kernel {
 public:
  Kernel(){};
  virtual ~Kernel() {}
  virtual MatrixType compute(const MatrixType&,
                             const size_type,
                             const size_type,
                             const MatrixType&,
                             const size_type,
                             const size_type,
                             const size_type,
                             const range_type) = 0;
};

template <typename MatrixType>
class GaussRBF : public Kernel<MatrixType> {
 public:
  GaussRBF() : gamma(0.0){};
  GaussRBF(const scalar_type gamma_) : gamma(gamma_){};
  virtual ~GaussRBF(){};

  struct {
    scalar_type time{0.0};
    scalar_type elapsed_time{0.0};
  } stats;

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
#endif /* SKEMA_KERNEL_H */