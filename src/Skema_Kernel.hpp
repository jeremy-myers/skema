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
class KernelNew {
 public:
  KernelNew(){};
  virtual ~KernelNew() {}
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
class GaussRBFNew : public KernelNew<MatrixType> {
 public:
  GaussRBFNew() : gamma(0.0){};
  GaussRBFNew(const scalar_type gamma_) : gamma(gamma_){};
  virtual ~GaussRBFNew(){};

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

template class GaussRBFNew<matrix_type>;
template class GaussRBFNew<crs_matrix_type>;
}  // namespace Skema
#endif /* SKEMA_KERNEL_H */