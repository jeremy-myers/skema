#pragma once
#ifndef SKEMA_ISVD_RESERVOIR_SAMPLER_HPP
#define SKEMA_ISVD_RESERVOIR_SAMPLER_HPP
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <memory>
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <typename MatrixType>
class Sampler {
 public:
  typedef Kokkos::Random_XorShift64_Pool<> pool_type;

  Sampler(const size_type nsamples_,
          const size_type nrow_,
          const size_type ncol_,
          const ordinal_type seed_,
          const ordinal_type print_level_)
      : nsamples(nsamples_),
        nrow(nrow_),
        ncol(ncol_),
        rand_pool(pool_type(seed_)),
        print_level(print_level_) {
    initialized = false;
    count = 0;
    offset = 0;
  };
  virtual ~Sampler() {}
  virtual void sample(const MatrixType& A) = 0;

 protected:
  size_type nsamples;
  size_type nrow;
  size_type ncol;
  pool_type rand_pool;
  ordinal_type print_level;
  bool initialized;
  ordinal_type count;
  ordinal_type offset;
};

template <typename MatrixType>
class ReservoirSampler : public Sampler<MatrixType> {
 public:
  typedef Sampler<MatrixType> base_sampler;
  typedef typename base_sampler::pool_type pool_type;

  ReservoirSampler(const size_type nsamples_,
                   const size_type nrow_,
                   const size_type ncol_,
                   const ordinal_type seed_,
                   const ordinal_type print_level_)
      : Sampler<MatrixType>(nsamples_, nrow_, ncol_, seed_, print_level_) {}
  virtual ~ReservoirSampler(){};

  void sample(const MatrixType& A) override;

  const inline MatrixType& matrix() const { return data; };
  const inline index_type& indices() const { return idxs; };
  struct {
    scalar_type elapsed_time{0.0};
  } stats;

 protected:
  MatrixType data;
  index_type idxs;
};

template class ReservoirSampler<matrix_type>;
template class ReservoirSampler<crs_matrix_type>;

// template <>
// class ReservoirSampler<crs_matrix_type> : public Sampler<crs_matrix_type> {
//  public:
//   ReservoirSampler(const AlgParams& algParams_)
//       : Sampler<crs_matrix_type>(algParams_) {}

//   void initialize() override { idxs = index_type("sample_indices", nsamples);
//   } void sample(const crs_matrix_type& A) override; inline crs_matrix_type&
//   matrix() override { return data; }; inline index_type& indices() override {
//   return idxs; };

//  protected:
//   crs_matrix_type data;
//   index_type idxs;
// };
}  // namespace Skema
#endif /* SKEMA_ISVD_RESERVOIR_SAMPLER_HPP */