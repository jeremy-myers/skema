#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_Timer.hpp>
#include <cstddef>
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
  Sampler(const Sampler&) = default;
  Sampler(Sampler&&) = default;
  Sampler& operator=(const Sampler&);
  Sampler& operator=(Sampler&&);
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
  ReservoirSampler(const ReservoirSampler&) = default;
  ReservoirSampler(ReservoirSampler&&) = default;
  ReservoirSampler& operator=(const ReservoirSampler&);
  ReservoirSampler& operator=(ReservoirSampler&&);
  virtual ~ReservoirSampler() {}

  void sample(const MatrixType& A) override;

  const inline MatrixType& matrix() const { return data; };
  const inline index_type& indices() const { return idxs; };
  const inline ordinal_type indices(const int i) const { return idxs(i); };
  struct {
    scalar_type elapsed_time{0.0};
  } stats;

 protected:
  MatrixType data;
  index_type idxs;
};

template class ReservoirSampler<matrix_type>;
template class ReservoirSampler<crs_matrix_type>;
}  // namespace Skema