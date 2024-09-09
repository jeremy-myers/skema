#pragma once
#ifndef SKEMA_EIGSVD_HPP
#define SKEMA_EIGSVD_HPP
#include "Skema_AlgParams.hpp"
#include "Skema_Sampler.hpp"
#include "Skema_Utils.hpp"
#include "primme.h"
#include <lapacke.h>

namespace Skema {

extern "C" {
void svds_monitorFun(void *basisSvals, int *basisSize, int *basisFlags,
                     int *iblock, int *blockSize, void *basisNorms,
                     int *numConverged, void *lockedSvals, int *numLocked,
                     int *lockedFlags, void *lockedNorms, int *inner_its,
                     void *LSRes, const char *msg, double *time,
                     primme_event *event, int *stage,
                     primme_svds_params *primme_svds, int *ierr);
}

template <typename MatrixType> class XVDS {
public:
  XVDS(){};
  virtual ~XVDS(){};

  virtual void compute(const MatrixType &, const size_type, const size_type,
                       const size_type, matrix_type &, vector_type &,
                       matrix_type &) = 0;

  virtual void reinitialize() = 0;
};

template <typename MatrixType> class PRIMME_SVDS : public XVDS<MatrixType> {
public:
  PRIMME_SVDS(const AlgParams &algParams_) : algParams(algParams_) {
    primme_svds_initialize(&params);
    params.monitorFun = svds_monitorFun;
  };

  ~PRIMME_SVDS() { primme_svds_free(&params); };

  void compute(const MatrixType &, const size_type, const size_type,
               const size_type, matrix_type &, vector_type &,
               matrix_type &) override;

  inline void reinitialize() override {
    primme_svds_initialize(&params);
    params.monitorFun = svds_monitorFun;
  }

protected:
  primme_svds_params params;
  AlgParams algParams;
};

template class PRIMME_SVDS<matrix_type>;
template class PRIMME_SVDS<crs_matrix_type>;

extern "C" {
inline void svds_monitorFun(void *basisSvals, int *basisSize, int *basisFlags,
                            int *iblock, int *blockSize, void *basisNorms,
                            int *numConverged, void *lockedSvals,
                            int *numLocked, int *lockedFlags, void *lockedNorms,
                            int *inner_its, void *LSRes, const char *msg,
                            double *time, primme_event *event, int *stage,
                            primme_svds_params *primme_svds, int *ierr) {
  assert(event != NULL && primme_svds != NULL);

  if (primme_svds->outputFile &&
      (primme_svds->procID == 0 || *event == primme_event_profile)) {
    switch (*event) {
    case primme_event_outer_iteration:
      assert(basisSize && (!*basisSize || (basisSvals && basisFlags)) &&
             blockSize && (!*blockSize || (iblock && basisNorms)) &&
             numConverged);
      for (int i = 0; i < *blockSize; ++i) {
        fprintf(primme_svds->outputFile,
                "OUT %lld blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
                "|r| %.16f\n",
                primme_svds->primme.stats.numOuterIterations, iblock[i],
                primme_svds->primme.stats.numMatvecs,
                primme_svds->primme.stats.elapsedTime,
                primme_svds->primme.stats.timeMatvec,
                primme_svds->primme.stats.timeOrtho,
                ((double *)basisSvals)[iblock[i]],
                ((double *)basisNorms)[iblock[i]]);
      }
      break;
    case primme_event_converged:
      assert(numConverged && iblock && basisSvals && basisNorms);
      fprintf(primme_svds->outputFile,
              "#Converged %d blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
              "|r| %.16f\n",
              *numConverged, iblock[0], primme_svds->primme.stats.numMatvecs,
              primme_svds->primme.stats.elapsedTime,
              primme_svds->primme.stats.timeMatvec,
              primme_svds->primme.stats.timeOrtho,
              ((double *)basisSvals)[iblock[0]],
              ((double *)basisNorms)[iblock[0]]);
      break;
    default:
      break;
    }
    fflush(primme_svds->outputFile);
  }
  *ierr = 0;
}
}

template <typename MatrixType>
void primme_svds(const MatrixType &, const AlgParams &);

} // namespace Skema

#endif /* SKEMA_EIGSVD_HPP */