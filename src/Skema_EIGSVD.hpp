#pragma once
#include "Skema_AlgParams.hpp"
#include "Skema_Utils.hpp"
#include "primme.h"

namespace Skema {

extern "C" {

void eigs_monitorFun(void* basisEvals,
                     int* basisSize,
                     int* basisFlags,
                     int* iblock,
                     int* blockSize,
                     void* basisNorms,
                     int* numConverged,
                     void* lockedEvals,
                     int* numLocked,
                     int* lockedFlags,
                     void* lockedNorms,
                     int* inner_its,
                     void* LSRes,
                     const char* msg,
                     double* time,
                     primme_event* event,
                     struct primme_params* primme,
                     int* ierr);

void svds_monitorFun(void* basisSvals,
                     int* basisSize,
                     int* basisFlags,
                     int* iblock,
                     int* blockSize,
                     void* basisNorms,
                     int* numConverged,
                     void* lockedSvals,
                     int* numLocked,
                     int* lockedFlags,
                     void* lockedNorms,
                     int* inner_its,
                     void* LSRes,
                     const char* msg,
                     double* time,
                     primme_event* event,
                     int* stage,
                     primme_svds_params* primme_svds,
                     int* ierr);
}

struct XVDS_stats {
  int64_t numOuterIterations{0};
  int64_t numMatvecs{0};
  double elapsedTime{0.0};
  double timeMatvec{0.0};
  double timeOrtho{0.0};
};

template <typename MatrixType>
class XVDS {
 public:
  XVDS() {};
  virtual ~XVDS() {};

  virtual void compute(const MatrixType&,
                       const size_type,
                       const size_type,
                       const size_type,
                       matrix_type&,
                       vector_type&,
                       matrix_type&) = 0;

  virtual void reinitialize() = 0;

  virtual std::shared_ptr<XVDS_stats> stats() = 0;
};

template <typename MatrixType>
class PRIMME_EIGS : public XVDS<MatrixType> {
 public:
  PRIMME_EIGS(const AlgParams& algParams_)
      : algParams(algParams_), primme_stats(std::make_shared<XVDS_stats>()) {
    primme_initialize(&params);
  };

  ~PRIMME_EIGS() {
    primme_free(&params);
    params.monitorFun = eigs_monitorFun;
  };

  void compute(const MatrixType&,
               const size_type,
               const size_type,
               const size_type,
               matrix_type&,
               vector_type&,
               matrix_type&) override;

  inline void reinitialize() override {
    primme_initialize(&params);
    params.monitorFun = eigs_monitorFun;
  }

  inline std::shared_ptr<XVDS_stats> stats() override {
    primme_stats->numOuterIterations = params.stats.numOuterIterations;
    primme_stats->numMatvecs = params.stats.numMatvecs;
    primme_stats->elapsedTime = params.stats.elapsedTime;
    primme_stats->timeMatvec = params.stats.timeMatvec;
    primme_stats->timeOrtho = params.stats.timeOrtho;
    return primme_stats;
  };

 protected:
  primme_params params;
  AlgParams algParams;
  std::shared_ptr<XVDS_stats> primme_stats;
};

template <typename MatrixType>
class PRIMME_SVDS : public XVDS<MatrixType> {
 public:
  PRIMME_SVDS(const AlgParams& algParams_)
      : algParams(algParams_), primme_stats(std::make_shared<XVDS_stats>()) {
    primme_svds_initialize(&params);
    params.monitorFun = svds_monitorFun;
  };

  ~PRIMME_SVDS() { primme_svds_free(&params); };

  void compute(const MatrixType&,
               const size_type,
               const size_type,
               const size_type,
               matrix_type&,
               vector_type&,
               matrix_type&) override;

  inline void reinitialize() override {
    primme_svds_initialize(&params);
    params.monitorFun = svds_monitorFun;
  }

  inline std::shared_ptr<XVDS_stats> stats() override {
    primme_stats->numOuterIterations = params.stats.numOuterIterations;
    primme_stats->numMatvecs = params.stats.numMatvecs;
    primme_stats->elapsedTime = params.stats.elapsedTime;
    primme_stats->timeMatvec = params.stats.timeMatvec;
    primme_stats->timeOrtho = params.stats.timeOrtho;
    return primme_stats;
  };

 protected:
  primme_svds_params params;
  AlgParams algParams;
  std::shared_ptr<XVDS_stats> primme_stats;
};

template class PRIMME_SVDS<matrix_type>;
template class PRIMME_SVDS<crs_matrix_type>;

extern "C" {
inline void eigs_monitorFun(void* basisEvals,
                            int* basisSize,
                            int* basisFlags,
                            int* iblock,
                            int* blockSize,
                            void* basisNorms,
                            int* numConverged,
                            void* lockedEvals,
                            int* numLocked,
                            int* lockedFlags,
                            void* lockedNorms,
                            int* inner_its,
                            void* LSRes,
                            const char* msg,
                            double* time,
                            primme_event* event,
                            struct primme_params* primme,
                            int* ierr) {
  assert(event != NULL && primme != NULL);

  if (primme->outputFile &&
      (primme->procID == 0 || *event == primme_event_profile)) {
    switch (*event) {
      case primme_event_outer_iteration:
        assert(basisSize && (!*basisSize || (basisEvals && basisFlags)) &&
               blockSize && (!*blockSize || (iblock && basisNorms)) &&
               numConverged);
        for (int i = 0; i < *blockSize; ++i) {
          fprintf(primme->outputFile,
                  "OUT %lld blk %d MV %lld Sec %E tMV %E tORTH %E SV "
                  "%.16f "
                  "|r| %.16f\n",
                  primme->stats.numOuterIterations, iblock[i],
                  primme->stats.numMatvecs, primme->stats.elapsedTime,
                  primme->stats.timeMatvec, primme->stats.timeOrtho,
                  ((double*)basisEvals)[iblock[i]],
                  ((double*)basisNorms)[iblock[i]]);
        }
        break;
      case primme_event_converged:
        assert(numConverged && iblock && basisEvals && basisNorms);
        fprintf(primme->outputFile,
                "#Converged %d blk %d MV %lld Sec %E tMV %E tORTH %E SV "
                "%.16f "
                "|r| %.16f\n",
                *numConverged, iblock[0], primme->stats.numMatvecs,
                primme->stats.elapsedTime, primme->stats.timeMatvec,
                primme->stats.timeOrtho, ((double*)basisEvals)[iblock[0]],
                ((double*)basisNorms)[iblock[0]]);
        break;
      default:
        break;
    }
    fflush(primme->outputFile);
  }
  *ierr = 0;
}

inline void svds_monitorFun(void* basisSvals,
                            int* basisSize,
                            int* basisFlags,
                            int* iblock,
                            int* blockSize,
                            void* basisNorms,
                            int* numConverged,
                            void* lockedSvals,
                            int* numLocked,
                            int* lockedFlags,
                            void* lockedNorms,
                            int* inner_its,
                            void* LSRes,
                            const char* msg,
                            double* time,
                            primme_event* event,
                            int* stage,
                            primme_svds_params* primme_svds,
                            int* ierr) {
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
                  ((double*)basisSvals)[iblock[i]],
                  ((double*)basisNorms)[iblock[i]]);
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
                ((double*)basisSvals)[iblock[0]],
                ((double*)basisNorms)[iblock[0]]);
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
void primme_eigs(const MatrixType&, const AlgParams&);

template <typename MatrixType>
void primme_svds(const MatrixType&, const AlgParams&);

}  // namespace Skema