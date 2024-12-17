#pragma once
#include <cstddef>
#include <map>
#include "Skema_AlgParams.hpp"
#include "Skema_EIGSVD.hpp"
#include "Skema_Sampler.hpp"
#include "Skema_Utils.hpp"

namespace Skema {
template <typename MatrixType>
class ISVD_SVDS : public PRIMME_SVDS<MatrixType> {
 public:
  ISVD_SVDS(const AlgParams& algParams_)
      : PRIMME_SVDS<MatrixType>(algParams_), algParams(algParams_), count(0) {
    using primme_svds = PRIMME_SVDS<MatrixType>;

    // History & logging
    std::string monitor_filename =
        algParams.outputfilename.filename().stem().string() + "_primme.txt";
    fp_primme_monitor = fopen(monitor_filename.c_str(), "w");
    if (fp_primme_monitor == NULL)
      perror("PRIMME output file failed to open: ");

    std::string json_filename =
        algParams.outputfilename.filename().stem().string() + "_primme.json";
    fp_primme_json = fopen(json_filename.c_str(), "w");
    if (fp_primme_json == NULL)
      perror("PRIMME json file failed to open: ");
    fprintf(fp_primme_json, "{\n");
  };

  ~ISVD_SVDS() {
    if (fp_primme_monitor != NULL)
      fclose(fp_primme_monitor);
    if (fp_primme_json != NULL) {
      fprintf(fp_primme_json, "\n}");
      fclose(fp_primme_json);
    }
  };

  void compute(const MatrixType&,
               const size_type,
               const size_type,
               const size_type,
               matrix_type&,
               vector_type&,
               matrix_type&) override;

  void compute(const MatrixType&,
               const size_type,
               const size_type,
               const size_type,
               matrix_type&,
               vector_type&,
               matrix_type&,
               const ReservoirSampler<MatrixType>&);

  void set_u0(const MatrixType&,
              const size_type,
              const size_type,
              const size_type,
              matrix_type&,
              vector_type&,
              matrix_type&);

  void print_stats(const vector_type&, const vector_type&, primme_svds_params&);

 protected:
  AlgParams algParams;
  FILE* fp_primme_monitor;
  FILE* fp_primme_json;
  int count;
};

template class ISVD_SVDS<matrix_type>;
template class ISVD_SVDS<crs_matrix_type>;

extern "C" {

void isvd_dense_convTestFun(double* sval,
                            void* leftsvec,
                            void* rightvec,
                            double* rnorm,
                            int* method,
                            int* isconv,
                            primme_svds_params* primme_svds,
                            int* ierr);

void isvd_sparse_convTestFun(double* sval,
                             void* leftsvec,
                             void* rightvec,
                             double* rnorm,
                             int* method,
                             int* isconv,
                             primme_svds_params* primme_svds,
                             int* ierr);

void isvd_monitorFun(void* basisSvals,
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
}  // namespace Skema