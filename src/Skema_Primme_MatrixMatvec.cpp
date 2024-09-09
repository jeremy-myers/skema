#include "Skema_Primme_MatrixMatvec.hpp"
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cstddef>
#include <iomanip>
#include <utility>
#include "Skema_AlgParams.hpp"
#include "Skema_Common.hpp"
#include "Skema_Kernel.hpp"
#include "primme.h"

extern "C" {

using scalar_type = default_scalar;
using ordinal_type = default_lno_t;
using size_type = default_size_type;
using layout_type = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;
using execution_space = typename device_type::execution_space;
using memory_space = typename device_type::memory_space;
using matrix_type = typename Kokkos::View<scalar_type**, layout_type>;
using unmanaged_matrix_type =
    typename Kokkos::View<scalar_type**,
                          Kokkos::LayoutLeft,
                          Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
using crs_matrix_type = typename KokkosSparse::
    CrsMatrix<scalar_type, ordinal_type, device_type, void, size_type>;
using vector_type = typename Kokkos::View<scalar_type*, layout_type>;
using range_type = typename std::pair<size_type, size_type>;

void primme_eigs_default_sparse_matvec(void* x,
                                       PRIMME_INT* ldx,
                                       void* y,
                                       PRIMME_INT* ldy,
                                       int* blockSize,
                                       primme_params* primme_svds,
                                       int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const crs_matrix_type spmatrix = *(crs_matrix_type*)primme_svds->matrix;
    const size_t n{static_cast<size_t>(primme_svds->n)};

    unmanaged_matrix_type x_view0((double*)x, static_cast<size_t>(*ldx),
                                  static_cast<size_t>(*blockSize));
    unmanaged_matrix_type y_view0((double*)y, static_cast<size_t>(*ldy),
                                  static_cast<size_t>(*blockSize));

    auto x_view = Kokkos::subview(x_view0, Kokkos::make_pair<size_type>(0, n),
                                  Kokkos::ALL());
    auto y_view = Kokkos::subview(y_view0, Kokkos::make_pair<size_type>(0, n),
                                  Kokkos::ALL());

    KokkosSparse::spmv("N", 1.0, spmatrix, x_view, 0.0, y_view);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout
        << "Skema::primme_eigs_default_sparse_matvec encountered an exception: "
        << e.what() << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}

void primme_eigs_default_monitorFun(void* basisEvals,
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
                                    primme_params* primme,
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
                  "OUT %lld blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
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
                "#Converged %d blk %d MV %lld Sec %E tMV %E tORTH %E SV %.16f "
                "|r| %.16f\n",
                *numConverged, iblock[0], primme->stats.numMatvecs,
                primme->stats.elapsedTime, primme->stats.timeMatvec,
                primme->stats.timeOrtho, ((double*)basisEvals)[iblock[0]],
                ((double*)basisNorms)[iblock[0]]);
        break;
      case primme_event_locked:
        fprintf(primme->outputFile,
                "Lock striplet[ %d ]= %e norm %.4e Mvecs %" PRIMME_INT_P
                " Time %.4e Flag %d\n",
                *numLocked - 1, ((double*)lockedEvals)[*numLocked - 1],
                ((double*)lockedNorms)[*numLocked - 1],
                primme->stats.numMatvecs, primme->stats.elapsedTime,
                lockedFlags[*numLocked - 1]);

        break;
      default:
        break;
    }
    fflush(primme->outputFile);
  }
  *ierr = 0;
}

// struct kernel_gaussrbf {
//   scalar_type* data;
//   size_type nfeat;
//   GaussRBF* kptr;
//   size_type window_size;
// };

// void primme_eigs_kernel_gaussrbf_matvec(void* x,
//                                         PRIMME_INT* ldx,
//                                         void* y,
//                                         PRIMME_INT* ldy,
//                                         int* blockSize,
//                                         primme_params* primme,
//                                         int* err) {
//   ///! Capture exceptions here; don't propagate them to C code
//   try {
//     kernel_gaussrbf kernel = *(kernel_gaussrbf*)primme->matrix;

//     const size_type nfeat = kernel.nfeat;
//     const size_type wsize = kernel.window_size;
//     range_type row_range;

//     const size_type nrow{static_cast<size_type>(primme->n)};
//     const size_type ncol{static_cast<size_type>(primme->n)};
//     const size_type lldx{static_cast<size_type>(*ldx)};
//     const size_type lldy{static_cast<size_type>(*ldy)};
//     const size_type lbsz{static_cast<size_type>(*blockSize)};

//     unmanaged_matrix_type x_view0((scalar_type*)x, lldx, lbsz);
//     unmanaged_matrix_type y_view0((scalar_type*)y, lldy, lbsz);
//     const unmanaged_matrix_type A_view((scalar_type*)kernel.data, nrow, nfeat);

//     Kokkos::Timer timer;
//     for (auto irow = 0; irow < nrow; irow += wsize) {
//       // Get row range of current tile
//       if (irow + wsize < nrow) {
//         row_range = std::make_pair(irow, irow + wsize);
//       } else {
//         row_range = std::make_pair(irow, nrow);
//       }

//       // Compute tile
//       matrix_type A_sub(A_view, row_range, Kokkos::ALL());
//       kernel.kptr->compute(A_sub, A_view, row_range);

//       // Perform matvec
//       auto y_view = Kokkos::subview(y_view0, row_range, Kokkos::ALL());
//       KokkosBlas::gemm("N", "N", 1.0, kernel.kptr->matrix(), x_view0, 0.0,
//                        y_view);
//     }

//     std::cout << "i " << primme->stats.numOuterIterations << " Sec "
//               << timer.seconds() << std::endl;

//     *err = 0;
//   } catch (const std::exception& e) {
//     fprintf(primme->outputFile,
//             "gauss_rbf_matvec encountered an exception: %s\n", e.what());
//     std::cout << "gauss_rbf_matvec encountered an exception: " << e.what()
//               << std::endl;
//     *err = 1;  ///! notify to primme that something went wrong
//   }
// }

// void primme_eigs_kernel_monitorFun(void* basisEvals,
//                                    int* basisSize,
//                                    int* basisFlags,
//                                    int* iblock,
//                                    int* blockSize,
//                                    void* basisNorms,
//                                    int* numConverged,
//                                    void* lockedEvals,
//                                    int* numLocked,
//                                    int* lockedFlags,
//                                    void* lockedNorms,
//                                    int* inner_its,
//                                    void* LSRes,
//                                    const char* msg,
//                                    double* time,
//                                    primme_event* event,
//                                    struct primme_params* primme,
//                                    int* ierr) {
//   assert(event != NULL && primme != NULL);

//   if (primme->outputFile &&
//       (primme->procID == 0 || *event == primme_event_profile)) {
//     kernel_gaussrbf kernel = *(kernel_gaussrbf*)primme->matrix;
//     switch (*event) {
//       case primme_event_outer_iteration:
//         assert(basisSize && (!*basisSize || (basisEvals && basisFlags)) &&
//                blockSize && (!*blockSize || (iblock && basisNorms)) &&
//                numConverged);
//         for (int i = 0; i < *blockSize; ++i) {
//           fprintf(
//               primme->outputFile,
//               "OUT %lld blk %d MV %lld Sec %E tKernel %E tMV %E tORTH %E SV "
//               "%.16f "
//               "|r| %.16f\n",
//               primme->stats.numOuterIterations, iblock[i],
//               primme->stats.numMatvecs, primme->stats.elapsedTime,
//               kernel.kptr->stats.elapsed_time,
//               (primme->stats.timeMatvec - kernel.kptr->stats.elapsed_time),
//               primme->stats.timeOrtho, ((double*)basisEvals)[iblock[i]],
//               ((double*)basisNorms)[iblock[i]]);
//         }
//         break;
//       case primme_event_converged:
//         assert(numConverged && iblock && basisEvals && basisNorms);
//         fprintf(
//             primme->outputFile,
//             "#Converged %d blk %d MV %lld Sec %E tKernel %E tMV %E tORTH %E SV "
//             "%.16f "
//             "|r| %.16f\n",
//             *numConverged, iblock[0], primme->stats.numMatvecs,
//             primme->stats.elapsedTime, kernel.kptr->stats.elapsed_time,
//             (primme->stats.timeMatvec - kernel.kptr->stats.elapsed_time),
//             primme->stats.timeOrtho, ((double*)basisEvals)[iblock[0]],
//             ((double*)basisNorms)[iblock[0]]);
//         break;
//       default:
//         break;
//     }
//     fflush(primme->outputFile);
//   }
//   *ierr = 0;
// }

void primme_svds_default_dense_matvec(void* x,
                                      PRIMME_INT* ldx,
                                      void* y,
                                      PRIMME_INT* ldy,
                                      int* blockSize,
                                      int* transpose,
                                      primme_svds_params* primme_svds,
                                      int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const size_type nrow{static_cast<size_type>(primme_svds->m)};
    const size_type ncol{static_cast<size_type>(primme_svds->n)};
    const size_type lldx{static_cast<size_type>(*ldx)};
    const size_type lldy{static_cast<size_type>(*ldy)};
    const size_type lbsz{static_cast<size_type>(*blockSize)};
    const char transp{(*transpose == 0) ? 'N' : 'T'};

    unmanaged_matrix_type x_view0((scalar_type*)x, lldx, lbsz);
    unmanaged_matrix_type y_view0((scalar_type*)y, lldy, lbsz);
    const unmanaged_matrix_type A_view((scalar_type*)primme_svds->matrix, nrow,
                                       ncol);

    size_type xrow{*transpose == 0 ? ncol : nrow};
    size_type yrow{*transpose == 0 ? nrow : ncol};

    std::pair<size_type, size_type> xidx = std::make_pair(0, xrow);
    std::pair<size_type, size_type> yidx = std::make_pair(0, yrow);

    auto x_view = Kokkos::subview(x_view0, xidx, Kokkos::ALL());
    auto y_view = Kokkos::subview(y_view0, yidx, Kokkos::ALL());

    KokkosBlas::gemm(&transp, "N", 1.0, A_view, x_view, 0.0, y_view);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout
        << "Skema::primme_svds_default_dense_matvec encountered an exception: "
        << e.what() << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}

void primme_svds_default_sparse_matvec(void* x,
                                       PRIMME_INT* ldx,
                                       void* y,
                                       PRIMME_INT* ldy,
                                       int* blockSize,
                                       int* transpose,
                                       primme_svds_params* primme_svds,
                                       int* err) {
  ///! Capture exceptions here; don't propagate them to C code
  try {
    const crs_matrix_type spmatrix = *(crs_matrix_type*)primme_svds->matrix;
    const size_t nrow{static_cast<size_t>(primme_svds->m)};
    const size_t ncol{static_cast<size_t>(primme_svds->n)};
    const char transp{(*transpose == 0) ? 'N' : 'T'};

    unmanaged_matrix_type x_view0((double*)x, static_cast<size_t>(*ldx),
                                  static_cast<size_t>(*blockSize));
    unmanaged_matrix_type y_view0((double*)y, static_cast<size_t>(*ldy),
                                  static_cast<size_t>(*blockSize));

    size_t xrow{*transpose == 0 ? ncol : nrow};
    size_t yrow{*transpose == 0 ? nrow : ncol};

    range_type xidx = std::make_pair(0, xrow);
    range_type yidx = std::make_pair(0, yrow);

    auto x_view = Kokkos::subview(x_view0, xidx, Kokkos::ALL());
    auto y_view = Kokkos::subview(y_view0, yidx, Kokkos::ALL());

    KokkosSparse::spmv(&transp, 1.0, spmatrix, x_view, 0.0, y_view);

    *err = 0;
  } catch (const std::exception& e) {
    std::cout
        << "Skema::primme_svds_default_sparse_matvec encountered an exception: "
        << e.what() << std::endl;
    *err = 1;  ///! notify to primme that something went wrong
  }
}
}