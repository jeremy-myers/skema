#include "Skema_IO.hpp"
#include <fstream>
#include "KokkosSparse_IOUtils.hpp"
#include "Skema_Utils.hpp"

namespace Skema {

template <>
auto read_mtx<matrix_type>(const std::filesystem::path& filename)
    -> matrix_type {
  std::ifstream mmfile(filename, std::ifstream::in);
  if (!mmfile.is_open()) {
    throw std::runtime_error("File cannot be opened\n");
  }

  std::string fline = "";
  std::getline(mmfile, fline);
  if (fline.size() < 2 || fline[0] != '%' || fline[1] != '%') {
    throw std::runtime_error("Invalid MatrixMarket file. Line-1\n");
  }

  if (fline.find("matrix") == std::string::npos) {
    throw std::runtime_error(
        "Only MatrixMarket \"matrix\" is supported by Skema read_matrix()");
  }

  if (fline.find("array") == std::string::npos) {
    throw std::runtime_error(
        "Only MatrixMarket \"array\" is supported by Skema read_matrix()");
  }

  while (1) {
    getline(mmfile, fline);
    if (fline[0] != '%')
      break;
  }
  std::stringstream ss(fline);
  size_type nr{0};
  size_type nc{0};
  size_type nnz{0};
  ss >> nr >> nc;
  nnz = nr * nc;

  matrix_type A("A", nr, nc);
  scalar_type v;
  for (size_type n = 0; n < nnz; ++n) {
    getline(mmfile, fline);
    std::stringstream ss2(fline);
    ss2 >> A(n % nr, n / nr);
  }
  mmfile.close();
  return A;
};

template <>
auto read_bin<matrix_type>(const std::filesystem::path& filename)
    -> matrix_type {
  throw std::runtime_error(
      "Only sparse binary inputs are supported by Skema read_matrix()");
}

template <>
auto read_mtx<crs_matrix_type>(const std::filesystem::path& filename)
    -> crs_matrix_type {
  return KokkosSparse::Impl::read_kokkos_crst_matrix<crs_matrix_type>(
      filename.c_str());
}

template <>
auto read_bin<crs_matrix_type>(const std::filesystem::path& filename)
    -> crs_matrix_type {
  return KokkosSparse::Impl::read_kokkos_crst_matrix<crs_matrix_type>(
      filename.c_str());
}

template <>
auto write_mtx<matrix_type>(const matrix_type A,
                            const std::filesystem::path& filename) -> void {
  std::cout << "Skema write_matrix() not implemented yet for dense matrices"
            << std::endl;
}

template <>
auto write_mtx<crs_matrix_type>(const crs_matrix_type A,
                                const std::filesystem::path& filename) -> void {
  KokkosSparse::Impl::write_kokkos_crst_matrix(A, filename.c_str());
}

template <>
auto write_bin<matrix_type>(const matrix_type A,
                            const std::filesystem::path& filename) -> void {
  std::cout << "Skema write_matrix() not implemented yet for dense matrices"
            << std::endl;
}

template <>
auto write_bin<crs_matrix_type>(const crs_matrix_type A,
                                const std::filesystem::path& filename) -> void {
  KokkosSparse::Impl::write_kokkos_crst_matrix(A, filename.c_str());
}

}  // namespace Skema