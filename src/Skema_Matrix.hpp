#pragma once
#include "Skema_Utils.hpp"

namespace Skema {
class MatrixBase {
 public:
  MatrixBase() {};
  MatrixBase(const MatrixBase&) = default;
  MatrixBase(MatrixBase&&) = default;
  MatrixBase& operator=(const MatrixBase&);
  MatrixBase& operator=(MatrixBase&&);
  virtual ~MatrixBase() = default;

  virtual size_type nrows() = 0;
  virtual size_type ncols() = 0;
  virtual bool issparse() = 0;
};

class Matrix : public MatrixBase {
 public:
  Matrix();
  ~Matrix() {};
}
}  // namespace Skema