#pragma once
#include <filesystem>

namespace Skema {
template <typename T>
auto read_mtx(const std::filesystem::path& filename) -> T;

template <typename T>
auto read_bin(const std::filesystem::path& filename) -> T;

template <typename T>
inline auto read_matrix(const std::filesystem::path& filename) -> T {
  if (filename.extension() == ".mtx") {
    return read_mtx<T>(filename);
  } else if (filename.extension() == ".bin") {
    return read_bin<T>(filename);
  } else {
    throw std::runtime_error(
        "Only \"mtx\" and \"bin\" file types are supported by Skema "
        "read_matrix()");
  }
};
}  // namespace Skema