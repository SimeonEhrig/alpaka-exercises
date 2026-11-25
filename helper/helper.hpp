#pragma once

#include <string>
#include <alpaka/alpaka.hpp>

// print data of a 1D DataSource
std::string dataSourceToStr(alpaka::concepts::IDataSource auto &data) {
  static_assert(data.dim() == 1);
  std::stringstream ss;
  ss << "[";
  for (auto i = 0; i < data.getExtents()[0]; ++i) {
    ss << data[i] << " ";
  }
  ss << "]";
  return ss.str();
}

