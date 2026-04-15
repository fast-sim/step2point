#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace step2point {

struct Shower {
  std::int32_t shower_id = -1;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  std::vector<float> E;
  std::optional<std::vector<float>> t;
  std::optional<std::vector<std::uint64_t>> cell_id;

  [[nodiscard]] std::size_t n_points() const noexcept { return E.size(); }
  [[nodiscard]] double total_energy() const noexcept;
  [[nodiscard]] bool valid() const noexcept;
};

struct CompressionStats {
  std::size_t n_points_before = 0;
  std::size_t n_points_after = 0;
  double energy_before = 0.0;
  double energy_after = 0.0;
  double compression_ratio = 1.0;
};

struct CompressionResult {
  Shower shower;
  CompressionStats stats;
  std::string algorithm;
};

} // namespace step2point
