#pragma once

#include "step2point/shower.hpp"

namespace step2point {

CompressionResult merge_within_cell(const Shower& input);
CompressionResult identity(const Shower& input);

} // namespace step2point
