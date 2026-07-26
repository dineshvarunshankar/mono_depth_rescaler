#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Grid-thin ToF: keep up to k per GxG image cell; indices into u/v. pick (k=1):
// "first" | "nearest" | "median" by depth. Parity with anchor_grid.py.
void grid_thin(
    const std::vector<float>& u, const std::vector<float>& v,
    const std::vector<float>& depth, int grid, int k, float width, float height,
    const std::string& pick, std::vector<size_t>& keep);
