#include "anchor_grid.h"

#include <algorithm>
#include <cstdint>
#include <map>

namespace {
int cell_id(float u, float v, int grid, float w, float h) {
    int cu = static_cast<int>(u * grid / w);
    int cv = static_cast<int>(v * grid / h);
    cu = cu < 0 ? 0 : (cu > grid - 1 ? grid - 1 : cu);
    cv = cv < 0 ? 0 : (cv > grid - 1 ? grid - 1 : cv);
    return cv * grid + cu;
}

void pick_cell(const std::vector<size_t>& members, const std::vector<float>& depth,
               int k, const std::string& pick, std::vector<size_t>& out) {
    const int n = static_cast<int>(members.size());
    if (n <= k) {
        out.insert(out.end(), members.begin(), members.end());
        return;
    }
    if (k == 1) {
        if (pick == "nearest") {
            out.push_back(*std::min_element(
                members.begin(), members.end(),
                [&](size_t a, size_t b) { return depth[a] < depth[b]; }));
        } else if (pick == "median") {
            std::vector<size_t> s = members;
            std::sort(s.begin(), s.end(),
                      [&](size_t a, size_t b) { return depth[a] < depth[b]; });
            out.push_back(s[n / 2]);
        } else {
            out.push_back(members[0]);  // first
        }
        return;
    }
    for (int i = 0; i < k; ++i) {
        out.push_back(members[static_cast<int>(
            static_cast<int64_t>(i) * (n - 1) / (k - 1))]);
    }
}
}  // namespace

void grid_thin(
    const std::vector<float>& u, const std::vector<float>& v,
    const std::vector<float>& depth, int grid, int k, float width, float height,
    const std::string& pick, std::vector<size_t>& keep) {
    std::map<int, std::vector<size_t>> by;  // ascending cell order
    for (size_t i = 0; i < u.size(); ++i) {
        by[cell_id(u[i], v[i], grid, width, height)].push_back(i);
    }
    for (const auto& cell : by) {
        pick_cell(cell.second, depth, k, pick, keep);
    }
}
