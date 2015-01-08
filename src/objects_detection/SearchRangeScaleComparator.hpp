#ifndef DOPPIA_SEARCHRANGESCALECOMPARATOR_HPP
#define DOPPIA_SEARCHRANGESCALECOMPARATOR_HPP

#include "DetectorSearchRange.hpp"

namespace doppia {

/// Helper class for sorting the search ranges by scale (indirectly)
class SearchRangeScaleComparator
{

public:
    SearchRangeScaleComparator(const detector_search_ranges_data_t &search_ranges);
    ~SearchRangeScaleComparator();

    bool operator()(const size_t a, const size_t b);

protected:
    const detector_search_ranges_data_t &search_ranges;

};

} // namespace doppia

#endif // DOPPIA_SEARCHRANGESCALECOMPARATOR_HPP
