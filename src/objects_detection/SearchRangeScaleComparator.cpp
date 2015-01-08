#include "SearchRangeScaleComparator.hpp"

namespace doppia {

SearchRangeScaleComparator::SearchRangeScaleComparator(const detector_search_ranges_data_t &search_ranges_)
    : search_ranges(search_ranges_)
{
    // nothing to do here
    return;
}

SearchRangeScaleComparator::~SearchRangeScaleComparator()
{
    // nothing to do here
    return;
}

bool SearchRangeScaleComparator::operator()(const size_t a, const size_t b)
{
    return search_ranges[a].detection_window_scale < search_ranges[b].detection_window_scale;
}


} // end of namespace doppia
