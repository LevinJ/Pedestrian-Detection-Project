#include "WeakClassifier.hpp"

using namespace boosted_learning;

WeakClassifier::WeakClassifier() :
    _cascade_threshold(-std::numeric_limits<weights_t::value_type>::max())
{
    return;
}

WeakClassifier::WeakClassifier(const bool silent_mode) :
    _silent_mode(silent_mode), _cascade_threshold(-std::numeric_limits<weights_t::value_type>::max())
{
    return;
}
