#include "EmptyGui.hpp"

namespace doppia {

EmptyGui::EmptyGui(const boost::program_options::variables_map &options)
    :AbstractGui(options)
{
    // nothing to do here
    return;
}

EmptyGui::~EmptyGui()
{
    // nothing to do here
    return;
}


/// @returns true if the application should stop
bool EmptyGui::update()
{
    // keep going until input data ends
    return false;
}

} // end of namespace doppia
