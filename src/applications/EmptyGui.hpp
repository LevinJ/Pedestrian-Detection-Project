#ifndef EMPTYGUI_HPP
#define EMPTYGUI_HPP

#include "AbstractGui.hpp"

namespace doppia {

class EmptyGui : public doppia::AbstractGui
{
public:

    EmptyGui(const boost::program_options::variables_map &options);
    ~EmptyGui();

    /// @returns true if the application should stop
    bool update();

};

} // end of namespace doppia

#endif // EMPTYGUI_HPP
