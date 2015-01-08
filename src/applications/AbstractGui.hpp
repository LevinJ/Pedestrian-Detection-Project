#ifndef ABSTRACTGUI_HPP
#define ABSTRACTGUI_HPP

#include <boost/program_options.hpp>

namespace doppia {

    class AbstractGui
    {
    public:
        //virtual static boost::program_options::options_description get_args_options() = 0;

        AbstractGui(const boost::program_options::variables_map &options);
        virtual ~AbstractGui();

        /// @returns true if the application should stop
        virtual bool update() = 0;
    };

} // end of namespace doppia

#endif // ABSTRACTGUI_HPP
