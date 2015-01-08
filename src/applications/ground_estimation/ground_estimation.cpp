
#include <cstdlib>
#include <iostream>

#include <boost/scoped_ptr.hpp>


#include "GroundEstimationApplication.hpp"


using namespace std;


// -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
int main(int argc, char *argv[])
{
    int ret = EXIT_SUCCESS;

    try
    {
            boost::scoped_ptr<doppia::AbstractApplication>
                application_p( new doppia::GroundEstimationApplication() );

            ret = application_p->main(argc, argv);
    }
    // on linux re-throw the exception in order to get the information
    catch (std::exception & e)
    {
        cout << "\033[1;31mA std::exception was raised:\033[0m " << e.what () << endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }
    catch (...)
    {
        cout << "\033[1;31mAn unknown exception was raised\033[0m " << endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }

    return ret;
}


