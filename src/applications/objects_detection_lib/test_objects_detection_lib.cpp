
#include "TestObjectsDetectionApplication.hpp"

#include <iostream>

int main(int argc, char *argv[])
{
    int ret = EXIT_SUCCESS;

    try
    {
        boost::scoped_ptr<objects_detection::TestObjectsDetectionApplication>
                application_p( new objects_detection::TestObjectsDetectionApplication() );

        ret = application_p->main(argc, argv);
    }
    // on linux re-throw the exception in order to get the information
    catch (std::exception & e)
    {
        std::cout << "\033[1;31mA std::exception was raised:\033[0m " << e.what () << std::endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }
    catch (...)
    {
        std::cout << "\033[1;31mAn unknown exception was raised\033[0m " << std::endl;
        ret = EXIT_FAILURE; // an error appeared
        throw;
    }

    return ret;
}


