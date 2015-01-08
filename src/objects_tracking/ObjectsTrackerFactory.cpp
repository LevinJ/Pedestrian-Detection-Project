#include "ObjectsTrackerFactory.hpp"
#include "DummyObjectsTracker.hpp"

#include "helpers/get_option_value.hpp"

namespace doppia {

using namespace std;
using boost::shared_ptr;
using namespace boost::program_options;


options_description ObjectsTrackerFactory::get_args_options()
{

    options_description desc("ObjectsTrackerFactory options");

    desc.add_options()

            ("objects_tracker.method", value<string>()->default_value("none"),
             "tracking methods: \n"\
             "\tdummy: simplistic 2d tracker\n" \
             "or none ")

            ;

    //desc.add(AbstractObjectsTracker::get_args_options());
    desc.add(DummyObjectsTracker::get_args_options());

    return desc;
}


AbstractObjectsTracker *ObjectsTrackerFactory::new_instance(const variables_map &options)
{
    AbstractObjectsTracker* objects_tracker_p = NULL;
    const string method = get_option_value<string>(options, "objects_tracker.method");
    if(method.compare("dummy") == 0)
    {
        objects_tracker_p = new DummyObjectsTracker(options);
    }
    else if (method.compare("none") == 0)
    {
        objects_tracker_p = NULL;
    }
    else
    {
        printf("ObjectsTrackerFactory received objects_tracker.method value == %s\n", method.c_str());
        throw std::runtime_error("Unknown 'objects_tracker.method' value");
    }

    return objects_tracker_p;
}

} // end of namespace doppia
