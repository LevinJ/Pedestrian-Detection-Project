#ifndef OBJECTSDETECTORFACTORY_HPP
#define OBJECTSDETECTORFACTORY_HPP

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <string>

namespace doppia_protobuf {
class DetectorModel;
}

namespace doppia {

// forward declaration
class AbstractObjectsDetector;

class ObjectsDetectorFactory
{
public:
    static boost::program_options::options_description get_args_options();
    static AbstractObjectsDetector* new_instance(const boost::program_options::variables_map &options);
};

void read_protobuf_model(const std::string &filename,
                         boost::shared_ptr<doppia_protobuf::DetectorModel> &detector_model_p);

} // end of namespace doppia

#endif // OBJECTSDETECTORFACTORY_HPP
