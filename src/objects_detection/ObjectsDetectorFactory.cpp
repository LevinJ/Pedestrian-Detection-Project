#include "ObjectsDetectorFactory.hpp"

#include "integral_channels/IntegralChannelsForPedestrians.hpp"

#include "IntegralChannelsLinearSvmSlidingWindow.hpp"
#include "IntegralChannelsDetector.hpp"
#include "FastestPedestrianDetectorInTheWest.hpp"
#include "FastestPedestrianDetectorInTheWestV2.hpp"
#include "MultiscalesIntegralChannelsDetector.hpp"
#include "VeryFastIntegralChannelsDetector.hpp"
#include "IntegralChannelsModelsBundleDetector.hpp"

#if defined(USE_GPU)
#include "GpuIntegralChannelsDetector.hpp"
#include "GpuMultiscalesIntegralChannelsDetector.hpp"
#include "GpuIntegralChannelsModelsBundleDetector.hpp"
#include "GpuFastestPedestrianDetectorInTheWest.hpp"
#include "GpuVeryFastIntegralChannelsDetector.hpp"
#endif

#include "non_maximal_suppression/NonMaximalSuppressionFactory.hpp"
#include "non_maximal_suppression/AbstractNonMaximalSuppression.hpp"
#include "LinearSvmModel.hpp"
#include "SoftCascadeOverIntegralChannelsModel.hpp"
#include "MultiScalesIntegralChannelsModel.hpp"
#include "IntegralChannelsDetectorModelsBundle.hpp"

#include "detector_model.pb.h"

#include "helpers/get_option_value.hpp"
#include "helpers/Log.hpp"
#include "helpers/replace_environment_variables.hpp"

#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include <stdexcept>
#include <string>


namespace
{

using namespace std;

std::ostream & log_info()
{
    return  logging::log(logging::InfoMessage, "ObjectsDetectorFactory");
}

std::ostream & log_debug()
{
    return  logging::log(logging::DebugMessage, "ObjectsDetectorFactory");
}

std::ostream & log_error()
{
    return  logging::log(logging::ErrorMessage, "ObjectsDetectorFactory");
}



} // end of anonymous namespace


namespace doppia
{

using namespace std;
using boost::shared_ptr;
using namespace boost::program_options;

options_description
ObjectsDetectorFactory::get_args_options()
{
    options_description desc("ObjectsDetectorFactory options");

    desc.add_options()

            ("objects_detector.method", value<string>()->default_value("cpu_fpdw_v2"),
             "detection methods: \n"\
             "\tcpu_linear_svm: sliding window linear svm over integral channels\n" \
             "\tcpu_channel: P. Dollar 2009 Integral Channel Features\n" \
             "\tgpu_channel: GPU implementation of gpu_channel\n" \
             "\tcpu_fpdw, cpu_fpdw_v2: P. Dollar 2010 Fastest pedestrian detector in the West\n" \
             "\tgpu_fpdw: GPU implementation of cpu_fpdw_v2\n" \
             "\tcpu_very_fast: CPU implementation of Benenson et al. CVPR2012 very fast detector\n" \
             "\tgpu_very_fast: GPU implementation of cpu_very_fast\n" \
             "or none ")

            ("objects_detector.model", value<string>(), "path to the detector model file")

            ("objects_detector.score_threshold", value<float>()->default_value(0.5),
             "minimum score needed to validate a detection." \
             "The score is assumed normalized across classes and models.")

            ("objects_detector.non_maximal_suppression_method", value<string>()->default_value("greedy"),
             "Indicate which method to use for non-maximal suppression. Options are:\n"\
             "\tgreedy\n" \
             "\tnone\n" )

            ("objects_detector.ignore_soft_cascade", value<bool>()->default_value(false),
             "if set to true, the model soft cascade will be ignored")

            // FIXME fix this mess redundant useless paremeters
            ("objects_detector.cascade_threshold_offset", value<float>()->default_value(0.0),
             "Negative offset added cascade threshold, so that they are less strict." \
             "The value is a fraction of the maximum (last) threshold value (e.g 0.1). " \
             "This value is used together with cascade_threshold_offset_decay to define an offset curve")

            ("objects_detector.cascade_threshold_offset_decay", value<float>()->default_value(1E-5),
             "This variable controls the speed of decay of the exponential that applies the cascade_threshold_offset." \
             "Initial scores have zero offset, and latest scores have the full offset value. " \
             "Very small decay value will provide a linear interpolation, " \
             "high values (~1E-2) will create an exponential offset.")

            ("objects_detector.cascade_threshold_additive_offset", value<float>()->default_value(0.0),
             "Negative offset added cascade threshold, so that they are less strict." \
             "The value is a fraction of the maximum (last) threshold value (e.g 0.1).")

            ("objects_detector.max_num_stages", value<int>()->default_value(-1),
             "This value sets the maximum number of stages used from a model" \
             "The parameter is ignored if the softcascade contains fewer stages than specified or if it is set to -1")

            ;

    desc.add(AbstractObjectsDetector::get_args_options());
    desc.add(IntegralChannelsLinearSvmSlidingWindow::get_args_options());
    desc.add(IntegralChannelsDetector::get_args_options());
    desc.add(FastestPedestrianDetectorInTheWest::get_args_options());
    //desc.add(MultiscalesIntegralChannelsDetector::get_args_options());
    //desc.add(VeryFastIntegralChannelsDetector::get_args_options());
    desc.add(NonMaximalSuppressionFactory::get_args_options());

#if defined(USE_GPU)
    desc.add(GpuIntegralChannelsDetector::get_args_options());
    //desc.add(GpuMultiscalesIntegralChannelsDetector::get_args_options());
    //desc.add(GpuVeryFastIntegralChannelsDetector::get_args_options());
#endif

    return desc;
}



/// will set detector_model_p if the read was successful, will left it empty else
template<typename ProtoBufferModelType>
void read_protobuf_model_impl(const string &filename, boost::shared_ptr<ProtoBufferModelType> &detector_model_p)
{
    using namespace google::protobuf;

    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    // parse the protocol buffer file

    // try parsing in binary format
    if(detector_model_p == false)
    {
        fstream input_stream(filename.c_str(), ios::in | ios::binary);
        detector_model_p.reset(new ProtoBufferModelType());
        io::ZeroCopyInputStream *zci_stream_p = new io::IstreamInputStream(&input_stream);
        const bool success = detector_model_p->ParseFromZeroCopyStream(zci_stream_p);
        delete zci_stream_p;

        if((success == false) or (detector_model_p->IsInitialized() == false))
        {
            log_info() << "Model file " << filename << " is not in binary protocol buffer format" << std::endl;
            detector_model_p.reset(); // detroy the allocated object to indicate the failure
        }
        else
        {
            log_info() << "Parsed the binary model file " << filename << std::endl;
        }
    }

    // try parsing in Text Format
    if(detector_model_p == false)
    {
        fstream input_stream(filename.c_str(), ios::in); // text is default stream format
        detector_model_p.reset(new ProtoBufferModelType());
        io::ZeroCopyInputStream *zci_stream_p = new io::IstreamInputStream(&input_stream);
        const bool success = TextFormat::Parse(zci_stream_p, detector_model_p.get());
        delete zci_stream_p;

        if(success == false or detector_model_p->IsInitialized() == false)
        {
            log_info() << "Model file " << filename << " is not in text protocol buffer format" << std::endl;
            detector_model_p.reset(); // detroy the allocated object to indicate the failure
        }
        else
        {
            log_info() << "Parsed the text model file " << filename << std::endl;
        }
    }

    return;
}


void read_protobuf_model(const std::string &filename,
                         boost::shared_ptr<doppia_protobuf::DetectorModel> &detector_model_p)
{
    return read_protobuf_model_impl(filename, detector_model_p);
}


void add_cascade_offset(
        doppia_protobuf::SoftCascadeOverIntegralChannelsModel &model,
        const float cascade_threshold_offset_fraction,
        const float cascade_threshold_offset_decay,
        const float cascade_threshold_additive_offset_fraction)
{
    const size_t num_stages = model.stages().size();

    if((num_stages == 0) or ((cascade_threshold_offset_fraction == 0) and (cascade_threshold_additive_offset_fraction == 0)))
    {
        if((num_stages == 0))
        {
            printf("model.stages().size() == 0\n");
        }
        // nothing to do here
        return;
    }

    const float
            //last_stage_threshold = model.stages(num_stages - 1).cascade_threshold(),
            // FIXME NOT USED AS FRACTIONS
            //cascade_threshold_offset =last_stage_threshold*cascade_threshold_offset_fraction,
            //cascade_threshold_additive_offset = last_stage_threshold*cascade_threshold_additive_offset_fraction;
            cascade_threshold_offset = cascade_threshold_offset_fraction,
            cascade_threshold_additive_offset = cascade_threshold_additive_offset_fraction;

    const float
            b = cascade_threshold_offset_decay,
            a =  -cascade_threshold_offset / (std::exp(b * (num_stages - 1)) - 1);

    std::vector<float> offset(num_stages);
    for(size_t stage_index=0; stage_index < num_stages; stage_index+=1)
    {
        offset[stage_index] = a*(std::exp(b* stage_index) - 1) - cascade_threshold_additive_offset;
        if(true and ( (stage_index == (num_stages / 2))
                      or (stage_index == (num_stages - 1))
                      or (stage_index == 0) ))
        {
            printf("add_cascade_offset: at stage %zi the offset value is %.5f\n", stage_index, offset[stage_index]);
        }
    }

    for(size_t stage_index=0; stage_index < num_stages; stage_index+=1)
    {
        doppia_protobuf::SoftCascadeOverIntegralChannelsStage &stage_data = *(model.mutable_stages((stage_index)));
        //printf("Stage %i cascade_threshold == %.3f\n", c, stage_data.cascade_threshold());
        stage_data.set_cascade_threshold(stage_data.cascade_threshold() + offset[stage_index]);
    }

    return;
}


AbstractObjectsDetector*
new_single_scale_detector_instance(const variables_map &options,
                                   boost::shared_ptr<doppia_protobuf::DetectorModel> detector_model_data_p,
                                   boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p)
{
    //mutable_stages
    const int max_num_stages = get_option_value<int>(options, "objects_detector.max_num_stages");
    doppia_protobuf::SoftCascadeOverIntegralChannelsModel &model = *(detector_model_data_p->mutable_soft_cascade_model());

    if ((max_num_stages > 0) and (max_num_stages< model.stages_size()))
    {
        log_info() << "Model num stages: " << model.stages_size() << std::endl;

        google::protobuf::RepeatedPtrField< doppia_protobuf::SoftCascadeOverIntegralChannelsStage > all_stages;
        all_stages.CopyFrom(model.stages());

#if GOOGLE_PROTOBUF_VERSION >= 2005000
        model.mutable_stages()->DeleteSubrange(max_num_stages, model.stages_size() - max_num_stages);
#else
        for(int c  = 0; c < max_num_stages; c+= 1)
        {
            doppia_protobuf::SoftCascadeOverIntegralChannelsStage *stage_p = model.mutable_stages()->Add();
            stage_p->CopyFrom(all_stages.Get(c));
        }
#endif
        log_info() << "Model num stages after cropping: " << model.stages_size() << std::endl;
    }

    if((detector_model_data_p.get() != NULL) and (detector_model_data_p->has_soft_cascade_model()))
    {
        const float
                cascade_threshold_offset_fraction = \
                get_option_value<float>(options, "objects_detector.cascade_threshold_offset"),
                cascade_threshold_offset_decay = \
                get_option_value<float>(options, "objects_detector.cascade_threshold_offset_decay"),
                cascade_threshold_additive_offset = \
                get_option_value<float>(options, "objects_detector.cascade_threshold_additive_offset");

        add_cascade_offset(*(detector_model_data_p->mutable_soft_cascade_model()),
                           cascade_threshold_offset_fraction, cascade_threshold_offset_decay,
                           cascade_threshold_additive_offset);
    }


    boost::shared_ptr<SoftCascadeOverIntegralChannelsModel> cascade_model_p;
    float score_threshold = get_option_value<float>(options, "objects_detector.score_threshold");

    const bool ignore_cascade = get_option_value<bool>(options, "objects_detector.ignore_soft_cascade");

    if(detector_model_data_p)
    {
        cascade_model_p.reset(new SoftCascadeOverIntegralChannelsModel(*detector_model_data_p));

        if((ignore_cascade == false) and cascade_model_p and cascade_model_p->has_soft_cascade())
        {
            const float last_cascade_threshold = cascade_model_p->get_last_cascade_threshold();
            // seems a "non trivial" threshold
            log_info() <<  boost::str(boost::format(
                                          "Updating the score threshold using the last cascade threshold. " \
                                          "New score threshold == %.3f (= %.3f + %.3f)")
                                      % (score_threshold + last_cascade_threshold)
                                      % score_threshold % last_cascade_threshold) << std::endl;
            score_threshold += last_cascade_threshold;
        }
    }

    int additional_border = 0;
    if(options.count("additional_border") > 0)
    {
        additional_border = get_option_value<int>(options, "additional_border");
    }

    AbstractObjectsDetector* objects_detector_p = NULL;
    const string method = get_option_value<string>(options, "objects_detector.method");
    if(method.compare("cpu_linear_svm") == 0)
    {
        boost::shared_ptr<IntegralChannelsForPedestrians>
                integral_channels_p(new IntegralChannelsForPedestrians());
        boost::shared_ptr<LinearSvmModel> linear_svm_model_p;

        if(detector_model_data_p)
        {
            linear_svm_model_p.reset(new LinearSvmModel(*detector_model_data_p));
        }
        else
        {
            const string model_path = get_option_value<string>(options, "objects_detector.model");
            linear_svm_model_p.reset(new LinearSvmModel(model_path));
        }

        objects_detector_p = new IntegralChannelsLinearSvmSlidingWindow(
                    options,
                    integral_channels_p, linear_svm_model_p, non_maximal_suppression_p,
                    score_threshold);
    }
    else if((method.compare("cpu_channel") == 0) or
            (method.compare("cpu_channels") == 0) or
            (method.compare("cpu_chnftrs") == 0) )
    {

        if(cascade_model_p == false)
        {
            throw std::runtime_error("Failed to read a model compatible with the selected IntegralChannelsDetector");
        }

        objects_detector_p = new IntegralChannelsDetector(
                    options,
                    cascade_model_p, non_maximal_suppression_p,
                    score_threshold, additional_border);

    }
    else if((method.compare("gpu_channel") == 0) or
            (method.compare("gpu_channels") == 0) or
            (method.compare("gpu_chnftrs") == 0) )
    {

        if(cascade_model_p == false)
        {
            throw std::runtime_error("Failed to read a model compatible with the selected GpuIntegralChannelsDetector");
        }

#if defined(USE_GPU) and (not defined(BOOTSTRAPPING_LIB))
        objects_detector_p = new GpuIntegralChannelsDetector(
                    options,
                    cascade_model_p, non_maximal_suppression_p,
                    score_threshold, additional_border);
#else
        throw std::runtime_error("This executable was compiled without support for GpuIntegralChannelsDetector");
#endif
    }
    else if(method.compare("cpu_fpdw") == 0)
    {

        if(cascade_model_p == false)
        {
            throw std::runtime_error("Failed to read a model compatible with the selected FastestPedestrianDetectorInTheWest");
        }

        objects_detector_p = new FastestPedestrianDetectorInTheWest(
                    options,
                    cascade_model_p, non_maximal_suppression_p,
                    score_threshold,
                    additional_border);

    }
    else if(method.compare("cpu_fpdw_v2") == 0)
    {

        if(cascade_model_p == false)
        {
            throw std::runtime_error("Failed to read a model compatible with the selected FastestPedestrianDetectorInTheWestV2");
        }

        objects_detector_p = new FastestPedestrianDetectorInTheWestV2(
                    options,
                    cascade_model_p, non_maximal_suppression_p,
                    score_threshold,
                    additional_border);

    }
    else if(method.compare("gpu_fpdw") == 0)
    {
#if defined(USE_GPU) and (not defined(BOOTSTRAPPING_LIB))
        objects_detector_p = new GpuFastestPedestrianDetectorInTheWest(
                    options,
                    cascade_model_p, non_maximal_suppression_p,
                    score_threshold, additional_border);
#else
        throw std::runtime_error("This executable was compiled without support for GpuFastestPedestrianDetectorInTheWest");
#endif
    }
    else if((method.compare("cpu_very_fast") == 0) or
            (method.compare("cpu_fast") == 0) or
            (method.compare("cpu_cvpr2012") == 0) or
            (method.compare("gpu_very_fast") == 0) or
            (method.compare("gpu_fast") == 0) or
            (method.compare("gpu_cvpr2012") == 0) )
    {
        throw std::runtime_error(
                    boost::str( boost::format(
                                    "method %s does not support single scale models as input") % method));
    }
    else if (method.compare("none") == 0)
    {
        objects_detector_p = NULL;
    }
    else
    {
        printf("ObjectsDetectorFactory received objects_detector.method value == %s\n", method.c_str());
        throw std::runtime_error("Unknown 'objects_detector.method' value (for single scale models)");
    }

    return objects_detector_p;
}


/// helper function that will modify the detector model data
template <typename ProtobufDetectorModel>
void add_cascade_offset(
        const variables_map &options,
        boost::shared_ptr<ProtobufDetectorModel> detector_model_data_p)
{
    assert(detector_model_data_p.get() != NULL);

    const float
            cascade_threshold_offset_fraction = \
            get_option_value<float>(options, "objects_detector.cascade_threshold_offset"),
            cascade_threshold_offset_decay = \
            get_option_value<float>(options, "objects_detector.cascade_threshold_offset_decay"),
            cascade_threshold_additive_offset = \
            get_option_value<float>(options, "objects_detector.cascade_threshold_additive_offset");


    for(int index=0; index < detector_model_data_p->detectors().size(); index += 1)
    {
        doppia_protobuf::DetectorModel &single_detector_model_data =
                *(detector_model_data_p->mutable_detectors(index));
        if(single_detector_model_data.has_soft_cascade_model())
        {
            add_cascade_offset(*(single_detector_model_data.mutable_soft_cascade_model()),
                               cascade_threshold_offset_fraction, cascade_threshold_offset_decay,
                               cascade_threshold_additive_offset);
        }

    } // end of "for each detector model element"

    return;
}




/// helper function that updates the score threshold based on the cascade threshold
template<typename DetectorModelType>
void add_last_cascade_threshold_to_score_threshold(
        const variables_map &options,
        boost::shared_ptr<DetectorModelType> detector_model_p,
        float &score_threshold)
{
    assert(detector_model_p.get() != NULL);

    const bool ignore_cascade = get_option_value<bool>(options, "objects_detector.ignore_soft_cascade");

    const typename DetectorModelType::detectors_t &detectors = detector_model_p->get_detectors();

    if(ignore_cascade == false)
    {
        for(size_t detector_index=0; detector_index < detectors.size(); detector_index += 1)
        {
            const typename DetectorModelType::detector_t &detector = detectors[detector_index];
            if(detector.has_soft_cascade())
            {
                const float last_cascade_threshold = detector.get_last_cascade_threshold();
                // seems a "non trivial" threshold
                log_info() <<  boost::str(boost::format(
                                              "Updating the score threshold using "
                                              "the last cascade threshold of detector %i. " \
                                              "New score threshold == %.3f (= %.3f + %.3f)")
                                          % detector_index
                                          % (score_threshold + last_cascade_threshold)
                                          % score_threshold % last_cascade_threshold) << std::endl;
                score_threshold += last_cascade_threshold;
                break;
            }
        } // end of "for each detector"
    } // end of "if we care about the cascade"

    return;
}


AbstractObjectsDetector*
new_multi_scales_detector_instance(const variables_map &options,
                                   boost::shared_ptr<doppia_protobuf::MultiScalesDetectorModel> detector_model_data_p,
                                   boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p)
{

    boost::shared_ptr<MultiScalesIntegralChannelsModel> detector_model_p;
    float score_threshold = get_option_value<float>(options, "objects_detector.score_threshold");

    if(detector_model_data_p)
    {
        add_cascade_offset(options, detector_model_data_p);

        // the MultiScalesIntegralChannelsModel is built _after_ applying cascade_threshold_offset
        detector_model_p.reset(new MultiScalesIntegralChannelsModel(*detector_model_data_p));

        // MultiScalesIntegralChannelsModel takes care of normalizing the models between themselves
        // this we only need to update the score_threshold based on one of models

        add_last_cascade_threshold_to_score_threshold(options, detector_model_p, score_threshold);

    } // end of "if detector model data is available"


    int additional_border = 0;
    if(options.count("additional_border") > 0)
    {
        additional_border = get_option_value<int>(options, "additional_border");
    }

    AbstractObjectsDetector* objects_detector_p = NULL;
    const string method = get_option_value<string>(options, "objects_detector.method");
    if((method.compare("cpu_channel") == 0) or
            (method.compare("cpu_channels") == 0) or
            (method.compare("cpu_chnftrs") == 0) )
    {

        if(detector_model_p == false)
        {
            throw std::runtime_error("Failed to read a model compatible with the selected IntegralChannelsDetector");
        }

        objects_detector_p = new MultiscalesIntegralChannelsDetector(
                    options,
                    detector_model_p, non_maximal_suppression_p,
                    score_threshold, additional_border);

    }
    else if((method.compare("gpu_channel") == 0) or
            (method.compare("gpu_channels") == 0) or
            (method.compare("gpu_chnftrs") == 0) )
    {

#if defined(USE_GPU) and (not defined(BOOTSTRAPPING_LIB))
        objects_detector_p = new GpuMultiscalesIntegralChannelsDetector(
                    options,
                    detector_model_p, non_maximal_suppression_p,
                    score_threshold, additional_border);
#else
        throw std::runtime_error("This executable was compiled without support for GpuIntegralChannelsDetector");
#endif
    }
    else if((method.compare("cpu_linear_svm") == 0) or
            (method.compare("cpu_fpdw") == 0) or
            (method.compare("gpu_fpdw") == 0) )
    {
        throw std::runtime_error(
                    boost::str( boost::format(
                                    "method %s does not support multi scale models as input") % method));
    }
    else if((method.compare("cpu_very_fast") == 0) or
            (method.compare("cpu_fast") == 0) or
            (method.compare("cpu_cvpr2012") == 0) )
    {

        if(detector_model_p == false)
        {
            throw std::runtime_error("Failed to read a model compatible with the selected IntegralChannelsDetector");
        }

        objects_detector_p = new VeryFastIntegralChannelsDetector(
                    options,
                    detector_model_p, non_maximal_suppression_p,
                    score_threshold, additional_border);

    }
    else if((method.compare("gpu_very_fast") == 0) or
            (method.compare("gpu_fast") == 0) or
            (method.compare("gpu_cvpr2012") == 0) )
    {

#if defined(USE_GPU) and (not defined(BOOTSTRAPPING_LIB))
        objects_detector_p = new GpuVeryFastIntegralChannelsDetector(
                    options,
                    detector_model_p, non_maximal_suppression_p,
                    score_threshold, additional_border);
#else
        throw std::runtime_error("This executable was compiled without support for GpuVeryFastIntegralChannelsDetector");
#endif
    }
    else if (method.compare("none") == 0)
    {
        objects_detector_p = NULL;
    }
    else
    {
        printf("ObjectsDetectorFactory received objects_detector.method value == %s\n", method.c_str());
        throw std::runtime_error("Unknown 'objects_detector.method' value  (for multiscales scale models)");
    }


    return objects_detector_p;
}


AbstractObjectsDetector*
new_detectors_bundle_instance(const variables_map &options,
                              boost::shared_ptr<doppia_protobuf::DetectorModelsBundle> detector_model_data_p,
                              boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p)
{

    boost::shared_ptr<IntegralChannelsDetectorModelsBundle> detector_model_p;
    float score_threshold = get_option_value<float>(options, "objects_detector.score_threshold");


    if(detector_model_data_p)
    {
        add_cascade_offset(options, detector_model_data_p);

        // the MultiScalesIntegralChannelsModel is built _after_ applying cascade_threshold_offset
        detector_model_p.reset(new IntegralChannelsDetectorModelsBundle(*detector_model_data_p));

        // MultiScalesIntegralChannelsModel takes care of normalizing the models between themselves
        // this we only need to update the score_threshold based on one of models

        add_last_cascade_threshold_to_score_threshold(options, detector_model_p, score_threshold);

    } // end of "if detector model data is available"


    AbstractObjectsDetector* objects_detector_p = NULL;
    const string method = get_option_value<string>(options, "objects_detector.method");
    if((method.compare("gpu_channel") == 0) or
            (method.compare("gpu_channels") == 0) or
            (method.compare("gpu_chnftrs") == 0) )
    {

#if defined(USE_GPU) and (not defined(BOOTSTRAPPING_LIB))
        int additional_border = 0;
        if(options.count("additional_border") > 0)
        {
            additional_border = get_option_value<int>(options, "additional_border");
        }

        objects_detector_p = new GpuIntegralChannelsModelsBundleDetector(
                    options,
                    detector_model_p, non_maximal_suppression_p,
                    score_threshold, additional_border);
#else
        throw std::runtime_error("This executable was compiled without support for GpuIntegralChannelsDetector");
#endif
    }
    else if((method.compare("cpu_channel") == 0) or
            (method.compare("cpu_channels") == 0) or
            (method.compare("cpu_chnftrs") == 0))
    {
        int additional_border = 0;
        if(options.count("additional_border") > 0)
        {
            additional_border = get_option_value<int>(options, "additional_border");
        }

        objects_detector_p = new IntegralChannelsModelsBundleDetector(
                    options,
                    detector_model_p, non_maximal_suppression_p,
                    score_threshold, additional_border);
    }
    else if((method.compare("cpu_very_fast") == 0) or
            (method.compare("cpu_fast") == 0) or
            (method.compare("cpu_cvpr2012") == 0) or
            (method.compare("gpu_very_fast") == 0) or
            (method.compare("gpu_fast") == 0) or
            (method.compare("gpu_cvpr2012") == 0) or
            (method.compare("cpu_linear_svm") == 0) or
            (method.compare("cpu_fpdw") == 0) or
            (method.compare("gpu_fpdw") == 0) )
    {
        throw std::runtime_error(
                    boost::str( boost::format(
                                    "method %s does not support a detector models bundle as input") % method));
    }
    else if (method.compare("none") == 0)
    {
        objects_detector_p = NULL;
    }
    else
    {
        printf("ObjectsDetectorFactory received objects_detector.method value == %s\n", method.c_str());
        throw std::runtime_error("Unknown 'objects_detector.method' value  (for detector models bundle)");
    }


    return objects_detector_p;
}



AbstractObjectsDetector*
ObjectsDetectorFactory::new_instance(const variables_map &options)
{

    const string non_maximal_suppression_method = \
            get_option_value<string>(options, "objects_detector.non_maximal_suppression_method");
    boost::shared_ptr<AbstractNonMaximalSuppression> non_maximal_suppression_p;
    non_maximal_suppression_p.reset( NonMaximalSuppressionFactory::new_instance(non_maximal_suppression_method, options) );

    boost::filesystem::path model_path = get_option_value<string>(options, "objects_detector.model");

    model_path = replace_environment_variables(model_path);

    if(boost::filesystem::exists(model_path) == false)
    {
        log_error() << "Could not find the objects_detector.model file " << model_path << std::endl;
        throw std::invalid_argument("Could not find the objects_detector.model file");
    }

    boost::shared_ptr<doppia_protobuf::DetectorModel> detector_model_data_p;
    read_protobuf_model(model_path.string(), detector_model_data_p);

    boost::shared_ptr<doppia_protobuf::MultiScalesDetectorModel> multi_scales_detector_model_data_p;
    read_protobuf_model_impl(model_path.string(), multi_scales_detector_model_data_p);

    boost::shared_ptr<doppia_protobuf::DetectorModelsBundle> detector_models_bundle_data_p;
    read_protobuf_model_impl(model_path.string(), detector_models_bundle_data_p);

    // current code cannot distinguish MultiScalesDetectorModel from DetectorModelsBundle
    // because the messages are compatible. We use the "method" as a quick hack.
    bool should_use_bundle = false;
    const string method = get_option_value<string>(options, "objects_detector.method");
    if((method.compare("gpu_channel") == 0) or (method.compare("cpu_channel") == 0) or
            (method.compare("gpu_channels") == 0) or (method.compare("cpu_channels") == 0) or
            (method.compare("gpu_chnftrs") == 0) or (method.compare("cpu_chnftrs") == 0) )
    {
        should_use_bundle = true;
    }

    if(detector_model_data_p)
    {
        return new_single_scale_detector_instance(options,
                                                  detector_model_data_p, non_maximal_suppression_p);
    }
    else if(detector_models_bundle_data_p and should_use_bundle)
    {
        return new_detectors_bundle_instance(options,
                                             detector_models_bundle_data_p, non_maximal_suppression_p);
    }
    else if(multi_scales_detector_model_data_p)
    {
        return new_multi_scales_detector_instance(options,
                                                  multi_scales_detector_model_data_p, non_maximal_suppression_p);
    }
    else
    {
        throw std::runtime_error("Received a model of an unknown type, model was not recognized "
                                 "as DetectorModel, MultiScalesDetectorModel nor DetectorModelsBundle");
    }

    return NULL;
}


} // end of namespace doppia
