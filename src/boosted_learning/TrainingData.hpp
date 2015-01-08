#ifndef BOOSTED_LEARNING_TRAININGDATA_HPP
#define BOOSTED_LEARNING_TRAININGDATA_HPP

#include "Feature.hpp"

#include <boost/shared_ptr.hpp>
#include "helpers/get_option_value.hpp"

#include <iostream>
#include <fstream>

namespace doppia
{
class IntegralChannelsFromFiles; // forward declaration
}

namespace boosted_learning
{


/// This class stores the strict minimum information required to be able to do Adaboost training.
/// In particular, we do _not_ store the integral images, but only the features responses
class TrainingData
{
public:

    typedef doppia::geometry::point_xy<int> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;

    typedef std::vector<double> weights_t;


    typedef ImageData meta_datum_t;
    typedef std::vector<meta_datum_t> meta_data_t;
    typedef ImageData::objectClassLabel_t objectClassLabel_t;

    typedef bootstrapping::integral_channels_t integral_channels_t;
    typedef bootstrapping::integral_channels_view_t integral_channels_view_t;
    typedef bootstrapping::integral_channels_const_view_t integral_channels_const_view_t;
    typedef std::vector<integral_channels_t> IntegralImages;

    typedef boost::shared_ptr<TrainingData> shared_ptr_t;
    typedef boost::shared_ptr<const TrainingData> const_shared_ptr_t;

    //typedef bootstrapping::integral_channels_computer_t integral_channels_computer_t;
    typedef doppia::AbstractIntegralChannelsComputer integral_channels_computer_t;
    typedef boost::shared_ptr<integral_channels_computer_t> integral_channels_computer_shared_ptr_t;

public:
    TrainingData(ConstFeaturesSharedPointer featuresConfigurations,
                 const std::vector<bool> &valid_features,
                 const size_t maxNumExamples,
                 const point_t modelWindow, const rectangle_t objectWindow,
                 integral_channels_computer_shared_ptr_t
                 integral_channels_computer_p = integral_channels_computer_shared_ptr_t());
    ~TrainingData();

    /// how many features in the pool ?
    size_t get_feature_pool_size() const;

    /// maximum number of examples that can be added ?
    size_t getMaxNumExamples() const;

    /// how many examples are currently here ?
    size_t get_num_examples() const;

    size_t get_num_positive_examples() const;
    size_t get_num_negative_examples() const;

    const rectangle_t & get_object_window() const;
    const point_t & get_model_window() const;

    /// meta-data access method,
    /// @param index is the training example index
    objectClassLabel_t get_class_label_for_sample(const size_t Index) const;

    /// meta-data access method,
    /// @param index is the training example index
    const string &get_file_name(const size_t Index) const;

    const ConstFeaturesSharedPointer getFeaturesConfigurations() const;


    /// the first index enumerates the features,
    /// the second index enumerates the training examples
    const FeaturesResponses &get_feature_responses() const;

    /// the first index enumerates the features,
    /// the second index enumerates the training examples
    const FeaturesBinResponses &get_feature_bin_responses() const;

    const Feature &get_feature(const size_t featureIndex) const;
    bool get_feature_validity(const size_t featureIndex) const;

    void setDatum(const size_t datumIndex,
                  const meta_datum_t &metaDatum, const bootstrapping::integral_channels_t &integralImage);


    void addPositiveSamples(const std::vector<std::string> &filenamesPositives,
                            const point_t &modelWindowSize, const point_t &dataOffset);

    void addHardNegativeSamples(const std::vector<std::string> &filenameHardNegatives,
                            const point_t &modelWindowSize, const point_t &dataOffset);

    void addNegativeSamples(const std::vector<std::string> &filenamesBackground,
                            const point_t &modelWindowSize, const point_t &dataOffset,
                            const size_t numNegativeSamplesToAdd);


    void addBootstrappingSamples(const std::string classifierPath,
                                 const std::vector<std::string> &filenamesBackground,
                                 const point_t &modelWindowSize, const point_t &dataOffset,
                                 const size_t numNegativeSamplesToAdd, const int maxFalsePositivesPerImage,
                                 const boost::program_options::variables_map &options);

    bintype get_bin_size() const;
    void setupBins(int num_bins);

    void dumpfeature_responses(const std::string &filename) const;

protected:

    const objectClassLabel_t _backgroundClassLabel; ///< Label of the class for background images

    ConstFeaturesSharedPointer _featuresConfigurations;
public:
    std::vector<bool> _validFeatures;
protected:
    point_t _modelWindow;
    rectangle_t _objectWindow;
    bintype _num_bins;

    FeaturesResponses _feature_responses;
    FeaturesBinResponses _featureBinResponses;

    meta_data_t _metaData; ///< labels of the classes
    size_t _numPositivesExamples, _numNegativesExamples;

    integral_channels_computer_shared_ptr_t _integralChannelsComputer;

};


} // end of namespace boosted_learning

#endif // BOOSTED_LEARNING_TRAININGDATA_HPP
