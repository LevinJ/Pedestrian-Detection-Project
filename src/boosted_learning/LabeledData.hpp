#ifndef __LABELED_DATA_H
#define __LABELED_DATA_H

#include "ImageData.hpp"

#include "helpers/geometry.hpp"

#include "applications/bootstrapping_lib/bootstrapping_lib.hpp"

#include <string>
#include <map>
#include <cassert>
#include <vector>


namespace boosted_learning {

using namespace std;
class StrongClassifier;

/// @deprecated, in general there is no enough memory to store _all_ the labeled data
/// this data container must die !
class LabeledData
{
public:
    typedef boost::shared_ptr<LabeledData> shared_ptr;
    typedef bootstrapping::integral_channels_t integral_channels_t;
    typedef bootstrapping::integral_channels_view_t integral_channels_view_t;
    typedef bootstrapping::integral_channels_const_view_t integral_channels_const_view_t;

    typedef doppia::geometry::point_xy<int> point_t;
    typedef doppia::geometry::box<point_t> rectangle_t;

    typedef std::vector<integral_channels_t> IntegralImages;
    typedef ImageData meta_datum_t;
    typedef std::vector<meta_datum_t> meta_data_t;

    LabeledData(bool silent_mode, int backgroundClassLabel);
    ~LabeledData();

    void createIntegralImages(const std::vector<std::string> &filenamesPositives,
                              const std::vector<std::string> &filenamesBackground,
                              const point_t modelWindow,
                              const int offsetX, const int offsetY,
                              const boost::program_options::variables_map &options,
                              const string bootStrapLearnerFile = std::string());

    /**
       * Returns the whole integral image of example \a Index.
       * @param Index The index of the example.
       */
    const doppia::IntegralChannelsForPedestrians::integral_channels_t &get_integral_image(const size_t Index) const;

    /// meta-data access method,
    /// @param index is the training example index
    int get_class_label_for_sample(const size_t Index) const;

    /// meta-data access method,
    /// @param index is the training example index
    const string &get_file_name(const size_t Index) const;

    int getX(const size_t Index) const;
    int getY(const size_t Index) const;

    const meta_datum_t &getMetaDatum(const size_t Index) const;
    const meta_data_t &getMetaData() const;

    size_t get_num_examples() const;
    size_t get_num_pos_examples() const;
    size_t getNumNegExamples() const;

    //void bootstrap(const StrongClassifier & classifier, int maxNr,std::vector<std::string> & filenames_background);
    void bootstrap(const std::string classifierName,
                   const point_t modelWindow, const int offsetX, const int offsetY,
                   const int maxNr, const int maxPerImage,
                   const std::vector<std::string> & filenames_background,
                   const boost::program_options::variables_map &options);

    void add(const ImageData &data, const integral_channels_t &integralImage);


    /// erases the currently stored integralImages and metaData
    void clear();

protected:

    bool _silent_mode;
    int _backgroundClassLabel; ///< Label of the class for background images

    /// @deprecated we need to get rid of this memory store
    IntegralImages _integralImages; ///< the data of the examples.
    meta_data_t _metaData; ///< labels of the classes
    size_t _numPosExamples, _numNegExamples;

};


} // end of namespace boosted_learning

#endif // __LABELED_DATA_H
