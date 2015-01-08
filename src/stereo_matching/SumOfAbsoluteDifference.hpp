#ifndef SUMOFABSOLUTEDIFFERENCE_HPP
#define SUMOFABSOLUTEDIFFERENCE_HPP

#include "AbstractStereoBlockMatcher.hpp"


namespace doppia
{

using namespace boost;
using namespace std;


//  ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

class SumOfAbsoluteDifference: public AbstractStereoBlockMatcher

{

public:

private:


public:
    SumOfAbsoluteDifference();


    static boost::program_options::options_description get_args_options(void);

    SumOfAbsoluteDifference(const boost::program_options::variables_map &options);
    ~SumOfAbsoluteDifference();

    void set_rectified_images_pair( gil::any_image<input_images_t>::const_view_t &left, gil::any_image<input_images_t>::const_view_t &right);

    void compute_disparity_map(gil::gray8c_view_t &left, gil::gray8c_view_t &right, bool left_right_are_inverted);

    void compute_disparity_map(gil::rgb8c_view_t  &left, gil::rgb8c_view_t &right, bool left_right_are_inverted);


private:


};


} // end of namespace bilcop


#endif // SUMOFABSOLUTEDIFFERENCE_HPP
