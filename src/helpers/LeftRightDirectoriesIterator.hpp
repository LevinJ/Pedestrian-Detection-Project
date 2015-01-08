#ifndef LEFTRIGHTDIRECTORIESITERATOR_HPP
#define LEFTRIGHTDIRECTORIESITERATOR_HPP


#include "IStereoImagesIterator.hpp"



// standard C++
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <utility> // defines class pair
#include <set>
using namespace std;

#include "for_each.hpp"

#include <CImg/CImg.h>
using namespace cimg_library;

#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>

#include <boost/scoped_ptr.hpp>



using namespace boost::filesystem;

class LeftRightDirectoriesIterator: public IStereoImagesIterator< CImg<uint8_t> >
{


public:
    typedef CImg<uint8_t> image_type;

    LeftRightDirectoriesIterator();

    LeftRightDirectoriesIterator(
        const directory_iterator &_left_itr,
        const directory_iterator &_right_itr);

    LeftRightDirectoriesIterator(const LeftRightDirectoriesIterator& it);

    ~LeftRightDirectoriesIterator();

    ///@name IStereoImagesIterator interface implementation
    ///{
    bool operator==(const LeftRightDirectoriesIterator &it);
    bool operator!=(const LeftRightDirectoriesIterator &it);

    const IStereoImagesIterator<image_type> & operator++();

    const image_type& left_image() const;
    const image_type& right_image() const;
    ///}

private:
    void read_first_images();
    void read_images();

    directory_iterator left_itr, right_itr, end_itr;
    image_type _left_image, _right_image;
    bool at_end;

};

#endif // LEFTRIGHTDIRECTORIESITERATOR_HPP
