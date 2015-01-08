#ifndef SPLITIMAGESDIRECTORYITERATOR_HPP
#define SPLITIMAGESDIRECTORYITERATOR_HPP

#include "IStereoImagesIterator.hpp"




#include <CImg/CImg.h>
using namespace cimg_library;


#include <boost/filesystem/operations.hpp>


using namespace boost::filesystem;

class SplitImagesDirectoryIterator: public IStereoImagesIterator< CImg<uint8_t> >
{


public:
    typedef CImg<uint8_t> image_type;

    SplitImagesDirectoryIterator();

    SplitImagesDirectoryIterator(const directory_iterator &_directory_itr);

    SplitImagesDirectoryIterator(const SplitImagesDirectoryIterator& it);

    ~SplitImagesDirectoryIterator();

    ///@name IStereoImagesIterator interface implementation
    ///{
    bool operator==(const SplitImagesDirectoryIterator &it);
    bool operator!=(const SplitImagesDirectoryIterator &it);

    const IStereoImagesIterator<image_type> & operator++();

    const image_type& left_image()  const;
    const image_type& right_image() const;
    ///}

private:
    void read_first_images();
    void read_images();
    directory_iterator directory_itr, end_itr;
    image_type original_image, _left_image, _right_image;
};
#endif // SPLITIMAGESDIRECTORYITERATOR_HPP
