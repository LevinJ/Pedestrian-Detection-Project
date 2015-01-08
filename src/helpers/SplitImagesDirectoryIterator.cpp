#include "SplitImagesDirectoryIterator.hpp"

// standard C++
#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <algorithm>
#include <utility> // defines class pair
#include <set>
using namespace std;


#include <boost/filesystem/path.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>

using namespace std;

void SplitImagesDirectoryIterator::read_first_images()
{

    bool valid_files = true;
    do
    {
        valid_files = true;
        if (directory_itr == end_itr) return; // reached the end, nothing to do

        if ( is_directory( *directory_itr ) )
        {
            cout << "Omiting subdirectory "
                 <<  directory_itr->string() << endl;
            valid_files = false;
        }
        else
            break;

        ++directory_itr;
    }
    while (valid_files == false);

    // got valid files, open the images --
    read_images();

    return;
}

SplitImagesDirectoryIterator::SplitImagesDirectoryIterator()
{

    return;
}

SplitImagesDirectoryIterator::SplitImagesDirectoryIterator(
    const directory_iterator &_directory_itr)
        : directory_itr(_directory_itr)
{
    read_first_images();
    return;
}

SplitImagesDirectoryIterator::SplitImagesDirectoryIterator(
    const SplitImagesDirectoryIterator& it)
        : 	IStereoImagesIterator<image_type>()
{

    directory_itr = it.directory_itr;
    read_first_images();
    return;
}


SplitImagesDirectoryIterator::~SplitImagesDirectoryIterator()
{
    // nothing to do here
    return;
}

bool SplitImagesDirectoryIterator::operator==(const SplitImagesDirectoryIterator &it)
{
    return (directory_itr == it.directory_itr);
}


bool SplitImagesDirectoryIterator::operator!=(const SplitImagesDirectoryIterator &it)
{
    return (directory_itr != it.directory_itr);
}

void SplitImagesDirectoryIterator::read_images()
{
    if (true)
    {
        cout << "Reading "  << "\t"
             << directory_itr->string() << endl;
    }

    original_image.assign(directory_itr->string().c_str());

    int dimx = original_image.dimx();
    const int dimy = original_image.dimy();
    if ((dimx % 2) == 1) dimx -= 1; // ensures dimx to be an even number

    _left_image = original_image.get_crop(0,0, dimx/2,dimy); // left side
    _right_image = original_image.get_crop(dimx/2,0, dimx,dimy); // right side

    return;
}


const IStereoImagesIterator<SplitImagesDirectoryIterator::image_type> &
SplitImagesDirectoryIterator::operator++()
{

    if (directory_itr == end_itr)
        throw std::runtime_error(
            "LeftRightDirectoriesIterator::operator++ called over "\
            "an iterator already at the end of one of the directories");
    else
    {
        bool valid_files = true;
        do
        {
            valid_files = true;
            ++directory_itr;
            if (directory_itr == end_itr) return *this;

            if ( is_directory( *directory_itr ) )
            {
                cout << "Omiting subdirectory "
                     <<  directory_itr->string() << endl;
                valid_files = false;
            }
        }
        while (valid_files == false);

        // got valid files, open the images --
        read_images();
    }
    return *this;
}


const SplitImagesDirectoryIterator::image_type &
SplitImagesDirectoryIterator::left_image() const

{
    return _left_image;
}

const SplitImagesDirectoryIterator::image_type &
SplitImagesDirectoryIterator::right_image() const
{
    return _right_image;
}


// Iterate on a directory where left right images are stored in a single image file
class SplitImagesDirectory
{
    directory_iterator directory_it;
    boost::scoped_ptr<SplitImagesDirectoryIterator>
    begin_iterator_p, end_iterator_p;
public:
    SplitImagesDirectory(const string &directory_name);
    ~SplitImagesDirectory();

    typedef SplitImagesDirectoryIterator  iterator;
    typedef SplitImagesDirectoryIterator const_iterator;
    // <<< how to define properly a const_iterator ? and non const_one ?

    const_iterator &begin();
    const_iterator &end();
};

SplitImagesDirectory::SplitImagesDirectory(const string &input_directory_name)
{

    path input_images_path(input_directory_name);

    if ( !exists(input_images_path))
        throw runtime_error("Indicated input images path does not exist");

    if ( !is_directory(input_images_path) )
        throw runtime_error("Indicated input images path is not a directory");


    directory_iterator directory_it( input_directory_name );
    begin_iterator_p.reset(new SplitImagesDirectoryIterator(directory_it));

    // default construction yields past-the-end
    end_iterator_p.reset(new SplitImagesDirectoryIterator());

    return;
}

SplitImagesDirectory::~SplitImagesDirectory()
{
    // nothing to do here
    return;
}

SplitImagesDirectory::const_iterator &SplitImagesDirectory::begin()
{
    return *begin_iterator_p;
};

SplitImagesDirectory::const_iterator &SplitImagesDirectory::end()
{
    return *end_iterator_p;
};
