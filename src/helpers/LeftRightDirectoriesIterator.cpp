#include "LeftRightDirectoriesIterator.hpp"


using namespace std;

void LeftRightDirectoriesIterator::read_first_images()
{
    // search for the first valid files ---
    bool valid_files = true;
    do
    {
        valid_files = true;
        if (at_end) return; // reached the end, nothing to do

        if ( is_directory( *left_itr ) || is_directory( *right_itr ) )
        {
            cout << "Omiting subdirectories "
                 <<  left_itr->string() << " and "
                 << right_itr->string() << endl;
            valid_files = false;
        }
        else
            break;

        ++left_itr;
        ++right_itr;
        at_end = (left_itr == end_itr) || (right_itr == end_itr);

    }
    while (valid_files == false);

    // got valid files, open the images --
    read_images();
    return;
}

LeftRightDirectoriesIterator::LeftRightDirectoriesIterator()
{
    at_end = (left_itr == end_itr) || (right_itr == end_itr);
    assert(at_end == true);
    return;
}

LeftRightDirectoriesIterator::LeftRightDirectoriesIterator(
    const directory_iterator &_left_itr,
    const directory_iterator &_right_itr)
{
    left_itr = _left_itr;
    right_itr = _right_itr;
    at_end = (left_itr == end_itr) || (right_itr == end_itr);


    read_first_images();
    return;
}

LeftRightDirectoriesIterator::LeftRightDirectoriesIterator(
    const LeftRightDirectoriesIterator& it)
        : 	IStereoImagesIterator<image_type>()
{

    left_itr = it.left_itr;
    right_itr = it.right_itr;

    at_end = (left_itr == end_itr) || (right_itr == end_itr);

    read_first_images();

    return;
}


LeftRightDirectoriesIterator::~LeftRightDirectoriesIterator()
{
    // nothing to do here
    return;
}

bool LeftRightDirectoriesIterator::operator==(const LeftRightDirectoriesIterator &it)
{
    if (it.at_end && this->at_end)
        return true;
    else
        return (it.left_itr == left_itr) && (it.right_itr == right_itr);
}


bool LeftRightDirectoriesIterator::operator!=(const LeftRightDirectoriesIterator &it)
{
    if (it.at_end && this->at_end)
        return false;
    else
        return (it.left_itr != left_itr) || (it.right_itr != right_itr);
}

void LeftRightDirectoriesIterator::read_images()
{
    if (true)
    {
        cout << "Reading "  << "\t"
             << left_itr->string() << "\t"
             << right_itr->string() << endl;
    }
    _left_image.assign(left_itr->string().c_str());
    _right_image.assign(right_itr->string().c_str());

    return;
}

const IStereoImagesIterator<LeftRightDirectoriesIterator::image_type> &
LeftRightDirectoriesIterator::operator++()
{

    if (at_end)
        throw std::runtime_error(
            "LeftRightDirectoriesIterator::operator++ called over "\
            "an iterator already at the end of one of the directories");
    else
    {
        bool valid_files = true;
        do
        {
            valid_files = true;
            ++left_itr;
            ++right_itr;
            at_end = (left_itr == end_itr) || (right_itr == end_itr);
            if (at_end) return *this; // reached the end, nothing to do

            if ( is_directory( *left_itr ) || is_directory( *right_itr ) )
            {
                cout << "Omiting subdirectories "
                     <<  left_itr->string() << " and "
                     << right_itr->string() << endl;
                valid_files = false;
            }
        }
        while (valid_files == false);

        // got valid files, open the images --
        read_images();
    }
    return *this;
}


const LeftRightDirectoriesIterator::image_type &
LeftRightDirectoriesIterator::left_image() const

{
    return _left_image;
}

const LeftRightDirectoriesIterator::image_type &
LeftRightDirectoriesIterator::right_image() const
{
    return _right_image;
}


// Iterable left right directories.
// IStereoImagesIterator provider
class LeftRightDirectories
{

    directory_iterator left_itr, right_itr;
    boost::scoped_ptr<LeftRightDirectoriesIterator>
    begin_iterator_p, end_iterator_p;
public:
    LeftRightDirectories(const string &directory_name);
    ~LeftRightDirectories();

    typedef LeftRightDirectoriesIterator  iterator;
    typedef LeftRightDirectoriesIterator const_iterator;
    // <<< how to define properly a const_iterator ? and non const_one ?

    const_iterator &begin();
    const_iterator &end();

};

LeftRightDirectories::LeftRightDirectories(const string &input_directory_name)
{

    path input_images_path(input_directory_name);

    if ( !exists(input_images_path))
        throw runtime_error("Indicated input images path does not exist");

    if ( !is_directory(input_images_path) )
        throw runtime_error("Indicated input images path is not a directory");

    path left_images_path( input_images_path / "left" );
    path right_images_path( input_images_path / "right" );

    if ( !exists(left_images_path)
            || !exists(right_images_path)
            || !is_directory(left_images_path)
            || !is_directory(right_images_path) )
        throw runtime_error("Did not find the 'left' and 'right' input subdirectories");

    directory_iterator left_itr( left_images_path ), right_itr(right_images_path);
    begin_iterator_p.reset(new LeftRightDirectoriesIterator(left_itr, right_itr));

    directory_iterator end_itr; // default construction yields past-the-end
    end_iterator_p.reset(new LeftRightDirectoriesIterator(end_itr, end_itr));

    return;
}

LeftRightDirectories::~LeftRightDirectories()
{
    // nothing to do here
    return;
}

LeftRightDirectories::const_iterator &LeftRightDirectories::begin()
{
    return *begin_iterator_p;
};

LeftRightDirectories::const_iterator &LeftRightDirectories::end()
{
    return *end_iterator_p;
};


