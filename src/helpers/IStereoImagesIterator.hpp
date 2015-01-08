#ifndef ISTEREOIMAGESITERATOR_HPP
#define ISTEREOIMAGESITERATOR_HPP



template<typename ImageType>
class IStereoImagesIterator
{
public:
    // methods to be implemented by childs

    virtual const IStereoImagesIterator<ImageType> & operator++() = 0;
    ///< prefix operator

    virtual const ImageType& right_image() const = 0;
    virtual const ImageType& left_image()  const = 0;

public:
    // base methods
    const IStereoImagesIterator<ImageType> *operator->() const
    {
        return this;
    }

    IStereoImagesIterator()
    {
        return;
    }

    virtual ~IStereoImagesIterator()
    {
        return;
    }
};



#endif // ISTEREOIMAGESITERATOR_HPP
