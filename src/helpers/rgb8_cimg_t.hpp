
#if not defined(RGB8_CIMG_T_HEADER)
#define RGB8_CIMG_T_HEADER


// import bug fixed version file
#include "../libs/boost/gil/color_base_algorithm.hpp"
#include "../libs/boost/gil/pixel.hpp"

#include <boost/gil/gil_all.hpp>

#include <stdexcept>

#include <CImg/CImg.h>


namespace cimg_library
{

using boost::gil::bits8;
using boost::gil::bits32f;
using boost::gil::image_view;
using boost::gil::rgb8_planar_image_t;
using boost::gil::rgb32f_planar_image_t;
using boost::gil::gray8_image_t;
using boost::gil::gray32f_image_t;

/**
Helper class that allows to exchange between image_view and CImg

GilPlanarT and CImgPixelT should be compatible types
GilPlanarT == rgb8_planar_image_t, C == bits8
*/


template<typename GilPlanarT, typename CImgPixelT>
class gil_cimg_t : public CImg<CImgPixelT>
{

public:

    typedef typename GilPlanarT::view_t view_t;

    typedef CImg<CImgPixelT> cimg_t;

    gil_cimg_t(const int width, const int height);
    gil_cimg_t(const boost::gil::point2<int> &size);
    gil_cimg_t(const boost::gil::point2<long int> &size);
    ~gil_cimg_t();


    template<typename T>
    void assign(const image_view<T> &view);

    template<typename T>
    void operator=(const image_view<T> &view);

    const view_t &get_view();

private:

    GilPlanarT planar_image;
    view_t view;

    void init(const int width, const int height);
};



template<typename GilPlanarT, typename CImgPixelT>
gil_cimg_t<GilPlanarT, CImgPixelT>::gil_cimg_t(const int width, const int height)
{

    init(width, height);
    return;
}

template<typename GilPlanarT, typename CImgPixelT>
gil_cimg_t<GilPlanarT, CImgPixelT>::gil_cimg_t(const boost::gil::point2<int> &size)
{

    init(size.x, size.y);
    return;
}

template<typename GilPlanarT, typename CImgPixelT>
gil_cimg_t<GilPlanarT, CImgPixelT>::gil_cimg_t(const boost::gil::point2<long int> &size)
{

    init(static_cast<int>(size.x), static_cast<int>(size.y));
    return;
}

template<typename GilPlanarT, typename CImgPixelT>
gil_cimg_t<GilPlanarT, CImgPixelT>::~gil_cimg_t()
{
    // nothing to do here
    return;
}


template<typename GilPlanarT, typename CImgPixelT>
void gil_cimg_t<GilPlanarT, CImgPixelT>::init(const int width, const int height)
{

    planar_image.recreate(width, height);
    view = boost::gil::view(planar_image);

    const CImgPixelT *data_p = planar_view_get_raw_data(view, 0);
    static_cast< CImg<CImgPixelT> *>(this)->assign( data_p, width, height, 1, boost::gil::num_channels<GilPlanarT>(), true );

    return;
}


template<typename GilPlanarT, typename CImgPixelT>
template<typename T>
void gil_cimg_t<GilPlanarT, CImgPixelT>::assign(const image_view<T> &reference_view)
{
    copy_and_convert_pixels(reference_view, view);
    return;
}



template<typename GilPlanarT, typename CImgPixelT>
template<typename T>
void gil_cimg_t<GilPlanarT, CImgPixelT>::operator=(const image_view<T> &reference_view)
{
    assign(reference_view);
    return;
}

template<typename GilPlanarT, typename CImgPixelT>
const typename gil_cimg_t<GilPlanarT, CImgPixelT>::view_t & gil_cimg_t<GilPlanarT, CImgPixelT>::get_view()
{
    if(view.size() == 0)
    {
        throw std::runtime_error("gil_cimg_t::get_view requesting the view of an unasigned image");
    }

    return view;
}

typedef gil_cimg_t<rgb8_planar_image_t,bits8> rgb8_cimg_t;
typedef gil_cimg_t<rgb32f_planar_image_t, bits32f> rgb32f_cimg_t;

typedef gil_cimg_t<gray8_image_t, bits8> gray8_cimg_t;
typedef gil_cimg_t<gray32f_image_t, bits32f> gray32f_cimg_t;


} // end of namespace cimg_library


#endif // not defined(RGB8_CIMG_T_HEADER)
