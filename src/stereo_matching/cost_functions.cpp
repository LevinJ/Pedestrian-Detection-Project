#include "cost_functions.hpp"

#include <stdexcept>

namespace doppia
{

template <>
template <typename PixelType>
float SadCostFunctionT<float>::get_maximum_cost_per_pixel() const {

    const int num_color_channels = gil::size< PixelType >::value;
    const float maximum_cost_per_channel = 255;

    return maximum_cost_per_channel * num_color_channels;
}

template <>
template <typename PixelType>
float SadCostFunctionT<int32_t>::get_maximum_cost_per_pixel() const {

    const int num_color_channels = gil::size< PixelType >::value;
    const float maximum_cost_per_channel = 255;

    return maximum_cost_per_channel * num_color_channels;
}

template <>
template <typename PixelType>
float SadCostFunctionT<uint8_t>::get_maximum_cost_per_pixel() const {

    return 255;
}


template float SadCostFunctionT<float>::get_maximum_cost_per_pixel<gil::gray8_pixel_t>() const;
template float SadCostFunctionT<float>::get_maximum_cost_per_pixel<gil::gray8c_pixel_t>() const;
template float SadCostFunctionT<float>::get_maximum_cost_per_pixel<gil::rgb8_pixel_t>() const;
template float SadCostFunctionT<float>::get_maximum_cost_per_pixel<gil::rgb8c_pixel_t>() const;

template int32_t SadCostFunctionT<int32_t>::get_maximum_cost_per_pixel<gil::gray8_pixel_t>() const;
template int32_t SadCostFunctionT<int32_t>::get_maximum_cost_per_pixel<gil::gray8c_pixel_t>() const;
template int32_t SadCostFunctionT<int32_t>::get_maximum_cost_per_pixel<gil::rgb8_pixel_t>() const;
template int32_t SadCostFunctionT<int32_t>::get_maximum_cost_per_pixel<gil::rgb8c_pixel_t>() const;

template uint8_t SadCostFunctionT<uint8_t>::get_maximum_cost_per_pixel<gil::gray8_pixel_t>() const;
template uint8_t SadCostFunctionT<uint8_t>::get_maximum_cost_per_pixel<gil::gray8c_pixel_t>() const;
template uint8_t SadCostFunctionT<uint8_t>::get_maximum_cost_per_pixel<gil::rgb8_pixel_t>() const;
template uint8_t SadCostFunctionT<uint8_t>::get_maximum_cost_per_pixel<gil::rgb8c_pixel_t>() const;

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-



template<typename PixelType>
float SsdCostFunction::get_maximum_cost_per_pixel() const {

    const int num_color_channels = gil::size< PixelType >::value;

    const float maximum_cost_per_channel = 255*255;

    return maximum_cost_per_channel * num_color_channels;
}

template float SsdCostFunction::get_maximum_cost_per_pixel<gil::gray8_pixel_t>() const;
template float SsdCostFunction::get_maximum_cost_per_pixel<gil::rgb8_pixel_t>() const;

template float SsdCostFunction::get_maximum_cost_per_pixel<gil::gray8c_pixel_t>() const;
template float SsdCostFunction::get_maximum_cost_per_pixel<gil::rgb8c_pixel_t>() const;

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-


LCDMCostFunction::LCDMCostFunction() {
    // nothing to do here
    return;
}

LCDMCostFunction::~LCDMCostFunction() {
    // nothing to do here
    return;
}

float LCDMCostFunction::operator()(const gil::gray8c_pixel_t &/*pixel_a*/,
                                   const gil::gray8c_pixel_t &/*pixel_b*/) const {

    throw std::runtime_error("LCDMCostFunction is not defined over gray pixels");

    return 0.0f;
}


template<typename PixelType>
float LCDMCostFunction::get_maximum_cost_per_pixel() const {

    const int num_color_channels = gil::size< PixelType >::value;

    const float maximum_cost_per_channel = 255*255;

    return maximum_cost_per_channel * num_color_channels;
}

template float LCDMCostFunction::get_maximum_cost_per_pixel<gil::gray8_pixel_t>() const;
template float LCDMCostFunction::get_maximum_cost_per_pixel<gil::rgb8_pixel_t>() const;

template float LCDMCostFunction::get_maximum_cost_per_pixel<gil::gray8c_pixel_t>() const;
template float LCDMCostFunction::get_maximum_cost_per_pixel<gil::rgb8c_pixel_t>() const;

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

GradientCostFunction::GradientCostFunction(const float gradient_weight_)
    : gradient_weight(gradient_weight_), color_weight(1.0f - gradient_weight_)
{
    // nothing to do here
    return;
}

GradientCostFunction::~GradientCostFunction() {
    // nothing to do here
    return;
}


template<typename PixelType>
float GradientCostFunction::get_maximum_cost_per_pixel() const {

    const int num_channels = gil::size< PixelType >::value;
    const int num_gradient_channels = 2;
    const int num_color_channels = num_channels - num_gradient_channels;

    const float maximum_cost_per_color_channel = 255;
    const float maximum_cost_per_gradient_channel = 255;

    return color_weight*(maximum_cost_per_color_channel * num_color_channels) +
            gradient_weight*(maximum_cost_per_gradient_channel*num_gradient_channels);
}

template float GradientCostFunction::get_maximum_cost_per_pixel<gil::dev3n8_pixel_t>() const;
template float GradientCostFunction::get_maximum_cost_per_pixel<gil::dev5n8_pixel_t>() const;

template float GradientCostFunction::get_maximum_cost_per_pixel<gil::dev3n8c_pixel_t>() const;
template float GradientCostFunction::get_maximum_cost_per_pixel<gil::dev5n8c_pixel_t>() const;

// ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

} // end of namespace doppia
