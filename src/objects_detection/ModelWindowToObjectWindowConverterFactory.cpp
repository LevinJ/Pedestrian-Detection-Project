#include "ModelWindowToObjectWindowConverterFactory.hpp"

#include "AbstractModelWindowToObjectWindowConverter.hpp"

#include "FastModelWindowToObjectWindowConverter.hpp"
#include "ModelWindowToObjectWindowConverter.hpp"

#include <cstdio>

namespace doppia {


AbstractModelWindowToObjectWindowConverter *
ModelWindowToObjectWindowConverterFactory::new_instance(const model_window_size_t &model_window_size,
                                                        const object_window_t &object_window)
{

    const int
            object_center_x = (object_window.max_corner().x() + object_window.min_corner().x())/2,
            object_center_y = (object_window.max_corner().y() + object_window.min_corner().y())/2;

    if(false)
    {
        printf("object_center x,y == %i, %i\n", object_center_x, object_center_y);
        printf("object_window min x,y ==%i,%i, object_window max x,y ==%i,%i\n", object_window.min_corner().x(), object_window.min_corner().y()
               , object_window.max_corner().x(), object_window.max_corner().y());
        printf("model_window_size x,y == %i, %i\n", model_window_size.x(), model_window_size.y());
    }

    // check if windows are centered
    bool windows_are_centered = false;

    // we give a 1 pixel slack to handle impair numbers
    if((std::abs(object_center_x - static_cast<int>(model_window_size.x()/2)) <= 1) and
            (std::abs(object_center_y - static_cast<int>(model_window_size.y()/2)) <= 1))
    {

        windows_are_centered = true;
    }

    if(windows_are_centered)
    {
        return new FastModelWindowToObjectWindowConverter(model_window_size, object_window);
    }
    // else

    return new ModelWindowToObjectWindowConverter(model_window_size, object_window);
}

} // namespace doppia
