/*
  #
  #  File        : gmic.h
  #                ( C++ header file )
  #
  #  Description : GREYC's Magic Image Converter
  #                ( http://gmic.sourceforge.net )
  #                This file is a part of the CImg Library project.
  #                ( http://cimg.sourceforge.net )
  #
  #  Note        : This file cannot be compiled on VC++ 6.
  #
  #  Copyright   : David Tschumperle
  #                ( http://www.greyc.ensicaen.fr/~dtschump/ )
  #
  #  License     : CeCILL v2.0
  #                ( http://www.cecill.info/licences/Licence_CeCILL_V2-en.html )
  #
  #  This software is governed by the CeCILL  license under French law and
  #  abiding by the rules of distribution of free software.  You can  use,
  #  modify and/ or redistribute the software under the terms of the CeCILL
  #  license as circulated by CEA, CNRS and INRIA at the following URL
  #  "http://www.cecill.info".
  #
  #  As a counterpart to the access to the source code and  rights to copy,
  #  modify and redistribute granted by the license, users are provided only
  #  with a limited warranty  and the software's author,  the holder of the
  #  economic rights,  and the successive licensors  have only  limited
  #  liability.
  #
  #  In this respect, the user's attention is drawn to the risks associated
  #  with loading,  using,  modifying and/or developing or reproducing the
  #  software by the user in light of its specific status of free software,
  #  that may mean  that it is complicated to manipulate,  and  that  also
  #  therefore means  that it is reserved for developers  and  experienced
  #  professionals having in-depth computer knowledge. Users are therefore
  #  encouraged to load and test the software's suitability as regards their
  #  requirements in conditions enabling the security of their systems and/or
  #  data to be ensured and,  more generally, to use and operate it in the
  #  same conditions as regards security.
  #
  #  The fact that you are presently reading this means that you have had
  #  knowledge of the CeCILL license and that you accept its terms.
  #
*/

#ifndef gmic_version
#include "CImg.h"
#define gmic_version 1317

// The lines below are necessary when using a non-standard compiler such as visualcpp6.
#ifdef cimg_use_visualcpp6
#define std
#endif
#ifdef min
#undef min
#undef max
#endif

// Define G'MIC Exception class.
//------------------------------
struct gmic_exception {
  char message[4096];
  gmic_exception();
  gmic_exception(const char *format, ...);
  gmic_exception(const char *format, std::va_list ap);
};

// Define G'MIC interpreter class.
//--------------------------------
struct gmic {

  // Internal variables.
  cimg_library::CImgDisplay instant_window;
  cimg_library::CImgList<char> macros, commands, scope;
  cimg_library::CImg<char> is_trace;
  float focale3d, light3d_x, light3d_y, light3d_z, specular_light3d, specular_shine3d;
  bool is_released, is_debug, is_fullpath, is_begin, is_end, is_quit, is_oriented3d;
  int verbosity_level, render3d, renderd3d;
  unsigned char background3d[3];
  unsigned int position;

  // Constructors - Destructors.
  gmic(const char *const command, const char *const custom_macros=0, const bool default_macros=true);
  template<typename T> gmic(const int argc, const char *const *const argv, cimg_library::CImgList<T>& images,
                            const char *const custom_macros=0, const bool default_macros=true);
  template<typename T> gmic(const char *const command, cimg_library::CImgList<T>& images,
                            const char *const custom_macros=0, const bool default_macros=true);
  gmic& assign(const char *const custom_macros=0, const bool default_macros=true);

  // Messages procedures.
  cimg_library::CImg<char> scope2string() const;
  static cimg_library::CImgList<char> commandline_to_CImgList(const char *const command);
  template<typename T>
  const gmic& error(const cimg_library::CImgList<T>& list, const char *format, ...) const;
  const gmic& error(const char *format, ...) const;
  template<typename T>
  const gmic& warning(const cimg_library::CImgList<T>& list, const char *format, ...) const;
  const gmic& warning(const char *format, ...) const;
  template<typename T>
  const gmic& debug(const cimg_library::CImgList<T>& list, const char *format, ...) const;
  const gmic& debug(const char *format, ...) const;
  template<typename T>
  const gmic& print(const cimg_library::CImgList<T>& list, const char *format, ...) const;
  const gmic& print(const char *format, ...) const;

  // Add macros.
  gmic& add_macros(const char *const data_macros);
  gmic& add_macros(std::FILE *const file);

  // Return indices of the images from a string.
  cimg_library::CImg<unsigned int> indices2cimg(const char *const string, const unsigned int indice_max,
                                                const char *const command) const;

  // Return stringified version of indices or filenames.
  char* indices2string(const cimg_library::CImg<unsigned int>& indices,
                       const cimg_library::CImgList<char>& filenames,
                       const bool display_indices) const;

  // Display image data.
  template<typename T>
  bool display_images(const cimg_library::CImgList<T>& images,
                      const cimg_library::CImgList<char>& filenames,
                      const cimg_library::CImg<unsigned int>& indices,
                      const bool verbose) const;
  template<typename T>
  bool display_objects3d(const cimg_library::CImgList<T>& images,
                         const cimg_library::CImgList<char>& filenames,
                         const cimg_library::CImg<unsigned int>& indices,
                         const bool verbose) const;
  template<typename T>
  bool display_plots(const cimg_library::CImgList<T>& images,
                     const cimg_library::CImgList<char>& filenames,
                     const cimg_library::CImg<unsigned int>& indices,
                     const unsigned int plot_type, const unsigned int vertex_type,
                     const double xmin, const double xmax,
                     const double ymin, const double ymax,
                     const bool verbose) const;

  // Substitute '@' and '{}' expressions.
  template<typename T>
  bool substitute_item(const char *const source, char *const destination,
                       const cimg_library::CImgList<T>& images, const cimg_library::CImgList<unsigned int>& repeatdones) const;

  // Main parsing procedure.
  template<typename T>
  gmic& parse(const cimg_library::CImgList<char>& command_line, unsigned int& position,
              cimg_library::CImgList<T> &images, cimg_library::CImgList<char> &filenames,
              cimg_library::CImgList<unsigned int>& dowhiles, cimg_library::CImgList<unsigned int>& repeatdones,
              cimg_library::CImgList<unsigned int>& locals, const bool initial_call);
  gmic& parse_bool(const cimg_library::CImgList<char>& command_line, unsigned int& position,
                   cimg_library::CImgList<bool>& images, cimg_library::CImgList<char> &filenames,
                   cimg_library::CImgList<unsigned int>& dowhiles, cimg_library::CImgList<unsigned int>& repeatdones,
                   cimg_library::CImgList<unsigned int>& locals, const bool initial_call);
  gmic& parse_uchar(const cimg_library::CImgList<char>& command_line, unsigned int& position,
                    cimg_library::CImgList<unsigned char>& images, cimg_library::CImgList<char> &filenames,
                    cimg_library::CImgList<unsigned int>& dowhiles, cimg_library::CImgList<unsigned int>& repeatdones,
                    cimg_library::CImgList<unsigned int>& locals, const bool initial_call);
  gmic& parse_char(const cimg_library::CImgList<char>& command_line, unsigned int& position,
                   cimg_library::CImgList<char>& images, cimg_library::CImgList<char> &filenames,
                   cimg_library::CImgList<unsigned int>& dowhiles, cimg_library::CImgList<unsigned int>& repeatdones,
                   cimg_library::CImgList<unsigned int>& locals, const bool initial_call);
  gmic& parse_ushort(const cimg_library::CImgList<char>& command_line, unsigned int& position,
                     cimg_library::CImgList<unsigned short>& images, cimg_library::CImgList<char> &filenames,
                     cimg_library::CImgList<unsigned int>& dowhiles, cimg_library::CImgList<unsigned int>& repeatdones,
                     cimg_library::CImgList<unsigned int>& locals, const bool initial_call);
  gmic& parse_short(const cimg_library::CImgList<char>& command_line, unsigned int& position,
                    cimg_library::CImgList<short>& images, cimg_library::CImgList<char> &filenames,
                    cimg_library::CImgList<unsigned int>& dowhiles, cimg_library::CImgList<unsigned int>& repeatdones,
                    cimg_library::CImgList<unsigned int>& locals, const bool initial_call);
  gmic& parse_uint(const cimg_library::CImgList<char>& command_line, unsigned int& position,
                   cimg_library::CImgList<unsigned int>& images, cimg_library::CImgList<char> &filenames,
                   cimg_library::CImgList<unsigned int>& dowhiles, cimg_library::CImgList<unsigned int>& repeatdones,
                   cimg_library::CImgList<unsigned int>& locals, const bool initial_call);
  gmic& parse_int(const cimg_library::CImgList<char>& command_line, unsigned int& position,
                  cimg_library::CImgList<int>& images, cimg_library::CImgList<char> &filenames,
                  cimg_library::CImgList<unsigned int>& dowhiles, cimg_library::CImgList<unsigned int>& repeatdones,
                  cimg_library::CImgList<unsigned int>& locals, const bool initial_call);
  gmic& parse_float(const cimg_library::CImgList<char>& command_line, unsigned int& position,
                    cimg_library::CImgList<float>& images, cimg_library::CImgList<char> &filenames,
                    cimg_library::CImgList<unsigned int>& dowhiles, cimg_library::CImgList<unsigned int>& repeatdones,
                    cimg_library::CImgList<unsigned int>& locals, const bool initial_call);
  gmic& parse_double(const cimg_library::CImgList<char>& command_line, unsigned int& position,
                     cimg_library::CImgList<double>& images, cimg_library::CImgList<char> &filenames,
                     cimg_library::CImgList<unsigned int>& dowhiles, cimg_library::CImgList<unsigned int>& repeatdones,
                     cimg_library::CImgList<unsigned int>& locals, const bool initial_call);

}; // End of the 'gmic' class.

cimg_library::CImgList<char> commandline_to_CImgList(const char *const command);

#endif

// Local Variables:
// mode: c++
// End:
