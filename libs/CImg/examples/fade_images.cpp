/*
 #
 #  File        : fade_images.cpp
 #                ( C++ source file )
 #
 #  Description : Compute a linear fading between two images.
 #                This file is a part of the CImg Library project.
 #                ( http://cimg.sourceforge.net )
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

#include "CImg.h"

// The lines below are necessary when using a non-standard compiler as visualcpp6.
#ifdef cimg_use_visualcpp6
#define std
#endif
#ifdef min
#undef min
#undef max
#endif

#ifndef cimg_imagepath
#define cimg_imagepath "img/"
#endif

// Main procedure
//---------------
int main(int argc,char **argv) {

  // Read and check command line parameters
  cimg_usage("Compute a linear fading between two 2D images");
  const char *file_i1 = cimg_option("-i1",cimg_imagepath "sh0r.pgm","Input Image 1");
  const char *file_i2 = cimg_option("-i2",cimg_imagepath "milla.bmp","Input Image 2");
  const char *file_o  = cimg_option("-o",(char*)0,"Output Image");
  const bool visu     = cimg_option("-visu",true,"Visualization mode");
  const double pmin   = cimg_option("-min",40.0,"Begin of the fade (in %)")/100.0;
  const double pmax   = cimg_option("-max",60.0,"End of the fade (in %)")/100.0;
  const double angle  = cimg_option("-angle",0.0,"Fade angle")*cimg_library::cimg::valuePI/180;

  // Init images
  cimg_library::CImg<unsigned char> img1(file_i1), img2(file_i2);
  if (img2.dimx()!=img1.dimx() || img2.dimy()!=img1.dimy() || img2.dimz()!=img1.dimz() || img2.dimv()!=img1.dimv()) {
    int
      dx = cimg_library::cimg::max(img1.dimx(),img2.dimx()),
      dy = cimg_library::cimg::max(img1.dimy(),img2.dimy()),
      dz = cimg_library::cimg::max(img1.dimz(),img2.dimz()),
      dv = cimg_library::cimg::max(img1.dimv(),img2.dimv());
    img1.resize(dx,dy,dz,dv,3);
    img2.resize(dx,dy,dz,dv,3);
  }
  cimg_library::CImg<unsigned char> dest(img1.dimx(),img1.dimy(),img1.dimz(),img1.dimv());

  // Compute the faded image
  const double ca=std::cos(angle), sa=std::sin(angle);
  double alpha;
  cimg_forXYZV(dest,x,y,z,k) {
    const double X = ((double)x/img1.dimx()-0.5)*ca + ((double)y/img1.dimy()-0.5)*sa;
    if (X+0.5<pmin) alpha=0; else {
      if (X+0.5>pmax) alpha=1; else alpha=(X+0.5-pmin)/(pmax-pmin);
    }
    dest(x,y,z,k) = (unsigned char)((1-alpha)*img1(x,y,z,k) + alpha*img2(x,y,z,k));
  }

  // Save and exit
  if (file_o) dest.save(file_o);
  if (visu) dest.display("Image fading");
  return 0;
}
