/*
 #
 #  File        : hough_transform.cpp
 #                ( C++ source file )
 #
 #  Description : Implementation of the Hough transform.
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
using namespace cimg_library;

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

int main(int argc,char **argv) {

  cimg_usage("Illustration of the Hough transform");
  CImg<unsigned char> src(cimg_option("-i",cimg_imagepath "parrot_original.ppm","Input image"));
  CImg<> vote(500,400,1,1,0), img = CImg<>(src).get_pointwise_norm().normalize(0,255).resize(-100,-100,1,2,2);

  CImgDisplay disp(src,"Image"), dispvote(vote,"Hough Transform");
  const unsigned char col1[3]={255,255,255}, col2[3]={0,0,0};
  const double
    alpha = cimg_option("-a",1.5,"Gradient smoothing"),
    sigma = cimg_option("-s",0.5,"Hough Transform smoothing"),
    rhomax = std::sqrt((double)(img.dimx()*img.dimx()+img.dimy()*img.dimy()))/2,
    thetamax = 2*cimg::valuePI;

  if (cimg::dialog(cimg::basename(argv[0]),
                   "Instructions : \n"
                   "------------\n\n"
                   "(1) When clicking on the image, all lines crossing the point\n"
                   "will be voted in the Hough Transform image.\n\n"
                   "(2) When clicking on the vote image, the corresponding line is drawn\n"
                   "on the image.\n\n"
                   "(3) When pressing the space bar, the image lines are detected from the\n"
                   "image gradients.\n\n"
                   "Note that a logarithmic scaling is performed for the vote image display.\n"
                   "See also the available options (option '-h')\n","Start !","Quit",0,0,0,0,
                   src.get_resize(100,100,1,3),true)) std::exit(0);

  while (!disp.is_closed && !dispvote.is_closed && !disp.is_keyQ && !dispvote.is_keyQ && !disp.is_keyESC && !dispvote.is_keyESC) {

    CImgDisplay::wait(disp,dispvote);

    // When pressing space bar, the vote is performed from the image gradients.
    if (dispvote.key==cimg::keySPACE || disp.key==cimg::keySPACE) {
      CImgList<> grad = img.get_gradient();
      cimglist_for(grad,l) grad[l].blur((float)alpha);
      vote.fill(0);
      cimg_forXY(img,x,y) {
        const double
          X = (double)x-img.dimx()/2,
          Y = (double)y-img.dimy()/2,
          gx = grad[0](x,y),
          gy = grad[1](x,y);
        double
          theta = std::atan2(gy,gx),
          rho   = std::sqrt(X*X+Y*Y)*std::cos(std::atan2(Y,X)-theta);
        if (rho<0) { rho=-rho; theta+=cimg::valuePI; }
        theta = cimg::mod(theta,thetamax);
        vote((int)(theta*dispvote.dimx()/thetamax),(int)(rho*dispvote.dimy()/rhomax))+=(float)std::sqrt(gx*gx+gy*gy);
      }
      vote.blur((float)sigma);
      CImg<> vote2(vote); { cimg_forXY(vote2,x,y) vote2(x,y) = (float)std::log(1+vote(x,y)); vote2.display(dispvote); }
    }

     // When clicking on the vote window.
    if (dispvote.button) {
      const double
        rho   = dispvote.mouse_y*rhomax/dispvote.dimy(),
        theta = dispvote.mouse_x*thetamax/dispvote.dimx(),
        x = img.dimx()/2  + rho*std::cos(theta),
        y = img.dimy()/2 + rho*std::sin(theta);
      const int
        x0 = (int)(x+1000*std::sin(theta)),
        y0 = (int)(y-1000*std::cos(theta)),
        x1 = (int)(x-1000*std::sin(theta)),
        y1 = (int)(y+1000*std::cos(theta));
      CImg<unsigned char>(src).
        draw_line(x0,y0,x1,y1,col1,1.0f,0xF0F0F0F0).draw_line(x0,y0,x1,y1,col2,1.0f,0x0F0F0F0F).
        draw_line(x0+1,y0,x1+1,y1,col1,1.0f,0xF0F0F0F0).draw_line(x0+1,y0,x1+1,y1,col2,1.0f,0x0F0F0F0F).
        draw_line(x0,y0+1,x1,y1+1,col1,1.0f,0xF0F0F0F0).draw_line(x0,y0+1,x1,y1+1,col2,1.0f,0x0F0F0F0F).
        display(disp);
     }

     // When clicking on the image.
    if (disp.button && disp.mouse_x>=0) {
       const double
         x0 = (double)disp.mouse_x-disp.dimx()/2,
         y0 = (double)disp.mouse_y-disp.dimy()/2,
         rho0 = std::sqrt(x0*x0+y0*y0),
         theta0 = std::atan2(y0,x0);

       for (double t=0; t<thetamax; t+=0.001) {
         double theta = t, rho = rho0*std::cos(theta0-t);
         if (rho<0) { rho=-rho; theta=cimg::mod(theta+cimg::valuePI,thetamax); }
         vote((int)(theta*vote.dimx()/thetamax),(int)(rho*vote.dimy()/rhomax))+=1;
       }
       CImg<> vote2(vote); cimg_forXY(vote2,x,y) vote2(x,y) = (float)std::log(1+vote(x,y)); vote2.display(dispvote);
    }
    dispvote.resize(dispvote);
    disp.resize(disp);
  }

  std::exit(0);
  return 0;
}
