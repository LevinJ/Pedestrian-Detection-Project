/*
 #
 #  File        : image_registration.cpp
 #                ( C++ source file )
 #
 #  Description : Compute a motion field between two images,
 #                with a multiscale and variational algorithm.
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
 #   same conditions as regards security.
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

// animate_warp() : Create warping animation from two images and a motion field
//----------------
void animate_warp(const CImg<unsigned char>& src, const CImg<unsigned char>& dest, const CImg<>& u,
                  const bool morph, const bool imode, const char *filename,int nb, CImgDisplay& disp) {
  CImg<unsigned char> visu = CImgList<unsigned char>(src,dest,src).get_append('x'), warp(src);
  float t=0;
  for (unsigned int iter=0; !disp || (!disp.is_closed && !disp.is_keyQ); iter++) {
    if (morph) cimg_forXYV(warp,x,y,k) {
      const float dx = u(x,y,0), dy = u(x,y,1),
        I1 = (float)src.linear_atXY(x-t*dx, y-t*dy, k),
        I2 = (float)dest.linear_atXY(x+(1-t)*dx,y+(1-t)*dy,k);
      warp(x,y,k) = (unsigned char)((1-t)*I1 + t*I2);
    } else cimg_forXYV(warp,x,y,k) {
      const float dx = u(x,y,0), dy = u(x,y,1), I1 = (float)src.linear_atXY(x-t*dx, y-t*dy, 0,k);
      warp(x,y,k) = (unsigned char)I1;
    }
    if (disp) visu.draw_image(2*src.dimx(),warp).display(disp.resize().wait(30));
    if (filename && *filename && (imode || (int)iter<nb)) {
      std::fprintf(stderr,"\r  > frame %d           ",iter);
      warp.save(filename,iter);
    }
    t+=1.0f/nb;
    if (t<0) { t=0; nb=-nb; }
    if (t>1) { t=1; nb=-nb; if (filename && *filename) std::exit(0); }
  }
}

// get_warp() : Return the image src warped by the motion field u.
//------------
template<typename T> CImg<T> getwarp(const CImg<T>& src, const CImg<>& u) {
  CImg<T> warp(src);
  cimg_forXY(warp,x,y) warp(x,y) = (T)src.linear_atXY(x - u(x,y,0), y - u(x,y,1));
  return warp;
}

// optmonoflow() : Register images for one scale ( semi-implicite PDE scheme ) between I2->I1
//---------------
CImg<> optmonoflow(const CImg<>& I1, const CImg<>& I2, const CImg<>& u0,
                   const float smooth, const float precision, CImgDisplay& disp) {

  CImg<> u = u0.get_resize(I1.dimx(),I1.dimy(),1,2,3),dI(u);
  CImg_3x3(I,float);
  float dt=2,E=1e20f;

  // compute first derivatives of I2
  cimg_for3x3(I2,x,y,0,0,I) {
    dI(x,y,0) = 0.5f*(Inc-Ipc);
    dI(x,y,1) = 0.5f*(Icn-Icp);
  }

  // Main PDE iteration
  for (unsigned int iter=0; iter<100000; iter++) {
    std::fprintf(stderr,"\r- Iteration %d - E = %g",iter,E); std::fflush(stderr);
    const float Eold = E;
    E = 0;
    cimg_for3XY(u,x,y) {
      const float
        X = x + u(x,y,0),
        Y = y + u(x,y,1),
        deltaI = (float)(I2.linear_atXY(X,Y) - I1(x,y));
      float tmpf = 0;
      cimg_forV(u,k) {
        const float
          ux  = 0.5f*(u(_n1x,y,k)-u(_p1x,y,k)),
          uy  = 0.5f*(u(x,_n1y,k)-u(x,_p1y,k));
        u(x,y,k) = (float)( u(x,y,k) +
                            dt*(
                                -deltaI*dI.linear_atXY(X,Y,k) +
                                smooth* ( u(_n1x,y,k) + u(_p1x,y,k) + u(x,_n1y,k) + u(x,_p1y,k) )
                                )
                            )/(1+4*smooth*dt);
        tmpf += ux*ux + uy*uy;
      }
      E += deltaI*deltaI + smooth * tmpf;
    }
    if (cimg::abs(Eold-E)<precision) break;
    if (Eold<E) dt*=0.5;
    if (disp) disp.resize();
    if (disp && disp.is_closed) std::exit(0);
    if (disp && !(iter%300)) {
      const unsigned char white = 255;
      CImg<unsigned char> tmp = getwarp(I1,u).normalize(0,200);
      tmp.resize(disp.dimx(),disp.dimy()).draw_quiver(u,&white,0.7f,15,-14,true).display(disp);
    }
  }
  return u;
}

// optflow() : multiscale version of the image registration algorithm
//-----------
CImg<> optflow(const CImg<>& xsrc, const CImg<>& xdest,
               const float smooth, const float precision, const unsigned int pnb_scale, CImgDisplay& disp) {
  const CImg<>
    src  = xsrc.get_pointwise_norm(1).resize(xdest.dimx(),xdest.dimy(),1,1,3).normalize(0,1),
    dest = xdest.get_pointwise_norm(1).resize(xdest.dimx(),xdest.dimy(),1,1,3).normalize(0,1);
  CImg<> u = CImg<>(src.dimx(),src.dimy(),1,2).fill(0);

  const unsigned int nb_scale = pnb_scale>0?pnb_scale:(unsigned int)(2*std::log((double)(cimg::max(src.dimx(),src.dimy()))));
  for (int scale=nb_scale-1; scale>=0; scale--) {
    const CImg<> I1 = src.get_resize((int)(src.dimx()/std::pow(1.5,scale)), (int)(src.dimy()/std::pow(1.5,scale)) ,1,1,3);
    const CImg<> I2 = dest.get_resize((int)(src.dimx()/std::pow(1.5,scale)), (int)(src.dimy()/std::pow(1.5,scale)) ,1,1,3);
    std::fprintf(stderr," * Scale %d\n",scale);
    u*=1.5;
    u = optmonoflow(I1,I2,u,smooth,(float)(precision/std::pow(2.25,1+scale)),disp);
    std::fprintf(stderr,"\n");
  }
  return u;
}

/*------------------------

  Main function

  ------------------------*/

int main(int argc,char **argv) {

  // Read command line parameters
  cimg_usage("Compute an optical flow between two 2D images, and create a warped animation");
  const char
    *name_i1   = cimg_option("-i",cimg_imagepath "sh0r.pgm","Input Image 1 (Destination)"),
    *name_i2   = cimg_option("-i2",cimg_imagepath "sh1r.pgm","Input Image 2 (Source)"),
    *name_o    = cimg_option("-o",(const char*)NULL,"Output 2D flow (inrimage)"),
    *name_seq  = cimg_option("-o2",(const char*)NULL,"Output Warping Sequence");
  const float
    smooth    = cimg_option("-s",0.1f,"Flow Smoothness"),
    precision = cimg_option("-p",0.9f,"Convergence precision");
  const unsigned int
    nb        = cimg_option("-n",40,"Number of warped frames"),
    nbscale   = cimg_option("-scale",0,"Number of scales (0=auto)");
  const bool
    normalize = cimg_option("-equalize",true,"Histogram normalization of the images"),
    morph     = cimg_option("-m",true,"Morphing mode"),
    imode     = cimg_option("-c",true,"Complete interpolation (or last frame is missing)"),
    dispflag = !cimg_option("-novisu",false,"Visualization");

  // Init images and display
  std::fprintf(stderr," - Init images.\n");
  const CImg<>
    src(name_i1),
    dest(CImg<>(name_i2).resize(src,3)),
    src_blur  = normalize?src.get_blur(0.5f).equalize(256):src.get_blur(0.5f),
    dest_blur = normalize?dest.get_blur(0.5f).equalize(256):dest.get_blur(0.5f);

  CImgDisplay disp;
  if (dispflag) {
    unsigned int w = src.dimx(), h = src.dimy();
    const unsigned int dmin = cimg::min(w,h), minsiz = 512;
    if (dmin<minsiz) { w=w*minsiz/dmin; h=h*minsiz/dmin; }
    const unsigned int dmax = cimg::max(w,h), maxsiz = 1024;
    if (dmax>maxsiz) { w=w*maxsiz/dmax; h=h*maxsiz/dmax; }
    disp.assign(w,h,"Estimated Motion",0);
  }

  // Run Motion estimation algorithm
  std::fprintf(stderr," - Compute optical flow.\n");
  const CImg<> u = optflow(src_blur,dest_blur,smooth,precision,nbscale,disp);
  if (name_o) u.save(name_o);
  u.print("Computed flow");

  // Do morphing animation
  std::fprintf(stderr," - Create warped animation.\n");
  CImgDisplay disp2;
  if (dispflag) {
    unsigned int w = src.dimx(), h = src.dimy();
    const unsigned int dmin = cimg::min(w,h), minsiz = 100;
    if (dmin<minsiz) { w=w*minsiz/dmin; h=h*minsiz/dmin; }
    const unsigned int dmax = cimg::max(w,h), maxsiz = 1024/3;
    if (dmax>maxsiz) { w=w*maxsiz/dmax; h=h*maxsiz/dmax; }
    disp2.assign(3*w,h,"Source/Destination images and Motion animation",0);
  }

  animate_warp(src.get_normalize(0,255),dest.get_normalize(0,255),u,morph,imode,name_seq,nb,disp2);

  std::exit(0);
  return 0;
}
