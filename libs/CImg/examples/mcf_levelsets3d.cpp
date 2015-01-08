/*
 #
 #  File        : mcf_levelsets3d.cpp
 #                ( C++ source file )
 #
 #  Description : Implementation of the Mean Curvature Flow on Surfaces
 #                using the framework of Level Sets 3D.
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

// Apply the Mean curvature flow PDE
//-----------------------------------
template<typename T> CImg<T>& mcf_PDE(CImg<T>& img, const unsigned int nb_iter,
                                      const float dt=0.25f, const float narrow=4.0f) {
  CImg<T> veloc(img.dimx(),img.dimy(),img.dimz(),img.dimv());
  CImg_3x3x3(I,float);
  for (unsigned int iter=0; iter<nb_iter; iter++) {
    cimg_for3x3x3(img,x,y,z,0,I) if (cimg::abs(Iccc)<narrow) {
      const float
        ix = 0.5f*(Incc-Ipcc),
        iy = 0.5f*(Icnc-Icpc),
        iz = 0.5f*(Iccn-Iccp),
        norm = (float)std::sqrt(1e-5f+ix*ix+iy*iy+iz*iz),
        ixx = Incc+Ipcc-2*Iccc,
        ixy = 0.25f*(Ippc+Innc-Inpc-Ipnc),
        ixz = 0.25f*(Ipcp+Incn-Incp-Ipcn),
        iyy = Icnc+Icpc-2*Iccc,
        iyz = 0.25f*(Icpp+Icnn-Icnp-Icpn),
        izz = Iccn+Iccp-2*Iccc,
        a = ix/norm,
        b = iy/norm,
        c = iz/norm,
        inn = a*a*ixx + b*b*iyy + c*c*izz + 2*a*b*ixy + 2*a*c*ixz + 2*b*c*iyz;
      veloc(x,y,z) = ixx+iyy+izz-inn;
    } else veloc(x,y,z) = 0;
    float m, M = veloc.maxmin(m);
    const double xdt = dt/cimg::max(cimg::abs(m),cimg::abs(M));
    img+=xdt*veloc;
  }
  return img;
}

// Main procedure
//----------------
int main(int argc,char **argv) {
  cimg_usage("Mean curvature flow of a surface, using 3D level sets");
  const char *file_i = cimg_option("-i",(char*)0,"Input image");
  const float dt = cimg_option("-dt",0.05f,"PDE Time step");
  const float narrow = cimg_option("-band",5.0f,"Size of the narrow band");
  const bool both = cimg_option("-both",false,"Show both evolving and initial surface");

  // Define the signed distance map of the initial surface
  CImg<> img;
  if (file_i) {
    const float sigma = cimg_option("-sigma",1.2f,"Segmentation regularity");
    const float alpha = cimg_option("-alpha",5.0f,"Region growing tolerance");
    img.load(file_i).channel(0);
    CImg<int> s;
    CImgDisplay disp(img,"Please select a starting point");
    while (!s || s[0]<0) s = img.get_select(0,disp);
    CImg<> region;
    float tmp[1] = { 0 };
    img.draw_fill(s[0],s[1],s[2],tmp,1,region,alpha);
    ((img = region.normalize(-1,1))*=-1).blur(sigma);

  }
  else { // Create synthetic implicit function
    img.assign(60,60,60);
    const float exte[1]={1}, inte[1]={-1};
    img.fill(*exte).draw_rectangle(15,15,15,45,45,45,inte).draw_rectangle(25,25,0,35,35,img.dimz()-1,exte).
      draw_rectangle(0,25,25,img.dimx()-1,35,35,exte).draw_rectangle(25,0,25,35,img.dimy()-1,35,exte);
  }
  img.distance_hamilton(10,0,0.1f);

  // Compute corresponding surface triangularization by the marching cube algorithm (isovalue 0)
  CImg<> points0;
  CImgList<unsigned int> faces0;
  if (both) points0 = img.get_isovalue3d(faces0,0);
  const CImgList<unsigned char> colors0(faces0.size,CImg<unsigned char>::vector(100,200,255));
  const CImgList<> opacities0(faces0.size,1,1,1,1,0.2f);

  // Perform MCF evolution
  CImgDisplay disp(256,256,"",1), disp3d(512,512,"",0);
  float alpha = 0, beta = 0;
  for (unsigned int iter=0; !disp.is_closed && !disp3d.is_closed && !disp.is_keyESC && !disp3d.is_keyESC &&
         !disp.is_keyQ && !disp3d.is_keyQ; iter++) {
    disp.set_title("3D implicit Function (iter. %u)",iter);
    disp3d.set_title("Mean curvature flow 3D - Isosurface (iter. %u)",iter);

    // Apply PDE on the distance function
    mcf_PDE(img,1,dt,narrow);                       // Do one iteration of mean curvature flow
    if (!(iter%10)) img.distance_hamilton(1,narrow,0.5f); // Every 10 steps, do one iteration of distance function re-initialization

    // Compute surface triangularization by the marching cube algorithm (isovalue 0)
    CImgList<unsigned int> faces;
    CImg<> points = img.get_isovalue3d(faces,0);
    CImgList<unsigned char> colors(faces.size,CImg<unsigned char>::vector(200,128,100));
    CImgList<> opacities(faces.size,CImg<>::vector(1.0f));
    const float fact = 3*cimg::max(disp3d.dimx(),disp3d.dimy())/(4.0f*cimg::max(img.dimx(),img.dimy()));

    // Append initial object if necessary.
    if (both) {
      points.append_object3d(faces,points0,faces0);
      colors.insert(colors0);
      opacities.insert(opacities0);
    }

    // center and rescale the objects
    cimg_forX(points,l) {
      points(l,0)=(points(l,0)-img.dimx()/2)*fact;
      points(l,1)=(points(l,1)-img.dimy()/2)*fact;
      points(l,2)=(points(l,2)-img.dimz()/2)*fact;
    }

    // Display 3D object on the display window.
    CImg<unsigned char> visu(disp3d.dimx(),disp3d.dimy(),1,3,0);
    const CImg<> rot = CImg<>::rotation_matrix(1,0,0,(beta+=0.01f))*CImg<>::rotation_matrix(0,1,1,(alpha+=0.05f));
    if (points.size()) {
      visu.draw_object3d(visu.dimx()/2.0f,visu.dimy()/2.0f,0.0f,
                         rot*points,faces,colors,opacities,3,
                         false,500.0,0.0f,0.0f,-8000.0f).display(disp3d);
    } else visu.fill(0).display(disp3d);
    img.display(disp.wait(20));

    if ((disp3d.button || disp3d.key) && points.size()) {
      unsigned char white[3]={ 255,255,255 };
      visu.fill(0).draw_text(10,10,"Time stopped, press any key to start again",white).
        display_object3d(disp3d,points,faces,colors,opacities,true,4,3,false,500,0.4f,0.3f);
      disp3d.key = 0;
    }
    if (disp.is_resized)   disp.resize(false);
    if (disp3d.is_resized) disp3d.resize(false);
  }

  // Exit
  return 0;
}
