/*
 #
 #  File        : scene3d.cpp
 #                ( C++ source file )
 #
 #  Description : A simple program that demonstrates the use of the
 #                3D functions of CImg, in conjonction with the Board library.
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

// Uncomment the line below to use the Board library.
// ( You will need to link your code with the board library object ).
// ( Get the Board Library at : http://libboard.sourceforge.net/ )
//#define cimg_use_board

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

//-------------------------
// Begin the main procedure
//-------------------------
int main() {

  // Define a simple 3D scene, composed of various basic objects (torus, cone, cube, ...)
  //-------------------------------------------------------------------------------------
  std::fprintf(stderr," - Create 3D Scene.\n");
  CImg<float> scene_pts, scene_opacs;
  CImgList<unsigned int> scene_prims;
  CImgList<unsigned char> scene_cols;

  CImgList<unsigned int>
    cube_prims,
    cone_prims,
    torus_prims,
    sphere_prims,
    plane_prims;

  // Define objects vertices and faces.
  const CImg<float>
    cube_pts = CImg<>::cube3d(cube_prims,60).translate_object3d(-50,50,0),
    cone_pts = CImg<>::cone3d(cone_prims,30,40).translate_object3d(50,50,0),
    torus_pts = CImg<>::torus3d(torus_prims,30,10).translate_object3d(-50,-50,0),
    sphere_pts = CImg<>::sphere3d(sphere_prims,30).translate_object3d(50,-50,0),
    plane_pts = CImg<>::plane3d(plane_prims,200,200,20,20,true).translate_object3d(0,0,40);

  // Define objects colors and textures.
  const CImgList<unsigned char>
    cone_cols = CImgList<unsigned char>(cone_prims.size,CImg<unsigned char>::vector(128,63,255)),
    torus_cols = CImgList<unsigned char>(torus_prims.size,CImg<unsigned char>::vector(255,55,163)),
    sphere_cols = CImgList<unsigned char>(sphere_prims.size,CImg<unsigned char>::vector(115,115,63)),
    plane_cols = CImgList<unsigned char>(plane_prims.size,CImg<unsigned char>::vector(60,120,180));

  const CImg<unsigned char> texture = CImg<unsigned char>(cimg_imagepath "milla.bmp").resize(128,128);
  CImgList<unsigned char> cube_cols;
  cimglist_for(cube_prims,p) {
    cube_cols.insert(texture,~0U,true);
    cube_prims[p].append(CImg<unsigned int>::vector(0,0,127,0,127,127,0,127),'y');
  }

  // Define objects opacities.
  const CImg<float>
    cube_opacs(cube_prims.size,1,1,1,1.0f),
    cone_opacs(cone_prims.size,1,1,1,0.8f),
    torus_opacs(torus_prims.size,1,1,1,0.6f),
    sphere_opacs(sphere_prims.size,1,1,1,0.4f),
    plane_opacs(plane_prims.size,1,1,1,0.4f);

  // Append all object in a single scene.
  scene_pts.
    append_object3d(scene_prims,cube_pts,cube_prims).
    append_object3d(scene_prims,cone_pts,cone_prims).
    append_object3d(scene_prims,torus_pts,torus_prims).
    append_object3d(scene_prims,sphere_pts,sphere_prims).
    append_object3d(scene_prims,plane_pts,plane_prims);
  scene_cols<<cube_cols<<cone_cols<<torus_cols<<sphere_cols<<plane_cols;
  scene_opacs = (cube_opacs<<cone_opacs<<torus_opacs<<sphere_opacs<<plane_opacs).get_append('x');

  // Display object3D in a user-interacted window and get final position matrix.
  std::fprintf(stderr," - Display 3D Scene.\n");
  const CImg<unsigned char> visu = CImg<unsigned char>(3,512,512,1).fill(230,230,255).permute_axes("yzvx");
  CImg<float> view_matrix = CImg<>::identity_matrix(4);
  visu.display_object3d("3D Scene",scene_pts,scene_prims,scene_cols,scene_opacs,true,4,4,false,
                        500.0f,0.5f,0.1f,true,view_matrix.ptr());

  // Save object 3D as OFF file.
  std::fprintf(stderr," - Save .OFF 3D object file.\n");
  scene_pts.save_off("output.off",scene_prims,scene_cols);

  // Save 3D view in SVG, EPS and FIG files.
  // (using the Board library : http://www.greyc.ensicaen.fr/~seb/board/ ).
#ifdef cimg_use_board

  // Define a Board instance
  BoardLib::Board B;

  // Set Background color of the board.
  B.clear(230,230,255);

  // Draw object both in 'visu' and in the board.
  (view_matrix.crop(0,0,2,2))*=20;
  (+visu).draw_object3d(B,visu.dimx()/2,visu.dimy()/2,visu.dimz()/2,view_matrix*scene_pts,scene_prims,scene_cols,scene_opacs,3).
  display("Snapshot for Board");

  // Save board into a vector graphics file format.
  std::fprintf(stderr," - Save .SVG, .EPS and .FIG snapshots\n");
  B.save("output.svg");
  B.save("output.eps");
  B.save("output.fig");
#endif

  // Exit.
  std::fprintf(stderr," - Exit.\n");
  return 0;
}
