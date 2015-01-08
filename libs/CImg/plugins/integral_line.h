/*
 #
 #  File        : integral_line.h
 #                ( C++ header file - CImg plug-in )
 #
 #  Description : This CImg plug-in defines function to track integral lines.
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

#ifndef cimg_plugin_integral_line
#define cimg_plugin_integral_line

#define pcimg_valign2d(i,j) \
    { restype &u = W(i,j,0,0), &v = W(i,j,0,1); \
    if (u*curru + v*currv<0) { u=-u; v=-v; }}
#define pcimg_valign3d(i,j,k) \
    { restype &u = W(i,j,k,0), &v = W(i,j,k,1), &w = W(i,j,k,2); \
    if (u*curru + v*currv + w*currw<0) { u=-u; v=-v; w=-w; }}

CImgList<typename cimg::superset<float,T>::type>
get_integral_line(const float x, const float y, const float z=0,
                  const float L=100, const float dl=0.5f, const unsigned int interpolation=3,
                  const bool orientations_only=false) const {

  typedef typename cimg::superset<float,T>::type restype;
  CImgList<restype> tracking;
  CImg<restype> W = (*this)*dl;

  const unsigned int
    dx1 = width-1,
    dy1 = height-1;
  const float
    L2 = L/2,
    cu = (float)(dl*W((int)x,(int)y,(int)z,0)),
    cv = (float)(dl*W((int)x,(int)y,(int)z,1));
  float
    pu = cu,
    pv = cv,
    X = x,
    Y = y;

  // 3D integral lines
  //-------------------
  switch (W.dimv()) {

  case 3: {
    const unsigned int
      dz1 = depth-1;
    const float
      cw = (float)(dl*W((int)x,(int)y,(int)z,2));
    float
      pw = cw,
      Z = z;

    switch (interpolation) {
    case 0: { // Nearest neighbor
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1 && Z>=0 && Z<=dz1; l+=dl) {
        tracking.insert(CImg<restype>::vector(X,Y,Z));
        const int
          cx = (int)(X+0.5f),
          cy = (int)(Y+0.5f),
          cz = (int)(Z+0.5f);
        float
          u = (float)(dl*W(cx,cy,cz,0)),
          v = (float)(dl*W(cx,cy,cz,1)),
          w = (float)(dl*W(cx,cy,cz,2));
        if (orientations_only && (pu*u + pv*v + pw*w)<0) { u=-u; v=-v; w=-w; }
        X+=(pu=u); Y+=(pv=v); Z+=(pw=w);
      }
      pu = cu;
      pv = cv;
      pw = cw;
      X  = x;
      Y  = y;
      Z  = z;
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1 && Z>=0 && Z<=dz1; l+=dl) {
        const int
          cx = (int)(X+0.5f),
          cy = (int)(Y+0.5f),
          cz = (int)(Z+0.5f);
        float
          u = (float)(dl*W(cx,cy,cz,0)),
          v = (float)(dl*W(cx,cy,cz,1)),
          w = (float)(dl*W(cx,cy,cz,2));
        if (orientations_only && (pu*u + pv*v + pw*w)<0) { u=-u; v=-v; w=-w; }
        X-=(pu=u); Y-=(pv=v); Z-=(pw=w);
        tracking.insert(CImg<restype>::vector(X,Y,Z),0);
      }
    } break;

    case 1: { // Linear
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1 && Z>=0 && Z<=dz1; l+=dl) {
        tracking.insert(CImg<restype>::vector(X,Y,Z));
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1,
          cz = (int)Z, pz = (cz-1<0)?0:cz-1, nz = (cz+1>(int)dz1)?(int)dz1:cz+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,cz,0),
            currv = (float)W(cx,cy,cz,1),
            currw = (float)W(cx,cy,cz,2);
          pcimg_valign3d(px,py,pz); pcimg_valign3d(cx,py,pz); pcimg_valign3d(nx,py,pz);
          pcimg_valign3d(px,cy,pz); pcimg_valign3d(cx,cy,pz); pcimg_valign3d(nx,cy,pz);
          pcimg_valign3d(px,ny,pz); pcimg_valign3d(cx,ny,pz); pcimg_valign3d(nx,ny,pz);
          pcimg_valign3d(px,py,cz); pcimg_valign3d(cx,py,cz); pcimg_valign3d(nx,py,cz);
          pcimg_valign3d(px,cy,cz);                           pcimg_valign3d(nx,cy,cz);
          pcimg_valign3d(px,ny,cz); pcimg_valign3d(cx,ny,cz); pcimg_valign3d(nx,ny,cz);
          pcimg_valign3d(px,py,nz); pcimg_valign3d(cx,py,nz); pcimg_valign3d(nx,py,nz);
          pcimg_valign3d(px,cy,nz); pcimg_valign3d(cx,cy,nz); pcimg_valign3d(nx,cy,nz);
          pcimg_valign3d(px,ny,nz); pcimg_valign3d(cx,ny,nz); pcimg_valign3d(nx,ny,nz);
        }
        float
          u = (float)(dl*W._linear_atXYZ(X,Y,Z,0)),
          v = (float)(dl*W._linear_atXYZ(X,Y,Z,1)),
          w = (float)(dl*W._linear_atXYZ(X,Y,Z,2));
        if (orientations_only && (pu*u + pv*v + pw*w)<0) { u=-u; v=-v; w=-w; }
        X+=(pu=u); Y+=(pv=v); Z+=(pw=w);
      }
      pu = cu;
      pv = cv;
      pw = cw;
      X  = x;
      Y  = y;
      Z  = z;
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1 && Z>=0 && Z<=dz1; l+=dl) {
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1,
          cz = (int)Z, pz = (cz-1<0)?0:cz-1, nz = (cz+1>(int)dz1)?(int)dz1:cz+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,cz,0),
            currv = (float)W(cx,cy,cz,1),
            currw = (float)W(cx,cy,cz,2);
          pcimg_valign3d(px,py,pz); pcimg_valign3d(cx,py,pz); pcimg_valign3d(nx,py,pz);
          pcimg_valign3d(px,cy,pz); pcimg_valign3d(cx,cy,pz); pcimg_valign3d(nx,cy,pz);
          pcimg_valign3d(px,ny,pz); pcimg_valign3d(cx,ny,pz); pcimg_valign3d(nx,ny,pz);
          pcimg_valign3d(px,py,cz); pcimg_valign3d(cx,py,cz); pcimg_valign3d(nx,py,cz);
          pcimg_valign3d(px,cy,cz);                           pcimg_valign3d(nx,cy,cz);
          pcimg_valign3d(px,ny,cz); pcimg_valign3d(cx,ny,cz); pcimg_valign3d(nx,ny,cz);
          pcimg_valign3d(px,py,nz); pcimg_valign3d(cx,py,nz); pcimg_valign3d(nx,py,nz);
          pcimg_valign3d(px,cy,nz); pcimg_valign3d(cx,cy,nz); pcimg_valign3d(nx,cy,nz);
          pcimg_valign3d(px,ny,nz); pcimg_valign3d(cx,ny,nz); pcimg_valign3d(nx,ny,nz);
        }
        float
          u = (float)(dl*W._linear_atXYZ(X,Y,Z,0)),
          v = (float)(dl*W._linear_atXYZ(X,Y,Z,1)),
          w = (float)(dl*W._linear_atXYZ(X,Y,Z,2));
        if (orientations_only && (pu*u+pv*v+pw*w)<0) { u=-u; v=-v; w=-w; }
        X-=(pu=u); Y-=(pv=v); Z-=(pw=w);
        tracking.insert(CImg<restype>::vector(X,Y,Z),0);
      }

    } break;

    case 2: { // 2nd order Runge Kutta
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1 && Z>=0 && Z<=dz1; l+=dl) {
        tracking.insert(CImg<restype>::vector(X,Y,Z));
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1,
          cz = (int)Z, pz = (cz-1<0)?0:cz-1, nz = (cz+1>(int)dz1)?(int)dz1:cz+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,cz,0),
            currv = (float)W(cx,cy,cz,1),
            currw = (float)W(cx,cy,cz,2);
          pcimg_valign3d(px,py,pz); pcimg_valign3d(cx,py,pz); pcimg_valign3d(nx,py,pz);
          pcimg_valign3d(px,cy,pz); pcimg_valign3d(cx,cy,pz); pcimg_valign3d(nx,cy,pz);
          pcimg_valign3d(px,ny,pz); pcimg_valign3d(cx,ny,pz); pcimg_valign3d(nx,ny,pz);
          pcimg_valign3d(px,py,cz); pcimg_valign3d(cx,py,cz); pcimg_valign3d(nx,py,cz);
          pcimg_valign3d(px,cy,cz);                           pcimg_valign3d(nx,cy,cz);
          pcimg_valign3d(px,ny,cz); pcimg_valign3d(cx,ny,cz); pcimg_valign3d(nx,ny,cz);
          pcimg_valign3d(px,py,nz); pcimg_valign3d(cx,py,nz); pcimg_valign3d(nx,py,nz);
          pcimg_valign3d(px,cy,nz); pcimg_valign3d(cx,cy,nz); pcimg_valign3d(nx,cy,nz);
          pcimg_valign3d(px,ny,nz); pcimg_valign3d(cx,ny,nz); pcimg_valign3d(nx,ny,nz);
        }
        const float
          u0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,0)),
          v0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,1)),
          w0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,2));
        float
          u = (float)(dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,0)),
          v = (float)(dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,1)),
          w = (float)(dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,2));
        if (orientations_only && (pu*u+pv*v+pw*w)<0) { u=-u; v=-v; w=-w; }
        X+=(pu=u); Y+=(pv=v); Z+=(pw=w);
      }
      pu = cu;
      pv = cv;
      pw = cw;
      X  = x;
      Y  = y;
      Z  = z;
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1 && Z>=0 && Z<=dz1; l+=dl) {
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1,
          cz = (int)Z, pz = (cz-1<0)?0:cz-1, nz = (cz+1>(int)dz1)?(int)dz1:cz+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,cz,0),
            currv = (float)W(cx,cy,cz,1),
            currw = (float)W(cx,cy,cz,2);
          pcimg_valign3d(px,py,pz); pcimg_valign3d(cx,py,pz); pcimg_valign3d(nx,py,pz);
          pcimg_valign3d(px,cy,pz); pcimg_valign3d(cx,cy,pz); pcimg_valign3d(nx,cy,pz);
          pcimg_valign3d(px,ny,pz); pcimg_valign3d(cx,ny,pz); pcimg_valign3d(nx,ny,pz);
          pcimg_valign3d(px,py,cz); pcimg_valign3d(cx,py,cz); pcimg_valign3d(nx,py,cz);
          pcimg_valign3d(px,cy,cz);                           pcimg_valign3d(nx,cy,cz);
          pcimg_valign3d(px,ny,cz); pcimg_valign3d(cx,ny,cz); pcimg_valign3d(nx,ny,cz);
          pcimg_valign3d(px,py,nz); pcimg_valign3d(cx,py,nz); pcimg_valign3d(nx,py,nz);
          pcimg_valign3d(px,cy,nz); pcimg_valign3d(cx,cy,nz); pcimg_valign3d(nx,cy,nz);
          pcimg_valign3d(px,ny,nz); pcimg_valign3d(cx,ny,nz); pcimg_valign3d(nx,ny,nz);
        }
        const float
          u0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,0)),
          v0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,1)),
          w0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,2));
        float
          u = (float)(dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,0)),
          v = (float)(dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,1)),
          w = (float)(dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,2));
        if (orientations_only && (pu*u+pv*v+pw*w)<0) { u=-u; v=-v; w=-w; }
        X-=(pu=u); Y-=(pv=v); Z-=(pw=w);
        tracking.insert(CImg<restype>::vector(X,Y,Z),0);
      }
    } break;

    case 3: {  // 4nd order Runge Kutta
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1 && Z>=0 && Z<=dz1; l+=dl) {
        tracking.insert(CImg<restype>::vector(X,Y,Z));
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1,
          cz = (int)Z, pz = (cz-1<0)?0:cz-1, nz = (cz+1>(int)dz1)?(int)dz1:cz+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,cz,0),
            currv = (float)W(cx,cy,cz,1),
            currw = (float)W(cx,cy,cz,2);
          pcimg_valign3d(px,py,pz); pcimg_valign3d(cx,py,pz); pcimg_valign3d(nx,py,pz);
          pcimg_valign3d(px,cy,pz); pcimg_valign3d(cx,cy,pz); pcimg_valign3d(nx,cy,pz);
          pcimg_valign3d(px,ny,pz); pcimg_valign3d(cx,ny,pz); pcimg_valign3d(nx,ny,pz);
          pcimg_valign3d(px,py,cz); pcimg_valign3d(cx,py,cz); pcimg_valign3d(nx,py,cz);
          pcimg_valign3d(px,cy,cz);                           pcimg_valign3d(nx,cy,cz);
          pcimg_valign3d(px,ny,cz); pcimg_valign3d(cx,ny,cz); pcimg_valign3d(nx,ny,cz);
          pcimg_valign3d(px,py,nz); pcimg_valign3d(cx,py,nz); pcimg_valign3d(nx,py,nz);
          pcimg_valign3d(px,cy,nz); pcimg_valign3d(cx,cy,nz); pcimg_valign3d(nx,cy,nz);
          pcimg_valign3d(px,ny,nz); pcimg_valign3d(cx,ny,nz); pcimg_valign3d(nx,ny,nz);
        }
        const float
          u0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,0)),
          v0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,1)),
          w0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,2)),
          u1 = (float)(0.5f*dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,0)),
          v1 = (float)(0.5f*dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,1)),
          w1 = (float)(0.5f*dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,2)),
          u2 = (float)(0.5f*dl*W._linear_atXYZ(X+u1,Y+v1,Z+w1,0)),
          v2 = (float)(0.5f*dl*W._linear_atXYZ(X+u1,Y+v1,Z+w1,1)),
          w2 = (float)(0.5f*dl*W._linear_atXYZ(X+u1,Y+v1,Z+w1,2)),
          u3 = (float)(0.5f*dl*W._linear_atXYZ(X+u2,Y+v2,Z+w2,0)),
          v3 = (float)(0.5f*dl*W._linear_atXYZ(X+u2,Y+v2,Z+w2,1)),
          w3 = (float)(0.5f*dl*W._linear_atXYZ(X+u2,Y+v2,Z+w2,2));
        float
          u = u0/6 + u1/3 + u2/3 + u3/6,
          v = v0/6 + v1/3 + v2/3 + v3/6,
          w = w0/6 + w1/3 + w2/3 + w3/6;
        if (orientations_only && (pu*u+pv*v+pw*w)<0) { u=-u; v=-v; w=-w; }
        X+=(pu=u); Y+=(pv=v); Z+=(pw=w);
      }
      pu = cu;
      pv = cv;
      pw = cw;
      X  = x;
      Y  = y;
      Z  = z;
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1 && Z>=0 && Z<=dz1; l+=dl) {
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1,
          cz = (int)Z, pz = (cz-1<0)?0:cz-1, nz = (cz+1>(int)dz1)?(int)dz1:cz+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,cz,0),
            currv = (float)W(cx,cy,cz,1),
            currw = (float)W(cx,cy,cz,2);
          pcimg_valign3d(px,py,pz); pcimg_valign3d(cx,py,pz); pcimg_valign3d(nx,py,pz);
          pcimg_valign3d(px,cy,pz); pcimg_valign3d(cx,cy,pz); pcimg_valign3d(nx,cy,pz);
          pcimg_valign3d(px,ny,pz); pcimg_valign3d(cx,ny,pz); pcimg_valign3d(nx,ny,pz);
          pcimg_valign3d(px,py,cz); pcimg_valign3d(cx,py,cz); pcimg_valign3d(nx,py,cz);
          pcimg_valign3d(px,cy,cz);                           pcimg_valign3d(nx,cy,cz);
          pcimg_valign3d(px,ny,cz); pcimg_valign3d(cx,ny,cz); pcimg_valign3d(nx,ny,cz);
          pcimg_valign3d(px,py,nz); pcimg_valign3d(cx,py,nz); pcimg_valign3d(nx,py,nz);
          pcimg_valign3d(px,cy,nz); pcimg_valign3d(cx,cy,nz); pcimg_valign3d(nx,cy,nz);
          pcimg_valign3d(px,ny,nz); pcimg_valign3d(cx,ny,nz); pcimg_valign3d(nx,ny,nz);
        }
        const float
          u0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,0)),
          v0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,1)),
          w0 = (float)(0.5f*dl*W._linear_atXYZ(X,Y,Z,2)),
          u1 = (float)(0.5f*dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,0)),
          v1 = (float)(0.5f*dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,1)),
          w1 = (float)(0.5f*dl*W._linear_atXYZ(X+u0,Y+v0,Z+w0,2)),
          u2 = (float)(0.5f*dl*W._linear_atXYZ(X+u1,Y+v1,Z+w1,0)),
          v2 = (float)(0.5f*dl*W._linear_atXYZ(X+u1,Y+v1,Z+w1,1)),
          w2 = (float)(0.5f*dl*W._linear_atXYZ(X+u1,Y+v1,Z+w1,2)),
          u3 = (float)(0.5f*dl*W._linear_atXYZ(X+u2,Y+v2,Z+w2,0)),
          v3 = (float)(0.5f*dl*W._linear_atXYZ(X+u2,Y+v2,Z+w2,1)),
          w3 = (float)(0.5f*dl*W._linear_atXYZ(X+u2,Y+v2,Z+w2,2));
        float
          u = u0/6 + u1/3 + u2/3 + u3/6,
          v = v0/6 + v1/3 + v2/3 + v3/6,
          w = w0/6 + w1/3 + w2/3 + w3/6;
        if (orientations_only && (pu*u+pv*v+pw*w)<0) { u=-u; v=-v; w=-w; }
        X-=(pu=u); Y-=(pv=v); Z-=(pw=w);
        tracking.insert(CImg<restype>::vector(X,Y,Z),0);
      }
    } break;
    }

  } break;

  // 2D integral lines
  //-------------------
  case 2: {

    switch (interpolation) {
    case 0: { // Nearest neighbor
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1; l+=dl) {
        tracking.insert(CImg<restype>::vector(X,Y));
        const int
          cx = (int)(X+0.5f),
          cy = (int)(Y+0.5f);
        float
          u = (float)(dl*W(cx,cy,0,0)),
          v = (float)(dl*W(cx,cy,0,1));
        if (orientations_only && (pu*u + pv*v)<0) { u=-u; v=-v; }
        X+=(pu=u); Y+=(pv=v);
      }
      pu = cu;
      pv = cv;
      X  = x;
      Y  = y;
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1; l+=dl) {
        const int
          cx = (int)(X+0.5f),
          cy = (int)(Y+0.5f);
        float
          u = (float)(dl*W(cx,cy,0,0)),
          v = (float)(dl*W(cx,cy,0,1));
        if (orientations_only && (pu*u + pv*v)<0) { u=-u; v=-v; }
        X-=(pu=u); Y-=(pv=v);
        tracking.insert(CImg<restype>::vector(X,Y),0);
      }
    } break;

    case 1: { // Linear
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1; l+=dl) {
        tracking.insert(CImg<restype>::vector(X,Y));
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,0,0),
            currv = (float)W(cx,cy,0,1);
          pcimg_valign2d(px,py); pcimg_valign2d(cx,py); pcimg_valign2d(nx,py);
          pcimg_valign2d(px,cy);                        pcimg_valign2d(nx,cy);
          pcimg_valign2d(px,ny); pcimg_valign2d(cx,ny); pcimg_valign2d(nx,ny);
        }
        float
          u = (float)(dl*W._linear_atXY(X,Y,0,0)),
          v = (float)(dl*W._linear_atXY(X,Y,0,1));
        if (orientations_only && (pu*u + pv*v)<0) { u=-u; v=-v; }
        X+=(pu=u); Y+=(pv=v);
      }
      pu = cu;
      pv = cv;
      X  = x;
      Y  = y;
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1; l+=dl) {
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,0,0),
            currv = (float)W(cx,cy,0,1);
          pcimg_valign2d(px,py); pcimg_valign2d(cx,py); pcimg_valign2d(nx,py);
          pcimg_valign2d(px,cy);                        pcimg_valign2d(nx,cy);
          pcimg_valign2d(px,ny); pcimg_valign2d(cx,ny); pcimg_valign2d(nx,ny);
        }
        float
          u = (float)(dl*W._linear_atXY(X,Y,0,0)),
          v = (float)(dl*W._linear_atXY(X,Y,0,1));
        if (orientations_only && (pu*u+pv*v)<0) { u=-u; v=-v; }
        X-=(pu=u); Y-=(pv=v);
        tracking.insert(CImg<restype>::vector(X,Y),0);
      }
    } break;

    case 2: {  // 2nd order Runge Kutta
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1; l+=dl) {
        tracking.insert(CImg<restype>::vector(X,Y));
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,0,0),
            currv = (float)W(cx,cy,0,1);
          pcimg_valign2d(px,py); pcimg_valign2d(cx,py); pcimg_valign2d(nx,py);
          pcimg_valign2d(px,cy);                        pcimg_valign2d(nx,cy);
          pcimg_valign2d(px,ny); pcimg_valign2d(cx,ny); pcimg_valign2d(nx,ny);
        }
        const float
          u0 = (float)(0.5f*dl*W._linear_atXY(X,Y,0,0)),
          v0 = (float)(0.5f*dl*W._linear_atXY(X,Y,0,1));
        float
          u = (float)(dl*W._linear_atXY(X+u0,Y+v0,0,0)),
          v = (float)(dl*W._linear_atXY(X+u0,Y+v0,0,1));
        if (orientations_only && (pu*u+pv*v)<0) { u=-u; v=-v; }
        X+=(pu=u); Y+=(pv=v);
      }
      pu = cu;
      pv = cv;
      X  = x;
      Y  = y;
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1; l+=dl) {
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,0,0),
            currv = (float)W(cx,cy,0,1);
          pcimg_valign2d(px,py); pcimg_valign2d(cx,py); pcimg_valign2d(nx,py);
          pcimg_valign2d(px,cy);                        pcimg_valign2d(nx,cy);
          pcimg_valign2d(px,ny); pcimg_valign2d(cx,ny); pcimg_valign2d(nx,ny);
        }
        const float
          u0 = (float)(0.5f*dl*W._linear_atXY(X,Y,0,0)),
          v0 = (float)(0.5f*dl*W._linear_atXY(X,Y,0,1));
        float
          u = (float)(dl*W._linear_atXY(X+u0,Y+v0,0,0)),
          v = (float)(dl*W._linear_atXY(X+u0,Y+v0,0,1));
        if (orientations_only && (pu*u+pv*v)<0) { u=-u; v=-v; }
        X-=(pu=u); Y-=(pv=v);
        tracking.insert(CImg<restype>::vector(X,Y),0);
      }
    } break;

    case 3: {  // 4nd order Runge Kutta
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1; l+=dl) {
        tracking.insert(CImg<restype>::vector(X,Y));
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,0,0),
            currv = (float)W(cx,cy,0,1);
          pcimg_valign2d(px,py); pcimg_valign2d(cx,py); pcimg_valign2d(nx,py);
          pcimg_valign2d(px,cy);                        pcimg_valign2d(nx,cy);
          pcimg_valign2d(px,ny); pcimg_valign2d(cx,ny); pcimg_valign2d(nx,ny);
        }
        const float
          u0 = (float)(0.5f*dl*W._linear_atXY(X,Y,0,0)),
          v0 = (float)(0.5f*dl*W._linear_atXY(X,Y,0,1)),
          u1 = (float)(0.5f*dl*W._linear_atXY(X+u0,Y+v0,0,0)),
          v1 = (float)(0.5f*dl*W._linear_atXY(X+u0,Y+v0,0,1)),
          u2 = (float)(0.5f*dl*W._linear_atXY(X+u1,Y+v1,0,0)),
          v2 = (float)(0.5f*dl*W._linear_atXY(X+u1,Y+v1,0,1)),
          u3 = (float)(0.5f*dl*W._linear_atXY(X+u2,Y+v2,0,0)),
          v3 = (float)(0.5f*dl*W._linear_atXY(X+u2,Y+v2,0,1));
        float
          u = u0/6 + u1/3 + u2/3 + u3/6,
          v = v0/6 + v1/3 + v2/3 + v3/6;
        if (orientations_only && (pu*u+pv*v)<0) { u=-u; v=-v; }
        X+=(pu=u); Y+=(pv=v);
      }
      pu = cu;
      pv = cv;
      X  = x;
      Y  = y;
      for (float l=0; l<L2 && X>=0 && X<=dx1 && Y>=0 && Y<=dy1; l+=dl) {
        const int
          cx = (int)X, px = (cx-1<0)?0:cx-1, nx = (cx+1>(int)dx1)?(int)dx1:cx+1,
          cy = (int)Y, py = (cy-1<0)?0:cy-1, ny = (cy+1>(int)dy1)?(int)dy1:cy+1;
        if (orientations_only) {
          const float
            curru = (float)W(cx,cy,0,0),
            currv = (float)W(cx,cy,0,1);
          pcimg_valign2d(px,py); pcimg_valign2d(cx,py); pcimg_valign2d(nx,py);
          pcimg_valign2d(px,cy);                        pcimg_valign2d(nx,cy);
          pcimg_valign2d(px,ny); pcimg_valign2d(cx,ny); pcimg_valign2d(nx,ny);
        }
        const float
          u0 = (float)(0.5f*dl*W._linear_atXY(X,Y,0,0)),
          v0 = (float)(0.5f*dl*W._linear_atXY(X,Y,0,1)),
          u1 = (float)(0.5f*dl*W._linear_atXY(X+u0,Y+v0,0,0)),
          v1 = (float)(0.5f*dl*W._linear_atXY(X+u0,Y+v0,0,1)),
          u2 = (float)(0.5f*dl*W._linear_atXY(X+u1,Y+v1,0,0)),
          v2 = (float)(0.5f*dl*W._linear_atXY(X+u1,Y+v1,0,1)),
          u3 = (float)(0.5f*dl*W._linear_atXY(X+u2,Y+v2,0,0)),
          v3 = (float)(0.5f*dl*W._linear_atXY(X+u2,Y+v2,0,1));
        float
          u = u0/6 + u1/3 + u2/3 + u3/6,
          v = v0/6 + v1/3 + v2/3 + v3/6;
        if (orientations_only && (pu*u+pv*v)<0) { u=-u; v=-v; }
        X-=(pu=u); Y-=(pv=v);
        tracking.insert(CImg<restype>::vector(X,Y),0);
      }
    } break;
    }

  } break;

  default:
    throw CImgInstanceException("CImg<%s>::get_integral_line() : Instance image must have dimv()=2 or 3 (current is %u).",
                                pixel_type(),dim);
    break;
  }

  return tracking;
}

#endif
