/*
 #
 #  File        : gmic.cpp
 #                ( C++ source file )
 #
 #  Description : GREYC's Magic Image Converter (library and executable)
 #                ( http://gmic.sourceforge.net )
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

// Add specific G'MIC methods to the CImg<T> class.
//-------------------------------------------------
#ifdef cimg_plugin

template<typename t>
CImg<T>& replace(CImg<t>& img) {
  return img.transfer_to(*this);
}

template<typename t>
CImg<T> get_replace(const CImg<t>& img) const {
  return +img;
}

CImg<T> get_gmic_set(const double value, const int x, const int y, const int z, const int v) const {
  return (+*this).gmic_set(value,x,y,z,v);
}

CImg<T>& gmic_set(const double value, const int x, const int y, const int z, const int v) {
  (*this).atXYZV(x,y,z,v,0) = (T)value;
  return *this;
}

CImg<T> get_draw_point(const int x, const int y, const int z, const CImg<T>& col, const float opacity) const {
  return (+*this).draw_point(x,y,z,col,opacity);
}

CImg<T> get_draw_line(const int x0, const int y0, const int x1, const int y1, const CImg<T>& col, const float opacity) const {
  return (+*this).draw_line(x0,y0,x1,y1,col,opacity);
}

template<typename t>
CImg<T> get_draw_polygon(const CImg<t>& pts, const CImg<T>& col, const float opacity) const {
  return (+*this).draw_polygon(pts,col,opacity);
}

CImg<T> get_draw_ellipse(const int x, const int y, const float r0, const float r1,
                         const float angle, const CImg<T>& col, const float opacity) const {
  return (+*this).draw_ellipse(x,y,r0,r1,angle,col,opacity);
}

CImg<T> get_draw_text(const int x, const int y, const char *const text, const T *const col,
                      const int bg, const float opacity,const int siz) const {
  return (+*this).draw_text(x,y,text,col,bg,opacity,siz);
}

CImg<T> get_draw_image(const int x, const int y, const int z,
                       const CImg<T>& sprite, const CImg<T>& mask, const float opacity) const {
  return (+*this).draw_image(x,y,z,sprite,mask,opacity);
}

CImg<T> get_draw_image(const int x, const int y, const int z,
                       const CImg<T>& sprite, const float opacity) const {
  return (+*this).draw_image(x,y,z,sprite,opacity);
}

CImg<T> get_draw_plasma(const float alpha, const float beta, const float opacity) const {
  return (+*this).draw_plasma(alpha,beta,opacity);
}

CImg<T> get_draw_mandelbrot(const CImg<T>& color_palette, const float opacity,
                            const double z0r, const double z0i, const double z1r, const double z1i,
                            const unsigned int itermax, const bool normalized_iteration,
                            const bool julia_set, const double paramr, const double parami) const {
  return (+*this).draw_mandelbrot(color_palette,opacity,z0r,z0i,z1r,z1i,itermax,
                                  normalized_iteration,julia_set,paramr,parami);
}

template<typename t1, typename t2>
CImg<T> get_draw_quiver(const CImg<t1>& flow,
                        const CImg<t2>& color, const float opacity=1,
                        const unsigned int sampling=25, const float factor=-20,
                        const bool arrows=true, const unsigned int pattern=~0U) {
  return (+*this).draw_quiver(flow,color,opacity,sampling,factor,arrows,pattern);
}

CImg<T> get_draw_fill(const int x, const int y, const int z,
                      const CImg<T>& col, const float opacity, const float tolerance) const {
  return (+*this).draw_fill(x,y,z,col,opacity,tolerance);
}

static bool is_almost(const T x, const T c) {
  return x>c && x<c+1;
}

bool is_CImg3d() const {
  const bool is_header = (width==1 && height>=8 && depth==1 && dim==1 &&
                          is_almost((*this)[0],(T)'C') && is_almost((*this)[1],(T)'I') &&
                          is_almost((*this)[2],(T)'m') && is_almost((*this)[3],(T)'g') &&
                          is_almost((*this)[4],(T)'3') && is_almost((*this)[5],(T)'d'));
  if (!is_header) return false;
  const int nbv = (int)(*this)[6], nbp = (int)(*this)[7];
  if (nbv<=0 || nbp<=0) return false;
  const T *ptrs = ptr() + 8 + 3*nbv, *const ptre = end();
  if (ptrs>=ptre) return false;
  for (int i = 0; i<nbp && ptrs<ptre; ++i) {
    const int N = (int)*(ptrs++);
    if (N<=0 || N>=8) return false;
    ptrs+=N;
  }
  ptrs+=4*nbp;
  if (ptrs>ptre) return false;
  return true;
}

template<typename tp, typename tf, typename tc, typename to>
CImg<T> get_draw_object3d(const float x0, const float y0, const float z0,
                          const CImg<tp>& points, const CImgList<tf>& primitives,
                          const CImgList<tc>& colors, const CImg<to>& opacities,
                          const unsigned int render_type, const bool double_sided,
                          const float focale, const float lightx, const float lighty,
                          const float lightz, const float specular_light, const float specular_shine,
                          CImg<floatT>& zbuffer=CImg<floatT>::empty()) const {
  return (+*this).draw_object3d(x0,y0,z0,points,primitives,colors,opacities,render_type,double_sided,focale,
                                lightx,lighty,lightz,specular_light,specular_shine,zbuffer);
}

template<typename tp, typename tc, typename to>
CImg<T>& object3dtoCImg3d(CImgList<tp>& primitives, CImgList<tc>& colors, CImg<to>& opacities) {
  if (is_empty() || !primitives) { primitives.assign(); colors.assign(); opacities.assign(); return *this; }
  const unsigned int primitives_size = primitives.size;
  CImgList<floatT> res;
  res.insert(CImg<floatT>("CImg3d",1,6,1,1,false)+=0.5f);
  res.insert(CImg<floatT>::vector((float)width,(float)primitives.size));
  resize(-100,3,1,1,0).transpose().unroll('y').transfer_to(res);
  cimglist_for(primitives,p) {
    res.insert(CImg<floatT>::vector((float)primitives[p].size())).insert(primitives[p]).last().unroll('y');
    primitives[p].assign();
  }
  primitives.assign();
  const unsigned int defined_colors = colors.size;
  cimglist_for(colors,c) { res.insert(colors[c]).last().resize(1,3,1,1,-1); colors[c].assign(); }
  colors.assign();
  if (defined_colors<primitives_size) CImg<floatT>(1,3*(primitives_size-defined_colors),1,1,200).transfer_to(res);
  const unsigned int defined_opacities = opacities.size();
  res.insert(opacities).last().unroll('y');
  opacities.assign();
  if (defined_opacities<primitives.size) CImg<floatT>(1,primitives_size-defined_opacities,1,1,1).transfer_to(res);
  return res.get_append('y').transfer_to(*this);
}

template<typename tp, typename tc, typename to>
CImg<T>& CImg3dtoobject3d(CImgList<tp>& primitives, CImgList<tc>& colors, CImg<to>& opacities) {
  const T *ptrs = ptr() + 6;
  const unsigned int
    nbv = (unsigned int)*(ptrs++),
    nbp = (unsigned int)*(ptrs++);
  CImg<T> points(nbv,3);
  primitives.assign(nbp);
  colors.assign(nbp,1,3,1,1);
  opacities.assign(nbp);
  cimg_forX(points,x) { points(x,0) = (T)*(ptrs++); points(x,1) = (T)*(ptrs++); points(x,2) = (T)*(ptrs++); }
  cimglist_for(primitives,p) {
    const unsigned int N = (unsigned int)*(ptrs++);
    primitives[p].assign(ptrs,1,N,1,1,false);
    ptrs+=N;
  }
  cimglist_for(colors,c) { colors(c,0) = (tc)*(ptrs++); colors(c,1) = (tc)*(ptrs++); colors(c,2) = (tc)*(ptrs++); }
  opacities.assign(ptrs,1,nbp,1,1,false);
  return assign(points);
}

CImg<T> get_appendCImg3d(const CImg<T>& img) const {
  CImg<T> res(1,img.size() + size() - 8);
  const T *ptrs = ptr() + 6, *ptrs0 = img.ptr() + 6;
  T *ptrd = res.ptr();
  *(ptrd++) = (T)('C' + 0.5f); *(ptrd++) = (T)('I' + 0.5f);
  *(ptrd++) = (T)('m' + 0.5f); *(ptrd++) = (T)('g' + 0.5f);
  *(ptrd++) = (T)('3' + 0.5f); *(ptrd++) = (T)('d' + 0.5f);
  const unsigned int
    nbv = (unsigned int)*(ptrs++),
    nbv0 = (unsigned int)*(ptrs0++),
    nbp = (unsigned int)*(ptrs++),
    nbp0 = (unsigned int)*(ptrs0++);
  *(ptrd++) = (T)(nbv + nbv0);
  *(ptrd++) = (T)(nbp + nbp0);
  std::memcpy(ptrd,ptrs,sizeof(T)*nbv*3);
  ptrd+=3*nbv; ptrs+=3*nbv;
  std::memcpy(ptrd,ptrs0,sizeof(T)*nbv0*3);
  ptrd+=3*nbv0; ptrs0+=3*nbv0;
  for (unsigned int i = 0; i<nbp; ++i) {
    const unsigned int N = (unsigned int)*(ptrs++);
    *(ptrd++) = (T)N;
    std::memcpy(ptrd,ptrs,sizeof(T)*N);
    ptrd+=N; ptrs+=N;
  }
  for (unsigned int i = 0; i<nbp0; ++i) {
    const unsigned int N = (unsigned int)*(ptrs0++);
    *(ptrd++) = (T)N;
    for (unsigned int j = 0; j<N; ++j) *(ptrd++) = (T)(*(ptrs0++) + nbv);
  }
  std::memcpy(ptrd,ptrs,sizeof(T)*nbp*3);
  ptrd+=3*nbp; ptrs+=3*nbp;
  std::memcpy(ptrd,ptrs0,sizeof(T)*nbp0*3);
  ptrd+=3*nbp0; ptrs0+=3*nbp0;
  std::memcpy(ptrd,ptrs,sizeof(T)*nbp);
  ptrd+=nbp;
  std::memcpy(ptrd,ptrs0,sizeof(T)*nbp0);
  return res;
}

CImg<T>& appendCImg3d(const CImg<T>& img) {
  return get_appendCImg3d(img).transfer_to(*this);
}

CImg<T>& centerCImg3d() {
  const unsigned int nbv = (unsigned int)(*this)[6];
  const T *ptrs = ptr() + 8;
  float xm = cimg::type<float>::max(), ym = xm, zm = xm, xM = cimg::type<float>::min(), yM = xM, zM = xM;
  for (unsigned int i = 0; i<nbv; ++i) {
    const float x = (float)*(ptrs++), y = (float)*(ptrs++), z = (float)*(ptrs++);
    if (x<xm) xm = x; if (x>xM) xM = x;
    if (y<ym) ym = y; if (y>yM) yM = y;
    if (z<zm) zm = z; if (z>zM) zM = z;
  }
  const float xc = (xm + xM)/2, yc = (ym + yM)/2, zc = (zm + zM)/2;
  T *ptrd = ptr() + 8;
  for (unsigned int i = 0; i<nbv; ++i) { *(ptrd++)-=(T)xc; *(ptrd++)-=(T)yc; *(ptrd++)-=(T)zc; }
  return *this;
}

CImg<T> get_centerCImg3d() const {
  return (+*this).centerCImg3d();
}

CImg<T>& normalizeCImg3d() {
  const unsigned int nbv = (unsigned int)(*this)[6];
  const T *ptrs = ptr() + 8;
  float xm = cimg::type<float>::max(), ym = xm, zm = xm, xM = cimg::type<float>::min(), yM = xM, zM = xM;
  for (unsigned int i = 0; i<nbv; ++i) {
    const float x = (float)*(ptrs++), y = (float)*(ptrs++), z = (float)*(ptrs++);
    if (x<xm) xm = x; if (x>xM) xM = x;
    if (y<ym) ym = y; if (y>yM) yM = y;
    if (z<zm) zm = z; if (z>zM) zM = z;
  }
  const float delta = cimg::max(xM-xm,yM-ym,zM-zm);
  if (delta>0) {
    T *ptrd = ptr() + 8;
    for (unsigned int i = 0; i<3*nbv; ++i) *(ptrd++)/=(T)delta;
  }
  return *this;
}

CImg<T> get_normalizeCImg3d() const {
  return (+*this).normalizeCImg3d();
}

template<typename t>
CImg<T>& rotateCImg3d(const CImg<t>& rot) {
  const unsigned int nbv = (unsigned int)(*this)[6];
  const T *ptrs = ptr() + 8;
  const float
    a = (float)rot(0,0), b = (float)rot(1,0), c = (float)rot(2,0),
    d = (float)rot(0,1), e = (float)rot(1,1), f = (float)rot(2,1),
    g = (float)rot(0,2), h = (float)rot(1,2), i = (float)rot(2,2);
  T *ptrd = ptr() + 8;
  for (unsigned int j = 0; j<nbv; ++j) {
    const float x = (float)*(ptrs++), y = (float)*(ptrs++), z = (float)*(ptrs++);
    *(ptrd++) = (T)(a*x + b*y + c*z);
    *(ptrd++) = (T)(d*x + e*y + f*z);
    *(ptrd++) = (T)(g*x + h*y + i*z);
  }
  return *this;
}

template<typename t>
CImg<T> get_rotateCImg3d(const CImg<t>& rot) const {
  return (+*this).rotateCImg3d(rot);
}

CImg<T>& translateCImg3d(const float tx, const float ty, const float tz) {
  const unsigned int nbv = (unsigned int)(*this)[6];
  T *ptrd = ptr() + 8;
  for (unsigned int j = 0; j<nbv; ++j) { *(ptrd++) += (T)tx; *(ptrd++) += (T)ty; *(ptrd++) += (T)tz; }
  return *this;
}

CImg<T> get_translateCImg3d(const float tx, const float ty, const float tz) const {
  return (+*this).translateCImg3d(tx,ty,tz);
}

CImg<T>& coloropacityCImg3d(const float R, const float G, const float B,
                            const float opacity, const bool set_RGB, const bool set_opacity) {
  T *ptrd = ptr() + 6;
  const unsigned int
    nbv = (unsigned int)*(ptrd++),
    nbp = (unsigned int)*(ptrd++);
  ptrd+=3*nbv;
  for (unsigned int i = 0; i<nbp; ++i) { const unsigned int N = (unsigned int)*(ptrd++); ptrd+=N; }
  if (set_RGB) for (unsigned int c = 0; c<nbp; ++c) { *(ptrd++) = (T)R; *(ptrd++) = (T)G; *(ptrd++) = (T)B; } else ptrd+=3*nbp;
  if (set_opacity) for (unsigned int o = 0; o<nbp; ++o) *(ptrd++) = (T)opacity;
  return *this;
}

CImg<T> get_coloropacityCImg3d(const float R, const float G, const float B,
                               const float opacity, const bool set_RGB, const bool set_opacity) const {
  return (+*this).coloropacityCImg3d(R,G,B,opacity,set_RGB,set_opacity);
}

template<typename t>
CImg<T>& inpaint(CImg<t>& mask) {
  if (!is_sameXYZ(mask))
    throw CImgArgumentException("CImg<%s>::inpaint() : Invalid mask (%u,%u,%u,%u,%p) for instance image (%u,%u,%u,%u,%p).",
                                pixel_type(),mask.width,mask.height,mask.depth,mask.dim,mask.data,width,height,depth,dim,data);
  CImg<t> nmask(mask);
  CImg_3x3(M,t); Mpp = Mnp = Mpn = Mnn = 0;
  CImg_3x3(I,T); Ipp = Inp = Icc = Ipn = Inn = 0;
  bool is_pixel = false;
  do {
    is_pixel = false;
    cimg_forZ(mask,z) cimg_for3x3(mask,x,y,z,0,M) if (Mcc && (!Mpc || !Mnc || !Mcp || !Mcn)) {
      is_pixel = true;
      const float wcp = Mcp?0.0f:1.0f, wpc = Mpc?0.0f:1.0f, wnc = Mnc?0.0f:1.0f, wcn = Mcn?0.0f:1.0f, sumw = wcp + wpc + wnc + wcn;
      cimg_forV(*this,k) {
        cimg_get3x3(*this,x,y,z,k,I);
        (*this)(x,y,z,k) = (T)((wcp*Icp + wpc*Ipc + wnc*Inc + wcn*Icn)/sumw);
      }
      nmask(x,y,z) = 0;
    }
    mask = nmask;
  } while (is_pixel);
  return *this;
}

template<typename t>
CImg<T> get_inpaint(CImg<t>& mask) const {
  return (+*this).inpaint(mask);
}

#else  // eq. to #ifndef cimg_plugin

#define cimg_debug 1
#ifndef cimg_gmic_cpp
#define cimg_gmic_cpp "examples/gmic.cpp"
#define cimg_cimg_h "../CImg.h"
#endif
#define cimg_stdout stdout
#define cimg_plugin cimg_gmic_cpp
#include cimg_cimg_h
#include "gmic.h"
using namespace cimg_library;

// The lines below are necessary when using a non-standard compiler such as visualcpp6.
#ifdef cimg_use_visualcpp6
#define std
#endif
#ifdef min
#undef min
#undef max
#endif

#if !defined(gmic_main) || !defined(gmic_separate_compilation)

// Define some useful macros.
//---------------------------

// Code for validity checking of indices.
#define gmic_inds indices2string(indices,filenames,true)
#define gmic_check_indice(ind,funcname) { \
  const int indo = (int)ind; \
  if (ind<0) ind+=images.size; \
  if (ind<0 || ind>=(int)images.size) { \
    if (images.size) error(images,funcname " : Invalid indice '[%d]' (valid indice range is -%u...%u).", \
                           gmic_inds,indo,images.size,images.size-1); \
    else error(images,funcname " : Invalid indice '[%d]' (image list is empty).",gmic_inds,indo); \
  } \
}

// Code for having 'get' or 'non-get' versions of G'MIC commands.
#define gmic_apply(instance,function) { \
  if (get_version) { \
    unsigned int posi = 0; \
    if (images.contains(instance,posi)) filenames.insert(filenames[posi]); \
    else filenames.insert(CImg<char>("(gmic)",7,1,1,1,false)); \
    CImg<T> res = instance.get_##function; \
    res.transfer_to(images); \
  } else instance.function; \
}

// Code for simple commands that has no parameters and act on images.
#define gmic_simple_item(option,function,description) \
  if (!cimg::strcmp(option,command_name)) { \
    print(images,description,gmic_inds); cimg_foroff(indices,l) gmic_apply(images[indices[l]],function()); \
    continue; \
}

// Code for the type cast command.
#define gmic_cast(pixel_type,st_type) \
  if (!cimg::strcmp(#pixel_type,argument)) { \
    print(images,"Set pixel type to '%s'.",#pixel_type); ++position; \
    if (!cimg::strcmp(st_type,cimg::type<T>::string())) continue; \
    CImgList<pixel_type> casted_images; \
    while (images) { casted_images.insert(images[0]); images.remove(0); } \
    return parse_##pixel_type(command_line,position,casted_images,filenames,dowhiles,repeatdones,locals,initial_call); \
}

// Code for G'MIC arithmetic commands.
#define gmic_arithmetic_item(option1,option2,\
                             function1,description1,arg1_1,arg1_2,value_type1, \
                             function2,description2_1,description2_2,arg2_1,arg2_2,description3) \
 if (!cimg::strcmp(option1,command_name) || !cimg::strcmp(option2,command_name)) { \
   double value = 0; char inds[4096] = { 0 }, sep = 0, end = 0; \
    if (std::sscanf(argument,"%lf%c",&value,&end)==1) { \
      print(images,description1 ".",arg1_1,arg1_2); \
      cimg_foroff(indices,l) \
       if (get_version) { \
         images.insert(images[indices[l]]); images.last().function1((value_type1)value); \
         filenames.insert(filenames[indices[l]]); \
       } else images[indices[l]].function1((value_type1)value); \
      ++position; \
    } else if (std::sscanf(argument,"[%4095[0-9.eE%+-]%c%c",inds,&sep,&end)==2 && sep==']') { \
      const CImg<unsigned int> ind = indices2cimg(inds,images.size,option1); \
      if (ind.size()!=1) error(images,description2_1 " : Argument '[%s]' should contain exactly one indice.",gmic_inds,inds); \
      print(images,description2_2 ".",arg2_1,arg2_2); \
      const CImg<T> img0 = images[ind[0]]; \
      cimg_foroff(indices,l) \
       if (get_version) { \
         images.insert(images[indices[l]]); images.last().function2(img0); \
         filenames.insert(filenames[indices[l]]); \
       } else images[indices[l]].function2(img0); \
      ++position; \
    } else if (std::sscanf(argument,"'%4095[^']%c%c",inds,&sep,&end)==2 && sep=='\'') { \
      cimg::strclean(inds); \
      cimg_foroff(indices,l) \
        if (get_version) { \
          images.insert(images[indices[l]]); images.last().function1((const char*)inds); \
          filenames.insert(filenames[indices[l]]); \
        } else images[indices[l]].function1((const char*)inds); \
      ++position; \
    } else { \
      print(images,description3 ".",gmic_inds); \
      if (images && indices) { \
        if (get_version) { \
          CImg<T> img0 = images[indices[0]]; \
          for (unsigned int siz = indices.size(), l = 1; l<siz; ++l) img0.function2(images[l]); \
          filenames.insert(filenames[indices[0]]); \
          img0.transfer_to(images); \
        } else for (unsigned int siz = indices.size(), ind0 = indices[0], off = 0, l = 1; l<siz; ++l) { \
          const unsigned int ind = indices[l] - off; \
          images[ind0].function2(images[ind]); \
          images.remove(ind); filenames.remove(ind); \
          ++off; \
        }}} continue; \
}

// Constructors.
//--------------
#if defined(gmic_float) || !defined(gmic_separate_compilation)
#include "gmic_def.h"

gmic_exception::gmic_exception() {
  message[0] = 0;
}

gmic_exception::gmic_exception(const char *format, ...) {
  std::va_list ap;
  va_start(ap,format);
  std::vsprintf(message,format,ap);
  va_end(ap);
}

gmic_exception::gmic_exception(const char *format, std::va_list ap) {
  std::vsprintf(message,format,ap);
}

gmic::gmic(const char *const command, const char *const custom_macros, const bool default_macros) {
  CImgList<float> images;
  gmic(command,images,custom_macros,default_macros);
}

// Get current scope as a string.
//-------------------------------
CImg<char> gmic::scope2string() const {
  CImgList<char> res(scope);
  cimglist_for(res,l) res[l].last() = '/';
  CImg<char>::vector(0).transfer_to(res);
  return res.get_append('x');
}

// Return command line items from a string.
//-----------------------------------------
CImgList<char> gmic::commandline_to_CImgList(const char *const command) {
  if (!command || !*command) return CImgList<char>();
  const unsigned int siz = std::strlen(command) + 1;
  char *const ncommand = new char[siz];
  std::strcpy(ncommand,command);
  cimg::strpare(ncommand);
  bool is_dquote = false;
  for (char *com = ncommand; *com;  ++com) switch (*com) {
  case '"' : is_dquote = !is_dquote; break;
  case ' ' : if (is_dquote) *com = 30; break;
  }
  CImgList<char> command_line = CImg<char>(ncommand,siz,1,1,1,true).get_split(' ',false,false,'x');
  const CImg<char> zero(1,1,1,1,0);
  cimglist_for(command_line,k) {
    CImg<char> &item = command_line[k];
    cimg_foroff(item,l) if (item[l]==30) item[l] = ' ';
    item.append(zero,'x');
  }
  delete[] ncommand;
  return command_line;
}

// Set default values of G'MIC parameters and macros.
//----------------------------------------------------
gmic& gmic::assign(const char *const custom_macros, const bool default_macros) {
  macros.assign();
  commands.assign();
  scope.assign(CImg<char>(".",2,1,1,1,false));
  position = 0;
  verbosity_level = 0;
  is_released = true;
  is_debug = false;
  is_trace.assign();
  is_begin = true;
  is_end = false;
  is_quit = false;
  background3d[0] = 120;
  background3d[1] = 120;
  background3d[2] = 140;
  render3d = 4;
  renderd3d = -1;
  is_oriented3d = false;
  focale3d = 500;
  light3d_x = 0;
  light3d_y = 0;
  light3d_z = -5000;
  specular_light3d = 0.15f;
  specular_shine3d = 0.8f;
  is_fullpath = false;
  if (default_macros) add_macros(data_gmic_def);
  add_macros(custom_macros);
  return *this;
}

// Error procedure.
//-----------------
const gmic& gmic::error(const char *format, ...) const {
  va_list ap;
  va_start(ap,format);
  char message[4096] = { 0 };
  std::vsprintf(message + std::sprintf(message,"** Error in %s ** ",scope2string().ptr()),format,ap);
  va_end(ap);
  if (verbosity_level>=0) {
    std::fprintf(cimg_stdout,"\n<gmic> %s",message);
    std::fprintf(cimg_stdout,"\n<gmic> Abort G'MIC instance.\n");
    std::fflush(cimg_stdout);
  }
  throw gmic_exception(message);
  return *this;
}

// Warning procedure.
//-------------------
const gmic& gmic::warning(const char *format, ...) const {
  va_list ap;
  va_start(ap,format);
  if (verbosity_level>=0) {
    std::fprintf(cimg_stdout,"\n<gmic> ** Warning in %s ** ",scope2string().ptr());
    std::vfprintf(cimg_stdout,format,ap);
    std::fflush(cimg_stdout);
  }
  va_end(ap);
  return *this;
}

// Print debug message.
//---------------------
const gmic& gmic::debug(const char *format, ...) const {
  const char t_normal[] = { 0x1b,'[','0',';','0',';','0','m','\0' };
  const char t_red[] = { 0x1b,'[','4',';','3','1',';','5','9','m','\0' };
  const char t_bold[] = { 0x1b,'[','1','m','\0' };
  if (is_debug) {
    va_list ap;
    va_start(ap,format);
    std::fprintf(cimg_stdout,"\n%s%s<gmic-debug>%s%s ",t_bold,t_red,scope2string().ptr(),t_normal);
    std::vfprintf(cimg_stdout,format,ap);
    std::fflush(cimg_stdout);
    va_end(ap);
  }
  return *this;
}

// Print status message.
//----------------------
const gmic& gmic::print(const char *format, ...) const {
  va_list ap;
  va_start(ap,format);
  if (verbosity_level>=0) {
    std::fprintf(cimg_stdout,"\n<gmic> ");
    std::vfprintf(cimg_stdout,format,ap);
    std::fflush(cimg_stdout);
  }
  va_end(ap);
  return *this;
}

// Add macros from a char* buffer.
//---------------------------------
gmic& gmic::add_macros(const char *const data_macros) {
  if (!data_macros) return *this;
  char mac[4096] = { 0 }, com[256*1024] = { 0 }, line[256*1024] = { 0 }, sep = 0;
  unsigned int pos = 0;
  for (const char *data = data_macros; *data; ) {
    if (*data=='\n') ++data;
    else {
      if (std::sscanf(data,"%262143[^\n]",line)>0) data += std::strlen(line) + 1;
      if (line[0]!='#') { // Useful line (not a comment)
        mac[0] = com[0] = 0;
        if (std::sscanf(line,"%4095[^: ] %c %262143[^\n]",mac,&sep,com)>=2 && sep==':' &&
            std::sscanf(mac,"%4095s",line)==1) { // Macro definition.
          macros.insert(CImg<char>(line,std::strlen(line)+1,1,1,1,false),pos);
          cimg::strpare(com);
          commands.insert(CImg<char>(com,std::strlen(com)+1,1,1,1,false),pos++);
        } else { // Possible continuation of a previous macro definition.
          if (!pos) error("Add macro commands : Invalid G'MIC macros data.");
          CImg<char> &last = commands[pos-1];
          last[last.size()-1] = ' ';
          cimg::strpare(line);
          last.append(CImg<char>(line,std::strlen(line)+1,1,1,1,false),'x');
        }
      }
    }
  }
  return *this;
}

// Add macros from a macro file.
//------------------------------
gmic& gmic::add_macros(std::FILE *const file) {
  if (!file) return *this;
  char mac[4096] = { 0 }, com[256*1024] = { 0 }, line[256*1024] = { 0 }, sep = 0;
  unsigned int pos = 0;
  int err = 0;
  while ((err=std::fscanf(file,"%262143[^\n] ",line)>=0)) {
    if (err) { // Non empty-line
      mac[0] = com[0] = 0;
      if (line[0]!='#') { // Useful line (not a comment).
        if (std::sscanf(line,"%4095[^: ] %c %262143[^\n]",mac,&sep,com)>=2 && sep==':' &&
            std::sscanf(mac,"%4095s",line)==1) { // Macro definition.
          macros.insert(CImg<char>(line,std::strlen(line)+1,1,1,1,false),pos);
          cimg::strpare(com);
          commands.insert(CImg<char>(com,std::strlen(com)+1,1,1,1,false),pos++);
        } else { // Possible continuation of a previous macro definition.
          if (!pos) error("Add macro commands : Invalid G'MIC macros data.");
          CImg<char> &last = commands[pos-1];
          last[last.size()-1] = ' ';
          cimg::strpare(line);
          last.append(CImg<char>(line,std::strlen(line)+1,1,1,1,false),'x');
        }
      }
    }
  }
  return *this;
}

// Return indices of the images from a string.
//--------------------------------------------
CImg<unsigned int> gmic::indices2cimg(const char *const string, const unsigned int indice_max,
                                      const char *const command) const {
  if (!string || !*string) {
    if (indice_max) return CImg<unsigned int>::sequence(indice_max,0,indice_max-1);
    else return CImg<unsigned int>();
  }
  CImgList<unsigned int> inds;
  const char *it = string;
  for (bool stopflag = false; !stopflag; ) {
    char sep = 0, end = 0, item0[4096] = { 0 }, item1[4096] = { 0 };
    float ind0 = 0, ind1 = 0, step = 1;
    if (std::sscanf(it,"%4095[^,]%c",item0,&end)!=2) stopflag = true;
    else it += 1 + std::strlen(item0);
    const int err = std::sscanf(item0,"%4095[^:]%c%f%c",item1,&sep,&step,&end);
    if (err!=1 && err!=3) error("%s : Invalid indice(s) '[%s]'.",command,string);
    if (std::sscanf(item1,"%f%%-%f%c%c",&ind0,&ind1,&sep,&end)==3 && sep=='%') {
      ind0 = (float)cimg::round(ind0*indice_max/100,1);
      ind1 = (float)cimg::round(ind1*indice_max/100,1);
    } else if (std::sscanf(item1,"%f%%-%f%c",&ind0,&ind1,&end)==2)
      ind0 = (float)cimg::round(ind0*indice_max/100,1);
    else if (std::sscanf(item1,"%f-%f%c%c",&ind0,&ind1,&sep,&end)==3 && sep=='%')
      ind1 = (float)cimg::round(ind1*indice_max/100,1);
    else if (std::sscanf(item1,"%f-%f%c",&ind0,&ind1,&end)==2) { }
    else if (std::sscanf(item1,"%f%c%c",&ind0,&sep,&end)==2 && sep=='%')
      ind1 = (ind0 = (float)cimg::round(ind0*indice_max/100,1));
    else if (std::sscanf(item1,"%f%c",&ind0,&end)==1)
      ind1 = ind0;
    else error("%s : Invalid indice(s) '[%s]'.",command,string);
    if (ind0<0) ind0+=indice_max;
    if (ind1<0) ind1+=indice_max;
    if (ind0<0 || ind0>=indice_max || ind1<0 || ind1>=indice_max || step<=0) {
      if (indice_max) error("%s : Invalid indice(s) '[%s]' (valid indice range is -%u...%u).",
                            command,string,indice_max,indice_max-1);
      else error("%s : Invalid indice(s) '[%s]' (list is empty).",
                 command,string);
    }
    if (ind0>ind1) cimg::swap(ind0,ind1);
    const unsigned int
      iind0 = (unsigned int)ind0,
      _ind1 = (unsigned int)ind1,
      iind1 = (unsigned int)(_ind1 - cimg::mod((float)_ind1,step));
    if (iind0==iind1) inds.insert(CImg<unsigned int>::vector(iind0));
    else inds.insert(CImg<unsigned int>::sequence((unsigned int)(1+(iind1-iind0)/step),
                                                  (unsigned int)iind0,
                                                  (unsigned int)iind1).get_split('y'));
  }
  inds = inds.get_append('y').sort().get_split('y');
  cimglist_for(inds,l) if (l!=inds.size-1 && inds(l,0)==inds(l+1,0)) inds.remove(l--);
  if (is_debug) {
    debug("Indices : ");
    inds.get_append('y').print(); // List indices if debug mode is activated.
  }
  return inds.get_append('y').sort();
}

// Return stringified version of indices or filenames.
//----------------------------------------------------
char* gmic::indices2string(const CImg<unsigned int>& indices, const CImgList<char>& filenames, const bool display_indices) const {
  static char res0[4096] = { 0 }, res1[4096] = { 0 };
  const unsigned int siz = indices.size();
  if (display_indices) {
    switch (siz) {
    case 0: std::sprintf(res0," []"); break;
    case 1: std::sprintf(res0," [%u]",indices[0]); break;
    case 2: std::sprintf(res0,"s [%u,%u]",indices[0],indices[1]); break;
    case 3: std::sprintf(res0,"s [%u,%u,%u]",indices[0],indices[1],indices[2]); break;
    case 4: std::sprintf(res0,"s [%u,%u,%u,%u]",indices[0],indices[1],indices[2],indices[3]); break;
    default: std::sprintf(res0,"s [%u,...,%u]",indices[0],indices[siz-1]);
    }
    return res0;
  }
  switch (siz) {
  case 0: std::sprintf(res1," "); break;
  case 1: std::sprintf(res1,"%s",filenames[indices[0]].ptr()); break;
  case 2: std::sprintf(res1,"%s, %s",filenames[indices[0]].ptr(),filenames[indices[1]].ptr()); break;
  case 3: std::sprintf(res1,"%s, %s, %s",filenames[indices[0]].ptr(),filenames[indices[1]].ptr(),
                       filenames[indices[2]].ptr()); break;
  case 4: std::sprintf(res1,"%s, %s, %s, %s",filenames[indices[0]].ptr(),filenames[indices[1]].ptr(),
                       filenames[indices[2]].ptr(), filenames[indices[3]].ptr()); break;
  default: std::sprintf(res1,"%s, ..., %s",filenames[indices[0]].ptr(),filenames[indices[siz-1]].ptr());
  }
  return res1;
}
#endif // #if defined(gmic_float) || !defined(gmic_separate_compilation)

// Error procedure.
//-----------------
template<typename T>
const gmic& gmic::error(const CImgList<T>& list, const char *format, ...) const {
  va_list ap;
  va_start(ap,format);
  char message[4096] = { 0 };
  std::vsprintf(message + std::sprintf(message,"** Error in %s ** ",scope2string().ptr()),format,ap);
  va_end(ap);
  if (verbosity_level>=0) {
    std::fprintf(cimg_stdout,"\n<gmic-#%u> %s",list.size,message);
    std::fprintf(cimg_stdout,"\n<gmic-#%u> Abort G'MIC instance.\n",list.size);
    std::fflush(cimg_stdout);
  }
  throw gmic_exception(message);
  return *this;
}

// Warning procedure.
//-------------------
template<typename T>
const gmic& gmic::warning(const CImgList<T>& list, const char *format, ...) const {
  va_list ap;
  va_start(ap,format);
  if (verbosity_level>=0) {
    std::fprintf(cimg_stdout,"\n<gmic-#%u> ** Warning in %s ** ",list.size,scope2string().ptr());
    std::vfprintf(cimg_stdout,format,ap);
    std::fflush(cimg_stdout);
  }
  va_end(ap);
  return *this;
}

// Print debug message.
//---------------------
template<typename T>
const gmic& gmic::debug(const CImgList<T>& list, const char *format, ...) const {
  const char t_normal[] = { 0x1b,'[','0',';','0',';','0','m','\0' };
  const char t_red[] = { 0x1b,'[','4',';','3','1',';','5','9','m','\0' };
  const char t_bold[] = { 0x1b,'[','1','m','\0' };
  if (is_debug) {
    va_list ap;
    va_start(ap,format);
    std::fprintf(cimg_stdout,"\n%s%s<gmic-debug-#%u>%s%s ",t_bold,t_red,list.size,scope2string().ptr(),t_normal);
    std::vfprintf(cimg_stdout,format,ap);
    std::fflush(cimg_stdout);
    va_end(ap);
  }
  return *this;
}

// Print status message.
//----------------------
template<typename T>
const gmic& gmic::print(const CImgList<T>& list, const char *format, ...) const {
  va_list ap;
  va_start(ap,format);
  if (verbosity_level>=0) {
    std::fprintf(cimg_stdout,"\n<gmic-#%u> ",list.size);
    std::vfprintf(cimg_stdout,format,ap);
    std::fflush(cimg_stdout);
  }
  va_end(ap);
  return *this;
}

// Template constructors.
//-----------------------
template<typename T>
gmic::gmic(const int argc, const char *const *const argv, CImgList<T>& images,
           const char *custom_macros, const bool default_macros) {
  assign(custom_macros,default_macros);
  CImgList<char> command_line;
  for (int pos = 1; pos<argc; ++pos)
    command_line.insert(CImg<char>(argv[pos],std::strlen(argv[pos])+1,1,1,1,false));
  is_released = false;
  CImgList<char> filenames;
  unsigned int position = 0;
  CImgList<unsigned int> dowhiles, repeatdones, locals;
  parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,true);
}

template<typename T>
gmic::gmic(const char *const command, CImgList<T>& images,
           const char *custom_macros, const bool default_macros) {
  assign(custom_macros,default_macros);
  is_released = true;
  const CImgList<char> command_line = commandline_to_CImgList(command);
  unsigned int position = 0;
  CImgList<char> filenames;
  CImgList<unsigned int> dowhiles, repeatdones, locals;
  parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,true);
}

// Display specified image(s).
//-----------------------------
template<typename T>
bool gmic::display_images(const CImgList<T>& images, const CImgList<char>& filenames, const CImg<unsigned int>& indices,
                          const bool verbose) const {
  if (!images || !filenames || !indices) { if (verbose) print(images,"Display image []."); return false; }
#if cimg_display==0
  if (verbose) print(images,"Display image%s skipped (no display available).",gmic_inds);
  return true;
#endif
  CImgList<unsigned int> inds = indices.get_unroll('x').get_split('x');
  CImgList<T> visu;
  unsigned int max_height = 0;
  cimglist_for(inds,l) {
    const CImg<T>& img = images[inds(l,0)];
    if (img.height>max_height && !img.is_CImg3d()) max_height = img.height;
  }
  cimglist_for(inds,l) {
    const unsigned int ind = inds(l,0);
    const CImg<T> &img = images[ind];
    if (img) {
      if (!max_height || img.height<max_height) visu.insert(img,~0U,true);
      else visu.insert(img.get_lines(0,max_height-1));
    } else if (verbose) { warning(images,"Display image : Image [%d] is empty.",ind); inds.remove(l--); }
  }
  const CImg<unsigned int> nindices = inds.get_append('x');
  const char *const fnames = indices2string(nindices,filenames,false);
  if (verbose) print(images,"Display image%s = '%s'.\n\n",gmic_inds,fnames);
  if (visu.size) {
    if (visu.size!=1) visu.display(fnames,verbosity_level>=0,'x','p');
    else {
      const CImg<T> &img = visu[0]; char title[4096] = { 0 };
      std::sprintf(title,"%s (%dx%dx%dx%d)",fnames,img.dimx(),img.dimy(),img.dimz(),img.dimv());
      img.display(title,verbosity_level>=0);
    }
  }
  return true;
}

// Display plots of specified image(s).
//--------------------------------------
template<typename T>
bool gmic::display_plots(const CImgList<T>& images, const CImgList<char>& filenames, const CImg<unsigned int>& indices,
                         const unsigned int plot_type, const unsigned int vertex_type,
                         const double xmin, const double xmax,
                         const double ymin, const double ymax,
                         const bool verbose) const {
  if (!images || !filenames || !indices) { print(images,"Plot image []."); return false; }
#if cimg_display==0
  print(images,"Plot image%s skipped (no display available).",gmic_inds);
  return true;
#endif
  CImgDisplay disp(cimg_fitscreen(640,480,1),0,0);
  cimg_foroff(indices,l) {
    const unsigned int ind = indices[l];
    const CImg<T>& img = images[ind];
    if (img) {
      print(images,"Plot image%s = '%s'.\n",gmic_inds,indices2string(indices,filenames,false));
      if (verbosity_level>=0) { std::fputc('\n',cimg_stdout); img.print(filenames[ind].ptr()); }
      char title[4096] = { 0 };
      std::sprintf(title,"%s (%dx%dx%dx%d)",
                   filenames[ind].ptr(),img.dimx(),img.dimy(),img.dimz(),img.dimv());
      img.display_graph(disp.set_title("%s",title),plot_type,vertex_type,0,xmin,xmax,0,ymin,ymax);
    } else if (verbose) warning(images,"Plot image : Image [%d] is empty.",ind);
  }
  return true;
}

// Display specified 3D object(s).
//--------------------------------
template<typename T>
bool gmic::display_objects3d(const CImgList<T>& images, const CImgList<char>& filenames,const CImg<unsigned int>& indices,
                             const bool verbose) const {
  if (!indices) { print(images,"Display 3D object []."); return false; }
#if cimg_display==0
  print(images,"Display 3D object%s skipped (no display available).",gmic_inds);
  return true;
#endif
  CImg<unsigned char> background;
  bool exist3d = false;
  CImgDisplay disp;
  cimg_foroff(indices,l) {
    const unsigned int ind = indices[l];
    const CImg<T> &img = images[ind];
    if (!img.is_CImg3d()) {
      if (verbose) warning(images,"Display 3D object : Image [%d] is not a 3D object.",ind);
    } else {
      exist3d = true;
      if (!background || !disp) {
        background.assign(cimg_fitscreen(640,480,1),1,3);
        cimg_forV(background,k) background.get_shared_channel(k).fill(background3d[k]);
        disp.assign(background);
      }
      CImgList<unsigned int> primitives3d;
      CImgList<unsigned char> colors3d;
      CImg<float> opacities3d, points3d(img);
      points3d.CImg3dtoobject3d(primitives3d,colors3d,opacities3d);
      print(images,"Display 3D object [%u] = '%s' (%d points, %u primitives).",
            ind,filenames[ind].ptr(),points3d.dimx(),primitives3d.size);
      disp.set_title("%s (%d points, %u primitives)",
                     filenames[ind].ptr(),points3d.dimx(),primitives3d.size);
      background.display_object3d(disp,points3d,primitives3d,colors3d,opacities3d,
                                  true,render3d,renderd3d,!is_oriented3d,focale3d,specular_light3d,specular_shine3d);
      if (disp.is_closed) break;
    }
  }
  return exist3d;
}

// Substitute '@' and '{}' expressions.
//--------------------------------------
template<typename T>
bool gmic::substitute_item(const char *const source, char *const destination,
                           const CImgList<T>& images, const CImgList<unsigned int>& repeatdones) const {
  if (!source || !destination) return false;
  bool substitution_done = false;
  char item[4096] = { 0 };
  CImgList<char> items;

  for (const char *nsource = source; *nsource; )
    if (*nsource!='@' && *nsource!='{') { // If not starting with '@' and '{'
      std::sscanf(nsource,"%4095[^@{]",item);
      const unsigned int l = std::strlen(item);
      items.insert(CImg<char>(item,l,1,1,1,true));
      nsource+=l;
    } else { // '@' or '{}' expression found.
      char argument[4096] = { 0 }, sep = 0, end = 0;
      double rand_m = 0, rand_M = 1, rand_round = 0;
      int ind = 0, rand_N = 1, larg = 0;
      bool no_braces = true;

      // Isolate arguments between '{}'.
      if (*nsource=='{') {
        const char *const ptr_beg = nsource + 1, *ptr_end = ptr_beg; unsigned int p = 0;
        for (p = 1; p>0 && *ptr_end; ++ptr_end) { if (*ptr_end=='{') ++p; if (*ptr_end=='}') --p; }
        if (p) { items.insert(CImg<char>(nsource++,1,1,1,1,true)); continue; }
        larg = ptr_end - ptr_beg - 1;
        if (larg>0) { char s[4096] = { 0 }; std::memcpy(s,ptr_beg,larg); substitute_item(s,argument,images,repeatdones); }
        nsource+=larg+2;
        const CImg<T> empty, &img = images.size?images.last():empty;
        cimg::strclean(argument);
        std::sprintf(item,"%g",img.eval(argument));
        items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        substitution_done = true;
        continue;
      } else if (nsource[1]=='{') {
        const char *const ptr_beg = nsource + 2, *ptr_end = ptr_beg; unsigned int p = 0;
        for (p = 1; p>0 && *ptr_end; ++ptr_end) { if (*ptr_end=='{') ++p; if (*ptr_end=='}') --p; }
        if (p) { items.insert(CImg<char>(nsource++,1,1,1,1,true)); continue; }
        larg = ptr_end - ptr_beg - 1;
        if (larg>0) { char s[4096] = { 0 }; std::memcpy(s,ptr_beg,larg); substitute_item(s,argument,images,repeatdones); }
        no_braces = false;
      }

      // Replace '@#' and '@{#}'
      if (nsource[1]=='#' || (argument[0]=='#' && argument[1]==0)) {
        nsource+=no_braces?2:4;
        std::sprintf(item,"%u",images.size);
        items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        substitution_done = true;

        // Replace '@!' and '@{!}'
      } else if (nsource[1]=='!' || (argument[0]=='!' && argument[1]==0)) {
        nsource+=no_braces?2:4;
        std::sprintf(item,"%u",instant_window?(instant_window.is_closed?0:1):0);
        items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        substitution_done = true;

        // Replace '@/', '@{/}' and '@{/,subset}'.
      } else if (nsource[1]=='/' || (argument[0]=='/' && (argument[1]==0 || (argument[1]==',' && argument[2])))) {
        nsource+=no_braces?2:(3+larg);
        const CImg<unsigned int> sub = indices2cimg(argument+2,scope.size,"Item substitution");
        cimg_foroff(sub,i) items.insert(scope[i]).last().last() = '/';
        substitution_done = true;

        // Replace '@>', '@{>}' and '@{>,subset}'.
      } else if (nsource[1]=='>' || (argument[0]=='>' && (argument[1]==0 || (argument[1]==',' && argument[2])))) {
        nsource+=no_braces?2:larg+3;
        const CImg<unsigned int> sub = indices2cimg(argument+2,repeatdones.size,"Item substitution");
        if (sub) {
          cimg_foroff(sub,i) {
            std::sprintf(item,"%u",repeatdones(sub[i],2));
            items.insert(CImg<char>(item,std::strlen(item)+1,1,1,1,true)).last().last()=',';
          }
          --(items.last().width);
        }
        substitution_done = true;

        // Replace '@<', '@{<}' and '@{<,subset}'.
      } else if (nsource[1]=='<' || (argument[0]=='<' && (argument[1]==0 || (argument[1]==',' && argument[2])))) {
        nsource+=no_braces?2:larg+3;
        const CImg<unsigned int> sub = indices2cimg(argument+2,repeatdones.size,"Item substitution");
        if (sub) {
          cimg_foroff(sub,i) {
            std::sprintf(item,"%u",repeatdones(sub[i],1)-1);
            items.insert(CImg<char>(item,std::strlen(item)+1,1,1,1,true)).last().last()=',';;
          }
          --(items.last().width);
        }
        substitution_done = true;

        // Replace '@?', '@{?}' and '@{?,min,max,N,round}'.
      } else if ((nsource[1]=='?' || (argument[0]=='?' && (argument[1]==0 || (argument[1]==',' && argument[2])))) &&
                 (!argument[1] ||
                  (std::sscanf(argument+2,"%lf%c",&rand_M,&end)==1 && !(rand_m=0)) ||
                  std::sscanf(argument+2,"%lf,%lf%c",&rand_m,&rand_M,&end)==2 ||
                  std::sscanf(argument+2,"%lf,%lf,%d%c",&rand_m,&rand_M,&rand_N,&end)==3 ||
                  std::sscanf(argument+2,"%lf,%lf,%d,%lf%c",&rand_m,&rand_M,&rand_N,&rand_round,&end)==4) &&
                 rand_round>=0) {
        nsource+=no_braces?2:larg+3;
        for (int i = 0; i<rand_N; ++i) {
          const double val = rand_m + (rand_M-rand_m)*cimg::rand();
          std::sprintf(item,"%g",rand_round>0?cimg::round(val,rand_round):val);
          if (i<rand_N-1) items.insert(CImg<char>(item,std::strlen(item)+1,1,1,1,true)).last().last()=',';
          else items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        }
        substitution_done = true;

        // Replace '@ind', '@{ind}' and '@{ind,argument}'.
      } else if (std::sscanf(nsource+1,"%d",&ind)==1 ||
                 std::sscanf(argument,"%d%c",&ind,&end)==1 ||
                 std::sscanf(argument,"%d,%c",&ind,&sep)==2) {
        static char buf[128] = { 0 };
        const unsigned int lind = std::sprintf(buf,"%d",ind);
        nsource+=no_braces?1+lind:larg+3;
        int nind = ind;
        if (nind<0) nind+=images.size;
        if (nind<0 || nind>=(int)images.size) {
          if (images.size) error(images,"Item substitution : Invalid indice '%d' in item '@ind' (valid indice range is -%u...%u).",
                                 ind,images.size,images.size-1);
          else error(images,"Item substitution : Invalid indice '%d' in item '@ind' (image list is empty).",ind);
        }
        const CImg<T>& img = images[nind];

        float x = 0, y = 0, z = 0, v = 0; char sepx = 0, sepy = 0, sepz = 0, sepv = 0;
        char argx[256] = { 0 }, argy[256] = { 0 }, argz[256] = { 0 }, argv[256] = { 0 }; int bcond = 0;
        const char *subset = sep?argument+lind+1:&sep;
        const unsigned int l = std::strlen(subset);
        if (*subset=='w' && l==1) {  // Replace by image width.
          std::sprintf(item,"%u",img.width);
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='h' && l==1) { // Replace by image height.
          std::sprintf(item,"%u",img.height);
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='d' && l==1) { // Replace by image depth.
          std::sprintf(item,"%u",img.depth);
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='c' && l==1) { // Replace by number of image channels.
          std::sprintf(item,"%u",img.dim);
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='#' && l==1) { // Replace by number of values.
          std::sprintf(item,"%lu",img.size());
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='m' && l==1) { // Replace by minimum value.
          std::sprintf(item,"%g",(double)img.min());
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='M' && l==1) { // Replace by maximum value.
          std::sprintf(item,"%g",(double)img.max());
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='a' && l==1) { // Replace by image average.
          std::sprintf(item,"%g",img.mean());
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='s' && l==1) { // Replace by image standard deviation.
          std::sprintf(item,"%g",std::sqrt(img.variance()));
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='v' && l==1) { // Replace by image variance.
          std::sprintf(item,"%g",img.variance());
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='-' && l==1) { // Replace by coordinates of minimum value.
          const CImg<unsigned int> st = img.get_stats();
          std::sprintf(item,"%u,%u,%u,%u",st[4],st[5],st[6],st[7]);
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if (*subset=='+' && l==1) { // Replace by coordinates of maximum value.
          const CImg<unsigned int> st = img.get_stats();
          std::sprintf(item,"%u,%u,%u,%u",st[8],st[9],st[10],st[11]);
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else if ((std::sscanf(subset,"(%255[0-9.eE%+-])%c",argx,&end)==1 || // Replace by value at specified coordinates.
                    std::sscanf(subset,"(%255[0-9.eE%+-],%255[0-9.eE%+-])%c",argx,argy,&end)==2 ||
                    std::sscanf(subset,"(%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-])%c",argx,argy,argz,&end)==3 ||
                    std::sscanf(subset,"(%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-])%c",argx,argy,argz,argv,&end)==4 ||
                    std::sscanf(subset,"(%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-],%d)%c",argx,argy,argz,argv,&bcond,&end)==5) &&
                   (!*argx || std::sscanf(argx,"%f%c",&x,&end)==1 || (std::sscanf(argx,"%f%c%c",&x,&sepx,&end)==2 && sepx=='%')) &&
                   (!*argy || std::sscanf(argy,"%f%c",&y,&end)==1 || (std::sscanf(argy,"%f%c%c",&y,&sepy,&end)==2 && sepy=='%')) &&
                   (!*argz || std::sscanf(argz,"%f%c",&z,&end)==1 || (std::sscanf(argz,"%f%c%c",&z,&sepz,&end)==2 && sepz=='%')) &&
                   (!*argv || std::sscanf(argv,"%f%c",&v,&end)==1 || (std::sscanf(argv,"%f%c%c",&v,&sepv,&end)==2 && sepv=='%'))) {
          const int
            nx = (int)cimg::round(sepx=='%'?x*(img.dimx()-1)/100:x,1),
            ny = (int)cimg::round(sepy=='%'?y*(img.dimy()-1)/100:y,1),
            nz = (int)cimg::round(sepz=='%'?z*(img.dimz()-1)/100:z,1),
            nv = (int)cimg::round(sepv=='%'?v*(img.dimv()-1)/100:v,1);
          std::sprintf(item,"%g",bcond?(double)img.atXYZV(nx,ny,nz,nv):(double)img.atXYZV(nx,ny,nz,nv,0));
          items.insert(CImg<char>(item,std::strlen(item),1,1,1,true));
        } else { // Replace by value subset (default).
          CImg<T> values;
          if (!*subset) values = img.get_shared();
          else {
            const CImg<unsigned int> inds = indices2cimg(subset,img.size(),"Item substitution");
            values.assign(inds.size());
            cimg_foroff(inds,p) values[p] = img[inds(p)];
          }
          CImg<char> s_values = values.value_string();
          --(s_values.width); s_values.transfer_to(items);
        }
        substitution_done = true;

        // Replace any other '@' expression.
      } else items.insert(CImg<char>(nsource++,1,1,1,1,true));
    }
  items.insert(CImg<char>::vector(0));
  const CImg<char> _items = items.get_append('x');
  if (_items.size()>4095) error(images,"Item substitution : Buffer overflow when substituting '%s'.",source);
  std::sprintf(destination,"%s",_items.ptr());
  if (substitution_done) debug(images,"Item '%s' substituted to '%s'.",source,destination);
  return substitution_done;
}

// Main parsing procedure.
//------------------------
template<typename T>
gmic& gmic::parse(const CImgList<char>& command_line, unsigned int& position, CImgList<T> &images, CImgList<char> &filenames,
                  CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones, CImgList<unsigned int>& locals,
                  const bool initial_call) {
  const unsigned int command_line_maxsize = 65535;
  const int no_ind = (int)(~0U>>1);
  cimg::exception_mode() = 0;
  if (images.size<filenames.size) filenames.remove(images.size,~0U);
  else if (images.size>filenames.size) filenames.insert(images.size-filenames.size,CImg<char>("(gmic)",7,1,1,1,false));

  // Begin command line parsing.
  while (position<command_line.size && command_line.size<command_line_maxsize && !is_quit) {
    const char
      *const orig_item = command_line[position].ptr(),
      *const orig_argument = position+1<command_line.size?command_line[position+1].ptr():"";

    // Get a constant reference to the last image.
    const CImg<T> _last_image, &last_image = images.size?images.last():_last_image;

    // Check consistency between variables.
    if (filenames.size!=images.size)
      error("Internal error : Number of images (%u) and filenames (%u) are not consistent.",filenames.size,images.size);
    if (!scope.size)
      error("Internal error : Scope of the current command is undefined.");
    if (scope.size>31)
      error("Internal error : Scope stack overflow (infinite recursion ?).");

    // Substitute '@' and '{}' expressions in 'orig_item' and 'orig_argument' if necessary.
    char sub_item[4096] = { 0 }, sub_argument[4096] = { 0 };
    bool sub_item_done = false, sub_argument_done = false;
    if (*orig_item=='-' || *orig_item=='[' || *orig_item=='(') {
      if (std::strchr(orig_item,'@') || std::strchr(orig_item,'{'))
        sub_item_done = substitute_item(orig_item,sub_item,images,repeatdones);
      if (*orig_item=='-' &&
          (*orig_argument!='-' || orig_argument[1]=='.' || orig_argument[1]=='@' || orig_argument[1]=='{' ||
           (orig_argument[1]>='0' && orig_argument[1]<='9')) &&
          (std::strchr(orig_argument,'@') || std::strchr(orig_argument,'{')))
        sub_argument_done = substitute_item(orig_argument,sub_argument,images,repeatdones);
    }
    const char *item = sub_item_done?sub_item:orig_item, *argument = sub_argument_done?sub_argument:orig_argument;
    char argument_text[64] = { 0 };
    if (std::strlen(argument)>=64) {
      std::memcpy(argument_text,argument,60*sizeof(char));
      argument_text[60] = argument_text[61] = argument_text[62] = '.'; argument_text[63] = 0;
    } else std::strcpy(argument_text,argument);

    // Get current item/command from the command line.
    char command_name[4096] = { 0 }, command_restriction[4096] = { 0 };
    bool get_version = false, is_restriction = true;
    CImg<unsigned int> indices;
    if (item[0]=='-' && item[1] && item[1]!='.') {
      char sep0 = 0, sep1 = 0, end = 0;
      if (item[1]=='-' && item[2] && item[2]!='[' && (item[2]!='3' || item[3]!='d')) { ++item; get_version = true; }
      const int err = std::sscanf(item,"%4095[^[]%c%4095[0-9.eE%,:+-]%c%c",command_name,&sep0,command_restriction,&sep1,&end);
      if (err==1) {
        indices = CImg<unsigned int>::sequence(images.size,0,images.size-1);
        is_restriction = false;
      } else if (err==2 && sep0=='[' && item[std::strlen(command_name)+1]==']')
        indices.assign();
      else if (err==4 && sep1==']')
        indices = indices2cimg(command_restriction,
                               (!strcmp(command_name,"-i") || !strcmp(command_name,"-input"))?images.size+1:images.size,command_name);
      else { std::strcpy(command_name,item); command_restriction[0] = 0; is_restriction = false; }
    }
    ++position;

    // Check for verbosity commands.
    if (*item=='-') {
      if (!cimg::strcmp("-verbose+",item) || !cimg::strcmp("-v+",item)) ++verbosity_level;
      else if (!cimg::strcmp("-verbose-",item) || !cimg::strcmp("-v-",item)) --verbosity_level;
    }

    if (is_begin) { print(images,"Start G'MIC instance."); is_begin = false; }
    debug(images,"Item : '%s', Selection%s, Argument : '%s'.",item,gmic_inds,argument);
    if (is_trace) {
      const CImg<unsigned int> trace_inds = indices2cimg(is_trace.ptr(),images.size,"-trace");
      display_images(images,filenames,trace_inds,false);
    }

    // Begin command interpretation.
    try {
      if (*item=='-') {

        //----------------
        // Global options
        //----------------

        // Verbosity (actually, just continue to next command since verbosity has been already processed above).
        if (!cimg::strcmp("-verbose+",item) || !cimg::strcmp("-v+",item) ||
            !cimg::strcmp("-verbose-",item) || !cimg::strcmp("-v-",item)) continue;

        // Load macro file.
        if (!cimg::strcmp("-macros",item) || !cimg::strcmp("-m",item)) {
          const char *const basename = cimg::basename(argument);
          print(images,"Load macro file '%s'",is_fullpath?argument:basename);
          std::FILE *const file = cimg::fopen(argument,"r");
          const unsigned int siz = macros.size;
          add_macros(file);
          cimg::fclose(file);
          if (verbosity_level>=0) {
            std::fprintf(cimg_stdout," (%u macros added).",macros.size-siz);
            std::fflush(cimg_stdout);
          }
          ++position; continue;
        }

        // Switch debug flag.
        if (!cimg::strcmp("-debug",item)) {
          is_debug = !is_debug;
          print(images,"%s debug mode.",is_debug?"Activate":"Deactivate");
          if (is_debug) ++verbosity_level;
          continue;
        }

        // Switch instructions trace.
        if (!cimg::strcmp("-trace",command_name)) {
          if (is_trace) {
            print(images,"Deactivate trace mode.");
            is_trace.assign();
          } else {
            if (*command_restriction) print(images,"Activate trace mode for indices [%s].",command_restriction);
            else print(images,"Activate trace mode for all indices.");
            is_trace = CImg<char>(command_restriction,std::strlen(command_restriction)+1);
          }
          continue;
        }

        // Switch fullpath mode.
        if (!cimg::strcmp("-fullpath",item)) {
          is_fullpath = !is_fullpath;
          print(images,"%s fullpath mode.",is_fullpath?"Activate":"Deactivate");
          continue;
        }

        //----------------------
        // Arithmetic operators
        //----------------------

        // Common arithmetic operators.
        gmic_arithmetic_item("-add","-+",
                             operator+=,"Add %g to image%s",value,gmic_inds,T,
                             operator+=,"Add to image%s",
                             "Add image [%d] to image%s",ind[0],gmic_inds,
                             "Add image%s together");

        gmic_arithmetic_item("-sub","--",
                             operator-=,"Substract %g to image%s",value,gmic_inds,T,
                             operator-=,"Substract to image%s",
                             "Substract image [%d] to image%s",ind[0],gmic_inds,
                             "Substract image%s together");

        gmic_arithmetic_item("-mul","-*",
                             operator*=,"Multiply image%s by %g",gmic_inds,value,double,
                             mul,"Multiply image%s",
                             "Multiply image%s by image [%d]",gmic_inds,ind[0],
                             "Multiply image%s together");

        gmic_arithmetic_item("-div","-/",
                             operator/=,"Divide image%s by %g",gmic_inds,value,double,
                             div,"Divide image%s",
                             "Divide image%s by image [%d]",gmic_inds,ind[0],
                             "Divide image%s together");

        gmic_arithmetic_item("-pow","-^",
                             pow,"Compute image%s to the power of %g",gmic_inds,value,double,
                             pow,"Compute power of image%s",
                             "Compute image%s to the power of image [%d]",gmic_inds,ind[0],
                             "Compute the power of image%s together");

        gmic_arithmetic_item("-min","-min",
                             min,"Compute pointwise minimum between image%s and %g",gmic_inds,value,T,
                             min,"Compute pointwise minimum with image%s",
                             "Compute pointwise minimum between image%s and image [%d]",gmic_inds,ind[0],
                             "Compute pointwise minimum between image%s together");

        gmic_arithmetic_item("-max","-max",
                             max,"Compute pointwise maximum between image%s and %g",gmic_inds,value,T,
                             max,"Compute pointwise maximum with image%s",
                             "Compute pointwise maximum between image%s and image [%d]",gmic_inds,ind[0],
                             "Compute pointwise maximum between image%s together");

        gmic_arithmetic_item("-mod","-%",
                             operator%=,"Compute pointwise modulo between image%s and %g.",gmic_inds,value,T,
                             operator%=,"Compute pointwise modulo with image%s",
                             "Compute pointwise modulo between image%s and image [%d]",gmic_inds,ind[0],
                             "Compute pointwise modulo between image%s together");

        gmic_arithmetic_item("-and","-and",
                             operator&=,"Compute bitwise AND between image%s and %g.",gmic_inds,value,T,
                             operator&=,"Compute bitwise AND with image%s",
                             "Compute bitwise AND between image%s and image [%d]",gmic_inds,ind[0],
                             "Compute bitwise AND between image%s together");

        gmic_arithmetic_item("-or","-or",
                             operator|=,"Compute bitwise OR between image%s and %g.",gmic_inds,value,T,
                             operator|=,"Compute bitwise OR with image%s",
                             "Compute bitwise OR between image%s and image [%d]",gmic_inds,ind[0],
                             "Compute bitwise OR between image%s together");

        gmic_arithmetic_item("-xor","-xor",
                             operator^=,"Compute bitwise XOR between image%s and %g.",gmic_inds,value,T,
                             operator^=,"Compute bitwise XOR with image%s",
                             "Compute bitwise XOR between image%s and image [%d]",gmic_inds,ind[0],
                             "Compute bitwise XOR between image%s together");

        // Other arithmetic operators.
        gmic_simple_item("-cos",cos,"Compute cosine of image%s.");
        gmic_simple_item("-sin",sin,"Compute sine of image%s.");
        gmic_simple_item("-tan",tan,"Compute tangent of image%s.");
        gmic_simple_item("-acos",acos,"Compute arc-cosine of image%s.");
        gmic_simple_item("-asin",asin,"Compute arc-sine of image%s.");
        gmic_simple_item("-atan",atan,"Compute arc-tangent of image%s.");
        gmic_simple_item("-abs",abs,"Compute absolute value of image%s.");
        gmic_simple_item("-sqr",sqr,"Compute square function of image%s.");
        gmic_simple_item("-sqrt",sqrt,"Compute square root of image%s.");
        gmic_simple_item("-exp",exp,"Compute exponential of image%s.");
        gmic_simple_item("-log",log,"Compute logarithm of image%s.");
        gmic_simple_item("-log10",log10,"Compute logarithm_10 of image%s.");

        // Arc-tangent2.
        if (!cimg::strcmp("-atan2",command_name)) {
          char sep = 0, end = 0; int ind = no_ind;
          if (std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') {
            gmic_check_indice(ind,"Compute oriented arc-tangent of image%s");
            print(images,"Compute oriented arc-tangent of image%s, using denominator [%d].",gmic_inds,ind);
            const CImg<T> img0 = images[ind];
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],atan2(img0));
          } else error("Compute oriented arc-tangent of image%s : Invalid argument '%s' "
                       "(should be '[indice]').",gmic_inds,argument_text);
          ++position; continue;
        }

        //---------------------------------------
        // Pointwise operations on pixel values
        //---------------------------------------

        // Type cast.
        if (!cimg::strcmp("-type",item) || !cimg::strcmp("-t",item)) {
          typedef unsigned char uchar;
          typedef unsigned short ushort;
          typedef unsigned int uint;
#ifndef gmic_minimal
          gmic_cast(bool,"bool");
          gmic_cast(uchar,"unsigned char");
          gmic_cast(char,"char");
          gmic_cast(ushort,"unsigned short");
          gmic_cast(short,"short");
          gmic_cast(uint,"unsigned int");
          gmic_cast(int,"int");
          gmic_cast(double,"double");
#endif
          gmic_cast(float,"float");
          error(images,"Set pixel type : Invalid argument '%s' "
                "(should be 'type={bool,uchar,char,ushort,short,uint,int,float,double}').",argument_text);
        }

        // Set pixel (scalar) value.
        if (!cimg::strcmp("-set",command_name) || !cimg::strcmp("-=",command_name)) {
          double value = 0; float x = 0, y = 0, z = 0, v = 0; char sepx = 0, sepy = 0, sepz = 0, sepv = 0, end = 0;
          char argx[4096] = { 0 }, argy[4096] = { 0 }, argz[4096] = { 0 }, argv[4096] = { 0 };
          if ((std::sscanf(argument,"%lf%c",&value,&end)==1 ||
               std::sscanf(argument,"%lf,%4095[0-9.eE%+-]%c",&value,argx,&end)==2 ||
               std::sscanf(argument,"%lf,%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",&value,argx,argy,&end)==3 ||
               std::sscanf(argument,"%lf,%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",&value,argx,argy,argz,&end)==4 ||
               std::sscanf(argument,"%lf,%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",&value,argx,argy,argz,argv,&end)==5) &&
              (!*argx || (std::sscanf(argx,"%f%c%c",&x,&sepx,&end)==2 && sepx=='%') || std::sscanf(argx,"%f%c",&x,&end)==1) &&
              (!*argy || (std::sscanf(argy,"%f%c%c",&y,&sepy,&end)==2 && sepy=='%') || std::sscanf(argy,"%f%c",&y,&end)==1) &&
              (!*argz || (std::sscanf(argz,"%f%c%c",&z,&sepz,&end)==2 && sepz=='%') || std::sscanf(argz,"%f%c",&z,&end)==1) &&
              (!*argv || (std::sscanf(argv,"%f%c%c",&v,&sepv,&end)==2 && sepv=='%') || std::sscanf(argv,"%f%c",&v,&end)==1)) {
            print(images,"Set scalar value %g at position (%g%s,%g%s,%g%s,%g%s), in image%s",
                  value,x,sepx=='%'?"%":"",y,sepy=='%'?"%":"",z,sepz=='%'?"%":"",v,sepv=='%'?"%":"",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                nx = (int)cimg::round(sepx=='%'?x*(img.dimx()-1)/100:x,1),
                ny = (int)cimg::round(sepy=='%'?y*(img.dimy()-1)/100:y,1),
                nz = (int)cimg::round(sepz=='%'?z*(img.dimz()-1)/100:z,1),
                nv = (int)cimg::round(sepv=='%'?v*(img.dimv()-1)/100:v,1);
              gmic_apply(images[indices[l]],gmic_set(value,nx,ny,nz,nv));
            }
          } else error(images,"Set scalar value in image%s : Invalid argument '%s' "
                       "(should be 'value,_x,_y,_z,_v').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Invert endianness.
        gmic_simple_item("-endian",invert_endianness,"Invert endianness of image%s.");

        // Fill.
        if (!cimg::strcmp("-fill",command_name) || !cimg::strcmp("-f",command_name)) {
          char sep = 0, end = 0; double value = 0; int ind = no_ind;
          if (std::sscanf(argument,"%lf%c",&value,&end)==1) {
            print(images,"Fill image%s with value %g.",gmic_inds,value);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],fill((T)value));
          } else if (std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') {
            gmic_check_indice(ind,"Fill image%s");
            print(images,"Fill image%s with values of image [%d].",gmic_inds,ind);
            const CImg<T> values = images[ind];
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],fill(values));
          } else {
            print(images,"Fill image%s with values '%s'.",gmic_inds,argument_text);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],fill(argument,true));
          }
          ++position; continue;
        }

        // Threshold.
        if (!cimg::strcmp("-threshold",command_name)) {
          char sep = 0, end = 0; unsigned int soft = 0; double value = 0;
          if (std::sscanf(argument,"%lf%c",&value,&end)==1 ||
              (std::sscanf(argument,"%lf%c%c",&value,&sep,&end)==2 && sep=='%') ||
              std::sscanf(argument,"%lf,%u%c",&value,&soft,&end)==2 ||
              std::sscanf(argument,"%lf%c,%u%c",&value,&sep,&soft,&end)==3) {
            print(images,"%s-threshold image%s with value %g%s.",soft?"Soft":"Hard",gmic_inds,value,sep=='%'?"%":"");
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              double vmin = 0, vmax = 0, nvalue = value;
              if (sep=='%') { vmin = img.minmax(vmax); nvalue = vmin + (vmax - vmin)*value/100; }
              gmic_apply(img,threshold((T)nvalue,soft?true:false));
            }
            ++position;
          } else {
#if cimg_display==0
            print(images,"Hard-threshold image%s : interactive mode skipped (no display available).",gmic_inds);
            continue;
#endif
            print(images,"Hard-threshold image%s : interactive mode.",gmic_inds);
            CImgDisplay disp;
            char title[4096] = { 0 };
            cimg_foroff(indices,l) {
              CImg<T>
                &img = images[indices[l]],
                visu = img.depth>1?img.get_projections2d(img.dimx()/2,img.dimy()/2,img.dimz()/2).
                channels(0,cimg::min(3,img.dimv())-1):img.get_channels(0,cimg::min(3,img.dimv()-1));
              if (disp) disp.resize(cimg_fitscreen(visu.dimx(),visu.dimy(),1),false);
              else disp.assign(cimg_fitscreen(visu.dimx(),visu.dimy(),1),0,1);
              double
                vmin = 0, vmax = (double)img.maxmin(vmin),
                distmax = std::sqrt(cimg::sqr(disp.dimx()-1.0) + cimg::sqr(disp.dimy()-1.0)),
                amount = 50;
              bool stopflag = false, obutt = false;
              int omx = -1, omy = -1;
              CImg<T> res;
              for (disp.show().button = disp.key = 0; !stopflag; ) {
                const unsigned int key = disp.key;
                if (!res) {
                  std::sprintf(title,"%s : threshold %.3g%%",filenames[indices[l]].ptr(),amount);
                  disp.display(res=visu.get_threshold((T)(vmin + amount*(vmax-vmin)/100))).
                    set_title("%s",title).wait();
                }
                const int mx = disp.mouse_x, my = disp.mouse_y;
                if (disp.button && mx>=0 && my>=0) {
                  if (omx==mx && omy==my && !obutt) break;
                  omx = mx; omy = my; obutt = true;
                  const double dist = std::sqrt((double)cimg::sqr(mx) + cimg::sqr(my));
                  amount = dist*100/distmax;
                  res.assign();
                } else if (!disp.button) obutt = false;
                if (disp.is_closed || (key && key!=cimg::keyCTRLLEFT)) stopflag = true;
                if (key==cimg::keyD && disp.is_keyCTRLLEFT &&
                    (disp.resize(cimg_fitscreen(3*disp.width/2,3*disp.height/2,1),stopflag=false).key=0)==0)
                  disp.is_resized = true;
                if (key==cimg::keyC && disp.is_keyCTRLLEFT &&
                    (disp.resize(cimg_fitscreen(2*disp.width/3,2*disp.height/3,1),stopflag=false).key=0)==0)
                  disp.is_resized = true;
                if (disp.is_resized) {
                  disp.resize(false).display(res);
                  distmax = std::sqrt(cimg::sqr(disp.dimx()-1.0) + cimg::sqr(disp.dimy()-1.0));
                }
              }
              gmic_apply(img,threshold((T)(vmin + amount*(vmax-vmin)/100)));
            }
          }
          continue;
        }

        // Cut.
        if (!cimg::strcmp("-cut",command_name)) {
          char sep0 = 0, sep1 = 0, end = 0, arg0[4096] = { 0 }, arg1[4096] = { 0 };
          double value0 = 0, value1 = 0; int ind0 = no_ind, ind1 = no_ind;
          if (std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",arg0,arg1,&end)==2 &&
              ((std::sscanf(arg0,"[%d%c%c",&ind0,&sep0,&end)==2 && sep0==']') ||
               (std::sscanf(arg0,"%lf%c%c",&value0,&sep0,&end)==2 && sep0=='%') ||
               std::sscanf(arg0,"%lf%c",&value0,&end)==1) &&
              ((std::sscanf(arg1,"[%d%c%c",&ind1,&sep1,&end)==2 && sep1==']') ||
               (std::sscanf(arg1,"%lf%c%c",&value1,&sep1,&end)==2 && sep1=='%') ||
               std::sscanf(arg1,"%lf%c",&value1,&end)==1)) {
            if (ind0!=no_ind) { gmic_check_indice(ind0,"Cut image%s"); value0 = images[ind0].min(); sep0 = 0; }
            if (ind1!=no_ind) { gmic_check_indice(ind1,"Cut image%s"); value1 = images[ind1].max(); sep1 = 0; }
            print(images,"Cut image%s in value range [%g%s,%g%s].",gmic_inds,value0,sep0=='%'?"%":"",value1,sep1=='%'?"%":"");
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              double vmin = 0, vmax = 0, nvalue0 = value0, nvalue1 = value1;
              if (sep0=='%') { vmin = img.minmax(vmax); nvalue0 = vmin + (vmax - vmin)*value0/100; }
              if (sep1=='%') { vmin = img.minmax(vmax); nvalue1 = vmin + (vmax - vmin)*value1/100; }
              gmic_apply(img,cut((T)nvalue0,(T)nvalue1));
            }
            ++position;
          } else if (std::sscanf(argument,"[%d%c%c",&(ind0=no_ind),&sep0,&end)==2) {
            if (ind0!=no_ind) gmic_check_indice(ind0,"Cut image%s");
            value0 = images[ind0].minmax(value1);
            print(images,"Cut image%s in value range [%g,%g].",gmic_inds,value0,value1);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],cut((T)value0,(T)value1));
            ++position;
          } else {
#if cimg_display==0
            print(images,"Cut image%s : interactive mode skipped (no display available).",gmic_inds);
            continue;
#endif
            print(images,"Cut image%s : interactive mode.",gmic_inds);
            CImgDisplay disp;
            char title[4096] = { 0 };
            cimg_foroff(indices,l) {
              CImg<T>
                &img = images[indices[l]],
                visu = img.depth>1?img.get_projections2d(img.dimx()/2,img.dimy()/2,img.dimz()/2).
                channels(0,cimg::min(3,img.dimv())-1):img.get_channels(0,cimg::min(3,img.dimv()-1));
              if (disp) disp.resize(cimg_fitscreen(visu.dimx(),visu.dimy(),1),false);
              else disp.assign(cimg_fitscreen(visu.dimx(),visu.dimy(),1),0,1);
              double vmin = 0, vmax = (double)img.maxmin(vmin), amount0 = 0, amount1 = 100;
              bool stopflag = false, obutt = false;
              int omx = -1, omy = -1;
              CImg<T> res;
              for (disp.show().button = disp.key = 0; !stopflag; ) {
                const unsigned int key = disp.key;
                if (!res) {
                  std::sprintf(title,"%s : cut [%.3g%%,%.3g%%]",
                               filenames[indices[l]].ptr(),amount0,amount1);
                  disp.display(res = visu.get_cut((T)(vmin + amount0*(vmax-vmin)/100),
                                                  (T)(vmin + amount1*(vmax-vmin)/100))).
                    set_title("%s",title).wait();
                }
                const int mx = disp.mouse_x, my = disp.mouse_y;
                if (disp.button && mx>=0 && my>=0) {
                  if (omx==mx && omy==my && !obutt) break;
                  omx = mx; omy = my; obutt = true;
                  amount0 = mx*100/disp.dimx(); amount1 = my*100/disp.dimy();
                  res.assign();
                } else if (!disp.button) obutt = false;
                if (disp.is_closed || (key && key!=cimg::keyCTRLLEFT)) stopflag = true;
                if (key==cimg::keyD && disp.is_keyCTRLLEFT &&
                    (disp.resize(cimg_fitscreen(3*disp.width/2,3*disp.height/2,1),stopflag=false).key=0)==0)
                  disp.is_resized = true;
                if (key==cimg::keyC && disp.is_keyCTRLLEFT &&
                    (disp.resize(cimg_fitscreen(2*disp.width/3,2*disp.height/3,1),stopflag=false).key=0)==0)
                  disp.is_resized = true;
                if (disp.is_resized) disp.resize(false).display(res);
              }
              gmic_apply(img,cut((T)(vmin + amount0*(vmax-vmin)/100),(T)(vmin + amount1*(vmax-vmin)/100)));
            }
          }
          continue;
        }

        // Normalize.
        if (!cimg::strcmp("-normalize",command_name) || !cimg::strcmp("-n",command_name)) {
          char sep0 = 0, sep1 = 0, end = 0, arg0[4096] = { 0 }, arg1[4096] = { 0 };
          double value0 = 0, value1 = 0; int ind0 = no_ind, ind1 = no_ind;
          if (std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",arg0,arg1,&end)==2 &&
              ((std::sscanf(arg0,"[%d%c%c",&ind0,&sep0,&end)==2 && sep0==']') ||
               (std::sscanf(arg0,"%lf%c%c",&value0,&sep0,&end)==2 && sep0=='%') ||
               std::sscanf(arg0,"%lf%c",&value0,&end)==1) &&
              ((std::sscanf(arg1,"[%d%c%c",&ind1,&sep1,&end)==2 && sep1==']') ||
               (std::sscanf(arg1,"%lf%c%c",&value1,&sep1,&end)==2 && sep1=='%') ||
               std::sscanf(arg1,"%lf%c",&value1,&end)==1)) {
            if (ind0!=no_ind) { gmic_check_indice(ind0,"Normalize image%s"); value0 = images[ind0].min(); sep0 = 0; }
            if (ind1!=no_ind) { gmic_check_indice(ind1,"Normalize image%s"); value1 = images[ind1].max(); sep1 = 0; }
            print(images,"Normalize image%s in value range [%g%s,%g%s].",gmic_inds,value0,sep0=='%'?"%":"",value1,sep1=='%'?"%":"");
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              double vmin = 0, vmax = 0, nvalue0 = value0, nvalue1 = value1;
              if (sep0=='%') { vmin = img.minmax(vmax); nvalue0 = vmin + (vmax - vmin)*value0/100; }
              if (sep1=='%') { vmin = img.minmax(vmax); nvalue1 = vmin + (vmax - vmin)*value1/100; }
              gmic_apply(img,normalize((T)nvalue0,(T)nvalue1));
            }
          } else if (std::sscanf(argument,"[%d%c%c",&(ind0=no_ind),&sep0,&end)==2) {
            if (ind0!=no_ind) gmic_check_indice(ind0,"Normalize image%s");
            value0 = images[ind0].minmax(value1);
            print(images,"Normalize image%s in value range [%g,%g].",gmic_inds,value0,value1);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],normalize((T)value0,(T)value1));
          } else error(images,"Normalize image%s : Invalid argument '%s' "
                       "(should be '{value1[%%],[indice]},{value2[%%],[indice]}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Round.
        if (!cimg::strcmp("-round",command_name)) {
          char end = 0; double value = 0; int rtype = 0;
          if ((std::sscanf(argument,"%lf%c",&value,&end)==1 ||
               std::sscanf(argument,"%lf,%d%c",&value,&rtype,&end)==2) &&
              rtype>=-1 && rtype<1) {
            print(images,"Round image%s with value %g and %s rounding.",
                  gmic_inds,value,rtype<0?"backward":rtype>0?"forward":"nearest");
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],round((float)value,rtype));
          } else error(images,"Round image%s : Invalid argument '%s' "
                       "(should be 'rounding_value,_rounding_type={-1,0,1}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Equalize.
        if (!cimg::strcmp("-equalize",command_name)) {
          float nb = 256; char sep = 0, end = 0;
          if ((std::sscanf(argument,"%f%c",&nb,&end)==1 ||
               (std::sscanf(argument,"%f%c%c",&nb,&sep,&end)==2 && sep=='%')) &&
              nb>0) {
            print(images,"Equalize image%s with %g%s clusters.",gmic_inds,nb,sep=='%'?"%":"");
            cimg_foroff(indices,l) {
              CImg<T>& img = images[indices[l]];
              unsigned int N = (unsigned int)nb;
              if (sep=='%') { double m, M = img.maxmin(m); N = (unsigned int)cimg::round((M-m)*nb/100,1); }
              gmic_apply(img,equalize((int)nb));
            }
          } else error(images,"Equalize image%s : Invalid argument '%s' "
                       "(should be 'nb_clusters[%%]>0').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Quantize.
        if (!cimg::strcmp("-quantize",command_name)) {
          int nb = 0; char end = 0;
          if (std::sscanf(argument,"%d%c",&nb,&end)==1 &&
              nb>0) {
            print(images,"Quantize image%s with %d levels.",gmic_inds,nb);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],quantize(nb));
          } else error(images,"Quantize image%s : Invalid argument '%s' "
                       "(should be 'nb_levels>0').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Add noise.
        if (!cimg::strcmp("-noise",command_name)) {
          float sigma = 0; char sep = 0, end = 0; int noise_type = 0;
          if ((std::sscanf(argument,"%f%c",&sigma,&end)==1 ||
               (std::sscanf(argument,"%f%c%c",&sigma,&sep,&end)==2 && sep=='%') ||
               std::sscanf(argument,"%f,%d%c",&sigma,&noise_type,&end)==2 ||
               (std::sscanf(argument,"%f%c,%d%c",&sigma,&sep,&noise_type,&end)==3 && sep=='%')) &&
              sigma>=0 && noise_type>=0 && noise_type<=3) {
            const char *st_type = noise_type==0?"gaussian":noise_type==1?"uniform":noise_type==2?"salt&pepper":"poisson";
            if (sep=='%') sigma = -sigma;
            print(images,"Add %s noise with standard deviation %g%s to image%s.",
                  st_type,cimg::abs(sigma),sep=='%'?"%":"",gmic_inds);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],noise(sigma,noise_type));
          } else error(images,"Add noise to image%s : Invalid argument '%s' "
                       "(should be 'std[%%]>=0,_noise_type={0,1,2,3}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Rand.
        if (!cimg::strcmp("-rand",command_name)) {
          double value0 = 0, value1 = 0; char end = 0;
          if (std::sscanf(argument,"%lf,%lf%c",&value0,&value1,&end)==2) {
            print(images,"Fill image%s with random values in range [%g,%g].",gmic_inds,value0,value1);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],rand((T)value0,(T)value1));
          } else error(images,"Fill image%s with random values : Invalid argument '%s' "
                       "(should be 'valmin,valmax').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Compute pointwise norms and orientations.
        gmic_simple_item("-norm",pointwise_norm,"Compute vector norm of image%s.");
        gmic_simple_item("-orientation",pointwise_orientation,"Compute vector orientation of image%s.");

        //------------------------
        // Color base conversions
        //------------------------
        gmic_simple_item("-rgb2hsv",RGBtoHSV,"Convert image%s from RGB to HSV colorbases.");
        gmic_simple_item("-rgb2hsl",RGBtoHSL,"Convert image%s from RGB to HSL colorbases.");
        gmic_simple_item("-rgb2hsi",RGBtoHSI,"Convert image%s from RGB to HSI colorbases.");
        gmic_simple_item("-rgb2yuv",RGBtoYUV,"Convert image%s from RGB to YUV colorbases.");
        gmic_simple_item("-rgb2ycbcr",RGBtoYCbCr,"Convert image%s from RGB to YCbCr colorbases.");
        gmic_simple_item("-rgb2xyz",RGBtoXYZ,"Convert image%s from RGB to XYZ colorbases.");
        gmic_simple_item("-rgb2lab",RGBtoLab,"Convert image%s from RGB to Lab colorbases.");
        gmic_simple_item("-rgb2cmy",RGBtoCMY,"Convert image%s from RGB to CMY colorbases.");
        gmic_simple_item("-rgb2cmyk",RGBtoCMYK,"Convert image%s from RGB to CMYK colorbases.");
        gmic_simple_item("-cmyk2rgb",CMYKtoRGB,"Convert image%s from CMYK to RGB colorbases.");
        gmic_simple_item("-cmy2rgb",CMYtoRGB,"Convert image%s from CMY to RGB colorbases.");
        gmic_simple_item("-lab2rgb",LabtoRGB,"Convert image%s from Lab to RGB colorbases.");
        gmic_simple_item("-xyz2rgb",XYZtoRGB,"Convert image%s from XYZ to RGB colorbases.");
        gmic_simple_item("-ycbcr2rgb",YCbCrtoRGB,"Convert image%s from YCbCr to RGB colorbases.");
        gmic_simple_item("-yuv2rgb",YUVtoRGB,"Convert image%s from YUV to RGB colorbases.");
        gmic_simple_item("-hsi2rgb",HSItoRGB,"Convert image%s from HSI to RGB colorbases.");
        gmic_simple_item("-hsl2rgb",HSLtoRGB,"Convert image%s from HSL to RGB colorbases.");
        gmic_simple_item("-hsv2rgb",HSVtoRGB,"Convert image%s from HSV to RGB colorbases.");

        // Map palette.
        if (!cimg::strcmp("-map",command_name)) {
          unsigned int lut_type = 0; int ind = 0; char sep = 0, end = 0;
          CImg<T> palette;
          if (std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') {
            gmic_check_indice(ind,"Map palette on image%s");
            print(images,"Map palette [%d] on image%s.",ind,gmic_inds);
            palette = images[ind];
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],map(palette));
          } else if (std::sscanf(argument,"%u%c",&lut_type,&end)==1 &&
                     lut_type<=2) {
            print(images,"Map %s palette on image%s.",lut_type==0?"default":lut_type==1?"rainbow":"cluster",gmic_inds);
            palette = lut_type==0?CImg<T>::default_LUT8():lut_type==1?CImg<T>::rainbow_LUT8():CImg<T>::contrast_LUT8();
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],map(palette));
          } else error(images,"Map palette on image%s : Invalid argument '%s' "
                       "(should be '[indice]' or 'predefined_palette={0,1,2}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Index/cluster image using a palette.
        if (!cimg::strcmp("-index",command_name)) {
          unsigned int lut_type = 0; int ind = 0, dithering = 0, map_indexes = 0; char sep = 0, end = 0;
          CImg<T> palette;
          if ((std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') ||
              std::sscanf(argument,"[%d],%d%c",&ind,&dithering,&end)==2 ||
              std::sscanf(argument,"[%d],%d,%d%c",&ind,&dithering,&map_indexes,&end)==3) {
            gmic_check_indice(ind,"Index vector values in image%s by palette");
            print(images,"Index vector values in image%s by palette [%d], %s dithering%s.",
                  gmic_inds,ind,dithering?"with":"without",map_indexes?" and index mapping":"");
            palette = images[ind];
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],index(palette,dithering?true:false,map_indexes?true:false));
          } else if ((std::sscanf(argument,"%u%c",&lut_type,&end)==1 ||
                      std::sscanf(argument,"%u,%d%c",&lut_type,&dithering,&end)==2 ||
                      std::sscanf(argument,"%u,%d,%d%c",&lut_type,&dithering,&map_indexes,&end)==3) &&
                     lut_type<=2) {
            print(images,"Index vector values in image%s by %s palette, %s dithering%s.",
                  gmic_inds,
                  lut_type==0?"default":lut_type==1?"rainbow":"cluster",dithering?"with":"without",
                  map_indexes?" and index mapping":"");
            palette = lut_type==0?CImg<T>::default_LUT8():lut_type==1?CImg<T>::rainbow_LUT8():CImg<T>::contrast_LUT8();
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],index(palette,dithering?true:false,map_indexes?true:false));
          } else error(images,"Index vector values in image%s by palette : Invalid argument '%s' "
                       "(should be '[indice],_dithering={0,1},_indexing={0,1}', or "
                       "'predefined_palette={0,1,2},_dithering={0,1},_mapping={0,1}').",gmic_inds,argument_text);
          ++position; continue;
        }

        //-----------------------
        // Geometric manipulation
        //-----------------------

        // Resize.
        if (!cimg::strcmp("-resize",command_name) || !cimg::strcmp("-r",command_name)) {
          char sepx = 0, sepy = 0, sepz = 0, sepv = 0, end = 0, argx[4096] = { 0 }, argy[4096] = { 0 }, argz[4096] = { 0 }, argv[4096] = { 0 };
          int borders = -1, indx = no_ind, indy = no_ind, indz = no_ind, indv = no_ind;
          float valx = -100, valy = -100, valz = -100, valv = -100;
          unsigned int interpolation = 1, center = 0;
          if (((std::sscanf(argument,"[%d%c%c",&indx,&sepx,&end)==2 && sepx==']') ||
               std::sscanf(argument,"[%d],%u%c",&indx,&interpolation,&end)==2 ||
               std::sscanf(argument,"[%d],%u,%d%c",&indx,&interpolation,&borders,&end)==3 ||
               std::sscanf(argument,"[%d],%u,%d,%u%c",&indx,&interpolation,&borders,&center,&end)==4) &&
              interpolation<=5 && borders>=-1 && borders<=2) {
            gmic_check_indice(indx,"Resize image%s");
            const int ivalx = images[indx].dimx(), ivaly = images[indx].dimy(), ivalz = images[indx].dimz(), ivalv = images[indx].dimv();
            print(images,"Resize image%s to %dx%dx%dx%d, with %s interpolation.",
                  gmic_inds,ivalx,ivaly,ivalz,ivalv,
                  interpolation==0?"no":interpolation==1?"nearest neighbor":
                  interpolation==2?"moving average":interpolation==3?"linear":
                  interpolation==4?"grid":"cubic");
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],resize(ivalx,ivaly,ivalz,ivalv,interpolation,borders,center?true:false));
            ++position;
          } else if ((std::sscanf(argument,"%4095[][0-9.eE%+-]%c",argx,&end)==1 ||
                      std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",argx,argy,&end)==2 ||
                      std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",argx,argy,argz,&end)==3 ||
                      std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",argx,argy,argz,argv,&end)==4 ||
                      std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%u%c",
                                  argx,argy,argz,argv,&interpolation,&end)==5 ||
                      std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%u,%d%c",
                                  argx,argy,argz,argv,&interpolation,&borders,&end)==6 ||
                      std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%u,%d,%u%c",
                                  argx,argy,argz,argv,&interpolation,&borders,&center,&end)==7) &&
                     (!*argx || std::sscanf(argx,"%f%c",&valx,&end)==1 ||
                      (std::sscanf(argx,"[%d%c%c",&indx,&sepx,&end)==2 && sepx==']') ||
                      (std::sscanf(argx,"%f%c%c",&valx,&sepx,&end)==2 && sepx=='%')) &&
                     (!*argy || std::sscanf(argy,"%f%c",&valy,&end)==1 ||
                      (std::sscanf(argy,"[%d%c%c",&indy,&sepy,&end)==2 && sepy==']') ||
                      (std::sscanf(argy,"%f%c%c",&valy,&sepy,&end)==2 && sepy=='%')) &&
                     (!*argz || std::sscanf(argz,"%f%c",&valz,&end)==1 ||
                      (std::sscanf(argz,"[%d%c%c",&indz,&sepz,&end)==2 && sepz==']') ||
                      (std::sscanf(argz,"%f%c%c",&valz,&sepz,&end)==2 && sepz=='%')) &&
                     (!*argv || std::sscanf(argv,"%f%c",&valv,&end)==1 ||
                      (std::sscanf(argv,"[%d%c%c",&indv,&sepv,&end)==2 && sepv==']') ||
                      (std::sscanf(argv,"%f%c%c",&valv,&sepv,&end)==2 && sepv=='%')) &&
                     interpolation<=5 && borders>=-1 && borders<=2) {
            if (indx!=no_ind) { gmic_check_indice(indx,"Resize image%s"); valx = (float)images[indx].dimx(); sepx = 0; }
            if (indy!=no_ind) { gmic_check_indice(indy,"Resize image%s"); valy = (float)images[indy].dimy(); sepy = 0; }
            if (indz!=no_ind) { gmic_check_indice(indz,"Resize image%s"); valz = (float)images[indz].dimz(); sepz = 0; }
            if (indv!=no_ind) { gmic_check_indice(indv,"Resize image%s"); valv = (float)images[indv].dimv(); sepv = 0; }
            if (valx<=0) { valx = -valx; sepx = '%'; }
            if (valy<=0) { valy = -valy; sepy = '%'; }
            if (valz<=0) { valz = -valz; sepz = '%'; }
            if (valv<=0) { valv = -valv; sepv = '%'; }
            print(images,"Resize image%s to %g%s%g%s%g%s%g%s, with %s interpolation.",
                  gmic_inds,valx,sepx=='%'?"%x":"x",valy,sepy=='%'?"%x":"x",valz,
                  sepz=='%'?"%x":"x",valv,sepv=='%'?"% ":"",
                  interpolation==0?"no":interpolation==1?"nearest neighbor":
                  interpolation==2?"moving average":interpolation==3?"linear":
                  interpolation==4?"grid":"cubic");
            cimg_foroff(indices,l) {
              CImg<T>& img = images[indices[l]];
              const int
                ivalx0 = (int)cimg::round(sepx=='%'?valx*img.dimx()/100:valx,1),
                ivaly0 = (int)cimg::round(sepy=='%'?valy*img.dimy()/100:valy,1),
                ivalz0 = (int)cimg::round(sepz=='%'?valz*img.dimz()/100:valz,1),
                ivalv0 = (int)cimg::round(sepv=='%'?valv*img.dimv()/100:valv,1),
                ivalx = ivalx0?ivalx0:1,
                ivaly = ivaly0?ivaly0:1,
                ivalz = ivalz0?ivalz0:1,
                ivalv = ivalv0?ivalv0:1;
              gmic_apply(img,resize(ivalx,ivaly,ivalz,ivalv,interpolation,borders,center?true:false));
            }
            ++position;
          } else {
#if cimg_display==0
            print(images,"Resize image%s : interactive mode skipped (no display available).",gmic_inds);
            continue;
#endif
            print(images,"Resize image%s : interactive mode.",gmic_inds);
            char title[4096] = { 0 };
            cimg_foroff(indices,l) {
              CImg<T>& img = images[indices[l]];
              CImgDisplay disp(img,0,1);
              std::sprintf(title,"%s : resize",filenames[indices[l]].ptr());
              disp.set_title("%s",title);
              img.get_select(disp,0);
              print(images,"Resize image [%d] to %dx%d, with nearest neighbor interpolation.",indices[l],disp.dimx(),disp.dimy());
              gmic_apply(img,resize(disp));
            }
          }
          continue;
        }

        // Resize2x. and Resize3x.
        gmic_simple_item("-resize2x",resize_doubleXY,"Double size of image%s, using Scale2x algorithm.");
        gmic_simple_item("-resize3x",resize_doubleXY,"Triple size of image%s, using Scale3x algorithm.");

        // Crop.
        if (!cimg::strcmp("-crop",command_name) || !cimg::strcmp("-c",command_name)) {
          char st0[4096] = { 0 }, st1[4096] = { 0 }, st2[4096] = { 0 }, st3[4096] = { 0 };
          char st4[4096] = { 0 }, st5[4096] = { 0 }, st6[4096] = { 0 }, st7[4096] = { 0 };
          char sep0 = 0, sep1 = 0, sep2 = 0, sep3 = 0, sep4 = 0, sep5 = 0, sep6 = 0, sep7 = 0, end = 0;
          float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0; unsigned int borders = 0;

          if ((std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",st0,st1,&end)==2 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%u%c",st0,st1,&borders,&end)==3) &&
              (std::sscanf(st0,"%f%c",&a0,&end)==1 || (std::sscanf(st0,"%f%c%c",&a0,&sep0,&end)==2 && sep0=='%')) &&
              (std::sscanf(st1,"%f%c",&a1,&end)==1 || (std::sscanf(st1,"%f%c%c",&a1,&sep1,&end)==2 && sep1=='%'))) {
            print(images,"Crop image%s with (%g%s x (%g%s.",gmic_inds,
                  a0,sep0=='%'?"%)":")",a1,sep1=='%'?"%)":")");
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                x0 = (int)cimg::round(sep0=='%'?a0*(img.dimx()-1)/100:a0,1),
                x1 = (int)cimg::round(sep1=='%'?a1*(img.dimx()-1)/100:a1,1);
              gmic_apply(img,crop(x0,x1,borders?true:false));
            }
            ++position;
          } else if ((std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",st0,st1,st2,st3,&end)==4 ||
                      std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%u%c",st0,st1,st2,st3,&borders,&end)==5) &&
                     (std::sscanf(st0,"%f%c",&a0,&end)==1 || (std::sscanf(st0,"%f%c%c",&a0,&sep0,&end)==2 && sep0=='%')) &&
                     (std::sscanf(st1,"%f%c",&a1,&end)==1 || (std::sscanf(st1,"%f%c%c",&a1,&sep1,&end)==2 && sep1=='%')) &&
                     (std::sscanf(st2,"%f%c",&a2,&end)==1 || (std::sscanf(st2,"%f%c%c",&a2,&sep2,&end)==2 && sep2=='%')) &&
                     (std::sscanf(st3,"%f%c",&a3,&end)==1 || (std::sscanf(st3,"%f%c%c",&a3,&sep3,&end)==2 && sep3=='%'))) {
            print(images,"Crop image%s with (%g%s%g%s x (%g%s%g%s.",gmic_inds,
                  a0,sep0=='%'?"%,":",",a1,sep1=='%'?"%)":")",
                  a2,sep2=='%'?"%,":",",a3,sep3=='%'?"%)":")");
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                x0 = (int)cimg::round(sep0=='%'?a0*(img.dimx()-1)/100:a0,1),
                y0 = (int)cimg::round(sep1=='%'?a1*(img.dimy()-1)/100:a1,1),
                x1 = (int)cimg::round(sep2=='%'?a2*(img.dimx()-1)/100:a2,1),
                y1 = (int)cimg::round(sep3=='%'?a3*(img.dimy()-1)/100:a3,1);
              gmic_apply(img,crop(x0,y0,x1,y1,borders?true:false));
            }
            ++position;
          } else if ((std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",
                                  st0,st1,st2,st3,st4,st5,&end)==6 ||
                      std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%u%c",
                                  st0,st1,st2,st3,st4,st5,&borders,&end)==7) &&
                     (std::sscanf(st0,"%f%c",&a0,&end)==1 || (std::sscanf(st0,"%f%c%c",&a0,&sep0,&end)==2 && sep0=='%')) &&
                     (std::sscanf(st1,"%f%c",&a1,&end)==1 || (std::sscanf(st1,"%f%c%c",&a1,&sep1,&end)==2 && sep1=='%')) &&
                     (std::sscanf(st2,"%f%c",&a2,&end)==1 || (std::sscanf(st2,"%f%c%c",&a2,&sep2,&end)==2 && sep2=='%')) &&
                     (std::sscanf(st3,"%f%c",&a3,&end)==1 || (std::sscanf(st3,"%f%c%c",&a3,&sep3,&end)==2 && sep3=='%')) &&
                     (std::sscanf(st4,"%f%c",&a4,&end)==1 || (std::sscanf(st4,"%f%c%c",&a4,&sep4,&end)==2 && sep4=='%')) &&
                     (std::sscanf(st5,"%f%c",&a5,&end)==1 || (std::sscanf(st5,"%f%c%c",&a5,&sep5,&end)==2 && sep5=='%'))) {
            print(images,"Crop image%s with (%g%s%g%s%g%s x (%g%s%g%s%g%s.",gmic_inds,
                  a0,sep0=='%'?"%,":",",a1,sep1=='%'?"%,":",",a2,sep2=='%'?"%)":")",
                  a3,sep3=='%'?"%,":",",a4,sep4=='%'?"%,":",",a5,sep5=='%'?"%)":")");
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                x0 = (int)cimg::round(sep0=='%'?a0*(img.dimx()-1)/100:a0,1),
                y0 = (int)cimg::round(sep1=='%'?a1*(img.dimy()-1)/100:a1,1),
                z0 = (int)cimg::round(sep2=='%'?a2*(img.dimz()-1)/100:a2,1),
                x1 = (int)cimg::round(sep3=='%'?a3*(img.dimx()-1)/100:a3,1),
                y1 = (int)cimg::round(sep4=='%'?a4*(img.dimy()-1)/100:a4,1),
                z1 = (int)cimg::round(sep5=='%'?a5*(img.dimz()-1)/100:a5,1);
              gmic_apply(img,crop(x0,y0,z0,x1,y1,z1,borders?true:false));
            }
            ++position;
          } else if ((std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],"
                                  "%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",
                                  st0,st1,st2,st3,st4,st5,st6,st7,&end)==8 ||
                      std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],"
                                  "%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%u%c",
                                  st0,st1,st2,st3,st4,st5,st6,st7,&borders,&end)==9) &&
                     (std::sscanf(st0,"%f%c",&a0,&end)==1 || (std::sscanf(st0,"%f%c%c",&a0,&sep0,&end)==2 && sep0=='%')) &&
                     (std::sscanf(st1,"%f%c",&a1,&end)==1 || (std::sscanf(st1,"%f%c%c",&a1,&sep1,&end)==2 && sep1=='%')) &&
                     (std::sscanf(st2,"%f%c",&a2,&end)==1 || (std::sscanf(st2,"%f%c%c",&a2,&sep2,&end)==2 && sep2=='%')) &&
                     (std::sscanf(st3,"%f%c",&a3,&end)==1 || (std::sscanf(st3,"%f%c%c",&a3,&sep3,&end)==2 && sep3=='%')) &&
                     (std::sscanf(st4,"%f%c",&a4,&end)==1 || (std::sscanf(st4,"%f%c%c",&a4,&sep4,&end)==2 && sep4=='%')) &&
                     (std::sscanf(st5,"%f%c",&a5,&end)==1 || (std::sscanf(st5,"%f%c%c",&a5,&sep5,&end)==2 && sep5=='%')) &&
                     (std::sscanf(st6,"%f%c",&a6,&end)==1 || (std::sscanf(st6,"%f%c%c",&a6,&sep6,&end)==2 && sep6=='%')) &&
                     (std::sscanf(st7,"%f%c",&a7,&end)==1 || (std::sscanf(st7,"%f%c%c",&a7,&sep7,&end)==2 && sep7=='%'))) {
            print(images,"Crop image%s with (%g%s%g%s%g%s%g%s x (%g%s%g%s%g%s%g%s.",gmic_inds,
                  a0,sep0=='%'?"%,":",",a1,sep1=='%'?"%,":",",
                  a2,sep2=='%'?"%,":",",a3,sep3=='%'?"%)":")",
                  a4,sep4=='%'?"%,":",",a5,sep5=='%'?"%,":",",
                  a6,sep6=='%'?"%,":",",a7,sep7=='%'?"%)":")");
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                x0 = (int)cimg::round(sep0=='%'?a0*(img.dimx()-1)/100:a0,1),
                y0 = (int)cimg::round(sep1=='%'?a1*(img.dimy()-1)/100:a1,1),
                z0 = (int)cimg::round(sep2=='%'?a2*(img.dimz()-1)/100:a2,1),
                v0 = (int)cimg::round(sep3=='%'?a3*(img.dimv()-1)/100:a3,1),
                x1 = (int)cimg::round(sep4=='%'?a4*(img.dimx()-1)/100:a4,1),
                y1 = (int)cimg::round(sep5=='%'?a5*(img.dimy()-1)/100:a5,1),
                z1 = (int)cimg::round(sep6=='%'?a6*(img.dimz()-1)/100:a6,1),
                v1 = (int)cimg::round(sep7=='%'?a7*(img.dimv()-1)/100:a7,1);
              gmic_apply(img,crop(x0,y0,z0,v0,x1,y1,z1,v1,borders?true:false));
            }
            ++position;
          } else {
            print(images,"Crop image%s : interactive mode.",gmic_inds);
            char title[4096] = { 0 };
            cimg_foroff(indices,l) {
              CImg<T>& img = images[indices[l]];
              CImgDisplay disp(cimg_fitscreen(img.dimx(),img.dimy(),1),0,1);
              std::sprintf(title,"%s : crop",filenames[indices[l]].ptr());
              disp.set_title("%s",title);
              const CImg<int> s = img.get_select(disp,2);
              print(images,"Crop image [%d] with (%d,%d,%d) x (%d,%d,%d).",indices[l],s[0],s[1],s[2],s[3],s[4],s[5]);
              gmic_apply(img,crop(s[0],s[1],s[2],s[3],s[4],s[5]));
            }
          }
          continue;
        }

        // Autocrop.
        if (!cimg::strcmp("-autocrop",command_name)) {
          print(images,"Auto-crop image%s by color '%s'.",gmic_inds,argument_text);
          cimg_foroff(indices,l) {
            CImg<T>& img = images[indices[l]];
            const CImg<T> col = CImg<T>(img.dimv()).fill(argument,true);
            gmic_apply(img,autocrop(col));
          }
          ++position; continue;
        }

        // Select channels.
        if (!cimg::strcmp("-channels",command_name)) {
          char sep0 = 0, sep1 = 0, end = 0, arg0[4096] = { 0 }, arg1[4096] = { 0 };
          float value0 = 0, value1 = 0; int ind0 = no_ind, ind1 = no_ind;
          if (std::sscanf(argument,"%4095[][0-9.eE%+-]%c",arg0,&end)==1 &&
              (std::sscanf(arg0,"%f%c",&value0,&end)==1 ||
               (std::sscanf(arg0,"[%d%c%c]",&ind0,&sep0,&end)==2 && sep0==']') ||
               (std::sscanf(arg0,"%f%c%c",&value0,&sep0,&end)==2 && sep0=='%'))) {
            if (ind0!=no_ind) { gmic_check_indice(ind0,"Keep channel of image%s"); value0 = images[ind0].dimv()-1.0f; sep0 = 0; }
            print(images,"Keep channel %g%s of image%s.",value0,sep0=='%'?"%":"",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int nvalue0 = (int)cimg::round(sep0=='%'?value0*(img.dimv()-1)/100:value0,1);
              gmic_apply(img,channel(nvalue0));
            }
          } else if (std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",arg0,arg1,&end)==2 &&
                     (std::sscanf(arg0,"%f%c",&value0,&end)==1 ||
                      (std::sscanf(arg0,"[%d%c%c",&ind0,&sep0,&end)==2 && sep0==']') ||
                      (std::sscanf(arg0,"%f%c%c",&value0,&sep0,&end)==2 && sep0=='%')) &&
                     (std::sscanf(arg1,"%f%c",&value1,&end)==1 ||
                      (std::sscanf(arg1,"[%d%c%c",&ind1,&sep1,&end)==2 && sep1==']') ||
                      (std::sscanf(arg1,"%f%c%c",&value1,&sep1,&end)==2 && sep1=='%'))) {
            if (ind0!=no_ind) { gmic_check_indice(ind0,"Keep channels of image%s"); value0 = images[ind0].dimv()-1.0f; sep0 = 0; }
            if (ind1!=no_ind) { gmic_check_indice(ind1,"Keep channels of image%s"); value1 = images[ind1].dimv()-1.0f; sep1 = 0; }
            print(images,"Keep channels %g%s..%g%s of image%s.",value0,sep0=='%'?"%":"",value1,sep1=='%'?"%":"",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                nvalue0 = (int)cimg::round(sep0=='%'?value0*(img.dimv()-1)/100:value0,1),
                nvalue1 = (int)cimg::round(sep1=='%'?value1*(img.dimv()-1)/100:value1,1);
              gmic_apply(img,channels(nvalue0,nvalue1));
            }
          } else error(images,"Keep channels of image%s : Invalid argument '%s' "
                       "(should be 'channel[%%]' or 'channel0[%%],channel1[%%]').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Select slices.
        if (!cimg::strcmp("-slices",command_name)) {
          char sep0 = 0, sep1 = 0, end = 0, arg0[4096] = { 0 }, arg1[4096] = { 0 };
          float value0 = 0, value1 = 0; int ind0 = no_ind, ind1 = no_ind;
          if (std::sscanf(argument,"%4095[][0-9.eE%+-]%c",arg0,&end)==1 &&
              (std::sscanf(arg0,"%f%c",&value0,&end)==1 ||
               (std::sscanf(arg0,"[%d%c%c]",&ind0,&sep0,&end)==2 && sep0==']') ||
               (std::sscanf(arg0,"%f%c%c",&value0,&sep0,&end)==2 && sep0=='%'))) {
            if (ind0!=no_ind) { gmic_check_indice(ind0,"Keep slice of image%s"); value0 = images[ind0].dimz()-1.0f; sep0 = 0; }
            print(images,"Keep slice %g%s of image%s.",value0,sep0=='%'?"%":"",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int nvalue0 = (int)cimg::round(sep0=='%'?value0*(img.dimz()-1)/100:value0,1);
              gmic_apply(img,slice(nvalue0));
            }
          } else if (std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",arg0,arg1,&end)==2 &&
                     (std::sscanf(arg0,"%f%c",&value0,&end)==1 ||
                      (std::sscanf(arg0,"[%d%c%c",&ind0,&sep0,&end)==2 && sep0==']') ||
                      (std::sscanf(arg0,"%f%c%c",&value0,&sep0,&end)==2 && sep0=='%')) &&
                     (std::sscanf(arg1,"%f%c",&value1,&end)==1 ||
                      (std::sscanf(arg1,"[%d%c%c",&ind1,&sep1,&end)==2 && sep1==']') ||
                      (std::sscanf(arg1,"%f%c%c",&value1,&sep1,&end)==2 && sep1=='%'))) {
            if (ind0!=no_ind) { gmic_check_indice(ind0,"Keep slices of image%s"); value0 = images[ind0].dimz()-1.0f; sep0 = 0; }
            if (ind1!=no_ind) { gmic_check_indice(ind1,"Keep slices of image%s"); value1 = images[ind1].dimz()-1.0f; sep1 = 0; }
            print(images,"Keep slices %g%s..%g%s of image%s.",value0,sep0=='%'?"%":"",value1,sep1=='%'?"%":"",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                nvalue0 = (int)cimg::round(sep0=='%'?value0*(img.dimz()-1)/100:value0,1),
                nvalue1 = (int)cimg::round(sep1=='%'?value1*(img.dimz()-1)/100:value1,1);
              gmic_apply(img,slices(nvalue0,nvalue1));
            }
          } else error(images,"Keep slices of image%s : Invalid argument '%s' "
                       "(should be 'slice[%%]' or 'slice0[%%],slice1[%%]').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Select lines.
        if (!cimg::strcmp("-lines",command_name)) {
          char sep0 = 0, sep1 = 0, end = 0, arg0[4096] = { 0 }, arg1[4096] = { 0 };
          float value0 = 0, value1 = 0; int ind0 = no_ind, ind1 = no_ind;
          if (std::sscanf(argument,"%4095[][0-9.eE%+-]%c",arg0,&end)==1 &&
              (std::sscanf(arg0,"%f%c",&value0,&end)==1 ||
               (std::sscanf(arg0,"[%d%c%c]",&ind0,&sep0,&end)==2 && sep0==']') ||
               (std::sscanf(arg0,"%f%c%c",&value0,&sep0,&end)==2 && sep0=='%'))) {
            if (ind0!=no_ind) { gmic_check_indice(ind0,"Keep line of image%s"); value0 = images[ind0].dimy()-1.0f; sep0 = 0; }
            print(images,"Keep line %g%s of image%s.",value0,sep0=='%'?"%":"",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int nvalue0 = (int)cimg::round(sep0=='%'?value0*(img.dimy()-1)/100:value0,1);
              gmic_apply(img,line(nvalue0));
            }
          } else if (std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",arg0,arg1,&end)==2 &&
                     (std::sscanf(arg0,"%f%c",&value0,&end)==1 ||
                      (std::sscanf(arg0,"[%d%c%c",&ind0,&sep0,&end)==2 && sep0==']') ||
                      (std::sscanf(arg0,"%f%c%c",&value0,&sep0,&end)==2 && sep0=='%')) &&
                     (std::sscanf(arg1,"%f%c",&value1,&end)==1 ||
                      (std::sscanf(arg1,"[%d%c%c",&ind1,&sep1,&end)==2 && sep1==']') ||
                      (std::sscanf(arg1,"%f%c%c",&value1,&sep1,&end)==2 && sep1=='%'))) {
            if (ind0!=no_ind) { gmic_check_indice(ind0,"Keep lines of image%s"); value0 = images[ind0].dimy()-1.0f; sep0 = 0; }
            if (ind1!=no_ind) { gmic_check_indice(ind1,"Keep lines of image%s"); value1 = images[ind1].dimy()-1.0f; sep1 = 0; }
            print(images,"Keep lines %g%s..%g%s of image%s.",value0,sep0=='%'?"%":"",value1,sep1=='%'?"%":"",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                nvalue0 = (int)cimg::round(sep0=='%'?value0*(img.dimy()-1)/100:value0,1),
                nvalue1 = (int)cimg::round(sep1=='%'?value1*(img.dimy()-1)/100:value1,1);
              gmic_apply(img,lines(nvalue0,nvalue1));
            }
          } else error(images,"Keep lines of image%s : Invalid argument '%s' "
                       "(should be 'line[%%]' or 'line0[%%],line1[%%]').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Select columns.
        if (!cimg::strcmp("-columns",command_name)) {
          char sep0 = 0, sep1 = 0, end = 0, arg0[4096] = { 0 }, arg1[4096] = { 0 };
          float value0 = 0, value1 = 0; int ind0 = no_ind, ind1 = no_ind;
          if (std::sscanf(argument,"%4095[][0-9.eE%+-]%c",arg0,&end)==1 &&
              (std::sscanf(arg0,"%f%c",&value0,&end)==1 ||
               (std::sscanf(arg0,"[%d%c%c]",&ind0,&sep0,&end)==2 && sep0==']') ||
               (std::sscanf(arg0,"%f%c%c",&value0,&sep0,&end)==2 && sep0=='%'))) {
            if (ind0!=no_ind) { gmic_check_indice(ind0,"Keep column of image%s"); value0 = images[ind0].dimx()-1.0f; sep0 = 0; }
            print(images,"Keep column %g%s of image%s.",value0,sep0=='%'?"%":"",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int nvalue0 = (int)cimg::round(sep0=='%'?value0*(img.dimx()-1)/100:value0,1);
              gmic_apply(img,column(nvalue0));
            }
          } else if (std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",arg0,arg1,&end)==2 &&
                     (std::sscanf(arg0,"%f%c",&value0,&end)==1 ||
                      (std::sscanf(arg0,"[%d%c%c",&ind0,&sep0,&end)==2 && sep0==']') ||
                      (std::sscanf(arg0,"%f%c%c",&value0,&sep0,&end)==2 && sep0=='%')) &&
                     (std::sscanf(arg1,"%f%c",&value1,&end)==1 ||
                      (std::sscanf(arg1,"[%d%c%c",&ind1,&sep1,&end)==2 && sep1==']') ||
                      (std::sscanf(arg1,"%f%c%c",&value1,&sep1,&end)==2 && sep1=='%'))) {
            if (ind0!=no_ind) { gmic_check_indice(ind0,"Keep columns of image%s"); value0 = images[ind0].dimx()-1.0f; sep0 = 0; }
            if (ind1!=no_ind) { gmic_check_indice(ind1,"Keep columns of image%s"); value1 = images[ind1].dimx()-1.0f; sep1 = 0; }
            print(images,"Keep columns %g%s..%g%s of image%s.",value0,sep0=='%'?"%":"",value1,sep1=='%'?"%":"",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                nvalue0 = (int)cimg::round(sep0=='%'?value0*(img.dimx()-1)/100:value0,1),
                nvalue1 = (int)cimg::round(sep1=='%'?value1*(img.dimx()-1)/100:value1,1);
              gmic_apply(img,columns(nvalue0,nvalue1));
            }
          } else error(images,"Keep columns of image%s : Invalid argument '%s' "
                       "(should be 'column[%%]' or 'column0[%%],column1[%%]').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Rotate.
        if (!cimg::strcmp("-rotate",command_name)) {
          float angle = 0; int borders = 0, interpolation = 1; char end = 0;
          if ((std::sscanf(argument,"%f%c",&angle,&end)==1 ||
               std::sscanf(argument,"%f,%d%c",&angle,&borders,&end)==2 ||
               std::sscanf(argument,"%f,%d,%d%c",&angle,&borders,&interpolation,&end)==3) &&
              borders>=-3 && borders<=2 && interpolation>=0 && interpolation<=2) {
            print(images,"Rotate image%s with an angle of %g deg and %s interpolation.",
                  gmic_inds,angle,interpolation==0?"nearest-neighbor":interpolation==1?"linear":"cubic");
            if (borders>=0) { cimg_foroff(indices,l) gmic_apply(images[indices[l]],rotate(angle,borders,interpolation)); }
            else cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              gmic_apply(img,rotate(angle,img.dimx()/2.0f,img.dimy()/2.0f,1,-1-borders,interpolation));
            }
          } else error(images,"Rotate image%s : Invalid argument '%s' "
                       "(should be 'angle,_border_conditions={-3,-2,-1,0,1,2},_interpolation={0,1,2}').",gmic_inds,argument_text);
          ++position;
          continue;
        }

        // Mirror.
        if (!cimg::strcmp("-mirror",command_name)) {
          const char axis = cimg::uncase(*argument);
          if (std::strlen(argument)==1 &&
              (axis=='x' || axis=='y' || axis=='z' || axis=='v')) {
            print(images,"Mirror image%s along the %c-axis.",gmic_inds,axis);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],mirror(axis));
          } else error(images,"Mirror image%s : Invalid argument '%s' "
                       "(should be 'axis={x,y,z,v}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Translate.
        if (!cimg::strcmp("-translate",command_name)) {
          char argx[4096] = { 0 }, argy[4096] = { 0 }, argz[4096] = { 0 }, argv[4096] = { 0 };
          char sepx = 0, sepy = 0, sepz = 0, sepv = 0, end = 0;
          float dx = 0, dy = 0, dz = 0, dv = 0; unsigned int borders = 0;
          if ((std::sscanf(argument,"%4095[0-9.eE%+-]%c",argx,&end)==1 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",argx,argy,&end)==2 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",argx,argy,argz,&end)==3 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",argx,argy,argz,argv,&end)==4 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%u%c",argx,argy,argz,argv,&borders,&end)==5) &&
              (!*argx || std::sscanf(argx,"%f%c",&dx,&end)==1 || (std::sscanf(argx,"%f%c%c",&dx,&sepx,&end)==2 && sepx=='%')) &&
              (!*argy || std::sscanf(argy,"%f%c",&dy,&end)==1 || (std::sscanf(argy,"%f%c%c",&dy,&sepy,&end)==2 && sepy=='%')) &&
              (!*argz || std::sscanf(argz,"%f%c",&dz,&end)==1 || (std::sscanf(argz,"%f%c%c",&dz,&sepz,&end)==2 && sepz=='%')) &&
              (!*argv || std::sscanf(argv,"%f%c",&dv,&end)==1 || (std::sscanf(argv,"%f%c%c",&dv,&sepv,&end)==2 && sepv=='%')) &&
              borders<=2) {
            print(images,"Translate image%s with vector (%g%s,%g%s,%g%s,%g%s).",
                  gmic_inds,dx,sepx=='%'?"%":"",dy,sepy=='%'?"%":"",dz,sepz=='%'?"%":"",dv,sepv=='%'?"%":"");
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                ndx = (int)cimg::round(sepx=='%'?dx*img.dimx()/100:dx,1),
                ndy = (int)cimg::round(sepy=='%'?dy*img.dimy()/100:dy,1),
                ndz = (int)cimg::round(sepz=='%'?dz*img.dimz()/100:dz,1),
                ndv = (int)cimg::round(sepv=='%'?dv*img.dimv()/100:dv,1);
              gmic_apply(images[indices[l]],translate(ndx,ndy,ndz,ndv,borders));
            }
          } else error(images,"Translate image%s : Invalid argument '%s' "
                       "(should be 'tx[%%],_ty[%%]=0,_tz[%%]=0,_tv[%%]=0,_border_conditions={0,1,2}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Transpose.
        gmic_simple_item("-transpose",transpose,"Transpose image%s.");

        // Invert.
        gmic_simple_item("-invert",invert,"Compute matrix inverse of image%s.");

        // Permute axes.
        if (!cimg::strcmp("-permute",command_name)) {
          print(images,"Permute axes of image%s, with permutation '%s'.",gmic_inds,argument_text);
          cimg_foroff(indices,l) gmic_apply(images[indices[l]],permute_axes(argument));
          ++position; continue;
        }

        // Unroll.
        if (!cimg::strcmp("-unroll",command_name)) {
          const char axis = cimg::uncase(*argument);
          if (std::strlen(argument)==1 &&
              (axis=='x' || axis=='y' || axis=='z' || axis=='v')) {
            print(images,"Unroll image%s along the %c-axis.",gmic_inds,axis);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],unroll(axis));
          } else error(images,"Unroll image%s : Invalid argument '%s' "
                       "(should be 'axis={x,y,z,v}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Split image(s).
        if (!cimg::strcmp("-split",command_name) || !cimg::strcmp("-s",command_name)) {
          char axis = cimg::uncase(*argument), foo = 0, end = 0; int nb = 0, keep_value = 0; double value = 0;
          if ((std::sscanf(argument,"%c%c",&foo,&end)==1 ||
               std::sscanf(argument,"%c,%d%c",&foo,&nb,&end)==2) &&
              (axis=='x' || axis=='y' || axis=='z' || axis=='v')) {
            if (nb>0) print(images,"Split image%s along the %c-axis into %d parts.",gmic_inds,axis,nb);
            else if (nb<0) print(images,"Split image%s along the %c-axis into blocs of %d pixels.",gmic_inds,axis,-nb);
            else print(images,"Split image%s along the %c-axis.",gmic_inds,axis);
            unsigned int off = 0;
            cimg_foroff(indices,l) {
              const unsigned int ind = indices[l] + off;
              const CImg<T>& img = images[ind];
              const CImg<char> filename = filenames[ind];
              const CImgList<T> split = img.get_split(axis,nb);
              if (get_version) {
                images.insert(split);
                filenames.insert(split.size,filenames);
              } else {
                images.remove(ind); images.insert(split,ind);
                filenames.remove(ind); filenames.insert(split.size,filename,ind);
                off+=split.size-1;
              }
            }
          } else if ((std::sscanf(argument,"%lf%c",&value,&end)==1 || std::sscanf(argument,"%lf,%d%c",&value,&keep_value,&end)==2) &&
                     (keep_value==0 || keep_value==1)) {
            print(images,"Split image%s according to value %g.",gmic_inds,value);
            unsigned int off = 0;
            cimg_foroff(indices,l) {
              const unsigned int ind = indices[l] + off;
              CImg<T>& img = images[ind];
              const CImg<char> filename = filenames[ind];
              const CImgList<T> split = img.get_split((T)value,keep_value,false);
              if (get_version) {
                images.insert(split);
                filenames.insert(split.size,filename);
              } else {
                images.remove(ind); images.insert(split,ind);
                filenames.remove(ind); filenames.insert(split.size,filename,ind);
                off+=split.size-1;
              }
            }
          } else error(images,"Split image%s : Invalid argument '%s' "
                       "(should be 'axis={x,y,z,v},_nb_parts' or 'value,_keep_value={0,1}).",gmic_inds,argument_text);
          ++position; continue;
        }

        // Append image(s).
        if (!cimg::strcmp("-append",command_name) || !cimg::strcmp("-a",command_name)) {
          char axis = 0, align='p', end = 0;
          if ((std::sscanf(argument,"%c%c",&axis,&end)==1 ||
               std::sscanf(argument,"%c,%c%c",&axis,&align,&end)==2) &&
              (axis=='x' || axis=='y' || axis=='z' || axis=='v') &&
              (align=='p' || align=='c' || align=='n')) {
            axis = cimg::uncase(axis);
            print(images,"Append image%s along the %c-axis with %s alignment.",
                  gmic_inds,axis,align=='p'?"left":align=='c'?"center":"right");
            CImgList<T> subimages; cimg_foroff(indices,l) subimages.insert(images[indices[l]],~0U,true);
            if (get_version) {
              images.insert(subimages.get_append(axis,align));
              filenames.insert(filenames[indices[0]]);
            } else {
              images.insert(subimages.get_append(axis,align),indices[0]);
              filenames.insert(filenames[indices[0]],indices[0]);
              int off = 1;
              cimg_foroff(indices,l) {
                const int ind = indices[l] + off;
                images.remove(ind); filenames.remove(ind);
                --off;
              }
            }
          } else error(images,"Append image%s : Invalid argument '%s' "
                       "(should be 'axis={x,y,z,v},_alignment={p,c,n}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Warp image(s).
        if (!cimg::strcmp("-warp",command_name)) {
          int ind0 = no_ind, nb_frames = 1; unsigned int interpolation = 1, relative = 0, borders = 1; char end = 0, sep = 0;
          if (((std::sscanf(argument,"[%d%c%c",&ind0,&sep,&end)==2 && sep==']')||
               std::sscanf(argument,"[%d],%u%c",&ind0,&relative,&end)==2 ||
               std::sscanf(argument,"[%d],%u,%u%c",&ind0,&relative,&interpolation,&end)==3 ||
               std::sscanf(argument,"[%d],%u,%u,%u%c",&ind0,&relative,&interpolation,&borders,&end)==4 ||
               std::sscanf(argument,"[%d],%u,%u,%u,%d%c",&ind0,&relative,&interpolation,&borders,&nb_frames,&end)==5) &&
              borders<=2 && nb_frames>=1) {
            gmic_check_indice(ind0,"Warp image%s");
            if (nb_frames>1) print(images,"Warp image%s with %s field [%u] and %d frames.",gmic_inds,relative?"relative":"absolute",ind0,nb_frames);
            else print(images,"Warp image%s with %s field [%u].",gmic_inds,relative?"relative":"absolute",ind0);
            const CImg<T> warp = images[ind0];
            unsigned int off = 0;
            cimg_foroff(indices,l) {
              const unsigned int ind = indices[l] + off;
              CImg<T> &img = images[ind];
              CImgList<T> frames(nb_frames);
              cimglist_for(frames,t) frames[t] = img.get_warp(warp*((t+1.0f)/nb_frames),relative?true:false,interpolation?true:false,borders);
              if (get_version) { images.insert(frames); filenames.insert(nb_frames,filenames[ind]); }
              else {
                images.remove(ind); images.insert(frames,ind);
                filenames.insert(nb_frames-1,filenames[ind],ind);
                off+=nb_frames-1;
              }
            }
          } else error(images,"Warp image%s : Invalid argument '%s' "
                       "(should be '[indice],_relative={0,1},_interpolation={0,1},_border_conditions={0,1,2},_nb_frames>=1').",
                       gmic_inds,argument_text);
          ++position; continue;
        }

        //-----------------------
        // Image filtering
        //-----------------------

        // Gaussian blur.
        if (!cimg::strcmp("-blur",command_name)) {
          float sigma = -1; unsigned int borders = 1; char sep = 0, end = 0;
          if (std::sscanf(argument,"%f%c",&sigma,&end)==1 ||
              (std::sscanf(argument,"%f%c%c",&sigma,&sep,&end)==2 && sep=='%') ||
              std::sscanf(argument,"%f,%u%c",&sigma,&borders,&end)==2 ||
              (std::sscanf(argument,"%f%c,%u%c",&sigma,&sep,&borders,&end)==3 && sep=='%')) {
            print(images,"Blur image%s with standard deviation %g%s.",gmic_inds,cimg::abs(sigma),(sigma<0 || sep=='%')?"%":"");
            if (sep=='%') sigma = -sigma;
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],blur(sigma,borders?true:false));
          } else error(images,"Blur image%s : Invalid argument '%s' "
                       "(should be 'stdev[%%],_border_conditions={0,1}'.",gmic_inds,argument_text);
          ++position; continue;
        }

        // Bilateral filter.
        if (!cimg::strcmp("-bilateral",command_name)) {
          float sigmas = 0, sigmar = 0; char sep =  0, end = 0;
          if (std::sscanf(argument,"%f,%f%c",&sigmas,&sigmar,&end)==2 ||
              (std::sscanf(argument,"%f%c,%f%c",&sigmas,&sep,&sigmar,&end)==3 && sep=='%')) {
            print(images,"Apply bilateral filter on image%s with standard deviations %g%s and %g.",
                  gmic_inds,cimg::abs(sigmas),(sigmas<0 || sep=='%')?"%":"",sigmar);
            if (sep=='%') sigmas = -sigmas;
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],blur_bilateral(sigmas,sigmar));
          } else error(images,"Apply bilateral filter on image%s : Invalid argument '%s' "
                       "(should be 'stdevs[%%],_stdevr').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Patch averaging.
        if (!cimg::strcmp("-denoise",command_name)) {
          float sigmas = 10, sigmar = 10, psmooth = 1; int psize = 5, rsize = 6; unsigned int fast_approximation = 1; char end = 0;
          if ((std::sscanf(argument,"%f%c",&sigmas,&end)==1 ||
               std::sscanf(argument,"%f,%f%c",&sigmas,&sigmar,&end)==2 ||
               std::sscanf(argument,"%f,%f,%d%c",&sigmas,&sigmar,&psize,&end)==3 ||
               std::sscanf(argument,"%f,%f,%d,%d%c",&sigmas,&sigmar,&psize,&rsize,&end)==4 ||
               std::sscanf(argument,"%f,%f,%d,%d,%f%c",&sigmas,&sigmar,&psize,&rsize,&psmooth,&end)==5 ||
               std::sscanf(argument,"%f,%f,%d,%d,%f,%u%c",&sigmas,&sigmar,&psize,&rsize,&psmooth,&fast_approximation,&end)==6) &&
              sigmas>=0 && sigmar>=0 && psize>0 && rsize>0) {
            print(images,"Denoise image%s with %dx%d patches, standard deviations %lg,%g, lookup size %d and smoothness %g.",
                  gmic_inds,psize,psize,sigmas,sigmar,rsize,psmooth);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],blur_patch(sigmas,sigmar,psize,rsize,psmooth,fast_approximation?true:false));
          } else error(images,"Denoise image%s : Invalid argument '%s' "
                       "(should be 'stdev_s>=0,_stdev_p>=0,_patch_size>0,_lookup_size>0,_smoothness,_fast_approximation={0,1}').",
                       gmic_inds,argument_text);
          ++position; continue;
        }

        // Smooth.
        if (!cimg::strcmp("-smooth",command_name)) {
          float amplitude = 0, sharpness = 0.7f, anisotropy = 0.3f, alpha = 0.6f, sigma = 1.1f, dl =0.8f, da = 30.0f, gauss_prec = 2.0f;
          int ind = no_ind; unsigned int interpolation_type = 0, fast_approx = 1; char sep = 0, end = 0;
          if ((std::sscanf(argument,"%f%c",&amplitude,&end)==1 ||
               std::sscanf(argument,"%f,%f%c",&amplitude,&sharpness,&end)==2 ||
               std::sscanf(argument,"%f,%f,%f%c",&amplitude,&sharpness,&anisotropy,&end)==3 ||
               std::sscanf(argument,"%f,%f,%f,%f%c",&amplitude,&sharpness,&anisotropy,&alpha,&end)==4 ||
               std::sscanf(argument,"%f,%f,%f,%f,%f%c",&amplitude,&sharpness,&anisotropy,&alpha,&sigma,&end)==5 ||
               std::sscanf(argument,"%f,%f,%f,%f,%f,%f%c",&amplitude,&sharpness,&anisotropy,&alpha,&sigma,&dl,&end)==6 ||
               std::sscanf(argument,"%f,%f,%f,%f,%f,%f,%f%c",&amplitude,&sharpness,&anisotropy,&alpha,&sigma,&dl,&da,&end)==7 ||
               std::sscanf(argument,"%f,%f,%f,%f,%f,%f,%f,%f%c",&amplitude,&sharpness,&anisotropy,&alpha,&sigma,&dl,&da,&gauss_prec,&end)==8 ||
               std::sscanf(argument,"%f,%f,%f,%f,%f,%f,%f,%f,%u%c",
                           &amplitude,&sharpness,&anisotropy,&alpha,&sigma,&dl,&da,&gauss_prec,&interpolation_type,&end)==9 ||
               std::sscanf(argument,"%f,%f,%f,%f,%f,%f,%f,%f,%u,%u%c",
                           &amplitude,&sharpness,&anisotropy,&alpha,&sigma,&dl,&da,&gauss_prec,&interpolation_type,&fast_approx,&end)==10) &&
              amplitude>=0 && sharpness>=0 && anisotropy>=0 && anisotropy<=1 && dl>0 && da>=0 && gauss_prec>0 &&
              interpolation_type<=2) {
            print(images,"Smooth image%s anisotropically with amplitude %g, sharpness %g, anisotropy %g, alpha %g and sigma %g.",
                  gmic_inds,amplitude,sharpness,anisotropy,alpha,sigma);
            cimg_foroff(indices,l)
              gmic_apply(images[indices[l]],blur_anisotropic(amplitude,sharpness,anisotropy,alpha,sigma,
                                                             dl,da,gauss_prec,interpolation_type,fast_approx?true:false));
          } else if (((std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') ||
                      std::sscanf(argument,"[%d],%f%c",&ind,&amplitude,&end)==2 ||
                      std::sscanf(argument,"[%d],%f,%f%c",&ind,&amplitude,&dl,&end)==3 ||
                      std::sscanf(argument,"[%d],%f,%f,%f%c",&ind,&amplitude,&dl,&da,&end)==4 ||
                      std::sscanf(argument,"[%d],%f,%f,%f,%f%c",&ind,&amplitude,&dl,&da,&gauss_prec,&end)==5 ||
                      std::sscanf(argument,"[%d],%f,%f,%f,%f,%u%c",&ind,&amplitude,&dl,&da,&gauss_prec,&interpolation_type,&end)==5 ||
                      std::sscanf(argument,"[%d],%f,%f,%f,%f,%u,%u%c",&ind,&amplitude,&dl,&da,&gauss_prec,&interpolation_type,&fast_approx,&end)==6) &&
                     amplitude>=0 && dl>0 && da>=0 && gauss_prec>0 && interpolation_type<=2) {
            gmic_check_indice(ind,"Smooth image%s anisotropically");
            const CImg<T> tensors = images[ind];
            print(images,"Smooth image%s anisotropically with tensor field [%d] and amplitude %g.",gmic_inds,ind,amplitude);
            cimg_foroff(indices,l)
              gmic_apply(images[indices[l]],blur_anisotropic(tensors,amplitude,dl,da,gauss_prec,interpolation_type,fast_approx));
          } else error(images,"Smooth image%s anisotropically : Invalid argument '%s' "
                       "(should be "
                       "'amplitude>=0,_sharpness>=0,_anisotropy=[0,1],_alpha,_sigma,_dl>0,"
                       "_da>0,_precision>0,_interpolation_type={0,1,2},_fast_approximation={0,1}' or "
                       "'nb_iters>=0,_sharpness>=0,_anisotropy=[0,1],_alpha,_sigma,_dt>0,0' or "
                       "'[ind],_amplitude>=0,_dl>0,_da>=0,_precision>0,_interpolation_type={0,1,2},_fast_approximation={0,1}' or "
                       "'[ind],_nb_iters>=0,_dt>0,0').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Get tensor geometry from image.
        if (!cimg::strcmp("-edgetensors",command_name)) {
          float sharpness = 0.7f, anisotropy = 0.3f, alpha = 0.6f, sigma = 1.1f; unsigned int is_sqrt = 0; char end = 0;
          if ((std::sscanf(argument,"%f%c",&sharpness,&end)==1 ||
               std::sscanf(argument,"%f,%f%c",&sharpness,&anisotropy,&end)==2 ||
               std::sscanf(argument,"%f,%f,%f%c",&sharpness,&anisotropy,&alpha,&end)==3 ||
               std::sscanf(argument,"%f,%f,%f,%f%c",&sharpness,&anisotropy,&alpha,&sigma,&end)==4 ||
               std::sscanf(argument,"%f,%f,%f,%f,%u%c",&sharpness,&anisotropy,&alpha,&sigma,&is_sqrt,&end)==5) &&
              sharpness>=0 && anisotropy>=0 && anisotropy<=1) {
            print(images,"Compute tensors for edge-preserving smoothing of image%s, with sharpness %g, anisotropy %g, alpha %g and sigma %g.",
                  gmic_inds,sharpness,anisotropy,alpha,sigma);
            cimg_foroff(indices,l)
              gmic_apply(images[indices[l]],edge_tensors(sharpness,anisotropy,alpha,sigma,is_sqrt?true:false));
          } else error(images,"Compute tensors for edge-preserving smoothing of image%s : Invalid argument '%s' "
                       "(should be 'sharpness>=0,_anisotropy=[0,1],_alpha,_sigma').",
                       gmic_inds,argument_text);
          ++position; continue;
        }

        // Median filter.
        if (!cimg::strcmp("-median",command_name)) {
          int siz = 3; char end = 0;
          if (std::sscanf(argument,"%d%c",&siz,&end)==1 &&
              siz>0) {
            print(images,"Apply median filter of size %d on image%s.",siz,gmic_inds);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],blur_median(siz));
          } else error(images,"Apply median filter on image%s : Invalid argument '%s' "
                       "(should be 'size>0').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Sharpen.
        if (!cimg::strcmp("-sharpen",command_name)) {
          float amplitude = 0, edge = 1, alpha = 0, sigma = 0; unsigned int sharpen_type = 0; char end = 0;
          if ((std::sscanf(argument,"%f%c",&amplitude,&end)==1 ||
               std::sscanf(argument,"%f,%u%c",&amplitude,&sharpen_type,&end)==2 ||
               std::sscanf(argument,"%f,%u,%f%c",&amplitude,&sharpen_type,&edge,&end)==3 ||
               std::sscanf(argument,"%f,%u,%f,%f%c",&amplitude,&sharpen_type,&edge,&alpha,&end)==4 ||
               std::sscanf(argument,"%f,%u,%f,%f,%f%c",&amplitude,&sharpen_type,&edge,&alpha,&sigma,&end)==5) &&
              amplitude>=0 && edge>=0) {
            if (sharpen_type) print(images,"Sharpen image%s with shock filters and amplitude %g, edge %g, alpha %g and sigma %g.",
                                    gmic_inds,amplitude,edge,alpha,sigma);
            else print(images,"Sharpen image%s with inverse diffusion and amplitude %g.",gmic_inds,amplitude);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],sharpen(amplitude,sharpen_type?true:false,edge,alpha,sigma));
          } else error(images,"Sharpen image%s : Invalid argument '%s' "
                       "(should be 'amplitude>=0,_sharpen_type={0,1},_edge>=0,_alpha,_sigma').",
                       gmic_inds,argument_text);
          ++position; continue;
        }

        // Convolve.
        if (!cimg::strcmp("-convolve",command_name)) {
          int ind = no_ind; unsigned int borders = 1; char sep = 0, end = 0;
          if ((std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') ||
               std::sscanf(argument,"[%d],%u%c",&ind,&borders,&end)==2) {
            gmic_check_indice(ind,"Convolve image%s");
            print(images,"Convolve image%s with mask [%d].",gmic_inds,ind);
            const CImg<T> mask = images[ind];
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],convolve(mask,borders));
          } else error(images,"Convolve image%s : Invalid argument '%s' "
                       "(should be '[indice],_border_conditions={0,1}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Correlate.
        if (!cimg::strcmp("-correlate",command_name)) {
          int ind = no_ind; unsigned int borders = 1; char sep = 0, end = 0;
          if ((std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') ||
              std::sscanf(argument,"[%d],%u%c",&ind,&borders,&end)==2) {
            gmic_check_indice(ind,"Correlate image%s");
            print(images,"Correlate image%s with mask [%d].",gmic_inds,ind);
            const CImg<T> mask = images[ind];
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],correlate(mask,borders));
          } else error(images,"Correlate image%s : Invalid argument '%s' "
                       "(should be '[indice],_border_conditions={0,1}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Erode.
        if (!cimg::strcmp("-erode",command_name)) {
          int siz = 3, ind = no_ind; unsigned int borders = 1; char sep = 0, end = 0;
          if ((std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') ||
              std::sscanf(argument,"[%d],%u%c",&ind,&borders,&end)==2) {
            gmic_check_indice(ind,"Erode image%s");
            print(images,"Erode image%s with mask [%d].",gmic_inds,ind);
            const CImg<T> mask = images[ind];
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],erode(mask,borders));
          } else if ((std::sscanf(argument,"%d%c",&siz,&end)==1 ||
                      std::sscanf(argument,"%d,%u%c",&siz,&borders,&end)==2) &&
                     siz>=0) {
            print(images,"Erode image%s with a %dx%d square mask.",gmic_inds,siz);
            if (siz>0) cimg_foroff(indices,l) gmic_apply(images[indices[l]],erode(siz,borders));
          } else error(images,"Erode image%s : Invalid argument '%s' "
                       "(should be '[indice],_border_conditions={0,1}' or 'size>=0,_border_conditions={0,1}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Dilate.
        if (!cimg::strcmp("-dilate",command_name)) {
          int siz = 3, ind = no_ind; unsigned int borders = 1; char sep = 0, end = 0;
          if ((std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') ||
              std::sscanf(argument,"[%d],%u%c",&ind,&borders,&end)==2) {
            gmic_check_indice(ind,"Dilate image%s");
            print(images,"Dilate image%s with mask [%d].",gmic_inds,ind);
            const CImg<T> mask = images[ind];
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],dilate(mask,borders));
          } else if ((std::sscanf(argument,"%d%c",&siz,&end)==1 ||
                      std::sscanf(argument,"%d,%u%c",&siz,&borders,&end)==2) &&
                     siz>=0) {
            print(images,"Dilate image%s with a %dx%d square mask.",gmic_inds,siz);
            if (siz>0) cimg_foroff(indices,l) gmic_apply(images[indices[l]],dilate(siz,borders));
          } else error(images,"Dilate image%s : Invalid argument '%s' "
                       "(should be '[indice],_border_conditions={0,1}' or 'size>=0,_border_conditions={0,1}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Inpaint
        if (!cimg::strcmp("-inpaint",command_name)) {
          int ind = no_ind; char sep = 0, end = 0;
          if (std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') {
            gmic_check_indice(ind,"Inpaint image%s");
            print(images,"Inpaint image%s with mask [%d].",gmic_inds,ind);
            CImg<T> mask = images[ind];
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],inpaint(mask));
          } else error(images,"Inpaint image%s : Invalid argument '%s' "
                       "(should be '[indice]').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Compute gradient.
        if (!cimg::strcmp("-gradient",command_name)) {
          char axes[4096] = { 0 }, *naxes = 0, end = 0; int scheme = 3;
          print(images,"Compute gradient of image%s.",gmic_inds);
          if (std::sscanf(argument,"%4095[xyz]%c",axes,&end)==1 ||
              std::sscanf(argument,"%4095[xyz],%d%c",axes,&scheme,&end)==2) { naxes = axes; ++position; }
          unsigned int off = 0;
          cimg_foroff(indices,l) {
            const unsigned int ind = indices[l] + off;
            CImg<T>& img = images[ind];
            const CImg<char> filename = filenames[ind];
            const CImgList<T> gradient = img.get_gradient(naxes,scheme);
            if (get_version) {
              images.insert(gradient);
              filenames.insert(gradient.size,filename);
            } else {
              images.remove(ind); images.insert(gradient,ind);
              filenames.remove(ind); filenames.insert(gradient.size,filename,ind);
              off+=gradient.size-1;
            }
          }
          continue;
        }

        // Compute Hessian.
        if (!cimg::strcmp("-hessian",command_name)) {
          char axes[4096] = { 0 }, *naxes = 0, end = 0;
          print(images,"Compute Hessian of image%s.",gmic_inds);
          if (std::sscanf(argument,"%4095[xyz]%c",axes,&end)==1) { naxes = axes; ++position; }
          unsigned int off = 0;
          cimg_foroff(indices,l) {
            const unsigned int ind = indices[l] + off;
            CImg<T>& img = images[ind];
            const CImg<char> filename = filenames[ind];
            const CImgList<T> hessian = img.get_hessian(naxes);
            if (get_version) {
              images.insert(hessian);
              filenames.insert(hessian.size,filename);
            } else {
              images.remove(ind); images.insert(hessian,ind);
              filenames.remove(ind); filenames.insert(hessian.size,filename,ind);
              off+=hessian.size-1;
            }
          }
          continue;
        }

        // Compute direct or inverse FFT.
        const bool inv_fft = !cimg::strcmp("-ifft",command_name);
        if (!cimg::strcmp("-fft",command_name) || inv_fft) {
          print(images,"Compute %sFourier Transform of complex data",inv_fft?"inverse ":"");
          cimg_foroff(indices,l) {
            const unsigned int ind0 = indices[l], ind1 = l+1<_maxl?indices[l+1]:~0U;
            if (ind1!=~0U) {
              if (verbosity_level>=0) {
                std::fprintf(cimg_stdout," ([%u],[%u])%c",ind0,ind1,l==_maxl-1?'.':',');
                std::fflush(cimg_stdout);
              }
              CImgList<T> fft(images[ind0],images[ind1],!get_version);
              fft.FFT(inv_fft);
              if (get_version) {
                images.insert(2);
                fft[0].transfer_to(images[images.size-2]);
                fft[1].transfer_to(images[images.size-1]);
                filenames.insert(filenames[ind0]);
                filenames.insert(filenames[ind1]);
              } else {
                fft[0].transfer_to(images[ind0]);
                fft[1].transfer_to(images[ind1]);
              }
              ++l;
            } else {
              if (verbosity_level>=0) {
                std::fprintf(cimg_stdout," ([%u],0)",ind0);
                std::fflush(cimg_stdout);
              }
              CImgList<T> fft(images[ind0],!get_version);
              fft.insert(fft[0],~0U,false);
              fft[1].fill(0);
              fft.FFT(inv_fft);
              if (get_version) {
                images.insert(2);
                fft[0].transfer_to(images[images.size-2]);
                fft[1].transfer_to(images[images.size-1]);
                filenames.insert(2,filenames[ind0]);
              } else {
                fft[0].transfer_to(images[ind0]);
                images.insert(fft[1],1+ind0);
                filenames.insert(filenames[ind0],1+ind0);
              }
            }
          }
          continue;
        }

        //-----------------------------
        // Image creation and drawing
        //-----------------------------

        // Histogram.
        if (!cimg::strcmp("-histogram",command_name)) {
          int nb_levels = 256; char sep = 0, end = 0;
          if ((std::sscanf(argument,"%d%c",&nb_levels,&end)==1 ||
               (std::sscanf(argument,"%d%c%c",&nb_levels,&sep,&end)==2 && sep=='%')) &&
              nb_levels>0) {
            print(images,"Compute histogram of image%s, using %d%s levels.",gmic_inds,nb_levels,sep=='%'?"%":"");
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              int nnb_levels = nb_levels;
              if (sep=='%') { double m, M = img.maxmin(m); nnb_levels = (int)cimg::round(nb_levels*(1+M-m)/100,1); }
              gmic_apply(images[indices[l]],histogram(nnb_levels));
            }
          } else error(images,"Compute histogram of image%s : Invalid argument '%s' "
                       "(should be 'nb_levels[%%]>0').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Distance function.
        if (!cimg::strcmp("-distance",command_name)) {
          double value = 0; char sep = 0, end = 0;
          if (std::sscanf(argument,"%lf%c",&value,&end)==1 ||
              (std::sscanf(argument,"%lf%c%c",&value,&sep,&end)==2 && sep=='%')) {
            print(images,"Compute distance map of image%s to isovalue %g%s.",gmic_inds,value,sep=='%'?"%":"");
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              double isovalue = value;
              if (sep=='%') { double m, M = img.maxmin(m); isovalue = m + value*(M - m)/100; }
              gmic_apply(img,distance((T)isovalue));
            }
          } else error(images,"Compute distance function of image%s : Invalid argument '%s' "
                       "(should be 'value[%%]').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Apply Hamilton-Jacobi PDE to compute distance to 0.
        if (!cimg::strcmp("-hamilton",command_name)) {
          int nb_iter = 0; float band_size = 0; char end = 0;
          if ((std::sscanf(argument,"%d%c",&nb_iter,&end)==1 ||
               std::sscanf(argument,"%d,%f%c",&nb_iter,&band_size,&end)==2) &&
              nb_iter>=0 && band_size>=0) {
            print(images,"Apply %d iterations of Hamilton-Jacobi PDE on image%s.",nb_iter,gmic_inds);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],distance_hamilton((unsigned int)nb_iter,band_size));
          } else error(images,"Apply %d iterations of Hamilton-Jacobi PDE on image%s : Invalid argument '%s' "
                       "(should be 'nb_iter>=0,_band_size>=0').",nb_iter,gmic_inds,argument_text);
          ++position; continue;
        }

        // Label regions.
        gmic_simple_item("-label",label_regions,"Label regions on image%s.");

        // Displacement field.
        if (!cimg::strcmp("-displacement",command_name)) {
          float smooth = 0.1f, precision = 0.1f; int ind0 = no_ind, nbscales = 0, itermax = 1000; unsigned int backward = 1; char sep = 0, end = 0;
          if (((std::sscanf(argument,"[%d%c%c",&ind0,&sep,&end)==2 && sep==']') ||
               std::sscanf(argument,"[%d],%f%c",&ind0,&smooth,&end)==2 ||
               std::sscanf(argument,"[%d],%f,%f%c",&ind0,&smooth,&precision,&end)==3 ||
               std::sscanf(argument,"[%d],%f,%f,%d%c",&ind0,&smooth,&precision,&nbscales,&end)==4 ||
               std::sscanf(argument,"[%d],%f,%f,%d,%d%c",&ind0,&smooth,&precision,&nbscales,&itermax,&end)==5 ||
               std::sscanf(argument,"[%d],%f,%f,%d,%d,%u%c",&ind0,&smooth,&precision,&nbscales,&itermax,&backward,&end)==6) &&
              smooth>=0 && precision>0 && nbscales>=0 && itermax>=0) {
            gmic_check_indice(ind0,"Compute displacement field of image%s");
            print(images,"Compute displacement field of image%s with target [%u] and smoothness %g.",gmic_inds,ind0,smooth);
            const CImg<T> target = images[ind0];
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],displacement_field(target,smooth,precision,nbscales,itermax,backward?true:false));
          } else error(images,"Compute displacement field of image%s : Invalid argument '%s' "
                       "(should be '[indice],_smoothness>=0,_precision>0,_nbscales>=0,_itermax>=0,_backward={0,1}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Sort.
        gmic_simple_item("-sort",sort,"Sort values in image%s.");

        // PSNR.
        if (!cimg::strcmp("-psnr",command_name)) {
          double valmax = 255; char end = 0;
          if (std::sscanf(argument,"%lf%c",&valmax,&end)==1) ++position;
          CImgList<T> subimages;
          cimg_foroff(indices,l) subimages.insert(images[l],~0U,true);
          print(images,"Compute %ux%u matrix of PSNR values from image%s, with maximum pixel value %g.",
                subimages.size,subimages.size,gmic_inds,valmax);
          CImg<T> res(subimages.size,subimages.size,1,1,(T)-1);
          cimg_forXY(res,x,y) if (x>y) res(x,y) = res(y,x) = (T)subimages[x].PSNR(subimages[y],(float)valmax);
          if (get_version) { res.transfer_to(images); filenames.insert(CImg<char>("PSNR",5,1,1,1,false)); }
          else {
            if (indices) {
              cimg_foroff(indices,l) { const unsigned int ind = indices[l] - l; images.remove(ind); filenames.remove(ind); }
              images.insert(res,indices[0]);
              filenames.insert(CImg<char>("PSNR",5,1,1,1,false),indices[0]);
            }
          }
          continue;
        }

        // Draw point.
        if (!cimg::strcmp("-point",command_name)) {
          char argx[4096] = { 0 }, argy[4096] = { 0 }, argz[4096] = { 0 }, color[4096] = { 0 }, sepx = 0, sepy = 0, sepz = 0, end = 0;
          float x = 0, y = 0, z = 0, opacity = 1;
          if ((std::sscanf(argument,"%4095[0-9.eE%+-]%c",argx,&end)==1 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",argx,argy,&end)==2 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",argx,argy,argz,&end)==3 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f%c",argx,argy,argz,&opacity,&end)==4 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f,%4095[0-9.eE,+-]%c",argx,argy,argz,&opacity,color,&end)==5) &&
              (!*argx || std::sscanf(argx,"%f%c",&x,&end)==1 || (std::sscanf(argx,"%f%c%c",&x,&sepx,&end)==2 && sepx=='%')) &&
              (!*argy || std::sscanf(argy,"%f%c",&y,&end)==1 || (std::sscanf(argy,"%f%c%c",&y,&sepy,&end)==2 && sepy=='%')) &&
              (!*argz || std::sscanf(argz,"%f%c",&z,&end)==1 || (std::sscanf(argz,"%f%c%c",&z,&sepz,&end)==2 && sepz=='%'))) {
            print(images,"Draw point (%g%s,%g%s,%g%s) with opacity %g and color '%s' on image%s.",
                  x,sepx=='%'?"%":"",y,sepy=='%'?"%":"",z,sepz=='%'?"%":"",opacity,color[0]?color:"default",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]], col(img.dimv(),1,1,1,0);
              col.fill(color,true);
              const int
                nx = (int)cimg::round(sepx=='%'?x*(img.dimx()-1)/100:x,1),
                ny = (int)cimg::round(sepy=='%'?y*(img.dimy()-1)/100:y,1),
                nz = (int)cimg::round(sepz=='%'?z*(img.dimz()-1)/100:z,1);
              gmic_apply(img,draw_point(nx,ny,nz,col,opacity));
            }
          } else error(images,"Draw point on image%s : Invalid argument '%s' "
                       "(should be 'x[%%],y[%%],_z[%%],_opacity,_color)",gmic_inds,argument_text);
          ++position; continue;
        }

        // Draw line.
        if (!cimg::strcmp("-line",command_name)) {
          char argx0[4096] = { 0 }, argy0[4096] = { 0 }, argx1[4096] = { 0 }, argy1[4096] = { 0 }, color[4096] = { 0 };
          char sepx0 = 0, sepy0 = 0, sepx1 = 0, sepy1 = 0, end = 0;
          float x0 = 0, y0 = 0, x1 = 0, y1 = 0, opacity = 1;
          if ((std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",argx0,argy0,argx1,argy1,&end)==4 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f%c",argx0,argy0,argx1,argy1,&opacity,&end)==5 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f,%4095[0-9.eE,+-]%c",
                           argx0,argy0,argx1,argy1,&opacity,color,&end)==6) &&
              (std::sscanf(argx0,"%f%c",&x0,&end)==1 || (std::sscanf(argx0,"%f%c%c",&x0,&sepx0,&end)==2 && sepx0=='%')) &&
              (std::sscanf(argy0,"%f%c",&y0,&end)==1 || (std::sscanf(argy0,"%f%c%c",&y0,&sepy0,&end)==2 && sepy0=='%')) &&
              (std::sscanf(argx1,"%f%c",&x1,&end)==1 || (std::sscanf(argx1,"%f%c%c",&x1,&sepx1,&end)==2 && sepx1=='%')) &&
              (std::sscanf(argy1,"%f%c",&y1,&end)==1 || (std::sscanf(argy1,"%f%c%c",&y1,&sepy1,&end)==2 && sepy1=='%'))) {
            print(images,"Draw line (%g%s,%g%s) - (%g%s,%g%s) with oapcity %g and color '%s' on image%s.",
                  x0,sepx0=='%'?"%":"",y0,sepy0=='%'?"%":"",x1,sepx1=='%'?"%":"",y1,sepy1=='%'?"%":"",opacity,color[0]?color:"default",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]], col(img.dimv(),1,1,1,0);
              col.fill(color,true);
              const int
                nx0 = (int)cimg::round(sepx0=='%'?x0*(img.dimx()-1)/100:x0,1),
                ny0 = (int)cimg::round(sepy0=='%'?y0*(img.dimy()-1)/100:y0,1),
                nx1 = (int)cimg::round(sepx1=='%'?x1*(img.dimx()-1)/100:x1,1),
                ny1 = (int)cimg::round(sepy1=='%'?y1*(img.dimy()-1)/100:y1,1);
              gmic_apply(img,draw_line(nx0,ny0,nx1,ny1,col,opacity));
            }
          } else error(images,"Draw line on image%s : Invalid argument '%s' "
                       "(should be 'x0[%%],y0[%%],x1[%%],y1[%%],_opacity,_color')",gmic_inds,argument_text);
          ++position; continue;
        }

        // Draw polygon.
        if (!cimg::strcmp("-polygon",command_name)) {
          char arg0[4096] = { 0 }, arg1[4096] = { 0 }, tmp[4096] = { 0 }, sepx0 = 0, sepy0 = 0, end = 0;
          int N = 0; float x0 = 0, y0 = 0, opacity = 1;
          if (std::sscanf(argument,"%d%c",&N,&end)==2 && N>2) {
            const char
              *nargument = argument + std::sprintf(tmp,"%d",N) + 1,
              *const eargument = argument + std::strlen(argument);
            CImg<float> coords0(N,2,1,1,0);
            CImg<bool> percents(N,2,1,1,0);
            for (int n = 0; n<N; ++n) if (nargument<eargument) {
              if (std::sscanf(nargument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-]",arg0,arg1)==2 &&
                  (std::sscanf(arg0,"%f%c",&x0,&end)==1 || (std::sscanf(arg0,"%f%c%c",&x0,&(sepx0=0),&end)==2 && sepx0=='%')) &&
                  (std::sscanf(arg1,"%f%c",&y0,&end)==1 || (std::sscanf(arg1,"%f%c%c",&y0,&(sepy0=0),&end)==2 && sepy0=='%'))) {
                coords0(n,0) = x0; percents(n,0) = (sepx0=='%');
                coords0(n,1) = y0; percents(n,1) = (sepy0=='%');
                nargument+=std::strlen(arg0) + std::strlen(arg1) + 2;
              } else error(images,"Draw polygon on image%s : Invalid or incomplete argument '%s' "
                           "(should be 'N>=3,x1[%%],y1[%%],..,xN[%%],yN[%%],_opacity,_color')",
                           gmic_inds,argument_text);
            } else error(images,"Draw polygon on image%s : Incomplete argument '%s' "
                         "(should be 'N>=3,x1[%%],y1[%%],..,xN[%%],yN[%%],_opacity,_color')",
                         gmic_inds,argument_text,N);
            if (nargument<eargument && std::sscanf(nargument,"%4095[0-9.eE+-]",arg0)==1 &&
                std::sscanf(arg0,"%f",&opacity)==1) nargument+=std::strlen(arg0)+1;
            const char *const color = nargument<eargument?nargument:&(end=0);
            print(images,"Draw %d-vertices polygon with opacity %g and color '%s' on image%s.",
                  N,opacity,color[0]?color:"default",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              CImg<int> coords(coords0);
              cimg_forX(coords,p) {
                if (percents(p,0)) coords(p,0) = (int)cimg::round(coords0(p,0)*(img.dimx()-1)/100,1);
                if (percents(p,1)) coords(p,1) = (int)cimg::round(coords0(p,1)*(img.dimy()-1)/100,1);
              }
              CImg<T> col(img.dimv(),1,1,1,0);
              col.fill(color,true);
              gmic_apply(img,draw_polygon(coords,col,opacity));
            }
          } else error(images,"Draw polygon on image%s : Invalid argument '%s' "
                       "(should be 'N>=3,x1[%%],y1[%%],..,xN[%%],yN[%%],_opacity,_color')",
                       gmic_inds,argument_text);
          ++position; continue;
        }

        // Draw ellipse.
        if (!cimg::strcmp("-ellipse",command_name)) {
          char argx[256] = { 0 }, argy[256] = { 0 }, argR[256] = { 0 }, argr[256] = { 0 }, color[4096] = { 0 };
          char sepx = 0, sepy = 0, sepR = 0, sepr = 0, end = 0;
          float x = 0, y = 0, R = 0, r = 0, angle = 0, opacity = 1;
          if ((std::sscanf(argument,"%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-]%c",argx,argy,argR,&end)==3 ||
               std::sscanf(argument,"%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-]%c",argx,argy,argR,argr,&end)==4 ||
               std::sscanf(argument,"%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-],%f%c",argx,argy,argR,argr,&angle,&end)==5 ||
               std::sscanf(argument,"%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-],%f,%f%c",argx,argy,argR,argr,&angle,&opacity,&end)==6 ||
               std::sscanf(argument,"%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-],%255[0-9.eE%+-],%f,%f,%4095[0-9.eE,+-]%c",
                           argx,argy,argR,argr,&angle,&opacity,color,&end)==7) &&
              (std::sscanf(argx,"%f%c",&x,&end)==1 || (std::sscanf(argx,"%f%c%c",&x,&sepx,&end)==2 && sepx=='%')) &&
              (std::sscanf(argy,"%f%c",&y,&end)==1 || (std::sscanf(argy,"%f%c%c",&y,&sepy,&end)==2 && sepy=='%')) &&
              (std::sscanf(argR,"%f%c",&R,&end)==1 || (std::sscanf(argR,"%f%c%c",&R,&sepR,&end)==2 && sepR=='%')) &&
              (!*argr || std::sscanf(argr,"%f%c",&r,&end)==1 || (std::sscanf(argr,"%f%c%c",&r,&sepr,&end)==2 && sepr=='%'))) {
            if (!*argr) r = R;
            print(images,"Draw ellipse centered at (%g%s,%g%s) with radii (%g%s,%g%s), orientation %g deg, opacity %g and color '%s' on image%s.",
                  x,sepx=='%'?"%":"",y,sepy=='%'?"%":"",R,sepR=='%'?"%":"",r,sepr=='%'?"%":"",angle,opacity,color[0]?color:"default",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]], col(img.dimv(),1,1,1,0);
              col.fill(color,true);
              const float rmax = std::sqrt((float)cimg::sqr(img.dimx()) + cimg::sqr(img.dimy()))/2;
              const int
                nx = (int)cimg::round(sepx=='%'?x*(img.dimx()-1)/100:x,1),
                ny = (int)cimg::round(sepy=='%'?y*(img.dimy()-1)/100:y,1);
                          const float
                nR = (float)cimg::round(sepR=='%'?R*rmax/100:R,1),
                nr = (float)cimg::round(sepr=='%'?r*rmax/100:r,1);
              gmic_apply(img,draw_ellipse(nx,ny,nR,nr,angle,col,opacity));
            }
          } else error(images,"Draw ellipse on image%s : Invalid argument '%s' "
                       "(should be 'x[%%],y[%%],r[%%],_R[%%],_theta,_opacity,_color)",
                       gmic_inds,argument_text);
          ++position; continue;
        }

        // Draw text.
        if (!cimg::strcmp("-text",command_name)) {
          char argx[4096] = { 0 }, argy[4096] = { 0 }, color[4096] = { 0 }, text[4096] = { 0 }, sepx = 0, sepy = 0, end = 0;
          float x = 0, y = 0, opacity = 1; int siz = 11;
          if ((std::sscanf(argument,"%4095[^,]%c",text,&end)==1 ||
               std::sscanf(argument,"%4095[^,],%4095[0-9.eE%+-]%c",text,argx,&end)==2 ||
               std::sscanf(argument,"%4095[^,],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",text,argx,argy,&end)==3 ||
               std::sscanf(argument,"%4095[^,],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%d%c",text,argx,argy,&siz,&end)==4 ||
               std::sscanf(argument,"%4095[^,],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%d,%f%c",text,argx,argy,&siz,&opacity,&end)==5 ||
               std::sscanf(argument,"%4095[^,],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%d,%f,%4095[0-9.eE,+-]%c",text,argx,argy,&siz,&opacity,color,&end)==6) &&
              (!*argx || std::sscanf(argx,"%f%c",&x,&end)==1 || (std::sscanf(argx,"%f%c%c",&x,&sepx,&end)==2 && sepx=='%')) &&
              (!*argy || std::sscanf(argy,"%f%c",&y,&end)==1 || (std::sscanf(argy,"%f%c%c",&y,&sepy,&end)==2 && sepy=='%')) &&
              siz>0) {
            cimg::strclean(text); cimg::strescape(text);
            print(images,"Draw text \"%s\" at position (%g%s,%g%s) with font size %d, opacity %g and color '%s' on image%s.",
                  text,x,sepx=='%'?"%":"",y,sepy=='%'?"%":"",siz,opacity,color[0]?color:"default",gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]], col(img.dimv(),1,1,1,0);
              col.fill(color,true);
              const int
                nx = (int)cimg::round(sepx=='%'?x*(img.dimx()-1)/100:x,1),
                ny = (int)cimg::round(sepy=='%'?y*(img.dimy()-1)/100:y,1);
              gmic_apply(img,draw_text(nx,ny,text,col.ptr(),0,opacity,siz));
            }
          } else error(images,"Draw text on image%s : Invalid argument '%s' "
                       "(should be 'text,_x[%%],_y[%%],_size>0,_opacity,_color').",
                       gmic_inds,argument_text);
          ++position; continue;
        }

        // Draw image.
        if (!cimg::strcmp("-image",command_name)) {
          char argx[4096] = { 0 }, argy[4096] = { 0 }, argz[4096] = { 0 }, sep = 0, sepx = 0, sepy = 0, sepz = 0, end = 0;
          int ind = no_ind, indm = no_ind; float x = 0, y = 0, z = 0, opacity = 1;
          if (((std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==1 && sep==']') ||
               std::sscanf(argument,"[%d],%4095[0-9.eE%+-]%c",&ind,argx,&end)==2 ||
               std::sscanf(argument,"[%d],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",&ind,argx,argy,&end)==3 ||
               std::sscanf(argument,"[%d],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",&ind,argx,argy,argz,&end)==4 ||
               std::sscanf(argument,"[%d],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f%c",&ind,argx,argy,argz,&opacity,&end)==5 ||
               (std::sscanf(argument,"[%d],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f,[%d%c%c",
                            &ind,argx,argy,argz,&opacity,&indm,&sep,&end)==7 && sep==']')) &&
              (!*argx || std::sscanf(argx,"%f%c",&x,&end)==1 || (std::sscanf(argx,"%f%c%c",&x,&sepx,&end)==2 && sepx=='%')) &&
              (!*argy || std::sscanf(argy,"%f%c",&y,&end)==1 || (std::sscanf(argy,"%f%c%c",&y,&sepy,&end)==2 && sepy=='%')) &&
              (!*argz || std::sscanf(argz,"%f%c",&z,&end)==1 || (std::sscanf(argz,"%f%c%c",&z,&sepz,&end)==2 && sepz=='%'))) {
            gmic_check_indice(ind,"Draw image on image%s");
            const CImg<T> sprite = images[ind];
            CImg<T> mask;
            if (indm!=no_ind) {
              gmic_check_indice(indm,"Draw image on image%s");
              mask = images[indm];
              print(images,"Draw image [%d] at (%g%s,%g%s,%g%s), with opacity %g and mask [%d] on image%s.",
                    ind,x,sepx=='%'?"%":"",y,sepy=='%'?"%":"",z,sepz=='%'?"%":"",opacity,indm,gmic_inds);
            } else print(images,"Draw image [%d] at (%g%s,%g%s,%g%s) with opacity %g on image%s.",
                         ind,x,sepx=='%'?"%":"",y,sepy=='%'?"%":"",z,sepz=='%'?"%":"",opacity,gmic_inds);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const int
                nx = (int)cimg::round(sepx=='%'?x*(img.dimx()-1)/100:x,1),
                ny = (int)cimg::round(sepy=='%'?y*(img.dimy()-1)/100:y,1),
                nz = (int)cimg::round(sepz=='%'?z*(img.dimz()-1)/100:z,1);
              if (indm!=no_ind) { gmic_apply(img,draw_image(nx,ny,nz,sprite,mask,opacity)); }
              else { gmic_apply(img,draw_image(nx,ny,nz,sprite,opacity)); }
            }
          } else error(images,"Draw image on image%s : Invalid argument '%s' "
                       "(should be '[indice],_x[%%],_y[%%],_z[%%],_opacity,_[indice_mask]').",
                       gmic_inds,argument_text);
          ++position; continue;
        }

        // Draw 3D object.
        if (!cimg::strcmp("-object3d",command_name)) {
          char argx[4096] = { 0 }, argy[4096] = { 0 }, sep = 0, sepx = 0, sepy = 0, end = 0;
          float x = 0, y = 0, z = 0, opacity = 1; int ind = no_ind;
          if (((std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']') ||
               std::sscanf(argument,"[%d],%4095[0-9.eE%+-]%c",&ind,argx,&end)==2 ||
               std::sscanf(argument,"[%d],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",&ind,argx,argy,&end)==3 ||
               std::sscanf(argument,"[%d],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f%c",&ind,argx,argy,&z,&end)==4 ||
               std::sscanf(argument,"[%d],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f,%f%c",&ind,argx,argy,&z,&opacity,&end)==5) &&
              (!*argx || std::sscanf(argx,"%f%c",&x,&end)==1 || (std::sscanf(argx,"%f%c%c",&x,&sepx,&end)==2 && sepx=='%')) &&
              (!*argy || std::sscanf(argy,"%f%c",&y,&end)==1 || (std::sscanf(argy,"%f%c%c",&y,&sepy,&end)==2 && sepy=='%'))) {
            gmic_check_indice(ind,"Draw 3D object on image%s");
            if (!images[ind].is_CImg3d())
              error(images,"Draw 3D object on image%s : Image [%d] is not a 3D object.",gmic_inds,ind);
            print(images,"Draw 3D object [%d] at (%g%s,%g%s,%g) with opacity %g on image%s.",
                  ind,x,sepx=='%'?"%":"",y,sepy=='%'?"%":"",z,opacity,gmic_inds);
            CImgList<unsigned int> primitives3d;
            CImgList<unsigned char> colors3d;
            CImg<float> opacities3d, points3d(images[ind]);
            points3d.CImg3dtoobject3d(primitives3d,colors3d,opacities3d);
            opacities3d*=opacity;
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]];
              const float
                nx = (float)cimg::round(sepx=='%'?x*(img.dimx()-1)/100:x,1),
                ny = (float)cimg::round(sepy=='%'?y*(img.dimy()-1)/100:y,1);
              gmic_apply(img,draw_object3d(nx,ny,z,points3d,primitives3d,colors3d,opacities3d,
                                           render3d,!is_oriented3d,focale3d,light3d_x,light3d_y,light3d_z,specular_light3d,specular_shine3d));
            }
          } else error(images,"Draw 3D object on image%s : Invalid argument '%s' "
                       "(should be '[indice],_x[%%],_y[%%],_z,_opacity').",
                       gmic_inds,argument_text);
          ++position; continue;
        }

        // Draw plasma fractal.
        if (!cimg::strcmp("-plasma",command_name)) {
          float alpha = 1, beta = 1, opacity = 1; char end = 0;
          if (std::sscanf(argument,"%f%c",&alpha,&end)==1 ||
              std::sscanf(argument,"%f,%f%c",&alpha,&beta,&end)==2 ||
              std::sscanf(argument,"%f,%f,%f%c",&alpha,&beta,&opacity,&end)==3) {
            print(images,"Draw plasma in image%s with alpha %g, beta %g and opacity %g.",gmic_inds,alpha,beta,opacity);
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],draw_plasma(alpha,beta,opacity));
          } else error(images,"Draw plasma in image%s : Invalid argument '%s' "
                       "(should be 'alpha,_beta,_opacity').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Draw Mandelbrot/Julia fractal.
        if (!cimg::strcmp("-mandelbrot",command_name)) {
          double z0r = -2, z0i = -2, z1r = 2, z1i = 2, paramr = 0, parami = 0; char end = 0;
          float opacity = 1; int itermax = 100; unsigned int julia = 0;
          if ((std::sscanf(argument,"%lf,%lf,%lf,%lf%c",&z0r,&z0i,&z1r,&z1i,&end)==4 ||
               std::sscanf(argument,"%lf,%lf,%lf,%lf,%d%c",&z0r,&z0i,&z1r,&z1i,&itermax,&end)==5 ||
               std::sscanf(argument,"%lf,%lf,%lf,%lf,%d,%u,%lf,%lf%c",&z0r,&z0i,&z1r,&z1i,&itermax,&julia,&paramr,&parami,&end)==8 ||
               std::sscanf(argument,"%lf,%lf,%lf,%lf,%d,%u,%lf,%lf,%f%c",&z0r,&z0i,&z1r,&z1i,&itermax,&julia,&paramr,&parami,&opacity,&end)==9) &&
              itermax>=0) {
            print(images,"Draw %s fractal in image%s from complex area (%g,%g)-(%g,%g) with c0 = (%g,%g) (%d iterations).",
                  julia?"Julia":"Mandelbrot",gmic_inds,z0r,z0i,z1r,z1i,paramr,parami,itermax);
            cimg_foroff(indices,l)
              gmic_apply(images[indices[l]],draw_mandelbrot(CImg<T>(),opacity,z0r,z0i,z1r,z1i,itermax,true,
                                                            julia?true:false,paramr,parami));
          } else error(images,"Draw fractal in image%s : Invalid argument '%s' "
                       "(should be 'z0r,z0i,z1r,z1i,_itermax>0,_julia={0,1},_c0r,_c0i,_opacity').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Draw quiver.
        if (!cimg::strcmp("-quiver",command_name)) {
          int ind = no_ind, sampling = 25; unsigned int arrows = 1; float factor = -20, opacity = 1; char color[4096] = { 0 }, end = 0;
          if ((std::sscanf(argument,"[%d]%c",&ind,&end)==1 ||
               std::sscanf(argument,"[%d],%d%c",&ind,&sampling,&end)==2 ||
               std::sscanf(argument,"[%d],%d,%f%c",&ind,&sampling,&factor,&end)==3 ||
               std::sscanf(argument,"[%d],%d,%f,%u%c",&ind,&sampling,&factor,&arrows,&end)==4 ||
               std::sscanf(argument,"[%d],%d,%f,%u,%f%c",&ind,&sampling,&factor,&arrows,&opacity,&end)==5 ||
               std::sscanf(argument,"[%d],%d,%f,%u,%f,%4095[0-9.eE,+-]%c",&ind,&sampling,&factor,&arrows,&opacity,color,&end)==6) &&
              sampling>0) {
            gmic_check_indice(ind,"Draw 2D vector field on image%s");
            print(images,"Draw 2D vector field on image%s with sampling %d, factor %g, opacity %g and color '%s'.",
                  gmic_inds,sampling,factor,opacity,color);
            const CImg<T> flow = images[ind];
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]], col(img.dimv(),1,1,1,0);
              col.fill(color,true);
              gmic_apply(img,draw_quiver(flow,col,opacity,sampling,factor,arrows?true:false));
            }
            ++position; continue;
          } else error(images,"Draw 2D vector field on image%s : Invalid argument '%s' "
                       "(should be '[ind],_sampling>0,_factor,_type={0,1},_opacity,_color').",gmic_inds,argument_text);
        }

        // Flood fill.
        if (!cimg::strcmp("-flood",command_name)) {
          char argx[4096] = { 0 }, argy[4096] = { 0 }, argz[4096] = { 0 }, color[4096] = { 0 }, sepx = 0, sepy = 0, sepz = 0, end = 0;
          float x = 0, y = 0, z = 0, tolerance = 0, opacity = 1;
          if ((std::sscanf(argument,"%4095[0-9.eE%+-]%c",argx,&end)==1 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",argx,argy,&end)==2 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-]%c",argx,argy,argz,&end)==3 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f%c",argx,argy,argz,&tolerance,&end)==4 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f,%f%c",argx,argy,argz,&tolerance,&opacity,&end)==5 ||
               std::sscanf(argument,"%4095[0-9.eE%+-],%4095[0-9.eE%+-],%4095[0-9.eE%+-],%f,%f,%4095[0-9.eE,+-]%c",
                           argx,argy,argz,&tolerance,&opacity,color,&end)==6) &&
              (!*argx || std::sscanf(argx,"%f%c",&x,&end)==1 || (std::sscanf(argx,"%f%c%c",&x,&sepx,&end)==2 && sepx=='%')) &&
              (!*argy || std::sscanf(argy,"%f%c",&y,&end)==1 || (std::sscanf(argy,"%f%c%c",&y,&sepy,&end)==2 && sepy=='%')) &&
              (!*argz || std::sscanf(argz,"%f%c",&z,&end)==1 || (std::sscanf(argz,"%f%c%c",&z,&sepz,&end)==2 && sepz=='%')) &&
              tolerance>=0) {
            print(images,"Flood fill image%s from (%g%s,%g%s,%g%s) with tolerance %g, opacity %g and color '%s'.",
                  gmic_inds,x,sepx=='%'?"%":"",y,sepy=='%'?"%":"",z,sepz=='%'?"%":"",tolerance,opacity,color);
            cimg_foroff(indices,l) {
              CImg<T> &img = images[indices[l]], col(img.dimv(),1,1,1,0);
              col.fill(color,true);
              const int
                nx = (int)cimg::round(sepx=='%'?x*(img.dimx()-1)/100:x,1),
                ny = (int)cimg::round(sepy=='%'?y*(img.dimy()-1)/100:y,1),
                nz = (int)cimg::round(sepz=='%'?z*(img.dimz()-1)/100:z,1);
              gmic_apply(img,draw_fill(nx,ny,nz,col,opacity,tolerance));
            }
          } else error(images,"Flood fill image%s : Invalid argument '%s' "
                       "(should be 'x,_y,_z,_tolerance>=0,_opacity,_color').",gmic_inds,argument_text);
          ++position; continue;
        }

        //-------------------------
        // Image list manipulation
        //-------------------------

        // Remove specified image(s).
        if (!cimg::strcmp("-remove",command_name) || !cimg::strcmp("-rm",command_name)) {
          print(images,"Remove image%s",gmic_inds);
          unsigned int off = 0;
          cimg_foroff(indices,l) {
            const unsigned int ind = indices[l] - off;
            images.remove(ind); filenames.remove(ind);
            ++off;
          }
          if (verbosity_level>=0) {
            std::fprintf(cimg_stdout," (%u image%s left).",images.size,images.size==1?"":"s");
            std::fflush(cimg_stdout);
          }
          continue;
        }

        // Keep specified image(s).
        if (!cimg::strcmp("-keep",command_name) || !cimg::strcmp("-k",command_name)) {
          print(images,"Keep image%s",gmic_inds);
          CImgList<T> nimages(indices.size());
          CImgList<char> nfilenames(indices.size());
          cimg_foroff(indices,l) {
            nimages[l].swap(images[indices[l]]);
            nfilenames[l].swap(filenames[indices[l]]);
          }
          nimages.transfer_to(images);
          nfilenames.transfer_to(filenames);
          if (verbosity_level>=0) {
            std::fprintf(cimg_stdout," (%u image%s left).",images.size,images.size==1?"":"s");
            std::fflush(cimg_stdout);
          }
          continue;
        }

        // Move image(s) to specified position.
        if (!cimg::strcmp("-move",command_name) || !cimg::strcmp("-mv",command_name)) {
          float number = 0; int ind0 = no_ind; char end = 0;
          if (std::sscanf(argument,"%f%c",&number,&end)==1) ind0 = (int)number;
          else ind0 = (int)last_image.eval(argument);
          if (ind0<0) ind0+=images.size;
          if (ind0<0) ind0 = 0;
          if (ind0>(int)images.size) ind0 = images.size;
          print(images,"Move image%s to position %d.",gmic_inds,ind0);
          CImgList<T> nimages;
          CImgList<char> nfilenames;
          cimg_foroff(indices,l) {
            const unsigned int ind = indices[l];
            images[ind].transfer_to(nimages);
            filenames[ind].transfer_to(nfilenames);
          }
          images.insert(nimages,ind0); filenames.insert(nfilenames,ind0);
          { cimglist_for(images,l) if (!images[l]) { images.remove(l); filenames.remove(l--); }}
          ++position; continue;
        }

        // Reverse images order.
        if (!cimg::strcmp("-reverse",command_name)) {
          print(images,"Reverse positions of image%s.",gmic_inds);
          CImgList<T> nimages(indices.size());
          CImgList<char> nfilenames(indices.size());
          cimg_foroff(indices,l) { nimages[l].swap(images[indices[l]]); nfilenames[l].swap(filenames[indices[l]]); }
          nimages.reverse(); nfilenames.reverse();
          { cimg_foroff(indices,l) { nimages[l].swap(images[indices[l]]); nfilenames[l].swap(filenames[indices[l]]); }}
          continue;
        }

        // Set image name.
        if (!cimg::strcmp("-name",command_name)) {
          char text[4096] = { 0 };
          std::strcpy(text,argument);
          cimg::strclean(text);
          cimg_foroff(indices,l) filenames[indices[l]].assign(text,std::strlen(text)+1,1,1,1,false);
          ++position; continue;
        }

        //-------------------------
        // 3D objects manipulation
        //-------------------------

        // Generate 3D cube.
        if (!cimg::strcmp("-cube3d",item)) {
          float size = 100; char end = 0;
          if (std::sscanf(argument,"%f%c",&size,&end)==1) {
            print(images,"Generate 3D cube with size %g.",size);
            CImgList<unsigned int> primitives3d;
            CImg<float> points3d = CImg<T>::cube3d(primitives3d,size);
            CImgList<unsigned char> colors3d(primitives3d.size,1,3,1,1,200);
            CImg<float> opacities3d(1,primitives3d.size,1,1,1);
            points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
            images.insert(points3d);
            filenames.insert(CImg<char>("(gmic)",7,1,1,1,false));
          } else error(images,"Generate 3D cube : Invalid argument '%s' "
                       "(should be 'size').",argument_text);
          ++position; continue;
        }

        // Generate 3D cone.
        if (!cimg::strcmp("-cone3d",item)) {
          float radius = 100, height = 200; char end = 0; int subdivisions = 24;
          if ((std::sscanf(argument,"%f%c",&radius,&end)==1 ||
               std::sscanf(argument,"%f,%f%c",&radius,&height,&end)==2 ||
               std::sscanf(argument,"%f,%f,%d%c",&radius,&height,&subdivisions,&end)==3) &&
              subdivisions>0) {
            print(images,"Generate 3D cone with radius %g, height %g and %d subdivisions.",radius,height,subdivisions);
            CImgList<unsigned int> primitives3d;
            CImg<float> points3d = CImg<T>::cone3d(primitives3d,radius,height,subdivisions);
            CImgList<unsigned char> colors3d(primitives3d.size,1,3,1,1,200);
            CImg<float> opacities3d(1,primitives3d.size,1,1,1);
            points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
            images.insert(points3d);
            filenames.insert(CImg<char>("(gmic)",7,1,1,1,false));
          } else error(images,"Generate 3D cone : Invalid argument '%s' "
                       "(should be 'radius,_height,_subdivisions>0').",argument_text);
          ++position; continue;
        }

        // Generate 3D cylinder.
        if (!cimg::strcmp("-cylinder3d",item)) {
          float radius = 100, height = 200; char end = 0; int subdivisions = 24;
          if ((std::sscanf(argument,"%f%c",&radius,&end)==1 ||
               std::sscanf(argument,"%f,%f%c",&radius,&height,&end)==2 ||
               std::sscanf(argument,"%f,%f,%d%c",&radius,&height,&subdivisions,&end)==3) &&
              subdivisions>0) {
            print(images,"Generate 3D cylinder with radius %g, height %g and %d subdivisions.",radius,height,subdivisions);
            CImgList<unsigned int> primitives3d;
            CImg<float> points3d = CImg<T>::cylinder3d(primitives3d,radius,height,subdivisions);
            CImgList<unsigned char> colors3d(primitives3d.size,1,3,1,1,200);
            CImg<float> opacities3d(1,primitives3d.size,1,1,1);
            points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
            images.insert(points3d);
            filenames.insert(CImg<char>("(gmic)",7,1,1,1,false));
          } else error(images,"Generate 3D cylinder : Invalid argument '%s' "
                       "(should be 'radius,_height,_subdivisions>0').",argument_text);
          ++position; continue;
        }

        // Generate 3D torus.
        if (!cimg::strcmp("-torus3d",item)) {
          float radius1 = 100, radius2 = 30; char end = 0; int subdivisions1 = 24, subdivisions2 = 12;
          if ((std::sscanf(argument,"%f%c",&radius1,&end)==1 ||
               std::sscanf(argument,"%f,%f%c",&radius1,&radius2,&end)==2 ||
               std::sscanf(argument,"%f,%f,%d,%c",&radius1,&radius2,&subdivisions1,&end)==3 ||
               std::sscanf(argument,"%f,%f,%d,%d%c",&radius1,&radius2,&subdivisions1,&subdivisions2,&end)==4) &&
              subdivisions1>0 && subdivisions2>0) {
            print(images,"Generate 3D torus with radii %g and %g, and subdivisions %d and %d.",radius1,radius2,subdivisions1,subdivisions2);
            CImgList<unsigned int> primitives3d;
            CImg<float> points3d = CImg<T>::torus3d(primitives3d,radius1,radius2,subdivisions1,subdivisions2);
            CImgList<unsigned char> colors3d(primitives3d.size,1,3,1,1,200);
            CImg<float> opacities3d(1,primitives3d.size,1,1,1);
            points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
            images.insert(points3d);
            filenames.insert(CImg<char>("(gmic)",7,1,1,1,false));
          } else error(images,"Generate 3D torus : Invalid argument '%s' "
                       "(should be 'radius1,_radius2,_subdivisions1>0,_subdivisions2>0').",argument_text);
          ++position; continue;
        }

        // Generate 3D plane.
        if (!cimg::strcmp("-plane3d",item)) {
          float sizex = 100, sizey = 30; char end = 0; int subdivisionsx = 24, subdivisionsy = 12;
          if ((std::sscanf(argument,"%f%c",&sizex,&end)==1 ||
               std::sscanf(argument,"%f,%f%c",&sizex,&sizey,&end)==2 ||
               std::sscanf(argument,"%f,%f,%d%c",&sizex,&sizey,&subdivisionsx,&end)==3 ||
               std::sscanf(argument,"%f,%f,%d,%d%c",&sizex,&sizey,&subdivisionsx,&subdivisionsy,&end)==4) &&
              subdivisionsx>0 && subdivisionsy>0) {
            print(images,"Generate 3D plane with dimensions %g and %g, and subdivisions %d and %d.",sizex,sizey,subdivisionsx,subdivisionsy);
            CImgList<unsigned int> primitives3d;
            CImg<float> points3d = CImg<T>::plane3d(primitives3d,sizex,sizey,subdivisionsx,subdivisionsy);
            CImgList<unsigned char> colors3d(primitives3d.size,1,3,1,1,200);
            CImg<float> opacities3d(1,primitives3d.size,1,1,1);
            points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
            images.insert(points3d);
            filenames.insert(CImg<char>("(gmic)",7,1,1,1,false));
          } else error(images,"Generate 3D plane : Invalid argument '%s' "
                       "(should be 'sizex,_sizey,_subdivisionsx>0,_subdivisionsy>0').",argument_text);
          ++position; continue;
        }

        // Generate 3D sphere.
        if (!cimg::strcmp("-sphere3d",item)) {
          float radius = 100; char end = 0; int recursions = 3;
          if ((std::sscanf(argument,"%f%c",&radius,&end)==1 ||
               std::sscanf(argument,"%f,%d%c",&radius,&recursions,&end)==2) &&
              recursions>=0) {
            print(images,"Generate 3D sphere with radius %g and %d recursions.",radius,recursions);
            CImgList<unsigned int> primitives3d;
            CImg<float> points3d = CImg<T>::sphere3d(primitives3d,radius,recursions);
            CImgList<unsigned char> colors3d(primitives3d.size,1,3,1,1,200);
            CImg<float> opacities3d(1,primitives3d.size,1,1,1);
            points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
            images.insert(points3d);
            filenames.insert(CImg<char>("(gmic)",7,1,1,1,false));
          } else error(images,"Generate 3D sphere : Invalid argument '%s' "
                       "(should be 'radius,_recursions>=0').",argument_text);
          ++position; continue;
        }

        // Build 3D elevation.
        if (!cimg::strcmp("-elevation3d",command_name)) {
          float zfact = 0.2f; char end = 0, sep = 0; int ind = no_ind;
          if (std::sscanf(argument,"%f%c",&zfact,&end)==1 ||
              (std::sscanf(argument,"[%d%c%c",&ind,&sep,&end)==2 && sep==']')) {
            CImg<typename CImg<T>::Tfloat> elev;
            if (ind!=no_ind) {
              gmic_check_indice(ind,"Build 3D elevation of image%s");
              print(images,"Build 3D elevation of image%s with elevation map [%d].",gmic_inds,ind);
              if (images[ind].dimv()==1) elev = images[ind];
              else elev = images[ind].get_pointwise_norm();
            } else print(images,"Build 3D elevation of image%s with z-factor %g.",gmic_inds,zfact);
            cimg_foroff(indices,l) {
              CImg<T>& img = images[indices[l]];
              CImgList<unsigned int> primitives3d;
              CImgList<unsigned char> colors3d;
              CImg<float> opacities3d, points3d;
              if (elev) points3d = img.get_elevation3d(primitives3d,colors3d,elev);
              else {
                if (img.dimv()==1) (elev = img)*=zfact; else (elev = img.get_pointwise_norm())*=zfact;
                points3d = img.get_elevation3d(primitives3d,colors3d,elev);
                elev.assign();
              }
              opacities3d.assign(1,primitives3d.size,1,1,1);
              points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
              gmic_apply(img,replace(points3d));
            }
          } else error(images,"Build 3D elevation : invalid argument '%s' "
                       "(should be 'z-factor' or '[indice]').",argument_text);
          ++position; continue;
        }

        // Build 3D isovalue.
        if (!cimg::strcmp("-isovalue3d",command_name)) {
          float value = 0; char end = 0;
          if (std::sscanf(argument,"%f%c",&value,&end)==1) {
            print(images,"Build 3D isovalue %g of image%s.",value,gmic_inds);
            cimg_foroff(indices,l) {
              const unsigned int ind = indices[l];
              CImg<T>& img = images[ind];
              CImg<float> points3d;
              CImgList<unsigned int> primitives3d;
              CImgList<unsigned char> colors3d;
              CImg<float> opacities3d;
              CImg<unsigned char> palette;
              palette.assign(3,img.dim,1,1,220).noise(35,1);
              if (img.dim==1) palette(0) = palette(1) = palette(2) = 255;
              else {
                palette(0,0) = 255; palette(1,0) = 30; palette(2,0) = 30;
                palette(0,1) = 30; palette(1,1) = 255; palette(2,1) = 30;
                if (img.dim>=3) palette(0,2) = 30; palette(1,2) = 30; palette(2,2) = 255;
              }
              cimg_forV(img,k) {
                CImgList<unsigned int> prims;
                const CImg<float> pts = img.get_shared_channel(k).get_isovalue3d(prims,value);
                if (pts) {
                  points3d.append_object3d(primitives3d,pts,prims);
                  colors3d.insert(prims.size,
                                  CImg<unsigned char>::vector(palette(0,k),palette(1,k),palette(2,k)));
                }
              }
              opacities3d.assign(1,primitives3d.size,1,1,1);
              if (!points3d)
                warning(images,"Build 3D isovalue of image [%u] : Isovalue %g not found.",ind,value);
              else points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
              gmic_apply(img,replace(points3d));
            }
          } else error(images,"Build 3D isovalue of image%s : Invalid argument '%s' "
                       "(should be 'isovalue').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Center a 3D object.
        if (!cimg::strcmp("-center3d",command_name) || !cimg::strcmp("-c3d",command_name)) {
          print(images,"Center 3D object%s.",gmic_inds);
          cimg_foroff(indices,l) {
            const unsigned int ind = indices[l];
            if (!images[ind].is_CImg3d())
              error(images,"Center 3D object%s : Image [%d] is not a 3D object.",gmic_inds,ind);
            gmic_apply(images[ind],centerCImg3d());
          }
          continue;
        }

        // Normalize a 3D object.
        if (!cimg::strcmp("-normalize3d",command_name) || !cimg::strcmp("-n3d",command_name)) {
          print(images,"Normalize 3D object%s.",gmic_inds);
          cimg_foroff(indices,l) {
            const unsigned int ind = indices[l];
            if (!images[ind].is_CImg3d())
              error(images,"Normalize 3D object%s : Image [%d] is not a 3D object.",gmic_inds,ind);
            gmic_apply(images[ind],normalizeCImg3d());
          }
          continue;
        }

        // Rotate a 3D object.
        if (!cimg::strcmp("-rotate3d",command_name) || !cimg::strcmp("-rot3d",command_name)) {
          float u = 0, v = 0, w = 1, angle = 0; char end = 0;
          if (std::sscanf(argument,"%f,%f,%f,%f%c",&u,&v,&w,&angle,&end)==4) {
            print(images,"Rotate 3D object%s around axis (%g,%g,%g) with angle %g.",gmic_inds,u,v,w,angle);
            const CImg<float> rot = CImg<float>::rotation_matrix(u,v,w,(float)(angle*cimg::valuePI/180));
            cimg_foroff(indices,l) {
              const unsigned int ind = indices[l];
              if (!images[ind].is_CImg3d())
                error(images,"Rotate 3D object%s : Image [%d] is not a 3D object.",gmic_inds,ind);
              gmic_apply(images[ind],rotateCImg3d(rot));
            }
          } else error(images,"Rotate 3D object%s : Invalid argument '%s' "
                       "(should be 'u,v,w,angle').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Add 3D objects together or translate a 3D object.
        if (!cimg::strcmp("-add3d",command_name) || !cimg::strcmp("-+3d",command_name)) {
          float tx = 0, ty = 0, tz = 0; int ind0 = no_ind; char sep = 0, end = 0;
          if (std::sscanf(argument,"%f%c",&tx,&end)==1 ||
              std::sscanf(argument,"%f,%f%c",&tx,&ty,&end)==2 ||
              std::sscanf(argument,"%f,%f,%f%c",&tx,&ty,&tz,&end)==3) {
            print(images,"Translate 3D object%s with vector (%g,%g,%g).",gmic_inds,tx,ty,tz);
            cimg_foroff(indices,l) {
              const unsigned int ind = indices[l];
              if (!images[ind].is_CImg3d())
                error(images,"Translate 3D object%s : Image [%d] is not a 3D object.",gmic_inds,ind);
              gmic_apply(images[ind],translateCImg3d(tx,ty,tz));
            }
            ++position;
          } else if (std::sscanf(argument,"[%d%c%c",&ind0,&sep,&end)==2 && sep==']') {
            gmic_check_indice(ind0,"Merge object with 3D object%s.");
            const CImg<T> img0 = images[ind0];
            if (!img0.is_CImg3d()) error(images,"Merge object [%d] with 3D object%s : Image [%d] is not a 3D object.",ind0,gmic_inds,ind0);
            print(images,"Merge object [%d] with 3D object%s.",ind0,gmic_inds);
            cimg_foroff(indices,l) {
              const unsigned int ind = indices[l];
              const CImg<T> &img = images[ind];
              if (!img.is_CImg3d())
                error(images,"Merge object [%d] with 3D object%s : Image [%d] is not a 3D object.",ind0,gmic_inds,ind);
              gmic_apply(images[ind],appendCImg3d(img0));
            }
            ++position;
          } else {
            print(images,"Merge 3D object%s together.",gmic_inds);
            if (indices) {
              const unsigned int ind0 = indices[0];
              if (!images[ind0].is_CImg3d())
                error(images,"Merge 3D object%s together : Image [%d] is not a 3D object.",gmic_inds,ind0);
              for (unsigned int siz = indices.size(), off = 0, l = 1; l<siz; ++l) {
                const unsigned int ind = indices[l] - off;
                if (!images[ind].is_CImg3d())
                  error(images,"Merge 3D object%s together : Image [%d] is not a 3D object.",gmic_inds,ind);
                images[ind0].appendCImg3d(images[ind]);
                images.remove(ind); filenames.remove(ind);
                ++off;
              }
            }
          }
          continue;
        }

        // Translate 3D object by the opposite vector.
        if (!cimg::strcmp("-sub3d",command_name) || !cimg::strcmp("--3d",command_name)) {
          float tx = 0, ty = 0, tz = 0; char end = 0;
          if (std::sscanf(argument,"%f%c",&tx,&end)==1 ||
              std::sscanf(argument,"%f,%f%c",&tx,&ty,&end)==2 ||
              std::sscanf(argument,"%f,%f,%f%c",&tx,&ty,&tz,&end)==3) {
            print(images,"Translate 3D object%s with vector -(%g,%g,%g).",gmic_inds,tx,ty,tz);
            cimg_foroff(indices,l) {
              CImg<T>& img = images[indices[l]];
              CImgList<unsigned int> primitives3d;
              CImgList<unsigned char> colors3d;
              CImg<float> opacities3d;
              CImg<T> points3d;
              if (get_version) points3d.assign(img); else img.transfer_to(points3d);
              points3d.CImg3dtoobject3d(primitives3d,colors3d,opacities3d);
              points3d.get_shared_line(0)-=tx;
              points3d.get_shared_line(1)-=ty;
              points3d.get_shared_line(2)-=tz;
              points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
              if (get_version) {
                points3d.transfer_to(images);
                filenames.insert(filenames[indices[l]]);
              } else points3d.transfer_to(images[indices[l]]);
            }
          } else error(images,"Translate 3D object%s : Invalid argument '%s' "
                       "(should be 'tx,_ty,_tz').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Scale a 3D object.
        bool divide = false;
        if (!cimg::strcmp("-mul3d",command_name) || !cimg::strcmp("-*3d",command_name) ||
            ((divide=true)==true && (!cimg::strcmp("-div3d",command_name) || !cimg::strcmp("-/3d",command_name)))) {
          float sx = 0, sy = 1, sz = 1; char end = 0;
          if ((std::sscanf(argument,"%f%c",&sx,&end)==1 && (sy = sz = sx),1) ||
              std::sscanf(argument,"%f,%f%c",&sx,&sy,&end)==2 ||
              std::sscanf(argument,"%f,%f,%f%c",&sx,&sy,&sz,&end)==3) {
            if (divide) print(images,"Scale 3D object%s with factors (1/%g,1/%g,1/%g).",gmic_inds,sx,sy,sz);
            else print(images,"Scale 3D object%s with factors (%g,%g,%g).",gmic_inds,sx,sy,sz);
            cimg_foroff(indices,l) {
              CImg<T>& img = images[indices[l]];
              CImgList<unsigned int> primitives3d;
              CImgList<unsigned char> colors3d;
              CImg<float> opacities3d;
              CImg<T> points3d;
              if (get_version) points3d.assign(img); else img.transfer_to(points3d);
              points3d.CImg3dtoobject3d(primitives3d,colors3d,opacities3d);
              if (divide) {
                points3d.get_shared_line(0)/=sx;
                points3d.get_shared_line(1)/=sy;
                points3d.get_shared_line(2)/=sz;
              } else {
                points3d.get_shared_line(0)*=sx;
                points3d.get_shared_line(1)*=sy;
                points3d.get_shared_line(2)*=sz;
              }
              points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
              if (get_version) {
                points3d.transfer_to(images);
                filenames.insert(filenames[indices[l]]);
              } else points3d.transfer_to(images[indices[l]]);
            }
          } else error(images,"Scale 3D object%s : Invalid argument '%s' "
                       "(should be 'fact' or 'factx,facty,_factz').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Set color of 3D object(s).
        if (!cimg::strcmp("-color3d",command_name) || !cimg::strcmp("-col3d",command_name)) {
          float R = 200, G = 200, B = 200, opacity = -1; char end = 0;
          if (std::sscanf(argument,"%f,%f,%f%c",&R,&G,&B,&end)==3 ||
              std::sscanf(argument,"%f,%f,%f,%f%c",&R,&G,&B,&opacity,&end)==4) {
            const bool set_opacity = (opacity>=0);
            R = (float)cimg::round(R,1); G = (float)cimg::round(G,1); B = (float)cimg::round(B,1);
            if (R<0) R = 0; if (R>255) R = 255;
            if (G<0) G = 0; if (G>255) G = 255;
            if (B<0) B = 0; if (B>255) B = 255;
            if (set_opacity) print(images,"Set colors of 3D object%s to (%g,%g,%g) and opacity to %g.",gmic_inds,R,G,B,opacity);
            else print(images,"Set color of 3D object%s to (%g,%g,%g).",gmic_inds,R,G,B);
            cimg_foroff(indices,l) {
              const unsigned int ind = indices[l];
              if (!images[ind].is_CImg3d())
                error(images,"Set color of 3D object%s : Image [%d] is not a 3D object.",gmic_inds,ind);
              gmic_apply(images[ind],coloropacityCImg3d(R,G,B,opacity,true,set_opacity));
            }
          } else error(images,"Set color of 3D object%s : Invalid argument '%s' "
                       "(should be 'R,G,B,_opacity').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Set opacity of 3D object(s).
        if (!cimg::strcmp("-opacity3d",command_name) || !cimg::strcmp("-opac3d",command_name)) {
          float opacity = 1; char end = 0;
          if (std::sscanf(argument,"%f%c",&opacity,&end)==1) {
            print(images,"Set opacity of 3D object%s to %g.",gmic_inds,opacity);
            cimg_foroff(indices,l) {
              const unsigned int ind = indices[l];
              if (!images[ind].is_CImg3d())
                error(images,"Set opacity of 3D object%s : Image [%d] is not a 3D object.",gmic_inds,ind);
              gmic_apply(images[ind],coloropacityCImg3d(0,0,0,opacity,false,true));
            }
          } else error(images,"Set opacity of 3D object%s : Invalid argument '%s' "
                       "(should be 'opacity').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Invert 3D orientation.
        if (!cimg::strcmp("-invert3d",command_name) || !cimg::strcmp("-i3d",command_name)) {
          print(images,"Invert orientation of 3D object%s.",gmic_inds);
          cimg_foroff(indices,l) {
            CImg<T> &img = images[indices[l]];
            CImgList<unsigned int> primitives3d;
            CImgList<unsigned char> colors3d;
            CImg<float> opacities3d;
            CImg<T> points3d;
            if (get_version) points3d.assign(img); else img.transfer_to(points3d);
            points3d.CImg3dtoobject3d(primitives3d,colors3d,opacities3d);
            if (primitives3d) primitives3d.invert_object3d();
            points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
            if (get_version) {
              points3d.transfer_to(images);
              filenames.insert(filenames[indices[l]]);
            } else points3d.transfer_to(images[indices[l]]);
          }
          continue;
        }

        // Split 3D object(s) into 6 vector images {header,N,vertices,primitives,colors,opacities}
        if (!cimg::strcmp("-split3d",command_name) || !cimg::strcmp("-s3d",command_name)) {
          print(images,"Split 3D object%s into its different characteristics.",gmic_inds);
          unsigned int off = 0;
          cimg_foroff(indices,l) {
            const unsigned int ind = indices[l] + off;
            CImg<T> &img = images[ind];
            const CImg<char> filename = filenames[ind];
            CImgList<unsigned int> primitives3d;
            CImgList<unsigned char> colors3d;
            CImg<float> opacities3d;
            CImg<T> points3d;
            if (get_version) points3d.assign(img); else img.transfer_to(points3d);
            points3d.CImg3dtoobject3d(primitives3d,colors3d,opacities3d);
            CImgList<T> split;
            split.insert(CImg<T>("CImg3d",1,6,1,1,false)+=0.5f);
            split.insert(CImg<T>::vector((T)points3d.dimx(),(T)primitives3d.size));
            points3d.resize(-100,3,1,1,0).transpose().unroll('y').transfer_to(split);
            points3d.assign();
            CImgList<T> _prims;
            cimglist_for(primitives3d,p)
              _prims.insert(CImg<T>::vector((T)primitives3d[p].size())).insert(primitives3d[p]).last().unroll('y');
            primitives3d.assign();
            split.insert(_prims.get_append('y')); _prims.assign();
            split.insert(colors3d.get_append('x').transpose().unroll('y')); colors3d.assign();
            opacities3d.transfer_to(split);
            if (get_version) {
              images.insert(split);
              filenames.insert(split.size,filename);
            } else {
              images.remove(ind); images.insert(split,ind);
              filenames.remove(ind); filenames.insert(split.size,filename,ind);
              off+=split.size-1;
            }
          }
          continue;
        }

        // Set 3D light position.
        if (!cimg::strcmp("-light3d",item) || !cimg::strcmp("-l3d",item)) {
          float lx = 0, ly = 0, lz = -5000; char end = 0;
          if (std::sscanf(argument,"%f,%f,%f%c",&lx,&ly,&lz,&end)==3) {
            print(images,"Set 3D light position at (%g,%g,%g).",lx,ly,lz);
            light3d_x = lx;
            light3d_y = ly;
            light3d_z = lz;
          } else error(images,"Set 3D light position : Invalid argument '%s' "
                       "(should be 'posx,posy,posz').",argument_text);
          ++position; continue;
        }

        // Set 3D focale.
        if (!cimg::strcmp("-focale3d",item) || !cimg::strcmp("-f3d",item)) {
          float focale = 500; char end = 0;
          if (std::sscanf(argument,"%f%c",&focale,&end)==1 && focale>0) {
            focale3d = focale;
            print(images,"Set 3D focale to %g.",focale);
          } else error(images,"Set 3D focale : Invalid argument '%s' "
                       "(should be 'focale>0').",argument_text);
          ++position; continue;
        }

        // Set 3D specular light parameters.
        if (!cimg::strcmp("-specl3d",item) || !cimg::strcmp("-sl3d",item)) {
          float value = 0; char end = 0;
          if (std::sscanf(argument,"%f%c",&value,&end)==1 && value>=0) {
            specular_light3d = value;
            print(images,"Set amount of 3D specular light to %g.",specular_light3d);
          } else error(images,"Set amount of 3D specular light : invalid argument '%s'"
                       "(should be 'value>=0').",argument_text);
          ++position; continue;
        }

        if (!cimg::strcmp("-specs3d",item) || !cimg::strcmp("-ss3d",item)) {
          float value = 0; char end = 0;
          if (std::sscanf(argument,"%f%c",&value,&end)==1 && value>=0) {
            specular_shine3d = value;
            print(images,"Set shininess of 3D specular light to %g.",specular_shine3d);
          }
          else error(images,"Set shininess of 3D specular light : invalid argument '%s'"
                     "(should be 'value>=0').",argument_text);
          ++position; continue;
        }

        // Switch double-sided mode for 3D rendering.
        if (!cimg::strcmp("-orient3d",item) || !cimg::strcmp("-o3d",item)) {
          is_oriented3d = !is_oriented3d;
          continue;
        }

        // Set 3D rendering mode.
        if (!cimg::strcmp("-render3d",item) || !cimg::strcmp("-r3d",item)) {
          int value = 0; char end = 0;
          if (std::sscanf(argument,"%d%c",&value,&end)==1 && value>=-1 && value<=5) {
            render3d = value;
            print(images,"Set static 3D render mode to %s.",
                  render3d==-1?"bounding-box":
                  render3d==0?"pointwise":render3d==1?"linear":render3d==2?"flat":
                  render3d==3?"flat-shaded":render3d==4?"Gouraud-shaded":
                  render3d==5?"Phong-shaded":"none");
          } else error(images,"Set static 3D render mode : invalid argument '%s'"
                       "(should be 'render_type={0,1,2,3,4,5}').",
                       argument_text);
          ++position; continue;
        }

        if (!cimg::strcmp("-renderd3d",item) || !cimg::strcmp("-rd3d",item)) {
          int value = 0; char end = 0;
          if (std::sscanf(argument,"%d%c",&value,&end)==1 && value>=-1 && value<=5) {
            renderd3d = value;
            print(images,"Set dynamic 3D render mode to %s.",
                  renderd3d==-1?"bounding-box":
                  renderd3d==0?"pointwise":renderd3d==1?"linear":renderd3d==2?"flat":
                  renderd3d==3?"flat-shaded":renderd3d==4?"Gouraud-shaded":
                  renderd3d==5?"Phong-shaded":"none");
          } else error(images,"Set dynamic 3D render mode : invalid argument '%s'"
                       "(should be 'render_type={0,1,2,3,4,5}').",
                       argument_text);
          ++position; continue;
        }

        // Set 3D background color.
        if (!cimg::strcmp("-background3d",item) || !cimg::strcmp("-b3d",item)) {
          int R = 0, G = 0, B = 0; char end = 0;
          const int nb = std::sscanf(argument,"%d,%d,%d%c",&R,&G,&B,&end);
          switch (nb) {
          case 1 : background3d[0] = background3d[1] = background3d[2] = R; break;
          case 2 : background3d[0] = R; background3d[1] = background3d[2] = G; break;
          case 3 : background3d[0] = R; background3d[1] = G; background3d[2] = B; break;
          default: error(images,"Set 3D background color : Invalid argument '%s'"
                         "(should be 'R,_G,_B').",argument_text);
          }
          print(images,"Set 3D background color to (%d,%d,%d).",
                (int)background3d[0],(int)background3d[1],(int)background3d[2]);
          ++position; continue;
        }

        //----------------------
        // Procedural commands.
        //----------------------

        // No operations : do nothing
        if (!cimg::strcmp("-nop",item)) {
          if (verbosity_level>0) print(images,"Do nothing.");
          continue;
        }

        // Skip next argument;
        if (!cimg::strcmp("-skip",item)) {
          if (verbosity_level>0) print(images,"Skip argument '%s'.",argument_text);
          ++position;
          continue;
        }

        // Echo.
        if (!cimg::strcmp("-echo",item) || !cimg::strcmp("-e",item)) {
          char text[4096] = { 0 };
          std::strcpy(text,argument);
          cimg::strclean(text);
          cimg::strescape(text);
          print(images,"%s",text);
          ++position; continue;
        }

        // Print.
        if (!cimg::strcmp("-print",command_name) || !cimg::strcmp("-p",command_name)) {
          if (images.size) {
            print(images,"Print image%s.\n\n",gmic_inds);
            char title[4096];
            if (verbosity_level>=0) cimg_foroff(indices,l) {
              const unsigned int ind = indices[l];
              std::sprintf(title,"image [%u] = '%s'",ind,filenames[ind].ptr());
              images[ind].print(title);
            }
            is_released = true;
          } else print(images,"Print image[].");
          continue;
        }

        // Return
        if (!cimg::strcmp("-return",item)) {
          if (verbosity_level>0) print(images,"Return.");
          dowhiles.assign();
          repeatdones.assign();
          locals.assign();
          position = command_line.size;
          continue;
        }

        // Quit.
        if (!cimg::strcmp("-quit",item) || !cimg::strcmp("-q",item)) {
          print(images,"Quit.");
          is_released = is_quit = true;
          dowhiles.assign();
          repeatdones.assign();
          locals.assign();
          position = command_line.size;
          continue;
        }

        // Exec
        if (!cimg::strcmp("-exec",item)) {
          char exec_command[4096] = { 0 };
          std::strcpy(exec_command,argument);
          cimg::strclean(exec_command);
          print(images,"Execute external command '%s'.\n",exec_command);
          int err = cimg::system(exec_command);
          err = ++position; continue;
        }

        // Do...while.
        if (!cimg::strcmp("-do",item)) {
          if (verbosity_level>0) print(images,"Do : Start do..while loop.");
          dowhiles.insert(CImg<unsigned int>::vector(position));
          continue;
        }

        if (!cimg::strcmp("-while",item)) {
          if (!dowhiles) error(images,"While : Missing associated '-do' command.");
          float number = 0; bool cond = false; char end = 0;
          if (std::sscanf(argument,"%f%c",&number,&end)==1) cond = (bool)number;
          else cond = (bool)last_image.eval(argument);
          if (verbosity_level>0) print(images,"While : Check '%s' -> %s.",argument_text,cond?"true":"false");
          if (cond) { position = dowhiles.last()(0); continue; }
          else dowhiles.remove();
          ++position; continue;
        }

        // If..else..endif
        if (!cimg::strcmp("-if",item)) {
          float number = 0; bool cond = false; char end = 0;
          if (std::sscanf(argument,"%f%c",&number,&end)==1) cond = (bool)number;
          else cond = (bool)last_image.eval(argument);
          if (verbosity_level>0) print(images,"If : Check '%s' -> %s.",argument_text,cond?"true":"false");
          if (!cond) {
            for (int nbifs = 1; nbifs && position<command_line.size; ++position) {
              const char *it = command_line[position].ptr();
              if (!cimg::strcmp("-if",it)) ++nbifs;
              if (!cimg::strcmp("-endif",it)) --nbifs;
              if (!cimg::strcmp("-else",it) && nbifs==1) --nbifs;
            }
            continue;
          }
          ++position; continue;
        }

        if (!cimg::strcmp("-else",item)) {
          for (int nbifs = 1; nbifs && position<command_line.size; ++position) {
            if (!cimg::strcmp("-if",command_line[position].ptr())) ++nbifs;
            if (!cimg::strcmp("-endif",command_line[position].ptr())) --nbifs;
          }
          continue;
        }

        if (!cimg::strcmp("-endif",item)) {
          if (verbosity_level>0) print(images,"End if.");
          continue;
        }

        // Repeat...done
        if (!cimg::strcmp("-repeat",item)) {
          float number = 0; int nb = 0; char end = 0;
          if (std::sscanf(argument,"%f%c",&number,&end)==1) nb = (int)number;
          else nb = (int)last_image.eval(argument);
          if (verbosity_level>0) print(images,"Start %d iteration%s of a repeat...done loop.",nb,nb>1?"s":"");
          if (nb>0) repeatdones.insert(CImg<unsigned int>::vector(position+1,nb,0));
          else {
            int nbrepeats = 0;
            for (nbrepeats = 1; nbrepeats && position<command_line.size; ++position) {
              const char *it = command_line[position].ptr();
              if (!cimg::strcmp("-repeat",it)) ++nbrepeats;
              if (!cimg::strcmp("-done",it)) --nbrepeats;
            }
            if (nbrepeats && position>=command_line.size)
              error(images,"Repeat : Missing associated '-done' command.");
            continue;
          }
          ++position; continue;
        }

        if (!cimg::strcmp("-done",item)) {
          if (!repeatdones) error(images,"Done : Missing associated '-repeat' command.");
          if (--repeatdones.last()(1)) {
            ++repeatdones.last()(2);
            position = repeatdones.last()(0);
          }
          else repeatdones.remove();
          continue;
        }

        // Check argument types.
        if (!cimg::strcmp("-int",item)) {
          char it[4096] = { 0 }, end = 0, sep = 0; int value = 0;
          for (const char *nargument = argument; *nargument; ) {
            const int nb = std::sscanf(nargument,"%4095[^,]%c",it,&sep);
            if (nb) {
              if (std::sscanf(it,"%d%c",&value,&end)==1) nargument+=std::strlen(it) + nb - 1;
              else error(images,"Check integer : Argument '%s' is not an integer value.",it);
            } else error(images,"Check integer : Argument '%s' is not an integer value.",argument_text);
          }
          ++position; continue;
        }

        if (!cimg::strcmp("-float",item)) {
          char it[4096] = { 0 }, end = 0, sep = 0; double value = 0;
          for (const char *nargument = argument; *nargument; ) {
            const int nb = std::sscanf(nargument,"%4095[^,]%c",it,&sep);
            if (nb) {
              if (std::sscanf(it,"%lf%c",&value,&end)==1) nargument+=std::strlen(it) + nb -1;
              else error(images,"Check float : Argument '%s' is not a float value.",it);
            } else error(images,"Check float : Argument '%s' is not a float value.",argument_text);
          }
          ++position; continue;
        }

        if (!cimg::strcmp("-error",item)) {
          char text[4096] = { 0 };
          std::strcpy(text,argument);
          cimg::strclean(text);
          cimg::strescape(text);
          error(images,text);
        }

        // Handle local environnements.
        if (!cimg::strcmp("-local",command_name) || !cimg::strcmp("-l",command_name)) {
          print(images,"Start local environment with image%s.",gmic_inds);
          locals.insert(indices);
          const unsigned int siz = indices.size();
          CImgList<T> nimages(siz);
          CImgList<char> nfilenames(siz);
          if (get_version) {
            cimg_foroff(indices,l) { nimages[l].assign(images[indices[l]]); nfilenames[l].assign(filenames[indices[l]]); }
            parse(command_line,position,nimages,nfilenames,dowhiles,repeatdones,locals,false);
            const unsigned int isiz = images.size;
            images.insert(nimages.size); filenames.insert(nimages.size);
            cimglist_for(nimages,i) { nimages[i].swap(images[isiz+i]); nfilenames[i].swap(filenames[isiz+i]); }
          } else {
            cimg_foroff(indices,l) { nimages[l].swap(images[indices[l]]); nfilenames[l].swap(filenames[indices[l]]); }
            parse(command_line,position,nimages,nfilenames,dowhiles,repeatdones,locals,false);
            const unsigned int nb = cimg::min(siz,nimages.size);
            for (unsigned int i = 0; i<nb; ++i) {
              images[indices[i]].swap(nimages[0]); filenames[indices[i]].swap(nfilenames[0]);
              nimages.remove(0); nfilenames.remove(0);
            }
            if (nb<siz) for (unsigned int off = 0, l = nb; l<siz; ++l, ++off) {
              const unsigned int ind = indices[l] - off;
              images.remove(ind); filenames.remove(ind);
            } else if (nimages) {
              const unsigned int ind0 = siz?indices[siz-1]+1:images.size;
              images.insert(nimages,ind0); filenames.insert(nimages.size,CImg<char>("(gmic)",7,1,1,1,false),ind0);
            }
          }
          continue;
        }

        if (!cimg::strcmp("-endlocal",item) || !cimg::strcmp("-endl",item)) {
          if (locals) {
            print(images,"End local environment with image%s.",indices2string(locals.last(),filenames,true));
            locals.remove();
            return *this;
          } else error(images,"End local environment : No local environment had been started.");
          continue;
        }

        //--------------------------
        // Input/Output and Display
        //--------------------------

        // Display.
        if (!cimg::strcmp("-display",command_name) || !cimg::strcmp("-d",command_name)) {
          is_released |= display_images(images,filenames,indices,true);
          continue;
        }

        // Display image(s) in an instant window.
        if (!cimg::strcmp("-window",command_name) || !cimg::strcmp("-w",command_name)) {
          int dimw = 0, dimh = 0, norm = 3, framerate = 20; char end = 0;
          if ((std::sscanf(argument,"%d,%d%c",&dimw,&dimh,&end)==2 ||
               std::sscanf(argument,"%d,%d,%d%c",&dimw,&dimh,&norm,&end)==3 ||
               std::sscanf(argument,"%d,%d,%d,%d%c",&dimw,&dimh,&norm,&framerate,&end)==4)
              && dimw>=0 && dimh>=0 && norm>=0 && norm<=3) ++position;
#if cimg_display==0
          print(images,"Display image%s in instant window skipped (no display available).",gmic_inds);
          continue;
#endif
          if (dimw>0 && dimh>0) {
            if (instant_window) instant_window.resize(dimw,dimh).normalization = norm;
            else instant_window.assign(dimw,dimh,"(gmic)",norm);
          } else instant_window.assign();
          if (instant_window) {
            if (framerate>0)
              print(images,"Display image%s in instant %dx%d window, with %d ms framerate and %snormalization.",
                    gmic_inds,instant_window.dimx(),instant_window.dimy(),framerate,norm==0?"no ":norm==1?"":norm==2?"1st-time ":"auto-");
            else if (framerate<0)
              print(images,"Display image%s in instant %dx%d window, with key-press and %snormalization.",
                    gmic_inds,instant_window.dimx(),instant_window.dimy(),norm==0?"no ":norm==1?"":norm==2?"1st-time ":"auto-");
            else
              print(images,"Display image%s in instant %dx%d window, with %snormalization.",
                    gmic_inds,instant_window.dimx(),instant_window.dimy(),norm==0?"no ":norm==1?"":norm==2?"1st-time ":"auto-");
            CImgList<T> subimages;
            cimg_foroff(indices,l) subimages.insert(images[indices[l]],~0U,true);
            if (subimages) { subimages.display(instant_window); is_released = true; }
            if (framerate>=0) cimg::wait(framerate);
            else { while (!instant_window.key && !instant_window.is_closed) instant_window.wait(); instant_window.key = 0; }
          }
          continue;
        }

        // Display 3D object.
        if (!cimg::strcmp("-display3d",command_name) || !cimg::strcmp("-d3d",command_name)) {
          is_released |= display_objects3d(images,filenames,indices,true);
          continue;
        }

        // Display as a graph plot.
        if (!cimg::strcmp("-plot",command_name)) {
          unsigned int plot_type = 1, vertex_type = 1; int resolution = 65536;
          double ymin = 0, ymax = 0, xmin = 0, xmax = 0; char end = 0, sep = 0, formula[4096] = { 0 };
          if (((std::sscanf(argument,"'%1023[^']%c%c",formula,&sep,&end)==2 && sep=='\'') ||
               std::sscanf(argument,"'%1023[^']',%lf,%lf%c",formula,&xmin,&xmax,&end)==3 ||
               std::sscanf(argument,"'%1023[^']',%lf,%lf,%lf,%lf%c",formula,&xmin,&xmax,&ymin,&ymax,&end)==5 ||
               std::sscanf(argument,"'%1023[^']',%lf,%lf,%lf,%lf,%d%c",formula,&xmin,&xmax,&ymin,&ymax,&resolution,&end)==6 ||
               std::sscanf(argument,"'%1023[^']',%lf,%lf,%lf,%lf,%d,%u%c",formula,&xmin,&xmax,&ymin,&ymax,&resolution,&plot_type,&end)==7 ||
               std::sscanf(argument,"'%1023[^']',%lf,%lf,%lf,%lf,%d,%u,%u%c",formula,&xmin,&xmax,&ymin,&ymax,&resolution,&plot_type,&vertex_type,&end)==8) &&
              resolution>0 && plot_type<=3 && vertex_type<=7) {
            if (xmin==0 && xmax==0) { xmin = -4; xmax = 4; }
            if (!plot_type && !vertex_type) plot_type = 1;
            if (resolution<1) resolution = 65536;
            CImgList<double> tmp_img(1);
            CImg<double> &img = tmp_img[0];
            img.assign(resolution--).eval(formula);
            const double dx = xmax - xmin;
            cimg_forX(img,X) img(X) = img.eval(0,xmin+X*dx/resolution);
            CImgList<char> tmp_filename;
            tmp_filename.insert(CImg<char>(formula,std::strlen(formula)+1,1,1,1,false));
            is_released |= display_plots(tmp_img,tmp_filename,CImg<unsigned int>::vector(0),plot_type,vertex_type,xmin,xmax,ymin,ymax,true);
            ++position;
          } else {
            plot_type = 1; vertex_type = 0; ymin = ymax = xmin = xmax = 0;
            if ((std::sscanf(argument,"%u%c",&plot_type,&end)==1 ||
                 std::sscanf(argument,"%u,%u%c",&plot_type,&vertex_type,&end)==2 ||
                 std::sscanf(argument,"%u,%u,%lf,%lf%c",&plot_type,&vertex_type,&xmin,&xmax,&end)==4 ||
                 std::sscanf(argument,"%u,%u,%lf,%lf,%lf,%lf%c",&plot_type,&vertex_type,&xmin,&xmax,&ymin,&ymax,&end)==6) &&
                plot_type<=3 && vertex_type<=7) ++position;
            if (!plot_type && !vertex_type) plot_type = 1;
            is_released |= display_plots(images,filenames,indices,plot_type,vertex_type,xmin,xmax,ymin,ymax,true);
          }
          continue;
        }

        // Select image feature.
        if (!cimg::strcmp("-select",command_name)) {
          unsigned int select_type = 0; char end = 0;
          if (std::sscanf(argument,"%u%c",&select_type,&end)==1 &&
              select_type<=3) {
            cimg_foroff(indices,l) gmic_apply(images[indices[l]],select(filenames[indices[l]].ptr(),select_type));
          } else error(images,"Select image%s : Invalid argument '%s' "
                       "(should be 'select_type={0,1,2}').",gmic_inds,argument_text);
          ++position; continue;
        }

        // Output.
        if (!cimg::strcmp("-output",command_name) || !cimg::strcmp("-o",command_name)) {
          char filename[4096] = { 0 }; char options[4096] = { 0 };
          if (std::sscanf(argument,"%4095[^,],%s",filename,options)!=2) std::strcpy(filename,argument);
          const char *const ext = cimg::split_filename(filename);
          if (!cimg::strcasecmp("off",ext)) {
            char nfilename[4096] = { 0 };
            std::strcpy(nfilename,filename);
            const unsigned int siz = indices.size();
            cimg_foroff(indices,l) {
              const unsigned int ind = indices[l];
              if (siz!=1) cimg::number_filename(filename,l,6,nfilename);
              if (!images[ind].is_CImg3d())
                error(images,"Output 3D object [%u] as file '%s' : Image [%u] is not a 3D object.",ind,nfilename,ind);
              print(images,"Output 3D object [%u] as file '%s'.",ind,nfilename);
              CImgList<unsigned int> primitives3d;
              CImgList<unsigned char> colors3d;
              CImg<float> opacities3d;
              CImg<float> points3d(images[ind]);
              points3d.CImg3dtoobject3d(primitives3d,colors3d,opacities3d).save_off(nfilename,primitives3d,colors3d);
            }
          } else if (!cimg::strcasecmp("jpeg",ext) || !cimg::strcasecmp("jpg",ext)) {
            int quality = 100; char end = 0;
            if (std::sscanf(options,"%d%c",&quality,&end)!=1) quality = 100;
            if (quality<0) quality = 0; else if (quality>100) quality = 100;
            CImgList<T> output_images;
            cimg_foroff(indices,l) output_images.insert(images[indices[l]],~0U,true);
            print(images,"Output image%s as file '%s', with quality %u%%",gmic_inds,filename,quality);
            if (!output_images) throw CImgInstanceException("CImgList<%s>::save() : File '%s, instance list (%u,%p) is empty.",
                                                            output_images.pixel_type(),filename,
                                                            output_images.size,output_images.data);
            if (output_images.size==1) output_images[0].save_jpeg(filename,quality);
            else {
              char nfilename[1024];
              cimglist_for(output_images,l) {
                cimg::number_filename(filename,l,6,nfilename);
                output_images[l].save_jpeg(nfilename,quality);
              }
            }
          } else {
            CImgList<T> output_images;
            cimg_foroff(indices,l) output_images.insert(images[indices[l]],~0U,true);
            print(images,"Output image%s as file '%s'.",gmic_inds,filename);
            output_images.save(filename);
          }
          is_released = true; ++position; continue;
        }

        // Run macro command, if found.
        if (cimg::strcmp("-i",command_name) && cimg::strcmp("-input",command_name)) {

          const char *macro = 0;
          bool macro_found = false, has_arguments = false;
          CImg<char> substituted_command;
          cimglist_for(macros,l) {
            macro = macros[l].ptr();
            const char *const command = commands[l].ptr();

            if (!cimg::strcmp(command_name+1,macro) && *command) {
              CImgList<char> arguments(256);
              unsigned int nb_arguments = 0;
              char s_argument[4096] = { 0 }, tmp[4096] = { 0 }, tmp2[4096] = { 0 };
              macro_found = true;
              debug(images,"Found macro '%s', substituting by '%s'.",macro,command);

              // Get command-line values of macro arguments.
              if (argument)
                for (const char *nargument = argument; nb_arguments<255 && *nargument; ) {
                  char *ns_argument = s_argument;
                  for (bool inside_dquote = false; *nargument && (*nargument!=',' || inside_dquote); ++nargument) {
                    if (*nargument=='\"') inside_dquote = !inside_dquote;
                    *(ns_argument++) = *nargument;
                  }
                  if (ns_argument!=s_argument) {
                    *ns_argument = 0;
                    CImg<char>(s_argument,std::strlen(s_argument)+1,1,1,1,false).transfer_to(arguments[++nb_arguments]);
                  }
                  if (*nargument) ++nargument;
                }

              // Substitute arguments in macro command expression.
              CImgList<char> lreplacement;
              for (const char *ncommand = command; *ncommand;) if (*ncommand=='$') {
                char *replace_text = &(s_argument[0] = 0), sep = 0; int ind = 0, ind1 = 0;

                // Replace $?.
                if (ncommand[1]=='?') {
                  std::sprintf(s_argument,"%s",gmic_inds);
                  ncommand+=2;

                // Replace $#.
                } else if (ncommand[1]=='#') {
                  std::sprintf(s_argument,"%u",nb_arguments);
                  ncommand+=2;
                  has_arguments = true;

                  // Replace $*.
                } else if (ncommand[1]=='*') {
                  for (unsigned int j = 1; j<=nb_arguments; ++j) {
                    replace_text+=std::sprintf(replace_text,"%s",arguments[j].ptr());
                    if (j<nb_arguments) *(replace_text++) = ',';
                  }
                  replace_text = s_argument;
                  ncommand+=2;
                  has_arguments = true;

                  // Replace ${i*}.
                } else if (std::sscanf(ncommand,"${%d*%c",&ind,&sep)==2 &&
                           ind>0 && ind<256 && sep=='}') {
                  for (unsigned int j = (unsigned int)ind; j<=nb_arguments; ++j) {
                    replace_text+=std::sprintf(replace_text,"%s",arguments[j].ptr());
                    if (j<nb_arguments) *(replace_text++) = ',';
                  }
                  replace_text = s_argument;
                  ncommand+=std::sprintf(tmp,"${%d*}",ind);
                  has_arguments = true;

                  // Replace $i and ${i}.
                } else if ((std::sscanf(ncommand,"$%d",&ind)==1 ||
                            (std::sscanf(ncommand,"${%d%c",&ind,&sep)==2 && sep=='}')) &&
                           ind>0 && ind<256) {
                  if (!arguments[ind]) {
                    if (sep=='}') error(images,"Macro '%s' : Argument '$%d' is undefined (in expression '${%d}').",macro,ind,ind);
                    else error(images,"Macro '%s' : Argument '$%d' is undefined (in expression '$%d').",macro,ind,ind);
                  }
                  replace_text = arguments[ind].ptr();
                  ncommand+=std::sprintf(tmp,"$%d",ind) + (sep=='}'?2:0);
                  if (ind>0) has_arguments = true;

                  // Replace ${i=$j}.
                } else if (std::sscanf(ncommand,"${%d=$%d%c",&ind,&ind1,&sep)==3 && sep=='}' &&
                           ind>0 && ind<256 && ind1>0 && ind1<256) {
                  if (!arguments[ind1])
                    error(images,"Macro '%s' : Argument '$%d' is undefined (in expression '${%d=$%d}').",macro,ind1,ind,ind1);
                  if (!arguments[ind]) arguments[ind] = arguments[ind1];
                  replace_text = arguments[ind].ptr();
                  ncommand+=std::sprintf(tmp,"${%d=$%d}",ind,ind1);
                  has_arguments = true;

                  // Replace ${i=$#}.
                } else if (std::sscanf(ncommand,"${%d=$#%c",&ind,&sep)==2 && sep=='}' &&
                           ind>0 && ind<256) {
                  if (!arguments[ind]) {
                    std::sprintf(s_argument,"%u",nb_arguments);
                    CImg<char>(s_argument,std::strlen(s_argument)+1,1,1,1,false).transfer_to(arguments[ind]);
                  }
                  replace_text = arguments[ind].ptr();
                  ncommand+=std::sprintf(tmp,"${%d=$#}",ind);
                  has_arguments = true;

                  // Replace ${i=default}.
                } else if (std::sscanf(ncommand,"${%d=%4095[^}]%c",&ind,tmp,&sep)==3 && sep=='}' &&
                           ind>0 && ind<256) {
                  if (!arguments[ind]) CImg<char>(tmp,std::strlen(tmp)+1,1,1,1,false).transfer_to(arguments[ind]);
                  replace_text = arguments[ind].ptr();
                  ncommand+=std::strlen(tmp) + 4 + std::sprintf(tmp2,"%d",ind);
                  has_arguments = true;

                  // Any other expression starting by '$'.
                } else {
                  s_argument[0] = '$';
                  if (std::sscanf(ncommand,"%4095[^$]",s_argument+1)!=1) { s_argument[1] = 0; ++ncommand; }
                  else ncommand+=std::strlen(s_argument);
                }

                const int replace_length = std::strlen(replace_text);
                if (replace_length)
                  CImg<char>(replace_text,replace_length,1,1,1,false).transfer_to(lreplacement);

              } else {
                std::sscanf(ncommand,"%4095[^$]",s_argument);
                const int replace_length = std::strlen(s_argument);
                if (replace_length) {
                  CImg<char>(s_argument,replace_length,1,1,1,false).transfer_to(lreplacement);
                  ncommand+=std::strlen(s_argument);
                }
              }
              const CImg<char> zero(1,1,1,1,0);
              lreplacement.insert(zero).get_append('x').transfer_to(substituted_command);
              debug(images,"Macro '%s' expanded to '%s'.",macro,substituted_command.ptr());
              break;
            }
          }

          if (macro_found) {
            const CImgList<char> ncommand_line = commandline_to_CImgList(substituted_command.ptr());
            CImgList<unsigned int> ndowhiles, nrepeatdones, nlocals;
            const unsigned int siz = indices.size();
            CImgList<char> nfilenames(siz);
            CImgList<T> nimages(siz);
            unsigned int nposition = 0;

            if (get_version) {
              cimg_foroff(indices,l) { nimages[l].assign(images[indices[l]]); nfilenames[l].assign(filenames[indices[l]]); }
              CImg<char>(macro,std::strlen(macro)+1,1,1,1,false).transfer_to(scope);
              parse(ncommand_line,nposition,nimages,nfilenames,ndowhiles,nrepeatdones,nlocals,false);
              scope.remove();
              const unsigned int isiz = images.size;
              images.insert(nimages.size); filenames.insert(nimages.size);
              cimglist_for(nimages,i) { nimages[i].swap(images[isiz+i]); nfilenames[i].swap(filenames[isiz+i]); }
            } else {
              cimg_foroff(indices,l) { nimages[l].swap(images[indices[l]]); nfilenames[l].swap(filenames[indices[l]]); }
              CImg<char>(macro,std::strlen(macro)+1,1,1,1,false).transfer_to(scope);
              parse(ncommand_line,nposition,nimages,nfilenames,ndowhiles,nrepeatdones,nlocals,false);
              scope.remove();
              const unsigned int nb = cimg::min(siz,nimages.size);
              for (unsigned int i = 0; i<nb; ++i) {
                images[indices[i]].swap(nimages[0]); filenames[indices[i]].swap(nfilenames[0]);
                nimages.remove(0); nfilenames.remove(0);
              }
              if (nb<siz) for (unsigned int off = 0, l = nb; l<siz; ++l, ++off) {
                const unsigned int ind = indices[l] - off;
                images.remove(ind); filenames.remove(ind);
              } else if (nimages) {
                const unsigned int ind0 = siz?indices[siz-1]+1:images.size;
                images.insert(nimages,ind0); filenames.insert(nimages.size,CImg<char>("(gmic)",7,1,1,1,false),ind0);
              }
            }
            if (has_arguments) ++position;
            continue;
          }
        }
      }

      // Input.
      if (!cimg::strcmp("-i",command_name) || !cimg::strcmp("-input",command_name)) ++position;
      else {
        if (get_version) --item;
        argument = item;
        if (std::strlen(argument)>=64) {
          std::memcpy(argument_text,argument,60*sizeof(char));
          argument_text[60] = argument_text[61] = argument_text[62] = '.'; argument_text[63] = 0;
        } else std::strcpy(argument_text,argument);
        command_restriction[0] = 0;
      }
      if (!std::strlen(command_restriction)) indices.assign(1,1,1,1,images.size);
      CImgList<T> input_images;
      CImgList<char> input_filenames;
      bool obj3d = false;
      char st_inds[4096] = { 0 }, stx[4096] = { 0 }, sty[4096] = { 0 }, stz[4096] = { 0 }, stv[4096] = { 0 }, st_values[4096] = { 0 };
      char end = 0, sep = 0, sepx = 0, sepy = 0, sepz = 0, sepv = 0;
      int nb = 1, indx = no_ind, indy = no_ind, indz = no_ind, indv = no_ind;
      float dx = 0, dy = 1, dz = 1, dv = 1;

      if ((std::sscanf(argument,"[%4095[0-9%,:-]%c%c",st_inds,&sep,&end)==2 && sep==']') ||
          std::sscanf(argument,"[%4095[0-9%,:-]],%d%c",st_inds,&nb,&end)==2) { // Nb copies of existing images.

        const CImg<unsigned int> indices0 = indices2cimg(st_inds,images.size,"-input");
        char st_tmp[4096] = { 0 }; std::strcpy(st_tmp,indices2string(indices0,filenames,true));
        if (nb<=0) error(images,"Input %d copies of image%s at position%s : Invalid argument '%s'.",
                         nb,st_tmp,gmic_inds,argument_text);
        if (nb!=1) print(images,"Input %d copies of image%s at position%s",nb,st_tmp,gmic_inds);
        else print(images,"Input copy of image%s at position%s",st_tmp,gmic_inds);
        for (int i = 0; i<nb; ++i) cimg_foroff(indices0,l) {
          input_images.insert(images[indices0[l]]);
          input_filenames.insert(filenames[indices0[l]]);
        }
      } else if ((std::sscanf(argument,"%4095[][0-9.eE%+-]%c",stx,&end)==1 ||
                  std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",stx,sty,&end)==2 ||
                  std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",stx,sty,stz,&end)==3 ||
                  std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-]%c",
                              stx,sty,stz,stv,&end)==4 ||
                  std::sscanf(argument,"%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[][0-9.eE%+-],%4095[^\n]",
                              stx,sty,stz,stv,&(st_values[0]=0))==5) &&
                 (!*stx || std::sscanf(stx,"%f%c",&dx,&end)==1 ||
                  (std::sscanf(stx,"%f%c%c",&dx,&sepx,&end)==2 && sepx=='%') ||
                  (std::sscanf(stx,"[%d%c%c",&indx,&sepx,&end)==2 && sepx==']')) &&
                 (!*sty || std::sscanf(sty,"%f%c",&dy,&end)==1 ||
                  (std::sscanf(sty,"%f%c%c",&dy,&sepy,&end)==2 && sepy=='%') ||
                  (std::sscanf(sty,"[%d%c%c",&indy,&sepy,&end)==2 && sepy==']')) &&
                 (!*stz || std::sscanf(stz,"%f%c",&dz,&end)==1 ||
                  (std::sscanf(stz,"%f%c%c",&dz,&sepz,&end)==2 && sepz=='%') ||
                  (std::sscanf(stz,"[%d%c%c",&indz,&sepz,&end)==2 && sepz==']')) &&
                 (!*stv || std::sscanf(stv,"%f%c",&dv,&end)==1 ||
                  (std::sscanf(stv,"%f%c%c",&dv,&sepv,&end)==2 && sepv=='%') ||
                  (std::sscanf(stv,"[%d%c%c",&indv,&sepv,&end)==2 && sepv==']'))) { // New image with specified dimensions and values.

        if (indx!=no_ind) { gmic_check_indice(indx,"Input image at position%s"); dx = (float)images[indx].dimx(); sepx = 0; }
        if (indy!=no_ind) { gmic_check_indice(indy,"Input image at position%s"); dy = (float)images[indy].dimy(); sepy = 0; }
        if (indz!=no_ind) { gmic_check_indice(indz,"Input image at position%s"); dz = (float)images[indz].dimz(); sepz = 0; }
        if (indv!=no_ind) { gmic_check_indice(indv,"Input image at position%s"); dv = (float)images[indv].dimv(); sepv = 0; }
        int idx = (int)dx, idy = (int)dy, idz = (int)dz, idv = (int)dv;
        if (sepx=='%') { idx = (int)cimg::round(dx*last_image.dimx()/100,1); if (!idx) ++idx; }
        if (sepy=='%') { idy = (int)cimg::round(dy*last_image.dimy()/100,1); if (!idy) ++idy; }
        if (sepz=='%') { idz = (int)cimg::round(dz*last_image.dimz()/100,1); if (!idz) ++idz; }
        if (sepv=='%') { idv = (int)cimg::round(dv*last_image.dimv()/100,1); if (!idv) ++idv; }
        if (idx<=0 || idy<=0 || idz<=0 || idv<=0)
          error(images,"Input image at position%s : Invalid image dimensions %dx%dx%dx%d.",gmic_inds,idx,idy,idz,idv);
        if (*st_values) print(images,"Input image at position%s, with values '%s'",gmic_inds,st_values);
        else print(images,"Input black image at position%s",gmic_inds);
        CImg<T> new_image(idx,idy,idz,idv,0);
        if (*st_values) {
          const unsigned int l = std::strlen(st_values);
          if (*st_values=='{' && st_values[l-1]=='}') {
            st_values[l-1] = 0;
            cimg::strclean(st_values+1);
            new_image.fill(st_values+1,true); }
          else new_image.fill(st_values,true);
        }
        new_image.transfer_to(input_images);
        filenames.insert(input_images.size,CImg<char>("(gmic)",7,1,1,1,false));
      } else if (*argument=='(' && argument[std::strlen(argument)-1]==')' &&
                 std::sscanf(argument,"(%4095[^\n]",stx)==1) { // New IxJxKxL image specified as array.
        stx[std::strlen(stx)-1] = 0;
        unsigned int cx = 0, cy = 0, cz = 0, cv = 0, maxcx = 0, maxcy = 0, maxcz = 0;
        const char *nargument = 0;
        for (nargument = stx; *nargument; ) {
          char s_value[256] = { 0 }, separator = 0; double value = 0;
          if (std::sscanf(nargument,"%255[0-9.eE+-]%c",s_value,&separator)>0 &&
              std::sscanf(s_value,"%lf%c",&value,&end)==1) {
            if (cx>maxcx) maxcx = cx;
            if (cy>maxcy) maxcy = cy;
            if (cz>maxcz) maxcz = cz;
            switch (separator) {
            case '^' : cx = cy = cz = 0; ++cv; break;
            case '/' : cx = cy = 0; ++cz; break;
            case ';' : cx = 0; ++cy; break;
            default : ++cx;
            }
            nargument+=std::strlen(s_value) + (separator?1:0);
          } else break;
        }
        if (*nargument) error(images,"Input image at position%s : Invalid input string '%s'.",gmic_inds,argument_text);

        CImg<T> img(maxcx+1,maxcy+1,maxcz+1,cv+1,0);
        cx = cy = cz = cv = 0;
        for (nargument = stx; *nargument; ) {
          char s_value[256] = { 0 }, separator = 0; double value = 0;
          if (std::sscanf(nargument,"%255[0-9.eE+-]%c",s_value,&separator)>0 &&
              std::sscanf(s_value,"%lf%c",&value,&end)==1) {
            img(cx,cy,cz,cv) = (T)value;
            switch (separator) {
            case '^' : cx = cy = cz = 0; ++cv; break;
            case '/' : cx = cy = 0; ++cz; break;
            case ';' : cx = 0; ++cy; break;
            default : ++cx;
            }
            nargument+=std::strlen(s_value) + (separator?1:0);
          } else break;
        }
        print(images,"Input image at position%s, with values '%s'",gmic_inds,argument_text);
        input_images.insert(img); filenames.insert(CImg<char>("(gmic)",7,1,1,1,false));
      } else { // Filename

        char filename[4096] = { 0 }, options[4096] = { 0 };
        if (argument[0]!='-' || (argument[1] && argument[1]!='.')) {
          std::FILE *file = std::fopen(argument,"r");
          if (file) { std::fclose(file); std::strcpy(filename,argument); }
          else {
            std::sscanf(argument,"%4095[^,],%s",filename,options);
            if (!(file=std::fopen(filename,"r"))) {
              if (filename[0]=='-') error(images,"Command '%s' on image%s : Command not found.",filename,gmic_inds);
              else error(images,"Input '%s' at position%s : File not found.",filename,gmic_inds);
            }
            std::fclose(file);
          }
        } else std::strcpy(filename,argument);

        const char *const basename = cimg::basename(filename), *const ext = cimg::split_filename(filename);
        if (!cimg::strcasecmp("off",ext)) {

          // 3D object file.
          print(images,"Input 3D object '%s' at position%s",is_fullpath?filename:basename,gmic_inds);
          CImgList<unsigned int> primitives3d;
          CImgList<unsigned char> colors3d;
          CImg<float> opacities3d, points3d = CImg<float>::get_load_off(filename,primitives3d,colors3d);
          opacities3d.assign(1,primitives3d.size,1,1,1);
          points3d.object3dtoCImg3d(primitives3d,colors3d,opacities3d);
          points3d.transfer_to(input_images);
          input_filenames.insert(CImg<char>(is_fullpath?filename:basename,
                                            std::strlen(is_fullpath?filename:basename)+1,1,1,1,false));
          obj3d = true;
        } else if (!cimg::strcasecmp(ext,"avi") ||
                   !cimg::strcasecmp(ext,"mov") ||
                   !cimg::strcasecmp(ext,"asf") ||
                   !cimg::strcasecmp(ext,"divx") ||
                   !cimg::strcasecmp(ext,"flv") ||
                   !cimg::strcasecmp(ext,"mpg") ||
                   !cimg::strcasecmp(ext,"m1v") ||
                   !cimg::strcasecmp(ext,"m2v") ||
                   !cimg::strcasecmp(ext,"m4v") ||
                   !cimg::strcasecmp(ext,"mjp") ||
                   !cimg::strcasecmp(ext,"mkv") ||
                   !cimg::strcasecmp(ext,"mpe") ||
                   !cimg::strcasecmp(ext,"movie") ||
                   !cimg::strcasecmp(ext,"ogm") ||
                   !cimg::strcasecmp(ext,"qt") ||
                   !cimg::strcasecmp(ext,"rm") ||
                   !cimg::strcasecmp(ext,"vob") ||
                   !cimg::strcasecmp(ext,"wmv") ||
                   !cimg::strcasecmp(ext,"xvid") ||
                   !cimg::strcasecmp(ext,"mpeg")) {

          // Image sequence file.
          unsigned int value0 = 0, value1 = 0, step = 1; char sep0 = 0, sep1 = 0, end = 0;
          if ((std::sscanf(options,"%u%c,%u%c,%u%c",&value0,&sep0,&value1,&sep1,&step,&end)==5 && sep0=='%' && sep1=='%') ||
              (std::sscanf(options,"%u%c,%u,%u%c",&value0,&sep0,&value1,&step,&end)==4 && sep0=='%') ||
              (std::sscanf(options,"%u,%u%c,%u%c",&value0,&value1,&sep1,&step,&end)==4 && sep1=='%') ||
              (std::sscanf(options,"%u,%u,%u%c",&value0,&value1,&step,&end)==3) ||
              (std::sscanf(options,"%u%c,%u%c%c",&value0,&sep0,&value1,&sep1,&end)==4 && sep0=='%' && sep1=='%') ||
              (std::sscanf(options,"%u%c,%u%c",&value0,&sep0,&value1,&end)==3 && sep0=='%') ||
              (std::sscanf(options,"%u,%u%c%c",&value0,&value1,&sep1,&end)==3 && sep1=='%') ||
              (std::sscanf(options,"%u,%u%c",&value0,&value1,&end)==2)) { // Read several frames
            print(images,"Input frames %u%s...%u%s with step %u of file '%s' at position%s",
                  value0,sep0=='%'?"%":"",value1,sep1=='%'?"%":"",step,is_fullpath?filename:basename,gmic_inds);
            if (sep0=='%' || sep1=='%') {
              const unsigned int nb_frames = CImg<unsigned int>::get_load_ffmpeg(filename,0,0,0)[0];
              if (sep0=='%') value0 = (unsigned int)cimg::round(value0*nb_frames/100,1);
              if (sep1=='%') value1 = (unsigned int)cimg::round(value1*nb_frames/100,1);
            }
          } else if ((std::sscanf(options,"%u%c%c",&value0,&sep0,&end)==2 && sep0=='%') ||
                     (std::sscanf(options,"%u%c",&value0,&end)==1)) { // Read one frame
            print(images,"Input frame %u%s of file '%s' at position%s",value0,sep0=='%'?"%":"",is_fullpath?filename:basename,gmic_inds);
            if (sep0=='%') {
              const unsigned int nb_frames = CImg<unsigned int>::get_load_ffmpeg(filename,0,0,0)[0];
              value0 = (unsigned int)cimg::round(value0*nb_frames/100,1);
            }
            value1 = value0; step = 1;
          } else { // Read all frames
            print(images,"Input all frames of file '%s' at position%s",is_fullpath?filename:basename,gmic_inds);
            value0 = 0; value1 = ~0U; sep0 = sep1 = 0; step = 1;
          }
          input_images.load_ffmpeg(filename,value0,value1,step);
          if (input_images)
            input_filenames.insert(input_images.size,CImg<char>(is_fullpath?filename:basename,
                                                                std::strlen(is_fullpath?filename:basename)+1,1,1,1,false));
        } else if (!cimg::strcasecmp("raw",ext)) {

          // Raw file.
          int dx = 0, dy = 1, dz = 1, dv = 1;
          if (std::sscanf(options,"%d,%d,%d,%d",&dx,&dy,&dz,&dv)>0) {
            if (dx<=0 || dy<=0 || dz<=0 || dv<=0)
              error(images,"Input raw file '%s' : Invalid specified dimensions %dx%dx%dx%d.",filename,dx,dy,dz,dv);
            print(images,"Input raw file '%s' at position%s",is_fullpath?filename:basename,gmic_inds);
            CImg<T>::get_load_raw(filename,dx,dy,dz,dv).transfer_to(input_images);
            input_filenames.insert(CImg<char>(is_fullpath?filename:basename,
                                              std::strlen(is_fullpath?filename:basename)+1,1,1,1,false));
          } else error(images,"Input raw file '%s' at position%s : Image dimensions must be specified.",filename,gmic_inds);
        } else if (!cimg::strcasecmp("yuv",ext)) {

          // YUV file.
          int dx = 0, dy = 0; unsigned int first = 0, last = ~0U, step = 1;
          if (std::sscanf(options,"%d,%d,%u,%u,%u",&dx,&dy,&first,&last,&step)>0) {
            if (dx<=0 || dy<=0)
              error(images,"Input yuv file '%s' at position%s : Invalid specified dimensions %dx%d.",filename,gmic_inds,dx,dy);
            print(images,"Input yuv file '%s' at position%s",is_fullpath?filename:basename,gmic_inds);
            input_images.load_yuv(filename,dx,dy,first,last,step);
            input_filenames.insert(input_images.size,CImg<char>(is_fullpath?filename:basename,
                                                                std::strlen(is_fullpath?filename:basename)+1,1,1,1,false));
          } else error(images,"Input yuv file '%s' at position%s : Image dimensions must be specified.",filename,gmic_inds);
        } else if (!cimg::strcasecmp("gmic",ext)) {

          // G'MIC macro file
          print(images,"Load macro file '%s'",is_fullpath?filename:basename);
          const unsigned int siz = macros.size;
          std::FILE *const file = cimg::fopen(argument,"r");
          add_macros(file);
          cimg::fclose(file);
          if (verbosity_level>=0) {
            std::fprintf(cimg_stdout," (%u macros added).",macros.size-siz);
            std::fflush(cimg_stdout);
          }
          continue;
        } else {

          // Other file type.
          print(images,"Input file '%s' at position%s",is_fullpath?filename:basename,gmic_inds);
          input_images.load(filename);
          input_filenames.insert(input_images.size,
                                 CImg<char>(is_fullpath?filename:basename,
                                            std::strlen(is_fullpath?filename:basename)+1,1,1,1,false));
        }
      }

      if (verbosity_level>=0) {
        if (input_images) {
          const unsigned int last = input_images.size-1;
          if (obj3d) {
            std::fprintf(cimg_stdout," (%d points, %u primitives, %u colors).",
                         (unsigned int)input_images(0,6),
                         (unsigned int)input_images(0,7),
                         (unsigned int)input_images(0,8));
            std::fflush(cimg_stdout);
          } else if (input_images.size==1) {
            std::fprintf(cimg_stdout," (1 image %ux%ux%ux%u).",
                         input_images[0].width,input_images[0].height,input_images[0].depth,
                         input_images[0].dim);
            std::fflush(cimg_stdout);
          } else {
            std::fprintf(cimg_stdout," (%u images [0] = %ux%ux%ux%u, %s[%u] = %ux%ux%ux%u).",
                         input_images.size,
                         input_images[0].width,input_images[0].height,input_images[0].depth,
                         input_images[0].dim,
                         last==1?"":"...,",
                         last,
                         input_images[last].width,input_images[last].height,input_images[last].depth,
                         input_images[last].dim);
            std::fflush(cimg_stdout);
          }
        } else {
          std::fprintf(cimg_stdout," (no available data).");
          std::fflush(cimg_stdout);
        }
      }

      for (unsigned int l = 0, siz = indices.size()-1, off = 0; l<=siz; ++l) {
        const unsigned int ind = indices[l] + off;
        if (l!=siz) images.insert(input_images,ind);
        else {
          images.insert(input_images.size,ind);
          cimglist_for(input_images,k) images[ind+k].swap(input_images[k]);
        }
        filenames.insert(input_filenames,ind);
        off+=input_images.size;
      }

    } catch (CImgException &e) {
      const char *error_message = e.message;
      char tmp[4096] = { 0 }, sep = 0;
      if (std::sscanf(error_message,"%4095[^>]>:%c",tmp,&sep)==2 && sep==':') error_message+=std::strlen(tmp)+3;
      error(images,error_message);
    }
  }

  // Post-checking.
  if (command_line.size>=command_line_maxsize)
    error(images,"Command line overflow : There are too much instructions specified on the command line.");
  if (dowhiles)
    warning(images,"A '-while' directive is missing somewhere.");
  if (repeatdones)
    warning(images,"A '-done' directive is missing somewhere.");
  if (locals)
    warning(images,"A '-endlocal' directive is missing somewhere.");

  // Display final result if necessary (not 'released' before).
  if (initial_call && !is_end) {
    if (images.size && !is_released) {
      if (!display_objects3d(images,filenames,CImg<unsigned int>::sequence(images.size,0,images.size-1),false))
        display_images(images,filenames,CImg<unsigned int>::sequence(images.size,0,images.size-1),true);
    }
    print(images,"End G'MIC instance.\n");
    is_end = true;
  }
  return *this;
}

// Small hack to separate the compilation of G'MIC in different pixel types.
// (only intended to save computer memory when compiling !)
//--------------------------------------------------------------------------
#ifdef gmic_minimal
gmic& gmic::parse_float(const CImgList<char>& command_line, unsigned int& position,CImgList<float>& images, CImgList<char>& filenames,
                        CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones,
                        CImgList<unsigned int>& locals, const bool initial_call) {
  return parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,initial_call);
}
template gmic::gmic(const int, const char *const *const, CImgList<float>&,
                    const char *const custom_macros, const bool default_macros);
template gmic::gmic(const char* const, CImgList<float>&,
                    const char *const custom_macros, const bool default_macros);
#else
#if defined(gmic_bool) || !defined(gmic_separate_compilation)
gmic& gmic::parse_bool(const CImgList<char>& command_line, unsigned int& position, CImgList<bool>& images, CImgList<char>& filenames,
                       CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones,
                       CImgList<unsigned int>& locals, const bool initial_call) {
  return parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,initial_call);
}
template gmic::gmic(const int, const char *const *const, CImgList<bool>&,
                    const char *const custom_macros, const bool default_macros);
template gmic::gmic(const char* const, CImgList<bool>&,
                    const char *const custom_macros, const bool default_macros);
#endif
#if defined(gmic_uchar) || !defined(gmic_separate_compilation)
gmic& gmic::parse_uchar(const CImgList<char>& command_line, unsigned int& position, CImgList<unsigned char>& images, CImgList<char>& filenames,
                        CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones,
                        CImgList<unsigned int>& locals, const bool initial_call) {
  return parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,initial_call);
}
template gmic::gmic(const int, const char *const *const, CImgList<unsigned char>&,
                    const char *const custom_macros, const bool default_macros);
template gmic::gmic(const char* const, CImgList<unsigned char>&,
                    const char *const custom_macros, const bool default_macros);
#endif
#if defined(gmic_char) || !defined(gmic_separate_compilation)
gmic& gmic::parse_char(const CImgList<char>& command_line, unsigned int& position, CImgList<char>& images, CImgList<char>& filenames,
                       CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones,
                       CImgList<unsigned int>& locals, const bool initial_call) {
  return parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,initial_call);
}
template gmic::gmic(const int, const char *const *const, CImgList<char>&,
                    const char *const custom_macros, const bool default_macros);
template gmic::gmic(const char* const, CImgList<char>&,
                    const char *const custom_macros, const bool default_macros);
#endif
#if defined(gmic_ushort) || !defined(gmic_separate_compilation)
gmic& gmic::parse_ushort(const CImgList<char>& command_line, unsigned int& position, CImgList<unsigned short>& images, CImgList<char>& filenames,
                         CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones,
                         CImgList<unsigned int>& locals, const bool initial_call) {
  return parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,initial_call);
}
template gmic::gmic(const int, const char *const *const, CImgList<unsigned short>&,
                    const char *const custom_macros, const bool default_macros);
template gmic::gmic(const char* const, CImgList<unsigned short>&,
                    const char *const custom_macros, const bool default_macros);
#endif
#if defined(gmic_short) || !defined(gmic_separate_compilation)
gmic& gmic::parse_short(const CImgList<char>& command_line, unsigned int& position, CImgList<short>& images, CImgList<char>& filenames,
                        CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones,
                        CImgList<unsigned int>& locals, const bool initial_call) {
  return parse(command_line,position,images,filenames,dowhiles,repeatdones,locals, initial_call);
}
template gmic::gmic(const int, const char *const *const, CImgList<short>&,
                    const char *const custom_macros, const bool default_macros);
template gmic::gmic(const char* const, CImgList<short>&,
                    const char *const custom_macros, const bool default_macros);
#endif
#if defined(gmic_uint) || !defined(gmic_separate_compilation)
gmic& gmic::parse_uint(const CImgList<char>& command_line, unsigned int& position, CImgList<unsigned int>& images, CImgList<char>& filenames,
                       CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones,
                       CImgList<unsigned int>& locals, const bool initial_call) {
  return parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,initial_call);
}
template gmic::gmic(const int, const char *const *const, CImgList<unsigned int>&,
                    const char *const custom_macros, const bool default_macros);
template gmic::gmic(const char* const, CImgList<unsigned int>&,
                    const char *const custom_macros, const bool default_macros);
#endif
#if defined(gmic_int) || !defined(gmic_separate_compilation)
gmic& gmic::parse_int(const CImgList<char>& command_line, unsigned int& position, CImgList<int>& images, CImgList<char>& filenames,
                      CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones,
                      CImgList<unsigned int>& locals, const bool initial_call) {
  return parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,initial_call);
}
template gmic::gmic(const int, const char *const *const, CImgList<int>&,
                    const char *const custom_macros, const bool default_macros);
template gmic::gmic(const char* const, CImgList<int>&,
                    const char *const custom_macros, const bool default_macros);
#endif
#if defined(gmic_float) || !defined(gmic_separate_compilation)
gmic& gmic::parse_float(const CImgList<char>& command_line, unsigned int& position, CImgList<float>& images, CImgList<char>& filenames,
                        CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones,
                        CImgList<unsigned int>& locals, const bool initial_call) {
  return parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,initial_call);
}
template gmic::gmic(const int, const char *const *const, CImgList<float>&,
                    const char *const custom_macros, const bool default_macros);
template gmic::gmic(const char* const, CImgList<float>&,
                    const char *const custom_macros, const bool default_macros);
#endif
#if defined(gmic_double) || !defined(gmic_separate_compilation)
gmic& gmic::parse_double(const CImgList<char>& command_line, unsigned int& position, CImgList<double>& images, CImgList<char>& filenames,
                         CImgList<unsigned int>& dowhiles, CImgList<unsigned int>& repeatdones,
                         CImgList<unsigned int>& locals, const bool initial_call) {
  return parse(command_line,position,images,filenames,dowhiles,repeatdones,locals,initial_call);
}
template gmic::gmic(const int, const char *const *const, CImgList<double>&,
                    const char *const custom_macros, const bool default_macros);
template gmic::gmic(const char* const, CImgList<double>&,
                    const char *const custom_macros, const bool default_macros);
#endif
#endif
#endif

//-----------------------
// Start main procedure.
//-----------------------
#if defined(gmic_main) || (!defined(gmic_separate_compilation) && !defined(gmic_minimal))
extern char data_gmic_def[];

int main(int argc, char **argv) {

  // Display help if necessary.
  //---------------------------
  if (argc==1) {
    std::fprintf(cimg_stdout,"<gmic> No options or data provided. Try '%s -h' for help.\n",cimg::basename(argv[0]));
    std::fflush(cimg_stdout);
    std::exit(0);
  }

  if (cimg_option("-h",false,0) || cimg_option("-help",false,0) || cimg_option("--help",false,0)) {
    cimg_usage("GREYC's Magic Image Converter");

    char version[1024] = { 0 };
    std::sprintf(version,"        Version %d.%d.%d.%d, Copyright (C) 2008-2009, David Tschumperle (http://gmic.sourceforge.net)",
                 gmic_version/1000,(gmic_version/100)%10,(gmic_version/10)%10,gmic_version%10);
    cimg_help(version);

    cimg_help("\n  Usage\n"
              "  -----");
    cimg_help("  gmic [file_1 | command_1] .. [file_n | command_n]\n");
    cimg_help("  G'MIC defines a script-based language for image processing, able to convert, manipulate and");
    cimg_help("  visualize generic 1D/2D/3D multi-spectral image and video files. It follows these simple rules :\n");
    cimg_help("    - G'MIC handles a numbered list of images which are all stored in computer memory.");
    cimg_help("    - Pixels of all stored images have the same datatype which can be");
    cimg_help("      'uchar', 'char', 'ushort', 'short', 'uint', 'int', 'float' or double'.");
    cimg_help("    - The first image of the list has indice '[0]'.");
    cimg_help("    - Negative indices are treated in a cyclic way (i.e. '[-1]' is the last image,");
    cimg_help("      '[-2]' the penultimate one, and so on...).");
    cimg_help("    - Command line items tell how to add/remove/manipulate/display images of the list.");
    cimg_help("    - Items are read and executed in the order they appear on the command line, from the left to the right.");
    cimg_help("    - Items can thus appear more than one time on the command line.");
    cimg_help("    - An item starting by '-' usually designates a G'MIC instruction.");
    cimg_help("    - One single instruction may have two equivalent names (regular and short).");
    cimg_help("    - A G'MIC instruction may have mandatory or optional arguments.");
    cimg_help("    - When arguments are required, they must be separated by commas ','.");
    cimg_help("    - Items that are not instructions are considered either as input filenames or input strings.");
    cimg_help("    - When an input filename is found, the corresponding image data are loaded");
    cimg_help("      and added to the end of the image list.");
    cimg_help("      (see section 'Filename options' below for more informations on file input/output).");
    cimg_help("    - Special filenames '-' and '-.ext' mean 'standard input/output' (optionally. in 'ext' format).");
    cimg_help("    - Special input strings can be also used to insert new images to the list. They can be :");
    cimg_help("        - 'width[%][,height[%][,depth[%][,dim[%][,values]]]]' : Insert new image with specified dimensions and values.");
    cimg_help("          (adding '%' to a dimension means 'percentage of the same dimension in the last image'),");
    cimg_help("        - '[indice]' or '[indice],N' : Insert 1 or N copies of an existing image [indice].");
    cimg_help("        - '(v1,v2,...)' : Insert new image with given values.");
    cimg_help("          Separators inside '(..)' can be ',' (column), ';' (line), '/' (slice) or '^' (channel).");
    cimg_help("    - A G'MIC instruction may be restricted to a specific subset of the image list, by adding '[subset]' to");
    cimg_help("      the instruction name. Several expressions are possible for 'subset', for instance : ");
    cimg_help("        '-command[0,1,3]' : Apply instruction on images 0,1 and 3.");
    cimg_help("        '-command[3-5]' : Apply instruction on images 3 to 5.");
    cimg_help("        '-command[50%-100%] : Apply instruction on the second half of the image list.");
    cimg_help("        '-command[0,-2,-1]' : Apply instruction on the first and two latest images.");
    cimg_help("        '-command[0-9:3]' : Apply instruction on images 0 to 9, with a step of 3 (i.e. images 0,3,6,9).");
    cimg_help("        '-command[0,2-4,50%--1]' : Apply instruction on images 0,2,3,4 and the second half of the list.");
    cimg_help("    - When no image subset is specified, a G'MIC instruction is applied on all images of the list.");
    cimg_help("    - A G'MIC instruction starting with '--' instead of '-' does not act 'in-place' but");
    cimg_help("      insert its result as a new image, at the end of the list.");
    cimg_help("    - On the command line, expressions starting with '@' are substituted this way :");
    cimg_help("        - '@#' or '@{#}' is replaced by the current size of the images list.");
    cimg_help("        - '@!' or '@{!}' is replaced by 0 or 1, according to the visibility state of the instant window.");
    cimg_help("        - '@/', '@{/}' or '@{/,subset}' is replaced by the execution scope, or a subset of it.");
    cimg_help("        - '@>', '@{>}' or '@{>,subset}' is replaced by the indices (or a subset of them) of the executing");
    cimg_help("          'repeat..done' loops. Indices are in ascending order and start from 0.");
    cimg_help("        - '@<', '@{<}' or '@{<,subset}' does the same but indices are in descending order toward 0.");
    cimg_help("        - '@?', '@{?}', '@{?,max}' or '@{?,_min,_max,_N,_rounding}' are replaced by one or N random values");
    cimg_help("          between [0,1], [0,max] or [min,max], eventually rounded to the specific rounding value.");
    cimg_help("        - '@indice' or '@{indice,feature}' is replaced by all pixel values of the image '[indice]' or only a feature of it.");
    cimg_help("          'feature' specifies what kind of feature are retrieved from the image. It can be one of the choice below) :");
    cimg_help("            - 'w' : retrieve image width.");
    cimg_help("            - 'h' : retrieve image height.");
    cimg_help("            - 'd' : retrieve image depth.");
    cimg_help("            - 'c' : retrieve number of image channels.");
    cimg_help("            - '#' : retrieve number of image values (i.e. width x height x depth x channels).");
    cimg_help("            - 'm' : retrieve minimum pixel value.");
    cimg_help("            - 'M' : retrieve maximum pixel value.");
    cimg_help("            - 'a' : retrieve average of pixel values.");
    cimg_help("            - 's' : retrieve standard deviation of pixel values.");
    cimg_help("            - 'v' : retrieve variance of pixel values.");
    cimg_help("            - '-' : retrieve coordinates of the minimum pixel value.");
    cimg_help("            - '+' : retrieve coordinates of the maximum pixel value.");
    cimg_help("            - '(x[,y[,z[,v[,border_conds]]]])' : retrieve pixel value at a specified coordinates (x,y,z,v).");
    cimg_help("            - 'subset' : retrieve pixel values defined as an image subset.");
    cimg_help("    - On the command line, an expression as '{formula}' is replaced by the evaluation of the given formula.");
    cimg_help("    - Input filenames or commands may result to the generation of 3D objects.");
    cimg_help("    - A 3D object viewer is included in G'MIC. It uses slow but portable software rendering.");
    cimg_help("    - A 3D object is stored as a single-column image containing all object data, in the following order :");
    cimg_help("        { magic_numbers, vertices, faces, colors, opacities }.");
    cimg_help("    - Custom instructions recognized by G'MIC can be defined by the user, with the use of a macro file.");
    cimg_help("    - A macro file is a simple ASCII text file, each line starting by");
    cimg_help("        'instruction_name : substitution' or 'substitution (continuation)' or '# comment'.");
    cimg_help("    - Each invoked macro instruction is substituted as its defined content when encountered on the command line.");
    cimg_help("    - A default macro file 'gmic_def.raw' is distributed within the G'MIC package.");
    cimg_help("    - The macros defined in this macro file are already included by default in G'MIC.");
    cimg_help("    - Macros arguments can be added after the invokation of a macro instruction.");
    cimg_help("    - In macros definitions, expressions starting with '$' have special meanings :");
    cimg_help("        '$#' is replaced by the number of arguments of the invoked macro instruction.");
    cimg_help("        '$*' is replaced by all arguments of the invoked macro instruction.");
    cimg_help("        '$i' and '${i}' are replaced, if i>0, by the i-th argument of the invoked macro instruction.");
    cimg_help("        '${i*}' is replaced, if i>0, by all arguments whose positions are higher or equal to i.");
    cimg_help("        '${i=default}' is replaced by the value of $i (i>0) if defined, or by its new value 'default' else");
    cimg_help("                       ('default' can be also a $ expression).");
    cimg_help("        '$?' is replaced by a string describing targeted images (only for use in '-echo' descriptions).");
    cimg_help("\n  All recognized G'MIC default commands are listed below.");
    cimg_help("  A parameter specified in '[]' or starting by '_' is optional, except when standing for '[indices]' where it");
    cimg_help("  corresponds to one or several indices of the image list. In this case, the '[' and ']'");
    cimg_help("  characters are mandatory when writting the corresponding item.");

    cimg_help("\n  Global options\n"
              "  --------------");
    cimg_option("-help","(no args)","Display this help (eq. to '-h').");
    cimg_option("-verbose+","(no args)","Increment verbosity level (eq. to '-v+').");
    cimg_option("-verbose-","(no args)","Decrement verbosity level (eq. to '-v-').");
    cimg_option("-macros","'filename'","Load macro file from specified filename (eq. to '-m').");
    cimg_option("-debug","(no args)","Switch debug flag (when on, displays internal debugging messages).");
    cimg_option("-trace","(no args)","Switch instructions trace for specified indices.");
    cimg_option("-fullpath","(no args)","Switch full path flag (when on, displays full filename paths).");

    cimg_help("\n  Arithmetic operators\n"
              "  --------------------");
    cimg_option("-add","'value', ''formula'', '[indice]' or (no args)","Add a 'value' or '[indice]' to image(s)");
    cimg_help("                                              "
              "or add images together (eq. to '-+').");
    cimg_option("-sub","'value', ''formula'', '[indice]' or (no args)","Substract a 'value' or '[indice]' to image(s)");
    cimg_help("                                              "
              "or substract images together (eq. to '--').");
    cimg_option("-mul","'value', ''formula'', '[indice]' or (no args)","Multiply image(s) by a 'value' or '[indice]'");
    cimg_help("                                              "
              "or multiply images together (eq. to '-*').");
    cimg_option("-div","'value', ''formula'', '[indice]' or (no args)","Divide image(s) by a 'value' or '[indice]'");
    cimg_help("                                              "
              "or divide images together (eq. to '-/').");
    cimg_option("-pow","'value', ''formula'', '[indice]' or (no args)","Compute image(s) to the power of a 'value' or '[indice]'");
    cimg_help("                                              "
              "or compute power of the images together (eq. to '-^').");
    cimg_option("-min","'value', ''formula'', '[indice]' or (no args)","Compute minimum between image(s) and a 'value' or '[indice]'");
    cimg_help("                                              "
              "or compute minimum of images together.");
    cimg_option("-max","'value', ''formula'', '[indice]' or (no args)","Compute maximum between image(s) and a 'value' or '[indice]'");
    cimg_help("                                              "
              "or compute maximum of images together.");
    cimg_option("-mod","'value', ''formula'', '[indice]' or (no args)","Compute modulo of image(s) with a 'value' or '[indice]'");
    cimg_help("                                              "
              "or compute modulo of images together.");
    cimg_option("-and","'value', ''formula'', '[indice]' or (no args)","Compute bitwise AND of image(s) with a 'value' or '[indice]'");
    cimg_help("                                              "
              "or compute bitwise AND of images together.");
    cimg_option("-or","'value', ''formula'', '[indice]' or (no args)","Compute bitwise OR of image(s) with a 'value' or '[indice]'");
    cimg_help("                                              "
              "or compute bitwise OR of images together.");
    cimg_option("-xor","'value', ''formula'', '[indice]' or (no args)","Compute bitwise XOR of image(s) with a 'value' '[indice]'");
    cimg_help("                                              "
              "or compute bitwise XOR of images together.");
    cimg_option("-cos","(no args)","Compute cosine of image values.");
    cimg_option("-sin","(no args)","Compute sine of image values.");
    cimg_option("-tan","(no args)","Compute tangent of image values.");
    cimg_option("-acos","(no args)","Compute arc-cosine of image values.");
    cimg_option("-asin","(no args)","Compute arc-sine of image values.");
    cimg_option("-atan","(no args)","Compute arc-tangent of image values.");
    cimg_option("-atan2","'[indice]'","Compute oriented arc-tangent of image values.");
    cimg_option("-abs","(no args)","Compute absolute value of image values.");
    cimg_option("-sqr","(no args)","Compute square of image values.");
    cimg_option("-sqrt","(no args)","Compute square root of image values.");
    cimg_option("-exp","(no args)","Compute exponential of image values.");
    cimg_option("-log","(no args)","Compute logarithm of image values.");
    cimg_option("-log10","(no args)","Compute logarithm_10 of image values.");

    cimg_help("\n  Pointwise pixel manipulation\n"
              "  ----------------------------");
    cimg_option("-type","'value_type'","Cast all images into a value type (eq. to '-t').");
    cimg_help("                                              "
              "('value_type' can be 'bool','uchar','char','ushort','short',");
    cimg_help("                                              "
              "'uint','int','float','double').");
    cimg_option("-set","'value,_x,_y,_z,_v'","Set scalar value in image(s) at given position (eq. to '-=').");
    cimg_option("-endian","(no args)","Invert endianness of image buffer(s).");
    cimg_option("-fill","'value1,value2,...' or 'formula'","Fill image(s) with list of values or mathematical formula (eq. to '-f').");
    cimg_option("-threshold","'value[%],_soft={0,1}' or (noargs)","Threshold pixel values ((noargs) for interactive mode).");
    cimg_option("-cut","'{value1[%],[indice]},{value2[%],[indice]}' or (noargs)","Cut pixel values ((noargs) for interactive mode).");
    cimg_option("-normalize","'{value1[%],[indice]},{value2[%],[indice]}'","Normalize pixel values (eq. to '-n').");
    cimg_option("-round","'rounding_value,_rounding_type={-1,0,1}'","Round pixel values.");
    cimg_option("-equalize","'nb_clusters[%]>0'","Equalize image histogram(s).");
    cimg_option("-quantize","'nb_levels>0'","Quantize image(s).");
    cimg_option("-noise","'std[%]>=0,_noise_type={0,1,2,3}'","Add noise to image(s)");
    cimg_help("                                              "
              "('noise_type' can be '{0=gaussian, 1=uniform, 2=salt&pepper, 3=poisson}'.");
    cimg_option("-rand","'valmin,valmax'","Fill image(s) with random values in given range.");
    cimg_option("-norm","(no args)","Compute pointwise L2-norm of multi-channel image(s).");
    cimg_option("-orientation","(no args)","Compute pointwise orientation of multi-channel image(s).");
    cimg_option("-index","'[indice],_dithering={0,1},_mapping={0,1}' or ","");
    cimg_help("              ");
    cimg_help("'predefined_palette={0,1,2},_dithering={0,1},_mapping={0,1}'    Index vector-values image(s) by palette");
    cimg_help("                                              "
              "('predefined_palette' can be '{0=default, 1=rainbow, 2=contrast}').");

    cimg_option("-map","'[indice]' or 'predefined_palette'","Map palette to indexed image(s)");
    cimg_help("                                              "
              "('predefined_palette' can be '{0=default, 1=rainbow, 2=contrast}'.");

    cimg_help("\n  Color bases conversions\n"
              "  -----------------------");
    cimg_option("-rgb2hsv","(no args)","Convert image(s) from RGB to HSV colorbases.");
    cimg_option("-rgb2hsl","(no args)","Convert image(s) from RGB to HSL colorbases.");
    cimg_option("-rgb2hsi","(no args)","Convert image(s) from RGB to HSI colorbases.");
    cimg_option("-rgb2yuv","(no args)","Convert image(s) from RGB to YUV colorbases.");
    cimg_option("-rgb2ycbcr","(no args)","Convert image(s) from RGB to YCbCr colorbases.");
    cimg_option("-rgb2xyz","(no args)","Convert image(s) from RGB to XYZ colorbases.");
    cimg_option("-rgb2lab","(no args)","Convert image(s) from RGB to Lab colorbases.");
    cimg_option("-rgb2cmy","(no args)","Convert image(s) from RGB to CMY colorbases.");
    cimg_option("-rgb2cmyk","(no args)","Convert image(s) from RGB to CMYK colorbases.");
    cimg_option("-hsv2rgb","(no args)","Convert image(s) from HSV to RGB colorbases.");
    cimg_option("-hsl2rgb","(no args)","Convert image(s) from HSL to RGB colorbases.");
    cimg_option("-hsi2rgb","(no args)","Convert image(s) from HSI to RGB colorbases.");
    cimg_option("-yuv2rgb","(no args)","Convert image(s) from YUV to RGB colorbases.");
    cimg_option("-ycbcr2rgb","(no args)","Convert image(s) from YCbCr to RGB colorbases.");
    cimg_option("-xyz2rgb","(no args)","Convert image(s) from XYZ to RGB colorbases.");
    cimg_option("-lab2rgb","(no args)","Convert image(s) from Lab to RGB colorbases.");
    cimg_option("-cmy2rgb","(no args)","Convert image(s) from CMY to RGB colorbases.");
    cimg_option("-cmyk2rgb","(no args)","Convert image(s) from CMYK to RGB colorbases.");

    cimg_help("\n  Geometric manipulation\n"
              "  ----------------------");
    cimg_option("-resize","'[indice],_interpolation={-1,0,1,2,3,4,5},_borders={-1,0,1,2},_center={0,1}' or ","");
    cimg_help("                     "
              "'{[indice],width[%]},_{[indice],height[%]},_{[indice],depth[%]},_{[indice],dim[%]},_interpolation,_borders,_center'");
    cimg_help("                                              "
              "or (noargs)");
    cimg_help("                                              "
              "Resize image(s) according to a given geometry ((noargs) for interactive mode) (eq. to '-r')");
    cimg_help("                                              "
              "('interpolation' can be '{0=none, 1=nearest, 2=average, 3=linear, 4=grid, 5=cubic}').");
    cimg_option("-resize2x","(no args)","Resize image(s) with the Scale2x algorithm.");
    cimg_option("-resize3x","(no args)","Resize image(s) with the Scale3x algorithm.");
    cimg_option("-crop","'x0[%],x1[%],_border_conditions={0,1}' or 'x0[%],y0[%],x1[%],y1[%],_border_conditions' or ","");
    cimg_help("                                              "
              "'x0[%],y0[%],z0[%],x1[%],y1[%],z1[%],_border_conditions' or ");
    cimg_help("                                              "
              "'x0[%],y0[%],z0[%],v0[%],x1[%],y1[%],z1[%],v1[%],_border_conditions' or (noargs).");
    cimg_help("                                              "
              "Crop image(s) ((noargs) for interactive mode) (eq. to '-c') ");
    cimg_help("                                              "
              "('border_conditions' can be '{0=zero, 1=nearest}')");
    cimg_help("                                              "
              "((no args) for interactive mode).");
    cimg_option("-autocrop","'value1,value2,...'","Autocrop image(s) using a background color.");
    cimg_option("-channels","'{[ind0],v0[%]},_{[ind1],v1[%]}'","Select channels v0..v1 of multi-channel image(s).");
    cimg_option("-slices","'{[ind0],z0[%]},_{[ind1],z1[%]}'","Select slices z0..z1 of volumetric image(s).");
    cimg_option("-lines","'{[ind0],y0[%]},_{[ind1],y1[%]}'","Select lines y0..y1 of image(s).");
    cimg_option("-columns","'{[ind0],x0[%]},_{[ind1],x1[%]}'","Select columns x0..x1 of image(s).");
    cimg_option("-rotate","'angle,_border_conditions={0,1,2},_interpolation={0,1,2}'","Rotate image(s) with a given angle");
    cimg_help("                                              "
              "('border_conditions' can be '{-3=cyclic (in-place), -2=nearest(ip), -1=zero(ip),");
    cimg_help("                                              "
              " 0=zero, 1=nearest, 2=cyclic}'");
    cimg_help("                                              "
              "and 'interpolation' can be '{0=none, 1=linear, 2=cubic}').");
    cimg_option("-mirror","'axis={x,y,z,v}'",
                "Mirror image(s) along a given axis.");
    cimg_option("-translate","'tx[%],_ty[%],_tz[%],_tv[%],_border_conditions={0,1,2}'",
                "Translate image(s) by vector (dx,dy,dz,dv)");
    cimg_help("                                              "
              "('border_conditions' can be '{0=zero, 1=nearest, 2=cyclic}').");
    cimg_option("-transpose","(no args)","Transpose image(s).");
    cimg_option("-invert","(no args)","Compute inverse image matrix(ces).");
    cimg_option("-permute","'permutation'","Permute image axes by a given permutation "
                "('permutation' can be 'yxzv',...).");
    cimg_option("-unroll","'axis={x,y,z,v}'",
                "Unroll image(s) along given axis.");
    cimg_option("-split","'axis={x,y,z,v},_nb_parts>=0' or 'value,_keep_value={0,1}'",
                "Split image(s) along given axis or value (eq. to '-s').");
    cimg_option("-append","'axis={x,y,z,v},_alignement={p,c,n}'","Append image(s) along given axis (eq. to '-a')");
    cimg_help("                                              "
              "('alignement' can be '{p=left, c=center, n=right)'.");
    cimg_option("-warp","'[indice],_relative={0,1},_interpolation={0,1},_border_conditions={0,1,2},_nb_frames>=1'",
                "Warp image(s) with given displacement field");
    cimg_help("                                              "
              "'border_conditions' can be '{0=zero, 1=nearest, 2=cyclic}').");

    cimg_help("\n  Image filtering\n"
              "  ---------------");
    cimg_option("-blur","'stdev>=0[%%],_border_conditions={0,1}'","Blur image(s) isotropically");
    cimg_help("                                              "
              "('border_conditions' can be '{0=zero, 1=nearest}').");
    cimg_option("-bilateral","'stdevs>=0[%%],stdevr>=0'",
                "Apply bilateral filtering.");
    cimg_option("-denoise","'stdevs>=0,_stdevp>=0,_patch_size>0,_lookup_size>0,_smoothness,_approx={0,1}'",
                "Denoise image(s) with a patch-averaging procedure.");
    cimg_option("-smooth",
                "'amplitude>=0,_sharpness>=0,_anisotropy=[0,1],_alpha,_sigma,_dl>0,_da>=0,_precision>0,_interpolation_type={0,1,2},_fast_approximation={0,1}'","");
    cimg_help("                  "
              "or 'nb_iters>=0,_sharpness>=0,_anisotropy=[0,1],_alpha,_sigma,_dt>0,0'");
    cimg_help("                  "
              "or '[ind],_amplitude>=0,_dl>0,_da>=0,_precision>0,_interpolation_type={0,1,2},_fast_approximation={0,1}'");
    cimg_help("                  "
              "or '[ind],_nb_iters>=0,_dt>0,0'");
    cimg_help("                                              "
              "Smooth image(s) anisotropically.");
    cimg_option("-edgetensors","'sharpness>=0,_anisotropy=[0,1],_alpha,_sigma,is_sqrt={0,1}'","");
    cimg_help("                                              "
              "Compute tensors for edge-preserving smoothing.");
    cimg_option("-median","'size>0'","Apply median filter.");
    cimg_option("-sharpen","'amplitude>=0' or 'amplitude>=0,1,_edge>=0,_alpha,_sigma'",
                "Sharpen image(s) with inverse diffusion or shock filters methods.");
    cimg_option("-convolve","'[indice],_border_conditions={0,1}'",
                "Convolve image(s) by a mask");
    cimg_help("                                              "
              "('border_conditions' can be '{0=zero, 1=nearest}').");
    cimg_option("-correlate","'[indice],_border_conditions={0,1}'",
                "Correlate image(s) by a mask (same parameters as above).");
    cimg_option("-erode","'size,_border_conditions={0,1}' or '[indice],_border_conditions'","");
    cimg_help("                                              "
              "Erode image(s) by a mask (same parameters as above)').");
    cimg_option("-dilate","'size,_border_conditions={0,1}' or '[indice],_border_conditions'","");
    cimg_help("                                              "
              "Dilate image(s) by a mask (same parameters as above).");
    cimg_option("-inpaint","'[ind]'","Inpaint image(s) by given inpainting mask.");
    cimg_option("-gradient","'{x,y,z}' or (no args)","Compute image gradient(s).");
    cimg_option("-hessian","'{xx,xy,xz,yy,yz,zz}' or (no args)","Compute image Hessian(s).");
    cimg_option("-fft","(no args)","Compute direct Fourier transform(s).");
    cimg_option("-ifft","(no args)","Compute inverse Fourier transform(s).");

    cimg_help("\n  Image creation and drawing\n"
              "  --------------------------");
    cimg_option("-histogram","'nb_levels[%]>0'","Compute image histogram(s).");
    cimg_option("-distance","'isovalue'","Compute distance function(s) to a given isovalue.");
    cimg_option("-hamilton","'nb_iter>=0,_band_size>=0'","Apply iterations of the Hamilton-Jacobi PDE to compute signed distance to 0.");
    cimg_option("-label","(no args)","Label connected components of image(s).");
    cimg_option("-displacement","'[indice],_smoothness>=0,_precision>0,_nbscales>=0,itermax>=0,backward={0,1}",
                "Estimate displacement field between images.");
    cimg_option("-sort","(no args)","Sort image values in increasing order.");
    cimg_option("-psnr","'_max_value'","Compute PSNR matrix between images.");
    cimg_option("-point","'x[%],y[%],_z[%],_opacity,_color'","Draw a 3D colored point on image(s).");
    cimg_option("-line","'x0[%],y0[%],x1[%],y1[%],_opacity,_color'","Draw a 2D colored line on image(s).");
    cimg_option("-polygon","'N,x1[%],y1[%],..,xN[%],yN[%],_opacity,_color'","Draw a 2D filled N-vertices polygon on image(s).");
    cimg_option("-ellipse","'x[%],y[%],r[%],R[%],_angle,_opacity,_color'","Draw a 2D colored ellipse on image(s).");
    cimg_option("-text","text,_x[%],_y[%],_size>0,_opacity,_color'", "Draw a text string on image(s).");
    cimg_option("-image","'[indice],_x[%],_y[%],_z[%],_opacity,_[ind_mask]'","Draw a sprite image on image(s).");
    cimg_option("-object3d","'[indice],_x[%],_y[%],_z,_opacity'","Draw a 3D object on image(s).");
    cimg_option("-plasma","'alpha,_beta,_opacity'","Draw a random plasma on image(s).");
    cimg_option("-mandelbrot","'z0r,z0i,z1r,z1i,_itermax>=0,_julia={0,1},_c0r,_c0i,_opacity'","Draw a Mandelbrot/Julia fractal on image(s).");
    cimg_option("-quiver","'[ind],_sampling>0,_factor,_type={0,1},_opacity,_color'","Draw a 2D vector field on image(s).");
    cimg_option("-flood","'x[%],_y[%],_z[%],_tolerance>=0,_opacity,_color'","Flood-fill image(s) using fill value and tolerance.");

    cimg_help("\n  List manipulation\n"
              "  -----------------");
    cimg_option("-remove","(no args)","Remove image(s) from list (eq. to '-rm').");
    cimg_option("-keep","(no args)","Keep only given image(s) in list (eq. to '-k').");
    cimg_option("-move","'position'","Move image(s) at given position (eq. to '-mv').");
    cimg_option("-reverse","(no args)","Reverse image(s) order in the list.");
    cimg_option("-name","\"name\"","Set display name of image(s).");

    cimg_help("\n  3D Rendering\n"
              "  ------------");
    cimg_option("-cube3d","'size'","Generate a 3D cube.");
    cimg_option("-cone3d","'radius,_height,_subdivisions>0'","Generate a 3D cone.");
    cimg_option("-cylinder3d","'radius,_height,_subdivisions>0'","Generate a 3D cylinder.");
    cimg_option("-torus3d","'radius1,_radius2,_subdivisions1>0,_subdivisions2>0'","Generate a 3D torus.");
    cimg_option("-plane3d","'sizex,_sizey,_subdivisionsx>0,_subdisivionsy>0'","Generate a 3D plane.");
    cimg_option("-sphere3d","'radius,_recursions>0'","Generate a 3D sphere.");
    cimg_option("-elevation3d","'z-factor' or '[indice]'","Generate a 3D elevation of image(s) using given factor or elevation map.");
    cimg_option("-isovalue3d","'value'","Generate 3D object(s) by retrieving isophote or isosurface from image(s).");
    cimg_option("-center3d","(no args)","Center 3D object(s) (eq. to '-c3d').");
    cimg_option("-normalize3d","(no args)","Normalize 3D object(s) to a unit size (eq. to '-n3d').");
    cimg_option("-rotate3d","'u,v,w,angle'","Rotate 3D object(s) around axis (u,v,w) with given angle (eq. to '-rot3d').");
    cimg_option("-add3d","'[indice]' or 'tx,_ty,_tz' or (noargs)","Append/translate 3D object(s) (eq. to '-+3d').");
    cimg_option("-sub3d","'tx,_ty,_tz'","Translate 3D object(s) with the opposite of the given vector (eq. to '--3d').");
    cimg_option("-mul3d","'fact' or 'factx,facty,_factz'","Scale 3D object(s) (eq. to '-*3d').");
    cimg_option("-div3d","'fact' or 'factx,facty,_factz'","Scale 3D object(s) with inverse factor (eq. to '-/3d').");
    cimg_option("-color3d","'R,G,B,_opacity'","Set color and opacity of 3D object(s) (eq. to '-col3d').");
    cimg_option("-opacity3d","'opacity'","Set opacity of 3D object(s) (eq. to '-opac3d').");
    cimg_option("-invert3d","(no args)","Invert primitive orientations of 3D object(s) (eq. to '-i3d').");
    cimg_option("-split3d","(no args)","Split 3D object(s) data into 6 data vectors");
    cimg_help("                                              "
              "'header,N,vertices,primitives,colors,opacities' (eq. to '-s3d').");
    cimg_option("-light3d","'posx,posy,posz'","Set 3D position of the light for 3D rendering (eq. to '-l3d').");
    cimg_option("-focale3d","'focale>0'","Set focale value for 3D rendering (eq. to '-f3d').");
    cimg_option("-specl3d","'value>=0'","Set amount of specular light for 3D rendering (eq. to '-sl3d').");
    cimg_option("-specs3d","'value>=0'","Set shininess of specular light for 3D rendering (eq. to '-ss3d').");
    cimg_option("-orient3d","(no args)","Switch double-sided mode for 3D rendering (eq. to '-o3d').");
    cimg_option("-render3d","'mode={-1,0,1,2,3}'","Set 3D rendering mode");
    cimg_help("                                              "
              "(can be '{-1=bounding-box, 0=pointwise, 1=linear, 2=flat, 3=flat-shaded,");
    cimg_help("                                              "
              "4=Gouraud-shaded, 5=Phong-shaded}') (eq. to '-r3d').");
    cimg_option("-renderd3d","'mode'","Set dynamic rendering mode in 3D viewer (same values as above) (eq. to '-rd3d').");
    cimg_option("-background3d","'R,_G,_B'","Define background color in 3D viewer (eq. to '-b3d').");

    cimg_help("\n  Program controls\n"
              "  ----------------");
    cimg_option("-nop","(no args)","Do nothing.");
    cimg_option("-skip","(any args)","Do nothing but skip the next argument.");
    cimg_option("-echo","'text'","Output textual message (eq. to '-e').");
    cimg_option("-print","(no args)","Print image(s) informations (eq. to '-p').");
    cimg_option("-return","(no args)","Return current function.");
    cimg_option("-quit","(no args)","Quit interpreter (eq. to '-q').");
    cimg_option("-exec","(no args)","Execute external command.");
    cimg_option("-do","(no args)","Start a 'do-while' code bloc.");
    cimg_option("-while","'cond'","End a 'do-while' code bloc and go back to associated '-do' if condition is true.");
    cimg_option("-if","'cond'","Start a 'if-then-else' code bloc and test if condition is true.");
    cimg_option("-else","(no args)","Execute following commands if previous '-if' condition failed.");
    cimg_option("-endif","(no args)","End a 'if-then-else' code bloc");
    cimg_option("-repeat","'N'","Start a 'repeat-done' code bloc (N can be a formula).");
    cimg_option("-done","(no args)","End a 'repeat-done' code bloc, and go to associated '-repeat' if iterations remain.");
    cimg_option("-int","'arg1,...,argN'","Check if all arguments are integer.");
    cimg_option("-float","'arg1,...,argN","Check if all arguments are float values.");
    cimg_option("-error","'message'","Print error message and exit.");
    cimg_option("-local","(no args)","Start a local environment with the given images (eq. to '-l').");
    cimg_option("-endlocal","(no args)","End the previous local environment (eq. to '-endl').");

    cimg_help("\n  Input/output\n"
              "  ------------");
    cimg_option("-input","'filename' or 'width[%]>0,_height[%]>0,_depth[%]>0,_dim[%]>0,_value(s)'","");
    cimg_help("                     "
              "or '[indice],_nb>0' or '(v11{,;/^}v21...vLM)'");
    cimg_help("                                              "
              "Input filename, image copy, or image with specified values");
    cimg_help("                                              "
              "(eq. to '-i' or (no args)).");
    cimg_option("-output","'filename'","Output image(s) into a filename (eq. to '-o').");
    cimg_option("-display","(no args)","Display image(s) on an interactive window (eq. to '-d').");
    cimg_option("-window","'width>=0,height>=0,normalization={0,1,2,3},_framerate' or '(no args)'","");
    cimg_help("                                              "
              "Display image(s) on an instant window of specified size (eq. to '-w').");
    cimg_help("                                              "
              "('normalization' can be '{0=none, 1=always, 2=1st-time, 3=auto}').");
    cimg_option("-display3d","(no args)","Display 3D object(s) (eq. to '-d3d').");
    cimg_option("-plot","'_plot_type={0,1,2,3},_vertex_type={0,1,2,3,4,5,6,7},_xmin,_xmax,_ymin,_ymax'","");
    cimg_help("                  "
              "or ''formula',_xmin,xmax,_ymin,_ymax,_resolution,_plot_type,_vertex_type'");
    cimg_help("                                              "
              "Display image or formula as a 1D plot on an interactive window");
    cimg_help("                                              "
              "('plot_type' can be '{0=none, 1=lines, 2=splines, 3=bar}').");
    cimg_help("                                              "
              "('vertex_type' can be '{0=none, 1=points, 2,3=crosses, 4,5=circles, 6,7=squares}').");
    cimg_option("-select","'select_type={0,1,2,3}'","Interactively select a feature from image(s)");
    cimg_help("                                              "
              "('select_type' can be in '{0=point, 1=line, 2=rectangle, 3=circle').");

    // Print descriptions of default macros.
    char line[256*1024] = { 0 }, name[4096] = { 0 }, args[4096] = { 0 }, desc[4096] = { 0 };
    bool first_description = true;
    for (const char *data = data_gmic_def; *data; ) {
      if (*data=='\n') ++data;
      else {
        if (std::sscanf(data,"%262143[^\n]",line)>0) data += std::strlen(line);
        if (line[0]=='#' && std::sscanf(line,"#@gmic %4095[^:] : %4095[^:] : %4095[^:]",name,args,desc)>0) {
          if (first_description) cimg_help("\n  Commands : Default macros\n"
                                           "  -------------------------");
          std::fprintf(cimg_stdout,"%s    %s-%-15s%s %-24s %s%s%s",
                       first_description?"":"\n",
                       cimg::t_bold,name,cimg::t_normal,args,cimg::t_green,desc,cimg::t_normal);
          std::fflush(cimg_stdout);
          first_description = false;
        }
      }
    }

    // Print descriptions of use-defined macros.
    first_description = true;
    for (int i = 1; i<argc-1; ++i) if (!cimg::strcmp("-m",argv[i]) || !cimg::strcmp("-macros",argv[i])) {
      std::FILE *file = cimg::fopen(argv[i+1],"r");
      if (file) {
        int err = 0;
        while ((err=std::fscanf(file,"%262143[^\n] ",line)>=0)) {
          if (err) { // Non empty-line
            name[0] = args[0] = desc[0] = 0;
            if (line[0]=='#' && std::sscanf(line,"#@gmic %4095[^:] : %4095[^:] : %4095[^:]",name,args,desc)>0) {
              if (first_description) cimg_help("\n\n  Commands : User-defined macros\n"
                                               "  ------------------------------");
              std::fprintf(cimg_stdout,"%s    %s-%-15s%s %-24s %s%s%s",
                           first_description?"":"\n",
                           cimg::t_bold,name,cimg::t_normal,args,cimg::t_green,desc,cimg::t_normal);
              std::fflush(cimg_stdout);
              first_description = false;
            }
          }
        }
      }
      cimg::fclose(file);
    }

    cimg_help("\n\n  Viewers shortcuts\n"
              "  -----------------");
    cimg_help("  When displaying image(s) or 3D object(s) with G'MIC, these keyboard shortcuts are available :");
    cimg_help("   - CTRL+D : Increase window size.");
    cimg_help("   - CTRL+C : Decrease window size.");
    cimg_help("   - CTRL+R : Reset window size.");
    cimg_help("   - CTRL+F : Toggle fullscreen mode.");
    cimg_help("   - CTRL+S : Save current window snapshot.");
    cimg_help("   - CTRL+O : Save current instance of viewed image or 3D object.\n");
    cimg_help("  Specific options for the image viewer :");
    cimg_help("   - CTRL+P             : Play stack of frames as a movie.");
    cimg_help("   - CTRL+(mousewheel)  : Zoom in/out.");
    cimg_help("   - SHIFT+(mousewheel) : Go left/right.");
    cimg_help("   - ALT+(mousewheel)   : Go up/down.");
    cimg_help("   - Numeric PAD        : Zoom in/out (+/-) and move zoomed region (digits).");
    cimg_help("   - BACKSPACE          : Reset zoom.\n");
    cimg_help("  Specific options for the 3D object viewer :");
    cimg_help("   - (mouse) + (left mouse button)   : Rotate object.");
    cimg_help("   - (mouse) + (right mouse button)  : Zoom object.");
    cimg_help("   - (mouse) + (middle mouse button) : Translate object.");
    cimg_help("   - (mousewheel)                    : Zoom in/out.");
    cimg_help("   - CTRL + Z : Enable/disable Z-buffer rendering");

    cimg_help("\n  File options\n"
              "  ------------");
    cimg_help("  G'MIC is able to read/write most of the classical image file formats, including :");
    cimg_help("   - 2D grayscale/color images : PNG,JPEG,GIF,PNM,TIFF,BMP,....");
    cimg_help("   - 3D volumetric images : DICOM,HDR,NII,PAN,CIMG,INR,....");
    cimg_help("   - Video files : MPEG,AVI,MOV,OGG,FLV,...");
    cimg_help("   - Generic data files : DLM,ASC,RAW,TXT.");
    cimg_help("   - 3D objects : OFF.\n");
    cimg_help("  Specific options :");
    cimg_help("   - For video files : you can read only sub-frames of the image sequence (recommended) with the expression");
    cimg_help("     'video.ext,[first_frame[%][,last_frame[%][,step]]]'.");
    cimg_help("   - For RAW files : you must specify the image dimensions with the expression");
    cimg_help("     'file.raw,width[,height[,depth[,dim]]]]'.");
    cimg_help("   - For YUV files : you must specify the image dimensions and can read only sub-frames of the image sequence with the expression");
    cimg_help("     'file.yuv,width,height[,first_frame[,last_frame[,step]]]'.");
    cimg_help("   - For JPEG files : you can specify the quality (in %) of an output jpeg file format with the expression");
    cimg_help("     'file.jpg,30%'.");
    cimg_help("   - If the input file has extension '.gmic', it is read as a G'MIC macro file.");

    cimg_help("\n  Examples of use\n"
              "  ---------------");
    cimg_help("  G'MIC is a simple but quite complete interpreter of image processing commands, and can be used for a wide variety of");
    cimg_help("  image processing tasks. Here are few examples of how the command line tool G'MIC may be used :\n");
    cimg_help("   - View image data : ");
    cimg_help("       gmic file1.bmp file2.jpeg");
    cimg_help("   - Convert image file : ");
    cimg_help("       gmic input.bmp -o output.jpg");
    cimg_help("   - Create volumetric image from movie sequence : ");
    cimg_help("       gmic input.mpg -a z -o output.hdr");
    cimg_help("   - Compute image gradient norm : ");
    cimg_help("       gmic input.bmp -gradient_norm");
    cimg_help("   - Create a G'MIC 3D logo : ");
    cimg_help("       gmic 180,70,1,3 -text G\\'MIC,30,5,50,1,1 -blur 2 -n 0,100 --plasma 0.4 -+ -blur 1 -elevation3d -0.1 -rd3d 4");
    cimg_help("\n  See also the macros defined in the provided macro file 'gmic_def.raw' for other examples.");

    cimg_help("\n  ** G'MIC comes with ABSOLUTELY NO WARRANTY; "
              "for details visit http://gmic.sourceforge.net **");
    std::exit(0);
  }

  // Launch G'MIC instance.
  //-----------------------
  CImgList<float> images;
  try { gmic(argc,argv,images); }
  catch (gmic_exception &e) { std::fprintf(stderr,"\n<gmic-error> %s\n",e.message); return -1; }
  return 0;
}
#endif

#endif // #ifdef cimg_plugin ... #else ...
