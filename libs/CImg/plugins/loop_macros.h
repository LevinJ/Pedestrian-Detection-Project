/*
 #
 #  File        : loop_macros.h
 #                ( C++ header file - CImg plug-in )
 #
 #  Description : CImg plug-in adding useful loop macros in CImg, in order to
 #                deal with NxN neighborhoods (where N=10..32)
 #                and NxNxN neighborhoods (where N=4..8)
 #                This file has been automatically generated using the loop
 #                macro generator available in 'examples/generate_loop_macros.cpp'
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

#ifndef cimg_plugin_loopmacros
#define cimg_plugin_loopmacros

// Define 10x10 loop macros for CImg
//----------------------------------
#define cimg_for10(bound,i) for (int i = 0, \
 _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5; \
 _n5##i<(int)(bound) || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i)

#define cimg_for10X(img,x) cimg_for10((img).width,x)
#define cimg_for10Y(img,y) cimg_for10((img).height,y)
#define cimg_for10Z(img,z) cimg_for10((img).depth,z)
#define cimg_for10V(img,v) cimg_for10((img).dim,v)
#define cimg_for10XY(img,x,y) cimg_for10Y(img,y) cimg_for10X(img,x)
#define cimg_for10XZ(img,x,z) cimg_for10Z(img,z) cimg_for10X(img,x)
#define cimg_for10XV(img,x,v) cimg_for10V(img,v) cimg_for10X(img,x)
#define cimg_for10YZ(img,y,z) cimg_for10Z(img,z) cimg_for10Y(img,y)
#define cimg_for10YV(img,y,v) cimg_for10V(img,v) cimg_for10Y(img,y)
#define cimg_for10ZV(img,z,v) cimg_for10V(img,v) cimg_for10Z(img,z)
#define cimg_for10XYZ(img,x,y,z) cimg_for10Z(img,z) cimg_for10XY(img,x,y)
#define cimg_for10XZV(img,x,z,v) cimg_for10V(img,v) cimg_for10XZ(img,x,z)
#define cimg_for10YZV(img,y,z,v) cimg_for10V(img,v) cimg_for10YZ(img,y,z)
#define cimg_for10XYZV(img,x,y,z,v) cimg_for10V(img,v) cimg_for10XYZ(img,x,y,z)

#define cimg_for_in10(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5; \
 i<=(int)(i1) && (_n5##i<(int)(bound) || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i)

#define cimg_for_in10X(img,x0,x1,x) cimg_for_in10((img).width,x0,x1,x)
#define cimg_for_in10Y(img,y0,y1,y) cimg_for_in10((img).height,y0,y1,y)
#define cimg_for_in10Z(img,z0,z1,z) cimg_for_in10((img).depth,z0,z1,z)
#define cimg_for_in10V(img,v0,v1,v) cimg_for_in10((img).dim,v0,v1,v)
#define cimg_for_in10XY(img,x0,y0,x1,y1,x,y) cimg_for_in10Y(img,y0,y1,y) cimg_for_in10X(img,x0,x1,x)
#define cimg_for_in10XZ(img,x0,z0,x1,z1,x,z) cimg_for_in10Z(img,z0,z1,z) cimg_for_in10X(img,x0,x1,x)
#define cimg_for_in10XV(img,x0,v0,x1,v1,x,v) cimg_for_in10V(img,v0,v1,v) cimg_for_in10X(img,x0,x1,x)
#define cimg_for_in10YZ(img,y0,z0,y1,z1,y,z) cimg_for_in10Z(img,z0,z1,z) cimg_for_in10Y(img,y0,y1,y)
#define cimg_for_in10YV(img,y0,v0,y1,v1,y,v) cimg_for_in10V(img,v0,v1,v) cimg_for_in10Y(img,y0,y1,y)
#define cimg_for_in10ZV(img,z0,v0,z1,v1,z,v) cimg_for_in10V(img,v0,v1,v) cimg_for_in10Z(img,z0,z1,z)
#define cimg_for_in10XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in10Z(img,z0,z1,z) cimg_for_in10XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in10XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in10V(img,v0,v1,v) cimg_for_in10XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in10YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in10V(img,v0,v1,v) cimg_for_in10YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in10XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in10V(img,v0,v1,v) cimg_for_in10XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for10x10(img,x,y,z,v,I) \
 cimg_for10((img).height,y) for (int x = 0, \
 _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = (img)(0,_p4##y,z,v)), \
 (I[10] = I[11] = I[12] = I[13] = I[14] = (img)(0,_p3##y,z,v)), \
 (I[20] = I[21] = I[22] = I[23] = I[24] = (img)(0,_p2##y,z,v)), \
 (I[30] = I[31] = I[32] = I[33] = I[34] = (img)(0,_p1##y,z,v)), \
 (I[40] = I[41] = I[42] = I[43] = I[44] = (img)(0,y,z,v)), \
 (I[50] = I[51] = I[52] = I[53] = I[54] = (img)(0,_n1##y,z,v)), \
 (I[60] = I[61] = I[62] = I[63] = I[64] = (img)(0,_n2##y,z,v)), \
 (I[70] = I[71] = I[72] = I[73] = I[74] = (img)(0,_n3##y,z,v)), \
 (I[80] = I[81] = I[82] = I[83] = I[84] = (img)(0,_n4##y,z,v)), \
 (I[90] = I[91] = I[92] = I[93] = I[94] = (img)(0,_n5##y,z,v)), \
 (I[5] = (img)(_n1##x,_p4##y,z,v)), \
 (I[15] = (img)(_n1##x,_p3##y,z,v)), \
 (I[25] = (img)(_n1##x,_p2##y,z,v)), \
 (I[35] = (img)(_n1##x,_p1##y,z,v)), \
 (I[45] = (img)(_n1##x,y,z,v)), \
 (I[55] = (img)(_n1##x,_n1##y,z,v)), \
 (I[65] = (img)(_n1##x,_n2##y,z,v)), \
 (I[75] = (img)(_n1##x,_n3##y,z,v)), \
 (I[85] = (img)(_n1##x,_n4##y,z,v)), \
 (I[95] = (img)(_n1##x,_n5##y,z,v)), \
 (I[6] = (img)(_n2##x,_p4##y,z,v)), \
 (I[16] = (img)(_n2##x,_p3##y,z,v)), \
 (I[26] = (img)(_n2##x,_p2##y,z,v)), \
 (I[36] = (img)(_n2##x,_p1##y,z,v)), \
 (I[46] = (img)(_n2##x,y,z,v)), \
 (I[56] = (img)(_n2##x,_n1##y,z,v)), \
 (I[66] = (img)(_n2##x,_n2##y,z,v)), \
 (I[76] = (img)(_n2##x,_n3##y,z,v)), \
 (I[86] = (img)(_n2##x,_n4##y,z,v)), \
 (I[96] = (img)(_n2##x,_n5##y,z,v)), \
 (I[7] = (img)(_n3##x,_p4##y,z,v)), \
 (I[17] = (img)(_n3##x,_p3##y,z,v)), \
 (I[27] = (img)(_n3##x,_p2##y,z,v)), \
 (I[37] = (img)(_n3##x,_p1##y,z,v)), \
 (I[47] = (img)(_n3##x,y,z,v)), \
 (I[57] = (img)(_n3##x,_n1##y,z,v)), \
 (I[67] = (img)(_n3##x,_n2##y,z,v)), \
 (I[77] = (img)(_n3##x,_n3##y,z,v)), \
 (I[87] = (img)(_n3##x,_n4##y,z,v)), \
 (I[97] = (img)(_n3##x,_n5##y,z,v)), \
 (I[8] = (img)(_n4##x,_p4##y,z,v)), \
 (I[18] = (img)(_n4##x,_p3##y,z,v)), \
 (I[28] = (img)(_n4##x,_p2##y,z,v)), \
 (I[38] = (img)(_n4##x,_p1##y,z,v)), \
 (I[48] = (img)(_n4##x,y,z,v)), \
 (I[58] = (img)(_n4##x,_n1##y,z,v)), \
 (I[68] = (img)(_n4##x,_n2##y,z,v)), \
 (I[78] = (img)(_n4##x,_n3##y,z,v)), \
 (I[88] = (img)(_n4##x,_n4##y,z,v)), \
 (I[98] = (img)(_n4##x,_n5##y,z,v)), \
 5>=((img).width)?(int)((img).width)-1:5); \
 (_n5##x<(int)((img).width) && ( \
 (I[9] = (img)(_n5##x,_p4##y,z,v)), \
 (I[19] = (img)(_n5##x,_p3##y,z,v)), \
 (I[29] = (img)(_n5##x,_p2##y,z,v)), \
 (I[39] = (img)(_n5##x,_p1##y,z,v)), \
 (I[49] = (img)(_n5##x,y,z,v)), \
 (I[59] = (img)(_n5##x,_n1##y,z,v)), \
 (I[69] = (img)(_n5##x,_n2##y,z,v)), \
 (I[79] = (img)(_n5##x,_n3##y,z,v)), \
 (I[89] = (img)(_n5##x,_n4##y,z,v)), \
 (I[99] = (img)(_n5##x,_n5##y,z,v)),1)) || \
 _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], \
 I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], \
 I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], \
 I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], \
 I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], \
 I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], \
 I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], \
 I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], \
 I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], \
 _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x)

#define cimg_for_in10x10(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in10((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = (int)( \
 (I[0] = (img)(_p4##x,_p4##y,z,v)), \
 (I[10] = (img)(_p4##x,_p3##y,z,v)), \
 (I[20] = (img)(_p4##x,_p2##y,z,v)), \
 (I[30] = (img)(_p4##x,_p1##y,z,v)), \
 (I[40] = (img)(_p4##x,y,z,v)), \
 (I[50] = (img)(_p4##x,_n1##y,z,v)), \
 (I[60] = (img)(_p4##x,_n2##y,z,v)), \
 (I[70] = (img)(_p4##x,_n3##y,z,v)), \
 (I[80] = (img)(_p4##x,_n4##y,z,v)), \
 (I[90] = (img)(_p4##x,_n5##y,z,v)), \
 (I[1] = (img)(_p3##x,_p4##y,z,v)), \
 (I[11] = (img)(_p3##x,_p3##y,z,v)), \
 (I[21] = (img)(_p3##x,_p2##y,z,v)), \
 (I[31] = (img)(_p3##x,_p1##y,z,v)), \
 (I[41] = (img)(_p3##x,y,z,v)), \
 (I[51] = (img)(_p3##x,_n1##y,z,v)), \
 (I[61] = (img)(_p3##x,_n2##y,z,v)), \
 (I[71] = (img)(_p3##x,_n3##y,z,v)), \
 (I[81] = (img)(_p3##x,_n4##y,z,v)), \
 (I[91] = (img)(_p3##x,_n5##y,z,v)), \
 (I[2] = (img)(_p2##x,_p4##y,z,v)), \
 (I[12] = (img)(_p2##x,_p3##y,z,v)), \
 (I[22] = (img)(_p2##x,_p2##y,z,v)), \
 (I[32] = (img)(_p2##x,_p1##y,z,v)), \
 (I[42] = (img)(_p2##x,y,z,v)), \
 (I[52] = (img)(_p2##x,_n1##y,z,v)), \
 (I[62] = (img)(_p2##x,_n2##y,z,v)), \
 (I[72] = (img)(_p2##x,_n3##y,z,v)), \
 (I[82] = (img)(_p2##x,_n4##y,z,v)), \
 (I[92] = (img)(_p2##x,_n5##y,z,v)), \
 (I[3] = (img)(_p1##x,_p4##y,z,v)), \
 (I[13] = (img)(_p1##x,_p3##y,z,v)), \
 (I[23] = (img)(_p1##x,_p2##y,z,v)), \
 (I[33] = (img)(_p1##x,_p1##y,z,v)), \
 (I[43] = (img)(_p1##x,y,z,v)), \
 (I[53] = (img)(_p1##x,_n1##y,z,v)), \
 (I[63] = (img)(_p1##x,_n2##y,z,v)), \
 (I[73] = (img)(_p1##x,_n3##y,z,v)), \
 (I[83] = (img)(_p1##x,_n4##y,z,v)), \
 (I[93] = (img)(_p1##x,_n5##y,z,v)), \
 (I[4] = (img)(x,_p4##y,z,v)), \
 (I[14] = (img)(x,_p3##y,z,v)), \
 (I[24] = (img)(x,_p2##y,z,v)), \
 (I[34] = (img)(x,_p1##y,z,v)), \
 (I[44] = (img)(x,y,z,v)), \
 (I[54] = (img)(x,_n1##y,z,v)), \
 (I[64] = (img)(x,_n2##y,z,v)), \
 (I[74] = (img)(x,_n3##y,z,v)), \
 (I[84] = (img)(x,_n4##y,z,v)), \
 (I[94] = (img)(x,_n5##y,z,v)), \
 (I[5] = (img)(_n1##x,_p4##y,z,v)), \
 (I[15] = (img)(_n1##x,_p3##y,z,v)), \
 (I[25] = (img)(_n1##x,_p2##y,z,v)), \
 (I[35] = (img)(_n1##x,_p1##y,z,v)), \
 (I[45] = (img)(_n1##x,y,z,v)), \
 (I[55] = (img)(_n1##x,_n1##y,z,v)), \
 (I[65] = (img)(_n1##x,_n2##y,z,v)), \
 (I[75] = (img)(_n1##x,_n3##y,z,v)), \
 (I[85] = (img)(_n1##x,_n4##y,z,v)), \
 (I[95] = (img)(_n1##x,_n5##y,z,v)), \
 (I[6] = (img)(_n2##x,_p4##y,z,v)), \
 (I[16] = (img)(_n2##x,_p3##y,z,v)), \
 (I[26] = (img)(_n2##x,_p2##y,z,v)), \
 (I[36] = (img)(_n2##x,_p1##y,z,v)), \
 (I[46] = (img)(_n2##x,y,z,v)), \
 (I[56] = (img)(_n2##x,_n1##y,z,v)), \
 (I[66] = (img)(_n2##x,_n2##y,z,v)), \
 (I[76] = (img)(_n2##x,_n3##y,z,v)), \
 (I[86] = (img)(_n2##x,_n4##y,z,v)), \
 (I[96] = (img)(_n2##x,_n5##y,z,v)), \
 (I[7] = (img)(_n3##x,_p4##y,z,v)), \
 (I[17] = (img)(_n3##x,_p3##y,z,v)), \
 (I[27] = (img)(_n3##x,_p2##y,z,v)), \
 (I[37] = (img)(_n3##x,_p1##y,z,v)), \
 (I[47] = (img)(_n3##x,y,z,v)), \
 (I[57] = (img)(_n3##x,_n1##y,z,v)), \
 (I[67] = (img)(_n3##x,_n2##y,z,v)), \
 (I[77] = (img)(_n3##x,_n3##y,z,v)), \
 (I[87] = (img)(_n3##x,_n4##y,z,v)), \
 (I[97] = (img)(_n3##x,_n5##y,z,v)), \
 (I[8] = (img)(_n4##x,_p4##y,z,v)), \
 (I[18] = (img)(_n4##x,_p3##y,z,v)), \
 (I[28] = (img)(_n4##x,_p2##y,z,v)), \
 (I[38] = (img)(_n4##x,_p1##y,z,v)), \
 (I[48] = (img)(_n4##x,y,z,v)), \
 (I[58] = (img)(_n4##x,_n1##y,z,v)), \
 (I[68] = (img)(_n4##x,_n2##y,z,v)), \
 (I[78] = (img)(_n4##x,_n3##y,z,v)), \
 (I[88] = (img)(_n4##x,_n4##y,z,v)), \
 (I[98] = (img)(_n4##x,_n5##y,z,v)), \
 x+5>=(int)((img).width)?(int)((img).width)-1:x+5); \
 x<=(int)(x1) && ((_n5##x<(int)((img).width) && ( \
 (I[9] = (img)(_n5##x,_p4##y,z,v)), \
 (I[19] = (img)(_n5##x,_p3##y,z,v)), \
 (I[29] = (img)(_n5##x,_p2##y,z,v)), \
 (I[39] = (img)(_n5##x,_p1##y,z,v)), \
 (I[49] = (img)(_n5##x,y,z,v)), \
 (I[59] = (img)(_n5##x,_n1##y,z,v)), \
 (I[69] = (img)(_n5##x,_n2##y,z,v)), \
 (I[79] = (img)(_n5##x,_n3##y,z,v)), \
 (I[89] = (img)(_n5##x,_n4##y,z,v)), \
 (I[99] = (img)(_n5##x,_n5##y,z,v)),1)) || \
 _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], \
 I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], \
 I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], \
 I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], \
 I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], \
 I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], \
 I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], \
 I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], \
 I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], \
 _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x)

#define cimg_get10x10(img,x,y,z,v,I) \
 I[0] = (img)(_p4##x,_p4##y,z,v), I[1] = (img)(_p3##x,_p4##y,z,v), I[2] = (img)(_p2##x,_p4##y,z,v), I[3] = (img)(_p1##x,_p4##y,z,v), I[4] = (img)(x,_p4##y,z,v), I[5] = (img)(_n1##x,_p4##y,z,v), I[6] = (img)(_n2##x,_p4##y,z,v), I[7] = (img)(_n3##x,_p4##y,z,v), I[8] = (img)(_n4##x,_p4##y,z,v), I[9] = (img)(_n5##x,_p4##y,z,v), \
 I[10] = (img)(_p4##x,_p3##y,z,v), I[11] = (img)(_p3##x,_p3##y,z,v), I[12] = (img)(_p2##x,_p3##y,z,v), I[13] = (img)(_p1##x,_p3##y,z,v), I[14] = (img)(x,_p3##y,z,v), I[15] = (img)(_n1##x,_p3##y,z,v), I[16] = (img)(_n2##x,_p3##y,z,v), I[17] = (img)(_n3##x,_p3##y,z,v), I[18] = (img)(_n4##x,_p3##y,z,v), I[19] = (img)(_n5##x,_p3##y,z,v), \
 I[20] = (img)(_p4##x,_p2##y,z,v), I[21] = (img)(_p3##x,_p2##y,z,v), I[22] = (img)(_p2##x,_p2##y,z,v), I[23] = (img)(_p1##x,_p2##y,z,v), I[24] = (img)(x,_p2##y,z,v), I[25] = (img)(_n1##x,_p2##y,z,v), I[26] = (img)(_n2##x,_p2##y,z,v), I[27] = (img)(_n3##x,_p2##y,z,v), I[28] = (img)(_n4##x,_p2##y,z,v), I[29] = (img)(_n5##x,_p2##y,z,v), \
 I[30] = (img)(_p4##x,_p1##y,z,v), I[31] = (img)(_p3##x,_p1##y,z,v), I[32] = (img)(_p2##x,_p1##y,z,v), I[33] = (img)(_p1##x,_p1##y,z,v), I[34] = (img)(x,_p1##y,z,v), I[35] = (img)(_n1##x,_p1##y,z,v), I[36] = (img)(_n2##x,_p1##y,z,v), I[37] = (img)(_n3##x,_p1##y,z,v), I[38] = (img)(_n4##x,_p1##y,z,v), I[39] = (img)(_n5##x,_p1##y,z,v), \
 I[40] = (img)(_p4##x,y,z,v), I[41] = (img)(_p3##x,y,z,v), I[42] = (img)(_p2##x,y,z,v), I[43] = (img)(_p1##x,y,z,v), I[44] = (img)(x,y,z,v), I[45] = (img)(_n1##x,y,z,v), I[46] = (img)(_n2##x,y,z,v), I[47] = (img)(_n3##x,y,z,v), I[48] = (img)(_n4##x,y,z,v), I[49] = (img)(_n5##x,y,z,v), \
 I[50] = (img)(_p4##x,_n1##y,z,v), I[51] = (img)(_p3##x,_n1##y,z,v), I[52] = (img)(_p2##x,_n1##y,z,v), I[53] = (img)(_p1##x,_n1##y,z,v), I[54] = (img)(x,_n1##y,z,v), I[55] = (img)(_n1##x,_n1##y,z,v), I[56] = (img)(_n2##x,_n1##y,z,v), I[57] = (img)(_n3##x,_n1##y,z,v), I[58] = (img)(_n4##x,_n1##y,z,v), I[59] = (img)(_n5##x,_n1##y,z,v), \
 I[60] = (img)(_p4##x,_n2##y,z,v), I[61] = (img)(_p3##x,_n2##y,z,v), I[62] = (img)(_p2##x,_n2##y,z,v), I[63] = (img)(_p1##x,_n2##y,z,v), I[64] = (img)(x,_n2##y,z,v), I[65] = (img)(_n1##x,_n2##y,z,v), I[66] = (img)(_n2##x,_n2##y,z,v), I[67] = (img)(_n3##x,_n2##y,z,v), I[68] = (img)(_n4##x,_n2##y,z,v), I[69] = (img)(_n5##x,_n2##y,z,v), \
 I[70] = (img)(_p4##x,_n3##y,z,v), I[71] = (img)(_p3##x,_n3##y,z,v), I[72] = (img)(_p2##x,_n3##y,z,v), I[73] = (img)(_p1##x,_n3##y,z,v), I[74] = (img)(x,_n3##y,z,v), I[75] = (img)(_n1##x,_n3##y,z,v), I[76] = (img)(_n2##x,_n3##y,z,v), I[77] = (img)(_n3##x,_n3##y,z,v), I[78] = (img)(_n4##x,_n3##y,z,v), I[79] = (img)(_n5##x,_n3##y,z,v), \
 I[80] = (img)(_p4##x,_n4##y,z,v), I[81] = (img)(_p3##x,_n4##y,z,v), I[82] = (img)(_p2##x,_n4##y,z,v), I[83] = (img)(_p1##x,_n4##y,z,v), I[84] = (img)(x,_n4##y,z,v), I[85] = (img)(_n1##x,_n4##y,z,v), I[86] = (img)(_n2##x,_n4##y,z,v), I[87] = (img)(_n3##x,_n4##y,z,v), I[88] = (img)(_n4##x,_n4##y,z,v), I[89] = (img)(_n5##x,_n4##y,z,v), \
 I[90] = (img)(_p4##x,_n5##y,z,v), I[91] = (img)(_p3##x,_n5##y,z,v), I[92] = (img)(_p2##x,_n5##y,z,v), I[93] = (img)(_p1##x,_n5##y,z,v), I[94] = (img)(x,_n5##y,z,v), I[95] = (img)(_n1##x,_n5##y,z,v), I[96] = (img)(_n2##x,_n5##y,z,v), I[97] = (img)(_n3##x,_n5##y,z,v), I[98] = (img)(_n4##x,_n5##y,z,v), I[99] = (img)(_n5##x,_n5##y,z,v);

// Define 11x11 loop macros for CImg
//----------------------------------
#define cimg_for11(bound,i) for (int i = 0, \
 _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5; \
 _n5##i<(int)(bound) || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i)

#define cimg_for11X(img,x) cimg_for11((img).width,x)
#define cimg_for11Y(img,y) cimg_for11((img).height,y)
#define cimg_for11Z(img,z) cimg_for11((img).depth,z)
#define cimg_for11V(img,v) cimg_for11((img).dim,v)
#define cimg_for11XY(img,x,y) cimg_for11Y(img,y) cimg_for11X(img,x)
#define cimg_for11XZ(img,x,z) cimg_for11Z(img,z) cimg_for11X(img,x)
#define cimg_for11XV(img,x,v) cimg_for11V(img,v) cimg_for11X(img,x)
#define cimg_for11YZ(img,y,z) cimg_for11Z(img,z) cimg_for11Y(img,y)
#define cimg_for11YV(img,y,v) cimg_for11V(img,v) cimg_for11Y(img,y)
#define cimg_for11ZV(img,z,v) cimg_for11V(img,v) cimg_for11Z(img,z)
#define cimg_for11XYZ(img,x,y,z) cimg_for11Z(img,z) cimg_for11XY(img,x,y)
#define cimg_for11XZV(img,x,z,v) cimg_for11V(img,v) cimg_for11XZ(img,x,z)
#define cimg_for11YZV(img,y,z,v) cimg_for11V(img,v) cimg_for11YZ(img,y,z)
#define cimg_for11XYZV(img,x,y,z,v) cimg_for11V(img,v) cimg_for11XYZ(img,x,y,z)

#define cimg_for_in11(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5; \
 i<=(int)(i1) && (_n5##i<(int)(bound) || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i)

#define cimg_for_in11X(img,x0,x1,x) cimg_for_in11((img).width,x0,x1,x)
#define cimg_for_in11Y(img,y0,y1,y) cimg_for_in11((img).height,y0,y1,y)
#define cimg_for_in11Z(img,z0,z1,z) cimg_for_in11((img).depth,z0,z1,z)
#define cimg_for_in11V(img,v0,v1,v) cimg_for_in11((img).dim,v0,v1,v)
#define cimg_for_in11XY(img,x0,y0,x1,y1,x,y) cimg_for_in11Y(img,y0,y1,y) cimg_for_in11X(img,x0,x1,x)
#define cimg_for_in11XZ(img,x0,z0,x1,z1,x,z) cimg_for_in11Z(img,z0,z1,z) cimg_for_in11X(img,x0,x1,x)
#define cimg_for_in11XV(img,x0,v0,x1,v1,x,v) cimg_for_in11V(img,v0,v1,v) cimg_for_in11X(img,x0,x1,x)
#define cimg_for_in11YZ(img,y0,z0,y1,z1,y,z) cimg_for_in11Z(img,z0,z1,z) cimg_for_in11Y(img,y0,y1,y)
#define cimg_for_in11YV(img,y0,v0,y1,v1,y,v) cimg_for_in11V(img,v0,v1,v) cimg_for_in11Y(img,y0,y1,y)
#define cimg_for_in11ZV(img,z0,v0,z1,v1,z,v) cimg_for_in11V(img,v0,v1,v) cimg_for_in11Z(img,z0,z1,z)
#define cimg_for_in11XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in11Z(img,z0,z1,z) cimg_for_in11XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in11XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in11V(img,v0,v1,v) cimg_for_in11XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in11YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in11V(img,v0,v1,v) cimg_for_in11YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in11XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in11V(img,v0,v1,v) cimg_for_in11XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for11x11(img,x,y,z,v,I) \
 cimg_for11((img).height,y) for (int x = 0, \
 _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = (img)(0,_p5##y,z,v)), \
 (I[11] = I[12] = I[13] = I[14] = I[15] = I[16] = (img)(0,_p4##y,z,v)), \
 (I[22] = I[23] = I[24] = I[25] = I[26] = I[27] = (img)(0,_p3##y,z,v)), \
 (I[33] = I[34] = I[35] = I[36] = I[37] = I[38] = (img)(0,_p2##y,z,v)), \
 (I[44] = I[45] = I[46] = I[47] = I[48] = I[49] = (img)(0,_p1##y,z,v)), \
 (I[55] = I[56] = I[57] = I[58] = I[59] = I[60] = (img)(0,y,z,v)), \
 (I[66] = I[67] = I[68] = I[69] = I[70] = I[71] = (img)(0,_n1##y,z,v)), \
 (I[77] = I[78] = I[79] = I[80] = I[81] = I[82] = (img)(0,_n2##y,z,v)), \
 (I[88] = I[89] = I[90] = I[91] = I[92] = I[93] = (img)(0,_n3##y,z,v)), \
 (I[99] = I[100] = I[101] = I[102] = I[103] = I[104] = (img)(0,_n4##y,z,v)), \
 (I[110] = I[111] = I[112] = I[113] = I[114] = I[115] = (img)(0,_n5##y,z,v)), \
 (I[6] = (img)(_n1##x,_p5##y,z,v)), \
 (I[17] = (img)(_n1##x,_p4##y,z,v)), \
 (I[28] = (img)(_n1##x,_p3##y,z,v)), \
 (I[39] = (img)(_n1##x,_p2##y,z,v)), \
 (I[50] = (img)(_n1##x,_p1##y,z,v)), \
 (I[61] = (img)(_n1##x,y,z,v)), \
 (I[72] = (img)(_n1##x,_n1##y,z,v)), \
 (I[83] = (img)(_n1##x,_n2##y,z,v)), \
 (I[94] = (img)(_n1##x,_n3##y,z,v)), \
 (I[105] = (img)(_n1##x,_n4##y,z,v)), \
 (I[116] = (img)(_n1##x,_n5##y,z,v)), \
 (I[7] = (img)(_n2##x,_p5##y,z,v)), \
 (I[18] = (img)(_n2##x,_p4##y,z,v)), \
 (I[29] = (img)(_n2##x,_p3##y,z,v)), \
 (I[40] = (img)(_n2##x,_p2##y,z,v)), \
 (I[51] = (img)(_n2##x,_p1##y,z,v)), \
 (I[62] = (img)(_n2##x,y,z,v)), \
 (I[73] = (img)(_n2##x,_n1##y,z,v)), \
 (I[84] = (img)(_n2##x,_n2##y,z,v)), \
 (I[95] = (img)(_n2##x,_n3##y,z,v)), \
 (I[106] = (img)(_n2##x,_n4##y,z,v)), \
 (I[117] = (img)(_n2##x,_n5##y,z,v)), \
 (I[8] = (img)(_n3##x,_p5##y,z,v)), \
 (I[19] = (img)(_n3##x,_p4##y,z,v)), \
 (I[30] = (img)(_n3##x,_p3##y,z,v)), \
 (I[41] = (img)(_n3##x,_p2##y,z,v)), \
 (I[52] = (img)(_n3##x,_p1##y,z,v)), \
 (I[63] = (img)(_n3##x,y,z,v)), \
 (I[74] = (img)(_n3##x,_n1##y,z,v)), \
 (I[85] = (img)(_n3##x,_n2##y,z,v)), \
 (I[96] = (img)(_n3##x,_n3##y,z,v)), \
 (I[107] = (img)(_n3##x,_n4##y,z,v)), \
 (I[118] = (img)(_n3##x,_n5##y,z,v)), \
 (I[9] = (img)(_n4##x,_p5##y,z,v)), \
 (I[20] = (img)(_n4##x,_p4##y,z,v)), \
 (I[31] = (img)(_n4##x,_p3##y,z,v)), \
 (I[42] = (img)(_n4##x,_p2##y,z,v)), \
 (I[53] = (img)(_n4##x,_p1##y,z,v)), \
 (I[64] = (img)(_n4##x,y,z,v)), \
 (I[75] = (img)(_n4##x,_n1##y,z,v)), \
 (I[86] = (img)(_n4##x,_n2##y,z,v)), \
 (I[97] = (img)(_n4##x,_n3##y,z,v)), \
 (I[108] = (img)(_n4##x,_n4##y,z,v)), \
 (I[119] = (img)(_n4##x,_n5##y,z,v)), \
 5>=((img).width)?(int)((img).width)-1:5); \
 (_n5##x<(int)((img).width) && ( \
 (I[10] = (img)(_n5##x,_p5##y,z,v)), \
 (I[21] = (img)(_n5##x,_p4##y,z,v)), \
 (I[32] = (img)(_n5##x,_p3##y,z,v)), \
 (I[43] = (img)(_n5##x,_p2##y,z,v)), \
 (I[54] = (img)(_n5##x,_p1##y,z,v)), \
 (I[65] = (img)(_n5##x,y,z,v)), \
 (I[76] = (img)(_n5##x,_n1##y,z,v)), \
 (I[87] = (img)(_n5##x,_n2##y,z,v)), \
 (I[98] = (img)(_n5##x,_n3##y,z,v)), \
 (I[109] = (img)(_n5##x,_n4##y,z,v)), \
 (I[120] = (img)(_n5##x,_n5##y,z,v)),1)) || \
 _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], \
 I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], \
 I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], \
 I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], \
 I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], \
 I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], \
 I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], \
 I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], \
 I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], \
 I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], \
 I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], \
 _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x)

#define cimg_for_in11x11(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in11((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = (int)( \
 (I[0] = (img)(_p5##x,_p5##y,z,v)), \
 (I[11] = (img)(_p5##x,_p4##y,z,v)), \
 (I[22] = (img)(_p5##x,_p3##y,z,v)), \
 (I[33] = (img)(_p5##x,_p2##y,z,v)), \
 (I[44] = (img)(_p5##x,_p1##y,z,v)), \
 (I[55] = (img)(_p5##x,y,z,v)), \
 (I[66] = (img)(_p5##x,_n1##y,z,v)), \
 (I[77] = (img)(_p5##x,_n2##y,z,v)), \
 (I[88] = (img)(_p5##x,_n3##y,z,v)), \
 (I[99] = (img)(_p5##x,_n4##y,z,v)), \
 (I[110] = (img)(_p5##x,_n5##y,z,v)), \
 (I[1] = (img)(_p4##x,_p5##y,z,v)), \
 (I[12] = (img)(_p4##x,_p4##y,z,v)), \
 (I[23] = (img)(_p4##x,_p3##y,z,v)), \
 (I[34] = (img)(_p4##x,_p2##y,z,v)), \
 (I[45] = (img)(_p4##x,_p1##y,z,v)), \
 (I[56] = (img)(_p4##x,y,z,v)), \
 (I[67] = (img)(_p4##x,_n1##y,z,v)), \
 (I[78] = (img)(_p4##x,_n2##y,z,v)), \
 (I[89] = (img)(_p4##x,_n3##y,z,v)), \
 (I[100] = (img)(_p4##x,_n4##y,z,v)), \
 (I[111] = (img)(_p4##x,_n5##y,z,v)), \
 (I[2] = (img)(_p3##x,_p5##y,z,v)), \
 (I[13] = (img)(_p3##x,_p4##y,z,v)), \
 (I[24] = (img)(_p3##x,_p3##y,z,v)), \
 (I[35] = (img)(_p3##x,_p2##y,z,v)), \
 (I[46] = (img)(_p3##x,_p1##y,z,v)), \
 (I[57] = (img)(_p3##x,y,z,v)), \
 (I[68] = (img)(_p3##x,_n1##y,z,v)), \
 (I[79] = (img)(_p3##x,_n2##y,z,v)), \
 (I[90] = (img)(_p3##x,_n3##y,z,v)), \
 (I[101] = (img)(_p3##x,_n4##y,z,v)), \
 (I[112] = (img)(_p3##x,_n5##y,z,v)), \
 (I[3] = (img)(_p2##x,_p5##y,z,v)), \
 (I[14] = (img)(_p2##x,_p4##y,z,v)), \
 (I[25] = (img)(_p2##x,_p3##y,z,v)), \
 (I[36] = (img)(_p2##x,_p2##y,z,v)), \
 (I[47] = (img)(_p2##x,_p1##y,z,v)), \
 (I[58] = (img)(_p2##x,y,z,v)), \
 (I[69] = (img)(_p2##x,_n1##y,z,v)), \
 (I[80] = (img)(_p2##x,_n2##y,z,v)), \
 (I[91] = (img)(_p2##x,_n3##y,z,v)), \
 (I[102] = (img)(_p2##x,_n4##y,z,v)), \
 (I[113] = (img)(_p2##x,_n5##y,z,v)), \
 (I[4] = (img)(_p1##x,_p5##y,z,v)), \
 (I[15] = (img)(_p1##x,_p4##y,z,v)), \
 (I[26] = (img)(_p1##x,_p3##y,z,v)), \
 (I[37] = (img)(_p1##x,_p2##y,z,v)), \
 (I[48] = (img)(_p1##x,_p1##y,z,v)), \
 (I[59] = (img)(_p1##x,y,z,v)), \
 (I[70] = (img)(_p1##x,_n1##y,z,v)), \
 (I[81] = (img)(_p1##x,_n2##y,z,v)), \
 (I[92] = (img)(_p1##x,_n3##y,z,v)), \
 (I[103] = (img)(_p1##x,_n4##y,z,v)), \
 (I[114] = (img)(_p1##x,_n5##y,z,v)), \
 (I[5] = (img)(x,_p5##y,z,v)), \
 (I[16] = (img)(x,_p4##y,z,v)), \
 (I[27] = (img)(x,_p3##y,z,v)), \
 (I[38] = (img)(x,_p2##y,z,v)), \
 (I[49] = (img)(x,_p1##y,z,v)), \
 (I[60] = (img)(x,y,z,v)), \
 (I[71] = (img)(x,_n1##y,z,v)), \
 (I[82] = (img)(x,_n2##y,z,v)), \
 (I[93] = (img)(x,_n3##y,z,v)), \
 (I[104] = (img)(x,_n4##y,z,v)), \
 (I[115] = (img)(x,_n5##y,z,v)), \
 (I[6] = (img)(_n1##x,_p5##y,z,v)), \
 (I[17] = (img)(_n1##x,_p4##y,z,v)), \
 (I[28] = (img)(_n1##x,_p3##y,z,v)), \
 (I[39] = (img)(_n1##x,_p2##y,z,v)), \
 (I[50] = (img)(_n1##x,_p1##y,z,v)), \
 (I[61] = (img)(_n1##x,y,z,v)), \
 (I[72] = (img)(_n1##x,_n1##y,z,v)), \
 (I[83] = (img)(_n1##x,_n2##y,z,v)), \
 (I[94] = (img)(_n1##x,_n3##y,z,v)), \
 (I[105] = (img)(_n1##x,_n4##y,z,v)), \
 (I[116] = (img)(_n1##x,_n5##y,z,v)), \
 (I[7] = (img)(_n2##x,_p5##y,z,v)), \
 (I[18] = (img)(_n2##x,_p4##y,z,v)), \
 (I[29] = (img)(_n2##x,_p3##y,z,v)), \
 (I[40] = (img)(_n2##x,_p2##y,z,v)), \
 (I[51] = (img)(_n2##x,_p1##y,z,v)), \
 (I[62] = (img)(_n2##x,y,z,v)), \
 (I[73] = (img)(_n2##x,_n1##y,z,v)), \
 (I[84] = (img)(_n2##x,_n2##y,z,v)), \
 (I[95] = (img)(_n2##x,_n3##y,z,v)), \
 (I[106] = (img)(_n2##x,_n4##y,z,v)), \
 (I[117] = (img)(_n2##x,_n5##y,z,v)), \
 (I[8] = (img)(_n3##x,_p5##y,z,v)), \
 (I[19] = (img)(_n3##x,_p4##y,z,v)), \
 (I[30] = (img)(_n3##x,_p3##y,z,v)), \
 (I[41] = (img)(_n3##x,_p2##y,z,v)), \
 (I[52] = (img)(_n3##x,_p1##y,z,v)), \
 (I[63] = (img)(_n3##x,y,z,v)), \
 (I[74] = (img)(_n3##x,_n1##y,z,v)), \
 (I[85] = (img)(_n3##x,_n2##y,z,v)), \
 (I[96] = (img)(_n3##x,_n3##y,z,v)), \
 (I[107] = (img)(_n3##x,_n4##y,z,v)), \
 (I[118] = (img)(_n3##x,_n5##y,z,v)), \
 (I[9] = (img)(_n4##x,_p5##y,z,v)), \
 (I[20] = (img)(_n4##x,_p4##y,z,v)), \
 (I[31] = (img)(_n4##x,_p3##y,z,v)), \
 (I[42] = (img)(_n4##x,_p2##y,z,v)), \
 (I[53] = (img)(_n4##x,_p1##y,z,v)), \
 (I[64] = (img)(_n4##x,y,z,v)), \
 (I[75] = (img)(_n4##x,_n1##y,z,v)), \
 (I[86] = (img)(_n4##x,_n2##y,z,v)), \
 (I[97] = (img)(_n4##x,_n3##y,z,v)), \
 (I[108] = (img)(_n4##x,_n4##y,z,v)), \
 (I[119] = (img)(_n4##x,_n5##y,z,v)), \
 x+5>=(int)((img).width)?(int)((img).width)-1:x+5); \
 x<=(int)(x1) && ((_n5##x<(int)((img).width) && ( \
 (I[10] = (img)(_n5##x,_p5##y,z,v)), \
 (I[21] = (img)(_n5##x,_p4##y,z,v)), \
 (I[32] = (img)(_n5##x,_p3##y,z,v)), \
 (I[43] = (img)(_n5##x,_p2##y,z,v)), \
 (I[54] = (img)(_n5##x,_p1##y,z,v)), \
 (I[65] = (img)(_n5##x,y,z,v)), \
 (I[76] = (img)(_n5##x,_n1##y,z,v)), \
 (I[87] = (img)(_n5##x,_n2##y,z,v)), \
 (I[98] = (img)(_n5##x,_n3##y,z,v)), \
 (I[109] = (img)(_n5##x,_n4##y,z,v)), \
 (I[120] = (img)(_n5##x,_n5##y,z,v)),1)) || \
 _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], \
 I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], \
 I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], \
 I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], \
 I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], \
 I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], \
 I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], \
 I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], \
 I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], \
 I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], \
 I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], \
 _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x)

#define cimg_get11x11(img,x,y,z,v,I) \
 I[0] = (img)(_p5##x,_p5##y,z,v), I[1] = (img)(_p4##x,_p5##y,z,v), I[2] = (img)(_p3##x,_p5##y,z,v), I[3] = (img)(_p2##x,_p5##y,z,v), I[4] = (img)(_p1##x,_p5##y,z,v), I[5] = (img)(x,_p5##y,z,v), I[6] = (img)(_n1##x,_p5##y,z,v), I[7] = (img)(_n2##x,_p5##y,z,v), I[8] = (img)(_n3##x,_p5##y,z,v), I[9] = (img)(_n4##x,_p5##y,z,v), I[10] = (img)(_n5##x,_p5##y,z,v), \
 I[11] = (img)(_p5##x,_p4##y,z,v), I[12] = (img)(_p4##x,_p4##y,z,v), I[13] = (img)(_p3##x,_p4##y,z,v), I[14] = (img)(_p2##x,_p4##y,z,v), I[15] = (img)(_p1##x,_p4##y,z,v), I[16] = (img)(x,_p4##y,z,v), I[17] = (img)(_n1##x,_p4##y,z,v), I[18] = (img)(_n2##x,_p4##y,z,v), I[19] = (img)(_n3##x,_p4##y,z,v), I[20] = (img)(_n4##x,_p4##y,z,v), I[21] = (img)(_n5##x,_p4##y,z,v), \
 I[22] = (img)(_p5##x,_p3##y,z,v), I[23] = (img)(_p4##x,_p3##y,z,v), I[24] = (img)(_p3##x,_p3##y,z,v), I[25] = (img)(_p2##x,_p3##y,z,v), I[26] = (img)(_p1##x,_p3##y,z,v), I[27] = (img)(x,_p3##y,z,v), I[28] = (img)(_n1##x,_p3##y,z,v), I[29] = (img)(_n2##x,_p3##y,z,v), I[30] = (img)(_n3##x,_p3##y,z,v), I[31] = (img)(_n4##x,_p3##y,z,v), I[32] = (img)(_n5##x,_p3##y,z,v), \
 I[33] = (img)(_p5##x,_p2##y,z,v), I[34] = (img)(_p4##x,_p2##y,z,v), I[35] = (img)(_p3##x,_p2##y,z,v), I[36] = (img)(_p2##x,_p2##y,z,v), I[37] = (img)(_p1##x,_p2##y,z,v), I[38] = (img)(x,_p2##y,z,v), I[39] = (img)(_n1##x,_p2##y,z,v), I[40] = (img)(_n2##x,_p2##y,z,v), I[41] = (img)(_n3##x,_p2##y,z,v), I[42] = (img)(_n4##x,_p2##y,z,v), I[43] = (img)(_n5##x,_p2##y,z,v), \
 I[44] = (img)(_p5##x,_p1##y,z,v), I[45] = (img)(_p4##x,_p1##y,z,v), I[46] = (img)(_p3##x,_p1##y,z,v), I[47] = (img)(_p2##x,_p1##y,z,v), I[48] = (img)(_p1##x,_p1##y,z,v), I[49] = (img)(x,_p1##y,z,v), I[50] = (img)(_n1##x,_p1##y,z,v), I[51] = (img)(_n2##x,_p1##y,z,v), I[52] = (img)(_n3##x,_p1##y,z,v), I[53] = (img)(_n4##x,_p1##y,z,v), I[54] = (img)(_n5##x,_p1##y,z,v), \
 I[55] = (img)(_p5##x,y,z,v), I[56] = (img)(_p4##x,y,z,v), I[57] = (img)(_p3##x,y,z,v), I[58] = (img)(_p2##x,y,z,v), I[59] = (img)(_p1##x,y,z,v), I[60] = (img)(x,y,z,v), I[61] = (img)(_n1##x,y,z,v), I[62] = (img)(_n2##x,y,z,v), I[63] = (img)(_n3##x,y,z,v), I[64] = (img)(_n4##x,y,z,v), I[65] = (img)(_n5##x,y,z,v), \
 I[66] = (img)(_p5##x,_n1##y,z,v), I[67] = (img)(_p4##x,_n1##y,z,v), I[68] = (img)(_p3##x,_n1##y,z,v), I[69] = (img)(_p2##x,_n1##y,z,v), I[70] = (img)(_p1##x,_n1##y,z,v), I[71] = (img)(x,_n1##y,z,v), I[72] = (img)(_n1##x,_n1##y,z,v), I[73] = (img)(_n2##x,_n1##y,z,v), I[74] = (img)(_n3##x,_n1##y,z,v), I[75] = (img)(_n4##x,_n1##y,z,v), I[76] = (img)(_n5##x,_n1##y,z,v), \
 I[77] = (img)(_p5##x,_n2##y,z,v), I[78] = (img)(_p4##x,_n2##y,z,v), I[79] = (img)(_p3##x,_n2##y,z,v), I[80] = (img)(_p2##x,_n2##y,z,v), I[81] = (img)(_p1##x,_n2##y,z,v), I[82] = (img)(x,_n2##y,z,v), I[83] = (img)(_n1##x,_n2##y,z,v), I[84] = (img)(_n2##x,_n2##y,z,v), I[85] = (img)(_n3##x,_n2##y,z,v), I[86] = (img)(_n4##x,_n2##y,z,v), I[87] = (img)(_n5##x,_n2##y,z,v), \
 I[88] = (img)(_p5##x,_n3##y,z,v), I[89] = (img)(_p4##x,_n3##y,z,v), I[90] = (img)(_p3##x,_n3##y,z,v), I[91] = (img)(_p2##x,_n3##y,z,v), I[92] = (img)(_p1##x,_n3##y,z,v), I[93] = (img)(x,_n3##y,z,v), I[94] = (img)(_n1##x,_n3##y,z,v), I[95] = (img)(_n2##x,_n3##y,z,v), I[96] = (img)(_n3##x,_n3##y,z,v), I[97] = (img)(_n4##x,_n3##y,z,v), I[98] = (img)(_n5##x,_n3##y,z,v), \
 I[99] = (img)(_p5##x,_n4##y,z,v), I[100] = (img)(_p4##x,_n4##y,z,v), I[101] = (img)(_p3##x,_n4##y,z,v), I[102] = (img)(_p2##x,_n4##y,z,v), I[103] = (img)(_p1##x,_n4##y,z,v), I[104] = (img)(x,_n4##y,z,v), I[105] = (img)(_n1##x,_n4##y,z,v), I[106] = (img)(_n2##x,_n4##y,z,v), I[107] = (img)(_n3##x,_n4##y,z,v), I[108] = (img)(_n4##x,_n4##y,z,v), I[109] = (img)(_n5##x,_n4##y,z,v), \
 I[110] = (img)(_p5##x,_n5##y,z,v), I[111] = (img)(_p4##x,_n5##y,z,v), I[112] = (img)(_p3##x,_n5##y,z,v), I[113] = (img)(_p2##x,_n5##y,z,v), I[114] = (img)(_p1##x,_n5##y,z,v), I[115] = (img)(x,_n5##y,z,v), I[116] = (img)(_n1##x,_n5##y,z,v), I[117] = (img)(_n2##x,_n5##y,z,v), I[118] = (img)(_n3##x,_n5##y,z,v), I[119] = (img)(_n4##x,_n5##y,z,v), I[120] = (img)(_n5##x,_n5##y,z,v);

// Define 12x12 loop macros for CImg
//----------------------------------
#define cimg_for12(bound,i) for (int i = 0, \
 _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6; \
 _n6##i<(int)(bound) || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i)

#define cimg_for12X(img,x) cimg_for12((img).width,x)
#define cimg_for12Y(img,y) cimg_for12((img).height,y)
#define cimg_for12Z(img,z) cimg_for12((img).depth,z)
#define cimg_for12V(img,v) cimg_for12((img).dim,v)
#define cimg_for12XY(img,x,y) cimg_for12Y(img,y) cimg_for12X(img,x)
#define cimg_for12XZ(img,x,z) cimg_for12Z(img,z) cimg_for12X(img,x)
#define cimg_for12XV(img,x,v) cimg_for12V(img,v) cimg_for12X(img,x)
#define cimg_for12YZ(img,y,z) cimg_for12Z(img,z) cimg_for12Y(img,y)
#define cimg_for12YV(img,y,v) cimg_for12V(img,v) cimg_for12Y(img,y)
#define cimg_for12ZV(img,z,v) cimg_for12V(img,v) cimg_for12Z(img,z)
#define cimg_for12XYZ(img,x,y,z) cimg_for12Z(img,z) cimg_for12XY(img,x,y)
#define cimg_for12XZV(img,x,z,v) cimg_for12V(img,v) cimg_for12XZ(img,x,z)
#define cimg_for12YZV(img,y,z,v) cimg_for12V(img,v) cimg_for12YZ(img,y,z)
#define cimg_for12XYZV(img,x,y,z,v) cimg_for12V(img,v) cimg_for12XYZ(img,x,y,z)

#define cimg_for_in12(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6; \
 i<=(int)(i1) && (_n6##i<(int)(bound) || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i)

#define cimg_for_in12X(img,x0,x1,x) cimg_for_in12((img).width,x0,x1,x)
#define cimg_for_in12Y(img,y0,y1,y) cimg_for_in12((img).height,y0,y1,y)
#define cimg_for_in12Z(img,z0,z1,z) cimg_for_in12((img).depth,z0,z1,z)
#define cimg_for_in12V(img,v0,v1,v) cimg_for_in12((img).dim,v0,v1,v)
#define cimg_for_in12XY(img,x0,y0,x1,y1,x,y) cimg_for_in12Y(img,y0,y1,y) cimg_for_in12X(img,x0,x1,x)
#define cimg_for_in12XZ(img,x0,z0,x1,z1,x,z) cimg_for_in12Z(img,z0,z1,z) cimg_for_in12X(img,x0,x1,x)
#define cimg_for_in12XV(img,x0,v0,x1,v1,x,v) cimg_for_in12V(img,v0,v1,v) cimg_for_in12X(img,x0,x1,x)
#define cimg_for_in12YZ(img,y0,z0,y1,z1,y,z) cimg_for_in12Z(img,z0,z1,z) cimg_for_in12Y(img,y0,y1,y)
#define cimg_for_in12YV(img,y0,v0,y1,v1,y,v) cimg_for_in12V(img,v0,v1,v) cimg_for_in12Y(img,y0,y1,y)
#define cimg_for_in12ZV(img,z0,v0,z1,v1,z,v) cimg_for_in12V(img,v0,v1,v) cimg_for_in12Z(img,z0,z1,z)
#define cimg_for_in12XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in12Z(img,z0,z1,z) cimg_for_in12XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in12XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in12V(img,v0,v1,v) cimg_for_in12XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in12YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in12V(img,v0,v1,v) cimg_for_in12YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in12XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in12V(img,v0,v1,v) cimg_for_in12XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for12x12(img,x,y,z,v,I) \
 cimg_for12((img).height,y) for (int x = 0, \
 _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = (img)(0,_p5##y,z,v)), \
 (I[12] = I[13] = I[14] = I[15] = I[16] = I[17] = (img)(0,_p4##y,z,v)), \
 (I[24] = I[25] = I[26] = I[27] = I[28] = I[29] = (img)(0,_p3##y,z,v)), \
 (I[36] = I[37] = I[38] = I[39] = I[40] = I[41] = (img)(0,_p2##y,z,v)), \
 (I[48] = I[49] = I[50] = I[51] = I[52] = I[53] = (img)(0,_p1##y,z,v)), \
 (I[60] = I[61] = I[62] = I[63] = I[64] = I[65] = (img)(0,y,z,v)), \
 (I[72] = I[73] = I[74] = I[75] = I[76] = I[77] = (img)(0,_n1##y,z,v)), \
 (I[84] = I[85] = I[86] = I[87] = I[88] = I[89] = (img)(0,_n2##y,z,v)), \
 (I[96] = I[97] = I[98] = I[99] = I[100] = I[101] = (img)(0,_n3##y,z,v)), \
 (I[108] = I[109] = I[110] = I[111] = I[112] = I[113] = (img)(0,_n4##y,z,v)), \
 (I[120] = I[121] = I[122] = I[123] = I[124] = I[125] = (img)(0,_n5##y,z,v)), \
 (I[132] = I[133] = I[134] = I[135] = I[136] = I[137] = (img)(0,_n6##y,z,v)), \
 (I[6] = (img)(_n1##x,_p5##y,z,v)), \
 (I[18] = (img)(_n1##x,_p4##y,z,v)), \
 (I[30] = (img)(_n1##x,_p3##y,z,v)), \
 (I[42] = (img)(_n1##x,_p2##y,z,v)), \
 (I[54] = (img)(_n1##x,_p1##y,z,v)), \
 (I[66] = (img)(_n1##x,y,z,v)), \
 (I[78] = (img)(_n1##x,_n1##y,z,v)), \
 (I[90] = (img)(_n1##x,_n2##y,z,v)), \
 (I[102] = (img)(_n1##x,_n3##y,z,v)), \
 (I[114] = (img)(_n1##x,_n4##y,z,v)), \
 (I[126] = (img)(_n1##x,_n5##y,z,v)), \
 (I[138] = (img)(_n1##x,_n6##y,z,v)), \
 (I[7] = (img)(_n2##x,_p5##y,z,v)), \
 (I[19] = (img)(_n2##x,_p4##y,z,v)), \
 (I[31] = (img)(_n2##x,_p3##y,z,v)), \
 (I[43] = (img)(_n2##x,_p2##y,z,v)), \
 (I[55] = (img)(_n2##x,_p1##y,z,v)), \
 (I[67] = (img)(_n2##x,y,z,v)), \
 (I[79] = (img)(_n2##x,_n1##y,z,v)), \
 (I[91] = (img)(_n2##x,_n2##y,z,v)), \
 (I[103] = (img)(_n2##x,_n3##y,z,v)), \
 (I[115] = (img)(_n2##x,_n4##y,z,v)), \
 (I[127] = (img)(_n2##x,_n5##y,z,v)), \
 (I[139] = (img)(_n2##x,_n6##y,z,v)), \
 (I[8] = (img)(_n3##x,_p5##y,z,v)), \
 (I[20] = (img)(_n3##x,_p4##y,z,v)), \
 (I[32] = (img)(_n3##x,_p3##y,z,v)), \
 (I[44] = (img)(_n3##x,_p2##y,z,v)), \
 (I[56] = (img)(_n3##x,_p1##y,z,v)), \
 (I[68] = (img)(_n3##x,y,z,v)), \
 (I[80] = (img)(_n3##x,_n1##y,z,v)), \
 (I[92] = (img)(_n3##x,_n2##y,z,v)), \
 (I[104] = (img)(_n3##x,_n3##y,z,v)), \
 (I[116] = (img)(_n3##x,_n4##y,z,v)), \
 (I[128] = (img)(_n3##x,_n5##y,z,v)), \
 (I[140] = (img)(_n3##x,_n6##y,z,v)), \
 (I[9] = (img)(_n4##x,_p5##y,z,v)), \
 (I[21] = (img)(_n4##x,_p4##y,z,v)), \
 (I[33] = (img)(_n4##x,_p3##y,z,v)), \
 (I[45] = (img)(_n4##x,_p2##y,z,v)), \
 (I[57] = (img)(_n4##x,_p1##y,z,v)), \
 (I[69] = (img)(_n4##x,y,z,v)), \
 (I[81] = (img)(_n4##x,_n1##y,z,v)), \
 (I[93] = (img)(_n4##x,_n2##y,z,v)), \
 (I[105] = (img)(_n4##x,_n3##y,z,v)), \
 (I[117] = (img)(_n4##x,_n4##y,z,v)), \
 (I[129] = (img)(_n4##x,_n5##y,z,v)), \
 (I[141] = (img)(_n4##x,_n6##y,z,v)), \
 (I[10] = (img)(_n5##x,_p5##y,z,v)), \
 (I[22] = (img)(_n5##x,_p4##y,z,v)), \
 (I[34] = (img)(_n5##x,_p3##y,z,v)), \
 (I[46] = (img)(_n5##x,_p2##y,z,v)), \
 (I[58] = (img)(_n5##x,_p1##y,z,v)), \
 (I[70] = (img)(_n5##x,y,z,v)), \
 (I[82] = (img)(_n5##x,_n1##y,z,v)), \
 (I[94] = (img)(_n5##x,_n2##y,z,v)), \
 (I[106] = (img)(_n5##x,_n3##y,z,v)), \
 (I[118] = (img)(_n5##x,_n4##y,z,v)), \
 (I[130] = (img)(_n5##x,_n5##y,z,v)), \
 (I[142] = (img)(_n5##x,_n6##y,z,v)), \
 6>=((img).width)?(int)((img).width)-1:6); \
 (_n6##x<(int)((img).width) && ( \
 (I[11] = (img)(_n6##x,_p5##y,z,v)), \
 (I[23] = (img)(_n6##x,_p4##y,z,v)), \
 (I[35] = (img)(_n6##x,_p3##y,z,v)), \
 (I[47] = (img)(_n6##x,_p2##y,z,v)), \
 (I[59] = (img)(_n6##x,_p1##y,z,v)), \
 (I[71] = (img)(_n6##x,y,z,v)), \
 (I[83] = (img)(_n6##x,_n1##y,z,v)), \
 (I[95] = (img)(_n6##x,_n2##y,z,v)), \
 (I[107] = (img)(_n6##x,_n3##y,z,v)), \
 (I[119] = (img)(_n6##x,_n4##y,z,v)), \
 (I[131] = (img)(_n6##x,_n5##y,z,v)), \
 (I[143] = (img)(_n6##x,_n6##y,z,v)),1)) || \
 _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], \
 I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], \
 I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], \
 I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], \
 I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], \
 I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], \
 I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], \
 I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], \
 I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x)

#define cimg_for_in12x12(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in12((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = (int)( \
 (I[0] = (img)(_p5##x,_p5##y,z,v)), \
 (I[12] = (img)(_p5##x,_p4##y,z,v)), \
 (I[24] = (img)(_p5##x,_p3##y,z,v)), \
 (I[36] = (img)(_p5##x,_p2##y,z,v)), \
 (I[48] = (img)(_p5##x,_p1##y,z,v)), \
 (I[60] = (img)(_p5##x,y,z,v)), \
 (I[72] = (img)(_p5##x,_n1##y,z,v)), \
 (I[84] = (img)(_p5##x,_n2##y,z,v)), \
 (I[96] = (img)(_p5##x,_n3##y,z,v)), \
 (I[108] = (img)(_p5##x,_n4##y,z,v)), \
 (I[120] = (img)(_p5##x,_n5##y,z,v)), \
 (I[132] = (img)(_p5##x,_n6##y,z,v)), \
 (I[1] = (img)(_p4##x,_p5##y,z,v)), \
 (I[13] = (img)(_p4##x,_p4##y,z,v)), \
 (I[25] = (img)(_p4##x,_p3##y,z,v)), \
 (I[37] = (img)(_p4##x,_p2##y,z,v)), \
 (I[49] = (img)(_p4##x,_p1##y,z,v)), \
 (I[61] = (img)(_p4##x,y,z,v)), \
 (I[73] = (img)(_p4##x,_n1##y,z,v)), \
 (I[85] = (img)(_p4##x,_n2##y,z,v)), \
 (I[97] = (img)(_p4##x,_n3##y,z,v)), \
 (I[109] = (img)(_p4##x,_n4##y,z,v)), \
 (I[121] = (img)(_p4##x,_n5##y,z,v)), \
 (I[133] = (img)(_p4##x,_n6##y,z,v)), \
 (I[2] = (img)(_p3##x,_p5##y,z,v)), \
 (I[14] = (img)(_p3##x,_p4##y,z,v)), \
 (I[26] = (img)(_p3##x,_p3##y,z,v)), \
 (I[38] = (img)(_p3##x,_p2##y,z,v)), \
 (I[50] = (img)(_p3##x,_p1##y,z,v)), \
 (I[62] = (img)(_p3##x,y,z,v)), \
 (I[74] = (img)(_p3##x,_n1##y,z,v)), \
 (I[86] = (img)(_p3##x,_n2##y,z,v)), \
 (I[98] = (img)(_p3##x,_n3##y,z,v)), \
 (I[110] = (img)(_p3##x,_n4##y,z,v)), \
 (I[122] = (img)(_p3##x,_n5##y,z,v)), \
 (I[134] = (img)(_p3##x,_n6##y,z,v)), \
 (I[3] = (img)(_p2##x,_p5##y,z,v)), \
 (I[15] = (img)(_p2##x,_p4##y,z,v)), \
 (I[27] = (img)(_p2##x,_p3##y,z,v)), \
 (I[39] = (img)(_p2##x,_p2##y,z,v)), \
 (I[51] = (img)(_p2##x,_p1##y,z,v)), \
 (I[63] = (img)(_p2##x,y,z,v)), \
 (I[75] = (img)(_p2##x,_n1##y,z,v)), \
 (I[87] = (img)(_p2##x,_n2##y,z,v)), \
 (I[99] = (img)(_p2##x,_n3##y,z,v)), \
 (I[111] = (img)(_p2##x,_n4##y,z,v)), \
 (I[123] = (img)(_p2##x,_n5##y,z,v)), \
 (I[135] = (img)(_p2##x,_n6##y,z,v)), \
 (I[4] = (img)(_p1##x,_p5##y,z,v)), \
 (I[16] = (img)(_p1##x,_p4##y,z,v)), \
 (I[28] = (img)(_p1##x,_p3##y,z,v)), \
 (I[40] = (img)(_p1##x,_p2##y,z,v)), \
 (I[52] = (img)(_p1##x,_p1##y,z,v)), \
 (I[64] = (img)(_p1##x,y,z,v)), \
 (I[76] = (img)(_p1##x,_n1##y,z,v)), \
 (I[88] = (img)(_p1##x,_n2##y,z,v)), \
 (I[100] = (img)(_p1##x,_n3##y,z,v)), \
 (I[112] = (img)(_p1##x,_n4##y,z,v)), \
 (I[124] = (img)(_p1##x,_n5##y,z,v)), \
 (I[136] = (img)(_p1##x,_n6##y,z,v)), \
 (I[5] = (img)(x,_p5##y,z,v)), \
 (I[17] = (img)(x,_p4##y,z,v)), \
 (I[29] = (img)(x,_p3##y,z,v)), \
 (I[41] = (img)(x,_p2##y,z,v)), \
 (I[53] = (img)(x,_p1##y,z,v)), \
 (I[65] = (img)(x,y,z,v)), \
 (I[77] = (img)(x,_n1##y,z,v)), \
 (I[89] = (img)(x,_n2##y,z,v)), \
 (I[101] = (img)(x,_n3##y,z,v)), \
 (I[113] = (img)(x,_n4##y,z,v)), \
 (I[125] = (img)(x,_n5##y,z,v)), \
 (I[137] = (img)(x,_n6##y,z,v)), \
 (I[6] = (img)(_n1##x,_p5##y,z,v)), \
 (I[18] = (img)(_n1##x,_p4##y,z,v)), \
 (I[30] = (img)(_n1##x,_p3##y,z,v)), \
 (I[42] = (img)(_n1##x,_p2##y,z,v)), \
 (I[54] = (img)(_n1##x,_p1##y,z,v)), \
 (I[66] = (img)(_n1##x,y,z,v)), \
 (I[78] = (img)(_n1##x,_n1##y,z,v)), \
 (I[90] = (img)(_n1##x,_n2##y,z,v)), \
 (I[102] = (img)(_n1##x,_n3##y,z,v)), \
 (I[114] = (img)(_n1##x,_n4##y,z,v)), \
 (I[126] = (img)(_n1##x,_n5##y,z,v)), \
 (I[138] = (img)(_n1##x,_n6##y,z,v)), \
 (I[7] = (img)(_n2##x,_p5##y,z,v)), \
 (I[19] = (img)(_n2##x,_p4##y,z,v)), \
 (I[31] = (img)(_n2##x,_p3##y,z,v)), \
 (I[43] = (img)(_n2##x,_p2##y,z,v)), \
 (I[55] = (img)(_n2##x,_p1##y,z,v)), \
 (I[67] = (img)(_n2##x,y,z,v)), \
 (I[79] = (img)(_n2##x,_n1##y,z,v)), \
 (I[91] = (img)(_n2##x,_n2##y,z,v)), \
 (I[103] = (img)(_n2##x,_n3##y,z,v)), \
 (I[115] = (img)(_n2##x,_n4##y,z,v)), \
 (I[127] = (img)(_n2##x,_n5##y,z,v)), \
 (I[139] = (img)(_n2##x,_n6##y,z,v)), \
 (I[8] = (img)(_n3##x,_p5##y,z,v)), \
 (I[20] = (img)(_n3##x,_p4##y,z,v)), \
 (I[32] = (img)(_n3##x,_p3##y,z,v)), \
 (I[44] = (img)(_n3##x,_p2##y,z,v)), \
 (I[56] = (img)(_n3##x,_p1##y,z,v)), \
 (I[68] = (img)(_n3##x,y,z,v)), \
 (I[80] = (img)(_n3##x,_n1##y,z,v)), \
 (I[92] = (img)(_n3##x,_n2##y,z,v)), \
 (I[104] = (img)(_n3##x,_n3##y,z,v)), \
 (I[116] = (img)(_n3##x,_n4##y,z,v)), \
 (I[128] = (img)(_n3##x,_n5##y,z,v)), \
 (I[140] = (img)(_n3##x,_n6##y,z,v)), \
 (I[9] = (img)(_n4##x,_p5##y,z,v)), \
 (I[21] = (img)(_n4##x,_p4##y,z,v)), \
 (I[33] = (img)(_n4##x,_p3##y,z,v)), \
 (I[45] = (img)(_n4##x,_p2##y,z,v)), \
 (I[57] = (img)(_n4##x,_p1##y,z,v)), \
 (I[69] = (img)(_n4##x,y,z,v)), \
 (I[81] = (img)(_n4##x,_n1##y,z,v)), \
 (I[93] = (img)(_n4##x,_n2##y,z,v)), \
 (I[105] = (img)(_n4##x,_n3##y,z,v)), \
 (I[117] = (img)(_n4##x,_n4##y,z,v)), \
 (I[129] = (img)(_n4##x,_n5##y,z,v)), \
 (I[141] = (img)(_n4##x,_n6##y,z,v)), \
 (I[10] = (img)(_n5##x,_p5##y,z,v)), \
 (I[22] = (img)(_n5##x,_p4##y,z,v)), \
 (I[34] = (img)(_n5##x,_p3##y,z,v)), \
 (I[46] = (img)(_n5##x,_p2##y,z,v)), \
 (I[58] = (img)(_n5##x,_p1##y,z,v)), \
 (I[70] = (img)(_n5##x,y,z,v)), \
 (I[82] = (img)(_n5##x,_n1##y,z,v)), \
 (I[94] = (img)(_n5##x,_n2##y,z,v)), \
 (I[106] = (img)(_n5##x,_n3##y,z,v)), \
 (I[118] = (img)(_n5##x,_n4##y,z,v)), \
 (I[130] = (img)(_n5##x,_n5##y,z,v)), \
 (I[142] = (img)(_n5##x,_n6##y,z,v)), \
 x+6>=(int)((img).width)?(int)((img).width)-1:x+6); \
 x<=(int)(x1) && ((_n6##x<(int)((img).width) && ( \
 (I[11] = (img)(_n6##x,_p5##y,z,v)), \
 (I[23] = (img)(_n6##x,_p4##y,z,v)), \
 (I[35] = (img)(_n6##x,_p3##y,z,v)), \
 (I[47] = (img)(_n6##x,_p2##y,z,v)), \
 (I[59] = (img)(_n6##x,_p1##y,z,v)), \
 (I[71] = (img)(_n6##x,y,z,v)), \
 (I[83] = (img)(_n6##x,_n1##y,z,v)), \
 (I[95] = (img)(_n6##x,_n2##y,z,v)), \
 (I[107] = (img)(_n6##x,_n3##y,z,v)), \
 (I[119] = (img)(_n6##x,_n4##y,z,v)), \
 (I[131] = (img)(_n6##x,_n5##y,z,v)), \
 (I[143] = (img)(_n6##x,_n6##y,z,v)),1)) || \
 _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], \
 I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], \
 I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], \
 I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], \
 I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], \
 I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], \
 I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], \
 I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], \
 I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x)

#define cimg_get12x12(img,x,y,z,v,I) \
 I[0] = (img)(_p5##x,_p5##y,z,v), I[1] = (img)(_p4##x,_p5##y,z,v), I[2] = (img)(_p3##x,_p5##y,z,v), I[3] = (img)(_p2##x,_p5##y,z,v), I[4] = (img)(_p1##x,_p5##y,z,v), I[5] = (img)(x,_p5##y,z,v), I[6] = (img)(_n1##x,_p5##y,z,v), I[7] = (img)(_n2##x,_p5##y,z,v), I[8] = (img)(_n3##x,_p5##y,z,v), I[9] = (img)(_n4##x,_p5##y,z,v), I[10] = (img)(_n5##x,_p5##y,z,v), I[11] = (img)(_n6##x,_p5##y,z,v), \
 I[12] = (img)(_p5##x,_p4##y,z,v), I[13] = (img)(_p4##x,_p4##y,z,v), I[14] = (img)(_p3##x,_p4##y,z,v), I[15] = (img)(_p2##x,_p4##y,z,v), I[16] = (img)(_p1##x,_p4##y,z,v), I[17] = (img)(x,_p4##y,z,v), I[18] = (img)(_n1##x,_p4##y,z,v), I[19] = (img)(_n2##x,_p4##y,z,v), I[20] = (img)(_n3##x,_p4##y,z,v), I[21] = (img)(_n4##x,_p4##y,z,v), I[22] = (img)(_n5##x,_p4##y,z,v), I[23] = (img)(_n6##x,_p4##y,z,v), \
 I[24] = (img)(_p5##x,_p3##y,z,v), I[25] = (img)(_p4##x,_p3##y,z,v), I[26] = (img)(_p3##x,_p3##y,z,v), I[27] = (img)(_p2##x,_p3##y,z,v), I[28] = (img)(_p1##x,_p3##y,z,v), I[29] = (img)(x,_p3##y,z,v), I[30] = (img)(_n1##x,_p3##y,z,v), I[31] = (img)(_n2##x,_p3##y,z,v), I[32] = (img)(_n3##x,_p3##y,z,v), I[33] = (img)(_n4##x,_p3##y,z,v), I[34] = (img)(_n5##x,_p3##y,z,v), I[35] = (img)(_n6##x,_p3##y,z,v), \
 I[36] = (img)(_p5##x,_p2##y,z,v), I[37] = (img)(_p4##x,_p2##y,z,v), I[38] = (img)(_p3##x,_p2##y,z,v), I[39] = (img)(_p2##x,_p2##y,z,v), I[40] = (img)(_p1##x,_p2##y,z,v), I[41] = (img)(x,_p2##y,z,v), I[42] = (img)(_n1##x,_p2##y,z,v), I[43] = (img)(_n2##x,_p2##y,z,v), I[44] = (img)(_n3##x,_p2##y,z,v), I[45] = (img)(_n4##x,_p2##y,z,v), I[46] = (img)(_n5##x,_p2##y,z,v), I[47] = (img)(_n6##x,_p2##y,z,v), \
 I[48] = (img)(_p5##x,_p1##y,z,v), I[49] = (img)(_p4##x,_p1##y,z,v), I[50] = (img)(_p3##x,_p1##y,z,v), I[51] = (img)(_p2##x,_p1##y,z,v), I[52] = (img)(_p1##x,_p1##y,z,v), I[53] = (img)(x,_p1##y,z,v), I[54] = (img)(_n1##x,_p1##y,z,v), I[55] = (img)(_n2##x,_p1##y,z,v), I[56] = (img)(_n3##x,_p1##y,z,v), I[57] = (img)(_n4##x,_p1##y,z,v), I[58] = (img)(_n5##x,_p1##y,z,v), I[59] = (img)(_n6##x,_p1##y,z,v), \
 I[60] = (img)(_p5##x,y,z,v), I[61] = (img)(_p4##x,y,z,v), I[62] = (img)(_p3##x,y,z,v), I[63] = (img)(_p2##x,y,z,v), I[64] = (img)(_p1##x,y,z,v), I[65] = (img)(x,y,z,v), I[66] = (img)(_n1##x,y,z,v), I[67] = (img)(_n2##x,y,z,v), I[68] = (img)(_n3##x,y,z,v), I[69] = (img)(_n4##x,y,z,v), I[70] = (img)(_n5##x,y,z,v), I[71] = (img)(_n6##x,y,z,v), \
 I[72] = (img)(_p5##x,_n1##y,z,v), I[73] = (img)(_p4##x,_n1##y,z,v), I[74] = (img)(_p3##x,_n1##y,z,v), I[75] = (img)(_p2##x,_n1##y,z,v), I[76] = (img)(_p1##x,_n1##y,z,v), I[77] = (img)(x,_n1##y,z,v), I[78] = (img)(_n1##x,_n1##y,z,v), I[79] = (img)(_n2##x,_n1##y,z,v), I[80] = (img)(_n3##x,_n1##y,z,v), I[81] = (img)(_n4##x,_n1##y,z,v), I[82] = (img)(_n5##x,_n1##y,z,v), I[83] = (img)(_n6##x,_n1##y,z,v), \
 I[84] = (img)(_p5##x,_n2##y,z,v), I[85] = (img)(_p4##x,_n2##y,z,v), I[86] = (img)(_p3##x,_n2##y,z,v), I[87] = (img)(_p2##x,_n2##y,z,v), I[88] = (img)(_p1##x,_n2##y,z,v), I[89] = (img)(x,_n2##y,z,v), I[90] = (img)(_n1##x,_n2##y,z,v), I[91] = (img)(_n2##x,_n2##y,z,v), I[92] = (img)(_n3##x,_n2##y,z,v), I[93] = (img)(_n4##x,_n2##y,z,v), I[94] = (img)(_n5##x,_n2##y,z,v), I[95] = (img)(_n6##x,_n2##y,z,v), \
 I[96] = (img)(_p5##x,_n3##y,z,v), I[97] = (img)(_p4##x,_n3##y,z,v), I[98] = (img)(_p3##x,_n3##y,z,v), I[99] = (img)(_p2##x,_n3##y,z,v), I[100] = (img)(_p1##x,_n3##y,z,v), I[101] = (img)(x,_n3##y,z,v), I[102] = (img)(_n1##x,_n3##y,z,v), I[103] = (img)(_n2##x,_n3##y,z,v), I[104] = (img)(_n3##x,_n3##y,z,v), I[105] = (img)(_n4##x,_n3##y,z,v), I[106] = (img)(_n5##x,_n3##y,z,v), I[107] = (img)(_n6##x,_n3##y,z,v), \
 I[108] = (img)(_p5##x,_n4##y,z,v), I[109] = (img)(_p4##x,_n4##y,z,v), I[110] = (img)(_p3##x,_n4##y,z,v), I[111] = (img)(_p2##x,_n4##y,z,v), I[112] = (img)(_p1##x,_n4##y,z,v), I[113] = (img)(x,_n4##y,z,v), I[114] = (img)(_n1##x,_n4##y,z,v), I[115] = (img)(_n2##x,_n4##y,z,v), I[116] = (img)(_n3##x,_n4##y,z,v), I[117] = (img)(_n4##x,_n4##y,z,v), I[118] = (img)(_n5##x,_n4##y,z,v), I[119] = (img)(_n6##x,_n4##y,z,v), \
 I[120] = (img)(_p5##x,_n5##y,z,v), I[121] = (img)(_p4##x,_n5##y,z,v), I[122] = (img)(_p3##x,_n5##y,z,v), I[123] = (img)(_p2##x,_n5##y,z,v), I[124] = (img)(_p1##x,_n5##y,z,v), I[125] = (img)(x,_n5##y,z,v), I[126] = (img)(_n1##x,_n5##y,z,v), I[127] = (img)(_n2##x,_n5##y,z,v), I[128] = (img)(_n3##x,_n5##y,z,v), I[129] = (img)(_n4##x,_n5##y,z,v), I[130] = (img)(_n5##x,_n5##y,z,v), I[131] = (img)(_n6##x,_n5##y,z,v), \
 I[132] = (img)(_p5##x,_n6##y,z,v), I[133] = (img)(_p4##x,_n6##y,z,v), I[134] = (img)(_p3##x,_n6##y,z,v), I[135] = (img)(_p2##x,_n6##y,z,v), I[136] = (img)(_p1##x,_n6##y,z,v), I[137] = (img)(x,_n6##y,z,v), I[138] = (img)(_n1##x,_n6##y,z,v), I[139] = (img)(_n2##x,_n6##y,z,v), I[140] = (img)(_n3##x,_n6##y,z,v), I[141] = (img)(_n4##x,_n6##y,z,v), I[142] = (img)(_n5##x,_n6##y,z,v), I[143] = (img)(_n6##x,_n6##y,z,v);

// Define 13x13 loop macros for CImg
//----------------------------------
#define cimg_for13(bound,i) for (int i = 0, \
 _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6; \
 _n6##i<(int)(bound) || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i)

#define cimg_for13X(img,x) cimg_for13((img).width,x)
#define cimg_for13Y(img,y) cimg_for13((img).height,y)
#define cimg_for13Z(img,z) cimg_for13((img).depth,z)
#define cimg_for13V(img,v) cimg_for13((img).dim,v)
#define cimg_for13XY(img,x,y) cimg_for13Y(img,y) cimg_for13X(img,x)
#define cimg_for13XZ(img,x,z) cimg_for13Z(img,z) cimg_for13X(img,x)
#define cimg_for13XV(img,x,v) cimg_for13V(img,v) cimg_for13X(img,x)
#define cimg_for13YZ(img,y,z) cimg_for13Z(img,z) cimg_for13Y(img,y)
#define cimg_for13YV(img,y,v) cimg_for13V(img,v) cimg_for13Y(img,y)
#define cimg_for13ZV(img,z,v) cimg_for13V(img,v) cimg_for13Z(img,z)
#define cimg_for13XYZ(img,x,y,z) cimg_for13Z(img,z) cimg_for13XY(img,x,y)
#define cimg_for13XZV(img,x,z,v) cimg_for13V(img,v) cimg_for13XZ(img,x,z)
#define cimg_for13YZV(img,y,z,v) cimg_for13V(img,v) cimg_for13YZ(img,y,z)
#define cimg_for13XYZV(img,x,y,z,v) cimg_for13V(img,v) cimg_for13XYZ(img,x,y,z)

#define cimg_for_in13(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6; \
 i<=(int)(i1) && (_n6##i<(int)(bound) || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i)

#define cimg_for_in13X(img,x0,x1,x) cimg_for_in13((img).width,x0,x1,x)
#define cimg_for_in13Y(img,y0,y1,y) cimg_for_in13((img).height,y0,y1,y)
#define cimg_for_in13Z(img,z0,z1,z) cimg_for_in13((img).depth,z0,z1,z)
#define cimg_for_in13V(img,v0,v1,v) cimg_for_in13((img).dim,v0,v1,v)
#define cimg_for_in13XY(img,x0,y0,x1,y1,x,y) cimg_for_in13Y(img,y0,y1,y) cimg_for_in13X(img,x0,x1,x)
#define cimg_for_in13XZ(img,x0,z0,x1,z1,x,z) cimg_for_in13Z(img,z0,z1,z) cimg_for_in13X(img,x0,x1,x)
#define cimg_for_in13XV(img,x0,v0,x1,v1,x,v) cimg_for_in13V(img,v0,v1,v) cimg_for_in13X(img,x0,x1,x)
#define cimg_for_in13YZ(img,y0,z0,y1,z1,y,z) cimg_for_in13Z(img,z0,z1,z) cimg_for_in13Y(img,y0,y1,y)
#define cimg_for_in13YV(img,y0,v0,y1,v1,y,v) cimg_for_in13V(img,v0,v1,v) cimg_for_in13Y(img,y0,y1,y)
#define cimg_for_in13ZV(img,z0,v0,z1,v1,z,v) cimg_for_in13V(img,v0,v1,v) cimg_for_in13Z(img,z0,z1,z)
#define cimg_for_in13XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in13Z(img,z0,z1,z) cimg_for_in13XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in13XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in13V(img,v0,v1,v) cimg_for_in13XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in13YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in13V(img,v0,v1,v) cimg_for_in13YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in13XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in13V(img,v0,v1,v) cimg_for_in13XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for13x13(img,x,y,z,v,I) \
 cimg_for13((img).height,y) for (int x = 0, \
 _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = (img)(0,_p6##y,z,v)), \
 (I[13] = I[14] = I[15] = I[16] = I[17] = I[18] = I[19] = (img)(0,_p5##y,z,v)), \
 (I[26] = I[27] = I[28] = I[29] = I[30] = I[31] = I[32] = (img)(0,_p4##y,z,v)), \
 (I[39] = I[40] = I[41] = I[42] = I[43] = I[44] = I[45] = (img)(0,_p3##y,z,v)), \
 (I[52] = I[53] = I[54] = I[55] = I[56] = I[57] = I[58] = (img)(0,_p2##y,z,v)), \
 (I[65] = I[66] = I[67] = I[68] = I[69] = I[70] = I[71] = (img)(0,_p1##y,z,v)), \
 (I[78] = I[79] = I[80] = I[81] = I[82] = I[83] = I[84] = (img)(0,y,z,v)), \
 (I[91] = I[92] = I[93] = I[94] = I[95] = I[96] = I[97] = (img)(0,_n1##y,z,v)), \
 (I[104] = I[105] = I[106] = I[107] = I[108] = I[109] = I[110] = (img)(0,_n2##y,z,v)), \
 (I[117] = I[118] = I[119] = I[120] = I[121] = I[122] = I[123] = (img)(0,_n3##y,z,v)), \
 (I[130] = I[131] = I[132] = I[133] = I[134] = I[135] = I[136] = (img)(0,_n4##y,z,v)), \
 (I[143] = I[144] = I[145] = I[146] = I[147] = I[148] = I[149] = (img)(0,_n5##y,z,v)), \
 (I[156] = I[157] = I[158] = I[159] = I[160] = I[161] = I[162] = (img)(0,_n6##y,z,v)), \
 (I[7] = (img)(_n1##x,_p6##y,z,v)), \
 (I[20] = (img)(_n1##x,_p5##y,z,v)), \
 (I[33] = (img)(_n1##x,_p4##y,z,v)), \
 (I[46] = (img)(_n1##x,_p3##y,z,v)), \
 (I[59] = (img)(_n1##x,_p2##y,z,v)), \
 (I[72] = (img)(_n1##x,_p1##y,z,v)), \
 (I[85] = (img)(_n1##x,y,z,v)), \
 (I[98] = (img)(_n1##x,_n1##y,z,v)), \
 (I[111] = (img)(_n1##x,_n2##y,z,v)), \
 (I[124] = (img)(_n1##x,_n3##y,z,v)), \
 (I[137] = (img)(_n1##x,_n4##y,z,v)), \
 (I[150] = (img)(_n1##x,_n5##y,z,v)), \
 (I[163] = (img)(_n1##x,_n6##y,z,v)), \
 (I[8] = (img)(_n2##x,_p6##y,z,v)), \
 (I[21] = (img)(_n2##x,_p5##y,z,v)), \
 (I[34] = (img)(_n2##x,_p4##y,z,v)), \
 (I[47] = (img)(_n2##x,_p3##y,z,v)), \
 (I[60] = (img)(_n2##x,_p2##y,z,v)), \
 (I[73] = (img)(_n2##x,_p1##y,z,v)), \
 (I[86] = (img)(_n2##x,y,z,v)), \
 (I[99] = (img)(_n2##x,_n1##y,z,v)), \
 (I[112] = (img)(_n2##x,_n2##y,z,v)), \
 (I[125] = (img)(_n2##x,_n3##y,z,v)), \
 (I[138] = (img)(_n2##x,_n4##y,z,v)), \
 (I[151] = (img)(_n2##x,_n5##y,z,v)), \
 (I[164] = (img)(_n2##x,_n6##y,z,v)), \
 (I[9] = (img)(_n3##x,_p6##y,z,v)), \
 (I[22] = (img)(_n3##x,_p5##y,z,v)), \
 (I[35] = (img)(_n3##x,_p4##y,z,v)), \
 (I[48] = (img)(_n3##x,_p3##y,z,v)), \
 (I[61] = (img)(_n3##x,_p2##y,z,v)), \
 (I[74] = (img)(_n3##x,_p1##y,z,v)), \
 (I[87] = (img)(_n3##x,y,z,v)), \
 (I[100] = (img)(_n3##x,_n1##y,z,v)), \
 (I[113] = (img)(_n3##x,_n2##y,z,v)), \
 (I[126] = (img)(_n3##x,_n3##y,z,v)), \
 (I[139] = (img)(_n3##x,_n4##y,z,v)), \
 (I[152] = (img)(_n3##x,_n5##y,z,v)), \
 (I[165] = (img)(_n3##x,_n6##y,z,v)), \
 (I[10] = (img)(_n4##x,_p6##y,z,v)), \
 (I[23] = (img)(_n4##x,_p5##y,z,v)), \
 (I[36] = (img)(_n4##x,_p4##y,z,v)), \
 (I[49] = (img)(_n4##x,_p3##y,z,v)), \
 (I[62] = (img)(_n4##x,_p2##y,z,v)), \
 (I[75] = (img)(_n4##x,_p1##y,z,v)), \
 (I[88] = (img)(_n4##x,y,z,v)), \
 (I[101] = (img)(_n4##x,_n1##y,z,v)), \
 (I[114] = (img)(_n4##x,_n2##y,z,v)), \
 (I[127] = (img)(_n4##x,_n3##y,z,v)), \
 (I[140] = (img)(_n4##x,_n4##y,z,v)), \
 (I[153] = (img)(_n4##x,_n5##y,z,v)), \
 (I[166] = (img)(_n4##x,_n6##y,z,v)), \
 (I[11] = (img)(_n5##x,_p6##y,z,v)), \
 (I[24] = (img)(_n5##x,_p5##y,z,v)), \
 (I[37] = (img)(_n5##x,_p4##y,z,v)), \
 (I[50] = (img)(_n5##x,_p3##y,z,v)), \
 (I[63] = (img)(_n5##x,_p2##y,z,v)), \
 (I[76] = (img)(_n5##x,_p1##y,z,v)), \
 (I[89] = (img)(_n5##x,y,z,v)), \
 (I[102] = (img)(_n5##x,_n1##y,z,v)), \
 (I[115] = (img)(_n5##x,_n2##y,z,v)), \
 (I[128] = (img)(_n5##x,_n3##y,z,v)), \
 (I[141] = (img)(_n5##x,_n4##y,z,v)), \
 (I[154] = (img)(_n5##x,_n5##y,z,v)), \
 (I[167] = (img)(_n5##x,_n6##y,z,v)), \
 6>=((img).width)?(int)((img).width)-1:6); \
 (_n6##x<(int)((img).width) && ( \
 (I[12] = (img)(_n6##x,_p6##y,z,v)), \
 (I[25] = (img)(_n6##x,_p5##y,z,v)), \
 (I[38] = (img)(_n6##x,_p4##y,z,v)), \
 (I[51] = (img)(_n6##x,_p3##y,z,v)), \
 (I[64] = (img)(_n6##x,_p2##y,z,v)), \
 (I[77] = (img)(_n6##x,_p1##y,z,v)), \
 (I[90] = (img)(_n6##x,y,z,v)), \
 (I[103] = (img)(_n6##x,_n1##y,z,v)), \
 (I[116] = (img)(_n6##x,_n2##y,z,v)), \
 (I[129] = (img)(_n6##x,_n3##y,z,v)), \
 (I[142] = (img)(_n6##x,_n4##y,z,v)), \
 (I[155] = (img)(_n6##x,_n5##y,z,v)), \
 (I[168] = (img)(_n6##x,_n6##y,z,v)),1)) || \
 _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], \
 I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], \
 I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], \
 I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], \
 I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], \
 I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], \
 I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], \
 I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], \
 I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], \
 I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], \
 I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], \
 I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], \
 I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], \
 _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x)

#define cimg_for_in13x13(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in13((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = (int)( \
 (I[0] = (img)(_p6##x,_p6##y,z,v)), \
 (I[13] = (img)(_p6##x,_p5##y,z,v)), \
 (I[26] = (img)(_p6##x,_p4##y,z,v)), \
 (I[39] = (img)(_p6##x,_p3##y,z,v)), \
 (I[52] = (img)(_p6##x,_p2##y,z,v)), \
 (I[65] = (img)(_p6##x,_p1##y,z,v)), \
 (I[78] = (img)(_p6##x,y,z,v)), \
 (I[91] = (img)(_p6##x,_n1##y,z,v)), \
 (I[104] = (img)(_p6##x,_n2##y,z,v)), \
 (I[117] = (img)(_p6##x,_n3##y,z,v)), \
 (I[130] = (img)(_p6##x,_n4##y,z,v)), \
 (I[143] = (img)(_p6##x,_n5##y,z,v)), \
 (I[156] = (img)(_p6##x,_n6##y,z,v)), \
 (I[1] = (img)(_p5##x,_p6##y,z,v)), \
 (I[14] = (img)(_p5##x,_p5##y,z,v)), \
 (I[27] = (img)(_p5##x,_p4##y,z,v)), \
 (I[40] = (img)(_p5##x,_p3##y,z,v)), \
 (I[53] = (img)(_p5##x,_p2##y,z,v)), \
 (I[66] = (img)(_p5##x,_p1##y,z,v)), \
 (I[79] = (img)(_p5##x,y,z,v)), \
 (I[92] = (img)(_p5##x,_n1##y,z,v)), \
 (I[105] = (img)(_p5##x,_n2##y,z,v)), \
 (I[118] = (img)(_p5##x,_n3##y,z,v)), \
 (I[131] = (img)(_p5##x,_n4##y,z,v)), \
 (I[144] = (img)(_p5##x,_n5##y,z,v)), \
 (I[157] = (img)(_p5##x,_n6##y,z,v)), \
 (I[2] = (img)(_p4##x,_p6##y,z,v)), \
 (I[15] = (img)(_p4##x,_p5##y,z,v)), \
 (I[28] = (img)(_p4##x,_p4##y,z,v)), \
 (I[41] = (img)(_p4##x,_p3##y,z,v)), \
 (I[54] = (img)(_p4##x,_p2##y,z,v)), \
 (I[67] = (img)(_p4##x,_p1##y,z,v)), \
 (I[80] = (img)(_p4##x,y,z,v)), \
 (I[93] = (img)(_p4##x,_n1##y,z,v)), \
 (I[106] = (img)(_p4##x,_n2##y,z,v)), \
 (I[119] = (img)(_p4##x,_n3##y,z,v)), \
 (I[132] = (img)(_p4##x,_n4##y,z,v)), \
 (I[145] = (img)(_p4##x,_n5##y,z,v)), \
 (I[158] = (img)(_p4##x,_n6##y,z,v)), \
 (I[3] = (img)(_p3##x,_p6##y,z,v)), \
 (I[16] = (img)(_p3##x,_p5##y,z,v)), \
 (I[29] = (img)(_p3##x,_p4##y,z,v)), \
 (I[42] = (img)(_p3##x,_p3##y,z,v)), \
 (I[55] = (img)(_p3##x,_p2##y,z,v)), \
 (I[68] = (img)(_p3##x,_p1##y,z,v)), \
 (I[81] = (img)(_p3##x,y,z,v)), \
 (I[94] = (img)(_p3##x,_n1##y,z,v)), \
 (I[107] = (img)(_p3##x,_n2##y,z,v)), \
 (I[120] = (img)(_p3##x,_n3##y,z,v)), \
 (I[133] = (img)(_p3##x,_n4##y,z,v)), \
 (I[146] = (img)(_p3##x,_n5##y,z,v)), \
 (I[159] = (img)(_p3##x,_n6##y,z,v)), \
 (I[4] = (img)(_p2##x,_p6##y,z,v)), \
 (I[17] = (img)(_p2##x,_p5##y,z,v)), \
 (I[30] = (img)(_p2##x,_p4##y,z,v)), \
 (I[43] = (img)(_p2##x,_p3##y,z,v)), \
 (I[56] = (img)(_p2##x,_p2##y,z,v)), \
 (I[69] = (img)(_p2##x,_p1##y,z,v)), \
 (I[82] = (img)(_p2##x,y,z,v)), \
 (I[95] = (img)(_p2##x,_n1##y,z,v)), \
 (I[108] = (img)(_p2##x,_n2##y,z,v)), \
 (I[121] = (img)(_p2##x,_n3##y,z,v)), \
 (I[134] = (img)(_p2##x,_n4##y,z,v)), \
 (I[147] = (img)(_p2##x,_n5##y,z,v)), \
 (I[160] = (img)(_p2##x,_n6##y,z,v)), \
 (I[5] = (img)(_p1##x,_p6##y,z,v)), \
 (I[18] = (img)(_p1##x,_p5##y,z,v)), \
 (I[31] = (img)(_p1##x,_p4##y,z,v)), \
 (I[44] = (img)(_p1##x,_p3##y,z,v)), \
 (I[57] = (img)(_p1##x,_p2##y,z,v)), \
 (I[70] = (img)(_p1##x,_p1##y,z,v)), \
 (I[83] = (img)(_p1##x,y,z,v)), \
 (I[96] = (img)(_p1##x,_n1##y,z,v)), \
 (I[109] = (img)(_p1##x,_n2##y,z,v)), \
 (I[122] = (img)(_p1##x,_n3##y,z,v)), \
 (I[135] = (img)(_p1##x,_n4##y,z,v)), \
 (I[148] = (img)(_p1##x,_n5##y,z,v)), \
 (I[161] = (img)(_p1##x,_n6##y,z,v)), \
 (I[6] = (img)(x,_p6##y,z,v)), \
 (I[19] = (img)(x,_p5##y,z,v)), \
 (I[32] = (img)(x,_p4##y,z,v)), \
 (I[45] = (img)(x,_p3##y,z,v)), \
 (I[58] = (img)(x,_p2##y,z,v)), \
 (I[71] = (img)(x,_p1##y,z,v)), \
 (I[84] = (img)(x,y,z,v)), \
 (I[97] = (img)(x,_n1##y,z,v)), \
 (I[110] = (img)(x,_n2##y,z,v)), \
 (I[123] = (img)(x,_n3##y,z,v)), \
 (I[136] = (img)(x,_n4##y,z,v)), \
 (I[149] = (img)(x,_n5##y,z,v)), \
 (I[162] = (img)(x,_n6##y,z,v)), \
 (I[7] = (img)(_n1##x,_p6##y,z,v)), \
 (I[20] = (img)(_n1##x,_p5##y,z,v)), \
 (I[33] = (img)(_n1##x,_p4##y,z,v)), \
 (I[46] = (img)(_n1##x,_p3##y,z,v)), \
 (I[59] = (img)(_n1##x,_p2##y,z,v)), \
 (I[72] = (img)(_n1##x,_p1##y,z,v)), \
 (I[85] = (img)(_n1##x,y,z,v)), \
 (I[98] = (img)(_n1##x,_n1##y,z,v)), \
 (I[111] = (img)(_n1##x,_n2##y,z,v)), \
 (I[124] = (img)(_n1##x,_n3##y,z,v)), \
 (I[137] = (img)(_n1##x,_n4##y,z,v)), \
 (I[150] = (img)(_n1##x,_n5##y,z,v)), \
 (I[163] = (img)(_n1##x,_n6##y,z,v)), \
 (I[8] = (img)(_n2##x,_p6##y,z,v)), \
 (I[21] = (img)(_n2##x,_p5##y,z,v)), \
 (I[34] = (img)(_n2##x,_p4##y,z,v)), \
 (I[47] = (img)(_n2##x,_p3##y,z,v)), \
 (I[60] = (img)(_n2##x,_p2##y,z,v)), \
 (I[73] = (img)(_n2##x,_p1##y,z,v)), \
 (I[86] = (img)(_n2##x,y,z,v)), \
 (I[99] = (img)(_n2##x,_n1##y,z,v)), \
 (I[112] = (img)(_n2##x,_n2##y,z,v)), \
 (I[125] = (img)(_n2##x,_n3##y,z,v)), \
 (I[138] = (img)(_n2##x,_n4##y,z,v)), \
 (I[151] = (img)(_n2##x,_n5##y,z,v)), \
 (I[164] = (img)(_n2##x,_n6##y,z,v)), \
 (I[9] = (img)(_n3##x,_p6##y,z,v)), \
 (I[22] = (img)(_n3##x,_p5##y,z,v)), \
 (I[35] = (img)(_n3##x,_p4##y,z,v)), \
 (I[48] = (img)(_n3##x,_p3##y,z,v)), \
 (I[61] = (img)(_n3##x,_p2##y,z,v)), \
 (I[74] = (img)(_n3##x,_p1##y,z,v)), \
 (I[87] = (img)(_n3##x,y,z,v)), \
 (I[100] = (img)(_n3##x,_n1##y,z,v)), \
 (I[113] = (img)(_n3##x,_n2##y,z,v)), \
 (I[126] = (img)(_n3##x,_n3##y,z,v)), \
 (I[139] = (img)(_n3##x,_n4##y,z,v)), \
 (I[152] = (img)(_n3##x,_n5##y,z,v)), \
 (I[165] = (img)(_n3##x,_n6##y,z,v)), \
 (I[10] = (img)(_n4##x,_p6##y,z,v)), \
 (I[23] = (img)(_n4##x,_p5##y,z,v)), \
 (I[36] = (img)(_n4##x,_p4##y,z,v)), \
 (I[49] = (img)(_n4##x,_p3##y,z,v)), \
 (I[62] = (img)(_n4##x,_p2##y,z,v)), \
 (I[75] = (img)(_n4##x,_p1##y,z,v)), \
 (I[88] = (img)(_n4##x,y,z,v)), \
 (I[101] = (img)(_n4##x,_n1##y,z,v)), \
 (I[114] = (img)(_n4##x,_n2##y,z,v)), \
 (I[127] = (img)(_n4##x,_n3##y,z,v)), \
 (I[140] = (img)(_n4##x,_n4##y,z,v)), \
 (I[153] = (img)(_n4##x,_n5##y,z,v)), \
 (I[166] = (img)(_n4##x,_n6##y,z,v)), \
 (I[11] = (img)(_n5##x,_p6##y,z,v)), \
 (I[24] = (img)(_n5##x,_p5##y,z,v)), \
 (I[37] = (img)(_n5##x,_p4##y,z,v)), \
 (I[50] = (img)(_n5##x,_p3##y,z,v)), \
 (I[63] = (img)(_n5##x,_p2##y,z,v)), \
 (I[76] = (img)(_n5##x,_p1##y,z,v)), \
 (I[89] = (img)(_n5##x,y,z,v)), \
 (I[102] = (img)(_n5##x,_n1##y,z,v)), \
 (I[115] = (img)(_n5##x,_n2##y,z,v)), \
 (I[128] = (img)(_n5##x,_n3##y,z,v)), \
 (I[141] = (img)(_n5##x,_n4##y,z,v)), \
 (I[154] = (img)(_n5##x,_n5##y,z,v)), \
 (I[167] = (img)(_n5##x,_n6##y,z,v)), \
 x+6>=(int)((img).width)?(int)((img).width)-1:x+6); \
 x<=(int)(x1) && ((_n6##x<(int)((img).width) && ( \
 (I[12] = (img)(_n6##x,_p6##y,z,v)), \
 (I[25] = (img)(_n6##x,_p5##y,z,v)), \
 (I[38] = (img)(_n6##x,_p4##y,z,v)), \
 (I[51] = (img)(_n6##x,_p3##y,z,v)), \
 (I[64] = (img)(_n6##x,_p2##y,z,v)), \
 (I[77] = (img)(_n6##x,_p1##y,z,v)), \
 (I[90] = (img)(_n6##x,y,z,v)), \
 (I[103] = (img)(_n6##x,_n1##y,z,v)), \
 (I[116] = (img)(_n6##x,_n2##y,z,v)), \
 (I[129] = (img)(_n6##x,_n3##y,z,v)), \
 (I[142] = (img)(_n6##x,_n4##y,z,v)), \
 (I[155] = (img)(_n6##x,_n5##y,z,v)), \
 (I[168] = (img)(_n6##x,_n6##y,z,v)),1)) || \
 _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], \
 I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], \
 I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], \
 I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], \
 I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], \
 I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], \
 I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], \
 I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], \
 I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], \
 I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], \
 I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], \
 I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], \
 I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], \
 _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x)

#define cimg_get13x13(img,x,y,z,v,I) \
 I[0] = (img)(_p6##x,_p6##y,z,v), I[1] = (img)(_p5##x,_p6##y,z,v), I[2] = (img)(_p4##x,_p6##y,z,v), I[3] = (img)(_p3##x,_p6##y,z,v), I[4] = (img)(_p2##x,_p6##y,z,v), I[5] = (img)(_p1##x,_p6##y,z,v), I[6] = (img)(x,_p6##y,z,v), I[7] = (img)(_n1##x,_p6##y,z,v), I[8] = (img)(_n2##x,_p6##y,z,v), I[9] = (img)(_n3##x,_p6##y,z,v), I[10] = (img)(_n4##x,_p6##y,z,v), I[11] = (img)(_n5##x,_p6##y,z,v), I[12] = (img)(_n6##x,_p6##y,z,v), \
 I[13] = (img)(_p6##x,_p5##y,z,v), I[14] = (img)(_p5##x,_p5##y,z,v), I[15] = (img)(_p4##x,_p5##y,z,v), I[16] = (img)(_p3##x,_p5##y,z,v), I[17] = (img)(_p2##x,_p5##y,z,v), I[18] = (img)(_p1##x,_p5##y,z,v), I[19] = (img)(x,_p5##y,z,v), I[20] = (img)(_n1##x,_p5##y,z,v), I[21] = (img)(_n2##x,_p5##y,z,v), I[22] = (img)(_n3##x,_p5##y,z,v), I[23] = (img)(_n4##x,_p5##y,z,v), I[24] = (img)(_n5##x,_p5##y,z,v), I[25] = (img)(_n6##x,_p5##y,z,v), \
 I[26] = (img)(_p6##x,_p4##y,z,v), I[27] = (img)(_p5##x,_p4##y,z,v), I[28] = (img)(_p4##x,_p4##y,z,v), I[29] = (img)(_p3##x,_p4##y,z,v), I[30] = (img)(_p2##x,_p4##y,z,v), I[31] = (img)(_p1##x,_p4##y,z,v), I[32] = (img)(x,_p4##y,z,v), I[33] = (img)(_n1##x,_p4##y,z,v), I[34] = (img)(_n2##x,_p4##y,z,v), I[35] = (img)(_n3##x,_p4##y,z,v), I[36] = (img)(_n4##x,_p4##y,z,v), I[37] = (img)(_n5##x,_p4##y,z,v), I[38] = (img)(_n6##x,_p4##y,z,v), \
 I[39] = (img)(_p6##x,_p3##y,z,v), I[40] = (img)(_p5##x,_p3##y,z,v), I[41] = (img)(_p4##x,_p3##y,z,v), I[42] = (img)(_p3##x,_p3##y,z,v), I[43] = (img)(_p2##x,_p3##y,z,v), I[44] = (img)(_p1##x,_p3##y,z,v), I[45] = (img)(x,_p3##y,z,v), I[46] = (img)(_n1##x,_p3##y,z,v), I[47] = (img)(_n2##x,_p3##y,z,v), I[48] = (img)(_n3##x,_p3##y,z,v), I[49] = (img)(_n4##x,_p3##y,z,v), I[50] = (img)(_n5##x,_p3##y,z,v), I[51] = (img)(_n6##x,_p3##y,z,v), \
 I[52] = (img)(_p6##x,_p2##y,z,v), I[53] = (img)(_p5##x,_p2##y,z,v), I[54] = (img)(_p4##x,_p2##y,z,v), I[55] = (img)(_p3##x,_p2##y,z,v), I[56] = (img)(_p2##x,_p2##y,z,v), I[57] = (img)(_p1##x,_p2##y,z,v), I[58] = (img)(x,_p2##y,z,v), I[59] = (img)(_n1##x,_p2##y,z,v), I[60] = (img)(_n2##x,_p2##y,z,v), I[61] = (img)(_n3##x,_p2##y,z,v), I[62] = (img)(_n4##x,_p2##y,z,v), I[63] = (img)(_n5##x,_p2##y,z,v), I[64] = (img)(_n6##x,_p2##y,z,v), \
 I[65] = (img)(_p6##x,_p1##y,z,v), I[66] = (img)(_p5##x,_p1##y,z,v), I[67] = (img)(_p4##x,_p1##y,z,v), I[68] = (img)(_p3##x,_p1##y,z,v), I[69] = (img)(_p2##x,_p1##y,z,v), I[70] = (img)(_p1##x,_p1##y,z,v), I[71] = (img)(x,_p1##y,z,v), I[72] = (img)(_n1##x,_p1##y,z,v), I[73] = (img)(_n2##x,_p1##y,z,v), I[74] = (img)(_n3##x,_p1##y,z,v), I[75] = (img)(_n4##x,_p1##y,z,v), I[76] = (img)(_n5##x,_p1##y,z,v), I[77] = (img)(_n6##x,_p1##y,z,v), \
 I[78] = (img)(_p6##x,y,z,v), I[79] = (img)(_p5##x,y,z,v), I[80] = (img)(_p4##x,y,z,v), I[81] = (img)(_p3##x,y,z,v), I[82] = (img)(_p2##x,y,z,v), I[83] = (img)(_p1##x,y,z,v), I[84] = (img)(x,y,z,v), I[85] = (img)(_n1##x,y,z,v), I[86] = (img)(_n2##x,y,z,v), I[87] = (img)(_n3##x,y,z,v), I[88] = (img)(_n4##x,y,z,v), I[89] = (img)(_n5##x,y,z,v), I[90] = (img)(_n6##x,y,z,v), \
 I[91] = (img)(_p6##x,_n1##y,z,v), I[92] = (img)(_p5##x,_n1##y,z,v), I[93] = (img)(_p4##x,_n1##y,z,v), I[94] = (img)(_p3##x,_n1##y,z,v), I[95] = (img)(_p2##x,_n1##y,z,v), I[96] = (img)(_p1##x,_n1##y,z,v), I[97] = (img)(x,_n1##y,z,v), I[98] = (img)(_n1##x,_n1##y,z,v), I[99] = (img)(_n2##x,_n1##y,z,v), I[100] = (img)(_n3##x,_n1##y,z,v), I[101] = (img)(_n4##x,_n1##y,z,v), I[102] = (img)(_n5##x,_n1##y,z,v), I[103] = (img)(_n6##x,_n1##y,z,v), \
 I[104] = (img)(_p6##x,_n2##y,z,v), I[105] = (img)(_p5##x,_n2##y,z,v), I[106] = (img)(_p4##x,_n2##y,z,v), I[107] = (img)(_p3##x,_n2##y,z,v), I[108] = (img)(_p2##x,_n2##y,z,v), I[109] = (img)(_p1##x,_n2##y,z,v), I[110] = (img)(x,_n2##y,z,v), I[111] = (img)(_n1##x,_n2##y,z,v), I[112] = (img)(_n2##x,_n2##y,z,v), I[113] = (img)(_n3##x,_n2##y,z,v), I[114] = (img)(_n4##x,_n2##y,z,v), I[115] = (img)(_n5##x,_n2##y,z,v), I[116] = (img)(_n6##x,_n2##y,z,v), \
 I[117] = (img)(_p6##x,_n3##y,z,v), I[118] = (img)(_p5##x,_n3##y,z,v), I[119] = (img)(_p4##x,_n3##y,z,v), I[120] = (img)(_p3##x,_n3##y,z,v), I[121] = (img)(_p2##x,_n3##y,z,v), I[122] = (img)(_p1##x,_n3##y,z,v), I[123] = (img)(x,_n3##y,z,v), I[124] = (img)(_n1##x,_n3##y,z,v), I[125] = (img)(_n2##x,_n3##y,z,v), I[126] = (img)(_n3##x,_n3##y,z,v), I[127] = (img)(_n4##x,_n3##y,z,v), I[128] = (img)(_n5##x,_n3##y,z,v), I[129] = (img)(_n6##x,_n3##y,z,v), \
 I[130] = (img)(_p6##x,_n4##y,z,v), I[131] = (img)(_p5##x,_n4##y,z,v), I[132] = (img)(_p4##x,_n4##y,z,v), I[133] = (img)(_p3##x,_n4##y,z,v), I[134] = (img)(_p2##x,_n4##y,z,v), I[135] = (img)(_p1##x,_n4##y,z,v), I[136] = (img)(x,_n4##y,z,v), I[137] = (img)(_n1##x,_n4##y,z,v), I[138] = (img)(_n2##x,_n4##y,z,v), I[139] = (img)(_n3##x,_n4##y,z,v), I[140] = (img)(_n4##x,_n4##y,z,v), I[141] = (img)(_n5##x,_n4##y,z,v), I[142] = (img)(_n6##x,_n4##y,z,v), \
 I[143] = (img)(_p6##x,_n5##y,z,v), I[144] = (img)(_p5##x,_n5##y,z,v), I[145] = (img)(_p4##x,_n5##y,z,v), I[146] = (img)(_p3##x,_n5##y,z,v), I[147] = (img)(_p2##x,_n5##y,z,v), I[148] = (img)(_p1##x,_n5##y,z,v), I[149] = (img)(x,_n5##y,z,v), I[150] = (img)(_n1##x,_n5##y,z,v), I[151] = (img)(_n2##x,_n5##y,z,v), I[152] = (img)(_n3##x,_n5##y,z,v), I[153] = (img)(_n4##x,_n5##y,z,v), I[154] = (img)(_n5##x,_n5##y,z,v), I[155] = (img)(_n6##x,_n5##y,z,v), \
 I[156] = (img)(_p6##x,_n6##y,z,v), I[157] = (img)(_p5##x,_n6##y,z,v), I[158] = (img)(_p4##x,_n6##y,z,v), I[159] = (img)(_p3##x,_n6##y,z,v), I[160] = (img)(_p2##x,_n6##y,z,v), I[161] = (img)(_p1##x,_n6##y,z,v), I[162] = (img)(x,_n6##y,z,v), I[163] = (img)(_n1##x,_n6##y,z,v), I[164] = (img)(_n2##x,_n6##y,z,v), I[165] = (img)(_n3##x,_n6##y,z,v), I[166] = (img)(_n4##x,_n6##y,z,v), I[167] = (img)(_n5##x,_n6##y,z,v), I[168] = (img)(_n6##x,_n6##y,z,v);

// Define 14x14 loop macros for CImg
//----------------------------------
#define cimg_for14(bound,i) for (int i = 0, \
 _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7; \
 _n7##i<(int)(bound) || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i)

#define cimg_for14X(img,x) cimg_for14((img).width,x)
#define cimg_for14Y(img,y) cimg_for14((img).height,y)
#define cimg_for14Z(img,z) cimg_for14((img).depth,z)
#define cimg_for14V(img,v) cimg_for14((img).dim,v)
#define cimg_for14XY(img,x,y) cimg_for14Y(img,y) cimg_for14X(img,x)
#define cimg_for14XZ(img,x,z) cimg_for14Z(img,z) cimg_for14X(img,x)
#define cimg_for14XV(img,x,v) cimg_for14V(img,v) cimg_for14X(img,x)
#define cimg_for14YZ(img,y,z) cimg_for14Z(img,z) cimg_for14Y(img,y)
#define cimg_for14YV(img,y,v) cimg_for14V(img,v) cimg_for14Y(img,y)
#define cimg_for14ZV(img,z,v) cimg_for14V(img,v) cimg_for14Z(img,z)
#define cimg_for14XYZ(img,x,y,z) cimg_for14Z(img,z) cimg_for14XY(img,x,y)
#define cimg_for14XZV(img,x,z,v) cimg_for14V(img,v) cimg_for14XZ(img,x,z)
#define cimg_for14YZV(img,y,z,v) cimg_for14V(img,v) cimg_for14YZ(img,y,z)
#define cimg_for14XYZV(img,x,y,z,v) cimg_for14V(img,v) cimg_for14XYZ(img,x,y,z)

#define cimg_for_in14(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7; \
 i<=(int)(i1) && (_n7##i<(int)(bound) || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i)

#define cimg_for_in14X(img,x0,x1,x) cimg_for_in14((img).width,x0,x1,x)
#define cimg_for_in14Y(img,y0,y1,y) cimg_for_in14((img).height,y0,y1,y)
#define cimg_for_in14Z(img,z0,z1,z) cimg_for_in14((img).depth,z0,z1,z)
#define cimg_for_in14V(img,v0,v1,v) cimg_for_in14((img).dim,v0,v1,v)
#define cimg_for_in14XY(img,x0,y0,x1,y1,x,y) cimg_for_in14Y(img,y0,y1,y) cimg_for_in14X(img,x0,x1,x)
#define cimg_for_in14XZ(img,x0,z0,x1,z1,x,z) cimg_for_in14Z(img,z0,z1,z) cimg_for_in14X(img,x0,x1,x)
#define cimg_for_in14XV(img,x0,v0,x1,v1,x,v) cimg_for_in14V(img,v0,v1,v) cimg_for_in14X(img,x0,x1,x)
#define cimg_for_in14YZ(img,y0,z0,y1,z1,y,z) cimg_for_in14Z(img,z0,z1,z) cimg_for_in14Y(img,y0,y1,y)
#define cimg_for_in14YV(img,y0,v0,y1,v1,y,v) cimg_for_in14V(img,v0,v1,v) cimg_for_in14Y(img,y0,y1,y)
#define cimg_for_in14ZV(img,z0,v0,z1,v1,z,v) cimg_for_in14V(img,v0,v1,v) cimg_for_in14Z(img,z0,z1,z)
#define cimg_for_in14XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in14Z(img,z0,z1,z) cimg_for_in14XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in14XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in14V(img,v0,v1,v) cimg_for_in14XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in14YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in14V(img,v0,v1,v) cimg_for_in14YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in14XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in14V(img,v0,v1,v) cimg_for_in14XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for14x14(img,x,y,z,v,I) \
 cimg_for14((img).height,y) for (int x = 0, \
 _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = (img)(0,_p6##y,z,v)), \
 (I[14] = I[15] = I[16] = I[17] = I[18] = I[19] = I[20] = (img)(0,_p5##y,z,v)), \
 (I[28] = I[29] = I[30] = I[31] = I[32] = I[33] = I[34] = (img)(0,_p4##y,z,v)), \
 (I[42] = I[43] = I[44] = I[45] = I[46] = I[47] = I[48] = (img)(0,_p3##y,z,v)), \
 (I[56] = I[57] = I[58] = I[59] = I[60] = I[61] = I[62] = (img)(0,_p2##y,z,v)), \
 (I[70] = I[71] = I[72] = I[73] = I[74] = I[75] = I[76] = (img)(0,_p1##y,z,v)), \
 (I[84] = I[85] = I[86] = I[87] = I[88] = I[89] = I[90] = (img)(0,y,z,v)), \
 (I[98] = I[99] = I[100] = I[101] = I[102] = I[103] = I[104] = (img)(0,_n1##y,z,v)), \
 (I[112] = I[113] = I[114] = I[115] = I[116] = I[117] = I[118] = (img)(0,_n2##y,z,v)), \
 (I[126] = I[127] = I[128] = I[129] = I[130] = I[131] = I[132] = (img)(0,_n3##y,z,v)), \
 (I[140] = I[141] = I[142] = I[143] = I[144] = I[145] = I[146] = (img)(0,_n4##y,z,v)), \
 (I[154] = I[155] = I[156] = I[157] = I[158] = I[159] = I[160] = (img)(0,_n5##y,z,v)), \
 (I[168] = I[169] = I[170] = I[171] = I[172] = I[173] = I[174] = (img)(0,_n6##y,z,v)), \
 (I[182] = I[183] = I[184] = I[185] = I[186] = I[187] = I[188] = (img)(0,_n7##y,z,v)), \
 (I[7] = (img)(_n1##x,_p6##y,z,v)), \
 (I[21] = (img)(_n1##x,_p5##y,z,v)), \
 (I[35] = (img)(_n1##x,_p4##y,z,v)), \
 (I[49] = (img)(_n1##x,_p3##y,z,v)), \
 (I[63] = (img)(_n1##x,_p2##y,z,v)), \
 (I[77] = (img)(_n1##x,_p1##y,z,v)), \
 (I[91] = (img)(_n1##x,y,z,v)), \
 (I[105] = (img)(_n1##x,_n1##y,z,v)), \
 (I[119] = (img)(_n1##x,_n2##y,z,v)), \
 (I[133] = (img)(_n1##x,_n3##y,z,v)), \
 (I[147] = (img)(_n1##x,_n4##y,z,v)), \
 (I[161] = (img)(_n1##x,_n5##y,z,v)), \
 (I[175] = (img)(_n1##x,_n6##y,z,v)), \
 (I[189] = (img)(_n1##x,_n7##y,z,v)), \
 (I[8] = (img)(_n2##x,_p6##y,z,v)), \
 (I[22] = (img)(_n2##x,_p5##y,z,v)), \
 (I[36] = (img)(_n2##x,_p4##y,z,v)), \
 (I[50] = (img)(_n2##x,_p3##y,z,v)), \
 (I[64] = (img)(_n2##x,_p2##y,z,v)), \
 (I[78] = (img)(_n2##x,_p1##y,z,v)), \
 (I[92] = (img)(_n2##x,y,z,v)), \
 (I[106] = (img)(_n2##x,_n1##y,z,v)), \
 (I[120] = (img)(_n2##x,_n2##y,z,v)), \
 (I[134] = (img)(_n2##x,_n3##y,z,v)), \
 (I[148] = (img)(_n2##x,_n4##y,z,v)), \
 (I[162] = (img)(_n2##x,_n5##y,z,v)), \
 (I[176] = (img)(_n2##x,_n6##y,z,v)), \
 (I[190] = (img)(_n2##x,_n7##y,z,v)), \
 (I[9] = (img)(_n3##x,_p6##y,z,v)), \
 (I[23] = (img)(_n3##x,_p5##y,z,v)), \
 (I[37] = (img)(_n3##x,_p4##y,z,v)), \
 (I[51] = (img)(_n3##x,_p3##y,z,v)), \
 (I[65] = (img)(_n3##x,_p2##y,z,v)), \
 (I[79] = (img)(_n3##x,_p1##y,z,v)), \
 (I[93] = (img)(_n3##x,y,z,v)), \
 (I[107] = (img)(_n3##x,_n1##y,z,v)), \
 (I[121] = (img)(_n3##x,_n2##y,z,v)), \
 (I[135] = (img)(_n3##x,_n3##y,z,v)), \
 (I[149] = (img)(_n3##x,_n4##y,z,v)), \
 (I[163] = (img)(_n3##x,_n5##y,z,v)), \
 (I[177] = (img)(_n3##x,_n6##y,z,v)), \
 (I[191] = (img)(_n3##x,_n7##y,z,v)), \
 (I[10] = (img)(_n4##x,_p6##y,z,v)), \
 (I[24] = (img)(_n4##x,_p5##y,z,v)), \
 (I[38] = (img)(_n4##x,_p4##y,z,v)), \
 (I[52] = (img)(_n4##x,_p3##y,z,v)), \
 (I[66] = (img)(_n4##x,_p2##y,z,v)), \
 (I[80] = (img)(_n4##x,_p1##y,z,v)), \
 (I[94] = (img)(_n4##x,y,z,v)), \
 (I[108] = (img)(_n4##x,_n1##y,z,v)), \
 (I[122] = (img)(_n4##x,_n2##y,z,v)), \
 (I[136] = (img)(_n4##x,_n3##y,z,v)), \
 (I[150] = (img)(_n4##x,_n4##y,z,v)), \
 (I[164] = (img)(_n4##x,_n5##y,z,v)), \
 (I[178] = (img)(_n4##x,_n6##y,z,v)), \
 (I[192] = (img)(_n4##x,_n7##y,z,v)), \
 (I[11] = (img)(_n5##x,_p6##y,z,v)), \
 (I[25] = (img)(_n5##x,_p5##y,z,v)), \
 (I[39] = (img)(_n5##x,_p4##y,z,v)), \
 (I[53] = (img)(_n5##x,_p3##y,z,v)), \
 (I[67] = (img)(_n5##x,_p2##y,z,v)), \
 (I[81] = (img)(_n5##x,_p1##y,z,v)), \
 (I[95] = (img)(_n5##x,y,z,v)), \
 (I[109] = (img)(_n5##x,_n1##y,z,v)), \
 (I[123] = (img)(_n5##x,_n2##y,z,v)), \
 (I[137] = (img)(_n5##x,_n3##y,z,v)), \
 (I[151] = (img)(_n5##x,_n4##y,z,v)), \
 (I[165] = (img)(_n5##x,_n5##y,z,v)), \
 (I[179] = (img)(_n5##x,_n6##y,z,v)), \
 (I[193] = (img)(_n5##x,_n7##y,z,v)), \
 (I[12] = (img)(_n6##x,_p6##y,z,v)), \
 (I[26] = (img)(_n6##x,_p5##y,z,v)), \
 (I[40] = (img)(_n6##x,_p4##y,z,v)), \
 (I[54] = (img)(_n6##x,_p3##y,z,v)), \
 (I[68] = (img)(_n6##x,_p2##y,z,v)), \
 (I[82] = (img)(_n6##x,_p1##y,z,v)), \
 (I[96] = (img)(_n6##x,y,z,v)), \
 (I[110] = (img)(_n6##x,_n1##y,z,v)), \
 (I[124] = (img)(_n6##x,_n2##y,z,v)), \
 (I[138] = (img)(_n6##x,_n3##y,z,v)), \
 (I[152] = (img)(_n6##x,_n4##y,z,v)), \
 (I[166] = (img)(_n6##x,_n5##y,z,v)), \
 (I[180] = (img)(_n6##x,_n6##y,z,v)), \
 (I[194] = (img)(_n6##x,_n7##y,z,v)), \
 7>=((img).width)?(int)((img).width)-1:7); \
 (_n7##x<(int)((img).width) && ( \
 (I[13] = (img)(_n7##x,_p6##y,z,v)), \
 (I[27] = (img)(_n7##x,_p5##y,z,v)), \
 (I[41] = (img)(_n7##x,_p4##y,z,v)), \
 (I[55] = (img)(_n7##x,_p3##y,z,v)), \
 (I[69] = (img)(_n7##x,_p2##y,z,v)), \
 (I[83] = (img)(_n7##x,_p1##y,z,v)), \
 (I[97] = (img)(_n7##x,y,z,v)), \
 (I[111] = (img)(_n7##x,_n1##y,z,v)), \
 (I[125] = (img)(_n7##x,_n2##y,z,v)), \
 (I[139] = (img)(_n7##x,_n3##y,z,v)), \
 (I[153] = (img)(_n7##x,_n4##y,z,v)), \
 (I[167] = (img)(_n7##x,_n5##y,z,v)), \
 (I[181] = (img)(_n7##x,_n6##y,z,v)), \
 (I[195] = (img)(_n7##x,_n7##y,z,v)),1)) || \
 _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], \
 I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], \
 I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], \
 I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], \
 I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], \
 I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], \
 I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], \
 I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], \
 I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], \
 I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], \
 I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], \
 I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], \
 I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], \
 _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x)

#define cimg_for_in14x14(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in14((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = (int)( \
 (I[0] = (img)(_p6##x,_p6##y,z,v)), \
 (I[14] = (img)(_p6##x,_p5##y,z,v)), \
 (I[28] = (img)(_p6##x,_p4##y,z,v)), \
 (I[42] = (img)(_p6##x,_p3##y,z,v)), \
 (I[56] = (img)(_p6##x,_p2##y,z,v)), \
 (I[70] = (img)(_p6##x,_p1##y,z,v)), \
 (I[84] = (img)(_p6##x,y,z,v)), \
 (I[98] = (img)(_p6##x,_n1##y,z,v)), \
 (I[112] = (img)(_p6##x,_n2##y,z,v)), \
 (I[126] = (img)(_p6##x,_n3##y,z,v)), \
 (I[140] = (img)(_p6##x,_n4##y,z,v)), \
 (I[154] = (img)(_p6##x,_n5##y,z,v)), \
 (I[168] = (img)(_p6##x,_n6##y,z,v)), \
 (I[182] = (img)(_p6##x,_n7##y,z,v)), \
 (I[1] = (img)(_p5##x,_p6##y,z,v)), \
 (I[15] = (img)(_p5##x,_p5##y,z,v)), \
 (I[29] = (img)(_p5##x,_p4##y,z,v)), \
 (I[43] = (img)(_p5##x,_p3##y,z,v)), \
 (I[57] = (img)(_p5##x,_p2##y,z,v)), \
 (I[71] = (img)(_p5##x,_p1##y,z,v)), \
 (I[85] = (img)(_p5##x,y,z,v)), \
 (I[99] = (img)(_p5##x,_n1##y,z,v)), \
 (I[113] = (img)(_p5##x,_n2##y,z,v)), \
 (I[127] = (img)(_p5##x,_n3##y,z,v)), \
 (I[141] = (img)(_p5##x,_n4##y,z,v)), \
 (I[155] = (img)(_p5##x,_n5##y,z,v)), \
 (I[169] = (img)(_p5##x,_n6##y,z,v)), \
 (I[183] = (img)(_p5##x,_n7##y,z,v)), \
 (I[2] = (img)(_p4##x,_p6##y,z,v)), \
 (I[16] = (img)(_p4##x,_p5##y,z,v)), \
 (I[30] = (img)(_p4##x,_p4##y,z,v)), \
 (I[44] = (img)(_p4##x,_p3##y,z,v)), \
 (I[58] = (img)(_p4##x,_p2##y,z,v)), \
 (I[72] = (img)(_p4##x,_p1##y,z,v)), \
 (I[86] = (img)(_p4##x,y,z,v)), \
 (I[100] = (img)(_p4##x,_n1##y,z,v)), \
 (I[114] = (img)(_p4##x,_n2##y,z,v)), \
 (I[128] = (img)(_p4##x,_n3##y,z,v)), \
 (I[142] = (img)(_p4##x,_n4##y,z,v)), \
 (I[156] = (img)(_p4##x,_n5##y,z,v)), \
 (I[170] = (img)(_p4##x,_n6##y,z,v)), \
 (I[184] = (img)(_p4##x,_n7##y,z,v)), \
 (I[3] = (img)(_p3##x,_p6##y,z,v)), \
 (I[17] = (img)(_p3##x,_p5##y,z,v)), \
 (I[31] = (img)(_p3##x,_p4##y,z,v)), \
 (I[45] = (img)(_p3##x,_p3##y,z,v)), \
 (I[59] = (img)(_p3##x,_p2##y,z,v)), \
 (I[73] = (img)(_p3##x,_p1##y,z,v)), \
 (I[87] = (img)(_p3##x,y,z,v)), \
 (I[101] = (img)(_p3##x,_n1##y,z,v)), \
 (I[115] = (img)(_p3##x,_n2##y,z,v)), \
 (I[129] = (img)(_p3##x,_n3##y,z,v)), \
 (I[143] = (img)(_p3##x,_n4##y,z,v)), \
 (I[157] = (img)(_p3##x,_n5##y,z,v)), \
 (I[171] = (img)(_p3##x,_n6##y,z,v)), \
 (I[185] = (img)(_p3##x,_n7##y,z,v)), \
 (I[4] = (img)(_p2##x,_p6##y,z,v)), \
 (I[18] = (img)(_p2##x,_p5##y,z,v)), \
 (I[32] = (img)(_p2##x,_p4##y,z,v)), \
 (I[46] = (img)(_p2##x,_p3##y,z,v)), \
 (I[60] = (img)(_p2##x,_p2##y,z,v)), \
 (I[74] = (img)(_p2##x,_p1##y,z,v)), \
 (I[88] = (img)(_p2##x,y,z,v)), \
 (I[102] = (img)(_p2##x,_n1##y,z,v)), \
 (I[116] = (img)(_p2##x,_n2##y,z,v)), \
 (I[130] = (img)(_p2##x,_n3##y,z,v)), \
 (I[144] = (img)(_p2##x,_n4##y,z,v)), \
 (I[158] = (img)(_p2##x,_n5##y,z,v)), \
 (I[172] = (img)(_p2##x,_n6##y,z,v)), \
 (I[186] = (img)(_p2##x,_n7##y,z,v)), \
 (I[5] = (img)(_p1##x,_p6##y,z,v)), \
 (I[19] = (img)(_p1##x,_p5##y,z,v)), \
 (I[33] = (img)(_p1##x,_p4##y,z,v)), \
 (I[47] = (img)(_p1##x,_p3##y,z,v)), \
 (I[61] = (img)(_p1##x,_p2##y,z,v)), \
 (I[75] = (img)(_p1##x,_p1##y,z,v)), \
 (I[89] = (img)(_p1##x,y,z,v)), \
 (I[103] = (img)(_p1##x,_n1##y,z,v)), \
 (I[117] = (img)(_p1##x,_n2##y,z,v)), \
 (I[131] = (img)(_p1##x,_n3##y,z,v)), \
 (I[145] = (img)(_p1##x,_n4##y,z,v)), \
 (I[159] = (img)(_p1##x,_n5##y,z,v)), \
 (I[173] = (img)(_p1##x,_n6##y,z,v)), \
 (I[187] = (img)(_p1##x,_n7##y,z,v)), \
 (I[6] = (img)(x,_p6##y,z,v)), \
 (I[20] = (img)(x,_p5##y,z,v)), \
 (I[34] = (img)(x,_p4##y,z,v)), \
 (I[48] = (img)(x,_p3##y,z,v)), \
 (I[62] = (img)(x,_p2##y,z,v)), \
 (I[76] = (img)(x,_p1##y,z,v)), \
 (I[90] = (img)(x,y,z,v)), \
 (I[104] = (img)(x,_n1##y,z,v)), \
 (I[118] = (img)(x,_n2##y,z,v)), \
 (I[132] = (img)(x,_n3##y,z,v)), \
 (I[146] = (img)(x,_n4##y,z,v)), \
 (I[160] = (img)(x,_n5##y,z,v)), \
 (I[174] = (img)(x,_n6##y,z,v)), \
 (I[188] = (img)(x,_n7##y,z,v)), \
 (I[7] = (img)(_n1##x,_p6##y,z,v)), \
 (I[21] = (img)(_n1##x,_p5##y,z,v)), \
 (I[35] = (img)(_n1##x,_p4##y,z,v)), \
 (I[49] = (img)(_n1##x,_p3##y,z,v)), \
 (I[63] = (img)(_n1##x,_p2##y,z,v)), \
 (I[77] = (img)(_n1##x,_p1##y,z,v)), \
 (I[91] = (img)(_n1##x,y,z,v)), \
 (I[105] = (img)(_n1##x,_n1##y,z,v)), \
 (I[119] = (img)(_n1##x,_n2##y,z,v)), \
 (I[133] = (img)(_n1##x,_n3##y,z,v)), \
 (I[147] = (img)(_n1##x,_n4##y,z,v)), \
 (I[161] = (img)(_n1##x,_n5##y,z,v)), \
 (I[175] = (img)(_n1##x,_n6##y,z,v)), \
 (I[189] = (img)(_n1##x,_n7##y,z,v)), \
 (I[8] = (img)(_n2##x,_p6##y,z,v)), \
 (I[22] = (img)(_n2##x,_p5##y,z,v)), \
 (I[36] = (img)(_n2##x,_p4##y,z,v)), \
 (I[50] = (img)(_n2##x,_p3##y,z,v)), \
 (I[64] = (img)(_n2##x,_p2##y,z,v)), \
 (I[78] = (img)(_n2##x,_p1##y,z,v)), \
 (I[92] = (img)(_n2##x,y,z,v)), \
 (I[106] = (img)(_n2##x,_n1##y,z,v)), \
 (I[120] = (img)(_n2##x,_n2##y,z,v)), \
 (I[134] = (img)(_n2##x,_n3##y,z,v)), \
 (I[148] = (img)(_n2##x,_n4##y,z,v)), \
 (I[162] = (img)(_n2##x,_n5##y,z,v)), \
 (I[176] = (img)(_n2##x,_n6##y,z,v)), \
 (I[190] = (img)(_n2##x,_n7##y,z,v)), \
 (I[9] = (img)(_n3##x,_p6##y,z,v)), \
 (I[23] = (img)(_n3##x,_p5##y,z,v)), \
 (I[37] = (img)(_n3##x,_p4##y,z,v)), \
 (I[51] = (img)(_n3##x,_p3##y,z,v)), \
 (I[65] = (img)(_n3##x,_p2##y,z,v)), \
 (I[79] = (img)(_n3##x,_p1##y,z,v)), \
 (I[93] = (img)(_n3##x,y,z,v)), \
 (I[107] = (img)(_n3##x,_n1##y,z,v)), \
 (I[121] = (img)(_n3##x,_n2##y,z,v)), \
 (I[135] = (img)(_n3##x,_n3##y,z,v)), \
 (I[149] = (img)(_n3##x,_n4##y,z,v)), \
 (I[163] = (img)(_n3##x,_n5##y,z,v)), \
 (I[177] = (img)(_n3##x,_n6##y,z,v)), \
 (I[191] = (img)(_n3##x,_n7##y,z,v)), \
 (I[10] = (img)(_n4##x,_p6##y,z,v)), \
 (I[24] = (img)(_n4##x,_p5##y,z,v)), \
 (I[38] = (img)(_n4##x,_p4##y,z,v)), \
 (I[52] = (img)(_n4##x,_p3##y,z,v)), \
 (I[66] = (img)(_n4##x,_p2##y,z,v)), \
 (I[80] = (img)(_n4##x,_p1##y,z,v)), \
 (I[94] = (img)(_n4##x,y,z,v)), \
 (I[108] = (img)(_n4##x,_n1##y,z,v)), \
 (I[122] = (img)(_n4##x,_n2##y,z,v)), \
 (I[136] = (img)(_n4##x,_n3##y,z,v)), \
 (I[150] = (img)(_n4##x,_n4##y,z,v)), \
 (I[164] = (img)(_n4##x,_n5##y,z,v)), \
 (I[178] = (img)(_n4##x,_n6##y,z,v)), \
 (I[192] = (img)(_n4##x,_n7##y,z,v)), \
 (I[11] = (img)(_n5##x,_p6##y,z,v)), \
 (I[25] = (img)(_n5##x,_p5##y,z,v)), \
 (I[39] = (img)(_n5##x,_p4##y,z,v)), \
 (I[53] = (img)(_n5##x,_p3##y,z,v)), \
 (I[67] = (img)(_n5##x,_p2##y,z,v)), \
 (I[81] = (img)(_n5##x,_p1##y,z,v)), \
 (I[95] = (img)(_n5##x,y,z,v)), \
 (I[109] = (img)(_n5##x,_n1##y,z,v)), \
 (I[123] = (img)(_n5##x,_n2##y,z,v)), \
 (I[137] = (img)(_n5##x,_n3##y,z,v)), \
 (I[151] = (img)(_n5##x,_n4##y,z,v)), \
 (I[165] = (img)(_n5##x,_n5##y,z,v)), \
 (I[179] = (img)(_n5##x,_n6##y,z,v)), \
 (I[193] = (img)(_n5##x,_n7##y,z,v)), \
 (I[12] = (img)(_n6##x,_p6##y,z,v)), \
 (I[26] = (img)(_n6##x,_p5##y,z,v)), \
 (I[40] = (img)(_n6##x,_p4##y,z,v)), \
 (I[54] = (img)(_n6##x,_p3##y,z,v)), \
 (I[68] = (img)(_n6##x,_p2##y,z,v)), \
 (I[82] = (img)(_n6##x,_p1##y,z,v)), \
 (I[96] = (img)(_n6##x,y,z,v)), \
 (I[110] = (img)(_n6##x,_n1##y,z,v)), \
 (I[124] = (img)(_n6##x,_n2##y,z,v)), \
 (I[138] = (img)(_n6##x,_n3##y,z,v)), \
 (I[152] = (img)(_n6##x,_n4##y,z,v)), \
 (I[166] = (img)(_n6##x,_n5##y,z,v)), \
 (I[180] = (img)(_n6##x,_n6##y,z,v)), \
 (I[194] = (img)(_n6##x,_n7##y,z,v)), \
 x+7>=(int)((img).width)?(int)((img).width)-1:x+7); \
 x<=(int)(x1) && ((_n7##x<(int)((img).width) && ( \
 (I[13] = (img)(_n7##x,_p6##y,z,v)), \
 (I[27] = (img)(_n7##x,_p5##y,z,v)), \
 (I[41] = (img)(_n7##x,_p4##y,z,v)), \
 (I[55] = (img)(_n7##x,_p3##y,z,v)), \
 (I[69] = (img)(_n7##x,_p2##y,z,v)), \
 (I[83] = (img)(_n7##x,_p1##y,z,v)), \
 (I[97] = (img)(_n7##x,y,z,v)), \
 (I[111] = (img)(_n7##x,_n1##y,z,v)), \
 (I[125] = (img)(_n7##x,_n2##y,z,v)), \
 (I[139] = (img)(_n7##x,_n3##y,z,v)), \
 (I[153] = (img)(_n7##x,_n4##y,z,v)), \
 (I[167] = (img)(_n7##x,_n5##y,z,v)), \
 (I[181] = (img)(_n7##x,_n6##y,z,v)), \
 (I[195] = (img)(_n7##x,_n7##y,z,v)),1)) || \
 _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], \
 I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], \
 I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], \
 I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], \
 I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], \
 I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], \
 I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], \
 I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], \
 I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], \
 I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], \
 I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], \
 I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], \
 I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], \
 _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x)

#define cimg_get14x14(img,x,y,z,v,I) \
 I[0] = (img)(_p6##x,_p6##y,z,v), I[1] = (img)(_p5##x,_p6##y,z,v), I[2] = (img)(_p4##x,_p6##y,z,v), I[3] = (img)(_p3##x,_p6##y,z,v), I[4] = (img)(_p2##x,_p6##y,z,v), I[5] = (img)(_p1##x,_p6##y,z,v), I[6] = (img)(x,_p6##y,z,v), I[7] = (img)(_n1##x,_p6##y,z,v), I[8] = (img)(_n2##x,_p6##y,z,v), I[9] = (img)(_n3##x,_p6##y,z,v), I[10] = (img)(_n4##x,_p6##y,z,v), I[11] = (img)(_n5##x,_p6##y,z,v), I[12] = (img)(_n6##x,_p6##y,z,v), I[13] = (img)(_n7##x,_p6##y,z,v), \
 I[14] = (img)(_p6##x,_p5##y,z,v), I[15] = (img)(_p5##x,_p5##y,z,v), I[16] = (img)(_p4##x,_p5##y,z,v), I[17] = (img)(_p3##x,_p5##y,z,v), I[18] = (img)(_p2##x,_p5##y,z,v), I[19] = (img)(_p1##x,_p5##y,z,v), I[20] = (img)(x,_p5##y,z,v), I[21] = (img)(_n1##x,_p5##y,z,v), I[22] = (img)(_n2##x,_p5##y,z,v), I[23] = (img)(_n3##x,_p5##y,z,v), I[24] = (img)(_n4##x,_p5##y,z,v), I[25] = (img)(_n5##x,_p5##y,z,v), I[26] = (img)(_n6##x,_p5##y,z,v), I[27] = (img)(_n7##x,_p5##y,z,v), \
 I[28] = (img)(_p6##x,_p4##y,z,v), I[29] = (img)(_p5##x,_p4##y,z,v), I[30] = (img)(_p4##x,_p4##y,z,v), I[31] = (img)(_p3##x,_p4##y,z,v), I[32] = (img)(_p2##x,_p4##y,z,v), I[33] = (img)(_p1##x,_p4##y,z,v), I[34] = (img)(x,_p4##y,z,v), I[35] = (img)(_n1##x,_p4##y,z,v), I[36] = (img)(_n2##x,_p4##y,z,v), I[37] = (img)(_n3##x,_p4##y,z,v), I[38] = (img)(_n4##x,_p4##y,z,v), I[39] = (img)(_n5##x,_p4##y,z,v), I[40] = (img)(_n6##x,_p4##y,z,v), I[41] = (img)(_n7##x,_p4##y,z,v), \
 I[42] = (img)(_p6##x,_p3##y,z,v), I[43] = (img)(_p5##x,_p3##y,z,v), I[44] = (img)(_p4##x,_p3##y,z,v), I[45] = (img)(_p3##x,_p3##y,z,v), I[46] = (img)(_p2##x,_p3##y,z,v), I[47] = (img)(_p1##x,_p3##y,z,v), I[48] = (img)(x,_p3##y,z,v), I[49] = (img)(_n1##x,_p3##y,z,v), I[50] = (img)(_n2##x,_p3##y,z,v), I[51] = (img)(_n3##x,_p3##y,z,v), I[52] = (img)(_n4##x,_p3##y,z,v), I[53] = (img)(_n5##x,_p3##y,z,v), I[54] = (img)(_n6##x,_p3##y,z,v), I[55] = (img)(_n7##x,_p3##y,z,v), \
 I[56] = (img)(_p6##x,_p2##y,z,v), I[57] = (img)(_p5##x,_p2##y,z,v), I[58] = (img)(_p4##x,_p2##y,z,v), I[59] = (img)(_p3##x,_p2##y,z,v), I[60] = (img)(_p2##x,_p2##y,z,v), I[61] = (img)(_p1##x,_p2##y,z,v), I[62] = (img)(x,_p2##y,z,v), I[63] = (img)(_n1##x,_p2##y,z,v), I[64] = (img)(_n2##x,_p2##y,z,v), I[65] = (img)(_n3##x,_p2##y,z,v), I[66] = (img)(_n4##x,_p2##y,z,v), I[67] = (img)(_n5##x,_p2##y,z,v), I[68] = (img)(_n6##x,_p2##y,z,v), I[69] = (img)(_n7##x,_p2##y,z,v), \
 I[70] = (img)(_p6##x,_p1##y,z,v), I[71] = (img)(_p5##x,_p1##y,z,v), I[72] = (img)(_p4##x,_p1##y,z,v), I[73] = (img)(_p3##x,_p1##y,z,v), I[74] = (img)(_p2##x,_p1##y,z,v), I[75] = (img)(_p1##x,_p1##y,z,v), I[76] = (img)(x,_p1##y,z,v), I[77] = (img)(_n1##x,_p1##y,z,v), I[78] = (img)(_n2##x,_p1##y,z,v), I[79] = (img)(_n3##x,_p1##y,z,v), I[80] = (img)(_n4##x,_p1##y,z,v), I[81] = (img)(_n5##x,_p1##y,z,v), I[82] = (img)(_n6##x,_p1##y,z,v), I[83] = (img)(_n7##x,_p1##y,z,v), \
 I[84] = (img)(_p6##x,y,z,v), I[85] = (img)(_p5##x,y,z,v), I[86] = (img)(_p4##x,y,z,v), I[87] = (img)(_p3##x,y,z,v), I[88] = (img)(_p2##x,y,z,v), I[89] = (img)(_p1##x,y,z,v), I[90] = (img)(x,y,z,v), I[91] = (img)(_n1##x,y,z,v), I[92] = (img)(_n2##x,y,z,v), I[93] = (img)(_n3##x,y,z,v), I[94] = (img)(_n4##x,y,z,v), I[95] = (img)(_n5##x,y,z,v), I[96] = (img)(_n6##x,y,z,v), I[97] = (img)(_n7##x,y,z,v), \
 I[98] = (img)(_p6##x,_n1##y,z,v), I[99] = (img)(_p5##x,_n1##y,z,v), I[100] = (img)(_p4##x,_n1##y,z,v), I[101] = (img)(_p3##x,_n1##y,z,v), I[102] = (img)(_p2##x,_n1##y,z,v), I[103] = (img)(_p1##x,_n1##y,z,v), I[104] = (img)(x,_n1##y,z,v), I[105] = (img)(_n1##x,_n1##y,z,v), I[106] = (img)(_n2##x,_n1##y,z,v), I[107] = (img)(_n3##x,_n1##y,z,v), I[108] = (img)(_n4##x,_n1##y,z,v), I[109] = (img)(_n5##x,_n1##y,z,v), I[110] = (img)(_n6##x,_n1##y,z,v), I[111] = (img)(_n7##x,_n1##y,z,v), \
 I[112] = (img)(_p6##x,_n2##y,z,v), I[113] = (img)(_p5##x,_n2##y,z,v), I[114] = (img)(_p4##x,_n2##y,z,v), I[115] = (img)(_p3##x,_n2##y,z,v), I[116] = (img)(_p2##x,_n2##y,z,v), I[117] = (img)(_p1##x,_n2##y,z,v), I[118] = (img)(x,_n2##y,z,v), I[119] = (img)(_n1##x,_n2##y,z,v), I[120] = (img)(_n2##x,_n2##y,z,v), I[121] = (img)(_n3##x,_n2##y,z,v), I[122] = (img)(_n4##x,_n2##y,z,v), I[123] = (img)(_n5##x,_n2##y,z,v), I[124] = (img)(_n6##x,_n2##y,z,v), I[125] = (img)(_n7##x,_n2##y,z,v), \
 I[126] = (img)(_p6##x,_n3##y,z,v), I[127] = (img)(_p5##x,_n3##y,z,v), I[128] = (img)(_p4##x,_n3##y,z,v), I[129] = (img)(_p3##x,_n3##y,z,v), I[130] = (img)(_p2##x,_n3##y,z,v), I[131] = (img)(_p1##x,_n3##y,z,v), I[132] = (img)(x,_n3##y,z,v), I[133] = (img)(_n1##x,_n3##y,z,v), I[134] = (img)(_n2##x,_n3##y,z,v), I[135] = (img)(_n3##x,_n3##y,z,v), I[136] = (img)(_n4##x,_n3##y,z,v), I[137] = (img)(_n5##x,_n3##y,z,v), I[138] = (img)(_n6##x,_n3##y,z,v), I[139] = (img)(_n7##x,_n3##y,z,v), \
 I[140] = (img)(_p6##x,_n4##y,z,v), I[141] = (img)(_p5##x,_n4##y,z,v), I[142] = (img)(_p4##x,_n4##y,z,v), I[143] = (img)(_p3##x,_n4##y,z,v), I[144] = (img)(_p2##x,_n4##y,z,v), I[145] = (img)(_p1##x,_n4##y,z,v), I[146] = (img)(x,_n4##y,z,v), I[147] = (img)(_n1##x,_n4##y,z,v), I[148] = (img)(_n2##x,_n4##y,z,v), I[149] = (img)(_n3##x,_n4##y,z,v), I[150] = (img)(_n4##x,_n4##y,z,v), I[151] = (img)(_n5##x,_n4##y,z,v), I[152] = (img)(_n6##x,_n4##y,z,v), I[153] = (img)(_n7##x,_n4##y,z,v), \
 I[154] = (img)(_p6##x,_n5##y,z,v), I[155] = (img)(_p5##x,_n5##y,z,v), I[156] = (img)(_p4##x,_n5##y,z,v), I[157] = (img)(_p3##x,_n5##y,z,v), I[158] = (img)(_p2##x,_n5##y,z,v), I[159] = (img)(_p1##x,_n5##y,z,v), I[160] = (img)(x,_n5##y,z,v), I[161] = (img)(_n1##x,_n5##y,z,v), I[162] = (img)(_n2##x,_n5##y,z,v), I[163] = (img)(_n3##x,_n5##y,z,v), I[164] = (img)(_n4##x,_n5##y,z,v), I[165] = (img)(_n5##x,_n5##y,z,v), I[166] = (img)(_n6##x,_n5##y,z,v), I[167] = (img)(_n7##x,_n5##y,z,v), \
 I[168] = (img)(_p6##x,_n6##y,z,v), I[169] = (img)(_p5##x,_n6##y,z,v), I[170] = (img)(_p4##x,_n6##y,z,v), I[171] = (img)(_p3##x,_n6##y,z,v), I[172] = (img)(_p2##x,_n6##y,z,v), I[173] = (img)(_p1##x,_n6##y,z,v), I[174] = (img)(x,_n6##y,z,v), I[175] = (img)(_n1##x,_n6##y,z,v), I[176] = (img)(_n2##x,_n6##y,z,v), I[177] = (img)(_n3##x,_n6##y,z,v), I[178] = (img)(_n4##x,_n6##y,z,v), I[179] = (img)(_n5##x,_n6##y,z,v), I[180] = (img)(_n6##x,_n6##y,z,v), I[181] = (img)(_n7##x,_n6##y,z,v), \
 I[182] = (img)(_p6##x,_n7##y,z,v), I[183] = (img)(_p5##x,_n7##y,z,v), I[184] = (img)(_p4##x,_n7##y,z,v), I[185] = (img)(_p3##x,_n7##y,z,v), I[186] = (img)(_p2##x,_n7##y,z,v), I[187] = (img)(_p1##x,_n7##y,z,v), I[188] = (img)(x,_n7##y,z,v), I[189] = (img)(_n1##x,_n7##y,z,v), I[190] = (img)(_n2##x,_n7##y,z,v), I[191] = (img)(_n3##x,_n7##y,z,v), I[192] = (img)(_n4##x,_n7##y,z,v), I[193] = (img)(_n5##x,_n7##y,z,v), I[194] = (img)(_n6##x,_n7##y,z,v), I[195] = (img)(_n7##x,_n7##y,z,v);

// Define 15x15 loop macros for CImg
//----------------------------------
#define cimg_for15(bound,i) for (int i = 0, \
 _p7##i = 0, _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7; \
 _n7##i<(int)(bound) || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i)

#define cimg_for15X(img,x) cimg_for15((img).width,x)
#define cimg_for15Y(img,y) cimg_for15((img).height,y)
#define cimg_for15Z(img,z) cimg_for15((img).depth,z)
#define cimg_for15V(img,v) cimg_for15((img).dim,v)
#define cimg_for15XY(img,x,y) cimg_for15Y(img,y) cimg_for15X(img,x)
#define cimg_for15XZ(img,x,z) cimg_for15Z(img,z) cimg_for15X(img,x)
#define cimg_for15XV(img,x,v) cimg_for15V(img,v) cimg_for15X(img,x)
#define cimg_for15YZ(img,y,z) cimg_for15Z(img,z) cimg_for15Y(img,y)
#define cimg_for15YV(img,y,v) cimg_for15V(img,v) cimg_for15Y(img,y)
#define cimg_for15ZV(img,z,v) cimg_for15V(img,v) cimg_for15Z(img,z)
#define cimg_for15XYZ(img,x,y,z) cimg_for15Z(img,z) cimg_for15XY(img,x,y)
#define cimg_for15XZV(img,x,z,v) cimg_for15V(img,v) cimg_for15XZ(img,x,z)
#define cimg_for15YZV(img,y,z,v) cimg_for15V(img,v) cimg_for15YZ(img,y,z)
#define cimg_for15XYZV(img,x,y,z,v) cimg_for15V(img,v) cimg_for15XYZ(img,x,y,z)

#define cimg_for_in15(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p7##i = i-7<0?0:i-7, \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7; \
 i<=(int)(i1) && (_n7##i<(int)(bound) || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i)

#define cimg_for_in15X(img,x0,x1,x) cimg_for_in15((img).width,x0,x1,x)
#define cimg_for_in15Y(img,y0,y1,y) cimg_for_in15((img).height,y0,y1,y)
#define cimg_for_in15Z(img,z0,z1,z) cimg_for_in15((img).depth,z0,z1,z)
#define cimg_for_in15V(img,v0,v1,v) cimg_for_in15((img).dim,v0,v1,v)
#define cimg_for_in15XY(img,x0,y0,x1,y1,x,y) cimg_for_in15Y(img,y0,y1,y) cimg_for_in15X(img,x0,x1,x)
#define cimg_for_in15XZ(img,x0,z0,x1,z1,x,z) cimg_for_in15Z(img,z0,z1,z) cimg_for_in15X(img,x0,x1,x)
#define cimg_for_in15XV(img,x0,v0,x1,v1,x,v) cimg_for_in15V(img,v0,v1,v) cimg_for_in15X(img,x0,x1,x)
#define cimg_for_in15YZ(img,y0,z0,y1,z1,y,z) cimg_for_in15Z(img,z0,z1,z) cimg_for_in15Y(img,y0,y1,y)
#define cimg_for_in15YV(img,y0,v0,y1,v1,y,v) cimg_for_in15V(img,v0,v1,v) cimg_for_in15Y(img,y0,y1,y)
#define cimg_for_in15ZV(img,z0,v0,z1,v1,z,v) cimg_for_in15V(img,v0,v1,v) cimg_for_in15Z(img,z0,z1,z)
#define cimg_for_in15XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in15Z(img,z0,z1,z) cimg_for_in15XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in15XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in15V(img,v0,v1,v) cimg_for_in15XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in15YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in15V(img,v0,v1,v) cimg_for_in15YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in15XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in15V(img,v0,v1,v) cimg_for_in15XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for15x15(img,x,y,z,v,I) \
 cimg_for15((img).height,y) for (int x = 0, \
 _p7##x = 0, _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = I[7] = (img)(0,_p7##y,z,v)), \
 (I[15] = I[16] = I[17] = I[18] = I[19] = I[20] = I[21] = I[22] = (img)(0,_p6##y,z,v)), \
 (I[30] = I[31] = I[32] = I[33] = I[34] = I[35] = I[36] = I[37] = (img)(0,_p5##y,z,v)), \
 (I[45] = I[46] = I[47] = I[48] = I[49] = I[50] = I[51] = I[52] = (img)(0,_p4##y,z,v)), \
 (I[60] = I[61] = I[62] = I[63] = I[64] = I[65] = I[66] = I[67] = (img)(0,_p3##y,z,v)), \
 (I[75] = I[76] = I[77] = I[78] = I[79] = I[80] = I[81] = I[82] = (img)(0,_p2##y,z,v)), \
 (I[90] = I[91] = I[92] = I[93] = I[94] = I[95] = I[96] = I[97] = (img)(0,_p1##y,z,v)), \
 (I[105] = I[106] = I[107] = I[108] = I[109] = I[110] = I[111] = I[112] = (img)(0,y,z,v)), \
 (I[120] = I[121] = I[122] = I[123] = I[124] = I[125] = I[126] = I[127] = (img)(0,_n1##y,z,v)), \
 (I[135] = I[136] = I[137] = I[138] = I[139] = I[140] = I[141] = I[142] = (img)(0,_n2##y,z,v)), \
 (I[150] = I[151] = I[152] = I[153] = I[154] = I[155] = I[156] = I[157] = (img)(0,_n3##y,z,v)), \
 (I[165] = I[166] = I[167] = I[168] = I[169] = I[170] = I[171] = I[172] = (img)(0,_n4##y,z,v)), \
 (I[180] = I[181] = I[182] = I[183] = I[184] = I[185] = I[186] = I[187] = (img)(0,_n5##y,z,v)), \
 (I[195] = I[196] = I[197] = I[198] = I[199] = I[200] = I[201] = I[202] = (img)(0,_n6##y,z,v)), \
 (I[210] = I[211] = I[212] = I[213] = I[214] = I[215] = I[216] = I[217] = (img)(0,_n7##y,z,v)), \
 (I[8] = (img)(_n1##x,_p7##y,z,v)), \
 (I[23] = (img)(_n1##x,_p6##y,z,v)), \
 (I[38] = (img)(_n1##x,_p5##y,z,v)), \
 (I[53] = (img)(_n1##x,_p4##y,z,v)), \
 (I[68] = (img)(_n1##x,_p3##y,z,v)), \
 (I[83] = (img)(_n1##x,_p2##y,z,v)), \
 (I[98] = (img)(_n1##x,_p1##y,z,v)), \
 (I[113] = (img)(_n1##x,y,z,v)), \
 (I[128] = (img)(_n1##x,_n1##y,z,v)), \
 (I[143] = (img)(_n1##x,_n2##y,z,v)), \
 (I[158] = (img)(_n1##x,_n3##y,z,v)), \
 (I[173] = (img)(_n1##x,_n4##y,z,v)), \
 (I[188] = (img)(_n1##x,_n5##y,z,v)), \
 (I[203] = (img)(_n1##x,_n6##y,z,v)), \
 (I[218] = (img)(_n1##x,_n7##y,z,v)), \
 (I[9] = (img)(_n2##x,_p7##y,z,v)), \
 (I[24] = (img)(_n2##x,_p6##y,z,v)), \
 (I[39] = (img)(_n2##x,_p5##y,z,v)), \
 (I[54] = (img)(_n2##x,_p4##y,z,v)), \
 (I[69] = (img)(_n2##x,_p3##y,z,v)), \
 (I[84] = (img)(_n2##x,_p2##y,z,v)), \
 (I[99] = (img)(_n2##x,_p1##y,z,v)), \
 (I[114] = (img)(_n2##x,y,z,v)), \
 (I[129] = (img)(_n2##x,_n1##y,z,v)), \
 (I[144] = (img)(_n2##x,_n2##y,z,v)), \
 (I[159] = (img)(_n2##x,_n3##y,z,v)), \
 (I[174] = (img)(_n2##x,_n4##y,z,v)), \
 (I[189] = (img)(_n2##x,_n5##y,z,v)), \
 (I[204] = (img)(_n2##x,_n6##y,z,v)), \
 (I[219] = (img)(_n2##x,_n7##y,z,v)), \
 (I[10] = (img)(_n3##x,_p7##y,z,v)), \
 (I[25] = (img)(_n3##x,_p6##y,z,v)), \
 (I[40] = (img)(_n3##x,_p5##y,z,v)), \
 (I[55] = (img)(_n3##x,_p4##y,z,v)), \
 (I[70] = (img)(_n3##x,_p3##y,z,v)), \
 (I[85] = (img)(_n3##x,_p2##y,z,v)), \
 (I[100] = (img)(_n3##x,_p1##y,z,v)), \
 (I[115] = (img)(_n3##x,y,z,v)), \
 (I[130] = (img)(_n3##x,_n1##y,z,v)), \
 (I[145] = (img)(_n3##x,_n2##y,z,v)), \
 (I[160] = (img)(_n3##x,_n3##y,z,v)), \
 (I[175] = (img)(_n3##x,_n4##y,z,v)), \
 (I[190] = (img)(_n3##x,_n5##y,z,v)), \
 (I[205] = (img)(_n3##x,_n6##y,z,v)), \
 (I[220] = (img)(_n3##x,_n7##y,z,v)), \
 (I[11] = (img)(_n4##x,_p7##y,z,v)), \
 (I[26] = (img)(_n4##x,_p6##y,z,v)), \
 (I[41] = (img)(_n4##x,_p5##y,z,v)), \
 (I[56] = (img)(_n4##x,_p4##y,z,v)), \
 (I[71] = (img)(_n4##x,_p3##y,z,v)), \
 (I[86] = (img)(_n4##x,_p2##y,z,v)), \
 (I[101] = (img)(_n4##x,_p1##y,z,v)), \
 (I[116] = (img)(_n4##x,y,z,v)), \
 (I[131] = (img)(_n4##x,_n1##y,z,v)), \
 (I[146] = (img)(_n4##x,_n2##y,z,v)), \
 (I[161] = (img)(_n4##x,_n3##y,z,v)), \
 (I[176] = (img)(_n4##x,_n4##y,z,v)), \
 (I[191] = (img)(_n4##x,_n5##y,z,v)), \
 (I[206] = (img)(_n4##x,_n6##y,z,v)), \
 (I[221] = (img)(_n4##x,_n7##y,z,v)), \
 (I[12] = (img)(_n5##x,_p7##y,z,v)), \
 (I[27] = (img)(_n5##x,_p6##y,z,v)), \
 (I[42] = (img)(_n5##x,_p5##y,z,v)), \
 (I[57] = (img)(_n5##x,_p4##y,z,v)), \
 (I[72] = (img)(_n5##x,_p3##y,z,v)), \
 (I[87] = (img)(_n5##x,_p2##y,z,v)), \
 (I[102] = (img)(_n5##x,_p1##y,z,v)), \
 (I[117] = (img)(_n5##x,y,z,v)), \
 (I[132] = (img)(_n5##x,_n1##y,z,v)), \
 (I[147] = (img)(_n5##x,_n2##y,z,v)), \
 (I[162] = (img)(_n5##x,_n3##y,z,v)), \
 (I[177] = (img)(_n5##x,_n4##y,z,v)), \
 (I[192] = (img)(_n5##x,_n5##y,z,v)), \
 (I[207] = (img)(_n5##x,_n6##y,z,v)), \
 (I[222] = (img)(_n5##x,_n7##y,z,v)), \
 (I[13] = (img)(_n6##x,_p7##y,z,v)), \
 (I[28] = (img)(_n6##x,_p6##y,z,v)), \
 (I[43] = (img)(_n6##x,_p5##y,z,v)), \
 (I[58] = (img)(_n6##x,_p4##y,z,v)), \
 (I[73] = (img)(_n6##x,_p3##y,z,v)), \
 (I[88] = (img)(_n6##x,_p2##y,z,v)), \
 (I[103] = (img)(_n6##x,_p1##y,z,v)), \
 (I[118] = (img)(_n6##x,y,z,v)), \
 (I[133] = (img)(_n6##x,_n1##y,z,v)), \
 (I[148] = (img)(_n6##x,_n2##y,z,v)), \
 (I[163] = (img)(_n6##x,_n3##y,z,v)), \
 (I[178] = (img)(_n6##x,_n4##y,z,v)), \
 (I[193] = (img)(_n6##x,_n5##y,z,v)), \
 (I[208] = (img)(_n6##x,_n6##y,z,v)), \
 (I[223] = (img)(_n6##x,_n7##y,z,v)), \
 7>=((img).width)?(int)((img).width)-1:7); \
 (_n7##x<(int)((img).width) && ( \
 (I[14] = (img)(_n7##x,_p7##y,z,v)), \
 (I[29] = (img)(_n7##x,_p6##y,z,v)), \
 (I[44] = (img)(_n7##x,_p5##y,z,v)), \
 (I[59] = (img)(_n7##x,_p4##y,z,v)), \
 (I[74] = (img)(_n7##x,_p3##y,z,v)), \
 (I[89] = (img)(_n7##x,_p2##y,z,v)), \
 (I[104] = (img)(_n7##x,_p1##y,z,v)), \
 (I[119] = (img)(_n7##x,y,z,v)), \
 (I[134] = (img)(_n7##x,_n1##y,z,v)), \
 (I[149] = (img)(_n7##x,_n2##y,z,v)), \
 (I[164] = (img)(_n7##x,_n3##y,z,v)), \
 (I[179] = (img)(_n7##x,_n4##y,z,v)), \
 (I[194] = (img)(_n7##x,_n5##y,z,v)), \
 (I[209] = (img)(_n7##x,_n6##y,z,v)), \
 (I[224] = (img)(_n7##x,_n7##y,z,v)),1)) || \
 _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], \
 I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], \
 I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], \
 I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], \
 I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], \
 I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], \
 I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], \
 I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], \
 I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], \
 I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], \
 I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], \
 I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], \
 I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], \
 _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x)

#define cimg_for_in15x15(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in15((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p7##x = x-7<0?0:x-7, \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = (int)( \
 (I[0] = (img)(_p7##x,_p7##y,z,v)), \
 (I[15] = (img)(_p7##x,_p6##y,z,v)), \
 (I[30] = (img)(_p7##x,_p5##y,z,v)), \
 (I[45] = (img)(_p7##x,_p4##y,z,v)), \
 (I[60] = (img)(_p7##x,_p3##y,z,v)), \
 (I[75] = (img)(_p7##x,_p2##y,z,v)), \
 (I[90] = (img)(_p7##x,_p1##y,z,v)), \
 (I[105] = (img)(_p7##x,y,z,v)), \
 (I[120] = (img)(_p7##x,_n1##y,z,v)), \
 (I[135] = (img)(_p7##x,_n2##y,z,v)), \
 (I[150] = (img)(_p7##x,_n3##y,z,v)), \
 (I[165] = (img)(_p7##x,_n4##y,z,v)), \
 (I[180] = (img)(_p7##x,_n5##y,z,v)), \
 (I[195] = (img)(_p7##x,_n6##y,z,v)), \
 (I[210] = (img)(_p7##x,_n7##y,z,v)), \
 (I[1] = (img)(_p6##x,_p7##y,z,v)), \
 (I[16] = (img)(_p6##x,_p6##y,z,v)), \
 (I[31] = (img)(_p6##x,_p5##y,z,v)), \
 (I[46] = (img)(_p6##x,_p4##y,z,v)), \
 (I[61] = (img)(_p6##x,_p3##y,z,v)), \
 (I[76] = (img)(_p6##x,_p2##y,z,v)), \
 (I[91] = (img)(_p6##x,_p1##y,z,v)), \
 (I[106] = (img)(_p6##x,y,z,v)), \
 (I[121] = (img)(_p6##x,_n1##y,z,v)), \
 (I[136] = (img)(_p6##x,_n2##y,z,v)), \
 (I[151] = (img)(_p6##x,_n3##y,z,v)), \
 (I[166] = (img)(_p6##x,_n4##y,z,v)), \
 (I[181] = (img)(_p6##x,_n5##y,z,v)), \
 (I[196] = (img)(_p6##x,_n6##y,z,v)), \
 (I[211] = (img)(_p6##x,_n7##y,z,v)), \
 (I[2] = (img)(_p5##x,_p7##y,z,v)), \
 (I[17] = (img)(_p5##x,_p6##y,z,v)), \
 (I[32] = (img)(_p5##x,_p5##y,z,v)), \
 (I[47] = (img)(_p5##x,_p4##y,z,v)), \
 (I[62] = (img)(_p5##x,_p3##y,z,v)), \
 (I[77] = (img)(_p5##x,_p2##y,z,v)), \
 (I[92] = (img)(_p5##x,_p1##y,z,v)), \
 (I[107] = (img)(_p5##x,y,z,v)), \
 (I[122] = (img)(_p5##x,_n1##y,z,v)), \
 (I[137] = (img)(_p5##x,_n2##y,z,v)), \
 (I[152] = (img)(_p5##x,_n3##y,z,v)), \
 (I[167] = (img)(_p5##x,_n4##y,z,v)), \
 (I[182] = (img)(_p5##x,_n5##y,z,v)), \
 (I[197] = (img)(_p5##x,_n6##y,z,v)), \
 (I[212] = (img)(_p5##x,_n7##y,z,v)), \
 (I[3] = (img)(_p4##x,_p7##y,z,v)), \
 (I[18] = (img)(_p4##x,_p6##y,z,v)), \
 (I[33] = (img)(_p4##x,_p5##y,z,v)), \
 (I[48] = (img)(_p4##x,_p4##y,z,v)), \
 (I[63] = (img)(_p4##x,_p3##y,z,v)), \
 (I[78] = (img)(_p4##x,_p2##y,z,v)), \
 (I[93] = (img)(_p4##x,_p1##y,z,v)), \
 (I[108] = (img)(_p4##x,y,z,v)), \
 (I[123] = (img)(_p4##x,_n1##y,z,v)), \
 (I[138] = (img)(_p4##x,_n2##y,z,v)), \
 (I[153] = (img)(_p4##x,_n3##y,z,v)), \
 (I[168] = (img)(_p4##x,_n4##y,z,v)), \
 (I[183] = (img)(_p4##x,_n5##y,z,v)), \
 (I[198] = (img)(_p4##x,_n6##y,z,v)), \
 (I[213] = (img)(_p4##x,_n7##y,z,v)), \
 (I[4] = (img)(_p3##x,_p7##y,z,v)), \
 (I[19] = (img)(_p3##x,_p6##y,z,v)), \
 (I[34] = (img)(_p3##x,_p5##y,z,v)), \
 (I[49] = (img)(_p3##x,_p4##y,z,v)), \
 (I[64] = (img)(_p3##x,_p3##y,z,v)), \
 (I[79] = (img)(_p3##x,_p2##y,z,v)), \
 (I[94] = (img)(_p3##x,_p1##y,z,v)), \
 (I[109] = (img)(_p3##x,y,z,v)), \
 (I[124] = (img)(_p3##x,_n1##y,z,v)), \
 (I[139] = (img)(_p3##x,_n2##y,z,v)), \
 (I[154] = (img)(_p3##x,_n3##y,z,v)), \
 (I[169] = (img)(_p3##x,_n4##y,z,v)), \
 (I[184] = (img)(_p3##x,_n5##y,z,v)), \
 (I[199] = (img)(_p3##x,_n6##y,z,v)), \
 (I[214] = (img)(_p3##x,_n7##y,z,v)), \
 (I[5] = (img)(_p2##x,_p7##y,z,v)), \
 (I[20] = (img)(_p2##x,_p6##y,z,v)), \
 (I[35] = (img)(_p2##x,_p5##y,z,v)), \
 (I[50] = (img)(_p2##x,_p4##y,z,v)), \
 (I[65] = (img)(_p2##x,_p3##y,z,v)), \
 (I[80] = (img)(_p2##x,_p2##y,z,v)), \
 (I[95] = (img)(_p2##x,_p1##y,z,v)), \
 (I[110] = (img)(_p2##x,y,z,v)), \
 (I[125] = (img)(_p2##x,_n1##y,z,v)), \
 (I[140] = (img)(_p2##x,_n2##y,z,v)), \
 (I[155] = (img)(_p2##x,_n3##y,z,v)), \
 (I[170] = (img)(_p2##x,_n4##y,z,v)), \
 (I[185] = (img)(_p2##x,_n5##y,z,v)), \
 (I[200] = (img)(_p2##x,_n6##y,z,v)), \
 (I[215] = (img)(_p2##x,_n7##y,z,v)), \
 (I[6] = (img)(_p1##x,_p7##y,z,v)), \
 (I[21] = (img)(_p1##x,_p6##y,z,v)), \
 (I[36] = (img)(_p1##x,_p5##y,z,v)), \
 (I[51] = (img)(_p1##x,_p4##y,z,v)), \
 (I[66] = (img)(_p1##x,_p3##y,z,v)), \
 (I[81] = (img)(_p1##x,_p2##y,z,v)), \
 (I[96] = (img)(_p1##x,_p1##y,z,v)), \
 (I[111] = (img)(_p1##x,y,z,v)), \
 (I[126] = (img)(_p1##x,_n1##y,z,v)), \
 (I[141] = (img)(_p1##x,_n2##y,z,v)), \
 (I[156] = (img)(_p1##x,_n3##y,z,v)), \
 (I[171] = (img)(_p1##x,_n4##y,z,v)), \
 (I[186] = (img)(_p1##x,_n5##y,z,v)), \
 (I[201] = (img)(_p1##x,_n6##y,z,v)), \
 (I[216] = (img)(_p1##x,_n7##y,z,v)), \
 (I[7] = (img)(x,_p7##y,z,v)), \
 (I[22] = (img)(x,_p6##y,z,v)), \
 (I[37] = (img)(x,_p5##y,z,v)), \
 (I[52] = (img)(x,_p4##y,z,v)), \
 (I[67] = (img)(x,_p3##y,z,v)), \
 (I[82] = (img)(x,_p2##y,z,v)), \
 (I[97] = (img)(x,_p1##y,z,v)), \
 (I[112] = (img)(x,y,z,v)), \
 (I[127] = (img)(x,_n1##y,z,v)), \
 (I[142] = (img)(x,_n2##y,z,v)), \
 (I[157] = (img)(x,_n3##y,z,v)), \
 (I[172] = (img)(x,_n4##y,z,v)), \
 (I[187] = (img)(x,_n5##y,z,v)), \
 (I[202] = (img)(x,_n6##y,z,v)), \
 (I[217] = (img)(x,_n7##y,z,v)), \
 (I[8] = (img)(_n1##x,_p7##y,z,v)), \
 (I[23] = (img)(_n1##x,_p6##y,z,v)), \
 (I[38] = (img)(_n1##x,_p5##y,z,v)), \
 (I[53] = (img)(_n1##x,_p4##y,z,v)), \
 (I[68] = (img)(_n1##x,_p3##y,z,v)), \
 (I[83] = (img)(_n1##x,_p2##y,z,v)), \
 (I[98] = (img)(_n1##x,_p1##y,z,v)), \
 (I[113] = (img)(_n1##x,y,z,v)), \
 (I[128] = (img)(_n1##x,_n1##y,z,v)), \
 (I[143] = (img)(_n1##x,_n2##y,z,v)), \
 (I[158] = (img)(_n1##x,_n3##y,z,v)), \
 (I[173] = (img)(_n1##x,_n4##y,z,v)), \
 (I[188] = (img)(_n1##x,_n5##y,z,v)), \
 (I[203] = (img)(_n1##x,_n6##y,z,v)), \
 (I[218] = (img)(_n1##x,_n7##y,z,v)), \
 (I[9] = (img)(_n2##x,_p7##y,z,v)), \
 (I[24] = (img)(_n2##x,_p6##y,z,v)), \
 (I[39] = (img)(_n2##x,_p5##y,z,v)), \
 (I[54] = (img)(_n2##x,_p4##y,z,v)), \
 (I[69] = (img)(_n2##x,_p3##y,z,v)), \
 (I[84] = (img)(_n2##x,_p2##y,z,v)), \
 (I[99] = (img)(_n2##x,_p1##y,z,v)), \
 (I[114] = (img)(_n2##x,y,z,v)), \
 (I[129] = (img)(_n2##x,_n1##y,z,v)), \
 (I[144] = (img)(_n2##x,_n2##y,z,v)), \
 (I[159] = (img)(_n2##x,_n3##y,z,v)), \
 (I[174] = (img)(_n2##x,_n4##y,z,v)), \
 (I[189] = (img)(_n2##x,_n5##y,z,v)), \
 (I[204] = (img)(_n2##x,_n6##y,z,v)), \
 (I[219] = (img)(_n2##x,_n7##y,z,v)), \
 (I[10] = (img)(_n3##x,_p7##y,z,v)), \
 (I[25] = (img)(_n3##x,_p6##y,z,v)), \
 (I[40] = (img)(_n3##x,_p5##y,z,v)), \
 (I[55] = (img)(_n3##x,_p4##y,z,v)), \
 (I[70] = (img)(_n3##x,_p3##y,z,v)), \
 (I[85] = (img)(_n3##x,_p2##y,z,v)), \
 (I[100] = (img)(_n3##x,_p1##y,z,v)), \
 (I[115] = (img)(_n3##x,y,z,v)), \
 (I[130] = (img)(_n3##x,_n1##y,z,v)), \
 (I[145] = (img)(_n3##x,_n2##y,z,v)), \
 (I[160] = (img)(_n3##x,_n3##y,z,v)), \
 (I[175] = (img)(_n3##x,_n4##y,z,v)), \
 (I[190] = (img)(_n3##x,_n5##y,z,v)), \
 (I[205] = (img)(_n3##x,_n6##y,z,v)), \
 (I[220] = (img)(_n3##x,_n7##y,z,v)), \
 (I[11] = (img)(_n4##x,_p7##y,z,v)), \
 (I[26] = (img)(_n4##x,_p6##y,z,v)), \
 (I[41] = (img)(_n4##x,_p5##y,z,v)), \
 (I[56] = (img)(_n4##x,_p4##y,z,v)), \
 (I[71] = (img)(_n4##x,_p3##y,z,v)), \
 (I[86] = (img)(_n4##x,_p2##y,z,v)), \
 (I[101] = (img)(_n4##x,_p1##y,z,v)), \
 (I[116] = (img)(_n4##x,y,z,v)), \
 (I[131] = (img)(_n4##x,_n1##y,z,v)), \
 (I[146] = (img)(_n4##x,_n2##y,z,v)), \
 (I[161] = (img)(_n4##x,_n3##y,z,v)), \
 (I[176] = (img)(_n4##x,_n4##y,z,v)), \
 (I[191] = (img)(_n4##x,_n5##y,z,v)), \
 (I[206] = (img)(_n4##x,_n6##y,z,v)), \
 (I[221] = (img)(_n4##x,_n7##y,z,v)), \
 (I[12] = (img)(_n5##x,_p7##y,z,v)), \
 (I[27] = (img)(_n5##x,_p6##y,z,v)), \
 (I[42] = (img)(_n5##x,_p5##y,z,v)), \
 (I[57] = (img)(_n5##x,_p4##y,z,v)), \
 (I[72] = (img)(_n5##x,_p3##y,z,v)), \
 (I[87] = (img)(_n5##x,_p2##y,z,v)), \
 (I[102] = (img)(_n5##x,_p1##y,z,v)), \
 (I[117] = (img)(_n5##x,y,z,v)), \
 (I[132] = (img)(_n5##x,_n1##y,z,v)), \
 (I[147] = (img)(_n5##x,_n2##y,z,v)), \
 (I[162] = (img)(_n5##x,_n3##y,z,v)), \
 (I[177] = (img)(_n5##x,_n4##y,z,v)), \
 (I[192] = (img)(_n5##x,_n5##y,z,v)), \
 (I[207] = (img)(_n5##x,_n6##y,z,v)), \
 (I[222] = (img)(_n5##x,_n7##y,z,v)), \
 (I[13] = (img)(_n6##x,_p7##y,z,v)), \
 (I[28] = (img)(_n6##x,_p6##y,z,v)), \
 (I[43] = (img)(_n6##x,_p5##y,z,v)), \
 (I[58] = (img)(_n6##x,_p4##y,z,v)), \
 (I[73] = (img)(_n6##x,_p3##y,z,v)), \
 (I[88] = (img)(_n6##x,_p2##y,z,v)), \
 (I[103] = (img)(_n6##x,_p1##y,z,v)), \
 (I[118] = (img)(_n6##x,y,z,v)), \
 (I[133] = (img)(_n6##x,_n1##y,z,v)), \
 (I[148] = (img)(_n6##x,_n2##y,z,v)), \
 (I[163] = (img)(_n6##x,_n3##y,z,v)), \
 (I[178] = (img)(_n6##x,_n4##y,z,v)), \
 (I[193] = (img)(_n6##x,_n5##y,z,v)), \
 (I[208] = (img)(_n6##x,_n6##y,z,v)), \
 (I[223] = (img)(_n6##x,_n7##y,z,v)), \
 x+7>=(int)((img).width)?(int)((img).width)-1:x+7); \
 x<=(int)(x1) && ((_n7##x<(int)((img).width) && ( \
 (I[14] = (img)(_n7##x,_p7##y,z,v)), \
 (I[29] = (img)(_n7##x,_p6##y,z,v)), \
 (I[44] = (img)(_n7##x,_p5##y,z,v)), \
 (I[59] = (img)(_n7##x,_p4##y,z,v)), \
 (I[74] = (img)(_n7##x,_p3##y,z,v)), \
 (I[89] = (img)(_n7##x,_p2##y,z,v)), \
 (I[104] = (img)(_n7##x,_p1##y,z,v)), \
 (I[119] = (img)(_n7##x,y,z,v)), \
 (I[134] = (img)(_n7##x,_n1##y,z,v)), \
 (I[149] = (img)(_n7##x,_n2##y,z,v)), \
 (I[164] = (img)(_n7##x,_n3##y,z,v)), \
 (I[179] = (img)(_n7##x,_n4##y,z,v)), \
 (I[194] = (img)(_n7##x,_n5##y,z,v)), \
 (I[209] = (img)(_n7##x,_n6##y,z,v)), \
 (I[224] = (img)(_n7##x,_n7##y,z,v)),1)) || \
 _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], \
 I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], \
 I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], \
 I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], \
 I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], \
 I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], \
 I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], \
 I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], \
 I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], \
 I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], \
 I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], \
 I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], \
 I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], \
 _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x)

#define cimg_get15x15(img,x,y,z,v,I) \
 I[0] = (img)(_p7##x,_p7##y,z,v), I[1] = (img)(_p6##x,_p7##y,z,v), I[2] = (img)(_p5##x,_p7##y,z,v), I[3] = (img)(_p4##x,_p7##y,z,v), I[4] = (img)(_p3##x,_p7##y,z,v), I[5] = (img)(_p2##x,_p7##y,z,v), I[6] = (img)(_p1##x,_p7##y,z,v), I[7] = (img)(x,_p7##y,z,v), I[8] = (img)(_n1##x,_p7##y,z,v), I[9] = (img)(_n2##x,_p7##y,z,v), I[10] = (img)(_n3##x,_p7##y,z,v), I[11] = (img)(_n4##x,_p7##y,z,v), I[12] = (img)(_n5##x,_p7##y,z,v), I[13] = (img)(_n6##x,_p7##y,z,v), I[14] = (img)(_n7##x,_p7##y,z,v), \
 I[15] = (img)(_p7##x,_p6##y,z,v), I[16] = (img)(_p6##x,_p6##y,z,v), I[17] = (img)(_p5##x,_p6##y,z,v), I[18] = (img)(_p4##x,_p6##y,z,v), I[19] = (img)(_p3##x,_p6##y,z,v), I[20] = (img)(_p2##x,_p6##y,z,v), I[21] = (img)(_p1##x,_p6##y,z,v), I[22] = (img)(x,_p6##y,z,v), I[23] = (img)(_n1##x,_p6##y,z,v), I[24] = (img)(_n2##x,_p6##y,z,v), I[25] = (img)(_n3##x,_p6##y,z,v), I[26] = (img)(_n4##x,_p6##y,z,v), I[27] = (img)(_n5##x,_p6##y,z,v), I[28] = (img)(_n6##x,_p6##y,z,v), I[29] = (img)(_n7##x,_p6##y,z,v), \
 I[30] = (img)(_p7##x,_p5##y,z,v), I[31] = (img)(_p6##x,_p5##y,z,v), I[32] = (img)(_p5##x,_p5##y,z,v), I[33] = (img)(_p4##x,_p5##y,z,v), I[34] = (img)(_p3##x,_p5##y,z,v), I[35] = (img)(_p2##x,_p5##y,z,v), I[36] = (img)(_p1##x,_p5##y,z,v), I[37] = (img)(x,_p5##y,z,v), I[38] = (img)(_n1##x,_p5##y,z,v), I[39] = (img)(_n2##x,_p5##y,z,v), I[40] = (img)(_n3##x,_p5##y,z,v), I[41] = (img)(_n4##x,_p5##y,z,v), I[42] = (img)(_n5##x,_p5##y,z,v), I[43] = (img)(_n6##x,_p5##y,z,v), I[44] = (img)(_n7##x,_p5##y,z,v), \
 I[45] = (img)(_p7##x,_p4##y,z,v), I[46] = (img)(_p6##x,_p4##y,z,v), I[47] = (img)(_p5##x,_p4##y,z,v), I[48] = (img)(_p4##x,_p4##y,z,v), I[49] = (img)(_p3##x,_p4##y,z,v), I[50] = (img)(_p2##x,_p4##y,z,v), I[51] = (img)(_p1##x,_p4##y,z,v), I[52] = (img)(x,_p4##y,z,v), I[53] = (img)(_n1##x,_p4##y,z,v), I[54] = (img)(_n2##x,_p4##y,z,v), I[55] = (img)(_n3##x,_p4##y,z,v), I[56] = (img)(_n4##x,_p4##y,z,v), I[57] = (img)(_n5##x,_p4##y,z,v), I[58] = (img)(_n6##x,_p4##y,z,v), I[59] = (img)(_n7##x,_p4##y,z,v), \
 I[60] = (img)(_p7##x,_p3##y,z,v), I[61] = (img)(_p6##x,_p3##y,z,v), I[62] = (img)(_p5##x,_p3##y,z,v), I[63] = (img)(_p4##x,_p3##y,z,v), I[64] = (img)(_p3##x,_p3##y,z,v), I[65] = (img)(_p2##x,_p3##y,z,v), I[66] = (img)(_p1##x,_p3##y,z,v), I[67] = (img)(x,_p3##y,z,v), I[68] = (img)(_n1##x,_p3##y,z,v), I[69] = (img)(_n2##x,_p3##y,z,v), I[70] = (img)(_n3##x,_p3##y,z,v), I[71] = (img)(_n4##x,_p3##y,z,v), I[72] = (img)(_n5##x,_p3##y,z,v), I[73] = (img)(_n6##x,_p3##y,z,v), I[74] = (img)(_n7##x,_p3##y,z,v), \
 I[75] = (img)(_p7##x,_p2##y,z,v), I[76] = (img)(_p6##x,_p2##y,z,v), I[77] = (img)(_p5##x,_p2##y,z,v), I[78] = (img)(_p4##x,_p2##y,z,v), I[79] = (img)(_p3##x,_p2##y,z,v), I[80] = (img)(_p2##x,_p2##y,z,v), I[81] = (img)(_p1##x,_p2##y,z,v), I[82] = (img)(x,_p2##y,z,v), I[83] = (img)(_n1##x,_p2##y,z,v), I[84] = (img)(_n2##x,_p2##y,z,v), I[85] = (img)(_n3##x,_p2##y,z,v), I[86] = (img)(_n4##x,_p2##y,z,v), I[87] = (img)(_n5##x,_p2##y,z,v), I[88] = (img)(_n6##x,_p2##y,z,v), I[89] = (img)(_n7##x,_p2##y,z,v), \
 I[90] = (img)(_p7##x,_p1##y,z,v), I[91] = (img)(_p6##x,_p1##y,z,v), I[92] = (img)(_p5##x,_p1##y,z,v), I[93] = (img)(_p4##x,_p1##y,z,v), I[94] = (img)(_p3##x,_p1##y,z,v), I[95] = (img)(_p2##x,_p1##y,z,v), I[96] = (img)(_p1##x,_p1##y,z,v), I[97] = (img)(x,_p1##y,z,v), I[98] = (img)(_n1##x,_p1##y,z,v), I[99] = (img)(_n2##x,_p1##y,z,v), I[100] = (img)(_n3##x,_p1##y,z,v), I[101] = (img)(_n4##x,_p1##y,z,v), I[102] = (img)(_n5##x,_p1##y,z,v), I[103] = (img)(_n6##x,_p1##y,z,v), I[104] = (img)(_n7##x,_p1##y,z,v), \
 I[105] = (img)(_p7##x,y,z,v), I[106] = (img)(_p6##x,y,z,v), I[107] = (img)(_p5##x,y,z,v), I[108] = (img)(_p4##x,y,z,v), I[109] = (img)(_p3##x,y,z,v), I[110] = (img)(_p2##x,y,z,v), I[111] = (img)(_p1##x,y,z,v), I[112] = (img)(x,y,z,v), I[113] = (img)(_n1##x,y,z,v), I[114] = (img)(_n2##x,y,z,v), I[115] = (img)(_n3##x,y,z,v), I[116] = (img)(_n4##x,y,z,v), I[117] = (img)(_n5##x,y,z,v), I[118] = (img)(_n6##x,y,z,v), I[119] = (img)(_n7##x,y,z,v), \
 I[120] = (img)(_p7##x,_n1##y,z,v), I[121] = (img)(_p6##x,_n1##y,z,v), I[122] = (img)(_p5##x,_n1##y,z,v), I[123] = (img)(_p4##x,_n1##y,z,v), I[124] = (img)(_p3##x,_n1##y,z,v), I[125] = (img)(_p2##x,_n1##y,z,v), I[126] = (img)(_p1##x,_n1##y,z,v), I[127] = (img)(x,_n1##y,z,v), I[128] = (img)(_n1##x,_n1##y,z,v), I[129] = (img)(_n2##x,_n1##y,z,v), I[130] = (img)(_n3##x,_n1##y,z,v), I[131] = (img)(_n4##x,_n1##y,z,v), I[132] = (img)(_n5##x,_n1##y,z,v), I[133] = (img)(_n6##x,_n1##y,z,v), I[134] = (img)(_n7##x,_n1##y,z,v), \
 I[135] = (img)(_p7##x,_n2##y,z,v), I[136] = (img)(_p6##x,_n2##y,z,v), I[137] = (img)(_p5##x,_n2##y,z,v), I[138] = (img)(_p4##x,_n2##y,z,v), I[139] = (img)(_p3##x,_n2##y,z,v), I[140] = (img)(_p2##x,_n2##y,z,v), I[141] = (img)(_p1##x,_n2##y,z,v), I[142] = (img)(x,_n2##y,z,v), I[143] = (img)(_n1##x,_n2##y,z,v), I[144] = (img)(_n2##x,_n2##y,z,v), I[145] = (img)(_n3##x,_n2##y,z,v), I[146] = (img)(_n4##x,_n2##y,z,v), I[147] = (img)(_n5##x,_n2##y,z,v), I[148] = (img)(_n6##x,_n2##y,z,v), I[149] = (img)(_n7##x,_n2##y,z,v), \
 I[150] = (img)(_p7##x,_n3##y,z,v), I[151] = (img)(_p6##x,_n3##y,z,v), I[152] = (img)(_p5##x,_n3##y,z,v), I[153] = (img)(_p4##x,_n3##y,z,v), I[154] = (img)(_p3##x,_n3##y,z,v), I[155] = (img)(_p2##x,_n3##y,z,v), I[156] = (img)(_p1##x,_n3##y,z,v), I[157] = (img)(x,_n3##y,z,v), I[158] = (img)(_n1##x,_n3##y,z,v), I[159] = (img)(_n2##x,_n3##y,z,v), I[160] = (img)(_n3##x,_n3##y,z,v), I[161] = (img)(_n4##x,_n3##y,z,v), I[162] = (img)(_n5##x,_n3##y,z,v), I[163] = (img)(_n6##x,_n3##y,z,v), I[164] = (img)(_n7##x,_n3##y,z,v), \
 I[165] = (img)(_p7##x,_n4##y,z,v), I[166] = (img)(_p6##x,_n4##y,z,v), I[167] = (img)(_p5##x,_n4##y,z,v), I[168] = (img)(_p4##x,_n4##y,z,v), I[169] = (img)(_p3##x,_n4##y,z,v), I[170] = (img)(_p2##x,_n4##y,z,v), I[171] = (img)(_p1##x,_n4##y,z,v), I[172] = (img)(x,_n4##y,z,v), I[173] = (img)(_n1##x,_n4##y,z,v), I[174] = (img)(_n2##x,_n4##y,z,v), I[175] = (img)(_n3##x,_n4##y,z,v), I[176] = (img)(_n4##x,_n4##y,z,v), I[177] = (img)(_n5##x,_n4##y,z,v), I[178] = (img)(_n6##x,_n4##y,z,v), I[179] = (img)(_n7##x,_n4##y,z,v), \
 I[180] = (img)(_p7##x,_n5##y,z,v), I[181] = (img)(_p6##x,_n5##y,z,v), I[182] = (img)(_p5##x,_n5##y,z,v), I[183] = (img)(_p4##x,_n5##y,z,v), I[184] = (img)(_p3##x,_n5##y,z,v), I[185] = (img)(_p2##x,_n5##y,z,v), I[186] = (img)(_p1##x,_n5##y,z,v), I[187] = (img)(x,_n5##y,z,v), I[188] = (img)(_n1##x,_n5##y,z,v), I[189] = (img)(_n2##x,_n5##y,z,v), I[190] = (img)(_n3##x,_n5##y,z,v), I[191] = (img)(_n4##x,_n5##y,z,v), I[192] = (img)(_n5##x,_n5##y,z,v), I[193] = (img)(_n6##x,_n5##y,z,v), I[194] = (img)(_n7##x,_n5##y,z,v), \
 I[195] = (img)(_p7##x,_n6##y,z,v), I[196] = (img)(_p6##x,_n6##y,z,v), I[197] = (img)(_p5##x,_n6##y,z,v), I[198] = (img)(_p4##x,_n6##y,z,v), I[199] = (img)(_p3##x,_n6##y,z,v), I[200] = (img)(_p2##x,_n6##y,z,v), I[201] = (img)(_p1##x,_n6##y,z,v), I[202] = (img)(x,_n6##y,z,v), I[203] = (img)(_n1##x,_n6##y,z,v), I[204] = (img)(_n2##x,_n6##y,z,v), I[205] = (img)(_n3##x,_n6##y,z,v), I[206] = (img)(_n4##x,_n6##y,z,v), I[207] = (img)(_n5##x,_n6##y,z,v), I[208] = (img)(_n6##x,_n6##y,z,v), I[209] = (img)(_n7##x,_n6##y,z,v), \
 I[210] = (img)(_p7##x,_n7##y,z,v), I[211] = (img)(_p6##x,_n7##y,z,v), I[212] = (img)(_p5##x,_n7##y,z,v), I[213] = (img)(_p4##x,_n7##y,z,v), I[214] = (img)(_p3##x,_n7##y,z,v), I[215] = (img)(_p2##x,_n7##y,z,v), I[216] = (img)(_p1##x,_n7##y,z,v), I[217] = (img)(x,_n7##y,z,v), I[218] = (img)(_n1##x,_n7##y,z,v), I[219] = (img)(_n2##x,_n7##y,z,v), I[220] = (img)(_n3##x,_n7##y,z,v), I[221] = (img)(_n4##x,_n7##y,z,v), I[222] = (img)(_n5##x,_n7##y,z,v), I[223] = (img)(_n6##x,_n7##y,z,v), I[224] = (img)(_n7##x,_n7##y,z,v);

// Define 16x16 loop macros for CImg
//----------------------------------
#define cimg_for16(bound,i) for (int i = 0, \
 _p7##i = 0, _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7, \
 _n8##i = 8>=(int)(bound)?(int)(bound)-1:8; \
 _n8##i<(int)(bound) || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i)

#define cimg_for16X(img,x) cimg_for16((img).width,x)
#define cimg_for16Y(img,y) cimg_for16((img).height,y)
#define cimg_for16Z(img,z) cimg_for16((img).depth,z)
#define cimg_for16V(img,v) cimg_for16((img).dim,v)
#define cimg_for16XY(img,x,y) cimg_for16Y(img,y) cimg_for16X(img,x)
#define cimg_for16XZ(img,x,z) cimg_for16Z(img,z) cimg_for16X(img,x)
#define cimg_for16XV(img,x,v) cimg_for16V(img,v) cimg_for16X(img,x)
#define cimg_for16YZ(img,y,z) cimg_for16Z(img,z) cimg_for16Y(img,y)
#define cimg_for16YV(img,y,v) cimg_for16V(img,v) cimg_for16Y(img,y)
#define cimg_for16ZV(img,z,v) cimg_for16V(img,v) cimg_for16Z(img,z)
#define cimg_for16XYZ(img,x,y,z) cimg_for16Z(img,z) cimg_for16XY(img,x,y)
#define cimg_for16XZV(img,x,z,v) cimg_for16V(img,v) cimg_for16XZ(img,x,z)
#define cimg_for16YZV(img,y,z,v) cimg_for16V(img,v) cimg_for16YZ(img,y,z)
#define cimg_for16XYZV(img,x,y,z,v) cimg_for16V(img,v) cimg_for16XYZ(img,x,y,z)

#define cimg_for_in16(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p7##i = i-7<0?0:i-7, \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7, \
 _n8##i = i+8>=(int)(bound)?(int)(bound)-1:i+8; \
 i<=(int)(i1) && (_n8##i<(int)(bound) || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i)

#define cimg_for_in16X(img,x0,x1,x) cimg_for_in16((img).width,x0,x1,x)
#define cimg_for_in16Y(img,y0,y1,y) cimg_for_in16((img).height,y0,y1,y)
#define cimg_for_in16Z(img,z0,z1,z) cimg_for_in16((img).depth,z0,z1,z)
#define cimg_for_in16V(img,v0,v1,v) cimg_for_in16((img).dim,v0,v1,v)
#define cimg_for_in16XY(img,x0,y0,x1,y1,x,y) cimg_for_in16Y(img,y0,y1,y) cimg_for_in16X(img,x0,x1,x)
#define cimg_for_in16XZ(img,x0,z0,x1,z1,x,z) cimg_for_in16Z(img,z0,z1,z) cimg_for_in16X(img,x0,x1,x)
#define cimg_for_in16XV(img,x0,v0,x1,v1,x,v) cimg_for_in16V(img,v0,v1,v) cimg_for_in16X(img,x0,x1,x)
#define cimg_for_in16YZ(img,y0,z0,y1,z1,y,z) cimg_for_in16Z(img,z0,z1,z) cimg_for_in16Y(img,y0,y1,y)
#define cimg_for_in16YV(img,y0,v0,y1,v1,y,v) cimg_for_in16V(img,v0,v1,v) cimg_for_in16Y(img,y0,y1,y)
#define cimg_for_in16ZV(img,z0,v0,z1,v1,z,v) cimg_for_in16V(img,v0,v1,v) cimg_for_in16Z(img,z0,z1,z)
#define cimg_for_in16XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in16Z(img,z0,z1,z) cimg_for_in16XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in16XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in16V(img,v0,v1,v) cimg_for_in16XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in16YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in16V(img,v0,v1,v) cimg_for_in16YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in16XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in16V(img,v0,v1,v) cimg_for_in16XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for16x16(img,x,y,z,v,I) \
 cimg_for16((img).height,y) for (int x = 0, \
 _p7##x = 0, _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = 7>=((img).width)?(int)((img).width)-1:7, \
 _n8##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = I[7] = (img)(0,_p7##y,z,v)), \
 (I[16] = I[17] = I[18] = I[19] = I[20] = I[21] = I[22] = I[23] = (img)(0,_p6##y,z,v)), \
 (I[32] = I[33] = I[34] = I[35] = I[36] = I[37] = I[38] = I[39] = (img)(0,_p5##y,z,v)), \
 (I[48] = I[49] = I[50] = I[51] = I[52] = I[53] = I[54] = I[55] = (img)(0,_p4##y,z,v)), \
 (I[64] = I[65] = I[66] = I[67] = I[68] = I[69] = I[70] = I[71] = (img)(0,_p3##y,z,v)), \
 (I[80] = I[81] = I[82] = I[83] = I[84] = I[85] = I[86] = I[87] = (img)(0,_p2##y,z,v)), \
 (I[96] = I[97] = I[98] = I[99] = I[100] = I[101] = I[102] = I[103] = (img)(0,_p1##y,z,v)), \
 (I[112] = I[113] = I[114] = I[115] = I[116] = I[117] = I[118] = I[119] = (img)(0,y,z,v)), \
 (I[128] = I[129] = I[130] = I[131] = I[132] = I[133] = I[134] = I[135] = (img)(0,_n1##y,z,v)), \
 (I[144] = I[145] = I[146] = I[147] = I[148] = I[149] = I[150] = I[151] = (img)(0,_n2##y,z,v)), \
 (I[160] = I[161] = I[162] = I[163] = I[164] = I[165] = I[166] = I[167] = (img)(0,_n3##y,z,v)), \
 (I[176] = I[177] = I[178] = I[179] = I[180] = I[181] = I[182] = I[183] = (img)(0,_n4##y,z,v)), \
 (I[192] = I[193] = I[194] = I[195] = I[196] = I[197] = I[198] = I[199] = (img)(0,_n5##y,z,v)), \
 (I[208] = I[209] = I[210] = I[211] = I[212] = I[213] = I[214] = I[215] = (img)(0,_n6##y,z,v)), \
 (I[224] = I[225] = I[226] = I[227] = I[228] = I[229] = I[230] = I[231] = (img)(0,_n7##y,z,v)), \
 (I[240] = I[241] = I[242] = I[243] = I[244] = I[245] = I[246] = I[247] = (img)(0,_n8##y,z,v)), \
 (I[8] = (img)(_n1##x,_p7##y,z,v)), \
 (I[24] = (img)(_n1##x,_p6##y,z,v)), \
 (I[40] = (img)(_n1##x,_p5##y,z,v)), \
 (I[56] = (img)(_n1##x,_p4##y,z,v)), \
 (I[72] = (img)(_n1##x,_p3##y,z,v)), \
 (I[88] = (img)(_n1##x,_p2##y,z,v)), \
 (I[104] = (img)(_n1##x,_p1##y,z,v)), \
 (I[120] = (img)(_n1##x,y,z,v)), \
 (I[136] = (img)(_n1##x,_n1##y,z,v)), \
 (I[152] = (img)(_n1##x,_n2##y,z,v)), \
 (I[168] = (img)(_n1##x,_n3##y,z,v)), \
 (I[184] = (img)(_n1##x,_n4##y,z,v)), \
 (I[200] = (img)(_n1##x,_n5##y,z,v)), \
 (I[216] = (img)(_n1##x,_n6##y,z,v)), \
 (I[232] = (img)(_n1##x,_n7##y,z,v)), \
 (I[248] = (img)(_n1##x,_n8##y,z,v)), \
 (I[9] = (img)(_n2##x,_p7##y,z,v)), \
 (I[25] = (img)(_n2##x,_p6##y,z,v)), \
 (I[41] = (img)(_n2##x,_p5##y,z,v)), \
 (I[57] = (img)(_n2##x,_p4##y,z,v)), \
 (I[73] = (img)(_n2##x,_p3##y,z,v)), \
 (I[89] = (img)(_n2##x,_p2##y,z,v)), \
 (I[105] = (img)(_n2##x,_p1##y,z,v)), \
 (I[121] = (img)(_n2##x,y,z,v)), \
 (I[137] = (img)(_n2##x,_n1##y,z,v)), \
 (I[153] = (img)(_n2##x,_n2##y,z,v)), \
 (I[169] = (img)(_n2##x,_n3##y,z,v)), \
 (I[185] = (img)(_n2##x,_n4##y,z,v)), \
 (I[201] = (img)(_n2##x,_n5##y,z,v)), \
 (I[217] = (img)(_n2##x,_n6##y,z,v)), \
 (I[233] = (img)(_n2##x,_n7##y,z,v)), \
 (I[249] = (img)(_n2##x,_n8##y,z,v)), \
 (I[10] = (img)(_n3##x,_p7##y,z,v)), \
 (I[26] = (img)(_n3##x,_p6##y,z,v)), \
 (I[42] = (img)(_n3##x,_p5##y,z,v)), \
 (I[58] = (img)(_n3##x,_p4##y,z,v)), \
 (I[74] = (img)(_n3##x,_p3##y,z,v)), \
 (I[90] = (img)(_n3##x,_p2##y,z,v)), \
 (I[106] = (img)(_n3##x,_p1##y,z,v)), \
 (I[122] = (img)(_n3##x,y,z,v)), \
 (I[138] = (img)(_n3##x,_n1##y,z,v)), \
 (I[154] = (img)(_n3##x,_n2##y,z,v)), \
 (I[170] = (img)(_n3##x,_n3##y,z,v)), \
 (I[186] = (img)(_n3##x,_n4##y,z,v)), \
 (I[202] = (img)(_n3##x,_n5##y,z,v)), \
 (I[218] = (img)(_n3##x,_n6##y,z,v)), \
 (I[234] = (img)(_n3##x,_n7##y,z,v)), \
 (I[250] = (img)(_n3##x,_n8##y,z,v)), \
 (I[11] = (img)(_n4##x,_p7##y,z,v)), \
 (I[27] = (img)(_n4##x,_p6##y,z,v)), \
 (I[43] = (img)(_n4##x,_p5##y,z,v)), \
 (I[59] = (img)(_n4##x,_p4##y,z,v)), \
 (I[75] = (img)(_n4##x,_p3##y,z,v)), \
 (I[91] = (img)(_n4##x,_p2##y,z,v)), \
 (I[107] = (img)(_n4##x,_p1##y,z,v)), \
 (I[123] = (img)(_n4##x,y,z,v)), \
 (I[139] = (img)(_n4##x,_n1##y,z,v)), \
 (I[155] = (img)(_n4##x,_n2##y,z,v)), \
 (I[171] = (img)(_n4##x,_n3##y,z,v)), \
 (I[187] = (img)(_n4##x,_n4##y,z,v)), \
 (I[203] = (img)(_n4##x,_n5##y,z,v)), \
 (I[219] = (img)(_n4##x,_n6##y,z,v)), \
 (I[235] = (img)(_n4##x,_n7##y,z,v)), \
 (I[251] = (img)(_n4##x,_n8##y,z,v)), \
 (I[12] = (img)(_n5##x,_p7##y,z,v)), \
 (I[28] = (img)(_n5##x,_p6##y,z,v)), \
 (I[44] = (img)(_n5##x,_p5##y,z,v)), \
 (I[60] = (img)(_n5##x,_p4##y,z,v)), \
 (I[76] = (img)(_n5##x,_p3##y,z,v)), \
 (I[92] = (img)(_n5##x,_p2##y,z,v)), \
 (I[108] = (img)(_n5##x,_p1##y,z,v)), \
 (I[124] = (img)(_n5##x,y,z,v)), \
 (I[140] = (img)(_n5##x,_n1##y,z,v)), \
 (I[156] = (img)(_n5##x,_n2##y,z,v)), \
 (I[172] = (img)(_n5##x,_n3##y,z,v)), \
 (I[188] = (img)(_n5##x,_n4##y,z,v)), \
 (I[204] = (img)(_n5##x,_n5##y,z,v)), \
 (I[220] = (img)(_n5##x,_n6##y,z,v)), \
 (I[236] = (img)(_n5##x,_n7##y,z,v)), \
 (I[252] = (img)(_n5##x,_n8##y,z,v)), \
 (I[13] = (img)(_n6##x,_p7##y,z,v)), \
 (I[29] = (img)(_n6##x,_p6##y,z,v)), \
 (I[45] = (img)(_n6##x,_p5##y,z,v)), \
 (I[61] = (img)(_n6##x,_p4##y,z,v)), \
 (I[77] = (img)(_n6##x,_p3##y,z,v)), \
 (I[93] = (img)(_n6##x,_p2##y,z,v)), \
 (I[109] = (img)(_n6##x,_p1##y,z,v)), \
 (I[125] = (img)(_n6##x,y,z,v)), \
 (I[141] = (img)(_n6##x,_n1##y,z,v)), \
 (I[157] = (img)(_n6##x,_n2##y,z,v)), \
 (I[173] = (img)(_n6##x,_n3##y,z,v)), \
 (I[189] = (img)(_n6##x,_n4##y,z,v)), \
 (I[205] = (img)(_n6##x,_n5##y,z,v)), \
 (I[221] = (img)(_n6##x,_n6##y,z,v)), \
 (I[237] = (img)(_n6##x,_n7##y,z,v)), \
 (I[253] = (img)(_n6##x,_n8##y,z,v)), \
 (I[14] = (img)(_n7##x,_p7##y,z,v)), \
 (I[30] = (img)(_n7##x,_p6##y,z,v)), \
 (I[46] = (img)(_n7##x,_p5##y,z,v)), \
 (I[62] = (img)(_n7##x,_p4##y,z,v)), \
 (I[78] = (img)(_n7##x,_p3##y,z,v)), \
 (I[94] = (img)(_n7##x,_p2##y,z,v)), \
 (I[110] = (img)(_n7##x,_p1##y,z,v)), \
 (I[126] = (img)(_n7##x,y,z,v)), \
 (I[142] = (img)(_n7##x,_n1##y,z,v)), \
 (I[158] = (img)(_n7##x,_n2##y,z,v)), \
 (I[174] = (img)(_n7##x,_n3##y,z,v)), \
 (I[190] = (img)(_n7##x,_n4##y,z,v)), \
 (I[206] = (img)(_n7##x,_n5##y,z,v)), \
 (I[222] = (img)(_n7##x,_n6##y,z,v)), \
 (I[238] = (img)(_n7##x,_n7##y,z,v)), \
 (I[254] = (img)(_n7##x,_n8##y,z,v)), \
 8>=((img).width)?(int)((img).width)-1:8); \
 (_n8##x<(int)((img).width) && ( \
 (I[15] = (img)(_n8##x,_p7##y,z,v)), \
 (I[31] = (img)(_n8##x,_p6##y,z,v)), \
 (I[47] = (img)(_n8##x,_p5##y,z,v)), \
 (I[63] = (img)(_n8##x,_p4##y,z,v)), \
 (I[79] = (img)(_n8##x,_p3##y,z,v)), \
 (I[95] = (img)(_n8##x,_p2##y,z,v)), \
 (I[111] = (img)(_n8##x,_p1##y,z,v)), \
 (I[127] = (img)(_n8##x,y,z,v)), \
 (I[143] = (img)(_n8##x,_n1##y,z,v)), \
 (I[159] = (img)(_n8##x,_n2##y,z,v)), \
 (I[175] = (img)(_n8##x,_n3##y,z,v)), \
 (I[191] = (img)(_n8##x,_n4##y,z,v)), \
 (I[207] = (img)(_n8##x,_n5##y,z,v)), \
 (I[223] = (img)(_n8##x,_n6##y,z,v)), \
 (I[239] = (img)(_n8##x,_n7##y,z,v)), \
 (I[255] = (img)(_n8##x,_n8##y,z,v)),1)) || \
 _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], \
 I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], \
 I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], \
 I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], \
 I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], \
 I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], \
 I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], \
 I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], \
 I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], \
 I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], \
 I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], \
 I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], \
 I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], \
 I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], \
 _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x)

#define cimg_for_in16x16(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in16((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p7##x = x-7<0?0:x-7, \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = x+7>=(int)((img).width)?(int)((img).width)-1:x+7, \
 _n8##x = (int)( \
 (I[0] = (img)(_p7##x,_p7##y,z,v)), \
 (I[16] = (img)(_p7##x,_p6##y,z,v)), \
 (I[32] = (img)(_p7##x,_p5##y,z,v)), \
 (I[48] = (img)(_p7##x,_p4##y,z,v)), \
 (I[64] = (img)(_p7##x,_p3##y,z,v)), \
 (I[80] = (img)(_p7##x,_p2##y,z,v)), \
 (I[96] = (img)(_p7##x,_p1##y,z,v)), \
 (I[112] = (img)(_p7##x,y,z,v)), \
 (I[128] = (img)(_p7##x,_n1##y,z,v)), \
 (I[144] = (img)(_p7##x,_n2##y,z,v)), \
 (I[160] = (img)(_p7##x,_n3##y,z,v)), \
 (I[176] = (img)(_p7##x,_n4##y,z,v)), \
 (I[192] = (img)(_p7##x,_n5##y,z,v)), \
 (I[208] = (img)(_p7##x,_n6##y,z,v)), \
 (I[224] = (img)(_p7##x,_n7##y,z,v)), \
 (I[240] = (img)(_p7##x,_n8##y,z,v)), \
 (I[1] = (img)(_p6##x,_p7##y,z,v)), \
 (I[17] = (img)(_p6##x,_p6##y,z,v)), \
 (I[33] = (img)(_p6##x,_p5##y,z,v)), \
 (I[49] = (img)(_p6##x,_p4##y,z,v)), \
 (I[65] = (img)(_p6##x,_p3##y,z,v)), \
 (I[81] = (img)(_p6##x,_p2##y,z,v)), \
 (I[97] = (img)(_p6##x,_p1##y,z,v)), \
 (I[113] = (img)(_p6##x,y,z,v)), \
 (I[129] = (img)(_p6##x,_n1##y,z,v)), \
 (I[145] = (img)(_p6##x,_n2##y,z,v)), \
 (I[161] = (img)(_p6##x,_n3##y,z,v)), \
 (I[177] = (img)(_p6##x,_n4##y,z,v)), \
 (I[193] = (img)(_p6##x,_n5##y,z,v)), \
 (I[209] = (img)(_p6##x,_n6##y,z,v)), \
 (I[225] = (img)(_p6##x,_n7##y,z,v)), \
 (I[241] = (img)(_p6##x,_n8##y,z,v)), \
 (I[2] = (img)(_p5##x,_p7##y,z,v)), \
 (I[18] = (img)(_p5##x,_p6##y,z,v)), \
 (I[34] = (img)(_p5##x,_p5##y,z,v)), \
 (I[50] = (img)(_p5##x,_p4##y,z,v)), \
 (I[66] = (img)(_p5##x,_p3##y,z,v)), \
 (I[82] = (img)(_p5##x,_p2##y,z,v)), \
 (I[98] = (img)(_p5##x,_p1##y,z,v)), \
 (I[114] = (img)(_p5##x,y,z,v)), \
 (I[130] = (img)(_p5##x,_n1##y,z,v)), \
 (I[146] = (img)(_p5##x,_n2##y,z,v)), \
 (I[162] = (img)(_p5##x,_n3##y,z,v)), \
 (I[178] = (img)(_p5##x,_n4##y,z,v)), \
 (I[194] = (img)(_p5##x,_n5##y,z,v)), \
 (I[210] = (img)(_p5##x,_n6##y,z,v)), \
 (I[226] = (img)(_p5##x,_n7##y,z,v)), \
 (I[242] = (img)(_p5##x,_n8##y,z,v)), \
 (I[3] = (img)(_p4##x,_p7##y,z,v)), \
 (I[19] = (img)(_p4##x,_p6##y,z,v)), \
 (I[35] = (img)(_p4##x,_p5##y,z,v)), \
 (I[51] = (img)(_p4##x,_p4##y,z,v)), \
 (I[67] = (img)(_p4##x,_p3##y,z,v)), \
 (I[83] = (img)(_p4##x,_p2##y,z,v)), \
 (I[99] = (img)(_p4##x,_p1##y,z,v)), \
 (I[115] = (img)(_p4##x,y,z,v)), \
 (I[131] = (img)(_p4##x,_n1##y,z,v)), \
 (I[147] = (img)(_p4##x,_n2##y,z,v)), \
 (I[163] = (img)(_p4##x,_n3##y,z,v)), \
 (I[179] = (img)(_p4##x,_n4##y,z,v)), \
 (I[195] = (img)(_p4##x,_n5##y,z,v)), \
 (I[211] = (img)(_p4##x,_n6##y,z,v)), \
 (I[227] = (img)(_p4##x,_n7##y,z,v)), \
 (I[243] = (img)(_p4##x,_n8##y,z,v)), \
 (I[4] = (img)(_p3##x,_p7##y,z,v)), \
 (I[20] = (img)(_p3##x,_p6##y,z,v)), \
 (I[36] = (img)(_p3##x,_p5##y,z,v)), \
 (I[52] = (img)(_p3##x,_p4##y,z,v)), \
 (I[68] = (img)(_p3##x,_p3##y,z,v)), \
 (I[84] = (img)(_p3##x,_p2##y,z,v)), \
 (I[100] = (img)(_p3##x,_p1##y,z,v)), \
 (I[116] = (img)(_p3##x,y,z,v)), \
 (I[132] = (img)(_p3##x,_n1##y,z,v)), \
 (I[148] = (img)(_p3##x,_n2##y,z,v)), \
 (I[164] = (img)(_p3##x,_n3##y,z,v)), \
 (I[180] = (img)(_p3##x,_n4##y,z,v)), \
 (I[196] = (img)(_p3##x,_n5##y,z,v)), \
 (I[212] = (img)(_p3##x,_n6##y,z,v)), \
 (I[228] = (img)(_p3##x,_n7##y,z,v)), \
 (I[244] = (img)(_p3##x,_n8##y,z,v)), \
 (I[5] = (img)(_p2##x,_p7##y,z,v)), \
 (I[21] = (img)(_p2##x,_p6##y,z,v)), \
 (I[37] = (img)(_p2##x,_p5##y,z,v)), \
 (I[53] = (img)(_p2##x,_p4##y,z,v)), \
 (I[69] = (img)(_p2##x,_p3##y,z,v)), \
 (I[85] = (img)(_p2##x,_p2##y,z,v)), \
 (I[101] = (img)(_p2##x,_p1##y,z,v)), \
 (I[117] = (img)(_p2##x,y,z,v)), \
 (I[133] = (img)(_p2##x,_n1##y,z,v)), \
 (I[149] = (img)(_p2##x,_n2##y,z,v)), \
 (I[165] = (img)(_p2##x,_n3##y,z,v)), \
 (I[181] = (img)(_p2##x,_n4##y,z,v)), \
 (I[197] = (img)(_p2##x,_n5##y,z,v)), \
 (I[213] = (img)(_p2##x,_n6##y,z,v)), \
 (I[229] = (img)(_p2##x,_n7##y,z,v)), \
 (I[245] = (img)(_p2##x,_n8##y,z,v)), \
 (I[6] = (img)(_p1##x,_p7##y,z,v)), \
 (I[22] = (img)(_p1##x,_p6##y,z,v)), \
 (I[38] = (img)(_p1##x,_p5##y,z,v)), \
 (I[54] = (img)(_p1##x,_p4##y,z,v)), \
 (I[70] = (img)(_p1##x,_p3##y,z,v)), \
 (I[86] = (img)(_p1##x,_p2##y,z,v)), \
 (I[102] = (img)(_p1##x,_p1##y,z,v)), \
 (I[118] = (img)(_p1##x,y,z,v)), \
 (I[134] = (img)(_p1##x,_n1##y,z,v)), \
 (I[150] = (img)(_p1##x,_n2##y,z,v)), \
 (I[166] = (img)(_p1##x,_n3##y,z,v)), \
 (I[182] = (img)(_p1##x,_n4##y,z,v)), \
 (I[198] = (img)(_p1##x,_n5##y,z,v)), \
 (I[214] = (img)(_p1##x,_n6##y,z,v)), \
 (I[230] = (img)(_p1##x,_n7##y,z,v)), \
 (I[246] = (img)(_p1##x,_n8##y,z,v)), \
 (I[7] = (img)(x,_p7##y,z,v)), \
 (I[23] = (img)(x,_p6##y,z,v)), \
 (I[39] = (img)(x,_p5##y,z,v)), \
 (I[55] = (img)(x,_p4##y,z,v)), \
 (I[71] = (img)(x,_p3##y,z,v)), \
 (I[87] = (img)(x,_p2##y,z,v)), \
 (I[103] = (img)(x,_p1##y,z,v)), \
 (I[119] = (img)(x,y,z,v)), \
 (I[135] = (img)(x,_n1##y,z,v)), \
 (I[151] = (img)(x,_n2##y,z,v)), \
 (I[167] = (img)(x,_n3##y,z,v)), \
 (I[183] = (img)(x,_n4##y,z,v)), \
 (I[199] = (img)(x,_n5##y,z,v)), \
 (I[215] = (img)(x,_n6##y,z,v)), \
 (I[231] = (img)(x,_n7##y,z,v)), \
 (I[247] = (img)(x,_n8##y,z,v)), \
 (I[8] = (img)(_n1##x,_p7##y,z,v)), \
 (I[24] = (img)(_n1##x,_p6##y,z,v)), \
 (I[40] = (img)(_n1##x,_p5##y,z,v)), \
 (I[56] = (img)(_n1##x,_p4##y,z,v)), \
 (I[72] = (img)(_n1##x,_p3##y,z,v)), \
 (I[88] = (img)(_n1##x,_p2##y,z,v)), \
 (I[104] = (img)(_n1##x,_p1##y,z,v)), \
 (I[120] = (img)(_n1##x,y,z,v)), \
 (I[136] = (img)(_n1##x,_n1##y,z,v)), \
 (I[152] = (img)(_n1##x,_n2##y,z,v)), \
 (I[168] = (img)(_n1##x,_n3##y,z,v)), \
 (I[184] = (img)(_n1##x,_n4##y,z,v)), \
 (I[200] = (img)(_n1##x,_n5##y,z,v)), \
 (I[216] = (img)(_n1##x,_n6##y,z,v)), \
 (I[232] = (img)(_n1##x,_n7##y,z,v)), \
 (I[248] = (img)(_n1##x,_n8##y,z,v)), \
 (I[9] = (img)(_n2##x,_p7##y,z,v)), \
 (I[25] = (img)(_n2##x,_p6##y,z,v)), \
 (I[41] = (img)(_n2##x,_p5##y,z,v)), \
 (I[57] = (img)(_n2##x,_p4##y,z,v)), \
 (I[73] = (img)(_n2##x,_p3##y,z,v)), \
 (I[89] = (img)(_n2##x,_p2##y,z,v)), \
 (I[105] = (img)(_n2##x,_p1##y,z,v)), \
 (I[121] = (img)(_n2##x,y,z,v)), \
 (I[137] = (img)(_n2##x,_n1##y,z,v)), \
 (I[153] = (img)(_n2##x,_n2##y,z,v)), \
 (I[169] = (img)(_n2##x,_n3##y,z,v)), \
 (I[185] = (img)(_n2##x,_n4##y,z,v)), \
 (I[201] = (img)(_n2##x,_n5##y,z,v)), \
 (I[217] = (img)(_n2##x,_n6##y,z,v)), \
 (I[233] = (img)(_n2##x,_n7##y,z,v)), \
 (I[249] = (img)(_n2##x,_n8##y,z,v)), \
 (I[10] = (img)(_n3##x,_p7##y,z,v)), \
 (I[26] = (img)(_n3##x,_p6##y,z,v)), \
 (I[42] = (img)(_n3##x,_p5##y,z,v)), \
 (I[58] = (img)(_n3##x,_p4##y,z,v)), \
 (I[74] = (img)(_n3##x,_p3##y,z,v)), \
 (I[90] = (img)(_n3##x,_p2##y,z,v)), \
 (I[106] = (img)(_n3##x,_p1##y,z,v)), \
 (I[122] = (img)(_n3##x,y,z,v)), \
 (I[138] = (img)(_n3##x,_n1##y,z,v)), \
 (I[154] = (img)(_n3##x,_n2##y,z,v)), \
 (I[170] = (img)(_n3##x,_n3##y,z,v)), \
 (I[186] = (img)(_n3##x,_n4##y,z,v)), \
 (I[202] = (img)(_n3##x,_n5##y,z,v)), \
 (I[218] = (img)(_n3##x,_n6##y,z,v)), \
 (I[234] = (img)(_n3##x,_n7##y,z,v)), \
 (I[250] = (img)(_n3##x,_n8##y,z,v)), \
 (I[11] = (img)(_n4##x,_p7##y,z,v)), \
 (I[27] = (img)(_n4##x,_p6##y,z,v)), \
 (I[43] = (img)(_n4##x,_p5##y,z,v)), \
 (I[59] = (img)(_n4##x,_p4##y,z,v)), \
 (I[75] = (img)(_n4##x,_p3##y,z,v)), \
 (I[91] = (img)(_n4##x,_p2##y,z,v)), \
 (I[107] = (img)(_n4##x,_p1##y,z,v)), \
 (I[123] = (img)(_n4##x,y,z,v)), \
 (I[139] = (img)(_n4##x,_n1##y,z,v)), \
 (I[155] = (img)(_n4##x,_n2##y,z,v)), \
 (I[171] = (img)(_n4##x,_n3##y,z,v)), \
 (I[187] = (img)(_n4##x,_n4##y,z,v)), \
 (I[203] = (img)(_n4##x,_n5##y,z,v)), \
 (I[219] = (img)(_n4##x,_n6##y,z,v)), \
 (I[235] = (img)(_n4##x,_n7##y,z,v)), \
 (I[251] = (img)(_n4##x,_n8##y,z,v)), \
 (I[12] = (img)(_n5##x,_p7##y,z,v)), \
 (I[28] = (img)(_n5##x,_p6##y,z,v)), \
 (I[44] = (img)(_n5##x,_p5##y,z,v)), \
 (I[60] = (img)(_n5##x,_p4##y,z,v)), \
 (I[76] = (img)(_n5##x,_p3##y,z,v)), \
 (I[92] = (img)(_n5##x,_p2##y,z,v)), \
 (I[108] = (img)(_n5##x,_p1##y,z,v)), \
 (I[124] = (img)(_n5##x,y,z,v)), \
 (I[140] = (img)(_n5##x,_n1##y,z,v)), \
 (I[156] = (img)(_n5##x,_n2##y,z,v)), \
 (I[172] = (img)(_n5##x,_n3##y,z,v)), \
 (I[188] = (img)(_n5##x,_n4##y,z,v)), \
 (I[204] = (img)(_n5##x,_n5##y,z,v)), \
 (I[220] = (img)(_n5##x,_n6##y,z,v)), \
 (I[236] = (img)(_n5##x,_n7##y,z,v)), \
 (I[252] = (img)(_n5##x,_n8##y,z,v)), \
 (I[13] = (img)(_n6##x,_p7##y,z,v)), \
 (I[29] = (img)(_n6##x,_p6##y,z,v)), \
 (I[45] = (img)(_n6##x,_p5##y,z,v)), \
 (I[61] = (img)(_n6##x,_p4##y,z,v)), \
 (I[77] = (img)(_n6##x,_p3##y,z,v)), \
 (I[93] = (img)(_n6##x,_p2##y,z,v)), \
 (I[109] = (img)(_n6##x,_p1##y,z,v)), \
 (I[125] = (img)(_n6##x,y,z,v)), \
 (I[141] = (img)(_n6##x,_n1##y,z,v)), \
 (I[157] = (img)(_n6##x,_n2##y,z,v)), \
 (I[173] = (img)(_n6##x,_n3##y,z,v)), \
 (I[189] = (img)(_n6##x,_n4##y,z,v)), \
 (I[205] = (img)(_n6##x,_n5##y,z,v)), \
 (I[221] = (img)(_n6##x,_n6##y,z,v)), \
 (I[237] = (img)(_n6##x,_n7##y,z,v)), \
 (I[253] = (img)(_n6##x,_n8##y,z,v)), \
 (I[14] = (img)(_n7##x,_p7##y,z,v)), \
 (I[30] = (img)(_n7##x,_p6##y,z,v)), \
 (I[46] = (img)(_n7##x,_p5##y,z,v)), \
 (I[62] = (img)(_n7##x,_p4##y,z,v)), \
 (I[78] = (img)(_n7##x,_p3##y,z,v)), \
 (I[94] = (img)(_n7##x,_p2##y,z,v)), \
 (I[110] = (img)(_n7##x,_p1##y,z,v)), \
 (I[126] = (img)(_n7##x,y,z,v)), \
 (I[142] = (img)(_n7##x,_n1##y,z,v)), \
 (I[158] = (img)(_n7##x,_n2##y,z,v)), \
 (I[174] = (img)(_n7##x,_n3##y,z,v)), \
 (I[190] = (img)(_n7##x,_n4##y,z,v)), \
 (I[206] = (img)(_n7##x,_n5##y,z,v)), \
 (I[222] = (img)(_n7##x,_n6##y,z,v)), \
 (I[238] = (img)(_n7##x,_n7##y,z,v)), \
 (I[254] = (img)(_n7##x,_n8##y,z,v)), \
 x+8>=(int)((img).width)?(int)((img).width)-1:x+8); \
 x<=(int)(x1) && ((_n8##x<(int)((img).width) && ( \
 (I[15] = (img)(_n8##x,_p7##y,z,v)), \
 (I[31] = (img)(_n8##x,_p6##y,z,v)), \
 (I[47] = (img)(_n8##x,_p5##y,z,v)), \
 (I[63] = (img)(_n8##x,_p4##y,z,v)), \
 (I[79] = (img)(_n8##x,_p3##y,z,v)), \
 (I[95] = (img)(_n8##x,_p2##y,z,v)), \
 (I[111] = (img)(_n8##x,_p1##y,z,v)), \
 (I[127] = (img)(_n8##x,y,z,v)), \
 (I[143] = (img)(_n8##x,_n1##y,z,v)), \
 (I[159] = (img)(_n8##x,_n2##y,z,v)), \
 (I[175] = (img)(_n8##x,_n3##y,z,v)), \
 (I[191] = (img)(_n8##x,_n4##y,z,v)), \
 (I[207] = (img)(_n8##x,_n5##y,z,v)), \
 (I[223] = (img)(_n8##x,_n6##y,z,v)), \
 (I[239] = (img)(_n8##x,_n7##y,z,v)), \
 (I[255] = (img)(_n8##x,_n8##y,z,v)),1)) || \
 _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], \
 I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], \
 I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], \
 I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], \
 I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], \
 I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], \
 I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], \
 I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], \
 I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], \
 I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], \
 I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], \
 I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], \
 I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], \
 I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], \
 _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x)

#define cimg_get16x16(img,x,y,z,v,I) \
 I[0] = (img)(_p7##x,_p7##y,z,v), I[1] = (img)(_p6##x,_p7##y,z,v), I[2] = (img)(_p5##x,_p7##y,z,v), I[3] = (img)(_p4##x,_p7##y,z,v), I[4] = (img)(_p3##x,_p7##y,z,v), I[5] = (img)(_p2##x,_p7##y,z,v), I[6] = (img)(_p1##x,_p7##y,z,v), I[7] = (img)(x,_p7##y,z,v), I[8] = (img)(_n1##x,_p7##y,z,v), I[9] = (img)(_n2##x,_p7##y,z,v), I[10] = (img)(_n3##x,_p7##y,z,v), I[11] = (img)(_n4##x,_p7##y,z,v), I[12] = (img)(_n5##x,_p7##y,z,v), I[13] = (img)(_n6##x,_p7##y,z,v), I[14] = (img)(_n7##x,_p7##y,z,v), I[15] = (img)(_n8##x,_p7##y,z,v), \
 I[16] = (img)(_p7##x,_p6##y,z,v), I[17] = (img)(_p6##x,_p6##y,z,v), I[18] = (img)(_p5##x,_p6##y,z,v), I[19] = (img)(_p4##x,_p6##y,z,v), I[20] = (img)(_p3##x,_p6##y,z,v), I[21] = (img)(_p2##x,_p6##y,z,v), I[22] = (img)(_p1##x,_p6##y,z,v), I[23] = (img)(x,_p6##y,z,v), I[24] = (img)(_n1##x,_p6##y,z,v), I[25] = (img)(_n2##x,_p6##y,z,v), I[26] = (img)(_n3##x,_p6##y,z,v), I[27] = (img)(_n4##x,_p6##y,z,v), I[28] = (img)(_n5##x,_p6##y,z,v), I[29] = (img)(_n6##x,_p6##y,z,v), I[30] = (img)(_n7##x,_p6##y,z,v), I[31] = (img)(_n8##x,_p6##y,z,v), \
 I[32] = (img)(_p7##x,_p5##y,z,v), I[33] = (img)(_p6##x,_p5##y,z,v), I[34] = (img)(_p5##x,_p5##y,z,v), I[35] = (img)(_p4##x,_p5##y,z,v), I[36] = (img)(_p3##x,_p5##y,z,v), I[37] = (img)(_p2##x,_p5##y,z,v), I[38] = (img)(_p1##x,_p5##y,z,v), I[39] = (img)(x,_p5##y,z,v), I[40] = (img)(_n1##x,_p5##y,z,v), I[41] = (img)(_n2##x,_p5##y,z,v), I[42] = (img)(_n3##x,_p5##y,z,v), I[43] = (img)(_n4##x,_p5##y,z,v), I[44] = (img)(_n5##x,_p5##y,z,v), I[45] = (img)(_n6##x,_p5##y,z,v), I[46] = (img)(_n7##x,_p5##y,z,v), I[47] = (img)(_n8##x,_p5##y,z,v), \
 I[48] = (img)(_p7##x,_p4##y,z,v), I[49] = (img)(_p6##x,_p4##y,z,v), I[50] = (img)(_p5##x,_p4##y,z,v), I[51] = (img)(_p4##x,_p4##y,z,v), I[52] = (img)(_p3##x,_p4##y,z,v), I[53] = (img)(_p2##x,_p4##y,z,v), I[54] = (img)(_p1##x,_p4##y,z,v), I[55] = (img)(x,_p4##y,z,v), I[56] = (img)(_n1##x,_p4##y,z,v), I[57] = (img)(_n2##x,_p4##y,z,v), I[58] = (img)(_n3##x,_p4##y,z,v), I[59] = (img)(_n4##x,_p4##y,z,v), I[60] = (img)(_n5##x,_p4##y,z,v), I[61] = (img)(_n6##x,_p4##y,z,v), I[62] = (img)(_n7##x,_p4##y,z,v), I[63] = (img)(_n8##x,_p4##y,z,v), \
 I[64] = (img)(_p7##x,_p3##y,z,v), I[65] = (img)(_p6##x,_p3##y,z,v), I[66] = (img)(_p5##x,_p3##y,z,v), I[67] = (img)(_p4##x,_p3##y,z,v), I[68] = (img)(_p3##x,_p3##y,z,v), I[69] = (img)(_p2##x,_p3##y,z,v), I[70] = (img)(_p1##x,_p3##y,z,v), I[71] = (img)(x,_p3##y,z,v), I[72] = (img)(_n1##x,_p3##y,z,v), I[73] = (img)(_n2##x,_p3##y,z,v), I[74] = (img)(_n3##x,_p3##y,z,v), I[75] = (img)(_n4##x,_p3##y,z,v), I[76] = (img)(_n5##x,_p3##y,z,v), I[77] = (img)(_n6##x,_p3##y,z,v), I[78] = (img)(_n7##x,_p3##y,z,v), I[79] = (img)(_n8##x,_p3##y,z,v), \
 I[80] = (img)(_p7##x,_p2##y,z,v), I[81] = (img)(_p6##x,_p2##y,z,v), I[82] = (img)(_p5##x,_p2##y,z,v), I[83] = (img)(_p4##x,_p2##y,z,v), I[84] = (img)(_p3##x,_p2##y,z,v), I[85] = (img)(_p2##x,_p2##y,z,v), I[86] = (img)(_p1##x,_p2##y,z,v), I[87] = (img)(x,_p2##y,z,v), I[88] = (img)(_n1##x,_p2##y,z,v), I[89] = (img)(_n2##x,_p2##y,z,v), I[90] = (img)(_n3##x,_p2##y,z,v), I[91] = (img)(_n4##x,_p2##y,z,v), I[92] = (img)(_n5##x,_p2##y,z,v), I[93] = (img)(_n6##x,_p2##y,z,v), I[94] = (img)(_n7##x,_p2##y,z,v), I[95] = (img)(_n8##x,_p2##y,z,v), \
 I[96] = (img)(_p7##x,_p1##y,z,v), I[97] = (img)(_p6##x,_p1##y,z,v), I[98] = (img)(_p5##x,_p1##y,z,v), I[99] = (img)(_p4##x,_p1##y,z,v), I[100] = (img)(_p3##x,_p1##y,z,v), I[101] = (img)(_p2##x,_p1##y,z,v), I[102] = (img)(_p1##x,_p1##y,z,v), I[103] = (img)(x,_p1##y,z,v), I[104] = (img)(_n1##x,_p1##y,z,v), I[105] = (img)(_n2##x,_p1##y,z,v), I[106] = (img)(_n3##x,_p1##y,z,v), I[107] = (img)(_n4##x,_p1##y,z,v), I[108] = (img)(_n5##x,_p1##y,z,v), I[109] = (img)(_n6##x,_p1##y,z,v), I[110] = (img)(_n7##x,_p1##y,z,v), I[111] = (img)(_n8##x,_p1##y,z,v), \
 I[112] = (img)(_p7##x,y,z,v), I[113] = (img)(_p6##x,y,z,v), I[114] = (img)(_p5##x,y,z,v), I[115] = (img)(_p4##x,y,z,v), I[116] = (img)(_p3##x,y,z,v), I[117] = (img)(_p2##x,y,z,v), I[118] = (img)(_p1##x,y,z,v), I[119] = (img)(x,y,z,v), I[120] = (img)(_n1##x,y,z,v), I[121] = (img)(_n2##x,y,z,v), I[122] = (img)(_n3##x,y,z,v), I[123] = (img)(_n4##x,y,z,v), I[124] = (img)(_n5##x,y,z,v), I[125] = (img)(_n6##x,y,z,v), I[126] = (img)(_n7##x,y,z,v), I[127] = (img)(_n8##x,y,z,v), \
 I[128] = (img)(_p7##x,_n1##y,z,v), I[129] = (img)(_p6##x,_n1##y,z,v), I[130] = (img)(_p5##x,_n1##y,z,v), I[131] = (img)(_p4##x,_n1##y,z,v), I[132] = (img)(_p3##x,_n1##y,z,v), I[133] = (img)(_p2##x,_n1##y,z,v), I[134] = (img)(_p1##x,_n1##y,z,v), I[135] = (img)(x,_n1##y,z,v), I[136] = (img)(_n1##x,_n1##y,z,v), I[137] = (img)(_n2##x,_n1##y,z,v), I[138] = (img)(_n3##x,_n1##y,z,v), I[139] = (img)(_n4##x,_n1##y,z,v), I[140] = (img)(_n5##x,_n1##y,z,v), I[141] = (img)(_n6##x,_n1##y,z,v), I[142] = (img)(_n7##x,_n1##y,z,v), I[143] = (img)(_n8##x,_n1##y,z,v), \
 I[144] = (img)(_p7##x,_n2##y,z,v), I[145] = (img)(_p6##x,_n2##y,z,v), I[146] = (img)(_p5##x,_n2##y,z,v), I[147] = (img)(_p4##x,_n2##y,z,v), I[148] = (img)(_p3##x,_n2##y,z,v), I[149] = (img)(_p2##x,_n2##y,z,v), I[150] = (img)(_p1##x,_n2##y,z,v), I[151] = (img)(x,_n2##y,z,v), I[152] = (img)(_n1##x,_n2##y,z,v), I[153] = (img)(_n2##x,_n2##y,z,v), I[154] = (img)(_n3##x,_n2##y,z,v), I[155] = (img)(_n4##x,_n2##y,z,v), I[156] = (img)(_n5##x,_n2##y,z,v), I[157] = (img)(_n6##x,_n2##y,z,v), I[158] = (img)(_n7##x,_n2##y,z,v), I[159] = (img)(_n8##x,_n2##y,z,v), \
 I[160] = (img)(_p7##x,_n3##y,z,v), I[161] = (img)(_p6##x,_n3##y,z,v), I[162] = (img)(_p5##x,_n3##y,z,v), I[163] = (img)(_p4##x,_n3##y,z,v), I[164] = (img)(_p3##x,_n3##y,z,v), I[165] = (img)(_p2##x,_n3##y,z,v), I[166] = (img)(_p1##x,_n3##y,z,v), I[167] = (img)(x,_n3##y,z,v), I[168] = (img)(_n1##x,_n3##y,z,v), I[169] = (img)(_n2##x,_n3##y,z,v), I[170] = (img)(_n3##x,_n3##y,z,v), I[171] = (img)(_n4##x,_n3##y,z,v), I[172] = (img)(_n5##x,_n3##y,z,v), I[173] = (img)(_n6##x,_n3##y,z,v), I[174] = (img)(_n7##x,_n3##y,z,v), I[175] = (img)(_n8##x,_n3##y,z,v), \
 I[176] = (img)(_p7##x,_n4##y,z,v), I[177] = (img)(_p6##x,_n4##y,z,v), I[178] = (img)(_p5##x,_n4##y,z,v), I[179] = (img)(_p4##x,_n4##y,z,v), I[180] = (img)(_p3##x,_n4##y,z,v), I[181] = (img)(_p2##x,_n4##y,z,v), I[182] = (img)(_p1##x,_n4##y,z,v), I[183] = (img)(x,_n4##y,z,v), I[184] = (img)(_n1##x,_n4##y,z,v), I[185] = (img)(_n2##x,_n4##y,z,v), I[186] = (img)(_n3##x,_n4##y,z,v), I[187] = (img)(_n4##x,_n4##y,z,v), I[188] = (img)(_n5##x,_n4##y,z,v), I[189] = (img)(_n6##x,_n4##y,z,v), I[190] = (img)(_n7##x,_n4##y,z,v), I[191] = (img)(_n8##x,_n4##y,z,v), \
 I[192] = (img)(_p7##x,_n5##y,z,v), I[193] = (img)(_p6##x,_n5##y,z,v), I[194] = (img)(_p5##x,_n5##y,z,v), I[195] = (img)(_p4##x,_n5##y,z,v), I[196] = (img)(_p3##x,_n5##y,z,v), I[197] = (img)(_p2##x,_n5##y,z,v), I[198] = (img)(_p1##x,_n5##y,z,v), I[199] = (img)(x,_n5##y,z,v), I[200] = (img)(_n1##x,_n5##y,z,v), I[201] = (img)(_n2##x,_n5##y,z,v), I[202] = (img)(_n3##x,_n5##y,z,v), I[203] = (img)(_n4##x,_n5##y,z,v), I[204] = (img)(_n5##x,_n5##y,z,v), I[205] = (img)(_n6##x,_n5##y,z,v), I[206] = (img)(_n7##x,_n5##y,z,v), I[207] = (img)(_n8##x,_n5##y,z,v), \
 I[208] = (img)(_p7##x,_n6##y,z,v), I[209] = (img)(_p6##x,_n6##y,z,v), I[210] = (img)(_p5##x,_n6##y,z,v), I[211] = (img)(_p4##x,_n6##y,z,v), I[212] = (img)(_p3##x,_n6##y,z,v), I[213] = (img)(_p2##x,_n6##y,z,v), I[214] = (img)(_p1##x,_n6##y,z,v), I[215] = (img)(x,_n6##y,z,v), I[216] = (img)(_n1##x,_n6##y,z,v), I[217] = (img)(_n2##x,_n6##y,z,v), I[218] = (img)(_n3##x,_n6##y,z,v), I[219] = (img)(_n4##x,_n6##y,z,v), I[220] = (img)(_n5##x,_n6##y,z,v), I[221] = (img)(_n6##x,_n6##y,z,v), I[222] = (img)(_n7##x,_n6##y,z,v), I[223] = (img)(_n8##x,_n6##y,z,v), \
 I[224] = (img)(_p7##x,_n7##y,z,v), I[225] = (img)(_p6##x,_n7##y,z,v), I[226] = (img)(_p5##x,_n7##y,z,v), I[227] = (img)(_p4##x,_n7##y,z,v), I[228] = (img)(_p3##x,_n7##y,z,v), I[229] = (img)(_p2##x,_n7##y,z,v), I[230] = (img)(_p1##x,_n7##y,z,v), I[231] = (img)(x,_n7##y,z,v), I[232] = (img)(_n1##x,_n7##y,z,v), I[233] = (img)(_n2##x,_n7##y,z,v), I[234] = (img)(_n3##x,_n7##y,z,v), I[235] = (img)(_n4##x,_n7##y,z,v), I[236] = (img)(_n5##x,_n7##y,z,v), I[237] = (img)(_n6##x,_n7##y,z,v), I[238] = (img)(_n7##x,_n7##y,z,v), I[239] = (img)(_n8##x,_n7##y,z,v), \
 I[240] = (img)(_p7##x,_n8##y,z,v), I[241] = (img)(_p6##x,_n8##y,z,v), I[242] = (img)(_p5##x,_n8##y,z,v), I[243] = (img)(_p4##x,_n8##y,z,v), I[244] = (img)(_p3##x,_n8##y,z,v), I[245] = (img)(_p2##x,_n8##y,z,v), I[246] = (img)(_p1##x,_n8##y,z,v), I[247] = (img)(x,_n8##y,z,v), I[248] = (img)(_n1##x,_n8##y,z,v), I[249] = (img)(_n2##x,_n8##y,z,v), I[250] = (img)(_n3##x,_n8##y,z,v), I[251] = (img)(_n4##x,_n8##y,z,v), I[252] = (img)(_n5##x,_n8##y,z,v), I[253] = (img)(_n6##x,_n8##y,z,v), I[254] = (img)(_n7##x,_n8##y,z,v), I[255] = (img)(_n8##x,_n8##y,z,v);

// Define 17x17 loop macros for CImg
//----------------------------------
#define cimg_for17(bound,i) for (int i = 0, \
 _p8##i = 0, _p7##i = 0, _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7, \
 _n8##i = 8>=(int)(bound)?(int)(bound)-1:8; \
 _n8##i<(int)(bound) || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i)

#define cimg_for17X(img,x) cimg_for17((img).width,x)
#define cimg_for17Y(img,y) cimg_for17((img).height,y)
#define cimg_for17Z(img,z) cimg_for17((img).depth,z)
#define cimg_for17V(img,v) cimg_for17((img).dim,v)
#define cimg_for17XY(img,x,y) cimg_for17Y(img,y) cimg_for17X(img,x)
#define cimg_for17XZ(img,x,z) cimg_for17Z(img,z) cimg_for17X(img,x)
#define cimg_for17XV(img,x,v) cimg_for17V(img,v) cimg_for17X(img,x)
#define cimg_for17YZ(img,y,z) cimg_for17Z(img,z) cimg_for17Y(img,y)
#define cimg_for17YV(img,y,v) cimg_for17V(img,v) cimg_for17Y(img,y)
#define cimg_for17ZV(img,z,v) cimg_for17V(img,v) cimg_for17Z(img,z)
#define cimg_for17XYZ(img,x,y,z) cimg_for17Z(img,z) cimg_for17XY(img,x,y)
#define cimg_for17XZV(img,x,z,v) cimg_for17V(img,v) cimg_for17XZ(img,x,z)
#define cimg_for17YZV(img,y,z,v) cimg_for17V(img,v) cimg_for17YZ(img,y,z)
#define cimg_for17XYZV(img,x,y,z,v) cimg_for17V(img,v) cimg_for17XYZ(img,x,y,z)

#define cimg_for_in17(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p8##i = i-8<0?0:i-8, \
 _p7##i = i-7<0?0:i-7, \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7, \
 _n8##i = i+8>=(int)(bound)?(int)(bound)-1:i+8; \
 i<=(int)(i1) && (_n8##i<(int)(bound) || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i)

#define cimg_for_in17X(img,x0,x1,x) cimg_for_in17((img).width,x0,x1,x)
#define cimg_for_in17Y(img,y0,y1,y) cimg_for_in17((img).height,y0,y1,y)
#define cimg_for_in17Z(img,z0,z1,z) cimg_for_in17((img).depth,z0,z1,z)
#define cimg_for_in17V(img,v0,v1,v) cimg_for_in17((img).dim,v0,v1,v)
#define cimg_for_in17XY(img,x0,y0,x1,y1,x,y) cimg_for_in17Y(img,y0,y1,y) cimg_for_in17X(img,x0,x1,x)
#define cimg_for_in17XZ(img,x0,z0,x1,z1,x,z) cimg_for_in17Z(img,z0,z1,z) cimg_for_in17X(img,x0,x1,x)
#define cimg_for_in17XV(img,x0,v0,x1,v1,x,v) cimg_for_in17V(img,v0,v1,v) cimg_for_in17X(img,x0,x1,x)
#define cimg_for_in17YZ(img,y0,z0,y1,z1,y,z) cimg_for_in17Z(img,z0,z1,z) cimg_for_in17Y(img,y0,y1,y)
#define cimg_for_in17YV(img,y0,v0,y1,v1,y,v) cimg_for_in17V(img,v0,v1,v) cimg_for_in17Y(img,y0,y1,y)
#define cimg_for_in17ZV(img,z0,v0,z1,v1,z,v) cimg_for_in17V(img,v0,v1,v) cimg_for_in17Z(img,z0,z1,z)
#define cimg_for_in17XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in17Z(img,z0,z1,z) cimg_for_in17XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in17XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in17V(img,v0,v1,v) cimg_for_in17XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in17YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in17V(img,v0,v1,v) cimg_for_in17YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in17XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in17V(img,v0,v1,v) cimg_for_in17XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for17x17(img,x,y,z,v,I) \
 cimg_for17((img).height,y) for (int x = 0, \
 _p8##x = 0, _p7##x = 0, _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = 7>=((img).width)?(int)((img).width)-1:7, \
 _n8##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = I[7] = I[8] = (img)(0,_p8##y,z,v)), \
 (I[17] = I[18] = I[19] = I[20] = I[21] = I[22] = I[23] = I[24] = I[25] = (img)(0,_p7##y,z,v)), \
 (I[34] = I[35] = I[36] = I[37] = I[38] = I[39] = I[40] = I[41] = I[42] = (img)(0,_p6##y,z,v)), \
 (I[51] = I[52] = I[53] = I[54] = I[55] = I[56] = I[57] = I[58] = I[59] = (img)(0,_p5##y,z,v)), \
 (I[68] = I[69] = I[70] = I[71] = I[72] = I[73] = I[74] = I[75] = I[76] = (img)(0,_p4##y,z,v)), \
 (I[85] = I[86] = I[87] = I[88] = I[89] = I[90] = I[91] = I[92] = I[93] = (img)(0,_p3##y,z,v)), \
 (I[102] = I[103] = I[104] = I[105] = I[106] = I[107] = I[108] = I[109] = I[110] = (img)(0,_p2##y,z,v)), \
 (I[119] = I[120] = I[121] = I[122] = I[123] = I[124] = I[125] = I[126] = I[127] = (img)(0,_p1##y,z,v)), \
 (I[136] = I[137] = I[138] = I[139] = I[140] = I[141] = I[142] = I[143] = I[144] = (img)(0,y,z,v)), \
 (I[153] = I[154] = I[155] = I[156] = I[157] = I[158] = I[159] = I[160] = I[161] = (img)(0,_n1##y,z,v)), \
 (I[170] = I[171] = I[172] = I[173] = I[174] = I[175] = I[176] = I[177] = I[178] = (img)(0,_n2##y,z,v)), \
 (I[187] = I[188] = I[189] = I[190] = I[191] = I[192] = I[193] = I[194] = I[195] = (img)(0,_n3##y,z,v)), \
 (I[204] = I[205] = I[206] = I[207] = I[208] = I[209] = I[210] = I[211] = I[212] = (img)(0,_n4##y,z,v)), \
 (I[221] = I[222] = I[223] = I[224] = I[225] = I[226] = I[227] = I[228] = I[229] = (img)(0,_n5##y,z,v)), \
 (I[238] = I[239] = I[240] = I[241] = I[242] = I[243] = I[244] = I[245] = I[246] = (img)(0,_n6##y,z,v)), \
 (I[255] = I[256] = I[257] = I[258] = I[259] = I[260] = I[261] = I[262] = I[263] = (img)(0,_n7##y,z,v)), \
 (I[272] = I[273] = I[274] = I[275] = I[276] = I[277] = I[278] = I[279] = I[280] = (img)(0,_n8##y,z,v)), \
 (I[9] = (img)(_n1##x,_p8##y,z,v)), \
 (I[26] = (img)(_n1##x,_p7##y,z,v)), \
 (I[43] = (img)(_n1##x,_p6##y,z,v)), \
 (I[60] = (img)(_n1##x,_p5##y,z,v)), \
 (I[77] = (img)(_n1##x,_p4##y,z,v)), \
 (I[94] = (img)(_n1##x,_p3##y,z,v)), \
 (I[111] = (img)(_n1##x,_p2##y,z,v)), \
 (I[128] = (img)(_n1##x,_p1##y,z,v)), \
 (I[145] = (img)(_n1##x,y,z,v)), \
 (I[162] = (img)(_n1##x,_n1##y,z,v)), \
 (I[179] = (img)(_n1##x,_n2##y,z,v)), \
 (I[196] = (img)(_n1##x,_n3##y,z,v)), \
 (I[213] = (img)(_n1##x,_n4##y,z,v)), \
 (I[230] = (img)(_n1##x,_n5##y,z,v)), \
 (I[247] = (img)(_n1##x,_n6##y,z,v)), \
 (I[264] = (img)(_n1##x,_n7##y,z,v)), \
 (I[281] = (img)(_n1##x,_n8##y,z,v)), \
 (I[10] = (img)(_n2##x,_p8##y,z,v)), \
 (I[27] = (img)(_n2##x,_p7##y,z,v)), \
 (I[44] = (img)(_n2##x,_p6##y,z,v)), \
 (I[61] = (img)(_n2##x,_p5##y,z,v)), \
 (I[78] = (img)(_n2##x,_p4##y,z,v)), \
 (I[95] = (img)(_n2##x,_p3##y,z,v)), \
 (I[112] = (img)(_n2##x,_p2##y,z,v)), \
 (I[129] = (img)(_n2##x,_p1##y,z,v)), \
 (I[146] = (img)(_n2##x,y,z,v)), \
 (I[163] = (img)(_n2##x,_n1##y,z,v)), \
 (I[180] = (img)(_n2##x,_n2##y,z,v)), \
 (I[197] = (img)(_n2##x,_n3##y,z,v)), \
 (I[214] = (img)(_n2##x,_n4##y,z,v)), \
 (I[231] = (img)(_n2##x,_n5##y,z,v)), \
 (I[248] = (img)(_n2##x,_n6##y,z,v)), \
 (I[265] = (img)(_n2##x,_n7##y,z,v)), \
 (I[282] = (img)(_n2##x,_n8##y,z,v)), \
 (I[11] = (img)(_n3##x,_p8##y,z,v)), \
 (I[28] = (img)(_n3##x,_p7##y,z,v)), \
 (I[45] = (img)(_n3##x,_p6##y,z,v)), \
 (I[62] = (img)(_n3##x,_p5##y,z,v)), \
 (I[79] = (img)(_n3##x,_p4##y,z,v)), \
 (I[96] = (img)(_n3##x,_p3##y,z,v)), \
 (I[113] = (img)(_n3##x,_p2##y,z,v)), \
 (I[130] = (img)(_n3##x,_p1##y,z,v)), \
 (I[147] = (img)(_n3##x,y,z,v)), \
 (I[164] = (img)(_n3##x,_n1##y,z,v)), \
 (I[181] = (img)(_n3##x,_n2##y,z,v)), \
 (I[198] = (img)(_n3##x,_n3##y,z,v)), \
 (I[215] = (img)(_n3##x,_n4##y,z,v)), \
 (I[232] = (img)(_n3##x,_n5##y,z,v)), \
 (I[249] = (img)(_n3##x,_n6##y,z,v)), \
 (I[266] = (img)(_n3##x,_n7##y,z,v)), \
 (I[283] = (img)(_n3##x,_n8##y,z,v)), \
 (I[12] = (img)(_n4##x,_p8##y,z,v)), \
 (I[29] = (img)(_n4##x,_p7##y,z,v)), \
 (I[46] = (img)(_n4##x,_p6##y,z,v)), \
 (I[63] = (img)(_n4##x,_p5##y,z,v)), \
 (I[80] = (img)(_n4##x,_p4##y,z,v)), \
 (I[97] = (img)(_n4##x,_p3##y,z,v)), \
 (I[114] = (img)(_n4##x,_p2##y,z,v)), \
 (I[131] = (img)(_n4##x,_p1##y,z,v)), \
 (I[148] = (img)(_n4##x,y,z,v)), \
 (I[165] = (img)(_n4##x,_n1##y,z,v)), \
 (I[182] = (img)(_n4##x,_n2##y,z,v)), \
 (I[199] = (img)(_n4##x,_n3##y,z,v)), \
 (I[216] = (img)(_n4##x,_n4##y,z,v)), \
 (I[233] = (img)(_n4##x,_n5##y,z,v)), \
 (I[250] = (img)(_n4##x,_n6##y,z,v)), \
 (I[267] = (img)(_n4##x,_n7##y,z,v)), \
 (I[284] = (img)(_n4##x,_n8##y,z,v)), \
 (I[13] = (img)(_n5##x,_p8##y,z,v)), \
 (I[30] = (img)(_n5##x,_p7##y,z,v)), \
 (I[47] = (img)(_n5##x,_p6##y,z,v)), \
 (I[64] = (img)(_n5##x,_p5##y,z,v)), \
 (I[81] = (img)(_n5##x,_p4##y,z,v)), \
 (I[98] = (img)(_n5##x,_p3##y,z,v)), \
 (I[115] = (img)(_n5##x,_p2##y,z,v)), \
 (I[132] = (img)(_n5##x,_p1##y,z,v)), \
 (I[149] = (img)(_n5##x,y,z,v)), \
 (I[166] = (img)(_n5##x,_n1##y,z,v)), \
 (I[183] = (img)(_n5##x,_n2##y,z,v)), \
 (I[200] = (img)(_n5##x,_n3##y,z,v)), \
 (I[217] = (img)(_n5##x,_n4##y,z,v)), \
 (I[234] = (img)(_n5##x,_n5##y,z,v)), \
 (I[251] = (img)(_n5##x,_n6##y,z,v)), \
 (I[268] = (img)(_n5##x,_n7##y,z,v)), \
 (I[285] = (img)(_n5##x,_n8##y,z,v)), \
 (I[14] = (img)(_n6##x,_p8##y,z,v)), \
 (I[31] = (img)(_n6##x,_p7##y,z,v)), \
 (I[48] = (img)(_n6##x,_p6##y,z,v)), \
 (I[65] = (img)(_n6##x,_p5##y,z,v)), \
 (I[82] = (img)(_n6##x,_p4##y,z,v)), \
 (I[99] = (img)(_n6##x,_p3##y,z,v)), \
 (I[116] = (img)(_n6##x,_p2##y,z,v)), \
 (I[133] = (img)(_n6##x,_p1##y,z,v)), \
 (I[150] = (img)(_n6##x,y,z,v)), \
 (I[167] = (img)(_n6##x,_n1##y,z,v)), \
 (I[184] = (img)(_n6##x,_n2##y,z,v)), \
 (I[201] = (img)(_n6##x,_n3##y,z,v)), \
 (I[218] = (img)(_n6##x,_n4##y,z,v)), \
 (I[235] = (img)(_n6##x,_n5##y,z,v)), \
 (I[252] = (img)(_n6##x,_n6##y,z,v)), \
 (I[269] = (img)(_n6##x,_n7##y,z,v)), \
 (I[286] = (img)(_n6##x,_n8##y,z,v)), \
 (I[15] = (img)(_n7##x,_p8##y,z,v)), \
 (I[32] = (img)(_n7##x,_p7##y,z,v)), \
 (I[49] = (img)(_n7##x,_p6##y,z,v)), \
 (I[66] = (img)(_n7##x,_p5##y,z,v)), \
 (I[83] = (img)(_n7##x,_p4##y,z,v)), \
 (I[100] = (img)(_n7##x,_p3##y,z,v)), \
 (I[117] = (img)(_n7##x,_p2##y,z,v)), \
 (I[134] = (img)(_n7##x,_p1##y,z,v)), \
 (I[151] = (img)(_n7##x,y,z,v)), \
 (I[168] = (img)(_n7##x,_n1##y,z,v)), \
 (I[185] = (img)(_n7##x,_n2##y,z,v)), \
 (I[202] = (img)(_n7##x,_n3##y,z,v)), \
 (I[219] = (img)(_n7##x,_n4##y,z,v)), \
 (I[236] = (img)(_n7##x,_n5##y,z,v)), \
 (I[253] = (img)(_n7##x,_n6##y,z,v)), \
 (I[270] = (img)(_n7##x,_n7##y,z,v)), \
 (I[287] = (img)(_n7##x,_n8##y,z,v)), \
 8>=((img).width)?(int)((img).width)-1:8); \
 (_n8##x<(int)((img).width) && ( \
 (I[16] = (img)(_n8##x,_p8##y,z,v)), \
 (I[33] = (img)(_n8##x,_p7##y,z,v)), \
 (I[50] = (img)(_n8##x,_p6##y,z,v)), \
 (I[67] = (img)(_n8##x,_p5##y,z,v)), \
 (I[84] = (img)(_n8##x,_p4##y,z,v)), \
 (I[101] = (img)(_n8##x,_p3##y,z,v)), \
 (I[118] = (img)(_n8##x,_p2##y,z,v)), \
 (I[135] = (img)(_n8##x,_p1##y,z,v)), \
 (I[152] = (img)(_n8##x,y,z,v)), \
 (I[169] = (img)(_n8##x,_n1##y,z,v)), \
 (I[186] = (img)(_n8##x,_n2##y,z,v)), \
 (I[203] = (img)(_n8##x,_n3##y,z,v)), \
 (I[220] = (img)(_n8##x,_n4##y,z,v)), \
 (I[237] = (img)(_n8##x,_n5##y,z,v)), \
 (I[254] = (img)(_n8##x,_n6##y,z,v)), \
 (I[271] = (img)(_n8##x,_n7##y,z,v)), \
 (I[288] = (img)(_n8##x,_n8##y,z,v)),1)) || \
 _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], \
 I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], \
 I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], \
 I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], \
 I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], \
 I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], \
 I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], \
 I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], \
 I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], \
 I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], \
 I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], \
 I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], \
 I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], \
 I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], \
 I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], \
 I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], \
 I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], I[287] = I[288], \
 _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x)

#define cimg_for_in17x17(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in17((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p8##x = x-8<0?0:x-8, \
 _p7##x = x-7<0?0:x-7, \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = x+7>=(int)((img).width)?(int)((img).width)-1:x+7, \
 _n8##x = (int)( \
 (I[0] = (img)(_p8##x,_p8##y,z,v)), \
 (I[17] = (img)(_p8##x,_p7##y,z,v)), \
 (I[34] = (img)(_p8##x,_p6##y,z,v)), \
 (I[51] = (img)(_p8##x,_p5##y,z,v)), \
 (I[68] = (img)(_p8##x,_p4##y,z,v)), \
 (I[85] = (img)(_p8##x,_p3##y,z,v)), \
 (I[102] = (img)(_p8##x,_p2##y,z,v)), \
 (I[119] = (img)(_p8##x,_p1##y,z,v)), \
 (I[136] = (img)(_p8##x,y,z,v)), \
 (I[153] = (img)(_p8##x,_n1##y,z,v)), \
 (I[170] = (img)(_p8##x,_n2##y,z,v)), \
 (I[187] = (img)(_p8##x,_n3##y,z,v)), \
 (I[204] = (img)(_p8##x,_n4##y,z,v)), \
 (I[221] = (img)(_p8##x,_n5##y,z,v)), \
 (I[238] = (img)(_p8##x,_n6##y,z,v)), \
 (I[255] = (img)(_p8##x,_n7##y,z,v)), \
 (I[272] = (img)(_p8##x,_n8##y,z,v)), \
 (I[1] = (img)(_p7##x,_p8##y,z,v)), \
 (I[18] = (img)(_p7##x,_p7##y,z,v)), \
 (I[35] = (img)(_p7##x,_p6##y,z,v)), \
 (I[52] = (img)(_p7##x,_p5##y,z,v)), \
 (I[69] = (img)(_p7##x,_p4##y,z,v)), \
 (I[86] = (img)(_p7##x,_p3##y,z,v)), \
 (I[103] = (img)(_p7##x,_p2##y,z,v)), \
 (I[120] = (img)(_p7##x,_p1##y,z,v)), \
 (I[137] = (img)(_p7##x,y,z,v)), \
 (I[154] = (img)(_p7##x,_n1##y,z,v)), \
 (I[171] = (img)(_p7##x,_n2##y,z,v)), \
 (I[188] = (img)(_p7##x,_n3##y,z,v)), \
 (I[205] = (img)(_p7##x,_n4##y,z,v)), \
 (I[222] = (img)(_p7##x,_n5##y,z,v)), \
 (I[239] = (img)(_p7##x,_n6##y,z,v)), \
 (I[256] = (img)(_p7##x,_n7##y,z,v)), \
 (I[273] = (img)(_p7##x,_n8##y,z,v)), \
 (I[2] = (img)(_p6##x,_p8##y,z,v)), \
 (I[19] = (img)(_p6##x,_p7##y,z,v)), \
 (I[36] = (img)(_p6##x,_p6##y,z,v)), \
 (I[53] = (img)(_p6##x,_p5##y,z,v)), \
 (I[70] = (img)(_p6##x,_p4##y,z,v)), \
 (I[87] = (img)(_p6##x,_p3##y,z,v)), \
 (I[104] = (img)(_p6##x,_p2##y,z,v)), \
 (I[121] = (img)(_p6##x,_p1##y,z,v)), \
 (I[138] = (img)(_p6##x,y,z,v)), \
 (I[155] = (img)(_p6##x,_n1##y,z,v)), \
 (I[172] = (img)(_p6##x,_n2##y,z,v)), \
 (I[189] = (img)(_p6##x,_n3##y,z,v)), \
 (I[206] = (img)(_p6##x,_n4##y,z,v)), \
 (I[223] = (img)(_p6##x,_n5##y,z,v)), \
 (I[240] = (img)(_p6##x,_n6##y,z,v)), \
 (I[257] = (img)(_p6##x,_n7##y,z,v)), \
 (I[274] = (img)(_p6##x,_n8##y,z,v)), \
 (I[3] = (img)(_p5##x,_p8##y,z,v)), \
 (I[20] = (img)(_p5##x,_p7##y,z,v)), \
 (I[37] = (img)(_p5##x,_p6##y,z,v)), \
 (I[54] = (img)(_p5##x,_p5##y,z,v)), \
 (I[71] = (img)(_p5##x,_p4##y,z,v)), \
 (I[88] = (img)(_p5##x,_p3##y,z,v)), \
 (I[105] = (img)(_p5##x,_p2##y,z,v)), \
 (I[122] = (img)(_p5##x,_p1##y,z,v)), \
 (I[139] = (img)(_p5##x,y,z,v)), \
 (I[156] = (img)(_p5##x,_n1##y,z,v)), \
 (I[173] = (img)(_p5##x,_n2##y,z,v)), \
 (I[190] = (img)(_p5##x,_n3##y,z,v)), \
 (I[207] = (img)(_p5##x,_n4##y,z,v)), \
 (I[224] = (img)(_p5##x,_n5##y,z,v)), \
 (I[241] = (img)(_p5##x,_n6##y,z,v)), \
 (I[258] = (img)(_p5##x,_n7##y,z,v)), \
 (I[275] = (img)(_p5##x,_n8##y,z,v)), \
 (I[4] = (img)(_p4##x,_p8##y,z,v)), \
 (I[21] = (img)(_p4##x,_p7##y,z,v)), \
 (I[38] = (img)(_p4##x,_p6##y,z,v)), \
 (I[55] = (img)(_p4##x,_p5##y,z,v)), \
 (I[72] = (img)(_p4##x,_p4##y,z,v)), \
 (I[89] = (img)(_p4##x,_p3##y,z,v)), \
 (I[106] = (img)(_p4##x,_p2##y,z,v)), \
 (I[123] = (img)(_p4##x,_p1##y,z,v)), \
 (I[140] = (img)(_p4##x,y,z,v)), \
 (I[157] = (img)(_p4##x,_n1##y,z,v)), \
 (I[174] = (img)(_p4##x,_n2##y,z,v)), \
 (I[191] = (img)(_p4##x,_n3##y,z,v)), \
 (I[208] = (img)(_p4##x,_n4##y,z,v)), \
 (I[225] = (img)(_p4##x,_n5##y,z,v)), \
 (I[242] = (img)(_p4##x,_n6##y,z,v)), \
 (I[259] = (img)(_p4##x,_n7##y,z,v)), \
 (I[276] = (img)(_p4##x,_n8##y,z,v)), \
 (I[5] = (img)(_p3##x,_p8##y,z,v)), \
 (I[22] = (img)(_p3##x,_p7##y,z,v)), \
 (I[39] = (img)(_p3##x,_p6##y,z,v)), \
 (I[56] = (img)(_p3##x,_p5##y,z,v)), \
 (I[73] = (img)(_p3##x,_p4##y,z,v)), \
 (I[90] = (img)(_p3##x,_p3##y,z,v)), \
 (I[107] = (img)(_p3##x,_p2##y,z,v)), \
 (I[124] = (img)(_p3##x,_p1##y,z,v)), \
 (I[141] = (img)(_p3##x,y,z,v)), \
 (I[158] = (img)(_p3##x,_n1##y,z,v)), \
 (I[175] = (img)(_p3##x,_n2##y,z,v)), \
 (I[192] = (img)(_p3##x,_n3##y,z,v)), \
 (I[209] = (img)(_p3##x,_n4##y,z,v)), \
 (I[226] = (img)(_p3##x,_n5##y,z,v)), \
 (I[243] = (img)(_p3##x,_n6##y,z,v)), \
 (I[260] = (img)(_p3##x,_n7##y,z,v)), \
 (I[277] = (img)(_p3##x,_n8##y,z,v)), \
 (I[6] = (img)(_p2##x,_p8##y,z,v)), \
 (I[23] = (img)(_p2##x,_p7##y,z,v)), \
 (I[40] = (img)(_p2##x,_p6##y,z,v)), \
 (I[57] = (img)(_p2##x,_p5##y,z,v)), \
 (I[74] = (img)(_p2##x,_p4##y,z,v)), \
 (I[91] = (img)(_p2##x,_p3##y,z,v)), \
 (I[108] = (img)(_p2##x,_p2##y,z,v)), \
 (I[125] = (img)(_p2##x,_p1##y,z,v)), \
 (I[142] = (img)(_p2##x,y,z,v)), \
 (I[159] = (img)(_p2##x,_n1##y,z,v)), \
 (I[176] = (img)(_p2##x,_n2##y,z,v)), \
 (I[193] = (img)(_p2##x,_n3##y,z,v)), \
 (I[210] = (img)(_p2##x,_n4##y,z,v)), \
 (I[227] = (img)(_p2##x,_n5##y,z,v)), \
 (I[244] = (img)(_p2##x,_n6##y,z,v)), \
 (I[261] = (img)(_p2##x,_n7##y,z,v)), \
 (I[278] = (img)(_p2##x,_n8##y,z,v)), \
 (I[7] = (img)(_p1##x,_p8##y,z,v)), \
 (I[24] = (img)(_p1##x,_p7##y,z,v)), \
 (I[41] = (img)(_p1##x,_p6##y,z,v)), \
 (I[58] = (img)(_p1##x,_p5##y,z,v)), \
 (I[75] = (img)(_p1##x,_p4##y,z,v)), \
 (I[92] = (img)(_p1##x,_p3##y,z,v)), \
 (I[109] = (img)(_p1##x,_p2##y,z,v)), \
 (I[126] = (img)(_p1##x,_p1##y,z,v)), \
 (I[143] = (img)(_p1##x,y,z,v)), \
 (I[160] = (img)(_p1##x,_n1##y,z,v)), \
 (I[177] = (img)(_p1##x,_n2##y,z,v)), \
 (I[194] = (img)(_p1##x,_n3##y,z,v)), \
 (I[211] = (img)(_p1##x,_n4##y,z,v)), \
 (I[228] = (img)(_p1##x,_n5##y,z,v)), \
 (I[245] = (img)(_p1##x,_n6##y,z,v)), \
 (I[262] = (img)(_p1##x,_n7##y,z,v)), \
 (I[279] = (img)(_p1##x,_n8##y,z,v)), \
 (I[8] = (img)(x,_p8##y,z,v)), \
 (I[25] = (img)(x,_p7##y,z,v)), \
 (I[42] = (img)(x,_p6##y,z,v)), \
 (I[59] = (img)(x,_p5##y,z,v)), \
 (I[76] = (img)(x,_p4##y,z,v)), \
 (I[93] = (img)(x,_p3##y,z,v)), \
 (I[110] = (img)(x,_p2##y,z,v)), \
 (I[127] = (img)(x,_p1##y,z,v)), \
 (I[144] = (img)(x,y,z,v)), \
 (I[161] = (img)(x,_n1##y,z,v)), \
 (I[178] = (img)(x,_n2##y,z,v)), \
 (I[195] = (img)(x,_n3##y,z,v)), \
 (I[212] = (img)(x,_n4##y,z,v)), \
 (I[229] = (img)(x,_n5##y,z,v)), \
 (I[246] = (img)(x,_n6##y,z,v)), \
 (I[263] = (img)(x,_n7##y,z,v)), \
 (I[280] = (img)(x,_n8##y,z,v)), \
 (I[9] = (img)(_n1##x,_p8##y,z,v)), \
 (I[26] = (img)(_n1##x,_p7##y,z,v)), \
 (I[43] = (img)(_n1##x,_p6##y,z,v)), \
 (I[60] = (img)(_n1##x,_p5##y,z,v)), \
 (I[77] = (img)(_n1##x,_p4##y,z,v)), \
 (I[94] = (img)(_n1##x,_p3##y,z,v)), \
 (I[111] = (img)(_n1##x,_p2##y,z,v)), \
 (I[128] = (img)(_n1##x,_p1##y,z,v)), \
 (I[145] = (img)(_n1##x,y,z,v)), \
 (I[162] = (img)(_n1##x,_n1##y,z,v)), \
 (I[179] = (img)(_n1##x,_n2##y,z,v)), \
 (I[196] = (img)(_n1##x,_n3##y,z,v)), \
 (I[213] = (img)(_n1##x,_n4##y,z,v)), \
 (I[230] = (img)(_n1##x,_n5##y,z,v)), \
 (I[247] = (img)(_n1##x,_n6##y,z,v)), \
 (I[264] = (img)(_n1##x,_n7##y,z,v)), \
 (I[281] = (img)(_n1##x,_n8##y,z,v)), \
 (I[10] = (img)(_n2##x,_p8##y,z,v)), \
 (I[27] = (img)(_n2##x,_p7##y,z,v)), \
 (I[44] = (img)(_n2##x,_p6##y,z,v)), \
 (I[61] = (img)(_n2##x,_p5##y,z,v)), \
 (I[78] = (img)(_n2##x,_p4##y,z,v)), \
 (I[95] = (img)(_n2##x,_p3##y,z,v)), \
 (I[112] = (img)(_n2##x,_p2##y,z,v)), \
 (I[129] = (img)(_n2##x,_p1##y,z,v)), \
 (I[146] = (img)(_n2##x,y,z,v)), \
 (I[163] = (img)(_n2##x,_n1##y,z,v)), \
 (I[180] = (img)(_n2##x,_n2##y,z,v)), \
 (I[197] = (img)(_n2##x,_n3##y,z,v)), \
 (I[214] = (img)(_n2##x,_n4##y,z,v)), \
 (I[231] = (img)(_n2##x,_n5##y,z,v)), \
 (I[248] = (img)(_n2##x,_n6##y,z,v)), \
 (I[265] = (img)(_n2##x,_n7##y,z,v)), \
 (I[282] = (img)(_n2##x,_n8##y,z,v)), \
 (I[11] = (img)(_n3##x,_p8##y,z,v)), \
 (I[28] = (img)(_n3##x,_p7##y,z,v)), \
 (I[45] = (img)(_n3##x,_p6##y,z,v)), \
 (I[62] = (img)(_n3##x,_p5##y,z,v)), \
 (I[79] = (img)(_n3##x,_p4##y,z,v)), \
 (I[96] = (img)(_n3##x,_p3##y,z,v)), \
 (I[113] = (img)(_n3##x,_p2##y,z,v)), \
 (I[130] = (img)(_n3##x,_p1##y,z,v)), \
 (I[147] = (img)(_n3##x,y,z,v)), \
 (I[164] = (img)(_n3##x,_n1##y,z,v)), \
 (I[181] = (img)(_n3##x,_n2##y,z,v)), \
 (I[198] = (img)(_n3##x,_n3##y,z,v)), \
 (I[215] = (img)(_n3##x,_n4##y,z,v)), \
 (I[232] = (img)(_n3##x,_n5##y,z,v)), \
 (I[249] = (img)(_n3##x,_n6##y,z,v)), \
 (I[266] = (img)(_n3##x,_n7##y,z,v)), \
 (I[283] = (img)(_n3##x,_n8##y,z,v)), \
 (I[12] = (img)(_n4##x,_p8##y,z,v)), \
 (I[29] = (img)(_n4##x,_p7##y,z,v)), \
 (I[46] = (img)(_n4##x,_p6##y,z,v)), \
 (I[63] = (img)(_n4##x,_p5##y,z,v)), \
 (I[80] = (img)(_n4##x,_p4##y,z,v)), \
 (I[97] = (img)(_n4##x,_p3##y,z,v)), \
 (I[114] = (img)(_n4##x,_p2##y,z,v)), \
 (I[131] = (img)(_n4##x,_p1##y,z,v)), \
 (I[148] = (img)(_n4##x,y,z,v)), \
 (I[165] = (img)(_n4##x,_n1##y,z,v)), \
 (I[182] = (img)(_n4##x,_n2##y,z,v)), \
 (I[199] = (img)(_n4##x,_n3##y,z,v)), \
 (I[216] = (img)(_n4##x,_n4##y,z,v)), \
 (I[233] = (img)(_n4##x,_n5##y,z,v)), \
 (I[250] = (img)(_n4##x,_n6##y,z,v)), \
 (I[267] = (img)(_n4##x,_n7##y,z,v)), \
 (I[284] = (img)(_n4##x,_n8##y,z,v)), \
 (I[13] = (img)(_n5##x,_p8##y,z,v)), \
 (I[30] = (img)(_n5##x,_p7##y,z,v)), \
 (I[47] = (img)(_n5##x,_p6##y,z,v)), \
 (I[64] = (img)(_n5##x,_p5##y,z,v)), \
 (I[81] = (img)(_n5##x,_p4##y,z,v)), \
 (I[98] = (img)(_n5##x,_p3##y,z,v)), \
 (I[115] = (img)(_n5##x,_p2##y,z,v)), \
 (I[132] = (img)(_n5##x,_p1##y,z,v)), \
 (I[149] = (img)(_n5##x,y,z,v)), \
 (I[166] = (img)(_n5##x,_n1##y,z,v)), \
 (I[183] = (img)(_n5##x,_n2##y,z,v)), \
 (I[200] = (img)(_n5##x,_n3##y,z,v)), \
 (I[217] = (img)(_n5##x,_n4##y,z,v)), \
 (I[234] = (img)(_n5##x,_n5##y,z,v)), \
 (I[251] = (img)(_n5##x,_n6##y,z,v)), \
 (I[268] = (img)(_n5##x,_n7##y,z,v)), \
 (I[285] = (img)(_n5##x,_n8##y,z,v)), \
 (I[14] = (img)(_n6##x,_p8##y,z,v)), \
 (I[31] = (img)(_n6##x,_p7##y,z,v)), \
 (I[48] = (img)(_n6##x,_p6##y,z,v)), \
 (I[65] = (img)(_n6##x,_p5##y,z,v)), \
 (I[82] = (img)(_n6##x,_p4##y,z,v)), \
 (I[99] = (img)(_n6##x,_p3##y,z,v)), \
 (I[116] = (img)(_n6##x,_p2##y,z,v)), \
 (I[133] = (img)(_n6##x,_p1##y,z,v)), \
 (I[150] = (img)(_n6##x,y,z,v)), \
 (I[167] = (img)(_n6##x,_n1##y,z,v)), \
 (I[184] = (img)(_n6##x,_n2##y,z,v)), \
 (I[201] = (img)(_n6##x,_n3##y,z,v)), \
 (I[218] = (img)(_n6##x,_n4##y,z,v)), \
 (I[235] = (img)(_n6##x,_n5##y,z,v)), \
 (I[252] = (img)(_n6##x,_n6##y,z,v)), \
 (I[269] = (img)(_n6##x,_n7##y,z,v)), \
 (I[286] = (img)(_n6##x,_n8##y,z,v)), \
 (I[15] = (img)(_n7##x,_p8##y,z,v)), \
 (I[32] = (img)(_n7##x,_p7##y,z,v)), \
 (I[49] = (img)(_n7##x,_p6##y,z,v)), \
 (I[66] = (img)(_n7##x,_p5##y,z,v)), \
 (I[83] = (img)(_n7##x,_p4##y,z,v)), \
 (I[100] = (img)(_n7##x,_p3##y,z,v)), \
 (I[117] = (img)(_n7##x,_p2##y,z,v)), \
 (I[134] = (img)(_n7##x,_p1##y,z,v)), \
 (I[151] = (img)(_n7##x,y,z,v)), \
 (I[168] = (img)(_n7##x,_n1##y,z,v)), \
 (I[185] = (img)(_n7##x,_n2##y,z,v)), \
 (I[202] = (img)(_n7##x,_n3##y,z,v)), \
 (I[219] = (img)(_n7##x,_n4##y,z,v)), \
 (I[236] = (img)(_n7##x,_n5##y,z,v)), \
 (I[253] = (img)(_n7##x,_n6##y,z,v)), \
 (I[270] = (img)(_n7##x,_n7##y,z,v)), \
 (I[287] = (img)(_n7##x,_n8##y,z,v)), \
 x+8>=(int)((img).width)?(int)((img).width)-1:x+8); \
 x<=(int)(x1) && ((_n8##x<(int)((img).width) && ( \
 (I[16] = (img)(_n8##x,_p8##y,z,v)), \
 (I[33] = (img)(_n8##x,_p7##y,z,v)), \
 (I[50] = (img)(_n8##x,_p6##y,z,v)), \
 (I[67] = (img)(_n8##x,_p5##y,z,v)), \
 (I[84] = (img)(_n8##x,_p4##y,z,v)), \
 (I[101] = (img)(_n8##x,_p3##y,z,v)), \
 (I[118] = (img)(_n8##x,_p2##y,z,v)), \
 (I[135] = (img)(_n8##x,_p1##y,z,v)), \
 (I[152] = (img)(_n8##x,y,z,v)), \
 (I[169] = (img)(_n8##x,_n1##y,z,v)), \
 (I[186] = (img)(_n8##x,_n2##y,z,v)), \
 (I[203] = (img)(_n8##x,_n3##y,z,v)), \
 (I[220] = (img)(_n8##x,_n4##y,z,v)), \
 (I[237] = (img)(_n8##x,_n5##y,z,v)), \
 (I[254] = (img)(_n8##x,_n6##y,z,v)), \
 (I[271] = (img)(_n8##x,_n7##y,z,v)), \
 (I[288] = (img)(_n8##x,_n8##y,z,v)),1)) || \
 _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], \
 I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], \
 I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], \
 I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], \
 I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], \
 I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], \
 I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], \
 I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], \
 I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], \
 I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], \
 I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], \
 I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], \
 I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], \
 I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], \
 I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], \
 I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], \
 I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], I[287] = I[288], \
 _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x)

#define cimg_get17x17(img,x,y,z,v,I) \
 I[0] = (img)(_p8##x,_p8##y,z,v), I[1] = (img)(_p7##x,_p8##y,z,v), I[2] = (img)(_p6##x,_p8##y,z,v), I[3] = (img)(_p5##x,_p8##y,z,v), I[4] = (img)(_p4##x,_p8##y,z,v), I[5] = (img)(_p3##x,_p8##y,z,v), I[6] = (img)(_p2##x,_p8##y,z,v), I[7] = (img)(_p1##x,_p8##y,z,v), I[8] = (img)(x,_p8##y,z,v), I[9] = (img)(_n1##x,_p8##y,z,v), I[10] = (img)(_n2##x,_p8##y,z,v), I[11] = (img)(_n3##x,_p8##y,z,v), I[12] = (img)(_n4##x,_p8##y,z,v), I[13] = (img)(_n5##x,_p8##y,z,v), I[14] = (img)(_n6##x,_p8##y,z,v), I[15] = (img)(_n7##x,_p8##y,z,v), I[16] = (img)(_n8##x,_p8##y,z,v), \
 I[17] = (img)(_p8##x,_p7##y,z,v), I[18] = (img)(_p7##x,_p7##y,z,v), I[19] = (img)(_p6##x,_p7##y,z,v), I[20] = (img)(_p5##x,_p7##y,z,v), I[21] = (img)(_p4##x,_p7##y,z,v), I[22] = (img)(_p3##x,_p7##y,z,v), I[23] = (img)(_p2##x,_p7##y,z,v), I[24] = (img)(_p1##x,_p7##y,z,v), I[25] = (img)(x,_p7##y,z,v), I[26] = (img)(_n1##x,_p7##y,z,v), I[27] = (img)(_n2##x,_p7##y,z,v), I[28] = (img)(_n3##x,_p7##y,z,v), I[29] = (img)(_n4##x,_p7##y,z,v), I[30] = (img)(_n5##x,_p7##y,z,v), I[31] = (img)(_n6##x,_p7##y,z,v), I[32] = (img)(_n7##x,_p7##y,z,v), I[33] = (img)(_n8##x,_p7##y,z,v), \
 I[34] = (img)(_p8##x,_p6##y,z,v), I[35] = (img)(_p7##x,_p6##y,z,v), I[36] = (img)(_p6##x,_p6##y,z,v), I[37] = (img)(_p5##x,_p6##y,z,v), I[38] = (img)(_p4##x,_p6##y,z,v), I[39] = (img)(_p3##x,_p6##y,z,v), I[40] = (img)(_p2##x,_p6##y,z,v), I[41] = (img)(_p1##x,_p6##y,z,v), I[42] = (img)(x,_p6##y,z,v), I[43] = (img)(_n1##x,_p6##y,z,v), I[44] = (img)(_n2##x,_p6##y,z,v), I[45] = (img)(_n3##x,_p6##y,z,v), I[46] = (img)(_n4##x,_p6##y,z,v), I[47] = (img)(_n5##x,_p6##y,z,v), I[48] = (img)(_n6##x,_p6##y,z,v), I[49] = (img)(_n7##x,_p6##y,z,v), I[50] = (img)(_n8##x,_p6##y,z,v), \
 I[51] = (img)(_p8##x,_p5##y,z,v), I[52] = (img)(_p7##x,_p5##y,z,v), I[53] = (img)(_p6##x,_p5##y,z,v), I[54] = (img)(_p5##x,_p5##y,z,v), I[55] = (img)(_p4##x,_p5##y,z,v), I[56] = (img)(_p3##x,_p5##y,z,v), I[57] = (img)(_p2##x,_p5##y,z,v), I[58] = (img)(_p1##x,_p5##y,z,v), I[59] = (img)(x,_p5##y,z,v), I[60] = (img)(_n1##x,_p5##y,z,v), I[61] = (img)(_n2##x,_p5##y,z,v), I[62] = (img)(_n3##x,_p5##y,z,v), I[63] = (img)(_n4##x,_p5##y,z,v), I[64] = (img)(_n5##x,_p5##y,z,v), I[65] = (img)(_n6##x,_p5##y,z,v), I[66] = (img)(_n7##x,_p5##y,z,v), I[67] = (img)(_n8##x,_p5##y,z,v), \
 I[68] = (img)(_p8##x,_p4##y,z,v), I[69] = (img)(_p7##x,_p4##y,z,v), I[70] = (img)(_p6##x,_p4##y,z,v), I[71] = (img)(_p5##x,_p4##y,z,v), I[72] = (img)(_p4##x,_p4##y,z,v), I[73] = (img)(_p3##x,_p4##y,z,v), I[74] = (img)(_p2##x,_p4##y,z,v), I[75] = (img)(_p1##x,_p4##y,z,v), I[76] = (img)(x,_p4##y,z,v), I[77] = (img)(_n1##x,_p4##y,z,v), I[78] = (img)(_n2##x,_p4##y,z,v), I[79] = (img)(_n3##x,_p4##y,z,v), I[80] = (img)(_n4##x,_p4##y,z,v), I[81] = (img)(_n5##x,_p4##y,z,v), I[82] = (img)(_n6##x,_p4##y,z,v), I[83] = (img)(_n7##x,_p4##y,z,v), I[84] = (img)(_n8##x,_p4##y,z,v), \
 I[85] = (img)(_p8##x,_p3##y,z,v), I[86] = (img)(_p7##x,_p3##y,z,v), I[87] = (img)(_p6##x,_p3##y,z,v), I[88] = (img)(_p5##x,_p3##y,z,v), I[89] = (img)(_p4##x,_p3##y,z,v), I[90] = (img)(_p3##x,_p3##y,z,v), I[91] = (img)(_p2##x,_p3##y,z,v), I[92] = (img)(_p1##x,_p3##y,z,v), I[93] = (img)(x,_p3##y,z,v), I[94] = (img)(_n1##x,_p3##y,z,v), I[95] = (img)(_n2##x,_p3##y,z,v), I[96] = (img)(_n3##x,_p3##y,z,v), I[97] = (img)(_n4##x,_p3##y,z,v), I[98] = (img)(_n5##x,_p3##y,z,v), I[99] = (img)(_n6##x,_p3##y,z,v), I[100] = (img)(_n7##x,_p3##y,z,v), I[101] = (img)(_n8##x,_p3##y,z,v), \
 I[102] = (img)(_p8##x,_p2##y,z,v), I[103] = (img)(_p7##x,_p2##y,z,v), I[104] = (img)(_p6##x,_p2##y,z,v), I[105] = (img)(_p5##x,_p2##y,z,v), I[106] = (img)(_p4##x,_p2##y,z,v), I[107] = (img)(_p3##x,_p2##y,z,v), I[108] = (img)(_p2##x,_p2##y,z,v), I[109] = (img)(_p1##x,_p2##y,z,v), I[110] = (img)(x,_p2##y,z,v), I[111] = (img)(_n1##x,_p2##y,z,v), I[112] = (img)(_n2##x,_p2##y,z,v), I[113] = (img)(_n3##x,_p2##y,z,v), I[114] = (img)(_n4##x,_p2##y,z,v), I[115] = (img)(_n5##x,_p2##y,z,v), I[116] = (img)(_n6##x,_p2##y,z,v), I[117] = (img)(_n7##x,_p2##y,z,v), I[118] = (img)(_n8##x,_p2##y,z,v), \
 I[119] = (img)(_p8##x,_p1##y,z,v), I[120] = (img)(_p7##x,_p1##y,z,v), I[121] = (img)(_p6##x,_p1##y,z,v), I[122] = (img)(_p5##x,_p1##y,z,v), I[123] = (img)(_p4##x,_p1##y,z,v), I[124] = (img)(_p3##x,_p1##y,z,v), I[125] = (img)(_p2##x,_p1##y,z,v), I[126] = (img)(_p1##x,_p1##y,z,v), I[127] = (img)(x,_p1##y,z,v), I[128] = (img)(_n1##x,_p1##y,z,v), I[129] = (img)(_n2##x,_p1##y,z,v), I[130] = (img)(_n3##x,_p1##y,z,v), I[131] = (img)(_n4##x,_p1##y,z,v), I[132] = (img)(_n5##x,_p1##y,z,v), I[133] = (img)(_n6##x,_p1##y,z,v), I[134] = (img)(_n7##x,_p1##y,z,v), I[135] = (img)(_n8##x,_p1##y,z,v), \
 I[136] = (img)(_p8##x,y,z,v), I[137] = (img)(_p7##x,y,z,v), I[138] = (img)(_p6##x,y,z,v), I[139] = (img)(_p5##x,y,z,v), I[140] = (img)(_p4##x,y,z,v), I[141] = (img)(_p3##x,y,z,v), I[142] = (img)(_p2##x,y,z,v), I[143] = (img)(_p1##x,y,z,v), I[144] = (img)(x,y,z,v), I[145] = (img)(_n1##x,y,z,v), I[146] = (img)(_n2##x,y,z,v), I[147] = (img)(_n3##x,y,z,v), I[148] = (img)(_n4##x,y,z,v), I[149] = (img)(_n5##x,y,z,v), I[150] = (img)(_n6##x,y,z,v), I[151] = (img)(_n7##x,y,z,v), I[152] = (img)(_n8##x,y,z,v), \
 I[153] = (img)(_p8##x,_n1##y,z,v), I[154] = (img)(_p7##x,_n1##y,z,v), I[155] = (img)(_p6##x,_n1##y,z,v), I[156] = (img)(_p5##x,_n1##y,z,v), I[157] = (img)(_p4##x,_n1##y,z,v), I[158] = (img)(_p3##x,_n1##y,z,v), I[159] = (img)(_p2##x,_n1##y,z,v), I[160] = (img)(_p1##x,_n1##y,z,v), I[161] = (img)(x,_n1##y,z,v), I[162] = (img)(_n1##x,_n1##y,z,v), I[163] = (img)(_n2##x,_n1##y,z,v), I[164] = (img)(_n3##x,_n1##y,z,v), I[165] = (img)(_n4##x,_n1##y,z,v), I[166] = (img)(_n5##x,_n1##y,z,v), I[167] = (img)(_n6##x,_n1##y,z,v), I[168] = (img)(_n7##x,_n1##y,z,v), I[169] = (img)(_n8##x,_n1##y,z,v), \
 I[170] = (img)(_p8##x,_n2##y,z,v), I[171] = (img)(_p7##x,_n2##y,z,v), I[172] = (img)(_p6##x,_n2##y,z,v), I[173] = (img)(_p5##x,_n2##y,z,v), I[174] = (img)(_p4##x,_n2##y,z,v), I[175] = (img)(_p3##x,_n2##y,z,v), I[176] = (img)(_p2##x,_n2##y,z,v), I[177] = (img)(_p1##x,_n2##y,z,v), I[178] = (img)(x,_n2##y,z,v), I[179] = (img)(_n1##x,_n2##y,z,v), I[180] = (img)(_n2##x,_n2##y,z,v), I[181] = (img)(_n3##x,_n2##y,z,v), I[182] = (img)(_n4##x,_n2##y,z,v), I[183] = (img)(_n5##x,_n2##y,z,v), I[184] = (img)(_n6##x,_n2##y,z,v), I[185] = (img)(_n7##x,_n2##y,z,v), I[186] = (img)(_n8##x,_n2##y,z,v), \
 I[187] = (img)(_p8##x,_n3##y,z,v), I[188] = (img)(_p7##x,_n3##y,z,v), I[189] = (img)(_p6##x,_n3##y,z,v), I[190] = (img)(_p5##x,_n3##y,z,v), I[191] = (img)(_p4##x,_n3##y,z,v), I[192] = (img)(_p3##x,_n3##y,z,v), I[193] = (img)(_p2##x,_n3##y,z,v), I[194] = (img)(_p1##x,_n3##y,z,v), I[195] = (img)(x,_n3##y,z,v), I[196] = (img)(_n1##x,_n3##y,z,v), I[197] = (img)(_n2##x,_n3##y,z,v), I[198] = (img)(_n3##x,_n3##y,z,v), I[199] = (img)(_n4##x,_n3##y,z,v), I[200] = (img)(_n5##x,_n3##y,z,v), I[201] = (img)(_n6##x,_n3##y,z,v), I[202] = (img)(_n7##x,_n3##y,z,v), I[203] = (img)(_n8##x,_n3##y,z,v), \
 I[204] = (img)(_p8##x,_n4##y,z,v), I[205] = (img)(_p7##x,_n4##y,z,v), I[206] = (img)(_p6##x,_n4##y,z,v), I[207] = (img)(_p5##x,_n4##y,z,v), I[208] = (img)(_p4##x,_n4##y,z,v), I[209] = (img)(_p3##x,_n4##y,z,v), I[210] = (img)(_p2##x,_n4##y,z,v), I[211] = (img)(_p1##x,_n4##y,z,v), I[212] = (img)(x,_n4##y,z,v), I[213] = (img)(_n1##x,_n4##y,z,v), I[214] = (img)(_n2##x,_n4##y,z,v), I[215] = (img)(_n3##x,_n4##y,z,v), I[216] = (img)(_n4##x,_n4##y,z,v), I[217] = (img)(_n5##x,_n4##y,z,v), I[218] = (img)(_n6##x,_n4##y,z,v), I[219] = (img)(_n7##x,_n4##y,z,v), I[220] = (img)(_n8##x,_n4##y,z,v), \
 I[221] = (img)(_p8##x,_n5##y,z,v), I[222] = (img)(_p7##x,_n5##y,z,v), I[223] = (img)(_p6##x,_n5##y,z,v), I[224] = (img)(_p5##x,_n5##y,z,v), I[225] = (img)(_p4##x,_n5##y,z,v), I[226] = (img)(_p3##x,_n5##y,z,v), I[227] = (img)(_p2##x,_n5##y,z,v), I[228] = (img)(_p1##x,_n5##y,z,v), I[229] = (img)(x,_n5##y,z,v), I[230] = (img)(_n1##x,_n5##y,z,v), I[231] = (img)(_n2##x,_n5##y,z,v), I[232] = (img)(_n3##x,_n5##y,z,v), I[233] = (img)(_n4##x,_n5##y,z,v), I[234] = (img)(_n5##x,_n5##y,z,v), I[235] = (img)(_n6##x,_n5##y,z,v), I[236] = (img)(_n7##x,_n5##y,z,v), I[237] = (img)(_n8##x,_n5##y,z,v), \
 I[238] = (img)(_p8##x,_n6##y,z,v), I[239] = (img)(_p7##x,_n6##y,z,v), I[240] = (img)(_p6##x,_n6##y,z,v), I[241] = (img)(_p5##x,_n6##y,z,v), I[242] = (img)(_p4##x,_n6##y,z,v), I[243] = (img)(_p3##x,_n6##y,z,v), I[244] = (img)(_p2##x,_n6##y,z,v), I[245] = (img)(_p1##x,_n6##y,z,v), I[246] = (img)(x,_n6##y,z,v), I[247] = (img)(_n1##x,_n6##y,z,v), I[248] = (img)(_n2##x,_n6##y,z,v), I[249] = (img)(_n3##x,_n6##y,z,v), I[250] = (img)(_n4##x,_n6##y,z,v), I[251] = (img)(_n5##x,_n6##y,z,v), I[252] = (img)(_n6##x,_n6##y,z,v), I[253] = (img)(_n7##x,_n6##y,z,v), I[254] = (img)(_n8##x,_n6##y,z,v), \
 I[255] = (img)(_p8##x,_n7##y,z,v), I[256] = (img)(_p7##x,_n7##y,z,v), I[257] = (img)(_p6##x,_n7##y,z,v), I[258] = (img)(_p5##x,_n7##y,z,v), I[259] = (img)(_p4##x,_n7##y,z,v), I[260] = (img)(_p3##x,_n7##y,z,v), I[261] = (img)(_p2##x,_n7##y,z,v), I[262] = (img)(_p1##x,_n7##y,z,v), I[263] = (img)(x,_n7##y,z,v), I[264] = (img)(_n1##x,_n7##y,z,v), I[265] = (img)(_n2##x,_n7##y,z,v), I[266] = (img)(_n3##x,_n7##y,z,v), I[267] = (img)(_n4##x,_n7##y,z,v), I[268] = (img)(_n5##x,_n7##y,z,v), I[269] = (img)(_n6##x,_n7##y,z,v), I[270] = (img)(_n7##x,_n7##y,z,v), I[271] = (img)(_n8##x,_n7##y,z,v), \
 I[272] = (img)(_p8##x,_n8##y,z,v), I[273] = (img)(_p7##x,_n8##y,z,v), I[274] = (img)(_p6##x,_n8##y,z,v), I[275] = (img)(_p5##x,_n8##y,z,v), I[276] = (img)(_p4##x,_n8##y,z,v), I[277] = (img)(_p3##x,_n8##y,z,v), I[278] = (img)(_p2##x,_n8##y,z,v), I[279] = (img)(_p1##x,_n8##y,z,v), I[280] = (img)(x,_n8##y,z,v), I[281] = (img)(_n1##x,_n8##y,z,v), I[282] = (img)(_n2##x,_n8##y,z,v), I[283] = (img)(_n3##x,_n8##y,z,v), I[284] = (img)(_n4##x,_n8##y,z,v), I[285] = (img)(_n5##x,_n8##y,z,v), I[286] = (img)(_n6##x,_n8##y,z,v), I[287] = (img)(_n7##x,_n8##y,z,v), I[288] = (img)(_n8##x,_n8##y,z,v);

// Define 18x18 loop macros for CImg
//----------------------------------
#define cimg_for18(bound,i) for (int i = 0, \
 _p8##i = 0, _p7##i = 0, _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7, \
 _n8##i = 8>=(int)(bound)?(int)(bound)-1:8, \
 _n9##i = 9>=(int)(bound)?(int)(bound)-1:9; \
 _n9##i<(int)(bound) || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i)

#define cimg_for18X(img,x) cimg_for18((img).width,x)
#define cimg_for18Y(img,y) cimg_for18((img).height,y)
#define cimg_for18Z(img,z) cimg_for18((img).depth,z)
#define cimg_for18V(img,v) cimg_for18((img).dim,v)
#define cimg_for18XY(img,x,y) cimg_for18Y(img,y) cimg_for18X(img,x)
#define cimg_for18XZ(img,x,z) cimg_for18Z(img,z) cimg_for18X(img,x)
#define cimg_for18XV(img,x,v) cimg_for18V(img,v) cimg_for18X(img,x)
#define cimg_for18YZ(img,y,z) cimg_for18Z(img,z) cimg_for18Y(img,y)
#define cimg_for18YV(img,y,v) cimg_for18V(img,v) cimg_for18Y(img,y)
#define cimg_for18ZV(img,z,v) cimg_for18V(img,v) cimg_for18Z(img,z)
#define cimg_for18XYZ(img,x,y,z) cimg_for18Z(img,z) cimg_for18XY(img,x,y)
#define cimg_for18XZV(img,x,z,v) cimg_for18V(img,v) cimg_for18XZ(img,x,z)
#define cimg_for18YZV(img,y,z,v) cimg_for18V(img,v) cimg_for18YZ(img,y,z)
#define cimg_for18XYZV(img,x,y,z,v) cimg_for18V(img,v) cimg_for18XYZ(img,x,y,z)

#define cimg_for_in18(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p8##i = i-8<0?0:i-8, \
 _p7##i = i-7<0?0:i-7, \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7, \
 _n8##i = i+8>=(int)(bound)?(int)(bound)-1:i+8, \
 _n9##i = i+9>=(int)(bound)?(int)(bound)-1:i+9; \
 i<=(int)(i1) && (_n9##i<(int)(bound) || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i)

#define cimg_for_in18X(img,x0,x1,x) cimg_for_in18((img).width,x0,x1,x)
#define cimg_for_in18Y(img,y0,y1,y) cimg_for_in18((img).height,y0,y1,y)
#define cimg_for_in18Z(img,z0,z1,z) cimg_for_in18((img).depth,z0,z1,z)
#define cimg_for_in18V(img,v0,v1,v) cimg_for_in18((img).dim,v0,v1,v)
#define cimg_for_in18XY(img,x0,y0,x1,y1,x,y) cimg_for_in18Y(img,y0,y1,y) cimg_for_in18X(img,x0,x1,x)
#define cimg_for_in18XZ(img,x0,z0,x1,z1,x,z) cimg_for_in18Z(img,z0,z1,z) cimg_for_in18X(img,x0,x1,x)
#define cimg_for_in18XV(img,x0,v0,x1,v1,x,v) cimg_for_in18V(img,v0,v1,v) cimg_for_in18X(img,x0,x1,x)
#define cimg_for_in18YZ(img,y0,z0,y1,z1,y,z) cimg_for_in18Z(img,z0,z1,z) cimg_for_in18Y(img,y0,y1,y)
#define cimg_for_in18YV(img,y0,v0,y1,v1,y,v) cimg_for_in18V(img,v0,v1,v) cimg_for_in18Y(img,y0,y1,y)
#define cimg_for_in18ZV(img,z0,v0,z1,v1,z,v) cimg_for_in18V(img,v0,v1,v) cimg_for_in18Z(img,z0,z1,z)
#define cimg_for_in18XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in18Z(img,z0,z1,z) cimg_for_in18XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in18XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in18V(img,v0,v1,v) cimg_for_in18XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in18YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in18V(img,v0,v1,v) cimg_for_in18YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in18XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in18V(img,v0,v1,v) cimg_for_in18XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for18x18(img,x,y,z,v,I) \
 cimg_for18((img).height,y) for (int x = 0, \
 _p8##x = 0, _p7##x = 0, _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = 7>=((img).width)?(int)((img).width)-1:7, \
 _n8##x = 8>=((img).width)?(int)((img).width)-1:8, \
 _n9##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = I[7] = I[8] = (img)(0,_p8##y,z,v)), \
 (I[18] = I[19] = I[20] = I[21] = I[22] = I[23] = I[24] = I[25] = I[26] = (img)(0,_p7##y,z,v)), \
 (I[36] = I[37] = I[38] = I[39] = I[40] = I[41] = I[42] = I[43] = I[44] = (img)(0,_p6##y,z,v)), \
 (I[54] = I[55] = I[56] = I[57] = I[58] = I[59] = I[60] = I[61] = I[62] = (img)(0,_p5##y,z,v)), \
 (I[72] = I[73] = I[74] = I[75] = I[76] = I[77] = I[78] = I[79] = I[80] = (img)(0,_p4##y,z,v)), \
 (I[90] = I[91] = I[92] = I[93] = I[94] = I[95] = I[96] = I[97] = I[98] = (img)(0,_p3##y,z,v)), \
 (I[108] = I[109] = I[110] = I[111] = I[112] = I[113] = I[114] = I[115] = I[116] = (img)(0,_p2##y,z,v)), \
 (I[126] = I[127] = I[128] = I[129] = I[130] = I[131] = I[132] = I[133] = I[134] = (img)(0,_p1##y,z,v)), \
 (I[144] = I[145] = I[146] = I[147] = I[148] = I[149] = I[150] = I[151] = I[152] = (img)(0,y,z,v)), \
 (I[162] = I[163] = I[164] = I[165] = I[166] = I[167] = I[168] = I[169] = I[170] = (img)(0,_n1##y,z,v)), \
 (I[180] = I[181] = I[182] = I[183] = I[184] = I[185] = I[186] = I[187] = I[188] = (img)(0,_n2##y,z,v)), \
 (I[198] = I[199] = I[200] = I[201] = I[202] = I[203] = I[204] = I[205] = I[206] = (img)(0,_n3##y,z,v)), \
 (I[216] = I[217] = I[218] = I[219] = I[220] = I[221] = I[222] = I[223] = I[224] = (img)(0,_n4##y,z,v)), \
 (I[234] = I[235] = I[236] = I[237] = I[238] = I[239] = I[240] = I[241] = I[242] = (img)(0,_n5##y,z,v)), \
 (I[252] = I[253] = I[254] = I[255] = I[256] = I[257] = I[258] = I[259] = I[260] = (img)(0,_n6##y,z,v)), \
 (I[270] = I[271] = I[272] = I[273] = I[274] = I[275] = I[276] = I[277] = I[278] = (img)(0,_n7##y,z,v)), \
 (I[288] = I[289] = I[290] = I[291] = I[292] = I[293] = I[294] = I[295] = I[296] = (img)(0,_n8##y,z,v)), \
 (I[306] = I[307] = I[308] = I[309] = I[310] = I[311] = I[312] = I[313] = I[314] = (img)(0,_n9##y,z,v)), \
 (I[9] = (img)(_n1##x,_p8##y,z,v)), \
 (I[27] = (img)(_n1##x,_p7##y,z,v)), \
 (I[45] = (img)(_n1##x,_p6##y,z,v)), \
 (I[63] = (img)(_n1##x,_p5##y,z,v)), \
 (I[81] = (img)(_n1##x,_p4##y,z,v)), \
 (I[99] = (img)(_n1##x,_p3##y,z,v)), \
 (I[117] = (img)(_n1##x,_p2##y,z,v)), \
 (I[135] = (img)(_n1##x,_p1##y,z,v)), \
 (I[153] = (img)(_n1##x,y,z,v)), \
 (I[171] = (img)(_n1##x,_n1##y,z,v)), \
 (I[189] = (img)(_n1##x,_n2##y,z,v)), \
 (I[207] = (img)(_n1##x,_n3##y,z,v)), \
 (I[225] = (img)(_n1##x,_n4##y,z,v)), \
 (I[243] = (img)(_n1##x,_n5##y,z,v)), \
 (I[261] = (img)(_n1##x,_n6##y,z,v)), \
 (I[279] = (img)(_n1##x,_n7##y,z,v)), \
 (I[297] = (img)(_n1##x,_n8##y,z,v)), \
 (I[315] = (img)(_n1##x,_n9##y,z,v)), \
 (I[10] = (img)(_n2##x,_p8##y,z,v)), \
 (I[28] = (img)(_n2##x,_p7##y,z,v)), \
 (I[46] = (img)(_n2##x,_p6##y,z,v)), \
 (I[64] = (img)(_n2##x,_p5##y,z,v)), \
 (I[82] = (img)(_n2##x,_p4##y,z,v)), \
 (I[100] = (img)(_n2##x,_p3##y,z,v)), \
 (I[118] = (img)(_n2##x,_p2##y,z,v)), \
 (I[136] = (img)(_n2##x,_p1##y,z,v)), \
 (I[154] = (img)(_n2##x,y,z,v)), \
 (I[172] = (img)(_n2##x,_n1##y,z,v)), \
 (I[190] = (img)(_n2##x,_n2##y,z,v)), \
 (I[208] = (img)(_n2##x,_n3##y,z,v)), \
 (I[226] = (img)(_n2##x,_n4##y,z,v)), \
 (I[244] = (img)(_n2##x,_n5##y,z,v)), \
 (I[262] = (img)(_n2##x,_n6##y,z,v)), \
 (I[280] = (img)(_n2##x,_n7##y,z,v)), \
 (I[298] = (img)(_n2##x,_n8##y,z,v)), \
 (I[316] = (img)(_n2##x,_n9##y,z,v)), \
 (I[11] = (img)(_n3##x,_p8##y,z,v)), \
 (I[29] = (img)(_n3##x,_p7##y,z,v)), \
 (I[47] = (img)(_n3##x,_p6##y,z,v)), \
 (I[65] = (img)(_n3##x,_p5##y,z,v)), \
 (I[83] = (img)(_n3##x,_p4##y,z,v)), \
 (I[101] = (img)(_n3##x,_p3##y,z,v)), \
 (I[119] = (img)(_n3##x,_p2##y,z,v)), \
 (I[137] = (img)(_n3##x,_p1##y,z,v)), \
 (I[155] = (img)(_n3##x,y,z,v)), \
 (I[173] = (img)(_n3##x,_n1##y,z,v)), \
 (I[191] = (img)(_n3##x,_n2##y,z,v)), \
 (I[209] = (img)(_n3##x,_n3##y,z,v)), \
 (I[227] = (img)(_n3##x,_n4##y,z,v)), \
 (I[245] = (img)(_n3##x,_n5##y,z,v)), \
 (I[263] = (img)(_n3##x,_n6##y,z,v)), \
 (I[281] = (img)(_n3##x,_n7##y,z,v)), \
 (I[299] = (img)(_n3##x,_n8##y,z,v)), \
 (I[317] = (img)(_n3##x,_n9##y,z,v)), \
 (I[12] = (img)(_n4##x,_p8##y,z,v)), \
 (I[30] = (img)(_n4##x,_p7##y,z,v)), \
 (I[48] = (img)(_n4##x,_p6##y,z,v)), \
 (I[66] = (img)(_n4##x,_p5##y,z,v)), \
 (I[84] = (img)(_n4##x,_p4##y,z,v)), \
 (I[102] = (img)(_n4##x,_p3##y,z,v)), \
 (I[120] = (img)(_n4##x,_p2##y,z,v)), \
 (I[138] = (img)(_n4##x,_p1##y,z,v)), \
 (I[156] = (img)(_n4##x,y,z,v)), \
 (I[174] = (img)(_n4##x,_n1##y,z,v)), \
 (I[192] = (img)(_n4##x,_n2##y,z,v)), \
 (I[210] = (img)(_n4##x,_n3##y,z,v)), \
 (I[228] = (img)(_n4##x,_n4##y,z,v)), \
 (I[246] = (img)(_n4##x,_n5##y,z,v)), \
 (I[264] = (img)(_n4##x,_n6##y,z,v)), \
 (I[282] = (img)(_n4##x,_n7##y,z,v)), \
 (I[300] = (img)(_n4##x,_n8##y,z,v)), \
 (I[318] = (img)(_n4##x,_n9##y,z,v)), \
 (I[13] = (img)(_n5##x,_p8##y,z,v)), \
 (I[31] = (img)(_n5##x,_p7##y,z,v)), \
 (I[49] = (img)(_n5##x,_p6##y,z,v)), \
 (I[67] = (img)(_n5##x,_p5##y,z,v)), \
 (I[85] = (img)(_n5##x,_p4##y,z,v)), \
 (I[103] = (img)(_n5##x,_p3##y,z,v)), \
 (I[121] = (img)(_n5##x,_p2##y,z,v)), \
 (I[139] = (img)(_n5##x,_p1##y,z,v)), \
 (I[157] = (img)(_n5##x,y,z,v)), \
 (I[175] = (img)(_n5##x,_n1##y,z,v)), \
 (I[193] = (img)(_n5##x,_n2##y,z,v)), \
 (I[211] = (img)(_n5##x,_n3##y,z,v)), \
 (I[229] = (img)(_n5##x,_n4##y,z,v)), \
 (I[247] = (img)(_n5##x,_n5##y,z,v)), \
 (I[265] = (img)(_n5##x,_n6##y,z,v)), \
 (I[283] = (img)(_n5##x,_n7##y,z,v)), \
 (I[301] = (img)(_n5##x,_n8##y,z,v)), \
 (I[319] = (img)(_n5##x,_n9##y,z,v)), \
 (I[14] = (img)(_n6##x,_p8##y,z,v)), \
 (I[32] = (img)(_n6##x,_p7##y,z,v)), \
 (I[50] = (img)(_n6##x,_p6##y,z,v)), \
 (I[68] = (img)(_n6##x,_p5##y,z,v)), \
 (I[86] = (img)(_n6##x,_p4##y,z,v)), \
 (I[104] = (img)(_n6##x,_p3##y,z,v)), \
 (I[122] = (img)(_n6##x,_p2##y,z,v)), \
 (I[140] = (img)(_n6##x,_p1##y,z,v)), \
 (I[158] = (img)(_n6##x,y,z,v)), \
 (I[176] = (img)(_n6##x,_n1##y,z,v)), \
 (I[194] = (img)(_n6##x,_n2##y,z,v)), \
 (I[212] = (img)(_n6##x,_n3##y,z,v)), \
 (I[230] = (img)(_n6##x,_n4##y,z,v)), \
 (I[248] = (img)(_n6##x,_n5##y,z,v)), \
 (I[266] = (img)(_n6##x,_n6##y,z,v)), \
 (I[284] = (img)(_n6##x,_n7##y,z,v)), \
 (I[302] = (img)(_n6##x,_n8##y,z,v)), \
 (I[320] = (img)(_n6##x,_n9##y,z,v)), \
 (I[15] = (img)(_n7##x,_p8##y,z,v)), \
 (I[33] = (img)(_n7##x,_p7##y,z,v)), \
 (I[51] = (img)(_n7##x,_p6##y,z,v)), \
 (I[69] = (img)(_n7##x,_p5##y,z,v)), \
 (I[87] = (img)(_n7##x,_p4##y,z,v)), \
 (I[105] = (img)(_n7##x,_p3##y,z,v)), \
 (I[123] = (img)(_n7##x,_p2##y,z,v)), \
 (I[141] = (img)(_n7##x,_p1##y,z,v)), \
 (I[159] = (img)(_n7##x,y,z,v)), \
 (I[177] = (img)(_n7##x,_n1##y,z,v)), \
 (I[195] = (img)(_n7##x,_n2##y,z,v)), \
 (I[213] = (img)(_n7##x,_n3##y,z,v)), \
 (I[231] = (img)(_n7##x,_n4##y,z,v)), \
 (I[249] = (img)(_n7##x,_n5##y,z,v)), \
 (I[267] = (img)(_n7##x,_n6##y,z,v)), \
 (I[285] = (img)(_n7##x,_n7##y,z,v)), \
 (I[303] = (img)(_n7##x,_n8##y,z,v)), \
 (I[321] = (img)(_n7##x,_n9##y,z,v)), \
 (I[16] = (img)(_n8##x,_p8##y,z,v)), \
 (I[34] = (img)(_n8##x,_p7##y,z,v)), \
 (I[52] = (img)(_n8##x,_p6##y,z,v)), \
 (I[70] = (img)(_n8##x,_p5##y,z,v)), \
 (I[88] = (img)(_n8##x,_p4##y,z,v)), \
 (I[106] = (img)(_n8##x,_p3##y,z,v)), \
 (I[124] = (img)(_n8##x,_p2##y,z,v)), \
 (I[142] = (img)(_n8##x,_p1##y,z,v)), \
 (I[160] = (img)(_n8##x,y,z,v)), \
 (I[178] = (img)(_n8##x,_n1##y,z,v)), \
 (I[196] = (img)(_n8##x,_n2##y,z,v)), \
 (I[214] = (img)(_n8##x,_n3##y,z,v)), \
 (I[232] = (img)(_n8##x,_n4##y,z,v)), \
 (I[250] = (img)(_n8##x,_n5##y,z,v)), \
 (I[268] = (img)(_n8##x,_n6##y,z,v)), \
 (I[286] = (img)(_n8##x,_n7##y,z,v)), \
 (I[304] = (img)(_n8##x,_n8##y,z,v)), \
 (I[322] = (img)(_n8##x,_n9##y,z,v)), \
 9>=((img).width)?(int)((img).width)-1:9); \
 (_n9##x<(int)((img).width) && ( \
 (I[17] = (img)(_n9##x,_p8##y,z,v)), \
 (I[35] = (img)(_n9##x,_p7##y,z,v)), \
 (I[53] = (img)(_n9##x,_p6##y,z,v)), \
 (I[71] = (img)(_n9##x,_p5##y,z,v)), \
 (I[89] = (img)(_n9##x,_p4##y,z,v)), \
 (I[107] = (img)(_n9##x,_p3##y,z,v)), \
 (I[125] = (img)(_n9##x,_p2##y,z,v)), \
 (I[143] = (img)(_n9##x,_p1##y,z,v)), \
 (I[161] = (img)(_n9##x,y,z,v)), \
 (I[179] = (img)(_n9##x,_n1##y,z,v)), \
 (I[197] = (img)(_n9##x,_n2##y,z,v)), \
 (I[215] = (img)(_n9##x,_n3##y,z,v)), \
 (I[233] = (img)(_n9##x,_n4##y,z,v)), \
 (I[251] = (img)(_n9##x,_n5##y,z,v)), \
 (I[269] = (img)(_n9##x,_n6##y,z,v)), \
 (I[287] = (img)(_n9##x,_n7##y,z,v)), \
 (I[305] = (img)(_n9##x,_n8##y,z,v)), \
 (I[323] = (img)(_n9##x,_n9##y,z,v)),1)) || \
 _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], \
 I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], \
 I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], \
 I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], \
 I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], \
 I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], \
 I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], \
 I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], \
 I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], \
 I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], \
 I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], \
 I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], \
 I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], \
 I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], \
 I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], \
 I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], \
 I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], I[321] = I[322], I[322] = I[323], \
 _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x)

#define cimg_for_in18x18(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in18((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p8##x = x-8<0?0:x-8, \
 _p7##x = x-7<0?0:x-7, \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = x+7>=(int)((img).width)?(int)((img).width)-1:x+7, \
 _n8##x = x+8>=(int)((img).width)?(int)((img).width)-1:x+8, \
 _n9##x = (int)( \
 (I[0] = (img)(_p8##x,_p8##y,z,v)), \
 (I[18] = (img)(_p8##x,_p7##y,z,v)), \
 (I[36] = (img)(_p8##x,_p6##y,z,v)), \
 (I[54] = (img)(_p8##x,_p5##y,z,v)), \
 (I[72] = (img)(_p8##x,_p4##y,z,v)), \
 (I[90] = (img)(_p8##x,_p3##y,z,v)), \
 (I[108] = (img)(_p8##x,_p2##y,z,v)), \
 (I[126] = (img)(_p8##x,_p1##y,z,v)), \
 (I[144] = (img)(_p8##x,y,z,v)), \
 (I[162] = (img)(_p8##x,_n1##y,z,v)), \
 (I[180] = (img)(_p8##x,_n2##y,z,v)), \
 (I[198] = (img)(_p8##x,_n3##y,z,v)), \
 (I[216] = (img)(_p8##x,_n4##y,z,v)), \
 (I[234] = (img)(_p8##x,_n5##y,z,v)), \
 (I[252] = (img)(_p8##x,_n6##y,z,v)), \
 (I[270] = (img)(_p8##x,_n7##y,z,v)), \
 (I[288] = (img)(_p8##x,_n8##y,z,v)), \
 (I[306] = (img)(_p8##x,_n9##y,z,v)), \
 (I[1] = (img)(_p7##x,_p8##y,z,v)), \
 (I[19] = (img)(_p7##x,_p7##y,z,v)), \
 (I[37] = (img)(_p7##x,_p6##y,z,v)), \
 (I[55] = (img)(_p7##x,_p5##y,z,v)), \
 (I[73] = (img)(_p7##x,_p4##y,z,v)), \
 (I[91] = (img)(_p7##x,_p3##y,z,v)), \
 (I[109] = (img)(_p7##x,_p2##y,z,v)), \
 (I[127] = (img)(_p7##x,_p1##y,z,v)), \
 (I[145] = (img)(_p7##x,y,z,v)), \
 (I[163] = (img)(_p7##x,_n1##y,z,v)), \
 (I[181] = (img)(_p7##x,_n2##y,z,v)), \
 (I[199] = (img)(_p7##x,_n3##y,z,v)), \
 (I[217] = (img)(_p7##x,_n4##y,z,v)), \
 (I[235] = (img)(_p7##x,_n5##y,z,v)), \
 (I[253] = (img)(_p7##x,_n6##y,z,v)), \
 (I[271] = (img)(_p7##x,_n7##y,z,v)), \
 (I[289] = (img)(_p7##x,_n8##y,z,v)), \
 (I[307] = (img)(_p7##x,_n9##y,z,v)), \
 (I[2] = (img)(_p6##x,_p8##y,z,v)), \
 (I[20] = (img)(_p6##x,_p7##y,z,v)), \
 (I[38] = (img)(_p6##x,_p6##y,z,v)), \
 (I[56] = (img)(_p6##x,_p5##y,z,v)), \
 (I[74] = (img)(_p6##x,_p4##y,z,v)), \
 (I[92] = (img)(_p6##x,_p3##y,z,v)), \
 (I[110] = (img)(_p6##x,_p2##y,z,v)), \
 (I[128] = (img)(_p6##x,_p1##y,z,v)), \
 (I[146] = (img)(_p6##x,y,z,v)), \
 (I[164] = (img)(_p6##x,_n1##y,z,v)), \
 (I[182] = (img)(_p6##x,_n2##y,z,v)), \
 (I[200] = (img)(_p6##x,_n3##y,z,v)), \
 (I[218] = (img)(_p6##x,_n4##y,z,v)), \
 (I[236] = (img)(_p6##x,_n5##y,z,v)), \
 (I[254] = (img)(_p6##x,_n6##y,z,v)), \
 (I[272] = (img)(_p6##x,_n7##y,z,v)), \
 (I[290] = (img)(_p6##x,_n8##y,z,v)), \
 (I[308] = (img)(_p6##x,_n9##y,z,v)), \
 (I[3] = (img)(_p5##x,_p8##y,z,v)), \
 (I[21] = (img)(_p5##x,_p7##y,z,v)), \
 (I[39] = (img)(_p5##x,_p6##y,z,v)), \
 (I[57] = (img)(_p5##x,_p5##y,z,v)), \
 (I[75] = (img)(_p5##x,_p4##y,z,v)), \
 (I[93] = (img)(_p5##x,_p3##y,z,v)), \
 (I[111] = (img)(_p5##x,_p2##y,z,v)), \
 (I[129] = (img)(_p5##x,_p1##y,z,v)), \
 (I[147] = (img)(_p5##x,y,z,v)), \
 (I[165] = (img)(_p5##x,_n1##y,z,v)), \
 (I[183] = (img)(_p5##x,_n2##y,z,v)), \
 (I[201] = (img)(_p5##x,_n3##y,z,v)), \
 (I[219] = (img)(_p5##x,_n4##y,z,v)), \
 (I[237] = (img)(_p5##x,_n5##y,z,v)), \
 (I[255] = (img)(_p5##x,_n6##y,z,v)), \
 (I[273] = (img)(_p5##x,_n7##y,z,v)), \
 (I[291] = (img)(_p5##x,_n8##y,z,v)), \
 (I[309] = (img)(_p5##x,_n9##y,z,v)), \
 (I[4] = (img)(_p4##x,_p8##y,z,v)), \
 (I[22] = (img)(_p4##x,_p7##y,z,v)), \
 (I[40] = (img)(_p4##x,_p6##y,z,v)), \
 (I[58] = (img)(_p4##x,_p5##y,z,v)), \
 (I[76] = (img)(_p4##x,_p4##y,z,v)), \
 (I[94] = (img)(_p4##x,_p3##y,z,v)), \
 (I[112] = (img)(_p4##x,_p2##y,z,v)), \
 (I[130] = (img)(_p4##x,_p1##y,z,v)), \
 (I[148] = (img)(_p4##x,y,z,v)), \
 (I[166] = (img)(_p4##x,_n1##y,z,v)), \
 (I[184] = (img)(_p4##x,_n2##y,z,v)), \
 (I[202] = (img)(_p4##x,_n3##y,z,v)), \
 (I[220] = (img)(_p4##x,_n4##y,z,v)), \
 (I[238] = (img)(_p4##x,_n5##y,z,v)), \
 (I[256] = (img)(_p4##x,_n6##y,z,v)), \
 (I[274] = (img)(_p4##x,_n7##y,z,v)), \
 (I[292] = (img)(_p4##x,_n8##y,z,v)), \
 (I[310] = (img)(_p4##x,_n9##y,z,v)), \
 (I[5] = (img)(_p3##x,_p8##y,z,v)), \
 (I[23] = (img)(_p3##x,_p7##y,z,v)), \
 (I[41] = (img)(_p3##x,_p6##y,z,v)), \
 (I[59] = (img)(_p3##x,_p5##y,z,v)), \
 (I[77] = (img)(_p3##x,_p4##y,z,v)), \
 (I[95] = (img)(_p3##x,_p3##y,z,v)), \
 (I[113] = (img)(_p3##x,_p2##y,z,v)), \
 (I[131] = (img)(_p3##x,_p1##y,z,v)), \
 (I[149] = (img)(_p3##x,y,z,v)), \
 (I[167] = (img)(_p3##x,_n1##y,z,v)), \
 (I[185] = (img)(_p3##x,_n2##y,z,v)), \
 (I[203] = (img)(_p3##x,_n3##y,z,v)), \
 (I[221] = (img)(_p3##x,_n4##y,z,v)), \
 (I[239] = (img)(_p3##x,_n5##y,z,v)), \
 (I[257] = (img)(_p3##x,_n6##y,z,v)), \
 (I[275] = (img)(_p3##x,_n7##y,z,v)), \
 (I[293] = (img)(_p3##x,_n8##y,z,v)), \
 (I[311] = (img)(_p3##x,_n9##y,z,v)), \
 (I[6] = (img)(_p2##x,_p8##y,z,v)), \
 (I[24] = (img)(_p2##x,_p7##y,z,v)), \
 (I[42] = (img)(_p2##x,_p6##y,z,v)), \
 (I[60] = (img)(_p2##x,_p5##y,z,v)), \
 (I[78] = (img)(_p2##x,_p4##y,z,v)), \
 (I[96] = (img)(_p2##x,_p3##y,z,v)), \
 (I[114] = (img)(_p2##x,_p2##y,z,v)), \
 (I[132] = (img)(_p2##x,_p1##y,z,v)), \
 (I[150] = (img)(_p2##x,y,z,v)), \
 (I[168] = (img)(_p2##x,_n1##y,z,v)), \
 (I[186] = (img)(_p2##x,_n2##y,z,v)), \
 (I[204] = (img)(_p2##x,_n3##y,z,v)), \
 (I[222] = (img)(_p2##x,_n4##y,z,v)), \
 (I[240] = (img)(_p2##x,_n5##y,z,v)), \
 (I[258] = (img)(_p2##x,_n6##y,z,v)), \
 (I[276] = (img)(_p2##x,_n7##y,z,v)), \
 (I[294] = (img)(_p2##x,_n8##y,z,v)), \
 (I[312] = (img)(_p2##x,_n9##y,z,v)), \
 (I[7] = (img)(_p1##x,_p8##y,z,v)), \
 (I[25] = (img)(_p1##x,_p7##y,z,v)), \
 (I[43] = (img)(_p1##x,_p6##y,z,v)), \
 (I[61] = (img)(_p1##x,_p5##y,z,v)), \
 (I[79] = (img)(_p1##x,_p4##y,z,v)), \
 (I[97] = (img)(_p1##x,_p3##y,z,v)), \
 (I[115] = (img)(_p1##x,_p2##y,z,v)), \
 (I[133] = (img)(_p1##x,_p1##y,z,v)), \
 (I[151] = (img)(_p1##x,y,z,v)), \
 (I[169] = (img)(_p1##x,_n1##y,z,v)), \
 (I[187] = (img)(_p1##x,_n2##y,z,v)), \
 (I[205] = (img)(_p1##x,_n3##y,z,v)), \
 (I[223] = (img)(_p1##x,_n4##y,z,v)), \
 (I[241] = (img)(_p1##x,_n5##y,z,v)), \
 (I[259] = (img)(_p1##x,_n6##y,z,v)), \
 (I[277] = (img)(_p1##x,_n7##y,z,v)), \
 (I[295] = (img)(_p1##x,_n8##y,z,v)), \
 (I[313] = (img)(_p1##x,_n9##y,z,v)), \
 (I[8] = (img)(x,_p8##y,z,v)), \
 (I[26] = (img)(x,_p7##y,z,v)), \
 (I[44] = (img)(x,_p6##y,z,v)), \
 (I[62] = (img)(x,_p5##y,z,v)), \
 (I[80] = (img)(x,_p4##y,z,v)), \
 (I[98] = (img)(x,_p3##y,z,v)), \
 (I[116] = (img)(x,_p2##y,z,v)), \
 (I[134] = (img)(x,_p1##y,z,v)), \
 (I[152] = (img)(x,y,z,v)), \
 (I[170] = (img)(x,_n1##y,z,v)), \
 (I[188] = (img)(x,_n2##y,z,v)), \
 (I[206] = (img)(x,_n3##y,z,v)), \
 (I[224] = (img)(x,_n4##y,z,v)), \
 (I[242] = (img)(x,_n5##y,z,v)), \
 (I[260] = (img)(x,_n6##y,z,v)), \
 (I[278] = (img)(x,_n7##y,z,v)), \
 (I[296] = (img)(x,_n8##y,z,v)), \
 (I[314] = (img)(x,_n9##y,z,v)), \
 (I[9] = (img)(_n1##x,_p8##y,z,v)), \
 (I[27] = (img)(_n1##x,_p7##y,z,v)), \
 (I[45] = (img)(_n1##x,_p6##y,z,v)), \
 (I[63] = (img)(_n1##x,_p5##y,z,v)), \
 (I[81] = (img)(_n1##x,_p4##y,z,v)), \
 (I[99] = (img)(_n1##x,_p3##y,z,v)), \
 (I[117] = (img)(_n1##x,_p2##y,z,v)), \
 (I[135] = (img)(_n1##x,_p1##y,z,v)), \
 (I[153] = (img)(_n1##x,y,z,v)), \
 (I[171] = (img)(_n1##x,_n1##y,z,v)), \
 (I[189] = (img)(_n1##x,_n2##y,z,v)), \
 (I[207] = (img)(_n1##x,_n3##y,z,v)), \
 (I[225] = (img)(_n1##x,_n4##y,z,v)), \
 (I[243] = (img)(_n1##x,_n5##y,z,v)), \
 (I[261] = (img)(_n1##x,_n6##y,z,v)), \
 (I[279] = (img)(_n1##x,_n7##y,z,v)), \
 (I[297] = (img)(_n1##x,_n8##y,z,v)), \
 (I[315] = (img)(_n1##x,_n9##y,z,v)), \
 (I[10] = (img)(_n2##x,_p8##y,z,v)), \
 (I[28] = (img)(_n2##x,_p7##y,z,v)), \
 (I[46] = (img)(_n2##x,_p6##y,z,v)), \
 (I[64] = (img)(_n2##x,_p5##y,z,v)), \
 (I[82] = (img)(_n2##x,_p4##y,z,v)), \
 (I[100] = (img)(_n2##x,_p3##y,z,v)), \
 (I[118] = (img)(_n2##x,_p2##y,z,v)), \
 (I[136] = (img)(_n2##x,_p1##y,z,v)), \
 (I[154] = (img)(_n2##x,y,z,v)), \
 (I[172] = (img)(_n2##x,_n1##y,z,v)), \
 (I[190] = (img)(_n2##x,_n2##y,z,v)), \
 (I[208] = (img)(_n2##x,_n3##y,z,v)), \
 (I[226] = (img)(_n2##x,_n4##y,z,v)), \
 (I[244] = (img)(_n2##x,_n5##y,z,v)), \
 (I[262] = (img)(_n2##x,_n6##y,z,v)), \
 (I[280] = (img)(_n2##x,_n7##y,z,v)), \
 (I[298] = (img)(_n2##x,_n8##y,z,v)), \
 (I[316] = (img)(_n2##x,_n9##y,z,v)), \
 (I[11] = (img)(_n3##x,_p8##y,z,v)), \
 (I[29] = (img)(_n3##x,_p7##y,z,v)), \
 (I[47] = (img)(_n3##x,_p6##y,z,v)), \
 (I[65] = (img)(_n3##x,_p5##y,z,v)), \
 (I[83] = (img)(_n3##x,_p4##y,z,v)), \
 (I[101] = (img)(_n3##x,_p3##y,z,v)), \
 (I[119] = (img)(_n3##x,_p2##y,z,v)), \
 (I[137] = (img)(_n3##x,_p1##y,z,v)), \
 (I[155] = (img)(_n3##x,y,z,v)), \
 (I[173] = (img)(_n3##x,_n1##y,z,v)), \
 (I[191] = (img)(_n3##x,_n2##y,z,v)), \
 (I[209] = (img)(_n3##x,_n3##y,z,v)), \
 (I[227] = (img)(_n3##x,_n4##y,z,v)), \
 (I[245] = (img)(_n3##x,_n5##y,z,v)), \
 (I[263] = (img)(_n3##x,_n6##y,z,v)), \
 (I[281] = (img)(_n3##x,_n7##y,z,v)), \
 (I[299] = (img)(_n3##x,_n8##y,z,v)), \
 (I[317] = (img)(_n3##x,_n9##y,z,v)), \
 (I[12] = (img)(_n4##x,_p8##y,z,v)), \
 (I[30] = (img)(_n4##x,_p7##y,z,v)), \
 (I[48] = (img)(_n4##x,_p6##y,z,v)), \
 (I[66] = (img)(_n4##x,_p5##y,z,v)), \
 (I[84] = (img)(_n4##x,_p4##y,z,v)), \
 (I[102] = (img)(_n4##x,_p3##y,z,v)), \
 (I[120] = (img)(_n4##x,_p2##y,z,v)), \
 (I[138] = (img)(_n4##x,_p1##y,z,v)), \
 (I[156] = (img)(_n4##x,y,z,v)), \
 (I[174] = (img)(_n4##x,_n1##y,z,v)), \
 (I[192] = (img)(_n4##x,_n2##y,z,v)), \
 (I[210] = (img)(_n4##x,_n3##y,z,v)), \
 (I[228] = (img)(_n4##x,_n4##y,z,v)), \
 (I[246] = (img)(_n4##x,_n5##y,z,v)), \
 (I[264] = (img)(_n4##x,_n6##y,z,v)), \
 (I[282] = (img)(_n4##x,_n7##y,z,v)), \
 (I[300] = (img)(_n4##x,_n8##y,z,v)), \
 (I[318] = (img)(_n4##x,_n9##y,z,v)), \
 (I[13] = (img)(_n5##x,_p8##y,z,v)), \
 (I[31] = (img)(_n5##x,_p7##y,z,v)), \
 (I[49] = (img)(_n5##x,_p6##y,z,v)), \
 (I[67] = (img)(_n5##x,_p5##y,z,v)), \
 (I[85] = (img)(_n5##x,_p4##y,z,v)), \
 (I[103] = (img)(_n5##x,_p3##y,z,v)), \
 (I[121] = (img)(_n5##x,_p2##y,z,v)), \
 (I[139] = (img)(_n5##x,_p1##y,z,v)), \
 (I[157] = (img)(_n5##x,y,z,v)), \
 (I[175] = (img)(_n5##x,_n1##y,z,v)), \
 (I[193] = (img)(_n5##x,_n2##y,z,v)), \
 (I[211] = (img)(_n5##x,_n3##y,z,v)), \
 (I[229] = (img)(_n5##x,_n4##y,z,v)), \
 (I[247] = (img)(_n5##x,_n5##y,z,v)), \
 (I[265] = (img)(_n5##x,_n6##y,z,v)), \
 (I[283] = (img)(_n5##x,_n7##y,z,v)), \
 (I[301] = (img)(_n5##x,_n8##y,z,v)), \
 (I[319] = (img)(_n5##x,_n9##y,z,v)), \
 (I[14] = (img)(_n6##x,_p8##y,z,v)), \
 (I[32] = (img)(_n6##x,_p7##y,z,v)), \
 (I[50] = (img)(_n6##x,_p6##y,z,v)), \
 (I[68] = (img)(_n6##x,_p5##y,z,v)), \
 (I[86] = (img)(_n6##x,_p4##y,z,v)), \
 (I[104] = (img)(_n6##x,_p3##y,z,v)), \
 (I[122] = (img)(_n6##x,_p2##y,z,v)), \
 (I[140] = (img)(_n6##x,_p1##y,z,v)), \
 (I[158] = (img)(_n6##x,y,z,v)), \
 (I[176] = (img)(_n6##x,_n1##y,z,v)), \
 (I[194] = (img)(_n6##x,_n2##y,z,v)), \
 (I[212] = (img)(_n6##x,_n3##y,z,v)), \
 (I[230] = (img)(_n6##x,_n4##y,z,v)), \
 (I[248] = (img)(_n6##x,_n5##y,z,v)), \
 (I[266] = (img)(_n6##x,_n6##y,z,v)), \
 (I[284] = (img)(_n6##x,_n7##y,z,v)), \
 (I[302] = (img)(_n6##x,_n8##y,z,v)), \
 (I[320] = (img)(_n6##x,_n9##y,z,v)), \
 (I[15] = (img)(_n7##x,_p8##y,z,v)), \
 (I[33] = (img)(_n7##x,_p7##y,z,v)), \
 (I[51] = (img)(_n7##x,_p6##y,z,v)), \
 (I[69] = (img)(_n7##x,_p5##y,z,v)), \
 (I[87] = (img)(_n7##x,_p4##y,z,v)), \
 (I[105] = (img)(_n7##x,_p3##y,z,v)), \
 (I[123] = (img)(_n7##x,_p2##y,z,v)), \
 (I[141] = (img)(_n7##x,_p1##y,z,v)), \
 (I[159] = (img)(_n7##x,y,z,v)), \
 (I[177] = (img)(_n7##x,_n1##y,z,v)), \
 (I[195] = (img)(_n7##x,_n2##y,z,v)), \
 (I[213] = (img)(_n7##x,_n3##y,z,v)), \
 (I[231] = (img)(_n7##x,_n4##y,z,v)), \
 (I[249] = (img)(_n7##x,_n5##y,z,v)), \
 (I[267] = (img)(_n7##x,_n6##y,z,v)), \
 (I[285] = (img)(_n7##x,_n7##y,z,v)), \
 (I[303] = (img)(_n7##x,_n8##y,z,v)), \
 (I[321] = (img)(_n7##x,_n9##y,z,v)), \
 (I[16] = (img)(_n8##x,_p8##y,z,v)), \
 (I[34] = (img)(_n8##x,_p7##y,z,v)), \
 (I[52] = (img)(_n8##x,_p6##y,z,v)), \
 (I[70] = (img)(_n8##x,_p5##y,z,v)), \
 (I[88] = (img)(_n8##x,_p4##y,z,v)), \
 (I[106] = (img)(_n8##x,_p3##y,z,v)), \
 (I[124] = (img)(_n8##x,_p2##y,z,v)), \
 (I[142] = (img)(_n8##x,_p1##y,z,v)), \
 (I[160] = (img)(_n8##x,y,z,v)), \
 (I[178] = (img)(_n8##x,_n1##y,z,v)), \
 (I[196] = (img)(_n8##x,_n2##y,z,v)), \
 (I[214] = (img)(_n8##x,_n3##y,z,v)), \
 (I[232] = (img)(_n8##x,_n4##y,z,v)), \
 (I[250] = (img)(_n8##x,_n5##y,z,v)), \
 (I[268] = (img)(_n8##x,_n6##y,z,v)), \
 (I[286] = (img)(_n8##x,_n7##y,z,v)), \
 (I[304] = (img)(_n8##x,_n8##y,z,v)), \
 (I[322] = (img)(_n8##x,_n9##y,z,v)), \
 x+9>=(int)((img).width)?(int)((img).width)-1:x+9); \
 x<=(int)(x1) && ((_n9##x<(int)((img).width) && ( \
 (I[17] = (img)(_n9##x,_p8##y,z,v)), \
 (I[35] = (img)(_n9##x,_p7##y,z,v)), \
 (I[53] = (img)(_n9##x,_p6##y,z,v)), \
 (I[71] = (img)(_n9##x,_p5##y,z,v)), \
 (I[89] = (img)(_n9##x,_p4##y,z,v)), \
 (I[107] = (img)(_n9##x,_p3##y,z,v)), \
 (I[125] = (img)(_n9##x,_p2##y,z,v)), \
 (I[143] = (img)(_n9##x,_p1##y,z,v)), \
 (I[161] = (img)(_n9##x,y,z,v)), \
 (I[179] = (img)(_n9##x,_n1##y,z,v)), \
 (I[197] = (img)(_n9##x,_n2##y,z,v)), \
 (I[215] = (img)(_n9##x,_n3##y,z,v)), \
 (I[233] = (img)(_n9##x,_n4##y,z,v)), \
 (I[251] = (img)(_n9##x,_n5##y,z,v)), \
 (I[269] = (img)(_n9##x,_n6##y,z,v)), \
 (I[287] = (img)(_n9##x,_n7##y,z,v)), \
 (I[305] = (img)(_n9##x,_n8##y,z,v)), \
 (I[323] = (img)(_n9##x,_n9##y,z,v)),1)) || \
 _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], \
 I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], \
 I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], \
 I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], \
 I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], \
 I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], \
 I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], \
 I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], \
 I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], \
 I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], \
 I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], \
 I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], \
 I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], \
 I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], \
 I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], \
 I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], \
 I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], I[321] = I[322], I[322] = I[323], \
 _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x)

#define cimg_get18x18(img,x,y,z,v,I) \
 I[0] = (img)(_p8##x,_p8##y,z,v), I[1] = (img)(_p7##x,_p8##y,z,v), I[2] = (img)(_p6##x,_p8##y,z,v), I[3] = (img)(_p5##x,_p8##y,z,v), I[4] = (img)(_p4##x,_p8##y,z,v), I[5] = (img)(_p3##x,_p8##y,z,v), I[6] = (img)(_p2##x,_p8##y,z,v), I[7] = (img)(_p1##x,_p8##y,z,v), I[8] = (img)(x,_p8##y,z,v), I[9] = (img)(_n1##x,_p8##y,z,v), I[10] = (img)(_n2##x,_p8##y,z,v), I[11] = (img)(_n3##x,_p8##y,z,v), I[12] = (img)(_n4##x,_p8##y,z,v), I[13] = (img)(_n5##x,_p8##y,z,v), I[14] = (img)(_n6##x,_p8##y,z,v), I[15] = (img)(_n7##x,_p8##y,z,v), I[16] = (img)(_n8##x,_p8##y,z,v), I[17] = (img)(_n9##x,_p8##y,z,v), \
 I[18] = (img)(_p8##x,_p7##y,z,v), I[19] = (img)(_p7##x,_p7##y,z,v), I[20] = (img)(_p6##x,_p7##y,z,v), I[21] = (img)(_p5##x,_p7##y,z,v), I[22] = (img)(_p4##x,_p7##y,z,v), I[23] = (img)(_p3##x,_p7##y,z,v), I[24] = (img)(_p2##x,_p7##y,z,v), I[25] = (img)(_p1##x,_p7##y,z,v), I[26] = (img)(x,_p7##y,z,v), I[27] = (img)(_n1##x,_p7##y,z,v), I[28] = (img)(_n2##x,_p7##y,z,v), I[29] = (img)(_n3##x,_p7##y,z,v), I[30] = (img)(_n4##x,_p7##y,z,v), I[31] = (img)(_n5##x,_p7##y,z,v), I[32] = (img)(_n6##x,_p7##y,z,v), I[33] = (img)(_n7##x,_p7##y,z,v), I[34] = (img)(_n8##x,_p7##y,z,v), I[35] = (img)(_n9##x,_p7##y,z,v), \
 I[36] = (img)(_p8##x,_p6##y,z,v), I[37] = (img)(_p7##x,_p6##y,z,v), I[38] = (img)(_p6##x,_p6##y,z,v), I[39] = (img)(_p5##x,_p6##y,z,v), I[40] = (img)(_p4##x,_p6##y,z,v), I[41] = (img)(_p3##x,_p6##y,z,v), I[42] = (img)(_p2##x,_p6##y,z,v), I[43] = (img)(_p1##x,_p6##y,z,v), I[44] = (img)(x,_p6##y,z,v), I[45] = (img)(_n1##x,_p6##y,z,v), I[46] = (img)(_n2##x,_p6##y,z,v), I[47] = (img)(_n3##x,_p6##y,z,v), I[48] = (img)(_n4##x,_p6##y,z,v), I[49] = (img)(_n5##x,_p6##y,z,v), I[50] = (img)(_n6##x,_p6##y,z,v), I[51] = (img)(_n7##x,_p6##y,z,v), I[52] = (img)(_n8##x,_p6##y,z,v), I[53] = (img)(_n9##x,_p6##y,z,v), \
 I[54] = (img)(_p8##x,_p5##y,z,v), I[55] = (img)(_p7##x,_p5##y,z,v), I[56] = (img)(_p6##x,_p5##y,z,v), I[57] = (img)(_p5##x,_p5##y,z,v), I[58] = (img)(_p4##x,_p5##y,z,v), I[59] = (img)(_p3##x,_p5##y,z,v), I[60] = (img)(_p2##x,_p5##y,z,v), I[61] = (img)(_p1##x,_p5##y,z,v), I[62] = (img)(x,_p5##y,z,v), I[63] = (img)(_n1##x,_p5##y,z,v), I[64] = (img)(_n2##x,_p5##y,z,v), I[65] = (img)(_n3##x,_p5##y,z,v), I[66] = (img)(_n4##x,_p5##y,z,v), I[67] = (img)(_n5##x,_p5##y,z,v), I[68] = (img)(_n6##x,_p5##y,z,v), I[69] = (img)(_n7##x,_p5##y,z,v), I[70] = (img)(_n8##x,_p5##y,z,v), I[71] = (img)(_n9##x,_p5##y,z,v), \
 I[72] = (img)(_p8##x,_p4##y,z,v), I[73] = (img)(_p7##x,_p4##y,z,v), I[74] = (img)(_p6##x,_p4##y,z,v), I[75] = (img)(_p5##x,_p4##y,z,v), I[76] = (img)(_p4##x,_p4##y,z,v), I[77] = (img)(_p3##x,_p4##y,z,v), I[78] = (img)(_p2##x,_p4##y,z,v), I[79] = (img)(_p1##x,_p4##y,z,v), I[80] = (img)(x,_p4##y,z,v), I[81] = (img)(_n1##x,_p4##y,z,v), I[82] = (img)(_n2##x,_p4##y,z,v), I[83] = (img)(_n3##x,_p4##y,z,v), I[84] = (img)(_n4##x,_p4##y,z,v), I[85] = (img)(_n5##x,_p4##y,z,v), I[86] = (img)(_n6##x,_p4##y,z,v), I[87] = (img)(_n7##x,_p4##y,z,v), I[88] = (img)(_n8##x,_p4##y,z,v), I[89] = (img)(_n9##x,_p4##y,z,v), \
 I[90] = (img)(_p8##x,_p3##y,z,v), I[91] = (img)(_p7##x,_p3##y,z,v), I[92] = (img)(_p6##x,_p3##y,z,v), I[93] = (img)(_p5##x,_p3##y,z,v), I[94] = (img)(_p4##x,_p3##y,z,v), I[95] = (img)(_p3##x,_p3##y,z,v), I[96] = (img)(_p2##x,_p3##y,z,v), I[97] = (img)(_p1##x,_p3##y,z,v), I[98] = (img)(x,_p3##y,z,v), I[99] = (img)(_n1##x,_p3##y,z,v), I[100] = (img)(_n2##x,_p3##y,z,v), I[101] = (img)(_n3##x,_p3##y,z,v), I[102] = (img)(_n4##x,_p3##y,z,v), I[103] = (img)(_n5##x,_p3##y,z,v), I[104] = (img)(_n6##x,_p3##y,z,v), I[105] = (img)(_n7##x,_p3##y,z,v), I[106] = (img)(_n8##x,_p3##y,z,v), I[107] = (img)(_n9##x,_p3##y,z,v), \
 I[108] = (img)(_p8##x,_p2##y,z,v), I[109] = (img)(_p7##x,_p2##y,z,v), I[110] = (img)(_p6##x,_p2##y,z,v), I[111] = (img)(_p5##x,_p2##y,z,v), I[112] = (img)(_p4##x,_p2##y,z,v), I[113] = (img)(_p3##x,_p2##y,z,v), I[114] = (img)(_p2##x,_p2##y,z,v), I[115] = (img)(_p1##x,_p2##y,z,v), I[116] = (img)(x,_p2##y,z,v), I[117] = (img)(_n1##x,_p2##y,z,v), I[118] = (img)(_n2##x,_p2##y,z,v), I[119] = (img)(_n3##x,_p2##y,z,v), I[120] = (img)(_n4##x,_p2##y,z,v), I[121] = (img)(_n5##x,_p2##y,z,v), I[122] = (img)(_n6##x,_p2##y,z,v), I[123] = (img)(_n7##x,_p2##y,z,v), I[124] = (img)(_n8##x,_p2##y,z,v), I[125] = (img)(_n9##x,_p2##y,z,v), \
 I[126] = (img)(_p8##x,_p1##y,z,v), I[127] = (img)(_p7##x,_p1##y,z,v), I[128] = (img)(_p6##x,_p1##y,z,v), I[129] = (img)(_p5##x,_p1##y,z,v), I[130] = (img)(_p4##x,_p1##y,z,v), I[131] = (img)(_p3##x,_p1##y,z,v), I[132] = (img)(_p2##x,_p1##y,z,v), I[133] = (img)(_p1##x,_p1##y,z,v), I[134] = (img)(x,_p1##y,z,v), I[135] = (img)(_n1##x,_p1##y,z,v), I[136] = (img)(_n2##x,_p1##y,z,v), I[137] = (img)(_n3##x,_p1##y,z,v), I[138] = (img)(_n4##x,_p1##y,z,v), I[139] = (img)(_n5##x,_p1##y,z,v), I[140] = (img)(_n6##x,_p1##y,z,v), I[141] = (img)(_n7##x,_p1##y,z,v), I[142] = (img)(_n8##x,_p1##y,z,v), I[143] = (img)(_n9##x,_p1##y,z,v), \
 I[144] = (img)(_p8##x,y,z,v), I[145] = (img)(_p7##x,y,z,v), I[146] = (img)(_p6##x,y,z,v), I[147] = (img)(_p5##x,y,z,v), I[148] = (img)(_p4##x,y,z,v), I[149] = (img)(_p3##x,y,z,v), I[150] = (img)(_p2##x,y,z,v), I[151] = (img)(_p1##x,y,z,v), I[152] = (img)(x,y,z,v), I[153] = (img)(_n1##x,y,z,v), I[154] = (img)(_n2##x,y,z,v), I[155] = (img)(_n3##x,y,z,v), I[156] = (img)(_n4##x,y,z,v), I[157] = (img)(_n5##x,y,z,v), I[158] = (img)(_n6##x,y,z,v), I[159] = (img)(_n7##x,y,z,v), I[160] = (img)(_n8##x,y,z,v), I[161] = (img)(_n9##x,y,z,v), \
 I[162] = (img)(_p8##x,_n1##y,z,v), I[163] = (img)(_p7##x,_n1##y,z,v), I[164] = (img)(_p6##x,_n1##y,z,v), I[165] = (img)(_p5##x,_n1##y,z,v), I[166] = (img)(_p4##x,_n1##y,z,v), I[167] = (img)(_p3##x,_n1##y,z,v), I[168] = (img)(_p2##x,_n1##y,z,v), I[169] = (img)(_p1##x,_n1##y,z,v), I[170] = (img)(x,_n1##y,z,v), I[171] = (img)(_n1##x,_n1##y,z,v), I[172] = (img)(_n2##x,_n1##y,z,v), I[173] = (img)(_n3##x,_n1##y,z,v), I[174] = (img)(_n4##x,_n1##y,z,v), I[175] = (img)(_n5##x,_n1##y,z,v), I[176] = (img)(_n6##x,_n1##y,z,v), I[177] = (img)(_n7##x,_n1##y,z,v), I[178] = (img)(_n8##x,_n1##y,z,v), I[179] = (img)(_n9##x,_n1##y,z,v), \
 I[180] = (img)(_p8##x,_n2##y,z,v), I[181] = (img)(_p7##x,_n2##y,z,v), I[182] = (img)(_p6##x,_n2##y,z,v), I[183] = (img)(_p5##x,_n2##y,z,v), I[184] = (img)(_p4##x,_n2##y,z,v), I[185] = (img)(_p3##x,_n2##y,z,v), I[186] = (img)(_p2##x,_n2##y,z,v), I[187] = (img)(_p1##x,_n2##y,z,v), I[188] = (img)(x,_n2##y,z,v), I[189] = (img)(_n1##x,_n2##y,z,v), I[190] = (img)(_n2##x,_n2##y,z,v), I[191] = (img)(_n3##x,_n2##y,z,v), I[192] = (img)(_n4##x,_n2##y,z,v), I[193] = (img)(_n5##x,_n2##y,z,v), I[194] = (img)(_n6##x,_n2##y,z,v), I[195] = (img)(_n7##x,_n2##y,z,v), I[196] = (img)(_n8##x,_n2##y,z,v), I[197] = (img)(_n9##x,_n2##y,z,v), \
 I[198] = (img)(_p8##x,_n3##y,z,v), I[199] = (img)(_p7##x,_n3##y,z,v), I[200] = (img)(_p6##x,_n3##y,z,v), I[201] = (img)(_p5##x,_n3##y,z,v), I[202] = (img)(_p4##x,_n3##y,z,v), I[203] = (img)(_p3##x,_n3##y,z,v), I[204] = (img)(_p2##x,_n3##y,z,v), I[205] = (img)(_p1##x,_n3##y,z,v), I[206] = (img)(x,_n3##y,z,v), I[207] = (img)(_n1##x,_n3##y,z,v), I[208] = (img)(_n2##x,_n3##y,z,v), I[209] = (img)(_n3##x,_n3##y,z,v), I[210] = (img)(_n4##x,_n3##y,z,v), I[211] = (img)(_n5##x,_n3##y,z,v), I[212] = (img)(_n6##x,_n3##y,z,v), I[213] = (img)(_n7##x,_n3##y,z,v), I[214] = (img)(_n8##x,_n3##y,z,v), I[215] = (img)(_n9##x,_n3##y,z,v), \
 I[216] = (img)(_p8##x,_n4##y,z,v), I[217] = (img)(_p7##x,_n4##y,z,v), I[218] = (img)(_p6##x,_n4##y,z,v), I[219] = (img)(_p5##x,_n4##y,z,v), I[220] = (img)(_p4##x,_n4##y,z,v), I[221] = (img)(_p3##x,_n4##y,z,v), I[222] = (img)(_p2##x,_n4##y,z,v), I[223] = (img)(_p1##x,_n4##y,z,v), I[224] = (img)(x,_n4##y,z,v), I[225] = (img)(_n1##x,_n4##y,z,v), I[226] = (img)(_n2##x,_n4##y,z,v), I[227] = (img)(_n3##x,_n4##y,z,v), I[228] = (img)(_n4##x,_n4##y,z,v), I[229] = (img)(_n5##x,_n4##y,z,v), I[230] = (img)(_n6##x,_n4##y,z,v), I[231] = (img)(_n7##x,_n4##y,z,v), I[232] = (img)(_n8##x,_n4##y,z,v), I[233] = (img)(_n9##x,_n4##y,z,v), \
 I[234] = (img)(_p8##x,_n5##y,z,v), I[235] = (img)(_p7##x,_n5##y,z,v), I[236] = (img)(_p6##x,_n5##y,z,v), I[237] = (img)(_p5##x,_n5##y,z,v), I[238] = (img)(_p4##x,_n5##y,z,v), I[239] = (img)(_p3##x,_n5##y,z,v), I[240] = (img)(_p2##x,_n5##y,z,v), I[241] = (img)(_p1##x,_n5##y,z,v), I[242] = (img)(x,_n5##y,z,v), I[243] = (img)(_n1##x,_n5##y,z,v), I[244] = (img)(_n2##x,_n5##y,z,v), I[245] = (img)(_n3##x,_n5##y,z,v), I[246] = (img)(_n4##x,_n5##y,z,v), I[247] = (img)(_n5##x,_n5##y,z,v), I[248] = (img)(_n6##x,_n5##y,z,v), I[249] = (img)(_n7##x,_n5##y,z,v), I[250] = (img)(_n8##x,_n5##y,z,v), I[251] = (img)(_n9##x,_n5##y,z,v), \
 I[252] = (img)(_p8##x,_n6##y,z,v), I[253] = (img)(_p7##x,_n6##y,z,v), I[254] = (img)(_p6##x,_n6##y,z,v), I[255] = (img)(_p5##x,_n6##y,z,v), I[256] = (img)(_p4##x,_n6##y,z,v), I[257] = (img)(_p3##x,_n6##y,z,v), I[258] = (img)(_p2##x,_n6##y,z,v), I[259] = (img)(_p1##x,_n6##y,z,v), I[260] = (img)(x,_n6##y,z,v), I[261] = (img)(_n1##x,_n6##y,z,v), I[262] = (img)(_n2##x,_n6##y,z,v), I[263] = (img)(_n3##x,_n6##y,z,v), I[264] = (img)(_n4##x,_n6##y,z,v), I[265] = (img)(_n5##x,_n6##y,z,v), I[266] = (img)(_n6##x,_n6##y,z,v), I[267] = (img)(_n7##x,_n6##y,z,v), I[268] = (img)(_n8##x,_n6##y,z,v), I[269] = (img)(_n9##x,_n6##y,z,v), \
 I[270] = (img)(_p8##x,_n7##y,z,v), I[271] = (img)(_p7##x,_n7##y,z,v), I[272] = (img)(_p6##x,_n7##y,z,v), I[273] = (img)(_p5##x,_n7##y,z,v), I[274] = (img)(_p4##x,_n7##y,z,v), I[275] = (img)(_p3##x,_n7##y,z,v), I[276] = (img)(_p2##x,_n7##y,z,v), I[277] = (img)(_p1##x,_n7##y,z,v), I[278] = (img)(x,_n7##y,z,v), I[279] = (img)(_n1##x,_n7##y,z,v), I[280] = (img)(_n2##x,_n7##y,z,v), I[281] = (img)(_n3##x,_n7##y,z,v), I[282] = (img)(_n4##x,_n7##y,z,v), I[283] = (img)(_n5##x,_n7##y,z,v), I[284] = (img)(_n6##x,_n7##y,z,v), I[285] = (img)(_n7##x,_n7##y,z,v), I[286] = (img)(_n8##x,_n7##y,z,v), I[287] = (img)(_n9##x,_n7##y,z,v), \
 I[288] = (img)(_p8##x,_n8##y,z,v), I[289] = (img)(_p7##x,_n8##y,z,v), I[290] = (img)(_p6##x,_n8##y,z,v), I[291] = (img)(_p5##x,_n8##y,z,v), I[292] = (img)(_p4##x,_n8##y,z,v), I[293] = (img)(_p3##x,_n8##y,z,v), I[294] = (img)(_p2##x,_n8##y,z,v), I[295] = (img)(_p1##x,_n8##y,z,v), I[296] = (img)(x,_n8##y,z,v), I[297] = (img)(_n1##x,_n8##y,z,v), I[298] = (img)(_n2##x,_n8##y,z,v), I[299] = (img)(_n3##x,_n8##y,z,v), I[300] = (img)(_n4##x,_n8##y,z,v), I[301] = (img)(_n5##x,_n8##y,z,v), I[302] = (img)(_n6##x,_n8##y,z,v), I[303] = (img)(_n7##x,_n8##y,z,v), I[304] = (img)(_n8##x,_n8##y,z,v), I[305] = (img)(_n9##x,_n8##y,z,v), \
 I[306] = (img)(_p8##x,_n9##y,z,v), I[307] = (img)(_p7##x,_n9##y,z,v), I[308] = (img)(_p6##x,_n9##y,z,v), I[309] = (img)(_p5##x,_n9##y,z,v), I[310] = (img)(_p4##x,_n9##y,z,v), I[311] = (img)(_p3##x,_n9##y,z,v), I[312] = (img)(_p2##x,_n9##y,z,v), I[313] = (img)(_p1##x,_n9##y,z,v), I[314] = (img)(x,_n9##y,z,v), I[315] = (img)(_n1##x,_n9##y,z,v), I[316] = (img)(_n2##x,_n9##y,z,v), I[317] = (img)(_n3##x,_n9##y,z,v), I[318] = (img)(_n4##x,_n9##y,z,v), I[319] = (img)(_n5##x,_n9##y,z,v), I[320] = (img)(_n6##x,_n9##y,z,v), I[321] = (img)(_n7##x,_n9##y,z,v), I[322] = (img)(_n8##x,_n9##y,z,v), I[323] = (img)(_n9##x,_n9##y,z,v);

// Define 19x19 loop macros for CImg
//----------------------------------
#define cimg_for19(bound,i) for (int i = 0, \
 _p9##i = 0, _p8##i = 0, _p7##i = 0, _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7, \
 _n8##i = 8>=(int)(bound)?(int)(bound)-1:8, \
 _n9##i = 9>=(int)(bound)?(int)(bound)-1:9; \
 _n9##i<(int)(bound) || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i)

#define cimg_for19X(img,x) cimg_for19((img).width,x)
#define cimg_for19Y(img,y) cimg_for19((img).height,y)
#define cimg_for19Z(img,z) cimg_for19((img).depth,z)
#define cimg_for19V(img,v) cimg_for19((img).dim,v)
#define cimg_for19XY(img,x,y) cimg_for19Y(img,y) cimg_for19X(img,x)
#define cimg_for19XZ(img,x,z) cimg_for19Z(img,z) cimg_for19X(img,x)
#define cimg_for19XV(img,x,v) cimg_for19V(img,v) cimg_for19X(img,x)
#define cimg_for19YZ(img,y,z) cimg_for19Z(img,z) cimg_for19Y(img,y)
#define cimg_for19YV(img,y,v) cimg_for19V(img,v) cimg_for19Y(img,y)
#define cimg_for19ZV(img,z,v) cimg_for19V(img,v) cimg_for19Z(img,z)
#define cimg_for19XYZ(img,x,y,z) cimg_for19Z(img,z) cimg_for19XY(img,x,y)
#define cimg_for19XZV(img,x,z,v) cimg_for19V(img,v) cimg_for19XZ(img,x,z)
#define cimg_for19YZV(img,y,z,v) cimg_for19V(img,v) cimg_for19YZ(img,y,z)
#define cimg_for19XYZV(img,x,y,z,v) cimg_for19V(img,v) cimg_for19XYZ(img,x,y,z)

#define cimg_for_in19(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p9##i = i-9<0?0:i-9, \
 _p8##i = i-8<0?0:i-8, \
 _p7##i = i-7<0?0:i-7, \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7, \
 _n8##i = i+8>=(int)(bound)?(int)(bound)-1:i+8, \
 _n9##i = i+9>=(int)(bound)?(int)(bound)-1:i+9; \
 i<=(int)(i1) && (_n9##i<(int)(bound) || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i)

#define cimg_for_in19X(img,x0,x1,x) cimg_for_in19((img).width,x0,x1,x)
#define cimg_for_in19Y(img,y0,y1,y) cimg_for_in19((img).height,y0,y1,y)
#define cimg_for_in19Z(img,z0,z1,z) cimg_for_in19((img).depth,z0,z1,z)
#define cimg_for_in19V(img,v0,v1,v) cimg_for_in19((img).dim,v0,v1,v)
#define cimg_for_in19XY(img,x0,y0,x1,y1,x,y) cimg_for_in19Y(img,y0,y1,y) cimg_for_in19X(img,x0,x1,x)
#define cimg_for_in19XZ(img,x0,z0,x1,z1,x,z) cimg_for_in19Z(img,z0,z1,z) cimg_for_in19X(img,x0,x1,x)
#define cimg_for_in19XV(img,x0,v0,x1,v1,x,v) cimg_for_in19V(img,v0,v1,v) cimg_for_in19X(img,x0,x1,x)
#define cimg_for_in19YZ(img,y0,z0,y1,z1,y,z) cimg_for_in19Z(img,z0,z1,z) cimg_for_in19Y(img,y0,y1,y)
#define cimg_for_in19YV(img,y0,v0,y1,v1,y,v) cimg_for_in19V(img,v0,v1,v) cimg_for_in19Y(img,y0,y1,y)
#define cimg_for_in19ZV(img,z0,v0,z1,v1,z,v) cimg_for_in19V(img,v0,v1,v) cimg_for_in19Z(img,z0,z1,z)
#define cimg_for_in19XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in19Z(img,z0,z1,z) cimg_for_in19XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in19XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in19V(img,v0,v1,v) cimg_for_in19XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in19YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in19V(img,v0,v1,v) cimg_for_in19YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in19XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in19V(img,v0,v1,v) cimg_for_in19XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for19x19(img,x,y,z,v,I) \
 cimg_for19((img).height,y) for (int x = 0, \
 _p9##x = 0, _p8##x = 0, _p7##x = 0, _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = 7>=((img).width)?(int)((img).width)-1:7, \
 _n8##x = 8>=((img).width)?(int)((img).width)-1:8, \
 _n9##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = I[7] = I[8] = I[9] = (img)(0,_p9##y,z,v)), \
 (I[19] = I[20] = I[21] = I[22] = I[23] = I[24] = I[25] = I[26] = I[27] = I[28] = (img)(0,_p8##y,z,v)), \
 (I[38] = I[39] = I[40] = I[41] = I[42] = I[43] = I[44] = I[45] = I[46] = I[47] = (img)(0,_p7##y,z,v)), \
 (I[57] = I[58] = I[59] = I[60] = I[61] = I[62] = I[63] = I[64] = I[65] = I[66] = (img)(0,_p6##y,z,v)), \
 (I[76] = I[77] = I[78] = I[79] = I[80] = I[81] = I[82] = I[83] = I[84] = I[85] = (img)(0,_p5##y,z,v)), \
 (I[95] = I[96] = I[97] = I[98] = I[99] = I[100] = I[101] = I[102] = I[103] = I[104] = (img)(0,_p4##y,z,v)), \
 (I[114] = I[115] = I[116] = I[117] = I[118] = I[119] = I[120] = I[121] = I[122] = I[123] = (img)(0,_p3##y,z,v)), \
 (I[133] = I[134] = I[135] = I[136] = I[137] = I[138] = I[139] = I[140] = I[141] = I[142] = (img)(0,_p2##y,z,v)), \
 (I[152] = I[153] = I[154] = I[155] = I[156] = I[157] = I[158] = I[159] = I[160] = I[161] = (img)(0,_p1##y,z,v)), \
 (I[171] = I[172] = I[173] = I[174] = I[175] = I[176] = I[177] = I[178] = I[179] = I[180] = (img)(0,y,z,v)), \
 (I[190] = I[191] = I[192] = I[193] = I[194] = I[195] = I[196] = I[197] = I[198] = I[199] = (img)(0,_n1##y,z,v)), \
 (I[209] = I[210] = I[211] = I[212] = I[213] = I[214] = I[215] = I[216] = I[217] = I[218] = (img)(0,_n2##y,z,v)), \
 (I[228] = I[229] = I[230] = I[231] = I[232] = I[233] = I[234] = I[235] = I[236] = I[237] = (img)(0,_n3##y,z,v)), \
 (I[247] = I[248] = I[249] = I[250] = I[251] = I[252] = I[253] = I[254] = I[255] = I[256] = (img)(0,_n4##y,z,v)), \
 (I[266] = I[267] = I[268] = I[269] = I[270] = I[271] = I[272] = I[273] = I[274] = I[275] = (img)(0,_n5##y,z,v)), \
 (I[285] = I[286] = I[287] = I[288] = I[289] = I[290] = I[291] = I[292] = I[293] = I[294] = (img)(0,_n6##y,z,v)), \
 (I[304] = I[305] = I[306] = I[307] = I[308] = I[309] = I[310] = I[311] = I[312] = I[313] = (img)(0,_n7##y,z,v)), \
 (I[323] = I[324] = I[325] = I[326] = I[327] = I[328] = I[329] = I[330] = I[331] = I[332] = (img)(0,_n8##y,z,v)), \
 (I[342] = I[343] = I[344] = I[345] = I[346] = I[347] = I[348] = I[349] = I[350] = I[351] = (img)(0,_n9##y,z,v)), \
 (I[10] = (img)(_n1##x,_p9##y,z,v)), \
 (I[29] = (img)(_n1##x,_p8##y,z,v)), \
 (I[48] = (img)(_n1##x,_p7##y,z,v)), \
 (I[67] = (img)(_n1##x,_p6##y,z,v)), \
 (I[86] = (img)(_n1##x,_p5##y,z,v)), \
 (I[105] = (img)(_n1##x,_p4##y,z,v)), \
 (I[124] = (img)(_n1##x,_p3##y,z,v)), \
 (I[143] = (img)(_n1##x,_p2##y,z,v)), \
 (I[162] = (img)(_n1##x,_p1##y,z,v)), \
 (I[181] = (img)(_n1##x,y,z,v)), \
 (I[200] = (img)(_n1##x,_n1##y,z,v)), \
 (I[219] = (img)(_n1##x,_n2##y,z,v)), \
 (I[238] = (img)(_n1##x,_n3##y,z,v)), \
 (I[257] = (img)(_n1##x,_n4##y,z,v)), \
 (I[276] = (img)(_n1##x,_n5##y,z,v)), \
 (I[295] = (img)(_n1##x,_n6##y,z,v)), \
 (I[314] = (img)(_n1##x,_n7##y,z,v)), \
 (I[333] = (img)(_n1##x,_n8##y,z,v)), \
 (I[352] = (img)(_n1##x,_n9##y,z,v)), \
 (I[11] = (img)(_n2##x,_p9##y,z,v)), \
 (I[30] = (img)(_n2##x,_p8##y,z,v)), \
 (I[49] = (img)(_n2##x,_p7##y,z,v)), \
 (I[68] = (img)(_n2##x,_p6##y,z,v)), \
 (I[87] = (img)(_n2##x,_p5##y,z,v)), \
 (I[106] = (img)(_n2##x,_p4##y,z,v)), \
 (I[125] = (img)(_n2##x,_p3##y,z,v)), \
 (I[144] = (img)(_n2##x,_p2##y,z,v)), \
 (I[163] = (img)(_n2##x,_p1##y,z,v)), \
 (I[182] = (img)(_n2##x,y,z,v)), \
 (I[201] = (img)(_n2##x,_n1##y,z,v)), \
 (I[220] = (img)(_n2##x,_n2##y,z,v)), \
 (I[239] = (img)(_n2##x,_n3##y,z,v)), \
 (I[258] = (img)(_n2##x,_n4##y,z,v)), \
 (I[277] = (img)(_n2##x,_n5##y,z,v)), \
 (I[296] = (img)(_n2##x,_n6##y,z,v)), \
 (I[315] = (img)(_n2##x,_n7##y,z,v)), \
 (I[334] = (img)(_n2##x,_n8##y,z,v)), \
 (I[353] = (img)(_n2##x,_n9##y,z,v)), \
 (I[12] = (img)(_n3##x,_p9##y,z,v)), \
 (I[31] = (img)(_n3##x,_p8##y,z,v)), \
 (I[50] = (img)(_n3##x,_p7##y,z,v)), \
 (I[69] = (img)(_n3##x,_p6##y,z,v)), \
 (I[88] = (img)(_n3##x,_p5##y,z,v)), \
 (I[107] = (img)(_n3##x,_p4##y,z,v)), \
 (I[126] = (img)(_n3##x,_p3##y,z,v)), \
 (I[145] = (img)(_n3##x,_p2##y,z,v)), \
 (I[164] = (img)(_n3##x,_p1##y,z,v)), \
 (I[183] = (img)(_n3##x,y,z,v)), \
 (I[202] = (img)(_n3##x,_n1##y,z,v)), \
 (I[221] = (img)(_n3##x,_n2##y,z,v)), \
 (I[240] = (img)(_n3##x,_n3##y,z,v)), \
 (I[259] = (img)(_n3##x,_n4##y,z,v)), \
 (I[278] = (img)(_n3##x,_n5##y,z,v)), \
 (I[297] = (img)(_n3##x,_n6##y,z,v)), \
 (I[316] = (img)(_n3##x,_n7##y,z,v)), \
 (I[335] = (img)(_n3##x,_n8##y,z,v)), \
 (I[354] = (img)(_n3##x,_n9##y,z,v)), \
 (I[13] = (img)(_n4##x,_p9##y,z,v)), \
 (I[32] = (img)(_n4##x,_p8##y,z,v)), \
 (I[51] = (img)(_n4##x,_p7##y,z,v)), \
 (I[70] = (img)(_n4##x,_p6##y,z,v)), \
 (I[89] = (img)(_n4##x,_p5##y,z,v)), \
 (I[108] = (img)(_n4##x,_p4##y,z,v)), \
 (I[127] = (img)(_n4##x,_p3##y,z,v)), \
 (I[146] = (img)(_n4##x,_p2##y,z,v)), \
 (I[165] = (img)(_n4##x,_p1##y,z,v)), \
 (I[184] = (img)(_n4##x,y,z,v)), \
 (I[203] = (img)(_n4##x,_n1##y,z,v)), \
 (I[222] = (img)(_n4##x,_n2##y,z,v)), \
 (I[241] = (img)(_n4##x,_n3##y,z,v)), \
 (I[260] = (img)(_n4##x,_n4##y,z,v)), \
 (I[279] = (img)(_n4##x,_n5##y,z,v)), \
 (I[298] = (img)(_n4##x,_n6##y,z,v)), \
 (I[317] = (img)(_n4##x,_n7##y,z,v)), \
 (I[336] = (img)(_n4##x,_n8##y,z,v)), \
 (I[355] = (img)(_n4##x,_n9##y,z,v)), \
 (I[14] = (img)(_n5##x,_p9##y,z,v)), \
 (I[33] = (img)(_n5##x,_p8##y,z,v)), \
 (I[52] = (img)(_n5##x,_p7##y,z,v)), \
 (I[71] = (img)(_n5##x,_p6##y,z,v)), \
 (I[90] = (img)(_n5##x,_p5##y,z,v)), \
 (I[109] = (img)(_n5##x,_p4##y,z,v)), \
 (I[128] = (img)(_n5##x,_p3##y,z,v)), \
 (I[147] = (img)(_n5##x,_p2##y,z,v)), \
 (I[166] = (img)(_n5##x,_p1##y,z,v)), \
 (I[185] = (img)(_n5##x,y,z,v)), \
 (I[204] = (img)(_n5##x,_n1##y,z,v)), \
 (I[223] = (img)(_n5##x,_n2##y,z,v)), \
 (I[242] = (img)(_n5##x,_n3##y,z,v)), \
 (I[261] = (img)(_n5##x,_n4##y,z,v)), \
 (I[280] = (img)(_n5##x,_n5##y,z,v)), \
 (I[299] = (img)(_n5##x,_n6##y,z,v)), \
 (I[318] = (img)(_n5##x,_n7##y,z,v)), \
 (I[337] = (img)(_n5##x,_n8##y,z,v)), \
 (I[356] = (img)(_n5##x,_n9##y,z,v)), \
 (I[15] = (img)(_n6##x,_p9##y,z,v)), \
 (I[34] = (img)(_n6##x,_p8##y,z,v)), \
 (I[53] = (img)(_n6##x,_p7##y,z,v)), \
 (I[72] = (img)(_n6##x,_p6##y,z,v)), \
 (I[91] = (img)(_n6##x,_p5##y,z,v)), \
 (I[110] = (img)(_n6##x,_p4##y,z,v)), \
 (I[129] = (img)(_n6##x,_p3##y,z,v)), \
 (I[148] = (img)(_n6##x,_p2##y,z,v)), \
 (I[167] = (img)(_n6##x,_p1##y,z,v)), \
 (I[186] = (img)(_n6##x,y,z,v)), \
 (I[205] = (img)(_n6##x,_n1##y,z,v)), \
 (I[224] = (img)(_n6##x,_n2##y,z,v)), \
 (I[243] = (img)(_n6##x,_n3##y,z,v)), \
 (I[262] = (img)(_n6##x,_n4##y,z,v)), \
 (I[281] = (img)(_n6##x,_n5##y,z,v)), \
 (I[300] = (img)(_n6##x,_n6##y,z,v)), \
 (I[319] = (img)(_n6##x,_n7##y,z,v)), \
 (I[338] = (img)(_n6##x,_n8##y,z,v)), \
 (I[357] = (img)(_n6##x,_n9##y,z,v)), \
 (I[16] = (img)(_n7##x,_p9##y,z,v)), \
 (I[35] = (img)(_n7##x,_p8##y,z,v)), \
 (I[54] = (img)(_n7##x,_p7##y,z,v)), \
 (I[73] = (img)(_n7##x,_p6##y,z,v)), \
 (I[92] = (img)(_n7##x,_p5##y,z,v)), \
 (I[111] = (img)(_n7##x,_p4##y,z,v)), \
 (I[130] = (img)(_n7##x,_p3##y,z,v)), \
 (I[149] = (img)(_n7##x,_p2##y,z,v)), \
 (I[168] = (img)(_n7##x,_p1##y,z,v)), \
 (I[187] = (img)(_n7##x,y,z,v)), \
 (I[206] = (img)(_n7##x,_n1##y,z,v)), \
 (I[225] = (img)(_n7##x,_n2##y,z,v)), \
 (I[244] = (img)(_n7##x,_n3##y,z,v)), \
 (I[263] = (img)(_n7##x,_n4##y,z,v)), \
 (I[282] = (img)(_n7##x,_n5##y,z,v)), \
 (I[301] = (img)(_n7##x,_n6##y,z,v)), \
 (I[320] = (img)(_n7##x,_n7##y,z,v)), \
 (I[339] = (img)(_n7##x,_n8##y,z,v)), \
 (I[358] = (img)(_n7##x,_n9##y,z,v)), \
 (I[17] = (img)(_n8##x,_p9##y,z,v)), \
 (I[36] = (img)(_n8##x,_p8##y,z,v)), \
 (I[55] = (img)(_n8##x,_p7##y,z,v)), \
 (I[74] = (img)(_n8##x,_p6##y,z,v)), \
 (I[93] = (img)(_n8##x,_p5##y,z,v)), \
 (I[112] = (img)(_n8##x,_p4##y,z,v)), \
 (I[131] = (img)(_n8##x,_p3##y,z,v)), \
 (I[150] = (img)(_n8##x,_p2##y,z,v)), \
 (I[169] = (img)(_n8##x,_p1##y,z,v)), \
 (I[188] = (img)(_n8##x,y,z,v)), \
 (I[207] = (img)(_n8##x,_n1##y,z,v)), \
 (I[226] = (img)(_n8##x,_n2##y,z,v)), \
 (I[245] = (img)(_n8##x,_n3##y,z,v)), \
 (I[264] = (img)(_n8##x,_n4##y,z,v)), \
 (I[283] = (img)(_n8##x,_n5##y,z,v)), \
 (I[302] = (img)(_n8##x,_n6##y,z,v)), \
 (I[321] = (img)(_n8##x,_n7##y,z,v)), \
 (I[340] = (img)(_n8##x,_n8##y,z,v)), \
 (I[359] = (img)(_n8##x,_n9##y,z,v)), \
 9>=((img).width)?(int)((img).width)-1:9); \
 (_n9##x<(int)((img).width) && ( \
 (I[18] = (img)(_n9##x,_p9##y,z,v)), \
 (I[37] = (img)(_n9##x,_p8##y,z,v)), \
 (I[56] = (img)(_n9##x,_p7##y,z,v)), \
 (I[75] = (img)(_n9##x,_p6##y,z,v)), \
 (I[94] = (img)(_n9##x,_p5##y,z,v)), \
 (I[113] = (img)(_n9##x,_p4##y,z,v)), \
 (I[132] = (img)(_n9##x,_p3##y,z,v)), \
 (I[151] = (img)(_n9##x,_p2##y,z,v)), \
 (I[170] = (img)(_n9##x,_p1##y,z,v)), \
 (I[189] = (img)(_n9##x,y,z,v)), \
 (I[208] = (img)(_n9##x,_n1##y,z,v)), \
 (I[227] = (img)(_n9##x,_n2##y,z,v)), \
 (I[246] = (img)(_n9##x,_n3##y,z,v)), \
 (I[265] = (img)(_n9##x,_n4##y,z,v)), \
 (I[284] = (img)(_n9##x,_n5##y,z,v)), \
 (I[303] = (img)(_n9##x,_n6##y,z,v)), \
 (I[322] = (img)(_n9##x,_n7##y,z,v)), \
 (I[341] = (img)(_n9##x,_n8##y,z,v)), \
 (I[360] = (img)(_n9##x,_n9##y,z,v)),1)) || \
 _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], \
 I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], \
 I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], \
 I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], \
 I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], \
 I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], \
 I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], \
 I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], \
 I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], \
 I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], \
 I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], \
 I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], \
 I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], \
 I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], \
 I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], \
 I[285] = I[286], I[286] = I[287], I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], \
 I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], I[321] = I[322], \
 I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], I[335] = I[336], I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], \
 I[342] = I[343], I[343] = I[344], I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], I[351] = I[352], I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], I[359] = I[360], \
 _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x)

#define cimg_for_in19x19(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in19((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p9##x = x-9<0?0:x-9, \
 _p8##x = x-8<0?0:x-8, \
 _p7##x = x-7<0?0:x-7, \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = x+7>=(int)((img).width)?(int)((img).width)-1:x+7, \
 _n8##x = x+8>=(int)((img).width)?(int)((img).width)-1:x+8, \
 _n9##x = (int)( \
 (I[0] = (img)(_p9##x,_p9##y,z,v)), \
 (I[19] = (img)(_p9##x,_p8##y,z,v)), \
 (I[38] = (img)(_p9##x,_p7##y,z,v)), \
 (I[57] = (img)(_p9##x,_p6##y,z,v)), \
 (I[76] = (img)(_p9##x,_p5##y,z,v)), \
 (I[95] = (img)(_p9##x,_p4##y,z,v)), \
 (I[114] = (img)(_p9##x,_p3##y,z,v)), \
 (I[133] = (img)(_p9##x,_p2##y,z,v)), \
 (I[152] = (img)(_p9##x,_p1##y,z,v)), \
 (I[171] = (img)(_p9##x,y,z,v)), \
 (I[190] = (img)(_p9##x,_n1##y,z,v)), \
 (I[209] = (img)(_p9##x,_n2##y,z,v)), \
 (I[228] = (img)(_p9##x,_n3##y,z,v)), \
 (I[247] = (img)(_p9##x,_n4##y,z,v)), \
 (I[266] = (img)(_p9##x,_n5##y,z,v)), \
 (I[285] = (img)(_p9##x,_n6##y,z,v)), \
 (I[304] = (img)(_p9##x,_n7##y,z,v)), \
 (I[323] = (img)(_p9##x,_n8##y,z,v)), \
 (I[342] = (img)(_p9##x,_n9##y,z,v)), \
 (I[1] = (img)(_p8##x,_p9##y,z,v)), \
 (I[20] = (img)(_p8##x,_p8##y,z,v)), \
 (I[39] = (img)(_p8##x,_p7##y,z,v)), \
 (I[58] = (img)(_p8##x,_p6##y,z,v)), \
 (I[77] = (img)(_p8##x,_p5##y,z,v)), \
 (I[96] = (img)(_p8##x,_p4##y,z,v)), \
 (I[115] = (img)(_p8##x,_p3##y,z,v)), \
 (I[134] = (img)(_p8##x,_p2##y,z,v)), \
 (I[153] = (img)(_p8##x,_p1##y,z,v)), \
 (I[172] = (img)(_p8##x,y,z,v)), \
 (I[191] = (img)(_p8##x,_n1##y,z,v)), \
 (I[210] = (img)(_p8##x,_n2##y,z,v)), \
 (I[229] = (img)(_p8##x,_n3##y,z,v)), \
 (I[248] = (img)(_p8##x,_n4##y,z,v)), \
 (I[267] = (img)(_p8##x,_n5##y,z,v)), \
 (I[286] = (img)(_p8##x,_n6##y,z,v)), \
 (I[305] = (img)(_p8##x,_n7##y,z,v)), \
 (I[324] = (img)(_p8##x,_n8##y,z,v)), \
 (I[343] = (img)(_p8##x,_n9##y,z,v)), \
 (I[2] = (img)(_p7##x,_p9##y,z,v)), \
 (I[21] = (img)(_p7##x,_p8##y,z,v)), \
 (I[40] = (img)(_p7##x,_p7##y,z,v)), \
 (I[59] = (img)(_p7##x,_p6##y,z,v)), \
 (I[78] = (img)(_p7##x,_p5##y,z,v)), \
 (I[97] = (img)(_p7##x,_p4##y,z,v)), \
 (I[116] = (img)(_p7##x,_p3##y,z,v)), \
 (I[135] = (img)(_p7##x,_p2##y,z,v)), \
 (I[154] = (img)(_p7##x,_p1##y,z,v)), \
 (I[173] = (img)(_p7##x,y,z,v)), \
 (I[192] = (img)(_p7##x,_n1##y,z,v)), \
 (I[211] = (img)(_p7##x,_n2##y,z,v)), \
 (I[230] = (img)(_p7##x,_n3##y,z,v)), \
 (I[249] = (img)(_p7##x,_n4##y,z,v)), \
 (I[268] = (img)(_p7##x,_n5##y,z,v)), \
 (I[287] = (img)(_p7##x,_n6##y,z,v)), \
 (I[306] = (img)(_p7##x,_n7##y,z,v)), \
 (I[325] = (img)(_p7##x,_n8##y,z,v)), \
 (I[344] = (img)(_p7##x,_n9##y,z,v)), \
 (I[3] = (img)(_p6##x,_p9##y,z,v)), \
 (I[22] = (img)(_p6##x,_p8##y,z,v)), \
 (I[41] = (img)(_p6##x,_p7##y,z,v)), \
 (I[60] = (img)(_p6##x,_p6##y,z,v)), \
 (I[79] = (img)(_p6##x,_p5##y,z,v)), \
 (I[98] = (img)(_p6##x,_p4##y,z,v)), \
 (I[117] = (img)(_p6##x,_p3##y,z,v)), \
 (I[136] = (img)(_p6##x,_p2##y,z,v)), \
 (I[155] = (img)(_p6##x,_p1##y,z,v)), \
 (I[174] = (img)(_p6##x,y,z,v)), \
 (I[193] = (img)(_p6##x,_n1##y,z,v)), \
 (I[212] = (img)(_p6##x,_n2##y,z,v)), \
 (I[231] = (img)(_p6##x,_n3##y,z,v)), \
 (I[250] = (img)(_p6##x,_n4##y,z,v)), \
 (I[269] = (img)(_p6##x,_n5##y,z,v)), \
 (I[288] = (img)(_p6##x,_n6##y,z,v)), \
 (I[307] = (img)(_p6##x,_n7##y,z,v)), \
 (I[326] = (img)(_p6##x,_n8##y,z,v)), \
 (I[345] = (img)(_p6##x,_n9##y,z,v)), \
 (I[4] = (img)(_p5##x,_p9##y,z,v)), \
 (I[23] = (img)(_p5##x,_p8##y,z,v)), \
 (I[42] = (img)(_p5##x,_p7##y,z,v)), \
 (I[61] = (img)(_p5##x,_p6##y,z,v)), \
 (I[80] = (img)(_p5##x,_p5##y,z,v)), \
 (I[99] = (img)(_p5##x,_p4##y,z,v)), \
 (I[118] = (img)(_p5##x,_p3##y,z,v)), \
 (I[137] = (img)(_p5##x,_p2##y,z,v)), \
 (I[156] = (img)(_p5##x,_p1##y,z,v)), \
 (I[175] = (img)(_p5##x,y,z,v)), \
 (I[194] = (img)(_p5##x,_n1##y,z,v)), \
 (I[213] = (img)(_p5##x,_n2##y,z,v)), \
 (I[232] = (img)(_p5##x,_n3##y,z,v)), \
 (I[251] = (img)(_p5##x,_n4##y,z,v)), \
 (I[270] = (img)(_p5##x,_n5##y,z,v)), \
 (I[289] = (img)(_p5##x,_n6##y,z,v)), \
 (I[308] = (img)(_p5##x,_n7##y,z,v)), \
 (I[327] = (img)(_p5##x,_n8##y,z,v)), \
 (I[346] = (img)(_p5##x,_n9##y,z,v)), \
 (I[5] = (img)(_p4##x,_p9##y,z,v)), \
 (I[24] = (img)(_p4##x,_p8##y,z,v)), \
 (I[43] = (img)(_p4##x,_p7##y,z,v)), \
 (I[62] = (img)(_p4##x,_p6##y,z,v)), \
 (I[81] = (img)(_p4##x,_p5##y,z,v)), \
 (I[100] = (img)(_p4##x,_p4##y,z,v)), \
 (I[119] = (img)(_p4##x,_p3##y,z,v)), \
 (I[138] = (img)(_p4##x,_p2##y,z,v)), \
 (I[157] = (img)(_p4##x,_p1##y,z,v)), \
 (I[176] = (img)(_p4##x,y,z,v)), \
 (I[195] = (img)(_p4##x,_n1##y,z,v)), \
 (I[214] = (img)(_p4##x,_n2##y,z,v)), \
 (I[233] = (img)(_p4##x,_n3##y,z,v)), \
 (I[252] = (img)(_p4##x,_n4##y,z,v)), \
 (I[271] = (img)(_p4##x,_n5##y,z,v)), \
 (I[290] = (img)(_p4##x,_n6##y,z,v)), \
 (I[309] = (img)(_p4##x,_n7##y,z,v)), \
 (I[328] = (img)(_p4##x,_n8##y,z,v)), \
 (I[347] = (img)(_p4##x,_n9##y,z,v)), \
 (I[6] = (img)(_p3##x,_p9##y,z,v)), \
 (I[25] = (img)(_p3##x,_p8##y,z,v)), \
 (I[44] = (img)(_p3##x,_p7##y,z,v)), \
 (I[63] = (img)(_p3##x,_p6##y,z,v)), \
 (I[82] = (img)(_p3##x,_p5##y,z,v)), \
 (I[101] = (img)(_p3##x,_p4##y,z,v)), \
 (I[120] = (img)(_p3##x,_p3##y,z,v)), \
 (I[139] = (img)(_p3##x,_p2##y,z,v)), \
 (I[158] = (img)(_p3##x,_p1##y,z,v)), \
 (I[177] = (img)(_p3##x,y,z,v)), \
 (I[196] = (img)(_p3##x,_n1##y,z,v)), \
 (I[215] = (img)(_p3##x,_n2##y,z,v)), \
 (I[234] = (img)(_p3##x,_n3##y,z,v)), \
 (I[253] = (img)(_p3##x,_n4##y,z,v)), \
 (I[272] = (img)(_p3##x,_n5##y,z,v)), \
 (I[291] = (img)(_p3##x,_n6##y,z,v)), \
 (I[310] = (img)(_p3##x,_n7##y,z,v)), \
 (I[329] = (img)(_p3##x,_n8##y,z,v)), \
 (I[348] = (img)(_p3##x,_n9##y,z,v)), \
 (I[7] = (img)(_p2##x,_p9##y,z,v)), \
 (I[26] = (img)(_p2##x,_p8##y,z,v)), \
 (I[45] = (img)(_p2##x,_p7##y,z,v)), \
 (I[64] = (img)(_p2##x,_p6##y,z,v)), \
 (I[83] = (img)(_p2##x,_p5##y,z,v)), \
 (I[102] = (img)(_p2##x,_p4##y,z,v)), \
 (I[121] = (img)(_p2##x,_p3##y,z,v)), \
 (I[140] = (img)(_p2##x,_p2##y,z,v)), \
 (I[159] = (img)(_p2##x,_p1##y,z,v)), \
 (I[178] = (img)(_p2##x,y,z,v)), \
 (I[197] = (img)(_p2##x,_n1##y,z,v)), \
 (I[216] = (img)(_p2##x,_n2##y,z,v)), \
 (I[235] = (img)(_p2##x,_n3##y,z,v)), \
 (I[254] = (img)(_p2##x,_n4##y,z,v)), \
 (I[273] = (img)(_p2##x,_n5##y,z,v)), \
 (I[292] = (img)(_p2##x,_n6##y,z,v)), \
 (I[311] = (img)(_p2##x,_n7##y,z,v)), \
 (I[330] = (img)(_p2##x,_n8##y,z,v)), \
 (I[349] = (img)(_p2##x,_n9##y,z,v)), \
 (I[8] = (img)(_p1##x,_p9##y,z,v)), \
 (I[27] = (img)(_p1##x,_p8##y,z,v)), \
 (I[46] = (img)(_p1##x,_p7##y,z,v)), \
 (I[65] = (img)(_p1##x,_p6##y,z,v)), \
 (I[84] = (img)(_p1##x,_p5##y,z,v)), \
 (I[103] = (img)(_p1##x,_p4##y,z,v)), \
 (I[122] = (img)(_p1##x,_p3##y,z,v)), \
 (I[141] = (img)(_p1##x,_p2##y,z,v)), \
 (I[160] = (img)(_p1##x,_p1##y,z,v)), \
 (I[179] = (img)(_p1##x,y,z,v)), \
 (I[198] = (img)(_p1##x,_n1##y,z,v)), \
 (I[217] = (img)(_p1##x,_n2##y,z,v)), \
 (I[236] = (img)(_p1##x,_n3##y,z,v)), \
 (I[255] = (img)(_p1##x,_n4##y,z,v)), \
 (I[274] = (img)(_p1##x,_n5##y,z,v)), \
 (I[293] = (img)(_p1##x,_n6##y,z,v)), \
 (I[312] = (img)(_p1##x,_n7##y,z,v)), \
 (I[331] = (img)(_p1##x,_n8##y,z,v)), \
 (I[350] = (img)(_p1##x,_n9##y,z,v)), \
 (I[9] = (img)(x,_p9##y,z,v)), \
 (I[28] = (img)(x,_p8##y,z,v)), \
 (I[47] = (img)(x,_p7##y,z,v)), \
 (I[66] = (img)(x,_p6##y,z,v)), \
 (I[85] = (img)(x,_p5##y,z,v)), \
 (I[104] = (img)(x,_p4##y,z,v)), \
 (I[123] = (img)(x,_p3##y,z,v)), \
 (I[142] = (img)(x,_p2##y,z,v)), \
 (I[161] = (img)(x,_p1##y,z,v)), \
 (I[180] = (img)(x,y,z,v)), \
 (I[199] = (img)(x,_n1##y,z,v)), \
 (I[218] = (img)(x,_n2##y,z,v)), \
 (I[237] = (img)(x,_n3##y,z,v)), \
 (I[256] = (img)(x,_n4##y,z,v)), \
 (I[275] = (img)(x,_n5##y,z,v)), \
 (I[294] = (img)(x,_n6##y,z,v)), \
 (I[313] = (img)(x,_n7##y,z,v)), \
 (I[332] = (img)(x,_n8##y,z,v)), \
 (I[351] = (img)(x,_n9##y,z,v)), \
 (I[10] = (img)(_n1##x,_p9##y,z,v)), \
 (I[29] = (img)(_n1##x,_p8##y,z,v)), \
 (I[48] = (img)(_n1##x,_p7##y,z,v)), \
 (I[67] = (img)(_n1##x,_p6##y,z,v)), \
 (I[86] = (img)(_n1##x,_p5##y,z,v)), \
 (I[105] = (img)(_n1##x,_p4##y,z,v)), \
 (I[124] = (img)(_n1##x,_p3##y,z,v)), \
 (I[143] = (img)(_n1##x,_p2##y,z,v)), \
 (I[162] = (img)(_n1##x,_p1##y,z,v)), \
 (I[181] = (img)(_n1##x,y,z,v)), \
 (I[200] = (img)(_n1##x,_n1##y,z,v)), \
 (I[219] = (img)(_n1##x,_n2##y,z,v)), \
 (I[238] = (img)(_n1##x,_n3##y,z,v)), \
 (I[257] = (img)(_n1##x,_n4##y,z,v)), \
 (I[276] = (img)(_n1##x,_n5##y,z,v)), \
 (I[295] = (img)(_n1##x,_n6##y,z,v)), \
 (I[314] = (img)(_n1##x,_n7##y,z,v)), \
 (I[333] = (img)(_n1##x,_n8##y,z,v)), \
 (I[352] = (img)(_n1##x,_n9##y,z,v)), \
 (I[11] = (img)(_n2##x,_p9##y,z,v)), \
 (I[30] = (img)(_n2##x,_p8##y,z,v)), \
 (I[49] = (img)(_n2##x,_p7##y,z,v)), \
 (I[68] = (img)(_n2##x,_p6##y,z,v)), \
 (I[87] = (img)(_n2##x,_p5##y,z,v)), \
 (I[106] = (img)(_n2##x,_p4##y,z,v)), \
 (I[125] = (img)(_n2##x,_p3##y,z,v)), \
 (I[144] = (img)(_n2##x,_p2##y,z,v)), \
 (I[163] = (img)(_n2##x,_p1##y,z,v)), \
 (I[182] = (img)(_n2##x,y,z,v)), \
 (I[201] = (img)(_n2##x,_n1##y,z,v)), \
 (I[220] = (img)(_n2##x,_n2##y,z,v)), \
 (I[239] = (img)(_n2##x,_n3##y,z,v)), \
 (I[258] = (img)(_n2##x,_n4##y,z,v)), \
 (I[277] = (img)(_n2##x,_n5##y,z,v)), \
 (I[296] = (img)(_n2##x,_n6##y,z,v)), \
 (I[315] = (img)(_n2##x,_n7##y,z,v)), \
 (I[334] = (img)(_n2##x,_n8##y,z,v)), \
 (I[353] = (img)(_n2##x,_n9##y,z,v)), \
 (I[12] = (img)(_n3##x,_p9##y,z,v)), \
 (I[31] = (img)(_n3##x,_p8##y,z,v)), \
 (I[50] = (img)(_n3##x,_p7##y,z,v)), \
 (I[69] = (img)(_n3##x,_p6##y,z,v)), \
 (I[88] = (img)(_n3##x,_p5##y,z,v)), \
 (I[107] = (img)(_n3##x,_p4##y,z,v)), \
 (I[126] = (img)(_n3##x,_p3##y,z,v)), \
 (I[145] = (img)(_n3##x,_p2##y,z,v)), \
 (I[164] = (img)(_n3##x,_p1##y,z,v)), \
 (I[183] = (img)(_n3##x,y,z,v)), \
 (I[202] = (img)(_n3##x,_n1##y,z,v)), \
 (I[221] = (img)(_n3##x,_n2##y,z,v)), \
 (I[240] = (img)(_n3##x,_n3##y,z,v)), \
 (I[259] = (img)(_n3##x,_n4##y,z,v)), \
 (I[278] = (img)(_n3##x,_n5##y,z,v)), \
 (I[297] = (img)(_n3##x,_n6##y,z,v)), \
 (I[316] = (img)(_n3##x,_n7##y,z,v)), \
 (I[335] = (img)(_n3##x,_n8##y,z,v)), \
 (I[354] = (img)(_n3##x,_n9##y,z,v)), \
 (I[13] = (img)(_n4##x,_p9##y,z,v)), \
 (I[32] = (img)(_n4##x,_p8##y,z,v)), \
 (I[51] = (img)(_n4##x,_p7##y,z,v)), \
 (I[70] = (img)(_n4##x,_p6##y,z,v)), \
 (I[89] = (img)(_n4##x,_p5##y,z,v)), \
 (I[108] = (img)(_n4##x,_p4##y,z,v)), \
 (I[127] = (img)(_n4##x,_p3##y,z,v)), \
 (I[146] = (img)(_n4##x,_p2##y,z,v)), \
 (I[165] = (img)(_n4##x,_p1##y,z,v)), \
 (I[184] = (img)(_n4##x,y,z,v)), \
 (I[203] = (img)(_n4##x,_n1##y,z,v)), \
 (I[222] = (img)(_n4##x,_n2##y,z,v)), \
 (I[241] = (img)(_n4##x,_n3##y,z,v)), \
 (I[260] = (img)(_n4##x,_n4##y,z,v)), \
 (I[279] = (img)(_n4##x,_n5##y,z,v)), \
 (I[298] = (img)(_n4##x,_n6##y,z,v)), \
 (I[317] = (img)(_n4##x,_n7##y,z,v)), \
 (I[336] = (img)(_n4##x,_n8##y,z,v)), \
 (I[355] = (img)(_n4##x,_n9##y,z,v)), \
 (I[14] = (img)(_n5##x,_p9##y,z,v)), \
 (I[33] = (img)(_n5##x,_p8##y,z,v)), \
 (I[52] = (img)(_n5##x,_p7##y,z,v)), \
 (I[71] = (img)(_n5##x,_p6##y,z,v)), \
 (I[90] = (img)(_n5##x,_p5##y,z,v)), \
 (I[109] = (img)(_n5##x,_p4##y,z,v)), \
 (I[128] = (img)(_n5##x,_p3##y,z,v)), \
 (I[147] = (img)(_n5##x,_p2##y,z,v)), \
 (I[166] = (img)(_n5##x,_p1##y,z,v)), \
 (I[185] = (img)(_n5##x,y,z,v)), \
 (I[204] = (img)(_n5##x,_n1##y,z,v)), \
 (I[223] = (img)(_n5##x,_n2##y,z,v)), \
 (I[242] = (img)(_n5##x,_n3##y,z,v)), \
 (I[261] = (img)(_n5##x,_n4##y,z,v)), \
 (I[280] = (img)(_n5##x,_n5##y,z,v)), \
 (I[299] = (img)(_n5##x,_n6##y,z,v)), \
 (I[318] = (img)(_n5##x,_n7##y,z,v)), \
 (I[337] = (img)(_n5##x,_n8##y,z,v)), \
 (I[356] = (img)(_n5##x,_n9##y,z,v)), \
 (I[15] = (img)(_n6##x,_p9##y,z,v)), \
 (I[34] = (img)(_n6##x,_p8##y,z,v)), \
 (I[53] = (img)(_n6##x,_p7##y,z,v)), \
 (I[72] = (img)(_n6##x,_p6##y,z,v)), \
 (I[91] = (img)(_n6##x,_p5##y,z,v)), \
 (I[110] = (img)(_n6##x,_p4##y,z,v)), \
 (I[129] = (img)(_n6##x,_p3##y,z,v)), \
 (I[148] = (img)(_n6##x,_p2##y,z,v)), \
 (I[167] = (img)(_n6##x,_p1##y,z,v)), \
 (I[186] = (img)(_n6##x,y,z,v)), \
 (I[205] = (img)(_n6##x,_n1##y,z,v)), \
 (I[224] = (img)(_n6##x,_n2##y,z,v)), \
 (I[243] = (img)(_n6##x,_n3##y,z,v)), \
 (I[262] = (img)(_n6##x,_n4##y,z,v)), \
 (I[281] = (img)(_n6##x,_n5##y,z,v)), \
 (I[300] = (img)(_n6##x,_n6##y,z,v)), \
 (I[319] = (img)(_n6##x,_n7##y,z,v)), \
 (I[338] = (img)(_n6##x,_n8##y,z,v)), \
 (I[357] = (img)(_n6##x,_n9##y,z,v)), \
 (I[16] = (img)(_n7##x,_p9##y,z,v)), \
 (I[35] = (img)(_n7##x,_p8##y,z,v)), \
 (I[54] = (img)(_n7##x,_p7##y,z,v)), \
 (I[73] = (img)(_n7##x,_p6##y,z,v)), \
 (I[92] = (img)(_n7##x,_p5##y,z,v)), \
 (I[111] = (img)(_n7##x,_p4##y,z,v)), \
 (I[130] = (img)(_n7##x,_p3##y,z,v)), \
 (I[149] = (img)(_n7##x,_p2##y,z,v)), \
 (I[168] = (img)(_n7##x,_p1##y,z,v)), \
 (I[187] = (img)(_n7##x,y,z,v)), \
 (I[206] = (img)(_n7##x,_n1##y,z,v)), \
 (I[225] = (img)(_n7##x,_n2##y,z,v)), \
 (I[244] = (img)(_n7##x,_n3##y,z,v)), \
 (I[263] = (img)(_n7##x,_n4##y,z,v)), \
 (I[282] = (img)(_n7##x,_n5##y,z,v)), \
 (I[301] = (img)(_n7##x,_n6##y,z,v)), \
 (I[320] = (img)(_n7##x,_n7##y,z,v)), \
 (I[339] = (img)(_n7##x,_n8##y,z,v)), \
 (I[358] = (img)(_n7##x,_n9##y,z,v)), \
 (I[17] = (img)(_n8##x,_p9##y,z,v)), \
 (I[36] = (img)(_n8##x,_p8##y,z,v)), \
 (I[55] = (img)(_n8##x,_p7##y,z,v)), \
 (I[74] = (img)(_n8##x,_p6##y,z,v)), \
 (I[93] = (img)(_n8##x,_p5##y,z,v)), \
 (I[112] = (img)(_n8##x,_p4##y,z,v)), \
 (I[131] = (img)(_n8##x,_p3##y,z,v)), \
 (I[150] = (img)(_n8##x,_p2##y,z,v)), \
 (I[169] = (img)(_n8##x,_p1##y,z,v)), \
 (I[188] = (img)(_n8##x,y,z,v)), \
 (I[207] = (img)(_n8##x,_n1##y,z,v)), \
 (I[226] = (img)(_n8##x,_n2##y,z,v)), \
 (I[245] = (img)(_n8##x,_n3##y,z,v)), \
 (I[264] = (img)(_n8##x,_n4##y,z,v)), \
 (I[283] = (img)(_n8##x,_n5##y,z,v)), \
 (I[302] = (img)(_n8##x,_n6##y,z,v)), \
 (I[321] = (img)(_n8##x,_n7##y,z,v)), \
 (I[340] = (img)(_n8##x,_n8##y,z,v)), \
 (I[359] = (img)(_n8##x,_n9##y,z,v)), \
 x+9>=(int)((img).width)?(int)((img).width)-1:x+9); \
 x<=(int)(x1) && ((_n9##x<(int)((img).width) && ( \
 (I[18] = (img)(_n9##x,_p9##y,z,v)), \
 (I[37] = (img)(_n9##x,_p8##y,z,v)), \
 (I[56] = (img)(_n9##x,_p7##y,z,v)), \
 (I[75] = (img)(_n9##x,_p6##y,z,v)), \
 (I[94] = (img)(_n9##x,_p5##y,z,v)), \
 (I[113] = (img)(_n9##x,_p4##y,z,v)), \
 (I[132] = (img)(_n9##x,_p3##y,z,v)), \
 (I[151] = (img)(_n9##x,_p2##y,z,v)), \
 (I[170] = (img)(_n9##x,_p1##y,z,v)), \
 (I[189] = (img)(_n9##x,y,z,v)), \
 (I[208] = (img)(_n9##x,_n1##y,z,v)), \
 (I[227] = (img)(_n9##x,_n2##y,z,v)), \
 (I[246] = (img)(_n9##x,_n3##y,z,v)), \
 (I[265] = (img)(_n9##x,_n4##y,z,v)), \
 (I[284] = (img)(_n9##x,_n5##y,z,v)), \
 (I[303] = (img)(_n9##x,_n6##y,z,v)), \
 (I[322] = (img)(_n9##x,_n7##y,z,v)), \
 (I[341] = (img)(_n9##x,_n8##y,z,v)), \
 (I[360] = (img)(_n9##x,_n9##y,z,v)),1)) || \
 _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], \
 I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], \
 I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], \
 I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], \
 I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], \
 I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], \
 I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], \
 I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], \
 I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], \
 I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], \
 I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], \
 I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], \
 I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], \
 I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], \
 I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], \
 I[285] = I[286], I[286] = I[287], I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], \
 I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], I[321] = I[322], \
 I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], I[335] = I[336], I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], \
 I[342] = I[343], I[343] = I[344], I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], I[351] = I[352], I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], I[359] = I[360], \
 _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x)

#define cimg_get19x19(img,x,y,z,v,I) \
 I[0] = (img)(_p9##x,_p9##y,z,v), I[1] = (img)(_p8##x,_p9##y,z,v), I[2] = (img)(_p7##x,_p9##y,z,v), I[3] = (img)(_p6##x,_p9##y,z,v), I[4] = (img)(_p5##x,_p9##y,z,v), I[5] = (img)(_p4##x,_p9##y,z,v), I[6] = (img)(_p3##x,_p9##y,z,v), I[7] = (img)(_p2##x,_p9##y,z,v), I[8] = (img)(_p1##x,_p9##y,z,v), I[9] = (img)(x,_p9##y,z,v), I[10] = (img)(_n1##x,_p9##y,z,v), I[11] = (img)(_n2##x,_p9##y,z,v), I[12] = (img)(_n3##x,_p9##y,z,v), I[13] = (img)(_n4##x,_p9##y,z,v), I[14] = (img)(_n5##x,_p9##y,z,v), I[15] = (img)(_n6##x,_p9##y,z,v), I[16] = (img)(_n7##x,_p9##y,z,v), I[17] = (img)(_n8##x,_p9##y,z,v), I[18] = (img)(_n9##x,_p9##y,z,v), \
 I[19] = (img)(_p9##x,_p8##y,z,v), I[20] = (img)(_p8##x,_p8##y,z,v), I[21] = (img)(_p7##x,_p8##y,z,v), I[22] = (img)(_p6##x,_p8##y,z,v), I[23] = (img)(_p5##x,_p8##y,z,v), I[24] = (img)(_p4##x,_p8##y,z,v), I[25] = (img)(_p3##x,_p8##y,z,v), I[26] = (img)(_p2##x,_p8##y,z,v), I[27] = (img)(_p1##x,_p8##y,z,v), I[28] = (img)(x,_p8##y,z,v), I[29] = (img)(_n1##x,_p8##y,z,v), I[30] = (img)(_n2##x,_p8##y,z,v), I[31] = (img)(_n3##x,_p8##y,z,v), I[32] = (img)(_n4##x,_p8##y,z,v), I[33] = (img)(_n5##x,_p8##y,z,v), I[34] = (img)(_n6##x,_p8##y,z,v), I[35] = (img)(_n7##x,_p8##y,z,v), I[36] = (img)(_n8##x,_p8##y,z,v), I[37] = (img)(_n9##x,_p8##y,z,v), \
 I[38] = (img)(_p9##x,_p7##y,z,v), I[39] = (img)(_p8##x,_p7##y,z,v), I[40] = (img)(_p7##x,_p7##y,z,v), I[41] = (img)(_p6##x,_p7##y,z,v), I[42] = (img)(_p5##x,_p7##y,z,v), I[43] = (img)(_p4##x,_p7##y,z,v), I[44] = (img)(_p3##x,_p7##y,z,v), I[45] = (img)(_p2##x,_p7##y,z,v), I[46] = (img)(_p1##x,_p7##y,z,v), I[47] = (img)(x,_p7##y,z,v), I[48] = (img)(_n1##x,_p7##y,z,v), I[49] = (img)(_n2##x,_p7##y,z,v), I[50] = (img)(_n3##x,_p7##y,z,v), I[51] = (img)(_n4##x,_p7##y,z,v), I[52] = (img)(_n5##x,_p7##y,z,v), I[53] = (img)(_n6##x,_p7##y,z,v), I[54] = (img)(_n7##x,_p7##y,z,v), I[55] = (img)(_n8##x,_p7##y,z,v), I[56] = (img)(_n9##x,_p7##y,z,v), \
 I[57] = (img)(_p9##x,_p6##y,z,v), I[58] = (img)(_p8##x,_p6##y,z,v), I[59] = (img)(_p7##x,_p6##y,z,v), I[60] = (img)(_p6##x,_p6##y,z,v), I[61] = (img)(_p5##x,_p6##y,z,v), I[62] = (img)(_p4##x,_p6##y,z,v), I[63] = (img)(_p3##x,_p6##y,z,v), I[64] = (img)(_p2##x,_p6##y,z,v), I[65] = (img)(_p1##x,_p6##y,z,v), I[66] = (img)(x,_p6##y,z,v), I[67] = (img)(_n1##x,_p6##y,z,v), I[68] = (img)(_n2##x,_p6##y,z,v), I[69] = (img)(_n3##x,_p6##y,z,v), I[70] = (img)(_n4##x,_p6##y,z,v), I[71] = (img)(_n5##x,_p6##y,z,v), I[72] = (img)(_n6##x,_p6##y,z,v), I[73] = (img)(_n7##x,_p6##y,z,v), I[74] = (img)(_n8##x,_p6##y,z,v), I[75] = (img)(_n9##x,_p6##y,z,v), \
 I[76] = (img)(_p9##x,_p5##y,z,v), I[77] = (img)(_p8##x,_p5##y,z,v), I[78] = (img)(_p7##x,_p5##y,z,v), I[79] = (img)(_p6##x,_p5##y,z,v), I[80] = (img)(_p5##x,_p5##y,z,v), I[81] = (img)(_p4##x,_p5##y,z,v), I[82] = (img)(_p3##x,_p5##y,z,v), I[83] = (img)(_p2##x,_p5##y,z,v), I[84] = (img)(_p1##x,_p5##y,z,v), I[85] = (img)(x,_p5##y,z,v), I[86] = (img)(_n1##x,_p5##y,z,v), I[87] = (img)(_n2##x,_p5##y,z,v), I[88] = (img)(_n3##x,_p5##y,z,v), I[89] = (img)(_n4##x,_p5##y,z,v), I[90] = (img)(_n5##x,_p5##y,z,v), I[91] = (img)(_n6##x,_p5##y,z,v), I[92] = (img)(_n7##x,_p5##y,z,v), I[93] = (img)(_n8##x,_p5##y,z,v), I[94] = (img)(_n9##x,_p5##y,z,v), \
 I[95] = (img)(_p9##x,_p4##y,z,v), I[96] = (img)(_p8##x,_p4##y,z,v), I[97] = (img)(_p7##x,_p4##y,z,v), I[98] = (img)(_p6##x,_p4##y,z,v), I[99] = (img)(_p5##x,_p4##y,z,v), I[100] = (img)(_p4##x,_p4##y,z,v), I[101] = (img)(_p3##x,_p4##y,z,v), I[102] = (img)(_p2##x,_p4##y,z,v), I[103] = (img)(_p1##x,_p4##y,z,v), I[104] = (img)(x,_p4##y,z,v), I[105] = (img)(_n1##x,_p4##y,z,v), I[106] = (img)(_n2##x,_p4##y,z,v), I[107] = (img)(_n3##x,_p4##y,z,v), I[108] = (img)(_n4##x,_p4##y,z,v), I[109] = (img)(_n5##x,_p4##y,z,v), I[110] = (img)(_n6##x,_p4##y,z,v), I[111] = (img)(_n7##x,_p4##y,z,v), I[112] = (img)(_n8##x,_p4##y,z,v), I[113] = (img)(_n9##x,_p4##y,z,v), \
 I[114] = (img)(_p9##x,_p3##y,z,v), I[115] = (img)(_p8##x,_p3##y,z,v), I[116] = (img)(_p7##x,_p3##y,z,v), I[117] = (img)(_p6##x,_p3##y,z,v), I[118] = (img)(_p5##x,_p3##y,z,v), I[119] = (img)(_p4##x,_p3##y,z,v), I[120] = (img)(_p3##x,_p3##y,z,v), I[121] = (img)(_p2##x,_p3##y,z,v), I[122] = (img)(_p1##x,_p3##y,z,v), I[123] = (img)(x,_p3##y,z,v), I[124] = (img)(_n1##x,_p3##y,z,v), I[125] = (img)(_n2##x,_p3##y,z,v), I[126] = (img)(_n3##x,_p3##y,z,v), I[127] = (img)(_n4##x,_p3##y,z,v), I[128] = (img)(_n5##x,_p3##y,z,v), I[129] = (img)(_n6##x,_p3##y,z,v), I[130] = (img)(_n7##x,_p3##y,z,v), I[131] = (img)(_n8##x,_p3##y,z,v), I[132] = (img)(_n9##x,_p3##y,z,v), \
 I[133] = (img)(_p9##x,_p2##y,z,v), I[134] = (img)(_p8##x,_p2##y,z,v), I[135] = (img)(_p7##x,_p2##y,z,v), I[136] = (img)(_p6##x,_p2##y,z,v), I[137] = (img)(_p5##x,_p2##y,z,v), I[138] = (img)(_p4##x,_p2##y,z,v), I[139] = (img)(_p3##x,_p2##y,z,v), I[140] = (img)(_p2##x,_p2##y,z,v), I[141] = (img)(_p1##x,_p2##y,z,v), I[142] = (img)(x,_p2##y,z,v), I[143] = (img)(_n1##x,_p2##y,z,v), I[144] = (img)(_n2##x,_p2##y,z,v), I[145] = (img)(_n3##x,_p2##y,z,v), I[146] = (img)(_n4##x,_p2##y,z,v), I[147] = (img)(_n5##x,_p2##y,z,v), I[148] = (img)(_n6##x,_p2##y,z,v), I[149] = (img)(_n7##x,_p2##y,z,v), I[150] = (img)(_n8##x,_p2##y,z,v), I[151] = (img)(_n9##x,_p2##y,z,v), \
 I[152] = (img)(_p9##x,_p1##y,z,v), I[153] = (img)(_p8##x,_p1##y,z,v), I[154] = (img)(_p7##x,_p1##y,z,v), I[155] = (img)(_p6##x,_p1##y,z,v), I[156] = (img)(_p5##x,_p1##y,z,v), I[157] = (img)(_p4##x,_p1##y,z,v), I[158] = (img)(_p3##x,_p1##y,z,v), I[159] = (img)(_p2##x,_p1##y,z,v), I[160] = (img)(_p1##x,_p1##y,z,v), I[161] = (img)(x,_p1##y,z,v), I[162] = (img)(_n1##x,_p1##y,z,v), I[163] = (img)(_n2##x,_p1##y,z,v), I[164] = (img)(_n3##x,_p1##y,z,v), I[165] = (img)(_n4##x,_p1##y,z,v), I[166] = (img)(_n5##x,_p1##y,z,v), I[167] = (img)(_n6##x,_p1##y,z,v), I[168] = (img)(_n7##x,_p1##y,z,v), I[169] = (img)(_n8##x,_p1##y,z,v), I[170] = (img)(_n9##x,_p1##y,z,v), \
 I[171] = (img)(_p9##x,y,z,v), I[172] = (img)(_p8##x,y,z,v), I[173] = (img)(_p7##x,y,z,v), I[174] = (img)(_p6##x,y,z,v), I[175] = (img)(_p5##x,y,z,v), I[176] = (img)(_p4##x,y,z,v), I[177] = (img)(_p3##x,y,z,v), I[178] = (img)(_p2##x,y,z,v), I[179] = (img)(_p1##x,y,z,v), I[180] = (img)(x,y,z,v), I[181] = (img)(_n1##x,y,z,v), I[182] = (img)(_n2##x,y,z,v), I[183] = (img)(_n3##x,y,z,v), I[184] = (img)(_n4##x,y,z,v), I[185] = (img)(_n5##x,y,z,v), I[186] = (img)(_n6##x,y,z,v), I[187] = (img)(_n7##x,y,z,v), I[188] = (img)(_n8##x,y,z,v), I[189] = (img)(_n9##x,y,z,v), \
 I[190] = (img)(_p9##x,_n1##y,z,v), I[191] = (img)(_p8##x,_n1##y,z,v), I[192] = (img)(_p7##x,_n1##y,z,v), I[193] = (img)(_p6##x,_n1##y,z,v), I[194] = (img)(_p5##x,_n1##y,z,v), I[195] = (img)(_p4##x,_n1##y,z,v), I[196] = (img)(_p3##x,_n1##y,z,v), I[197] = (img)(_p2##x,_n1##y,z,v), I[198] = (img)(_p1##x,_n1##y,z,v), I[199] = (img)(x,_n1##y,z,v), I[200] = (img)(_n1##x,_n1##y,z,v), I[201] = (img)(_n2##x,_n1##y,z,v), I[202] = (img)(_n3##x,_n1##y,z,v), I[203] = (img)(_n4##x,_n1##y,z,v), I[204] = (img)(_n5##x,_n1##y,z,v), I[205] = (img)(_n6##x,_n1##y,z,v), I[206] = (img)(_n7##x,_n1##y,z,v), I[207] = (img)(_n8##x,_n1##y,z,v), I[208] = (img)(_n9##x,_n1##y,z,v), \
 I[209] = (img)(_p9##x,_n2##y,z,v), I[210] = (img)(_p8##x,_n2##y,z,v), I[211] = (img)(_p7##x,_n2##y,z,v), I[212] = (img)(_p6##x,_n2##y,z,v), I[213] = (img)(_p5##x,_n2##y,z,v), I[214] = (img)(_p4##x,_n2##y,z,v), I[215] = (img)(_p3##x,_n2##y,z,v), I[216] = (img)(_p2##x,_n2##y,z,v), I[217] = (img)(_p1##x,_n2##y,z,v), I[218] = (img)(x,_n2##y,z,v), I[219] = (img)(_n1##x,_n2##y,z,v), I[220] = (img)(_n2##x,_n2##y,z,v), I[221] = (img)(_n3##x,_n2##y,z,v), I[222] = (img)(_n4##x,_n2##y,z,v), I[223] = (img)(_n5##x,_n2##y,z,v), I[224] = (img)(_n6##x,_n2##y,z,v), I[225] = (img)(_n7##x,_n2##y,z,v), I[226] = (img)(_n8##x,_n2##y,z,v), I[227] = (img)(_n9##x,_n2##y,z,v), \
 I[228] = (img)(_p9##x,_n3##y,z,v), I[229] = (img)(_p8##x,_n3##y,z,v), I[230] = (img)(_p7##x,_n3##y,z,v), I[231] = (img)(_p6##x,_n3##y,z,v), I[232] = (img)(_p5##x,_n3##y,z,v), I[233] = (img)(_p4##x,_n3##y,z,v), I[234] = (img)(_p3##x,_n3##y,z,v), I[235] = (img)(_p2##x,_n3##y,z,v), I[236] = (img)(_p1##x,_n3##y,z,v), I[237] = (img)(x,_n3##y,z,v), I[238] = (img)(_n1##x,_n3##y,z,v), I[239] = (img)(_n2##x,_n3##y,z,v), I[240] = (img)(_n3##x,_n3##y,z,v), I[241] = (img)(_n4##x,_n3##y,z,v), I[242] = (img)(_n5##x,_n3##y,z,v), I[243] = (img)(_n6##x,_n3##y,z,v), I[244] = (img)(_n7##x,_n3##y,z,v), I[245] = (img)(_n8##x,_n3##y,z,v), I[246] = (img)(_n9##x,_n3##y,z,v), \
 I[247] = (img)(_p9##x,_n4##y,z,v), I[248] = (img)(_p8##x,_n4##y,z,v), I[249] = (img)(_p7##x,_n4##y,z,v), I[250] = (img)(_p6##x,_n4##y,z,v), I[251] = (img)(_p5##x,_n4##y,z,v), I[252] = (img)(_p4##x,_n4##y,z,v), I[253] = (img)(_p3##x,_n4##y,z,v), I[254] = (img)(_p2##x,_n4##y,z,v), I[255] = (img)(_p1##x,_n4##y,z,v), I[256] = (img)(x,_n4##y,z,v), I[257] = (img)(_n1##x,_n4##y,z,v), I[258] = (img)(_n2##x,_n4##y,z,v), I[259] = (img)(_n3##x,_n4##y,z,v), I[260] = (img)(_n4##x,_n4##y,z,v), I[261] = (img)(_n5##x,_n4##y,z,v), I[262] = (img)(_n6##x,_n4##y,z,v), I[263] = (img)(_n7##x,_n4##y,z,v), I[264] = (img)(_n8##x,_n4##y,z,v), I[265] = (img)(_n9##x,_n4##y,z,v), \
 I[266] = (img)(_p9##x,_n5##y,z,v), I[267] = (img)(_p8##x,_n5##y,z,v), I[268] = (img)(_p7##x,_n5##y,z,v), I[269] = (img)(_p6##x,_n5##y,z,v), I[270] = (img)(_p5##x,_n5##y,z,v), I[271] = (img)(_p4##x,_n5##y,z,v), I[272] = (img)(_p3##x,_n5##y,z,v), I[273] = (img)(_p2##x,_n5##y,z,v), I[274] = (img)(_p1##x,_n5##y,z,v), I[275] = (img)(x,_n5##y,z,v), I[276] = (img)(_n1##x,_n5##y,z,v), I[277] = (img)(_n2##x,_n5##y,z,v), I[278] = (img)(_n3##x,_n5##y,z,v), I[279] = (img)(_n4##x,_n5##y,z,v), I[280] = (img)(_n5##x,_n5##y,z,v), I[281] = (img)(_n6##x,_n5##y,z,v), I[282] = (img)(_n7##x,_n5##y,z,v), I[283] = (img)(_n8##x,_n5##y,z,v), I[284] = (img)(_n9##x,_n5##y,z,v), \
 I[285] = (img)(_p9##x,_n6##y,z,v), I[286] = (img)(_p8##x,_n6##y,z,v), I[287] = (img)(_p7##x,_n6##y,z,v), I[288] = (img)(_p6##x,_n6##y,z,v), I[289] = (img)(_p5##x,_n6##y,z,v), I[290] = (img)(_p4##x,_n6##y,z,v), I[291] = (img)(_p3##x,_n6##y,z,v), I[292] = (img)(_p2##x,_n6##y,z,v), I[293] = (img)(_p1##x,_n6##y,z,v), I[294] = (img)(x,_n6##y,z,v), I[295] = (img)(_n1##x,_n6##y,z,v), I[296] = (img)(_n2##x,_n6##y,z,v), I[297] = (img)(_n3##x,_n6##y,z,v), I[298] = (img)(_n4##x,_n6##y,z,v), I[299] = (img)(_n5##x,_n6##y,z,v), I[300] = (img)(_n6##x,_n6##y,z,v), I[301] = (img)(_n7##x,_n6##y,z,v), I[302] = (img)(_n8##x,_n6##y,z,v), I[303] = (img)(_n9##x,_n6##y,z,v), \
 I[304] = (img)(_p9##x,_n7##y,z,v), I[305] = (img)(_p8##x,_n7##y,z,v), I[306] = (img)(_p7##x,_n7##y,z,v), I[307] = (img)(_p6##x,_n7##y,z,v), I[308] = (img)(_p5##x,_n7##y,z,v), I[309] = (img)(_p4##x,_n7##y,z,v), I[310] = (img)(_p3##x,_n7##y,z,v), I[311] = (img)(_p2##x,_n7##y,z,v), I[312] = (img)(_p1##x,_n7##y,z,v), I[313] = (img)(x,_n7##y,z,v), I[314] = (img)(_n1##x,_n7##y,z,v), I[315] = (img)(_n2##x,_n7##y,z,v), I[316] = (img)(_n3##x,_n7##y,z,v), I[317] = (img)(_n4##x,_n7##y,z,v), I[318] = (img)(_n5##x,_n7##y,z,v), I[319] = (img)(_n6##x,_n7##y,z,v), I[320] = (img)(_n7##x,_n7##y,z,v), I[321] = (img)(_n8##x,_n7##y,z,v), I[322] = (img)(_n9##x,_n7##y,z,v), \
 I[323] = (img)(_p9##x,_n8##y,z,v), I[324] = (img)(_p8##x,_n8##y,z,v), I[325] = (img)(_p7##x,_n8##y,z,v), I[326] = (img)(_p6##x,_n8##y,z,v), I[327] = (img)(_p5##x,_n8##y,z,v), I[328] = (img)(_p4##x,_n8##y,z,v), I[329] = (img)(_p3##x,_n8##y,z,v), I[330] = (img)(_p2##x,_n8##y,z,v), I[331] = (img)(_p1##x,_n8##y,z,v), I[332] = (img)(x,_n8##y,z,v), I[333] = (img)(_n1##x,_n8##y,z,v), I[334] = (img)(_n2##x,_n8##y,z,v), I[335] = (img)(_n3##x,_n8##y,z,v), I[336] = (img)(_n4##x,_n8##y,z,v), I[337] = (img)(_n5##x,_n8##y,z,v), I[338] = (img)(_n6##x,_n8##y,z,v), I[339] = (img)(_n7##x,_n8##y,z,v), I[340] = (img)(_n8##x,_n8##y,z,v), I[341] = (img)(_n9##x,_n8##y,z,v), \
 I[342] = (img)(_p9##x,_n9##y,z,v), I[343] = (img)(_p8##x,_n9##y,z,v), I[344] = (img)(_p7##x,_n9##y,z,v), I[345] = (img)(_p6##x,_n9##y,z,v), I[346] = (img)(_p5##x,_n9##y,z,v), I[347] = (img)(_p4##x,_n9##y,z,v), I[348] = (img)(_p3##x,_n9##y,z,v), I[349] = (img)(_p2##x,_n9##y,z,v), I[350] = (img)(_p1##x,_n9##y,z,v), I[351] = (img)(x,_n9##y,z,v), I[352] = (img)(_n1##x,_n9##y,z,v), I[353] = (img)(_n2##x,_n9##y,z,v), I[354] = (img)(_n3##x,_n9##y,z,v), I[355] = (img)(_n4##x,_n9##y,z,v), I[356] = (img)(_n5##x,_n9##y,z,v), I[357] = (img)(_n6##x,_n9##y,z,v), I[358] = (img)(_n7##x,_n9##y,z,v), I[359] = (img)(_n8##x,_n9##y,z,v), I[360] = (img)(_n9##x,_n9##y,z,v);

// Define 20x20 loop macros for CImg
//----------------------------------
#define cimg_for20(bound,i) for (int i = 0, \
 _p9##i = 0, _p8##i = 0, _p7##i = 0, _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7, \
 _n8##i = 8>=(int)(bound)?(int)(bound)-1:8, \
 _n9##i = 9>=(int)(bound)?(int)(bound)-1:9, \
 _n10##i = 10>=(int)(bound)?(int)(bound)-1:10; \
 _n10##i<(int)(bound) || _n9##i==--_n10##i || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n10##i = _n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i, ++_n10##i)

#define cimg_for20X(img,x) cimg_for20((img).width,x)
#define cimg_for20Y(img,y) cimg_for20((img).height,y)
#define cimg_for20Z(img,z) cimg_for20((img).depth,z)
#define cimg_for20V(img,v) cimg_for20((img).dim,v)
#define cimg_for20XY(img,x,y) cimg_for20Y(img,y) cimg_for20X(img,x)
#define cimg_for20XZ(img,x,z) cimg_for20Z(img,z) cimg_for20X(img,x)
#define cimg_for20XV(img,x,v) cimg_for20V(img,v) cimg_for20X(img,x)
#define cimg_for20YZ(img,y,z) cimg_for20Z(img,z) cimg_for20Y(img,y)
#define cimg_for20YV(img,y,v) cimg_for20V(img,v) cimg_for20Y(img,y)
#define cimg_for20ZV(img,z,v) cimg_for20V(img,v) cimg_for20Z(img,z)
#define cimg_for20XYZ(img,x,y,z) cimg_for20Z(img,z) cimg_for20XY(img,x,y)
#define cimg_for20XZV(img,x,z,v) cimg_for20V(img,v) cimg_for20XZ(img,x,z)
#define cimg_for20YZV(img,y,z,v) cimg_for20V(img,v) cimg_for20YZ(img,y,z)
#define cimg_for20XYZV(img,x,y,z,v) cimg_for20V(img,v) cimg_for20XYZ(img,x,y,z)

#define cimg_for_in20(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p9##i = i-9<0?0:i-9, \
 _p8##i = i-8<0?0:i-8, \
 _p7##i = i-7<0?0:i-7, \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7, \
 _n8##i = i+8>=(int)(bound)?(int)(bound)-1:i+8, \
 _n9##i = i+9>=(int)(bound)?(int)(bound)-1:i+9, \
 _n10##i = i+10>=(int)(bound)?(int)(bound)-1:i+10; \
 i<=(int)(i1) && (_n10##i<(int)(bound) || _n9##i==--_n10##i || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n10##i = _n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i, ++_n10##i)

#define cimg_for_in20X(img,x0,x1,x) cimg_for_in20((img).width,x0,x1,x)
#define cimg_for_in20Y(img,y0,y1,y) cimg_for_in20((img).height,y0,y1,y)
#define cimg_for_in20Z(img,z0,z1,z) cimg_for_in20((img).depth,z0,z1,z)
#define cimg_for_in20V(img,v0,v1,v) cimg_for_in20((img).dim,v0,v1,v)
#define cimg_for_in20XY(img,x0,y0,x1,y1,x,y) cimg_for_in20Y(img,y0,y1,y) cimg_for_in20X(img,x0,x1,x)
#define cimg_for_in20XZ(img,x0,z0,x1,z1,x,z) cimg_for_in20Z(img,z0,z1,z) cimg_for_in20X(img,x0,x1,x)
#define cimg_for_in20XV(img,x0,v0,x1,v1,x,v) cimg_for_in20V(img,v0,v1,v) cimg_for_in20X(img,x0,x1,x)
#define cimg_for_in20YZ(img,y0,z0,y1,z1,y,z) cimg_for_in20Z(img,z0,z1,z) cimg_for_in20Y(img,y0,y1,y)
#define cimg_for_in20YV(img,y0,v0,y1,v1,y,v) cimg_for_in20V(img,v0,v1,v) cimg_for_in20Y(img,y0,y1,y)
#define cimg_for_in20ZV(img,z0,v0,z1,v1,z,v) cimg_for_in20V(img,v0,v1,v) cimg_for_in20Z(img,z0,z1,z)
#define cimg_for_in20XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in20Z(img,z0,z1,z) cimg_for_in20XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in20XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in20V(img,v0,v1,v) cimg_for_in20XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in20YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in20V(img,v0,v1,v) cimg_for_in20YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in20XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in20V(img,v0,v1,v) cimg_for_in20XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for20x20(img,x,y,z,v,I) \
 cimg_for20((img).height,y) for (int x = 0, \
 _p9##x = 0, _p8##x = 0, _p7##x = 0, _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = 7>=((img).width)?(int)((img).width)-1:7, \
 _n8##x = 8>=((img).width)?(int)((img).width)-1:8, \
 _n9##x = 9>=((img).width)?(int)((img).width)-1:9, \
 _n10##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = I[7] = I[8] = I[9] = (img)(0,_p9##y,z,v)), \
 (I[20] = I[21] = I[22] = I[23] = I[24] = I[25] = I[26] = I[27] = I[28] = I[29] = (img)(0,_p8##y,z,v)), \
 (I[40] = I[41] = I[42] = I[43] = I[44] = I[45] = I[46] = I[47] = I[48] = I[49] = (img)(0,_p7##y,z,v)), \
 (I[60] = I[61] = I[62] = I[63] = I[64] = I[65] = I[66] = I[67] = I[68] = I[69] = (img)(0,_p6##y,z,v)), \
 (I[80] = I[81] = I[82] = I[83] = I[84] = I[85] = I[86] = I[87] = I[88] = I[89] = (img)(0,_p5##y,z,v)), \
 (I[100] = I[101] = I[102] = I[103] = I[104] = I[105] = I[106] = I[107] = I[108] = I[109] = (img)(0,_p4##y,z,v)), \
 (I[120] = I[121] = I[122] = I[123] = I[124] = I[125] = I[126] = I[127] = I[128] = I[129] = (img)(0,_p3##y,z,v)), \
 (I[140] = I[141] = I[142] = I[143] = I[144] = I[145] = I[146] = I[147] = I[148] = I[149] = (img)(0,_p2##y,z,v)), \
 (I[160] = I[161] = I[162] = I[163] = I[164] = I[165] = I[166] = I[167] = I[168] = I[169] = (img)(0,_p1##y,z,v)), \
 (I[180] = I[181] = I[182] = I[183] = I[184] = I[185] = I[186] = I[187] = I[188] = I[189] = (img)(0,y,z,v)), \
 (I[200] = I[201] = I[202] = I[203] = I[204] = I[205] = I[206] = I[207] = I[208] = I[209] = (img)(0,_n1##y,z,v)), \
 (I[220] = I[221] = I[222] = I[223] = I[224] = I[225] = I[226] = I[227] = I[228] = I[229] = (img)(0,_n2##y,z,v)), \
 (I[240] = I[241] = I[242] = I[243] = I[244] = I[245] = I[246] = I[247] = I[248] = I[249] = (img)(0,_n3##y,z,v)), \
 (I[260] = I[261] = I[262] = I[263] = I[264] = I[265] = I[266] = I[267] = I[268] = I[269] = (img)(0,_n4##y,z,v)), \
 (I[280] = I[281] = I[282] = I[283] = I[284] = I[285] = I[286] = I[287] = I[288] = I[289] = (img)(0,_n5##y,z,v)), \
 (I[300] = I[301] = I[302] = I[303] = I[304] = I[305] = I[306] = I[307] = I[308] = I[309] = (img)(0,_n6##y,z,v)), \
 (I[320] = I[321] = I[322] = I[323] = I[324] = I[325] = I[326] = I[327] = I[328] = I[329] = (img)(0,_n7##y,z,v)), \
 (I[340] = I[341] = I[342] = I[343] = I[344] = I[345] = I[346] = I[347] = I[348] = I[349] = (img)(0,_n8##y,z,v)), \
 (I[360] = I[361] = I[362] = I[363] = I[364] = I[365] = I[366] = I[367] = I[368] = I[369] = (img)(0,_n9##y,z,v)), \
 (I[380] = I[381] = I[382] = I[383] = I[384] = I[385] = I[386] = I[387] = I[388] = I[389] = (img)(0,_n10##y,z,v)), \
 (I[10] = (img)(_n1##x,_p9##y,z,v)), \
 (I[30] = (img)(_n1##x,_p8##y,z,v)), \
 (I[50] = (img)(_n1##x,_p7##y,z,v)), \
 (I[70] = (img)(_n1##x,_p6##y,z,v)), \
 (I[90] = (img)(_n1##x,_p5##y,z,v)), \
 (I[110] = (img)(_n1##x,_p4##y,z,v)), \
 (I[130] = (img)(_n1##x,_p3##y,z,v)), \
 (I[150] = (img)(_n1##x,_p2##y,z,v)), \
 (I[170] = (img)(_n1##x,_p1##y,z,v)), \
 (I[190] = (img)(_n1##x,y,z,v)), \
 (I[210] = (img)(_n1##x,_n1##y,z,v)), \
 (I[230] = (img)(_n1##x,_n2##y,z,v)), \
 (I[250] = (img)(_n1##x,_n3##y,z,v)), \
 (I[270] = (img)(_n1##x,_n4##y,z,v)), \
 (I[290] = (img)(_n1##x,_n5##y,z,v)), \
 (I[310] = (img)(_n1##x,_n6##y,z,v)), \
 (I[330] = (img)(_n1##x,_n7##y,z,v)), \
 (I[350] = (img)(_n1##x,_n8##y,z,v)), \
 (I[370] = (img)(_n1##x,_n9##y,z,v)), \
 (I[390] = (img)(_n1##x,_n10##y,z,v)), \
 (I[11] = (img)(_n2##x,_p9##y,z,v)), \
 (I[31] = (img)(_n2##x,_p8##y,z,v)), \
 (I[51] = (img)(_n2##x,_p7##y,z,v)), \
 (I[71] = (img)(_n2##x,_p6##y,z,v)), \
 (I[91] = (img)(_n2##x,_p5##y,z,v)), \
 (I[111] = (img)(_n2##x,_p4##y,z,v)), \
 (I[131] = (img)(_n2##x,_p3##y,z,v)), \
 (I[151] = (img)(_n2##x,_p2##y,z,v)), \
 (I[171] = (img)(_n2##x,_p1##y,z,v)), \
 (I[191] = (img)(_n2##x,y,z,v)), \
 (I[211] = (img)(_n2##x,_n1##y,z,v)), \
 (I[231] = (img)(_n2##x,_n2##y,z,v)), \
 (I[251] = (img)(_n2##x,_n3##y,z,v)), \
 (I[271] = (img)(_n2##x,_n4##y,z,v)), \
 (I[291] = (img)(_n2##x,_n5##y,z,v)), \
 (I[311] = (img)(_n2##x,_n6##y,z,v)), \
 (I[331] = (img)(_n2##x,_n7##y,z,v)), \
 (I[351] = (img)(_n2##x,_n8##y,z,v)), \
 (I[371] = (img)(_n2##x,_n9##y,z,v)), \
 (I[391] = (img)(_n2##x,_n10##y,z,v)), \
 (I[12] = (img)(_n3##x,_p9##y,z,v)), \
 (I[32] = (img)(_n3##x,_p8##y,z,v)), \
 (I[52] = (img)(_n3##x,_p7##y,z,v)), \
 (I[72] = (img)(_n3##x,_p6##y,z,v)), \
 (I[92] = (img)(_n3##x,_p5##y,z,v)), \
 (I[112] = (img)(_n3##x,_p4##y,z,v)), \
 (I[132] = (img)(_n3##x,_p3##y,z,v)), \
 (I[152] = (img)(_n3##x,_p2##y,z,v)), \
 (I[172] = (img)(_n3##x,_p1##y,z,v)), \
 (I[192] = (img)(_n3##x,y,z,v)), \
 (I[212] = (img)(_n3##x,_n1##y,z,v)), \
 (I[232] = (img)(_n3##x,_n2##y,z,v)), \
 (I[252] = (img)(_n3##x,_n3##y,z,v)), \
 (I[272] = (img)(_n3##x,_n4##y,z,v)), \
 (I[292] = (img)(_n3##x,_n5##y,z,v)), \
 (I[312] = (img)(_n3##x,_n6##y,z,v)), \
 (I[332] = (img)(_n3##x,_n7##y,z,v)), \
 (I[352] = (img)(_n3##x,_n8##y,z,v)), \
 (I[372] = (img)(_n3##x,_n9##y,z,v)), \
 (I[392] = (img)(_n3##x,_n10##y,z,v)), \
 (I[13] = (img)(_n4##x,_p9##y,z,v)), \
 (I[33] = (img)(_n4##x,_p8##y,z,v)), \
 (I[53] = (img)(_n4##x,_p7##y,z,v)), \
 (I[73] = (img)(_n4##x,_p6##y,z,v)), \
 (I[93] = (img)(_n4##x,_p5##y,z,v)), \
 (I[113] = (img)(_n4##x,_p4##y,z,v)), \
 (I[133] = (img)(_n4##x,_p3##y,z,v)), \
 (I[153] = (img)(_n4##x,_p2##y,z,v)), \
 (I[173] = (img)(_n4##x,_p1##y,z,v)), \
 (I[193] = (img)(_n4##x,y,z,v)), \
 (I[213] = (img)(_n4##x,_n1##y,z,v)), \
 (I[233] = (img)(_n4##x,_n2##y,z,v)), \
 (I[253] = (img)(_n4##x,_n3##y,z,v)), \
 (I[273] = (img)(_n4##x,_n4##y,z,v)), \
 (I[293] = (img)(_n4##x,_n5##y,z,v)), \
 (I[313] = (img)(_n4##x,_n6##y,z,v)), \
 (I[333] = (img)(_n4##x,_n7##y,z,v)), \
 (I[353] = (img)(_n4##x,_n8##y,z,v)), \
 (I[373] = (img)(_n4##x,_n9##y,z,v)), \
 (I[393] = (img)(_n4##x,_n10##y,z,v)), \
 (I[14] = (img)(_n5##x,_p9##y,z,v)), \
 (I[34] = (img)(_n5##x,_p8##y,z,v)), \
 (I[54] = (img)(_n5##x,_p7##y,z,v)), \
 (I[74] = (img)(_n5##x,_p6##y,z,v)), \
 (I[94] = (img)(_n5##x,_p5##y,z,v)), \
 (I[114] = (img)(_n5##x,_p4##y,z,v)), \
 (I[134] = (img)(_n5##x,_p3##y,z,v)), \
 (I[154] = (img)(_n5##x,_p2##y,z,v)), \
 (I[174] = (img)(_n5##x,_p1##y,z,v)), \
 (I[194] = (img)(_n5##x,y,z,v)), \
 (I[214] = (img)(_n5##x,_n1##y,z,v)), \
 (I[234] = (img)(_n5##x,_n2##y,z,v)), \
 (I[254] = (img)(_n5##x,_n3##y,z,v)), \
 (I[274] = (img)(_n5##x,_n4##y,z,v)), \
 (I[294] = (img)(_n5##x,_n5##y,z,v)), \
 (I[314] = (img)(_n5##x,_n6##y,z,v)), \
 (I[334] = (img)(_n5##x,_n7##y,z,v)), \
 (I[354] = (img)(_n5##x,_n8##y,z,v)), \
 (I[374] = (img)(_n5##x,_n9##y,z,v)), \
 (I[394] = (img)(_n5##x,_n10##y,z,v)), \
 (I[15] = (img)(_n6##x,_p9##y,z,v)), \
 (I[35] = (img)(_n6##x,_p8##y,z,v)), \
 (I[55] = (img)(_n6##x,_p7##y,z,v)), \
 (I[75] = (img)(_n6##x,_p6##y,z,v)), \
 (I[95] = (img)(_n6##x,_p5##y,z,v)), \
 (I[115] = (img)(_n6##x,_p4##y,z,v)), \
 (I[135] = (img)(_n6##x,_p3##y,z,v)), \
 (I[155] = (img)(_n6##x,_p2##y,z,v)), \
 (I[175] = (img)(_n6##x,_p1##y,z,v)), \
 (I[195] = (img)(_n6##x,y,z,v)), \
 (I[215] = (img)(_n6##x,_n1##y,z,v)), \
 (I[235] = (img)(_n6##x,_n2##y,z,v)), \
 (I[255] = (img)(_n6##x,_n3##y,z,v)), \
 (I[275] = (img)(_n6##x,_n4##y,z,v)), \
 (I[295] = (img)(_n6##x,_n5##y,z,v)), \
 (I[315] = (img)(_n6##x,_n6##y,z,v)), \
 (I[335] = (img)(_n6##x,_n7##y,z,v)), \
 (I[355] = (img)(_n6##x,_n8##y,z,v)), \
 (I[375] = (img)(_n6##x,_n9##y,z,v)), \
 (I[395] = (img)(_n6##x,_n10##y,z,v)), \
 (I[16] = (img)(_n7##x,_p9##y,z,v)), \
 (I[36] = (img)(_n7##x,_p8##y,z,v)), \
 (I[56] = (img)(_n7##x,_p7##y,z,v)), \
 (I[76] = (img)(_n7##x,_p6##y,z,v)), \
 (I[96] = (img)(_n7##x,_p5##y,z,v)), \
 (I[116] = (img)(_n7##x,_p4##y,z,v)), \
 (I[136] = (img)(_n7##x,_p3##y,z,v)), \
 (I[156] = (img)(_n7##x,_p2##y,z,v)), \
 (I[176] = (img)(_n7##x,_p1##y,z,v)), \
 (I[196] = (img)(_n7##x,y,z,v)), \
 (I[216] = (img)(_n7##x,_n1##y,z,v)), \
 (I[236] = (img)(_n7##x,_n2##y,z,v)), \
 (I[256] = (img)(_n7##x,_n3##y,z,v)), \
 (I[276] = (img)(_n7##x,_n4##y,z,v)), \
 (I[296] = (img)(_n7##x,_n5##y,z,v)), \
 (I[316] = (img)(_n7##x,_n6##y,z,v)), \
 (I[336] = (img)(_n7##x,_n7##y,z,v)), \
 (I[356] = (img)(_n7##x,_n8##y,z,v)), \
 (I[376] = (img)(_n7##x,_n9##y,z,v)), \
 (I[396] = (img)(_n7##x,_n10##y,z,v)), \
 (I[17] = (img)(_n8##x,_p9##y,z,v)), \
 (I[37] = (img)(_n8##x,_p8##y,z,v)), \
 (I[57] = (img)(_n8##x,_p7##y,z,v)), \
 (I[77] = (img)(_n8##x,_p6##y,z,v)), \
 (I[97] = (img)(_n8##x,_p5##y,z,v)), \
 (I[117] = (img)(_n8##x,_p4##y,z,v)), \
 (I[137] = (img)(_n8##x,_p3##y,z,v)), \
 (I[157] = (img)(_n8##x,_p2##y,z,v)), \
 (I[177] = (img)(_n8##x,_p1##y,z,v)), \
 (I[197] = (img)(_n8##x,y,z,v)), \
 (I[217] = (img)(_n8##x,_n1##y,z,v)), \
 (I[237] = (img)(_n8##x,_n2##y,z,v)), \
 (I[257] = (img)(_n8##x,_n3##y,z,v)), \
 (I[277] = (img)(_n8##x,_n4##y,z,v)), \
 (I[297] = (img)(_n8##x,_n5##y,z,v)), \
 (I[317] = (img)(_n8##x,_n6##y,z,v)), \
 (I[337] = (img)(_n8##x,_n7##y,z,v)), \
 (I[357] = (img)(_n8##x,_n8##y,z,v)), \
 (I[377] = (img)(_n8##x,_n9##y,z,v)), \
 (I[397] = (img)(_n8##x,_n10##y,z,v)), \
 (I[18] = (img)(_n9##x,_p9##y,z,v)), \
 (I[38] = (img)(_n9##x,_p8##y,z,v)), \
 (I[58] = (img)(_n9##x,_p7##y,z,v)), \
 (I[78] = (img)(_n9##x,_p6##y,z,v)), \
 (I[98] = (img)(_n9##x,_p5##y,z,v)), \
 (I[118] = (img)(_n9##x,_p4##y,z,v)), \
 (I[138] = (img)(_n9##x,_p3##y,z,v)), \
 (I[158] = (img)(_n9##x,_p2##y,z,v)), \
 (I[178] = (img)(_n9##x,_p1##y,z,v)), \
 (I[198] = (img)(_n9##x,y,z,v)), \
 (I[218] = (img)(_n9##x,_n1##y,z,v)), \
 (I[238] = (img)(_n9##x,_n2##y,z,v)), \
 (I[258] = (img)(_n9##x,_n3##y,z,v)), \
 (I[278] = (img)(_n9##x,_n4##y,z,v)), \
 (I[298] = (img)(_n9##x,_n5##y,z,v)), \
 (I[318] = (img)(_n9##x,_n6##y,z,v)), \
 (I[338] = (img)(_n9##x,_n7##y,z,v)), \
 (I[358] = (img)(_n9##x,_n8##y,z,v)), \
 (I[378] = (img)(_n9##x,_n9##y,z,v)), \
 (I[398] = (img)(_n9##x,_n10##y,z,v)), \
 10>=((img).width)?(int)((img).width)-1:10); \
 (_n10##x<(int)((img).width) && ( \
 (I[19] = (img)(_n10##x,_p9##y,z,v)), \
 (I[39] = (img)(_n10##x,_p8##y,z,v)), \
 (I[59] = (img)(_n10##x,_p7##y,z,v)), \
 (I[79] = (img)(_n10##x,_p6##y,z,v)), \
 (I[99] = (img)(_n10##x,_p5##y,z,v)), \
 (I[119] = (img)(_n10##x,_p4##y,z,v)), \
 (I[139] = (img)(_n10##x,_p3##y,z,v)), \
 (I[159] = (img)(_n10##x,_p2##y,z,v)), \
 (I[179] = (img)(_n10##x,_p1##y,z,v)), \
 (I[199] = (img)(_n10##x,y,z,v)), \
 (I[219] = (img)(_n10##x,_n1##y,z,v)), \
 (I[239] = (img)(_n10##x,_n2##y,z,v)), \
 (I[259] = (img)(_n10##x,_n3##y,z,v)), \
 (I[279] = (img)(_n10##x,_n4##y,z,v)), \
 (I[299] = (img)(_n10##x,_n5##y,z,v)), \
 (I[319] = (img)(_n10##x,_n6##y,z,v)), \
 (I[339] = (img)(_n10##x,_n7##y,z,v)), \
 (I[359] = (img)(_n10##x,_n8##y,z,v)), \
 (I[379] = (img)(_n10##x,_n9##y,z,v)), \
 (I[399] = (img)(_n10##x,_n10##y,z,v)),1)) || \
 _n9##x==--_n10##x || _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n10##x = _n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], \
 I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], \
 I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], \
 I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], \
 I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], \
 I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], \
 I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], \
 I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], \
 I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], \
 I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], \
 I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], \
 I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], \
 I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], \
 I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], \
 I[320] = I[321], I[321] = I[322], I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], I[335] = I[336], I[336] = I[337], I[337] = I[338], I[338] = I[339], \
 I[340] = I[341], I[341] = I[342], I[342] = I[343], I[343] = I[344], I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], I[351] = I[352], I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], \
 I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], I[367] = I[368], I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], I[373] = I[374], I[374] = I[375], I[375] = I[376], I[376] = I[377], I[377] = I[378], I[378] = I[379], \
 I[380] = I[381], I[381] = I[382], I[382] = I[383], I[383] = I[384], I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], I[390] = I[391], I[391] = I[392], I[392] = I[393], I[393] = I[394], I[394] = I[395], I[395] = I[396], I[396] = I[397], I[397] = I[398], I[398] = I[399], \
 _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x, ++_n10##x)

#define cimg_for_in20x20(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in20((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p9##x = x-9<0?0:x-9, \
 _p8##x = x-8<0?0:x-8, \
 _p7##x = x-7<0?0:x-7, \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = x+7>=(int)((img).width)?(int)((img).width)-1:x+7, \
 _n8##x = x+8>=(int)((img).width)?(int)((img).width)-1:x+8, \
 _n9##x = x+9>=(int)((img).width)?(int)((img).width)-1:x+9, \
 _n10##x = (int)( \
 (I[0] = (img)(_p9##x,_p9##y,z,v)), \
 (I[20] = (img)(_p9##x,_p8##y,z,v)), \
 (I[40] = (img)(_p9##x,_p7##y,z,v)), \
 (I[60] = (img)(_p9##x,_p6##y,z,v)), \
 (I[80] = (img)(_p9##x,_p5##y,z,v)), \
 (I[100] = (img)(_p9##x,_p4##y,z,v)), \
 (I[120] = (img)(_p9##x,_p3##y,z,v)), \
 (I[140] = (img)(_p9##x,_p2##y,z,v)), \
 (I[160] = (img)(_p9##x,_p1##y,z,v)), \
 (I[180] = (img)(_p9##x,y,z,v)), \
 (I[200] = (img)(_p9##x,_n1##y,z,v)), \
 (I[220] = (img)(_p9##x,_n2##y,z,v)), \
 (I[240] = (img)(_p9##x,_n3##y,z,v)), \
 (I[260] = (img)(_p9##x,_n4##y,z,v)), \
 (I[280] = (img)(_p9##x,_n5##y,z,v)), \
 (I[300] = (img)(_p9##x,_n6##y,z,v)), \
 (I[320] = (img)(_p9##x,_n7##y,z,v)), \
 (I[340] = (img)(_p9##x,_n8##y,z,v)), \
 (I[360] = (img)(_p9##x,_n9##y,z,v)), \
 (I[380] = (img)(_p9##x,_n10##y,z,v)), \
 (I[1] = (img)(_p8##x,_p9##y,z,v)), \
 (I[21] = (img)(_p8##x,_p8##y,z,v)), \
 (I[41] = (img)(_p8##x,_p7##y,z,v)), \
 (I[61] = (img)(_p8##x,_p6##y,z,v)), \
 (I[81] = (img)(_p8##x,_p5##y,z,v)), \
 (I[101] = (img)(_p8##x,_p4##y,z,v)), \
 (I[121] = (img)(_p8##x,_p3##y,z,v)), \
 (I[141] = (img)(_p8##x,_p2##y,z,v)), \
 (I[161] = (img)(_p8##x,_p1##y,z,v)), \
 (I[181] = (img)(_p8##x,y,z,v)), \
 (I[201] = (img)(_p8##x,_n1##y,z,v)), \
 (I[221] = (img)(_p8##x,_n2##y,z,v)), \
 (I[241] = (img)(_p8##x,_n3##y,z,v)), \
 (I[261] = (img)(_p8##x,_n4##y,z,v)), \
 (I[281] = (img)(_p8##x,_n5##y,z,v)), \
 (I[301] = (img)(_p8##x,_n6##y,z,v)), \
 (I[321] = (img)(_p8##x,_n7##y,z,v)), \
 (I[341] = (img)(_p8##x,_n8##y,z,v)), \
 (I[361] = (img)(_p8##x,_n9##y,z,v)), \
 (I[381] = (img)(_p8##x,_n10##y,z,v)), \
 (I[2] = (img)(_p7##x,_p9##y,z,v)), \
 (I[22] = (img)(_p7##x,_p8##y,z,v)), \
 (I[42] = (img)(_p7##x,_p7##y,z,v)), \
 (I[62] = (img)(_p7##x,_p6##y,z,v)), \
 (I[82] = (img)(_p7##x,_p5##y,z,v)), \
 (I[102] = (img)(_p7##x,_p4##y,z,v)), \
 (I[122] = (img)(_p7##x,_p3##y,z,v)), \
 (I[142] = (img)(_p7##x,_p2##y,z,v)), \
 (I[162] = (img)(_p7##x,_p1##y,z,v)), \
 (I[182] = (img)(_p7##x,y,z,v)), \
 (I[202] = (img)(_p7##x,_n1##y,z,v)), \
 (I[222] = (img)(_p7##x,_n2##y,z,v)), \
 (I[242] = (img)(_p7##x,_n3##y,z,v)), \
 (I[262] = (img)(_p7##x,_n4##y,z,v)), \
 (I[282] = (img)(_p7##x,_n5##y,z,v)), \
 (I[302] = (img)(_p7##x,_n6##y,z,v)), \
 (I[322] = (img)(_p7##x,_n7##y,z,v)), \
 (I[342] = (img)(_p7##x,_n8##y,z,v)), \
 (I[362] = (img)(_p7##x,_n9##y,z,v)), \
 (I[382] = (img)(_p7##x,_n10##y,z,v)), \
 (I[3] = (img)(_p6##x,_p9##y,z,v)), \
 (I[23] = (img)(_p6##x,_p8##y,z,v)), \
 (I[43] = (img)(_p6##x,_p7##y,z,v)), \
 (I[63] = (img)(_p6##x,_p6##y,z,v)), \
 (I[83] = (img)(_p6##x,_p5##y,z,v)), \
 (I[103] = (img)(_p6##x,_p4##y,z,v)), \
 (I[123] = (img)(_p6##x,_p3##y,z,v)), \
 (I[143] = (img)(_p6##x,_p2##y,z,v)), \
 (I[163] = (img)(_p6##x,_p1##y,z,v)), \
 (I[183] = (img)(_p6##x,y,z,v)), \
 (I[203] = (img)(_p6##x,_n1##y,z,v)), \
 (I[223] = (img)(_p6##x,_n2##y,z,v)), \
 (I[243] = (img)(_p6##x,_n3##y,z,v)), \
 (I[263] = (img)(_p6##x,_n4##y,z,v)), \
 (I[283] = (img)(_p6##x,_n5##y,z,v)), \
 (I[303] = (img)(_p6##x,_n6##y,z,v)), \
 (I[323] = (img)(_p6##x,_n7##y,z,v)), \
 (I[343] = (img)(_p6##x,_n8##y,z,v)), \
 (I[363] = (img)(_p6##x,_n9##y,z,v)), \
 (I[383] = (img)(_p6##x,_n10##y,z,v)), \
 (I[4] = (img)(_p5##x,_p9##y,z,v)), \
 (I[24] = (img)(_p5##x,_p8##y,z,v)), \
 (I[44] = (img)(_p5##x,_p7##y,z,v)), \
 (I[64] = (img)(_p5##x,_p6##y,z,v)), \
 (I[84] = (img)(_p5##x,_p5##y,z,v)), \
 (I[104] = (img)(_p5##x,_p4##y,z,v)), \
 (I[124] = (img)(_p5##x,_p3##y,z,v)), \
 (I[144] = (img)(_p5##x,_p2##y,z,v)), \
 (I[164] = (img)(_p5##x,_p1##y,z,v)), \
 (I[184] = (img)(_p5##x,y,z,v)), \
 (I[204] = (img)(_p5##x,_n1##y,z,v)), \
 (I[224] = (img)(_p5##x,_n2##y,z,v)), \
 (I[244] = (img)(_p5##x,_n3##y,z,v)), \
 (I[264] = (img)(_p5##x,_n4##y,z,v)), \
 (I[284] = (img)(_p5##x,_n5##y,z,v)), \
 (I[304] = (img)(_p5##x,_n6##y,z,v)), \
 (I[324] = (img)(_p5##x,_n7##y,z,v)), \
 (I[344] = (img)(_p5##x,_n8##y,z,v)), \
 (I[364] = (img)(_p5##x,_n9##y,z,v)), \
 (I[384] = (img)(_p5##x,_n10##y,z,v)), \
 (I[5] = (img)(_p4##x,_p9##y,z,v)), \
 (I[25] = (img)(_p4##x,_p8##y,z,v)), \
 (I[45] = (img)(_p4##x,_p7##y,z,v)), \
 (I[65] = (img)(_p4##x,_p6##y,z,v)), \
 (I[85] = (img)(_p4##x,_p5##y,z,v)), \
 (I[105] = (img)(_p4##x,_p4##y,z,v)), \
 (I[125] = (img)(_p4##x,_p3##y,z,v)), \
 (I[145] = (img)(_p4##x,_p2##y,z,v)), \
 (I[165] = (img)(_p4##x,_p1##y,z,v)), \
 (I[185] = (img)(_p4##x,y,z,v)), \
 (I[205] = (img)(_p4##x,_n1##y,z,v)), \
 (I[225] = (img)(_p4##x,_n2##y,z,v)), \
 (I[245] = (img)(_p4##x,_n3##y,z,v)), \
 (I[265] = (img)(_p4##x,_n4##y,z,v)), \
 (I[285] = (img)(_p4##x,_n5##y,z,v)), \
 (I[305] = (img)(_p4##x,_n6##y,z,v)), \
 (I[325] = (img)(_p4##x,_n7##y,z,v)), \
 (I[345] = (img)(_p4##x,_n8##y,z,v)), \
 (I[365] = (img)(_p4##x,_n9##y,z,v)), \
 (I[385] = (img)(_p4##x,_n10##y,z,v)), \
 (I[6] = (img)(_p3##x,_p9##y,z,v)), \
 (I[26] = (img)(_p3##x,_p8##y,z,v)), \
 (I[46] = (img)(_p3##x,_p7##y,z,v)), \
 (I[66] = (img)(_p3##x,_p6##y,z,v)), \
 (I[86] = (img)(_p3##x,_p5##y,z,v)), \
 (I[106] = (img)(_p3##x,_p4##y,z,v)), \
 (I[126] = (img)(_p3##x,_p3##y,z,v)), \
 (I[146] = (img)(_p3##x,_p2##y,z,v)), \
 (I[166] = (img)(_p3##x,_p1##y,z,v)), \
 (I[186] = (img)(_p3##x,y,z,v)), \
 (I[206] = (img)(_p3##x,_n1##y,z,v)), \
 (I[226] = (img)(_p3##x,_n2##y,z,v)), \
 (I[246] = (img)(_p3##x,_n3##y,z,v)), \
 (I[266] = (img)(_p3##x,_n4##y,z,v)), \
 (I[286] = (img)(_p3##x,_n5##y,z,v)), \
 (I[306] = (img)(_p3##x,_n6##y,z,v)), \
 (I[326] = (img)(_p3##x,_n7##y,z,v)), \
 (I[346] = (img)(_p3##x,_n8##y,z,v)), \
 (I[366] = (img)(_p3##x,_n9##y,z,v)), \
 (I[386] = (img)(_p3##x,_n10##y,z,v)), \
 (I[7] = (img)(_p2##x,_p9##y,z,v)), \
 (I[27] = (img)(_p2##x,_p8##y,z,v)), \
 (I[47] = (img)(_p2##x,_p7##y,z,v)), \
 (I[67] = (img)(_p2##x,_p6##y,z,v)), \
 (I[87] = (img)(_p2##x,_p5##y,z,v)), \
 (I[107] = (img)(_p2##x,_p4##y,z,v)), \
 (I[127] = (img)(_p2##x,_p3##y,z,v)), \
 (I[147] = (img)(_p2##x,_p2##y,z,v)), \
 (I[167] = (img)(_p2##x,_p1##y,z,v)), \
 (I[187] = (img)(_p2##x,y,z,v)), \
 (I[207] = (img)(_p2##x,_n1##y,z,v)), \
 (I[227] = (img)(_p2##x,_n2##y,z,v)), \
 (I[247] = (img)(_p2##x,_n3##y,z,v)), \
 (I[267] = (img)(_p2##x,_n4##y,z,v)), \
 (I[287] = (img)(_p2##x,_n5##y,z,v)), \
 (I[307] = (img)(_p2##x,_n6##y,z,v)), \
 (I[327] = (img)(_p2##x,_n7##y,z,v)), \
 (I[347] = (img)(_p2##x,_n8##y,z,v)), \
 (I[367] = (img)(_p2##x,_n9##y,z,v)), \
 (I[387] = (img)(_p2##x,_n10##y,z,v)), \
 (I[8] = (img)(_p1##x,_p9##y,z,v)), \
 (I[28] = (img)(_p1##x,_p8##y,z,v)), \
 (I[48] = (img)(_p1##x,_p7##y,z,v)), \
 (I[68] = (img)(_p1##x,_p6##y,z,v)), \
 (I[88] = (img)(_p1##x,_p5##y,z,v)), \
 (I[108] = (img)(_p1##x,_p4##y,z,v)), \
 (I[128] = (img)(_p1##x,_p3##y,z,v)), \
 (I[148] = (img)(_p1##x,_p2##y,z,v)), \
 (I[168] = (img)(_p1##x,_p1##y,z,v)), \
 (I[188] = (img)(_p1##x,y,z,v)), \
 (I[208] = (img)(_p1##x,_n1##y,z,v)), \
 (I[228] = (img)(_p1##x,_n2##y,z,v)), \
 (I[248] = (img)(_p1##x,_n3##y,z,v)), \
 (I[268] = (img)(_p1##x,_n4##y,z,v)), \
 (I[288] = (img)(_p1##x,_n5##y,z,v)), \
 (I[308] = (img)(_p1##x,_n6##y,z,v)), \
 (I[328] = (img)(_p1##x,_n7##y,z,v)), \
 (I[348] = (img)(_p1##x,_n8##y,z,v)), \
 (I[368] = (img)(_p1##x,_n9##y,z,v)), \
 (I[388] = (img)(_p1##x,_n10##y,z,v)), \
 (I[9] = (img)(x,_p9##y,z,v)), \
 (I[29] = (img)(x,_p8##y,z,v)), \
 (I[49] = (img)(x,_p7##y,z,v)), \
 (I[69] = (img)(x,_p6##y,z,v)), \
 (I[89] = (img)(x,_p5##y,z,v)), \
 (I[109] = (img)(x,_p4##y,z,v)), \
 (I[129] = (img)(x,_p3##y,z,v)), \
 (I[149] = (img)(x,_p2##y,z,v)), \
 (I[169] = (img)(x,_p1##y,z,v)), \
 (I[189] = (img)(x,y,z,v)), \
 (I[209] = (img)(x,_n1##y,z,v)), \
 (I[229] = (img)(x,_n2##y,z,v)), \
 (I[249] = (img)(x,_n3##y,z,v)), \
 (I[269] = (img)(x,_n4##y,z,v)), \
 (I[289] = (img)(x,_n5##y,z,v)), \
 (I[309] = (img)(x,_n6##y,z,v)), \
 (I[329] = (img)(x,_n7##y,z,v)), \
 (I[349] = (img)(x,_n8##y,z,v)), \
 (I[369] = (img)(x,_n9##y,z,v)), \
 (I[389] = (img)(x,_n10##y,z,v)), \
 (I[10] = (img)(_n1##x,_p9##y,z,v)), \
 (I[30] = (img)(_n1##x,_p8##y,z,v)), \
 (I[50] = (img)(_n1##x,_p7##y,z,v)), \
 (I[70] = (img)(_n1##x,_p6##y,z,v)), \
 (I[90] = (img)(_n1##x,_p5##y,z,v)), \
 (I[110] = (img)(_n1##x,_p4##y,z,v)), \
 (I[130] = (img)(_n1##x,_p3##y,z,v)), \
 (I[150] = (img)(_n1##x,_p2##y,z,v)), \
 (I[170] = (img)(_n1##x,_p1##y,z,v)), \
 (I[190] = (img)(_n1##x,y,z,v)), \
 (I[210] = (img)(_n1##x,_n1##y,z,v)), \
 (I[230] = (img)(_n1##x,_n2##y,z,v)), \
 (I[250] = (img)(_n1##x,_n3##y,z,v)), \
 (I[270] = (img)(_n1##x,_n4##y,z,v)), \
 (I[290] = (img)(_n1##x,_n5##y,z,v)), \
 (I[310] = (img)(_n1##x,_n6##y,z,v)), \
 (I[330] = (img)(_n1##x,_n7##y,z,v)), \
 (I[350] = (img)(_n1##x,_n8##y,z,v)), \
 (I[370] = (img)(_n1##x,_n9##y,z,v)), \
 (I[390] = (img)(_n1##x,_n10##y,z,v)), \
 (I[11] = (img)(_n2##x,_p9##y,z,v)), \
 (I[31] = (img)(_n2##x,_p8##y,z,v)), \
 (I[51] = (img)(_n2##x,_p7##y,z,v)), \
 (I[71] = (img)(_n2##x,_p6##y,z,v)), \
 (I[91] = (img)(_n2##x,_p5##y,z,v)), \
 (I[111] = (img)(_n2##x,_p4##y,z,v)), \
 (I[131] = (img)(_n2##x,_p3##y,z,v)), \
 (I[151] = (img)(_n2##x,_p2##y,z,v)), \
 (I[171] = (img)(_n2##x,_p1##y,z,v)), \
 (I[191] = (img)(_n2##x,y,z,v)), \
 (I[211] = (img)(_n2##x,_n1##y,z,v)), \
 (I[231] = (img)(_n2##x,_n2##y,z,v)), \
 (I[251] = (img)(_n2##x,_n3##y,z,v)), \
 (I[271] = (img)(_n2##x,_n4##y,z,v)), \
 (I[291] = (img)(_n2##x,_n5##y,z,v)), \
 (I[311] = (img)(_n2##x,_n6##y,z,v)), \
 (I[331] = (img)(_n2##x,_n7##y,z,v)), \
 (I[351] = (img)(_n2##x,_n8##y,z,v)), \
 (I[371] = (img)(_n2##x,_n9##y,z,v)), \
 (I[391] = (img)(_n2##x,_n10##y,z,v)), \
 (I[12] = (img)(_n3##x,_p9##y,z,v)), \
 (I[32] = (img)(_n3##x,_p8##y,z,v)), \
 (I[52] = (img)(_n3##x,_p7##y,z,v)), \
 (I[72] = (img)(_n3##x,_p6##y,z,v)), \
 (I[92] = (img)(_n3##x,_p5##y,z,v)), \
 (I[112] = (img)(_n3##x,_p4##y,z,v)), \
 (I[132] = (img)(_n3##x,_p3##y,z,v)), \
 (I[152] = (img)(_n3##x,_p2##y,z,v)), \
 (I[172] = (img)(_n3##x,_p1##y,z,v)), \
 (I[192] = (img)(_n3##x,y,z,v)), \
 (I[212] = (img)(_n3##x,_n1##y,z,v)), \
 (I[232] = (img)(_n3##x,_n2##y,z,v)), \
 (I[252] = (img)(_n3##x,_n3##y,z,v)), \
 (I[272] = (img)(_n3##x,_n4##y,z,v)), \
 (I[292] = (img)(_n3##x,_n5##y,z,v)), \
 (I[312] = (img)(_n3##x,_n6##y,z,v)), \
 (I[332] = (img)(_n3##x,_n7##y,z,v)), \
 (I[352] = (img)(_n3##x,_n8##y,z,v)), \
 (I[372] = (img)(_n3##x,_n9##y,z,v)), \
 (I[392] = (img)(_n3##x,_n10##y,z,v)), \
 (I[13] = (img)(_n4##x,_p9##y,z,v)), \
 (I[33] = (img)(_n4##x,_p8##y,z,v)), \
 (I[53] = (img)(_n4##x,_p7##y,z,v)), \
 (I[73] = (img)(_n4##x,_p6##y,z,v)), \
 (I[93] = (img)(_n4##x,_p5##y,z,v)), \
 (I[113] = (img)(_n4##x,_p4##y,z,v)), \
 (I[133] = (img)(_n4##x,_p3##y,z,v)), \
 (I[153] = (img)(_n4##x,_p2##y,z,v)), \
 (I[173] = (img)(_n4##x,_p1##y,z,v)), \
 (I[193] = (img)(_n4##x,y,z,v)), \
 (I[213] = (img)(_n4##x,_n1##y,z,v)), \
 (I[233] = (img)(_n4##x,_n2##y,z,v)), \
 (I[253] = (img)(_n4##x,_n3##y,z,v)), \
 (I[273] = (img)(_n4##x,_n4##y,z,v)), \
 (I[293] = (img)(_n4##x,_n5##y,z,v)), \
 (I[313] = (img)(_n4##x,_n6##y,z,v)), \
 (I[333] = (img)(_n4##x,_n7##y,z,v)), \
 (I[353] = (img)(_n4##x,_n8##y,z,v)), \
 (I[373] = (img)(_n4##x,_n9##y,z,v)), \
 (I[393] = (img)(_n4##x,_n10##y,z,v)), \
 (I[14] = (img)(_n5##x,_p9##y,z,v)), \
 (I[34] = (img)(_n5##x,_p8##y,z,v)), \
 (I[54] = (img)(_n5##x,_p7##y,z,v)), \
 (I[74] = (img)(_n5##x,_p6##y,z,v)), \
 (I[94] = (img)(_n5##x,_p5##y,z,v)), \
 (I[114] = (img)(_n5##x,_p4##y,z,v)), \
 (I[134] = (img)(_n5##x,_p3##y,z,v)), \
 (I[154] = (img)(_n5##x,_p2##y,z,v)), \
 (I[174] = (img)(_n5##x,_p1##y,z,v)), \
 (I[194] = (img)(_n5##x,y,z,v)), \
 (I[214] = (img)(_n5##x,_n1##y,z,v)), \
 (I[234] = (img)(_n5##x,_n2##y,z,v)), \
 (I[254] = (img)(_n5##x,_n3##y,z,v)), \
 (I[274] = (img)(_n5##x,_n4##y,z,v)), \
 (I[294] = (img)(_n5##x,_n5##y,z,v)), \
 (I[314] = (img)(_n5##x,_n6##y,z,v)), \
 (I[334] = (img)(_n5##x,_n7##y,z,v)), \
 (I[354] = (img)(_n5##x,_n8##y,z,v)), \
 (I[374] = (img)(_n5##x,_n9##y,z,v)), \
 (I[394] = (img)(_n5##x,_n10##y,z,v)), \
 (I[15] = (img)(_n6##x,_p9##y,z,v)), \
 (I[35] = (img)(_n6##x,_p8##y,z,v)), \
 (I[55] = (img)(_n6##x,_p7##y,z,v)), \
 (I[75] = (img)(_n6##x,_p6##y,z,v)), \
 (I[95] = (img)(_n6##x,_p5##y,z,v)), \
 (I[115] = (img)(_n6##x,_p4##y,z,v)), \
 (I[135] = (img)(_n6##x,_p3##y,z,v)), \
 (I[155] = (img)(_n6##x,_p2##y,z,v)), \
 (I[175] = (img)(_n6##x,_p1##y,z,v)), \
 (I[195] = (img)(_n6##x,y,z,v)), \
 (I[215] = (img)(_n6##x,_n1##y,z,v)), \
 (I[235] = (img)(_n6##x,_n2##y,z,v)), \
 (I[255] = (img)(_n6##x,_n3##y,z,v)), \
 (I[275] = (img)(_n6##x,_n4##y,z,v)), \
 (I[295] = (img)(_n6##x,_n5##y,z,v)), \
 (I[315] = (img)(_n6##x,_n6##y,z,v)), \
 (I[335] = (img)(_n6##x,_n7##y,z,v)), \
 (I[355] = (img)(_n6##x,_n8##y,z,v)), \
 (I[375] = (img)(_n6##x,_n9##y,z,v)), \
 (I[395] = (img)(_n6##x,_n10##y,z,v)), \
 (I[16] = (img)(_n7##x,_p9##y,z,v)), \
 (I[36] = (img)(_n7##x,_p8##y,z,v)), \
 (I[56] = (img)(_n7##x,_p7##y,z,v)), \
 (I[76] = (img)(_n7##x,_p6##y,z,v)), \
 (I[96] = (img)(_n7##x,_p5##y,z,v)), \
 (I[116] = (img)(_n7##x,_p4##y,z,v)), \
 (I[136] = (img)(_n7##x,_p3##y,z,v)), \
 (I[156] = (img)(_n7##x,_p2##y,z,v)), \
 (I[176] = (img)(_n7##x,_p1##y,z,v)), \
 (I[196] = (img)(_n7##x,y,z,v)), \
 (I[216] = (img)(_n7##x,_n1##y,z,v)), \
 (I[236] = (img)(_n7##x,_n2##y,z,v)), \
 (I[256] = (img)(_n7##x,_n3##y,z,v)), \
 (I[276] = (img)(_n7##x,_n4##y,z,v)), \
 (I[296] = (img)(_n7##x,_n5##y,z,v)), \
 (I[316] = (img)(_n7##x,_n6##y,z,v)), \
 (I[336] = (img)(_n7##x,_n7##y,z,v)), \
 (I[356] = (img)(_n7##x,_n8##y,z,v)), \
 (I[376] = (img)(_n7##x,_n9##y,z,v)), \
 (I[396] = (img)(_n7##x,_n10##y,z,v)), \
 (I[17] = (img)(_n8##x,_p9##y,z,v)), \
 (I[37] = (img)(_n8##x,_p8##y,z,v)), \
 (I[57] = (img)(_n8##x,_p7##y,z,v)), \
 (I[77] = (img)(_n8##x,_p6##y,z,v)), \
 (I[97] = (img)(_n8##x,_p5##y,z,v)), \
 (I[117] = (img)(_n8##x,_p4##y,z,v)), \
 (I[137] = (img)(_n8##x,_p3##y,z,v)), \
 (I[157] = (img)(_n8##x,_p2##y,z,v)), \
 (I[177] = (img)(_n8##x,_p1##y,z,v)), \
 (I[197] = (img)(_n8##x,y,z,v)), \
 (I[217] = (img)(_n8##x,_n1##y,z,v)), \
 (I[237] = (img)(_n8##x,_n2##y,z,v)), \
 (I[257] = (img)(_n8##x,_n3##y,z,v)), \
 (I[277] = (img)(_n8##x,_n4##y,z,v)), \
 (I[297] = (img)(_n8##x,_n5##y,z,v)), \
 (I[317] = (img)(_n8##x,_n6##y,z,v)), \
 (I[337] = (img)(_n8##x,_n7##y,z,v)), \
 (I[357] = (img)(_n8##x,_n8##y,z,v)), \
 (I[377] = (img)(_n8##x,_n9##y,z,v)), \
 (I[397] = (img)(_n8##x,_n10##y,z,v)), \
 (I[18] = (img)(_n9##x,_p9##y,z,v)), \
 (I[38] = (img)(_n9##x,_p8##y,z,v)), \
 (I[58] = (img)(_n9##x,_p7##y,z,v)), \
 (I[78] = (img)(_n9##x,_p6##y,z,v)), \
 (I[98] = (img)(_n9##x,_p5##y,z,v)), \
 (I[118] = (img)(_n9##x,_p4##y,z,v)), \
 (I[138] = (img)(_n9##x,_p3##y,z,v)), \
 (I[158] = (img)(_n9##x,_p2##y,z,v)), \
 (I[178] = (img)(_n9##x,_p1##y,z,v)), \
 (I[198] = (img)(_n9##x,y,z,v)), \
 (I[218] = (img)(_n9##x,_n1##y,z,v)), \
 (I[238] = (img)(_n9##x,_n2##y,z,v)), \
 (I[258] = (img)(_n9##x,_n3##y,z,v)), \
 (I[278] = (img)(_n9##x,_n4##y,z,v)), \
 (I[298] = (img)(_n9##x,_n5##y,z,v)), \
 (I[318] = (img)(_n9##x,_n6##y,z,v)), \
 (I[338] = (img)(_n9##x,_n7##y,z,v)), \
 (I[358] = (img)(_n9##x,_n8##y,z,v)), \
 (I[378] = (img)(_n9##x,_n9##y,z,v)), \
 (I[398] = (img)(_n9##x,_n10##y,z,v)), \
 x+10>=(int)((img).width)?(int)((img).width)-1:x+10); \
 x<=(int)(x1) && ((_n10##x<(int)((img).width) && ( \
 (I[19] = (img)(_n10##x,_p9##y,z,v)), \
 (I[39] = (img)(_n10##x,_p8##y,z,v)), \
 (I[59] = (img)(_n10##x,_p7##y,z,v)), \
 (I[79] = (img)(_n10##x,_p6##y,z,v)), \
 (I[99] = (img)(_n10##x,_p5##y,z,v)), \
 (I[119] = (img)(_n10##x,_p4##y,z,v)), \
 (I[139] = (img)(_n10##x,_p3##y,z,v)), \
 (I[159] = (img)(_n10##x,_p2##y,z,v)), \
 (I[179] = (img)(_n10##x,_p1##y,z,v)), \
 (I[199] = (img)(_n10##x,y,z,v)), \
 (I[219] = (img)(_n10##x,_n1##y,z,v)), \
 (I[239] = (img)(_n10##x,_n2##y,z,v)), \
 (I[259] = (img)(_n10##x,_n3##y,z,v)), \
 (I[279] = (img)(_n10##x,_n4##y,z,v)), \
 (I[299] = (img)(_n10##x,_n5##y,z,v)), \
 (I[319] = (img)(_n10##x,_n6##y,z,v)), \
 (I[339] = (img)(_n10##x,_n7##y,z,v)), \
 (I[359] = (img)(_n10##x,_n8##y,z,v)), \
 (I[379] = (img)(_n10##x,_n9##y,z,v)), \
 (I[399] = (img)(_n10##x,_n10##y,z,v)),1)) || \
 _n9##x==--_n10##x || _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n10##x = _n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], \
 I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], \
 I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], \
 I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], \
 I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], \
 I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], \
 I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], \
 I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], \
 I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], \
 I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], \
 I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], \
 I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], \
 I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], \
 I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], \
 I[320] = I[321], I[321] = I[322], I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], I[335] = I[336], I[336] = I[337], I[337] = I[338], I[338] = I[339], \
 I[340] = I[341], I[341] = I[342], I[342] = I[343], I[343] = I[344], I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], I[351] = I[352], I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], \
 I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], I[367] = I[368], I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], I[373] = I[374], I[374] = I[375], I[375] = I[376], I[376] = I[377], I[377] = I[378], I[378] = I[379], \
 I[380] = I[381], I[381] = I[382], I[382] = I[383], I[383] = I[384], I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], I[390] = I[391], I[391] = I[392], I[392] = I[393], I[393] = I[394], I[394] = I[395], I[395] = I[396], I[396] = I[397], I[397] = I[398], I[398] = I[399], \
 _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x, ++_n10##x)

#define cimg_get20x20(img,x,y,z,v,I) \
 I[0] = (img)(_p9##x,_p9##y,z,v), I[1] = (img)(_p8##x,_p9##y,z,v), I[2] = (img)(_p7##x,_p9##y,z,v), I[3] = (img)(_p6##x,_p9##y,z,v), I[4] = (img)(_p5##x,_p9##y,z,v), I[5] = (img)(_p4##x,_p9##y,z,v), I[6] = (img)(_p3##x,_p9##y,z,v), I[7] = (img)(_p2##x,_p9##y,z,v), I[8] = (img)(_p1##x,_p9##y,z,v), I[9] = (img)(x,_p9##y,z,v), I[10] = (img)(_n1##x,_p9##y,z,v), I[11] = (img)(_n2##x,_p9##y,z,v), I[12] = (img)(_n3##x,_p9##y,z,v), I[13] = (img)(_n4##x,_p9##y,z,v), I[14] = (img)(_n5##x,_p9##y,z,v), I[15] = (img)(_n6##x,_p9##y,z,v), I[16] = (img)(_n7##x,_p9##y,z,v), I[17] = (img)(_n8##x,_p9##y,z,v), I[18] = (img)(_n9##x,_p9##y,z,v), I[19] = (img)(_n10##x,_p9##y,z,v), \
 I[20] = (img)(_p9##x,_p8##y,z,v), I[21] = (img)(_p8##x,_p8##y,z,v), I[22] = (img)(_p7##x,_p8##y,z,v), I[23] = (img)(_p6##x,_p8##y,z,v), I[24] = (img)(_p5##x,_p8##y,z,v), I[25] = (img)(_p4##x,_p8##y,z,v), I[26] = (img)(_p3##x,_p8##y,z,v), I[27] = (img)(_p2##x,_p8##y,z,v), I[28] = (img)(_p1##x,_p8##y,z,v), I[29] = (img)(x,_p8##y,z,v), I[30] = (img)(_n1##x,_p8##y,z,v), I[31] = (img)(_n2##x,_p8##y,z,v), I[32] = (img)(_n3##x,_p8##y,z,v), I[33] = (img)(_n4##x,_p8##y,z,v), I[34] = (img)(_n5##x,_p8##y,z,v), I[35] = (img)(_n6##x,_p8##y,z,v), I[36] = (img)(_n7##x,_p8##y,z,v), I[37] = (img)(_n8##x,_p8##y,z,v), I[38] = (img)(_n9##x,_p8##y,z,v), I[39] = (img)(_n10##x,_p8##y,z,v), \
 I[40] = (img)(_p9##x,_p7##y,z,v), I[41] = (img)(_p8##x,_p7##y,z,v), I[42] = (img)(_p7##x,_p7##y,z,v), I[43] = (img)(_p6##x,_p7##y,z,v), I[44] = (img)(_p5##x,_p7##y,z,v), I[45] = (img)(_p4##x,_p7##y,z,v), I[46] = (img)(_p3##x,_p7##y,z,v), I[47] = (img)(_p2##x,_p7##y,z,v), I[48] = (img)(_p1##x,_p7##y,z,v), I[49] = (img)(x,_p7##y,z,v), I[50] = (img)(_n1##x,_p7##y,z,v), I[51] = (img)(_n2##x,_p7##y,z,v), I[52] = (img)(_n3##x,_p7##y,z,v), I[53] = (img)(_n4##x,_p7##y,z,v), I[54] = (img)(_n5##x,_p7##y,z,v), I[55] = (img)(_n6##x,_p7##y,z,v), I[56] = (img)(_n7##x,_p7##y,z,v), I[57] = (img)(_n8##x,_p7##y,z,v), I[58] = (img)(_n9##x,_p7##y,z,v), I[59] = (img)(_n10##x,_p7##y,z,v), \
 I[60] = (img)(_p9##x,_p6##y,z,v), I[61] = (img)(_p8##x,_p6##y,z,v), I[62] = (img)(_p7##x,_p6##y,z,v), I[63] = (img)(_p6##x,_p6##y,z,v), I[64] = (img)(_p5##x,_p6##y,z,v), I[65] = (img)(_p4##x,_p6##y,z,v), I[66] = (img)(_p3##x,_p6##y,z,v), I[67] = (img)(_p2##x,_p6##y,z,v), I[68] = (img)(_p1##x,_p6##y,z,v), I[69] = (img)(x,_p6##y,z,v), I[70] = (img)(_n1##x,_p6##y,z,v), I[71] = (img)(_n2##x,_p6##y,z,v), I[72] = (img)(_n3##x,_p6##y,z,v), I[73] = (img)(_n4##x,_p6##y,z,v), I[74] = (img)(_n5##x,_p6##y,z,v), I[75] = (img)(_n6##x,_p6##y,z,v), I[76] = (img)(_n7##x,_p6##y,z,v), I[77] = (img)(_n8##x,_p6##y,z,v), I[78] = (img)(_n9##x,_p6##y,z,v), I[79] = (img)(_n10##x,_p6##y,z,v), \
 I[80] = (img)(_p9##x,_p5##y,z,v), I[81] = (img)(_p8##x,_p5##y,z,v), I[82] = (img)(_p7##x,_p5##y,z,v), I[83] = (img)(_p6##x,_p5##y,z,v), I[84] = (img)(_p5##x,_p5##y,z,v), I[85] = (img)(_p4##x,_p5##y,z,v), I[86] = (img)(_p3##x,_p5##y,z,v), I[87] = (img)(_p2##x,_p5##y,z,v), I[88] = (img)(_p1##x,_p5##y,z,v), I[89] = (img)(x,_p5##y,z,v), I[90] = (img)(_n1##x,_p5##y,z,v), I[91] = (img)(_n2##x,_p5##y,z,v), I[92] = (img)(_n3##x,_p5##y,z,v), I[93] = (img)(_n4##x,_p5##y,z,v), I[94] = (img)(_n5##x,_p5##y,z,v), I[95] = (img)(_n6##x,_p5##y,z,v), I[96] = (img)(_n7##x,_p5##y,z,v), I[97] = (img)(_n8##x,_p5##y,z,v), I[98] = (img)(_n9##x,_p5##y,z,v), I[99] = (img)(_n10##x,_p5##y,z,v), \
 I[100] = (img)(_p9##x,_p4##y,z,v), I[101] = (img)(_p8##x,_p4##y,z,v), I[102] = (img)(_p7##x,_p4##y,z,v), I[103] = (img)(_p6##x,_p4##y,z,v), I[104] = (img)(_p5##x,_p4##y,z,v), I[105] = (img)(_p4##x,_p4##y,z,v), I[106] = (img)(_p3##x,_p4##y,z,v), I[107] = (img)(_p2##x,_p4##y,z,v), I[108] = (img)(_p1##x,_p4##y,z,v), I[109] = (img)(x,_p4##y,z,v), I[110] = (img)(_n1##x,_p4##y,z,v), I[111] = (img)(_n2##x,_p4##y,z,v), I[112] = (img)(_n3##x,_p4##y,z,v), I[113] = (img)(_n4##x,_p4##y,z,v), I[114] = (img)(_n5##x,_p4##y,z,v), I[115] = (img)(_n6##x,_p4##y,z,v), I[116] = (img)(_n7##x,_p4##y,z,v), I[117] = (img)(_n8##x,_p4##y,z,v), I[118] = (img)(_n9##x,_p4##y,z,v), I[119] = (img)(_n10##x,_p4##y,z,v), \
 I[120] = (img)(_p9##x,_p3##y,z,v), I[121] = (img)(_p8##x,_p3##y,z,v), I[122] = (img)(_p7##x,_p3##y,z,v), I[123] = (img)(_p6##x,_p3##y,z,v), I[124] = (img)(_p5##x,_p3##y,z,v), I[125] = (img)(_p4##x,_p3##y,z,v), I[126] = (img)(_p3##x,_p3##y,z,v), I[127] = (img)(_p2##x,_p3##y,z,v), I[128] = (img)(_p1##x,_p3##y,z,v), I[129] = (img)(x,_p3##y,z,v), I[130] = (img)(_n1##x,_p3##y,z,v), I[131] = (img)(_n2##x,_p3##y,z,v), I[132] = (img)(_n3##x,_p3##y,z,v), I[133] = (img)(_n4##x,_p3##y,z,v), I[134] = (img)(_n5##x,_p3##y,z,v), I[135] = (img)(_n6##x,_p3##y,z,v), I[136] = (img)(_n7##x,_p3##y,z,v), I[137] = (img)(_n8##x,_p3##y,z,v), I[138] = (img)(_n9##x,_p3##y,z,v), I[139] = (img)(_n10##x,_p3##y,z,v), \
 I[140] = (img)(_p9##x,_p2##y,z,v), I[141] = (img)(_p8##x,_p2##y,z,v), I[142] = (img)(_p7##x,_p2##y,z,v), I[143] = (img)(_p6##x,_p2##y,z,v), I[144] = (img)(_p5##x,_p2##y,z,v), I[145] = (img)(_p4##x,_p2##y,z,v), I[146] = (img)(_p3##x,_p2##y,z,v), I[147] = (img)(_p2##x,_p2##y,z,v), I[148] = (img)(_p1##x,_p2##y,z,v), I[149] = (img)(x,_p2##y,z,v), I[150] = (img)(_n1##x,_p2##y,z,v), I[151] = (img)(_n2##x,_p2##y,z,v), I[152] = (img)(_n3##x,_p2##y,z,v), I[153] = (img)(_n4##x,_p2##y,z,v), I[154] = (img)(_n5##x,_p2##y,z,v), I[155] = (img)(_n6##x,_p2##y,z,v), I[156] = (img)(_n7##x,_p2##y,z,v), I[157] = (img)(_n8##x,_p2##y,z,v), I[158] = (img)(_n9##x,_p2##y,z,v), I[159] = (img)(_n10##x,_p2##y,z,v), \
 I[160] = (img)(_p9##x,_p1##y,z,v), I[161] = (img)(_p8##x,_p1##y,z,v), I[162] = (img)(_p7##x,_p1##y,z,v), I[163] = (img)(_p6##x,_p1##y,z,v), I[164] = (img)(_p5##x,_p1##y,z,v), I[165] = (img)(_p4##x,_p1##y,z,v), I[166] = (img)(_p3##x,_p1##y,z,v), I[167] = (img)(_p2##x,_p1##y,z,v), I[168] = (img)(_p1##x,_p1##y,z,v), I[169] = (img)(x,_p1##y,z,v), I[170] = (img)(_n1##x,_p1##y,z,v), I[171] = (img)(_n2##x,_p1##y,z,v), I[172] = (img)(_n3##x,_p1##y,z,v), I[173] = (img)(_n4##x,_p1##y,z,v), I[174] = (img)(_n5##x,_p1##y,z,v), I[175] = (img)(_n6##x,_p1##y,z,v), I[176] = (img)(_n7##x,_p1##y,z,v), I[177] = (img)(_n8##x,_p1##y,z,v), I[178] = (img)(_n9##x,_p1##y,z,v), I[179] = (img)(_n10##x,_p1##y,z,v), \
 I[180] = (img)(_p9##x,y,z,v), I[181] = (img)(_p8##x,y,z,v), I[182] = (img)(_p7##x,y,z,v), I[183] = (img)(_p6##x,y,z,v), I[184] = (img)(_p5##x,y,z,v), I[185] = (img)(_p4##x,y,z,v), I[186] = (img)(_p3##x,y,z,v), I[187] = (img)(_p2##x,y,z,v), I[188] = (img)(_p1##x,y,z,v), I[189] = (img)(x,y,z,v), I[190] = (img)(_n1##x,y,z,v), I[191] = (img)(_n2##x,y,z,v), I[192] = (img)(_n3##x,y,z,v), I[193] = (img)(_n4##x,y,z,v), I[194] = (img)(_n5##x,y,z,v), I[195] = (img)(_n6##x,y,z,v), I[196] = (img)(_n7##x,y,z,v), I[197] = (img)(_n8##x,y,z,v), I[198] = (img)(_n9##x,y,z,v), I[199] = (img)(_n10##x,y,z,v), \
 I[200] = (img)(_p9##x,_n1##y,z,v), I[201] = (img)(_p8##x,_n1##y,z,v), I[202] = (img)(_p7##x,_n1##y,z,v), I[203] = (img)(_p6##x,_n1##y,z,v), I[204] = (img)(_p5##x,_n1##y,z,v), I[205] = (img)(_p4##x,_n1##y,z,v), I[206] = (img)(_p3##x,_n1##y,z,v), I[207] = (img)(_p2##x,_n1##y,z,v), I[208] = (img)(_p1##x,_n1##y,z,v), I[209] = (img)(x,_n1##y,z,v), I[210] = (img)(_n1##x,_n1##y,z,v), I[211] = (img)(_n2##x,_n1##y,z,v), I[212] = (img)(_n3##x,_n1##y,z,v), I[213] = (img)(_n4##x,_n1##y,z,v), I[214] = (img)(_n5##x,_n1##y,z,v), I[215] = (img)(_n6##x,_n1##y,z,v), I[216] = (img)(_n7##x,_n1##y,z,v), I[217] = (img)(_n8##x,_n1##y,z,v), I[218] = (img)(_n9##x,_n1##y,z,v), I[219] = (img)(_n10##x,_n1##y,z,v), \
 I[220] = (img)(_p9##x,_n2##y,z,v), I[221] = (img)(_p8##x,_n2##y,z,v), I[222] = (img)(_p7##x,_n2##y,z,v), I[223] = (img)(_p6##x,_n2##y,z,v), I[224] = (img)(_p5##x,_n2##y,z,v), I[225] = (img)(_p4##x,_n2##y,z,v), I[226] = (img)(_p3##x,_n2##y,z,v), I[227] = (img)(_p2##x,_n2##y,z,v), I[228] = (img)(_p1##x,_n2##y,z,v), I[229] = (img)(x,_n2##y,z,v), I[230] = (img)(_n1##x,_n2##y,z,v), I[231] = (img)(_n2##x,_n2##y,z,v), I[232] = (img)(_n3##x,_n2##y,z,v), I[233] = (img)(_n4##x,_n2##y,z,v), I[234] = (img)(_n5##x,_n2##y,z,v), I[235] = (img)(_n6##x,_n2##y,z,v), I[236] = (img)(_n7##x,_n2##y,z,v), I[237] = (img)(_n8##x,_n2##y,z,v), I[238] = (img)(_n9##x,_n2##y,z,v), I[239] = (img)(_n10##x,_n2##y,z,v), \
 I[240] = (img)(_p9##x,_n3##y,z,v), I[241] = (img)(_p8##x,_n3##y,z,v), I[242] = (img)(_p7##x,_n3##y,z,v), I[243] = (img)(_p6##x,_n3##y,z,v), I[244] = (img)(_p5##x,_n3##y,z,v), I[245] = (img)(_p4##x,_n3##y,z,v), I[246] = (img)(_p3##x,_n3##y,z,v), I[247] = (img)(_p2##x,_n3##y,z,v), I[248] = (img)(_p1##x,_n3##y,z,v), I[249] = (img)(x,_n3##y,z,v), I[250] = (img)(_n1##x,_n3##y,z,v), I[251] = (img)(_n2##x,_n3##y,z,v), I[252] = (img)(_n3##x,_n3##y,z,v), I[253] = (img)(_n4##x,_n3##y,z,v), I[254] = (img)(_n5##x,_n3##y,z,v), I[255] = (img)(_n6##x,_n3##y,z,v), I[256] = (img)(_n7##x,_n3##y,z,v), I[257] = (img)(_n8##x,_n3##y,z,v), I[258] = (img)(_n9##x,_n3##y,z,v), I[259] = (img)(_n10##x,_n3##y,z,v), \
 I[260] = (img)(_p9##x,_n4##y,z,v), I[261] = (img)(_p8##x,_n4##y,z,v), I[262] = (img)(_p7##x,_n4##y,z,v), I[263] = (img)(_p6##x,_n4##y,z,v), I[264] = (img)(_p5##x,_n4##y,z,v), I[265] = (img)(_p4##x,_n4##y,z,v), I[266] = (img)(_p3##x,_n4##y,z,v), I[267] = (img)(_p2##x,_n4##y,z,v), I[268] = (img)(_p1##x,_n4##y,z,v), I[269] = (img)(x,_n4##y,z,v), I[270] = (img)(_n1##x,_n4##y,z,v), I[271] = (img)(_n2##x,_n4##y,z,v), I[272] = (img)(_n3##x,_n4##y,z,v), I[273] = (img)(_n4##x,_n4##y,z,v), I[274] = (img)(_n5##x,_n4##y,z,v), I[275] = (img)(_n6##x,_n4##y,z,v), I[276] = (img)(_n7##x,_n4##y,z,v), I[277] = (img)(_n8##x,_n4##y,z,v), I[278] = (img)(_n9##x,_n4##y,z,v), I[279] = (img)(_n10##x,_n4##y,z,v), \
 I[280] = (img)(_p9##x,_n5##y,z,v), I[281] = (img)(_p8##x,_n5##y,z,v), I[282] = (img)(_p7##x,_n5##y,z,v), I[283] = (img)(_p6##x,_n5##y,z,v), I[284] = (img)(_p5##x,_n5##y,z,v), I[285] = (img)(_p4##x,_n5##y,z,v), I[286] = (img)(_p3##x,_n5##y,z,v), I[287] = (img)(_p2##x,_n5##y,z,v), I[288] = (img)(_p1##x,_n5##y,z,v), I[289] = (img)(x,_n5##y,z,v), I[290] = (img)(_n1##x,_n5##y,z,v), I[291] = (img)(_n2##x,_n5##y,z,v), I[292] = (img)(_n3##x,_n5##y,z,v), I[293] = (img)(_n4##x,_n5##y,z,v), I[294] = (img)(_n5##x,_n5##y,z,v), I[295] = (img)(_n6##x,_n5##y,z,v), I[296] = (img)(_n7##x,_n5##y,z,v), I[297] = (img)(_n8##x,_n5##y,z,v), I[298] = (img)(_n9##x,_n5##y,z,v), I[299] = (img)(_n10##x,_n5##y,z,v), \
 I[300] = (img)(_p9##x,_n6##y,z,v), I[301] = (img)(_p8##x,_n6##y,z,v), I[302] = (img)(_p7##x,_n6##y,z,v), I[303] = (img)(_p6##x,_n6##y,z,v), I[304] = (img)(_p5##x,_n6##y,z,v), I[305] = (img)(_p4##x,_n6##y,z,v), I[306] = (img)(_p3##x,_n6##y,z,v), I[307] = (img)(_p2##x,_n6##y,z,v), I[308] = (img)(_p1##x,_n6##y,z,v), I[309] = (img)(x,_n6##y,z,v), I[310] = (img)(_n1##x,_n6##y,z,v), I[311] = (img)(_n2##x,_n6##y,z,v), I[312] = (img)(_n3##x,_n6##y,z,v), I[313] = (img)(_n4##x,_n6##y,z,v), I[314] = (img)(_n5##x,_n6##y,z,v), I[315] = (img)(_n6##x,_n6##y,z,v), I[316] = (img)(_n7##x,_n6##y,z,v), I[317] = (img)(_n8##x,_n6##y,z,v), I[318] = (img)(_n9##x,_n6##y,z,v), I[319] = (img)(_n10##x,_n6##y,z,v), \
 I[320] = (img)(_p9##x,_n7##y,z,v), I[321] = (img)(_p8##x,_n7##y,z,v), I[322] = (img)(_p7##x,_n7##y,z,v), I[323] = (img)(_p6##x,_n7##y,z,v), I[324] = (img)(_p5##x,_n7##y,z,v), I[325] = (img)(_p4##x,_n7##y,z,v), I[326] = (img)(_p3##x,_n7##y,z,v), I[327] = (img)(_p2##x,_n7##y,z,v), I[328] = (img)(_p1##x,_n7##y,z,v), I[329] = (img)(x,_n7##y,z,v), I[330] = (img)(_n1##x,_n7##y,z,v), I[331] = (img)(_n2##x,_n7##y,z,v), I[332] = (img)(_n3##x,_n7##y,z,v), I[333] = (img)(_n4##x,_n7##y,z,v), I[334] = (img)(_n5##x,_n7##y,z,v), I[335] = (img)(_n6##x,_n7##y,z,v), I[336] = (img)(_n7##x,_n7##y,z,v), I[337] = (img)(_n8##x,_n7##y,z,v), I[338] = (img)(_n9##x,_n7##y,z,v), I[339] = (img)(_n10##x,_n7##y,z,v), \
 I[340] = (img)(_p9##x,_n8##y,z,v), I[341] = (img)(_p8##x,_n8##y,z,v), I[342] = (img)(_p7##x,_n8##y,z,v), I[343] = (img)(_p6##x,_n8##y,z,v), I[344] = (img)(_p5##x,_n8##y,z,v), I[345] = (img)(_p4##x,_n8##y,z,v), I[346] = (img)(_p3##x,_n8##y,z,v), I[347] = (img)(_p2##x,_n8##y,z,v), I[348] = (img)(_p1##x,_n8##y,z,v), I[349] = (img)(x,_n8##y,z,v), I[350] = (img)(_n1##x,_n8##y,z,v), I[351] = (img)(_n2##x,_n8##y,z,v), I[352] = (img)(_n3##x,_n8##y,z,v), I[353] = (img)(_n4##x,_n8##y,z,v), I[354] = (img)(_n5##x,_n8##y,z,v), I[355] = (img)(_n6##x,_n8##y,z,v), I[356] = (img)(_n7##x,_n8##y,z,v), I[357] = (img)(_n8##x,_n8##y,z,v), I[358] = (img)(_n9##x,_n8##y,z,v), I[359] = (img)(_n10##x,_n8##y,z,v), \
 I[360] = (img)(_p9##x,_n9##y,z,v), I[361] = (img)(_p8##x,_n9##y,z,v), I[362] = (img)(_p7##x,_n9##y,z,v), I[363] = (img)(_p6##x,_n9##y,z,v), I[364] = (img)(_p5##x,_n9##y,z,v), I[365] = (img)(_p4##x,_n9##y,z,v), I[366] = (img)(_p3##x,_n9##y,z,v), I[367] = (img)(_p2##x,_n9##y,z,v), I[368] = (img)(_p1##x,_n9##y,z,v), I[369] = (img)(x,_n9##y,z,v), I[370] = (img)(_n1##x,_n9##y,z,v), I[371] = (img)(_n2##x,_n9##y,z,v), I[372] = (img)(_n3##x,_n9##y,z,v), I[373] = (img)(_n4##x,_n9##y,z,v), I[374] = (img)(_n5##x,_n9##y,z,v), I[375] = (img)(_n6##x,_n9##y,z,v), I[376] = (img)(_n7##x,_n9##y,z,v), I[377] = (img)(_n8##x,_n9##y,z,v), I[378] = (img)(_n9##x,_n9##y,z,v), I[379] = (img)(_n10##x,_n9##y,z,v), \
 I[380] = (img)(_p9##x,_n10##y,z,v), I[381] = (img)(_p8##x,_n10##y,z,v), I[382] = (img)(_p7##x,_n10##y,z,v), I[383] = (img)(_p6##x,_n10##y,z,v), I[384] = (img)(_p5##x,_n10##y,z,v), I[385] = (img)(_p4##x,_n10##y,z,v), I[386] = (img)(_p3##x,_n10##y,z,v), I[387] = (img)(_p2##x,_n10##y,z,v), I[388] = (img)(_p1##x,_n10##y,z,v), I[389] = (img)(x,_n10##y,z,v), I[390] = (img)(_n1##x,_n10##y,z,v), I[391] = (img)(_n2##x,_n10##y,z,v), I[392] = (img)(_n3##x,_n10##y,z,v), I[393] = (img)(_n4##x,_n10##y,z,v), I[394] = (img)(_n5##x,_n10##y,z,v), I[395] = (img)(_n6##x,_n10##y,z,v), I[396] = (img)(_n7##x,_n10##y,z,v), I[397] = (img)(_n8##x,_n10##y,z,v), I[398] = (img)(_n9##x,_n10##y,z,v), I[399] = (img)(_n10##x,_n10##y,z,v);

// Define 21x21 loop macros for CImg
//----------------------------------
#define cimg_for21(bound,i) for (int i = 0, \
 _p10##i = 0, _p9##i = 0, _p8##i = 0, _p7##i = 0, _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7, \
 _n8##i = 8>=(int)(bound)?(int)(bound)-1:8, \
 _n9##i = 9>=(int)(bound)?(int)(bound)-1:9, \
 _n10##i = 10>=(int)(bound)?(int)(bound)-1:10; \
 _n10##i<(int)(bound) || _n9##i==--_n10##i || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n10##i = _n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p10##i = _p9##i, _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i, ++_n10##i)

#define cimg_for21X(img,x) cimg_for21((img).width,x)
#define cimg_for21Y(img,y) cimg_for21((img).height,y)
#define cimg_for21Z(img,z) cimg_for21((img).depth,z)
#define cimg_for21V(img,v) cimg_for21((img).dim,v)
#define cimg_for21XY(img,x,y) cimg_for21Y(img,y) cimg_for21X(img,x)
#define cimg_for21XZ(img,x,z) cimg_for21Z(img,z) cimg_for21X(img,x)
#define cimg_for21XV(img,x,v) cimg_for21V(img,v) cimg_for21X(img,x)
#define cimg_for21YZ(img,y,z) cimg_for21Z(img,z) cimg_for21Y(img,y)
#define cimg_for21YV(img,y,v) cimg_for21V(img,v) cimg_for21Y(img,y)
#define cimg_for21ZV(img,z,v) cimg_for21V(img,v) cimg_for21Z(img,z)
#define cimg_for21XYZ(img,x,y,z) cimg_for21Z(img,z) cimg_for21XY(img,x,y)
#define cimg_for21XZV(img,x,z,v) cimg_for21V(img,v) cimg_for21XZ(img,x,z)
#define cimg_for21YZV(img,y,z,v) cimg_for21V(img,v) cimg_for21YZ(img,y,z)
#define cimg_for21XYZV(img,x,y,z,v) cimg_for21V(img,v) cimg_for21XYZ(img,x,y,z)

#define cimg_for_in21(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p10##i = i-10<0?0:i-10, \
 _p9##i = i-9<0?0:i-9, \
 _p8##i = i-8<0?0:i-8, \
 _p7##i = i-7<0?0:i-7, \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7, \
 _n8##i = i+8>=(int)(bound)?(int)(bound)-1:i+8, \
 _n9##i = i+9>=(int)(bound)?(int)(bound)-1:i+9, \
 _n10##i = i+10>=(int)(bound)?(int)(bound)-1:i+10; \
 i<=(int)(i1) && (_n10##i<(int)(bound) || _n9##i==--_n10##i || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n10##i = _n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p10##i = _p9##i, _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i, ++_n10##i)

#define cimg_for_in21X(img,x0,x1,x) cimg_for_in21((img).width,x0,x1,x)
#define cimg_for_in21Y(img,y0,y1,y) cimg_for_in21((img).height,y0,y1,y)
#define cimg_for_in21Z(img,z0,z1,z) cimg_for_in21((img).depth,z0,z1,z)
#define cimg_for_in21V(img,v0,v1,v) cimg_for_in21((img).dim,v0,v1,v)
#define cimg_for_in21XY(img,x0,y0,x1,y1,x,y) cimg_for_in21Y(img,y0,y1,y) cimg_for_in21X(img,x0,x1,x)
#define cimg_for_in21XZ(img,x0,z0,x1,z1,x,z) cimg_for_in21Z(img,z0,z1,z) cimg_for_in21X(img,x0,x1,x)
#define cimg_for_in21XV(img,x0,v0,x1,v1,x,v) cimg_for_in21V(img,v0,v1,v) cimg_for_in21X(img,x0,x1,x)
#define cimg_for_in21YZ(img,y0,z0,y1,z1,y,z) cimg_for_in21Z(img,z0,z1,z) cimg_for_in21Y(img,y0,y1,y)
#define cimg_for_in21YV(img,y0,v0,y1,v1,y,v) cimg_for_in21V(img,v0,v1,v) cimg_for_in21Y(img,y0,y1,y)
#define cimg_for_in21ZV(img,z0,v0,z1,v1,z,v) cimg_for_in21V(img,v0,v1,v) cimg_for_in21Z(img,z0,z1,z)
#define cimg_for_in21XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in21Z(img,z0,z1,z) cimg_for_in21XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in21XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in21V(img,v0,v1,v) cimg_for_in21XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in21YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in21V(img,v0,v1,v) cimg_for_in21YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in21XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in21V(img,v0,v1,v) cimg_for_in21XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for21x21(img,x,y,z,v,I) \
 cimg_for21((img).height,y) for (int x = 0, \
 _p10##x = 0, _p9##x = 0, _p8##x = 0, _p7##x = 0, _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = 7>=((img).width)?(int)((img).width)-1:7, \
 _n8##x = 8>=((img).width)?(int)((img).width)-1:8, \
 _n9##x = 9>=((img).width)?(int)((img).width)-1:9, \
 _n10##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = I[7] = I[8] = I[9] = I[10] = (img)(0,_p10##y,z,v)), \
 (I[21] = I[22] = I[23] = I[24] = I[25] = I[26] = I[27] = I[28] = I[29] = I[30] = I[31] = (img)(0,_p9##y,z,v)), \
 (I[42] = I[43] = I[44] = I[45] = I[46] = I[47] = I[48] = I[49] = I[50] = I[51] = I[52] = (img)(0,_p8##y,z,v)), \
 (I[63] = I[64] = I[65] = I[66] = I[67] = I[68] = I[69] = I[70] = I[71] = I[72] = I[73] = (img)(0,_p7##y,z,v)), \
 (I[84] = I[85] = I[86] = I[87] = I[88] = I[89] = I[90] = I[91] = I[92] = I[93] = I[94] = (img)(0,_p6##y,z,v)), \
 (I[105] = I[106] = I[107] = I[108] = I[109] = I[110] = I[111] = I[112] = I[113] = I[114] = I[115] = (img)(0,_p5##y,z,v)), \
 (I[126] = I[127] = I[128] = I[129] = I[130] = I[131] = I[132] = I[133] = I[134] = I[135] = I[136] = (img)(0,_p4##y,z,v)), \
 (I[147] = I[148] = I[149] = I[150] = I[151] = I[152] = I[153] = I[154] = I[155] = I[156] = I[157] = (img)(0,_p3##y,z,v)), \
 (I[168] = I[169] = I[170] = I[171] = I[172] = I[173] = I[174] = I[175] = I[176] = I[177] = I[178] = (img)(0,_p2##y,z,v)), \
 (I[189] = I[190] = I[191] = I[192] = I[193] = I[194] = I[195] = I[196] = I[197] = I[198] = I[199] = (img)(0,_p1##y,z,v)), \
 (I[210] = I[211] = I[212] = I[213] = I[214] = I[215] = I[216] = I[217] = I[218] = I[219] = I[220] = (img)(0,y,z,v)), \
 (I[231] = I[232] = I[233] = I[234] = I[235] = I[236] = I[237] = I[238] = I[239] = I[240] = I[241] = (img)(0,_n1##y,z,v)), \
 (I[252] = I[253] = I[254] = I[255] = I[256] = I[257] = I[258] = I[259] = I[260] = I[261] = I[262] = (img)(0,_n2##y,z,v)), \
 (I[273] = I[274] = I[275] = I[276] = I[277] = I[278] = I[279] = I[280] = I[281] = I[282] = I[283] = (img)(0,_n3##y,z,v)), \
 (I[294] = I[295] = I[296] = I[297] = I[298] = I[299] = I[300] = I[301] = I[302] = I[303] = I[304] = (img)(0,_n4##y,z,v)), \
 (I[315] = I[316] = I[317] = I[318] = I[319] = I[320] = I[321] = I[322] = I[323] = I[324] = I[325] = (img)(0,_n5##y,z,v)), \
 (I[336] = I[337] = I[338] = I[339] = I[340] = I[341] = I[342] = I[343] = I[344] = I[345] = I[346] = (img)(0,_n6##y,z,v)), \
 (I[357] = I[358] = I[359] = I[360] = I[361] = I[362] = I[363] = I[364] = I[365] = I[366] = I[367] = (img)(0,_n7##y,z,v)), \
 (I[378] = I[379] = I[380] = I[381] = I[382] = I[383] = I[384] = I[385] = I[386] = I[387] = I[388] = (img)(0,_n8##y,z,v)), \
 (I[399] = I[400] = I[401] = I[402] = I[403] = I[404] = I[405] = I[406] = I[407] = I[408] = I[409] = (img)(0,_n9##y,z,v)), \
 (I[420] = I[421] = I[422] = I[423] = I[424] = I[425] = I[426] = I[427] = I[428] = I[429] = I[430] = (img)(0,_n10##y,z,v)), \
 (I[11] = (img)(_n1##x,_p10##y,z,v)), \
 (I[32] = (img)(_n1##x,_p9##y,z,v)), \
 (I[53] = (img)(_n1##x,_p8##y,z,v)), \
 (I[74] = (img)(_n1##x,_p7##y,z,v)), \
 (I[95] = (img)(_n1##x,_p6##y,z,v)), \
 (I[116] = (img)(_n1##x,_p5##y,z,v)), \
 (I[137] = (img)(_n1##x,_p4##y,z,v)), \
 (I[158] = (img)(_n1##x,_p3##y,z,v)), \
 (I[179] = (img)(_n1##x,_p2##y,z,v)), \
 (I[200] = (img)(_n1##x,_p1##y,z,v)), \
 (I[221] = (img)(_n1##x,y,z,v)), \
 (I[242] = (img)(_n1##x,_n1##y,z,v)), \
 (I[263] = (img)(_n1##x,_n2##y,z,v)), \
 (I[284] = (img)(_n1##x,_n3##y,z,v)), \
 (I[305] = (img)(_n1##x,_n4##y,z,v)), \
 (I[326] = (img)(_n1##x,_n5##y,z,v)), \
 (I[347] = (img)(_n1##x,_n6##y,z,v)), \
 (I[368] = (img)(_n1##x,_n7##y,z,v)), \
 (I[389] = (img)(_n1##x,_n8##y,z,v)), \
 (I[410] = (img)(_n1##x,_n9##y,z,v)), \
 (I[431] = (img)(_n1##x,_n10##y,z,v)), \
 (I[12] = (img)(_n2##x,_p10##y,z,v)), \
 (I[33] = (img)(_n2##x,_p9##y,z,v)), \
 (I[54] = (img)(_n2##x,_p8##y,z,v)), \
 (I[75] = (img)(_n2##x,_p7##y,z,v)), \
 (I[96] = (img)(_n2##x,_p6##y,z,v)), \
 (I[117] = (img)(_n2##x,_p5##y,z,v)), \
 (I[138] = (img)(_n2##x,_p4##y,z,v)), \
 (I[159] = (img)(_n2##x,_p3##y,z,v)), \
 (I[180] = (img)(_n2##x,_p2##y,z,v)), \
 (I[201] = (img)(_n2##x,_p1##y,z,v)), \
 (I[222] = (img)(_n2##x,y,z,v)), \
 (I[243] = (img)(_n2##x,_n1##y,z,v)), \
 (I[264] = (img)(_n2##x,_n2##y,z,v)), \
 (I[285] = (img)(_n2##x,_n3##y,z,v)), \
 (I[306] = (img)(_n2##x,_n4##y,z,v)), \
 (I[327] = (img)(_n2##x,_n5##y,z,v)), \
 (I[348] = (img)(_n2##x,_n6##y,z,v)), \
 (I[369] = (img)(_n2##x,_n7##y,z,v)), \
 (I[390] = (img)(_n2##x,_n8##y,z,v)), \
 (I[411] = (img)(_n2##x,_n9##y,z,v)), \
 (I[432] = (img)(_n2##x,_n10##y,z,v)), \
 (I[13] = (img)(_n3##x,_p10##y,z,v)), \
 (I[34] = (img)(_n3##x,_p9##y,z,v)), \
 (I[55] = (img)(_n3##x,_p8##y,z,v)), \
 (I[76] = (img)(_n3##x,_p7##y,z,v)), \
 (I[97] = (img)(_n3##x,_p6##y,z,v)), \
 (I[118] = (img)(_n3##x,_p5##y,z,v)), \
 (I[139] = (img)(_n3##x,_p4##y,z,v)), \
 (I[160] = (img)(_n3##x,_p3##y,z,v)), \
 (I[181] = (img)(_n3##x,_p2##y,z,v)), \
 (I[202] = (img)(_n3##x,_p1##y,z,v)), \
 (I[223] = (img)(_n3##x,y,z,v)), \
 (I[244] = (img)(_n3##x,_n1##y,z,v)), \
 (I[265] = (img)(_n3##x,_n2##y,z,v)), \
 (I[286] = (img)(_n3##x,_n3##y,z,v)), \
 (I[307] = (img)(_n3##x,_n4##y,z,v)), \
 (I[328] = (img)(_n3##x,_n5##y,z,v)), \
 (I[349] = (img)(_n3##x,_n6##y,z,v)), \
 (I[370] = (img)(_n3##x,_n7##y,z,v)), \
 (I[391] = (img)(_n3##x,_n8##y,z,v)), \
 (I[412] = (img)(_n3##x,_n9##y,z,v)), \
 (I[433] = (img)(_n3##x,_n10##y,z,v)), \
 (I[14] = (img)(_n4##x,_p10##y,z,v)), \
 (I[35] = (img)(_n4##x,_p9##y,z,v)), \
 (I[56] = (img)(_n4##x,_p8##y,z,v)), \
 (I[77] = (img)(_n4##x,_p7##y,z,v)), \
 (I[98] = (img)(_n4##x,_p6##y,z,v)), \
 (I[119] = (img)(_n4##x,_p5##y,z,v)), \
 (I[140] = (img)(_n4##x,_p4##y,z,v)), \
 (I[161] = (img)(_n4##x,_p3##y,z,v)), \
 (I[182] = (img)(_n4##x,_p2##y,z,v)), \
 (I[203] = (img)(_n4##x,_p1##y,z,v)), \
 (I[224] = (img)(_n4##x,y,z,v)), \
 (I[245] = (img)(_n4##x,_n1##y,z,v)), \
 (I[266] = (img)(_n4##x,_n2##y,z,v)), \
 (I[287] = (img)(_n4##x,_n3##y,z,v)), \
 (I[308] = (img)(_n4##x,_n4##y,z,v)), \
 (I[329] = (img)(_n4##x,_n5##y,z,v)), \
 (I[350] = (img)(_n4##x,_n6##y,z,v)), \
 (I[371] = (img)(_n4##x,_n7##y,z,v)), \
 (I[392] = (img)(_n4##x,_n8##y,z,v)), \
 (I[413] = (img)(_n4##x,_n9##y,z,v)), \
 (I[434] = (img)(_n4##x,_n10##y,z,v)), \
 (I[15] = (img)(_n5##x,_p10##y,z,v)), \
 (I[36] = (img)(_n5##x,_p9##y,z,v)), \
 (I[57] = (img)(_n5##x,_p8##y,z,v)), \
 (I[78] = (img)(_n5##x,_p7##y,z,v)), \
 (I[99] = (img)(_n5##x,_p6##y,z,v)), \
 (I[120] = (img)(_n5##x,_p5##y,z,v)), \
 (I[141] = (img)(_n5##x,_p4##y,z,v)), \
 (I[162] = (img)(_n5##x,_p3##y,z,v)), \
 (I[183] = (img)(_n5##x,_p2##y,z,v)), \
 (I[204] = (img)(_n5##x,_p1##y,z,v)), \
 (I[225] = (img)(_n5##x,y,z,v)), \
 (I[246] = (img)(_n5##x,_n1##y,z,v)), \
 (I[267] = (img)(_n5##x,_n2##y,z,v)), \
 (I[288] = (img)(_n5##x,_n3##y,z,v)), \
 (I[309] = (img)(_n5##x,_n4##y,z,v)), \
 (I[330] = (img)(_n5##x,_n5##y,z,v)), \
 (I[351] = (img)(_n5##x,_n6##y,z,v)), \
 (I[372] = (img)(_n5##x,_n7##y,z,v)), \
 (I[393] = (img)(_n5##x,_n8##y,z,v)), \
 (I[414] = (img)(_n5##x,_n9##y,z,v)), \
 (I[435] = (img)(_n5##x,_n10##y,z,v)), \
 (I[16] = (img)(_n6##x,_p10##y,z,v)), \
 (I[37] = (img)(_n6##x,_p9##y,z,v)), \
 (I[58] = (img)(_n6##x,_p8##y,z,v)), \
 (I[79] = (img)(_n6##x,_p7##y,z,v)), \
 (I[100] = (img)(_n6##x,_p6##y,z,v)), \
 (I[121] = (img)(_n6##x,_p5##y,z,v)), \
 (I[142] = (img)(_n6##x,_p4##y,z,v)), \
 (I[163] = (img)(_n6##x,_p3##y,z,v)), \
 (I[184] = (img)(_n6##x,_p2##y,z,v)), \
 (I[205] = (img)(_n6##x,_p1##y,z,v)), \
 (I[226] = (img)(_n6##x,y,z,v)), \
 (I[247] = (img)(_n6##x,_n1##y,z,v)), \
 (I[268] = (img)(_n6##x,_n2##y,z,v)), \
 (I[289] = (img)(_n6##x,_n3##y,z,v)), \
 (I[310] = (img)(_n6##x,_n4##y,z,v)), \
 (I[331] = (img)(_n6##x,_n5##y,z,v)), \
 (I[352] = (img)(_n6##x,_n6##y,z,v)), \
 (I[373] = (img)(_n6##x,_n7##y,z,v)), \
 (I[394] = (img)(_n6##x,_n8##y,z,v)), \
 (I[415] = (img)(_n6##x,_n9##y,z,v)), \
 (I[436] = (img)(_n6##x,_n10##y,z,v)), \
 (I[17] = (img)(_n7##x,_p10##y,z,v)), \
 (I[38] = (img)(_n7##x,_p9##y,z,v)), \
 (I[59] = (img)(_n7##x,_p8##y,z,v)), \
 (I[80] = (img)(_n7##x,_p7##y,z,v)), \
 (I[101] = (img)(_n7##x,_p6##y,z,v)), \
 (I[122] = (img)(_n7##x,_p5##y,z,v)), \
 (I[143] = (img)(_n7##x,_p4##y,z,v)), \
 (I[164] = (img)(_n7##x,_p3##y,z,v)), \
 (I[185] = (img)(_n7##x,_p2##y,z,v)), \
 (I[206] = (img)(_n7##x,_p1##y,z,v)), \
 (I[227] = (img)(_n7##x,y,z,v)), \
 (I[248] = (img)(_n7##x,_n1##y,z,v)), \
 (I[269] = (img)(_n7##x,_n2##y,z,v)), \
 (I[290] = (img)(_n7##x,_n3##y,z,v)), \
 (I[311] = (img)(_n7##x,_n4##y,z,v)), \
 (I[332] = (img)(_n7##x,_n5##y,z,v)), \
 (I[353] = (img)(_n7##x,_n6##y,z,v)), \
 (I[374] = (img)(_n7##x,_n7##y,z,v)), \
 (I[395] = (img)(_n7##x,_n8##y,z,v)), \
 (I[416] = (img)(_n7##x,_n9##y,z,v)), \
 (I[437] = (img)(_n7##x,_n10##y,z,v)), \
 (I[18] = (img)(_n8##x,_p10##y,z,v)), \
 (I[39] = (img)(_n8##x,_p9##y,z,v)), \
 (I[60] = (img)(_n8##x,_p8##y,z,v)), \
 (I[81] = (img)(_n8##x,_p7##y,z,v)), \
 (I[102] = (img)(_n8##x,_p6##y,z,v)), \
 (I[123] = (img)(_n8##x,_p5##y,z,v)), \
 (I[144] = (img)(_n8##x,_p4##y,z,v)), \
 (I[165] = (img)(_n8##x,_p3##y,z,v)), \
 (I[186] = (img)(_n8##x,_p2##y,z,v)), \
 (I[207] = (img)(_n8##x,_p1##y,z,v)), \
 (I[228] = (img)(_n8##x,y,z,v)), \
 (I[249] = (img)(_n8##x,_n1##y,z,v)), \
 (I[270] = (img)(_n8##x,_n2##y,z,v)), \
 (I[291] = (img)(_n8##x,_n3##y,z,v)), \
 (I[312] = (img)(_n8##x,_n4##y,z,v)), \
 (I[333] = (img)(_n8##x,_n5##y,z,v)), \
 (I[354] = (img)(_n8##x,_n6##y,z,v)), \
 (I[375] = (img)(_n8##x,_n7##y,z,v)), \
 (I[396] = (img)(_n8##x,_n8##y,z,v)), \
 (I[417] = (img)(_n8##x,_n9##y,z,v)), \
 (I[438] = (img)(_n8##x,_n10##y,z,v)), \
 (I[19] = (img)(_n9##x,_p10##y,z,v)), \
 (I[40] = (img)(_n9##x,_p9##y,z,v)), \
 (I[61] = (img)(_n9##x,_p8##y,z,v)), \
 (I[82] = (img)(_n9##x,_p7##y,z,v)), \
 (I[103] = (img)(_n9##x,_p6##y,z,v)), \
 (I[124] = (img)(_n9##x,_p5##y,z,v)), \
 (I[145] = (img)(_n9##x,_p4##y,z,v)), \
 (I[166] = (img)(_n9##x,_p3##y,z,v)), \
 (I[187] = (img)(_n9##x,_p2##y,z,v)), \
 (I[208] = (img)(_n9##x,_p1##y,z,v)), \
 (I[229] = (img)(_n9##x,y,z,v)), \
 (I[250] = (img)(_n9##x,_n1##y,z,v)), \
 (I[271] = (img)(_n9##x,_n2##y,z,v)), \
 (I[292] = (img)(_n9##x,_n3##y,z,v)), \
 (I[313] = (img)(_n9##x,_n4##y,z,v)), \
 (I[334] = (img)(_n9##x,_n5##y,z,v)), \
 (I[355] = (img)(_n9##x,_n6##y,z,v)), \
 (I[376] = (img)(_n9##x,_n7##y,z,v)), \
 (I[397] = (img)(_n9##x,_n8##y,z,v)), \
 (I[418] = (img)(_n9##x,_n9##y,z,v)), \
 (I[439] = (img)(_n9##x,_n10##y,z,v)), \
 10>=((img).width)?(int)((img).width)-1:10); \
 (_n10##x<(int)((img).width) && ( \
 (I[20] = (img)(_n10##x,_p10##y,z,v)), \
 (I[41] = (img)(_n10##x,_p9##y,z,v)), \
 (I[62] = (img)(_n10##x,_p8##y,z,v)), \
 (I[83] = (img)(_n10##x,_p7##y,z,v)), \
 (I[104] = (img)(_n10##x,_p6##y,z,v)), \
 (I[125] = (img)(_n10##x,_p5##y,z,v)), \
 (I[146] = (img)(_n10##x,_p4##y,z,v)), \
 (I[167] = (img)(_n10##x,_p3##y,z,v)), \
 (I[188] = (img)(_n10##x,_p2##y,z,v)), \
 (I[209] = (img)(_n10##x,_p1##y,z,v)), \
 (I[230] = (img)(_n10##x,y,z,v)), \
 (I[251] = (img)(_n10##x,_n1##y,z,v)), \
 (I[272] = (img)(_n10##x,_n2##y,z,v)), \
 (I[293] = (img)(_n10##x,_n3##y,z,v)), \
 (I[314] = (img)(_n10##x,_n4##y,z,v)), \
 (I[335] = (img)(_n10##x,_n5##y,z,v)), \
 (I[356] = (img)(_n10##x,_n6##y,z,v)), \
 (I[377] = (img)(_n10##x,_n7##y,z,v)), \
 (I[398] = (img)(_n10##x,_n8##y,z,v)), \
 (I[419] = (img)(_n10##x,_n9##y,z,v)), \
 (I[440] = (img)(_n10##x,_n10##y,z,v)),1)) || \
 _n9##x==--_n10##x || _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n10##x = _n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], \
 I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], \
 I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], \
 I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], \
 I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], \
 I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], \
 I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], \
 I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], \
 I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], \
 I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], \
 I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], \
 I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], \
 I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], \
 I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], \
 I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], I[321] = I[322], I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], \
 I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], I[342] = I[343], I[343] = I[344], I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], I[351] = I[352], I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], \
 I[357] = I[358], I[358] = I[359], I[359] = I[360], I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], I[367] = I[368], I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], I[373] = I[374], I[374] = I[375], I[375] = I[376], I[376] = I[377], \
 I[378] = I[379], I[379] = I[380], I[380] = I[381], I[381] = I[382], I[382] = I[383], I[383] = I[384], I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], I[390] = I[391], I[391] = I[392], I[392] = I[393], I[393] = I[394], I[394] = I[395], I[395] = I[396], I[396] = I[397], I[397] = I[398], \
 I[399] = I[400], I[400] = I[401], I[401] = I[402], I[402] = I[403], I[403] = I[404], I[404] = I[405], I[405] = I[406], I[406] = I[407], I[407] = I[408], I[408] = I[409], I[409] = I[410], I[410] = I[411], I[411] = I[412], I[412] = I[413], I[413] = I[414], I[414] = I[415], I[415] = I[416], I[416] = I[417], I[417] = I[418], I[418] = I[419], \
 I[420] = I[421], I[421] = I[422], I[422] = I[423], I[423] = I[424], I[424] = I[425], I[425] = I[426], I[426] = I[427], I[427] = I[428], I[428] = I[429], I[429] = I[430], I[430] = I[431], I[431] = I[432], I[432] = I[433], I[433] = I[434], I[434] = I[435], I[435] = I[436], I[436] = I[437], I[437] = I[438], I[438] = I[439], I[439] = I[440], \
 _p10##x = _p9##x, _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x, ++_n10##x)

#define cimg_for_in21x21(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in21((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p10##x = x-10<0?0:x-10, \
 _p9##x = x-9<0?0:x-9, \
 _p8##x = x-8<0?0:x-8, \
 _p7##x = x-7<0?0:x-7, \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = x+7>=(int)((img).width)?(int)((img).width)-1:x+7, \
 _n8##x = x+8>=(int)((img).width)?(int)((img).width)-1:x+8, \
 _n9##x = x+9>=(int)((img).width)?(int)((img).width)-1:x+9, \
 _n10##x = (int)( \
 (I[0] = (img)(_p10##x,_p10##y,z,v)), \
 (I[21] = (img)(_p10##x,_p9##y,z,v)), \
 (I[42] = (img)(_p10##x,_p8##y,z,v)), \
 (I[63] = (img)(_p10##x,_p7##y,z,v)), \
 (I[84] = (img)(_p10##x,_p6##y,z,v)), \
 (I[105] = (img)(_p10##x,_p5##y,z,v)), \
 (I[126] = (img)(_p10##x,_p4##y,z,v)), \
 (I[147] = (img)(_p10##x,_p3##y,z,v)), \
 (I[168] = (img)(_p10##x,_p2##y,z,v)), \
 (I[189] = (img)(_p10##x,_p1##y,z,v)), \
 (I[210] = (img)(_p10##x,y,z,v)), \
 (I[231] = (img)(_p10##x,_n1##y,z,v)), \
 (I[252] = (img)(_p10##x,_n2##y,z,v)), \
 (I[273] = (img)(_p10##x,_n3##y,z,v)), \
 (I[294] = (img)(_p10##x,_n4##y,z,v)), \
 (I[315] = (img)(_p10##x,_n5##y,z,v)), \
 (I[336] = (img)(_p10##x,_n6##y,z,v)), \
 (I[357] = (img)(_p10##x,_n7##y,z,v)), \
 (I[378] = (img)(_p10##x,_n8##y,z,v)), \
 (I[399] = (img)(_p10##x,_n9##y,z,v)), \
 (I[420] = (img)(_p10##x,_n10##y,z,v)), \
 (I[1] = (img)(_p9##x,_p10##y,z,v)), \
 (I[22] = (img)(_p9##x,_p9##y,z,v)), \
 (I[43] = (img)(_p9##x,_p8##y,z,v)), \
 (I[64] = (img)(_p9##x,_p7##y,z,v)), \
 (I[85] = (img)(_p9##x,_p6##y,z,v)), \
 (I[106] = (img)(_p9##x,_p5##y,z,v)), \
 (I[127] = (img)(_p9##x,_p4##y,z,v)), \
 (I[148] = (img)(_p9##x,_p3##y,z,v)), \
 (I[169] = (img)(_p9##x,_p2##y,z,v)), \
 (I[190] = (img)(_p9##x,_p1##y,z,v)), \
 (I[211] = (img)(_p9##x,y,z,v)), \
 (I[232] = (img)(_p9##x,_n1##y,z,v)), \
 (I[253] = (img)(_p9##x,_n2##y,z,v)), \
 (I[274] = (img)(_p9##x,_n3##y,z,v)), \
 (I[295] = (img)(_p9##x,_n4##y,z,v)), \
 (I[316] = (img)(_p9##x,_n5##y,z,v)), \
 (I[337] = (img)(_p9##x,_n6##y,z,v)), \
 (I[358] = (img)(_p9##x,_n7##y,z,v)), \
 (I[379] = (img)(_p9##x,_n8##y,z,v)), \
 (I[400] = (img)(_p9##x,_n9##y,z,v)), \
 (I[421] = (img)(_p9##x,_n10##y,z,v)), \
 (I[2] = (img)(_p8##x,_p10##y,z,v)), \
 (I[23] = (img)(_p8##x,_p9##y,z,v)), \
 (I[44] = (img)(_p8##x,_p8##y,z,v)), \
 (I[65] = (img)(_p8##x,_p7##y,z,v)), \
 (I[86] = (img)(_p8##x,_p6##y,z,v)), \
 (I[107] = (img)(_p8##x,_p5##y,z,v)), \
 (I[128] = (img)(_p8##x,_p4##y,z,v)), \
 (I[149] = (img)(_p8##x,_p3##y,z,v)), \
 (I[170] = (img)(_p8##x,_p2##y,z,v)), \
 (I[191] = (img)(_p8##x,_p1##y,z,v)), \
 (I[212] = (img)(_p8##x,y,z,v)), \
 (I[233] = (img)(_p8##x,_n1##y,z,v)), \
 (I[254] = (img)(_p8##x,_n2##y,z,v)), \
 (I[275] = (img)(_p8##x,_n3##y,z,v)), \
 (I[296] = (img)(_p8##x,_n4##y,z,v)), \
 (I[317] = (img)(_p8##x,_n5##y,z,v)), \
 (I[338] = (img)(_p8##x,_n6##y,z,v)), \
 (I[359] = (img)(_p8##x,_n7##y,z,v)), \
 (I[380] = (img)(_p8##x,_n8##y,z,v)), \
 (I[401] = (img)(_p8##x,_n9##y,z,v)), \
 (I[422] = (img)(_p8##x,_n10##y,z,v)), \
 (I[3] = (img)(_p7##x,_p10##y,z,v)), \
 (I[24] = (img)(_p7##x,_p9##y,z,v)), \
 (I[45] = (img)(_p7##x,_p8##y,z,v)), \
 (I[66] = (img)(_p7##x,_p7##y,z,v)), \
 (I[87] = (img)(_p7##x,_p6##y,z,v)), \
 (I[108] = (img)(_p7##x,_p5##y,z,v)), \
 (I[129] = (img)(_p7##x,_p4##y,z,v)), \
 (I[150] = (img)(_p7##x,_p3##y,z,v)), \
 (I[171] = (img)(_p7##x,_p2##y,z,v)), \
 (I[192] = (img)(_p7##x,_p1##y,z,v)), \
 (I[213] = (img)(_p7##x,y,z,v)), \
 (I[234] = (img)(_p7##x,_n1##y,z,v)), \
 (I[255] = (img)(_p7##x,_n2##y,z,v)), \
 (I[276] = (img)(_p7##x,_n3##y,z,v)), \
 (I[297] = (img)(_p7##x,_n4##y,z,v)), \
 (I[318] = (img)(_p7##x,_n5##y,z,v)), \
 (I[339] = (img)(_p7##x,_n6##y,z,v)), \
 (I[360] = (img)(_p7##x,_n7##y,z,v)), \
 (I[381] = (img)(_p7##x,_n8##y,z,v)), \
 (I[402] = (img)(_p7##x,_n9##y,z,v)), \
 (I[423] = (img)(_p7##x,_n10##y,z,v)), \
 (I[4] = (img)(_p6##x,_p10##y,z,v)), \
 (I[25] = (img)(_p6##x,_p9##y,z,v)), \
 (I[46] = (img)(_p6##x,_p8##y,z,v)), \
 (I[67] = (img)(_p6##x,_p7##y,z,v)), \
 (I[88] = (img)(_p6##x,_p6##y,z,v)), \
 (I[109] = (img)(_p6##x,_p5##y,z,v)), \
 (I[130] = (img)(_p6##x,_p4##y,z,v)), \
 (I[151] = (img)(_p6##x,_p3##y,z,v)), \
 (I[172] = (img)(_p6##x,_p2##y,z,v)), \
 (I[193] = (img)(_p6##x,_p1##y,z,v)), \
 (I[214] = (img)(_p6##x,y,z,v)), \
 (I[235] = (img)(_p6##x,_n1##y,z,v)), \
 (I[256] = (img)(_p6##x,_n2##y,z,v)), \
 (I[277] = (img)(_p6##x,_n3##y,z,v)), \
 (I[298] = (img)(_p6##x,_n4##y,z,v)), \
 (I[319] = (img)(_p6##x,_n5##y,z,v)), \
 (I[340] = (img)(_p6##x,_n6##y,z,v)), \
 (I[361] = (img)(_p6##x,_n7##y,z,v)), \
 (I[382] = (img)(_p6##x,_n8##y,z,v)), \
 (I[403] = (img)(_p6##x,_n9##y,z,v)), \
 (I[424] = (img)(_p6##x,_n10##y,z,v)), \
 (I[5] = (img)(_p5##x,_p10##y,z,v)), \
 (I[26] = (img)(_p5##x,_p9##y,z,v)), \
 (I[47] = (img)(_p5##x,_p8##y,z,v)), \
 (I[68] = (img)(_p5##x,_p7##y,z,v)), \
 (I[89] = (img)(_p5##x,_p6##y,z,v)), \
 (I[110] = (img)(_p5##x,_p5##y,z,v)), \
 (I[131] = (img)(_p5##x,_p4##y,z,v)), \
 (I[152] = (img)(_p5##x,_p3##y,z,v)), \
 (I[173] = (img)(_p5##x,_p2##y,z,v)), \
 (I[194] = (img)(_p5##x,_p1##y,z,v)), \
 (I[215] = (img)(_p5##x,y,z,v)), \
 (I[236] = (img)(_p5##x,_n1##y,z,v)), \
 (I[257] = (img)(_p5##x,_n2##y,z,v)), \
 (I[278] = (img)(_p5##x,_n3##y,z,v)), \
 (I[299] = (img)(_p5##x,_n4##y,z,v)), \
 (I[320] = (img)(_p5##x,_n5##y,z,v)), \
 (I[341] = (img)(_p5##x,_n6##y,z,v)), \
 (I[362] = (img)(_p5##x,_n7##y,z,v)), \
 (I[383] = (img)(_p5##x,_n8##y,z,v)), \
 (I[404] = (img)(_p5##x,_n9##y,z,v)), \
 (I[425] = (img)(_p5##x,_n10##y,z,v)), \
 (I[6] = (img)(_p4##x,_p10##y,z,v)), \
 (I[27] = (img)(_p4##x,_p9##y,z,v)), \
 (I[48] = (img)(_p4##x,_p8##y,z,v)), \
 (I[69] = (img)(_p4##x,_p7##y,z,v)), \
 (I[90] = (img)(_p4##x,_p6##y,z,v)), \
 (I[111] = (img)(_p4##x,_p5##y,z,v)), \
 (I[132] = (img)(_p4##x,_p4##y,z,v)), \
 (I[153] = (img)(_p4##x,_p3##y,z,v)), \
 (I[174] = (img)(_p4##x,_p2##y,z,v)), \
 (I[195] = (img)(_p4##x,_p1##y,z,v)), \
 (I[216] = (img)(_p4##x,y,z,v)), \
 (I[237] = (img)(_p4##x,_n1##y,z,v)), \
 (I[258] = (img)(_p4##x,_n2##y,z,v)), \
 (I[279] = (img)(_p4##x,_n3##y,z,v)), \
 (I[300] = (img)(_p4##x,_n4##y,z,v)), \
 (I[321] = (img)(_p4##x,_n5##y,z,v)), \
 (I[342] = (img)(_p4##x,_n6##y,z,v)), \
 (I[363] = (img)(_p4##x,_n7##y,z,v)), \
 (I[384] = (img)(_p4##x,_n8##y,z,v)), \
 (I[405] = (img)(_p4##x,_n9##y,z,v)), \
 (I[426] = (img)(_p4##x,_n10##y,z,v)), \
 (I[7] = (img)(_p3##x,_p10##y,z,v)), \
 (I[28] = (img)(_p3##x,_p9##y,z,v)), \
 (I[49] = (img)(_p3##x,_p8##y,z,v)), \
 (I[70] = (img)(_p3##x,_p7##y,z,v)), \
 (I[91] = (img)(_p3##x,_p6##y,z,v)), \
 (I[112] = (img)(_p3##x,_p5##y,z,v)), \
 (I[133] = (img)(_p3##x,_p4##y,z,v)), \
 (I[154] = (img)(_p3##x,_p3##y,z,v)), \
 (I[175] = (img)(_p3##x,_p2##y,z,v)), \
 (I[196] = (img)(_p3##x,_p1##y,z,v)), \
 (I[217] = (img)(_p3##x,y,z,v)), \
 (I[238] = (img)(_p3##x,_n1##y,z,v)), \
 (I[259] = (img)(_p3##x,_n2##y,z,v)), \
 (I[280] = (img)(_p3##x,_n3##y,z,v)), \
 (I[301] = (img)(_p3##x,_n4##y,z,v)), \
 (I[322] = (img)(_p3##x,_n5##y,z,v)), \
 (I[343] = (img)(_p3##x,_n6##y,z,v)), \
 (I[364] = (img)(_p3##x,_n7##y,z,v)), \
 (I[385] = (img)(_p3##x,_n8##y,z,v)), \
 (I[406] = (img)(_p3##x,_n9##y,z,v)), \
 (I[427] = (img)(_p3##x,_n10##y,z,v)), \
 (I[8] = (img)(_p2##x,_p10##y,z,v)), \
 (I[29] = (img)(_p2##x,_p9##y,z,v)), \
 (I[50] = (img)(_p2##x,_p8##y,z,v)), \
 (I[71] = (img)(_p2##x,_p7##y,z,v)), \
 (I[92] = (img)(_p2##x,_p6##y,z,v)), \
 (I[113] = (img)(_p2##x,_p5##y,z,v)), \
 (I[134] = (img)(_p2##x,_p4##y,z,v)), \
 (I[155] = (img)(_p2##x,_p3##y,z,v)), \
 (I[176] = (img)(_p2##x,_p2##y,z,v)), \
 (I[197] = (img)(_p2##x,_p1##y,z,v)), \
 (I[218] = (img)(_p2##x,y,z,v)), \
 (I[239] = (img)(_p2##x,_n1##y,z,v)), \
 (I[260] = (img)(_p2##x,_n2##y,z,v)), \
 (I[281] = (img)(_p2##x,_n3##y,z,v)), \
 (I[302] = (img)(_p2##x,_n4##y,z,v)), \
 (I[323] = (img)(_p2##x,_n5##y,z,v)), \
 (I[344] = (img)(_p2##x,_n6##y,z,v)), \
 (I[365] = (img)(_p2##x,_n7##y,z,v)), \
 (I[386] = (img)(_p2##x,_n8##y,z,v)), \
 (I[407] = (img)(_p2##x,_n9##y,z,v)), \
 (I[428] = (img)(_p2##x,_n10##y,z,v)), \
 (I[9] = (img)(_p1##x,_p10##y,z,v)), \
 (I[30] = (img)(_p1##x,_p9##y,z,v)), \
 (I[51] = (img)(_p1##x,_p8##y,z,v)), \
 (I[72] = (img)(_p1##x,_p7##y,z,v)), \
 (I[93] = (img)(_p1##x,_p6##y,z,v)), \
 (I[114] = (img)(_p1##x,_p5##y,z,v)), \
 (I[135] = (img)(_p1##x,_p4##y,z,v)), \
 (I[156] = (img)(_p1##x,_p3##y,z,v)), \
 (I[177] = (img)(_p1##x,_p2##y,z,v)), \
 (I[198] = (img)(_p1##x,_p1##y,z,v)), \
 (I[219] = (img)(_p1##x,y,z,v)), \
 (I[240] = (img)(_p1##x,_n1##y,z,v)), \
 (I[261] = (img)(_p1##x,_n2##y,z,v)), \
 (I[282] = (img)(_p1##x,_n3##y,z,v)), \
 (I[303] = (img)(_p1##x,_n4##y,z,v)), \
 (I[324] = (img)(_p1##x,_n5##y,z,v)), \
 (I[345] = (img)(_p1##x,_n6##y,z,v)), \
 (I[366] = (img)(_p1##x,_n7##y,z,v)), \
 (I[387] = (img)(_p1##x,_n8##y,z,v)), \
 (I[408] = (img)(_p1##x,_n9##y,z,v)), \
 (I[429] = (img)(_p1##x,_n10##y,z,v)), \
 (I[10] = (img)(x,_p10##y,z,v)), \
 (I[31] = (img)(x,_p9##y,z,v)), \
 (I[52] = (img)(x,_p8##y,z,v)), \
 (I[73] = (img)(x,_p7##y,z,v)), \
 (I[94] = (img)(x,_p6##y,z,v)), \
 (I[115] = (img)(x,_p5##y,z,v)), \
 (I[136] = (img)(x,_p4##y,z,v)), \
 (I[157] = (img)(x,_p3##y,z,v)), \
 (I[178] = (img)(x,_p2##y,z,v)), \
 (I[199] = (img)(x,_p1##y,z,v)), \
 (I[220] = (img)(x,y,z,v)), \
 (I[241] = (img)(x,_n1##y,z,v)), \
 (I[262] = (img)(x,_n2##y,z,v)), \
 (I[283] = (img)(x,_n3##y,z,v)), \
 (I[304] = (img)(x,_n4##y,z,v)), \
 (I[325] = (img)(x,_n5##y,z,v)), \
 (I[346] = (img)(x,_n6##y,z,v)), \
 (I[367] = (img)(x,_n7##y,z,v)), \
 (I[388] = (img)(x,_n8##y,z,v)), \
 (I[409] = (img)(x,_n9##y,z,v)), \
 (I[430] = (img)(x,_n10##y,z,v)), \
 (I[11] = (img)(_n1##x,_p10##y,z,v)), \
 (I[32] = (img)(_n1##x,_p9##y,z,v)), \
 (I[53] = (img)(_n1##x,_p8##y,z,v)), \
 (I[74] = (img)(_n1##x,_p7##y,z,v)), \
 (I[95] = (img)(_n1##x,_p6##y,z,v)), \
 (I[116] = (img)(_n1##x,_p5##y,z,v)), \
 (I[137] = (img)(_n1##x,_p4##y,z,v)), \
 (I[158] = (img)(_n1##x,_p3##y,z,v)), \
 (I[179] = (img)(_n1##x,_p2##y,z,v)), \
 (I[200] = (img)(_n1##x,_p1##y,z,v)), \
 (I[221] = (img)(_n1##x,y,z,v)), \
 (I[242] = (img)(_n1##x,_n1##y,z,v)), \
 (I[263] = (img)(_n1##x,_n2##y,z,v)), \
 (I[284] = (img)(_n1##x,_n3##y,z,v)), \
 (I[305] = (img)(_n1##x,_n4##y,z,v)), \
 (I[326] = (img)(_n1##x,_n5##y,z,v)), \
 (I[347] = (img)(_n1##x,_n6##y,z,v)), \
 (I[368] = (img)(_n1##x,_n7##y,z,v)), \
 (I[389] = (img)(_n1##x,_n8##y,z,v)), \
 (I[410] = (img)(_n1##x,_n9##y,z,v)), \
 (I[431] = (img)(_n1##x,_n10##y,z,v)), \
 (I[12] = (img)(_n2##x,_p10##y,z,v)), \
 (I[33] = (img)(_n2##x,_p9##y,z,v)), \
 (I[54] = (img)(_n2##x,_p8##y,z,v)), \
 (I[75] = (img)(_n2##x,_p7##y,z,v)), \
 (I[96] = (img)(_n2##x,_p6##y,z,v)), \
 (I[117] = (img)(_n2##x,_p5##y,z,v)), \
 (I[138] = (img)(_n2##x,_p4##y,z,v)), \
 (I[159] = (img)(_n2##x,_p3##y,z,v)), \
 (I[180] = (img)(_n2##x,_p2##y,z,v)), \
 (I[201] = (img)(_n2##x,_p1##y,z,v)), \
 (I[222] = (img)(_n2##x,y,z,v)), \
 (I[243] = (img)(_n2##x,_n1##y,z,v)), \
 (I[264] = (img)(_n2##x,_n2##y,z,v)), \
 (I[285] = (img)(_n2##x,_n3##y,z,v)), \
 (I[306] = (img)(_n2##x,_n4##y,z,v)), \
 (I[327] = (img)(_n2##x,_n5##y,z,v)), \
 (I[348] = (img)(_n2##x,_n6##y,z,v)), \
 (I[369] = (img)(_n2##x,_n7##y,z,v)), \
 (I[390] = (img)(_n2##x,_n8##y,z,v)), \
 (I[411] = (img)(_n2##x,_n9##y,z,v)), \
 (I[432] = (img)(_n2##x,_n10##y,z,v)), \
 (I[13] = (img)(_n3##x,_p10##y,z,v)), \
 (I[34] = (img)(_n3##x,_p9##y,z,v)), \
 (I[55] = (img)(_n3##x,_p8##y,z,v)), \
 (I[76] = (img)(_n3##x,_p7##y,z,v)), \
 (I[97] = (img)(_n3##x,_p6##y,z,v)), \
 (I[118] = (img)(_n3##x,_p5##y,z,v)), \
 (I[139] = (img)(_n3##x,_p4##y,z,v)), \
 (I[160] = (img)(_n3##x,_p3##y,z,v)), \
 (I[181] = (img)(_n3##x,_p2##y,z,v)), \
 (I[202] = (img)(_n3##x,_p1##y,z,v)), \
 (I[223] = (img)(_n3##x,y,z,v)), \
 (I[244] = (img)(_n3##x,_n1##y,z,v)), \
 (I[265] = (img)(_n3##x,_n2##y,z,v)), \
 (I[286] = (img)(_n3##x,_n3##y,z,v)), \
 (I[307] = (img)(_n3##x,_n4##y,z,v)), \
 (I[328] = (img)(_n3##x,_n5##y,z,v)), \
 (I[349] = (img)(_n3##x,_n6##y,z,v)), \
 (I[370] = (img)(_n3##x,_n7##y,z,v)), \
 (I[391] = (img)(_n3##x,_n8##y,z,v)), \
 (I[412] = (img)(_n3##x,_n9##y,z,v)), \
 (I[433] = (img)(_n3##x,_n10##y,z,v)), \
 (I[14] = (img)(_n4##x,_p10##y,z,v)), \
 (I[35] = (img)(_n4##x,_p9##y,z,v)), \
 (I[56] = (img)(_n4##x,_p8##y,z,v)), \
 (I[77] = (img)(_n4##x,_p7##y,z,v)), \
 (I[98] = (img)(_n4##x,_p6##y,z,v)), \
 (I[119] = (img)(_n4##x,_p5##y,z,v)), \
 (I[140] = (img)(_n4##x,_p4##y,z,v)), \
 (I[161] = (img)(_n4##x,_p3##y,z,v)), \
 (I[182] = (img)(_n4##x,_p2##y,z,v)), \
 (I[203] = (img)(_n4##x,_p1##y,z,v)), \
 (I[224] = (img)(_n4##x,y,z,v)), \
 (I[245] = (img)(_n4##x,_n1##y,z,v)), \
 (I[266] = (img)(_n4##x,_n2##y,z,v)), \
 (I[287] = (img)(_n4##x,_n3##y,z,v)), \
 (I[308] = (img)(_n4##x,_n4##y,z,v)), \
 (I[329] = (img)(_n4##x,_n5##y,z,v)), \
 (I[350] = (img)(_n4##x,_n6##y,z,v)), \
 (I[371] = (img)(_n4##x,_n7##y,z,v)), \
 (I[392] = (img)(_n4##x,_n8##y,z,v)), \
 (I[413] = (img)(_n4##x,_n9##y,z,v)), \
 (I[434] = (img)(_n4##x,_n10##y,z,v)), \
 (I[15] = (img)(_n5##x,_p10##y,z,v)), \
 (I[36] = (img)(_n5##x,_p9##y,z,v)), \
 (I[57] = (img)(_n5##x,_p8##y,z,v)), \
 (I[78] = (img)(_n5##x,_p7##y,z,v)), \
 (I[99] = (img)(_n5##x,_p6##y,z,v)), \
 (I[120] = (img)(_n5##x,_p5##y,z,v)), \
 (I[141] = (img)(_n5##x,_p4##y,z,v)), \
 (I[162] = (img)(_n5##x,_p3##y,z,v)), \
 (I[183] = (img)(_n5##x,_p2##y,z,v)), \
 (I[204] = (img)(_n5##x,_p1##y,z,v)), \
 (I[225] = (img)(_n5##x,y,z,v)), \
 (I[246] = (img)(_n5##x,_n1##y,z,v)), \
 (I[267] = (img)(_n5##x,_n2##y,z,v)), \
 (I[288] = (img)(_n5##x,_n3##y,z,v)), \
 (I[309] = (img)(_n5##x,_n4##y,z,v)), \
 (I[330] = (img)(_n5##x,_n5##y,z,v)), \
 (I[351] = (img)(_n5##x,_n6##y,z,v)), \
 (I[372] = (img)(_n5##x,_n7##y,z,v)), \
 (I[393] = (img)(_n5##x,_n8##y,z,v)), \
 (I[414] = (img)(_n5##x,_n9##y,z,v)), \
 (I[435] = (img)(_n5##x,_n10##y,z,v)), \
 (I[16] = (img)(_n6##x,_p10##y,z,v)), \
 (I[37] = (img)(_n6##x,_p9##y,z,v)), \
 (I[58] = (img)(_n6##x,_p8##y,z,v)), \
 (I[79] = (img)(_n6##x,_p7##y,z,v)), \
 (I[100] = (img)(_n6##x,_p6##y,z,v)), \
 (I[121] = (img)(_n6##x,_p5##y,z,v)), \
 (I[142] = (img)(_n6##x,_p4##y,z,v)), \
 (I[163] = (img)(_n6##x,_p3##y,z,v)), \
 (I[184] = (img)(_n6##x,_p2##y,z,v)), \
 (I[205] = (img)(_n6##x,_p1##y,z,v)), \
 (I[226] = (img)(_n6##x,y,z,v)), \
 (I[247] = (img)(_n6##x,_n1##y,z,v)), \
 (I[268] = (img)(_n6##x,_n2##y,z,v)), \
 (I[289] = (img)(_n6##x,_n3##y,z,v)), \
 (I[310] = (img)(_n6##x,_n4##y,z,v)), \
 (I[331] = (img)(_n6##x,_n5##y,z,v)), \
 (I[352] = (img)(_n6##x,_n6##y,z,v)), \
 (I[373] = (img)(_n6##x,_n7##y,z,v)), \
 (I[394] = (img)(_n6##x,_n8##y,z,v)), \
 (I[415] = (img)(_n6##x,_n9##y,z,v)), \
 (I[436] = (img)(_n6##x,_n10##y,z,v)), \
 (I[17] = (img)(_n7##x,_p10##y,z,v)), \
 (I[38] = (img)(_n7##x,_p9##y,z,v)), \
 (I[59] = (img)(_n7##x,_p8##y,z,v)), \
 (I[80] = (img)(_n7##x,_p7##y,z,v)), \
 (I[101] = (img)(_n7##x,_p6##y,z,v)), \
 (I[122] = (img)(_n7##x,_p5##y,z,v)), \
 (I[143] = (img)(_n7##x,_p4##y,z,v)), \
 (I[164] = (img)(_n7##x,_p3##y,z,v)), \
 (I[185] = (img)(_n7##x,_p2##y,z,v)), \
 (I[206] = (img)(_n7##x,_p1##y,z,v)), \
 (I[227] = (img)(_n7##x,y,z,v)), \
 (I[248] = (img)(_n7##x,_n1##y,z,v)), \
 (I[269] = (img)(_n7##x,_n2##y,z,v)), \
 (I[290] = (img)(_n7##x,_n3##y,z,v)), \
 (I[311] = (img)(_n7##x,_n4##y,z,v)), \
 (I[332] = (img)(_n7##x,_n5##y,z,v)), \
 (I[353] = (img)(_n7##x,_n6##y,z,v)), \
 (I[374] = (img)(_n7##x,_n7##y,z,v)), \
 (I[395] = (img)(_n7##x,_n8##y,z,v)), \
 (I[416] = (img)(_n7##x,_n9##y,z,v)), \
 (I[437] = (img)(_n7##x,_n10##y,z,v)), \
 (I[18] = (img)(_n8##x,_p10##y,z,v)), \
 (I[39] = (img)(_n8##x,_p9##y,z,v)), \
 (I[60] = (img)(_n8##x,_p8##y,z,v)), \
 (I[81] = (img)(_n8##x,_p7##y,z,v)), \
 (I[102] = (img)(_n8##x,_p6##y,z,v)), \
 (I[123] = (img)(_n8##x,_p5##y,z,v)), \
 (I[144] = (img)(_n8##x,_p4##y,z,v)), \
 (I[165] = (img)(_n8##x,_p3##y,z,v)), \
 (I[186] = (img)(_n8##x,_p2##y,z,v)), \
 (I[207] = (img)(_n8##x,_p1##y,z,v)), \
 (I[228] = (img)(_n8##x,y,z,v)), \
 (I[249] = (img)(_n8##x,_n1##y,z,v)), \
 (I[270] = (img)(_n8##x,_n2##y,z,v)), \
 (I[291] = (img)(_n8##x,_n3##y,z,v)), \
 (I[312] = (img)(_n8##x,_n4##y,z,v)), \
 (I[333] = (img)(_n8##x,_n5##y,z,v)), \
 (I[354] = (img)(_n8##x,_n6##y,z,v)), \
 (I[375] = (img)(_n8##x,_n7##y,z,v)), \
 (I[396] = (img)(_n8##x,_n8##y,z,v)), \
 (I[417] = (img)(_n8##x,_n9##y,z,v)), \
 (I[438] = (img)(_n8##x,_n10##y,z,v)), \
 (I[19] = (img)(_n9##x,_p10##y,z,v)), \
 (I[40] = (img)(_n9##x,_p9##y,z,v)), \
 (I[61] = (img)(_n9##x,_p8##y,z,v)), \
 (I[82] = (img)(_n9##x,_p7##y,z,v)), \
 (I[103] = (img)(_n9##x,_p6##y,z,v)), \
 (I[124] = (img)(_n9##x,_p5##y,z,v)), \
 (I[145] = (img)(_n9##x,_p4##y,z,v)), \
 (I[166] = (img)(_n9##x,_p3##y,z,v)), \
 (I[187] = (img)(_n9##x,_p2##y,z,v)), \
 (I[208] = (img)(_n9##x,_p1##y,z,v)), \
 (I[229] = (img)(_n9##x,y,z,v)), \
 (I[250] = (img)(_n9##x,_n1##y,z,v)), \
 (I[271] = (img)(_n9##x,_n2##y,z,v)), \
 (I[292] = (img)(_n9##x,_n3##y,z,v)), \
 (I[313] = (img)(_n9##x,_n4##y,z,v)), \
 (I[334] = (img)(_n9##x,_n5##y,z,v)), \
 (I[355] = (img)(_n9##x,_n6##y,z,v)), \
 (I[376] = (img)(_n9##x,_n7##y,z,v)), \
 (I[397] = (img)(_n9##x,_n8##y,z,v)), \
 (I[418] = (img)(_n9##x,_n9##y,z,v)), \
 (I[439] = (img)(_n9##x,_n10##y,z,v)), \
 x+10>=(int)((img).width)?(int)((img).width)-1:x+10); \
 x<=(int)(x1) && ((_n10##x<(int)((img).width) && ( \
 (I[20] = (img)(_n10##x,_p10##y,z,v)), \
 (I[41] = (img)(_n10##x,_p9##y,z,v)), \
 (I[62] = (img)(_n10##x,_p8##y,z,v)), \
 (I[83] = (img)(_n10##x,_p7##y,z,v)), \
 (I[104] = (img)(_n10##x,_p6##y,z,v)), \
 (I[125] = (img)(_n10##x,_p5##y,z,v)), \
 (I[146] = (img)(_n10##x,_p4##y,z,v)), \
 (I[167] = (img)(_n10##x,_p3##y,z,v)), \
 (I[188] = (img)(_n10##x,_p2##y,z,v)), \
 (I[209] = (img)(_n10##x,_p1##y,z,v)), \
 (I[230] = (img)(_n10##x,y,z,v)), \
 (I[251] = (img)(_n10##x,_n1##y,z,v)), \
 (I[272] = (img)(_n10##x,_n2##y,z,v)), \
 (I[293] = (img)(_n10##x,_n3##y,z,v)), \
 (I[314] = (img)(_n10##x,_n4##y,z,v)), \
 (I[335] = (img)(_n10##x,_n5##y,z,v)), \
 (I[356] = (img)(_n10##x,_n6##y,z,v)), \
 (I[377] = (img)(_n10##x,_n7##y,z,v)), \
 (I[398] = (img)(_n10##x,_n8##y,z,v)), \
 (I[419] = (img)(_n10##x,_n9##y,z,v)), \
 (I[440] = (img)(_n10##x,_n10##y,z,v)),1)) || \
 _n9##x==--_n10##x || _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n10##x = _n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], \
 I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], \
 I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], \
 I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], \
 I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], \
 I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], \
 I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], \
 I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], \
 I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], \
 I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], \
 I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], \
 I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], \
 I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], \
 I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], \
 I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], I[321] = I[322], I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], \
 I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], I[342] = I[343], I[343] = I[344], I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], I[351] = I[352], I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], \
 I[357] = I[358], I[358] = I[359], I[359] = I[360], I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], I[367] = I[368], I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], I[373] = I[374], I[374] = I[375], I[375] = I[376], I[376] = I[377], \
 I[378] = I[379], I[379] = I[380], I[380] = I[381], I[381] = I[382], I[382] = I[383], I[383] = I[384], I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], I[390] = I[391], I[391] = I[392], I[392] = I[393], I[393] = I[394], I[394] = I[395], I[395] = I[396], I[396] = I[397], I[397] = I[398], \
 I[399] = I[400], I[400] = I[401], I[401] = I[402], I[402] = I[403], I[403] = I[404], I[404] = I[405], I[405] = I[406], I[406] = I[407], I[407] = I[408], I[408] = I[409], I[409] = I[410], I[410] = I[411], I[411] = I[412], I[412] = I[413], I[413] = I[414], I[414] = I[415], I[415] = I[416], I[416] = I[417], I[417] = I[418], I[418] = I[419], \
 I[420] = I[421], I[421] = I[422], I[422] = I[423], I[423] = I[424], I[424] = I[425], I[425] = I[426], I[426] = I[427], I[427] = I[428], I[428] = I[429], I[429] = I[430], I[430] = I[431], I[431] = I[432], I[432] = I[433], I[433] = I[434], I[434] = I[435], I[435] = I[436], I[436] = I[437], I[437] = I[438], I[438] = I[439], I[439] = I[440], \
 _p10##x = _p9##x, _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x, ++_n10##x)

#define cimg_get21x21(img,x,y,z,v,I) \
 I[0] = (img)(_p10##x,_p10##y,z,v), I[1] = (img)(_p9##x,_p10##y,z,v), I[2] = (img)(_p8##x,_p10##y,z,v), I[3] = (img)(_p7##x,_p10##y,z,v), I[4] = (img)(_p6##x,_p10##y,z,v), I[5] = (img)(_p5##x,_p10##y,z,v), I[6] = (img)(_p4##x,_p10##y,z,v), I[7] = (img)(_p3##x,_p10##y,z,v), I[8] = (img)(_p2##x,_p10##y,z,v), I[9] = (img)(_p1##x,_p10##y,z,v), I[10] = (img)(x,_p10##y,z,v), I[11] = (img)(_n1##x,_p10##y,z,v), I[12] = (img)(_n2##x,_p10##y,z,v), I[13] = (img)(_n3##x,_p10##y,z,v), I[14] = (img)(_n4##x,_p10##y,z,v), I[15] = (img)(_n5##x,_p10##y,z,v), I[16] = (img)(_n6##x,_p10##y,z,v), I[17] = (img)(_n7##x,_p10##y,z,v), I[18] = (img)(_n8##x,_p10##y,z,v), I[19] = (img)(_n9##x,_p10##y,z,v), I[20] = (img)(_n10##x,_p10##y,z,v), \
 I[21] = (img)(_p10##x,_p9##y,z,v), I[22] = (img)(_p9##x,_p9##y,z,v), I[23] = (img)(_p8##x,_p9##y,z,v), I[24] = (img)(_p7##x,_p9##y,z,v), I[25] = (img)(_p6##x,_p9##y,z,v), I[26] = (img)(_p5##x,_p9##y,z,v), I[27] = (img)(_p4##x,_p9##y,z,v), I[28] = (img)(_p3##x,_p9##y,z,v), I[29] = (img)(_p2##x,_p9##y,z,v), I[30] = (img)(_p1##x,_p9##y,z,v), I[31] = (img)(x,_p9##y,z,v), I[32] = (img)(_n1##x,_p9##y,z,v), I[33] = (img)(_n2##x,_p9##y,z,v), I[34] = (img)(_n3##x,_p9##y,z,v), I[35] = (img)(_n4##x,_p9##y,z,v), I[36] = (img)(_n5##x,_p9##y,z,v), I[37] = (img)(_n6##x,_p9##y,z,v), I[38] = (img)(_n7##x,_p9##y,z,v), I[39] = (img)(_n8##x,_p9##y,z,v), I[40] = (img)(_n9##x,_p9##y,z,v), I[41] = (img)(_n10##x,_p9##y,z,v), \
 I[42] = (img)(_p10##x,_p8##y,z,v), I[43] = (img)(_p9##x,_p8##y,z,v), I[44] = (img)(_p8##x,_p8##y,z,v), I[45] = (img)(_p7##x,_p8##y,z,v), I[46] = (img)(_p6##x,_p8##y,z,v), I[47] = (img)(_p5##x,_p8##y,z,v), I[48] = (img)(_p4##x,_p8##y,z,v), I[49] = (img)(_p3##x,_p8##y,z,v), I[50] = (img)(_p2##x,_p8##y,z,v), I[51] = (img)(_p1##x,_p8##y,z,v), I[52] = (img)(x,_p8##y,z,v), I[53] = (img)(_n1##x,_p8##y,z,v), I[54] = (img)(_n2##x,_p8##y,z,v), I[55] = (img)(_n3##x,_p8##y,z,v), I[56] = (img)(_n4##x,_p8##y,z,v), I[57] = (img)(_n5##x,_p8##y,z,v), I[58] = (img)(_n6##x,_p8##y,z,v), I[59] = (img)(_n7##x,_p8##y,z,v), I[60] = (img)(_n8##x,_p8##y,z,v), I[61] = (img)(_n9##x,_p8##y,z,v), I[62] = (img)(_n10##x,_p8##y,z,v), \
 I[63] = (img)(_p10##x,_p7##y,z,v), I[64] = (img)(_p9##x,_p7##y,z,v), I[65] = (img)(_p8##x,_p7##y,z,v), I[66] = (img)(_p7##x,_p7##y,z,v), I[67] = (img)(_p6##x,_p7##y,z,v), I[68] = (img)(_p5##x,_p7##y,z,v), I[69] = (img)(_p4##x,_p7##y,z,v), I[70] = (img)(_p3##x,_p7##y,z,v), I[71] = (img)(_p2##x,_p7##y,z,v), I[72] = (img)(_p1##x,_p7##y,z,v), I[73] = (img)(x,_p7##y,z,v), I[74] = (img)(_n1##x,_p7##y,z,v), I[75] = (img)(_n2##x,_p7##y,z,v), I[76] = (img)(_n3##x,_p7##y,z,v), I[77] = (img)(_n4##x,_p7##y,z,v), I[78] = (img)(_n5##x,_p7##y,z,v), I[79] = (img)(_n6##x,_p7##y,z,v), I[80] = (img)(_n7##x,_p7##y,z,v), I[81] = (img)(_n8##x,_p7##y,z,v), I[82] = (img)(_n9##x,_p7##y,z,v), I[83] = (img)(_n10##x,_p7##y,z,v), \
 I[84] = (img)(_p10##x,_p6##y,z,v), I[85] = (img)(_p9##x,_p6##y,z,v), I[86] = (img)(_p8##x,_p6##y,z,v), I[87] = (img)(_p7##x,_p6##y,z,v), I[88] = (img)(_p6##x,_p6##y,z,v), I[89] = (img)(_p5##x,_p6##y,z,v), I[90] = (img)(_p4##x,_p6##y,z,v), I[91] = (img)(_p3##x,_p6##y,z,v), I[92] = (img)(_p2##x,_p6##y,z,v), I[93] = (img)(_p1##x,_p6##y,z,v), I[94] = (img)(x,_p6##y,z,v), I[95] = (img)(_n1##x,_p6##y,z,v), I[96] = (img)(_n2##x,_p6##y,z,v), I[97] = (img)(_n3##x,_p6##y,z,v), I[98] = (img)(_n4##x,_p6##y,z,v), I[99] = (img)(_n5##x,_p6##y,z,v), I[100] = (img)(_n6##x,_p6##y,z,v), I[101] = (img)(_n7##x,_p6##y,z,v), I[102] = (img)(_n8##x,_p6##y,z,v), I[103] = (img)(_n9##x,_p6##y,z,v), I[104] = (img)(_n10##x,_p6##y,z,v), \
 I[105] = (img)(_p10##x,_p5##y,z,v), I[106] = (img)(_p9##x,_p5##y,z,v), I[107] = (img)(_p8##x,_p5##y,z,v), I[108] = (img)(_p7##x,_p5##y,z,v), I[109] = (img)(_p6##x,_p5##y,z,v), I[110] = (img)(_p5##x,_p5##y,z,v), I[111] = (img)(_p4##x,_p5##y,z,v), I[112] = (img)(_p3##x,_p5##y,z,v), I[113] = (img)(_p2##x,_p5##y,z,v), I[114] = (img)(_p1##x,_p5##y,z,v), I[115] = (img)(x,_p5##y,z,v), I[116] = (img)(_n1##x,_p5##y,z,v), I[117] = (img)(_n2##x,_p5##y,z,v), I[118] = (img)(_n3##x,_p5##y,z,v), I[119] = (img)(_n4##x,_p5##y,z,v), I[120] = (img)(_n5##x,_p5##y,z,v), I[121] = (img)(_n6##x,_p5##y,z,v), I[122] = (img)(_n7##x,_p5##y,z,v), I[123] = (img)(_n8##x,_p5##y,z,v), I[124] = (img)(_n9##x,_p5##y,z,v), I[125] = (img)(_n10##x,_p5##y,z,v), \
 I[126] = (img)(_p10##x,_p4##y,z,v), I[127] = (img)(_p9##x,_p4##y,z,v), I[128] = (img)(_p8##x,_p4##y,z,v), I[129] = (img)(_p7##x,_p4##y,z,v), I[130] = (img)(_p6##x,_p4##y,z,v), I[131] = (img)(_p5##x,_p4##y,z,v), I[132] = (img)(_p4##x,_p4##y,z,v), I[133] = (img)(_p3##x,_p4##y,z,v), I[134] = (img)(_p2##x,_p4##y,z,v), I[135] = (img)(_p1##x,_p4##y,z,v), I[136] = (img)(x,_p4##y,z,v), I[137] = (img)(_n1##x,_p4##y,z,v), I[138] = (img)(_n2##x,_p4##y,z,v), I[139] = (img)(_n3##x,_p4##y,z,v), I[140] = (img)(_n4##x,_p4##y,z,v), I[141] = (img)(_n5##x,_p4##y,z,v), I[142] = (img)(_n6##x,_p4##y,z,v), I[143] = (img)(_n7##x,_p4##y,z,v), I[144] = (img)(_n8##x,_p4##y,z,v), I[145] = (img)(_n9##x,_p4##y,z,v), I[146] = (img)(_n10##x,_p4##y,z,v), \
 I[147] = (img)(_p10##x,_p3##y,z,v), I[148] = (img)(_p9##x,_p3##y,z,v), I[149] = (img)(_p8##x,_p3##y,z,v), I[150] = (img)(_p7##x,_p3##y,z,v), I[151] = (img)(_p6##x,_p3##y,z,v), I[152] = (img)(_p5##x,_p3##y,z,v), I[153] = (img)(_p4##x,_p3##y,z,v), I[154] = (img)(_p3##x,_p3##y,z,v), I[155] = (img)(_p2##x,_p3##y,z,v), I[156] = (img)(_p1##x,_p3##y,z,v), I[157] = (img)(x,_p3##y,z,v), I[158] = (img)(_n1##x,_p3##y,z,v), I[159] = (img)(_n2##x,_p3##y,z,v), I[160] = (img)(_n3##x,_p3##y,z,v), I[161] = (img)(_n4##x,_p3##y,z,v), I[162] = (img)(_n5##x,_p3##y,z,v), I[163] = (img)(_n6##x,_p3##y,z,v), I[164] = (img)(_n7##x,_p3##y,z,v), I[165] = (img)(_n8##x,_p3##y,z,v), I[166] = (img)(_n9##x,_p3##y,z,v), I[167] = (img)(_n10##x,_p3##y,z,v), \
 I[168] = (img)(_p10##x,_p2##y,z,v), I[169] = (img)(_p9##x,_p2##y,z,v), I[170] = (img)(_p8##x,_p2##y,z,v), I[171] = (img)(_p7##x,_p2##y,z,v), I[172] = (img)(_p6##x,_p2##y,z,v), I[173] = (img)(_p5##x,_p2##y,z,v), I[174] = (img)(_p4##x,_p2##y,z,v), I[175] = (img)(_p3##x,_p2##y,z,v), I[176] = (img)(_p2##x,_p2##y,z,v), I[177] = (img)(_p1##x,_p2##y,z,v), I[178] = (img)(x,_p2##y,z,v), I[179] = (img)(_n1##x,_p2##y,z,v), I[180] = (img)(_n2##x,_p2##y,z,v), I[181] = (img)(_n3##x,_p2##y,z,v), I[182] = (img)(_n4##x,_p2##y,z,v), I[183] = (img)(_n5##x,_p2##y,z,v), I[184] = (img)(_n6##x,_p2##y,z,v), I[185] = (img)(_n7##x,_p2##y,z,v), I[186] = (img)(_n8##x,_p2##y,z,v), I[187] = (img)(_n9##x,_p2##y,z,v), I[188] = (img)(_n10##x,_p2##y,z,v), \
 I[189] = (img)(_p10##x,_p1##y,z,v), I[190] = (img)(_p9##x,_p1##y,z,v), I[191] = (img)(_p8##x,_p1##y,z,v), I[192] = (img)(_p7##x,_p1##y,z,v), I[193] = (img)(_p6##x,_p1##y,z,v), I[194] = (img)(_p5##x,_p1##y,z,v), I[195] = (img)(_p4##x,_p1##y,z,v), I[196] = (img)(_p3##x,_p1##y,z,v), I[197] = (img)(_p2##x,_p1##y,z,v), I[198] = (img)(_p1##x,_p1##y,z,v), I[199] = (img)(x,_p1##y,z,v), I[200] = (img)(_n1##x,_p1##y,z,v), I[201] = (img)(_n2##x,_p1##y,z,v), I[202] = (img)(_n3##x,_p1##y,z,v), I[203] = (img)(_n4##x,_p1##y,z,v), I[204] = (img)(_n5##x,_p1##y,z,v), I[205] = (img)(_n6##x,_p1##y,z,v), I[206] = (img)(_n7##x,_p1##y,z,v), I[207] = (img)(_n8##x,_p1##y,z,v), I[208] = (img)(_n9##x,_p1##y,z,v), I[209] = (img)(_n10##x,_p1##y,z,v), \
 I[210] = (img)(_p10##x,y,z,v), I[211] = (img)(_p9##x,y,z,v), I[212] = (img)(_p8##x,y,z,v), I[213] = (img)(_p7##x,y,z,v), I[214] = (img)(_p6##x,y,z,v), I[215] = (img)(_p5##x,y,z,v), I[216] = (img)(_p4##x,y,z,v), I[217] = (img)(_p3##x,y,z,v), I[218] = (img)(_p2##x,y,z,v), I[219] = (img)(_p1##x,y,z,v), I[220] = (img)(x,y,z,v), I[221] = (img)(_n1##x,y,z,v), I[222] = (img)(_n2##x,y,z,v), I[223] = (img)(_n3##x,y,z,v), I[224] = (img)(_n4##x,y,z,v), I[225] = (img)(_n5##x,y,z,v), I[226] = (img)(_n6##x,y,z,v), I[227] = (img)(_n7##x,y,z,v), I[228] = (img)(_n8##x,y,z,v), I[229] = (img)(_n9##x,y,z,v), I[230] = (img)(_n10##x,y,z,v), \
 I[231] = (img)(_p10##x,_n1##y,z,v), I[232] = (img)(_p9##x,_n1##y,z,v), I[233] = (img)(_p8##x,_n1##y,z,v), I[234] = (img)(_p7##x,_n1##y,z,v), I[235] = (img)(_p6##x,_n1##y,z,v), I[236] = (img)(_p5##x,_n1##y,z,v), I[237] = (img)(_p4##x,_n1##y,z,v), I[238] = (img)(_p3##x,_n1##y,z,v), I[239] = (img)(_p2##x,_n1##y,z,v), I[240] = (img)(_p1##x,_n1##y,z,v), I[241] = (img)(x,_n1##y,z,v), I[242] = (img)(_n1##x,_n1##y,z,v), I[243] = (img)(_n2##x,_n1##y,z,v), I[244] = (img)(_n3##x,_n1##y,z,v), I[245] = (img)(_n4##x,_n1##y,z,v), I[246] = (img)(_n5##x,_n1##y,z,v), I[247] = (img)(_n6##x,_n1##y,z,v), I[248] = (img)(_n7##x,_n1##y,z,v), I[249] = (img)(_n8##x,_n1##y,z,v), I[250] = (img)(_n9##x,_n1##y,z,v), I[251] = (img)(_n10##x,_n1##y,z,v), \
 I[252] = (img)(_p10##x,_n2##y,z,v), I[253] = (img)(_p9##x,_n2##y,z,v), I[254] = (img)(_p8##x,_n2##y,z,v), I[255] = (img)(_p7##x,_n2##y,z,v), I[256] = (img)(_p6##x,_n2##y,z,v), I[257] = (img)(_p5##x,_n2##y,z,v), I[258] = (img)(_p4##x,_n2##y,z,v), I[259] = (img)(_p3##x,_n2##y,z,v), I[260] = (img)(_p2##x,_n2##y,z,v), I[261] = (img)(_p1##x,_n2##y,z,v), I[262] = (img)(x,_n2##y,z,v), I[263] = (img)(_n1##x,_n2##y,z,v), I[264] = (img)(_n2##x,_n2##y,z,v), I[265] = (img)(_n3##x,_n2##y,z,v), I[266] = (img)(_n4##x,_n2##y,z,v), I[267] = (img)(_n5##x,_n2##y,z,v), I[268] = (img)(_n6##x,_n2##y,z,v), I[269] = (img)(_n7##x,_n2##y,z,v), I[270] = (img)(_n8##x,_n2##y,z,v), I[271] = (img)(_n9##x,_n2##y,z,v), I[272] = (img)(_n10##x,_n2##y,z,v), \
 I[273] = (img)(_p10##x,_n3##y,z,v), I[274] = (img)(_p9##x,_n3##y,z,v), I[275] = (img)(_p8##x,_n3##y,z,v), I[276] = (img)(_p7##x,_n3##y,z,v), I[277] = (img)(_p6##x,_n3##y,z,v), I[278] = (img)(_p5##x,_n3##y,z,v), I[279] = (img)(_p4##x,_n3##y,z,v), I[280] = (img)(_p3##x,_n3##y,z,v), I[281] = (img)(_p2##x,_n3##y,z,v), I[282] = (img)(_p1##x,_n3##y,z,v), I[283] = (img)(x,_n3##y,z,v), I[284] = (img)(_n1##x,_n3##y,z,v), I[285] = (img)(_n2##x,_n3##y,z,v), I[286] = (img)(_n3##x,_n3##y,z,v), I[287] = (img)(_n4##x,_n3##y,z,v), I[288] = (img)(_n5##x,_n3##y,z,v), I[289] = (img)(_n6##x,_n3##y,z,v), I[290] = (img)(_n7##x,_n3##y,z,v), I[291] = (img)(_n8##x,_n3##y,z,v), I[292] = (img)(_n9##x,_n3##y,z,v), I[293] = (img)(_n10##x,_n3##y,z,v), \
 I[294] = (img)(_p10##x,_n4##y,z,v), I[295] = (img)(_p9##x,_n4##y,z,v), I[296] = (img)(_p8##x,_n4##y,z,v), I[297] = (img)(_p7##x,_n4##y,z,v), I[298] = (img)(_p6##x,_n4##y,z,v), I[299] = (img)(_p5##x,_n4##y,z,v), I[300] = (img)(_p4##x,_n4##y,z,v), I[301] = (img)(_p3##x,_n4##y,z,v), I[302] = (img)(_p2##x,_n4##y,z,v), I[303] = (img)(_p1##x,_n4##y,z,v), I[304] = (img)(x,_n4##y,z,v), I[305] = (img)(_n1##x,_n4##y,z,v), I[306] = (img)(_n2##x,_n4##y,z,v), I[307] = (img)(_n3##x,_n4##y,z,v), I[308] = (img)(_n4##x,_n4##y,z,v), I[309] = (img)(_n5##x,_n4##y,z,v), I[310] = (img)(_n6##x,_n4##y,z,v), I[311] = (img)(_n7##x,_n4##y,z,v), I[312] = (img)(_n8##x,_n4##y,z,v), I[313] = (img)(_n9##x,_n4##y,z,v), I[314] = (img)(_n10##x,_n4##y,z,v), \
 I[315] = (img)(_p10##x,_n5##y,z,v), I[316] = (img)(_p9##x,_n5##y,z,v), I[317] = (img)(_p8##x,_n5##y,z,v), I[318] = (img)(_p7##x,_n5##y,z,v), I[319] = (img)(_p6##x,_n5##y,z,v), I[320] = (img)(_p5##x,_n5##y,z,v), I[321] = (img)(_p4##x,_n5##y,z,v), I[322] = (img)(_p3##x,_n5##y,z,v), I[323] = (img)(_p2##x,_n5##y,z,v), I[324] = (img)(_p1##x,_n5##y,z,v), I[325] = (img)(x,_n5##y,z,v), I[326] = (img)(_n1##x,_n5##y,z,v), I[327] = (img)(_n2##x,_n5##y,z,v), I[328] = (img)(_n3##x,_n5##y,z,v), I[329] = (img)(_n4##x,_n5##y,z,v), I[330] = (img)(_n5##x,_n5##y,z,v), I[331] = (img)(_n6##x,_n5##y,z,v), I[332] = (img)(_n7##x,_n5##y,z,v), I[333] = (img)(_n8##x,_n5##y,z,v), I[334] = (img)(_n9##x,_n5##y,z,v), I[335] = (img)(_n10##x,_n5##y,z,v), \
 I[336] = (img)(_p10##x,_n6##y,z,v), I[337] = (img)(_p9##x,_n6##y,z,v), I[338] = (img)(_p8##x,_n6##y,z,v), I[339] = (img)(_p7##x,_n6##y,z,v), I[340] = (img)(_p6##x,_n6##y,z,v), I[341] = (img)(_p5##x,_n6##y,z,v), I[342] = (img)(_p4##x,_n6##y,z,v), I[343] = (img)(_p3##x,_n6##y,z,v), I[344] = (img)(_p2##x,_n6##y,z,v), I[345] = (img)(_p1##x,_n6##y,z,v), I[346] = (img)(x,_n6##y,z,v), I[347] = (img)(_n1##x,_n6##y,z,v), I[348] = (img)(_n2##x,_n6##y,z,v), I[349] = (img)(_n3##x,_n6##y,z,v), I[350] = (img)(_n4##x,_n6##y,z,v), I[351] = (img)(_n5##x,_n6##y,z,v), I[352] = (img)(_n6##x,_n6##y,z,v), I[353] = (img)(_n7##x,_n6##y,z,v), I[354] = (img)(_n8##x,_n6##y,z,v), I[355] = (img)(_n9##x,_n6##y,z,v), I[356] = (img)(_n10##x,_n6##y,z,v), \
 I[357] = (img)(_p10##x,_n7##y,z,v), I[358] = (img)(_p9##x,_n7##y,z,v), I[359] = (img)(_p8##x,_n7##y,z,v), I[360] = (img)(_p7##x,_n7##y,z,v), I[361] = (img)(_p6##x,_n7##y,z,v), I[362] = (img)(_p5##x,_n7##y,z,v), I[363] = (img)(_p4##x,_n7##y,z,v), I[364] = (img)(_p3##x,_n7##y,z,v), I[365] = (img)(_p2##x,_n7##y,z,v), I[366] = (img)(_p1##x,_n7##y,z,v), I[367] = (img)(x,_n7##y,z,v), I[368] = (img)(_n1##x,_n7##y,z,v), I[369] = (img)(_n2##x,_n7##y,z,v), I[370] = (img)(_n3##x,_n7##y,z,v), I[371] = (img)(_n4##x,_n7##y,z,v), I[372] = (img)(_n5##x,_n7##y,z,v), I[373] = (img)(_n6##x,_n7##y,z,v), I[374] = (img)(_n7##x,_n7##y,z,v), I[375] = (img)(_n8##x,_n7##y,z,v), I[376] = (img)(_n9##x,_n7##y,z,v), I[377] = (img)(_n10##x,_n7##y,z,v), \
 I[378] = (img)(_p10##x,_n8##y,z,v), I[379] = (img)(_p9##x,_n8##y,z,v), I[380] = (img)(_p8##x,_n8##y,z,v), I[381] = (img)(_p7##x,_n8##y,z,v), I[382] = (img)(_p6##x,_n8##y,z,v), I[383] = (img)(_p5##x,_n8##y,z,v), I[384] = (img)(_p4##x,_n8##y,z,v), I[385] = (img)(_p3##x,_n8##y,z,v), I[386] = (img)(_p2##x,_n8##y,z,v), I[387] = (img)(_p1##x,_n8##y,z,v), I[388] = (img)(x,_n8##y,z,v), I[389] = (img)(_n1##x,_n8##y,z,v), I[390] = (img)(_n2##x,_n8##y,z,v), I[391] = (img)(_n3##x,_n8##y,z,v), I[392] = (img)(_n4##x,_n8##y,z,v), I[393] = (img)(_n5##x,_n8##y,z,v), I[394] = (img)(_n6##x,_n8##y,z,v), I[395] = (img)(_n7##x,_n8##y,z,v), I[396] = (img)(_n8##x,_n8##y,z,v), I[397] = (img)(_n9##x,_n8##y,z,v), I[398] = (img)(_n10##x,_n8##y,z,v), \
 I[399] = (img)(_p10##x,_n9##y,z,v), I[400] = (img)(_p9##x,_n9##y,z,v), I[401] = (img)(_p8##x,_n9##y,z,v), I[402] = (img)(_p7##x,_n9##y,z,v), I[403] = (img)(_p6##x,_n9##y,z,v), I[404] = (img)(_p5##x,_n9##y,z,v), I[405] = (img)(_p4##x,_n9##y,z,v), I[406] = (img)(_p3##x,_n9##y,z,v), I[407] = (img)(_p2##x,_n9##y,z,v), I[408] = (img)(_p1##x,_n9##y,z,v), I[409] = (img)(x,_n9##y,z,v), I[410] = (img)(_n1##x,_n9##y,z,v), I[411] = (img)(_n2##x,_n9##y,z,v), I[412] = (img)(_n3##x,_n9##y,z,v), I[413] = (img)(_n4##x,_n9##y,z,v), I[414] = (img)(_n5##x,_n9##y,z,v), I[415] = (img)(_n6##x,_n9##y,z,v), I[416] = (img)(_n7##x,_n9##y,z,v), I[417] = (img)(_n8##x,_n9##y,z,v), I[418] = (img)(_n9##x,_n9##y,z,v), I[419] = (img)(_n10##x,_n9##y,z,v), \
 I[420] = (img)(_p10##x,_n10##y,z,v), I[421] = (img)(_p9##x,_n10##y,z,v), I[422] = (img)(_p8##x,_n10##y,z,v), I[423] = (img)(_p7##x,_n10##y,z,v), I[424] = (img)(_p6##x,_n10##y,z,v), I[425] = (img)(_p5##x,_n10##y,z,v), I[426] = (img)(_p4##x,_n10##y,z,v), I[427] = (img)(_p3##x,_n10##y,z,v), I[428] = (img)(_p2##x,_n10##y,z,v), I[429] = (img)(_p1##x,_n10##y,z,v), I[430] = (img)(x,_n10##y,z,v), I[431] = (img)(_n1##x,_n10##y,z,v), I[432] = (img)(_n2##x,_n10##y,z,v), I[433] = (img)(_n3##x,_n10##y,z,v), I[434] = (img)(_n4##x,_n10##y,z,v), I[435] = (img)(_n5##x,_n10##y,z,v), I[436] = (img)(_n6##x,_n10##y,z,v), I[437] = (img)(_n7##x,_n10##y,z,v), I[438] = (img)(_n8##x,_n10##y,z,v), I[439] = (img)(_n9##x,_n10##y,z,v), I[440] = (img)(_n10##x,_n10##y,z,v);

// Define 22x22 loop macros for CImg
//----------------------------------
#define cimg_for22(bound,i) for (int i = 0, \
 _p10##i = 0, _p9##i = 0, _p8##i = 0, _p7##i = 0, _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7, \
 _n8##i = 8>=(int)(bound)?(int)(bound)-1:8, \
 _n9##i = 9>=(int)(bound)?(int)(bound)-1:9, \
 _n10##i = 10>=(int)(bound)?(int)(bound)-1:10, \
 _n11##i = 11>=(int)(bound)?(int)(bound)-1:11; \
 _n11##i<(int)(bound) || _n10##i==--_n11##i || _n9##i==--_n10##i || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n11##i = _n10##i = _n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p10##i = _p9##i, _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i, ++_n10##i, ++_n11##i)

#define cimg_for22X(img,x) cimg_for22((img).width,x)
#define cimg_for22Y(img,y) cimg_for22((img).height,y)
#define cimg_for22Z(img,z) cimg_for22((img).depth,z)
#define cimg_for22V(img,v) cimg_for22((img).dim,v)
#define cimg_for22XY(img,x,y) cimg_for22Y(img,y) cimg_for22X(img,x)
#define cimg_for22XZ(img,x,z) cimg_for22Z(img,z) cimg_for22X(img,x)
#define cimg_for22XV(img,x,v) cimg_for22V(img,v) cimg_for22X(img,x)
#define cimg_for22YZ(img,y,z) cimg_for22Z(img,z) cimg_for22Y(img,y)
#define cimg_for22YV(img,y,v) cimg_for22V(img,v) cimg_for22Y(img,y)
#define cimg_for22ZV(img,z,v) cimg_for22V(img,v) cimg_for22Z(img,z)
#define cimg_for22XYZ(img,x,y,z) cimg_for22Z(img,z) cimg_for22XY(img,x,y)
#define cimg_for22XZV(img,x,z,v) cimg_for22V(img,v) cimg_for22XZ(img,x,z)
#define cimg_for22YZV(img,y,z,v) cimg_for22V(img,v) cimg_for22YZ(img,y,z)
#define cimg_for22XYZV(img,x,y,z,v) cimg_for22V(img,v) cimg_for22XYZ(img,x,y,z)

#define cimg_for_in22(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p10##i = i-10<0?0:i-10, \
 _p9##i = i-9<0?0:i-9, \
 _p8##i = i-8<0?0:i-8, \
 _p7##i = i-7<0?0:i-7, \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7, \
 _n8##i = i+8>=(int)(bound)?(int)(bound)-1:i+8, \
 _n9##i = i+9>=(int)(bound)?(int)(bound)-1:i+9, \
 _n10##i = i+10>=(int)(bound)?(int)(bound)-1:i+10, \
 _n11##i = i+11>=(int)(bound)?(int)(bound)-1:i+11; \
 i<=(int)(i1) && (_n11##i<(int)(bound) || _n10##i==--_n11##i || _n9##i==--_n10##i || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n11##i = _n10##i = _n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p10##i = _p9##i, _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i, ++_n10##i, ++_n11##i)

#define cimg_for_in22X(img,x0,x1,x) cimg_for_in22((img).width,x0,x1,x)
#define cimg_for_in22Y(img,y0,y1,y) cimg_for_in22((img).height,y0,y1,y)
#define cimg_for_in22Z(img,z0,z1,z) cimg_for_in22((img).depth,z0,z1,z)
#define cimg_for_in22V(img,v0,v1,v) cimg_for_in22((img).dim,v0,v1,v)
#define cimg_for_in22XY(img,x0,y0,x1,y1,x,y) cimg_for_in22Y(img,y0,y1,y) cimg_for_in22X(img,x0,x1,x)
#define cimg_for_in22XZ(img,x0,z0,x1,z1,x,z) cimg_for_in22Z(img,z0,z1,z) cimg_for_in22X(img,x0,x1,x)
#define cimg_for_in22XV(img,x0,v0,x1,v1,x,v) cimg_for_in22V(img,v0,v1,v) cimg_for_in22X(img,x0,x1,x)
#define cimg_for_in22YZ(img,y0,z0,y1,z1,y,z) cimg_for_in22Z(img,z0,z1,z) cimg_for_in22Y(img,y0,y1,y)
#define cimg_for_in22YV(img,y0,v0,y1,v1,y,v) cimg_for_in22V(img,v0,v1,v) cimg_for_in22Y(img,y0,y1,y)
#define cimg_for_in22ZV(img,z0,v0,z1,v1,z,v) cimg_for_in22V(img,v0,v1,v) cimg_for_in22Z(img,z0,z1,z)
#define cimg_for_in22XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in22Z(img,z0,z1,z) cimg_for_in22XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in22XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in22V(img,v0,v1,v) cimg_for_in22XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in22YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in22V(img,v0,v1,v) cimg_for_in22YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in22XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in22V(img,v0,v1,v) cimg_for_in22XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for22x22(img,x,y,z,v,I) \
 cimg_for22((img).height,y) for (int x = 0, \
 _p10##x = 0, _p9##x = 0, _p8##x = 0, _p7##x = 0, _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = 7>=((img).width)?(int)((img).width)-1:7, \
 _n8##x = 8>=((img).width)?(int)((img).width)-1:8, \
 _n9##x = 9>=((img).width)?(int)((img).width)-1:9, \
 _n10##x = 10>=((img).width)?(int)((img).width)-1:10, \
 _n11##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = I[7] = I[8] = I[9] = I[10] = (img)(0,_p10##y,z,v)), \
 (I[22] = I[23] = I[24] = I[25] = I[26] = I[27] = I[28] = I[29] = I[30] = I[31] = I[32] = (img)(0,_p9##y,z,v)), \
 (I[44] = I[45] = I[46] = I[47] = I[48] = I[49] = I[50] = I[51] = I[52] = I[53] = I[54] = (img)(0,_p8##y,z,v)), \
 (I[66] = I[67] = I[68] = I[69] = I[70] = I[71] = I[72] = I[73] = I[74] = I[75] = I[76] = (img)(0,_p7##y,z,v)), \
 (I[88] = I[89] = I[90] = I[91] = I[92] = I[93] = I[94] = I[95] = I[96] = I[97] = I[98] = (img)(0,_p6##y,z,v)), \
 (I[110] = I[111] = I[112] = I[113] = I[114] = I[115] = I[116] = I[117] = I[118] = I[119] = I[120] = (img)(0,_p5##y,z,v)), \
 (I[132] = I[133] = I[134] = I[135] = I[136] = I[137] = I[138] = I[139] = I[140] = I[141] = I[142] = (img)(0,_p4##y,z,v)), \
 (I[154] = I[155] = I[156] = I[157] = I[158] = I[159] = I[160] = I[161] = I[162] = I[163] = I[164] = (img)(0,_p3##y,z,v)), \
 (I[176] = I[177] = I[178] = I[179] = I[180] = I[181] = I[182] = I[183] = I[184] = I[185] = I[186] = (img)(0,_p2##y,z,v)), \
 (I[198] = I[199] = I[200] = I[201] = I[202] = I[203] = I[204] = I[205] = I[206] = I[207] = I[208] = (img)(0,_p1##y,z,v)), \
 (I[220] = I[221] = I[222] = I[223] = I[224] = I[225] = I[226] = I[227] = I[228] = I[229] = I[230] = (img)(0,y,z,v)), \
 (I[242] = I[243] = I[244] = I[245] = I[246] = I[247] = I[248] = I[249] = I[250] = I[251] = I[252] = (img)(0,_n1##y,z,v)), \
 (I[264] = I[265] = I[266] = I[267] = I[268] = I[269] = I[270] = I[271] = I[272] = I[273] = I[274] = (img)(0,_n2##y,z,v)), \
 (I[286] = I[287] = I[288] = I[289] = I[290] = I[291] = I[292] = I[293] = I[294] = I[295] = I[296] = (img)(0,_n3##y,z,v)), \
 (I[308] = I[309] = I[310] = I[311] = I[312] = I[313] = I[314] = I[315] = I[316] = I[317] = I[318] = (img)(0,_n4##y,z,v)), \
 (I[330] = I[331] = I[332] = I[333] = I[334] = I[335] = I[336] = I[337] = I[338] = I[339] = I[340] = (img)(0,_n5##y,z,v)), \
 (I[352] = I[353] = I[354] = I[355] = I[356] = I[357] = I[358] = I[359] = I[360] = I[361] = I[362] = (img)(0,_n6##y,z,v)), \
 (I[374] = I[375] = I[376] = I[377] = I[378] = I[379] = I[380] = I[381] = I[382] = I[383] = I[384] = (img)(0,_n7##y,z,v)), \
 (I[396] = I[397] = I[398] = I[399] = I[400] = I[401] = I[402] = I[403] = I[404] = I[405] = I[406] = (img)(0,_n8##y,z,v)), \
 (I[418] = I[419] = I[420] = I[421] = I[422] = I[423] = I[424] = I[425] = I[426] = I[427] = I[428] = (img)(0,_n9##y,z,v)), \
 (I[440] = I[441] = I[442] = I[443] = I[444] = I[445] = I[446] = I[447] = I[448] = I[449] = I[450] = (img)(0,_n10##y,z,v)), \
 (I[462] = I[463] = I[464] = I[465] = I[466] = I[467] = I[468] = I[469] = I[470] = I[471] = I[472] = (img)(0,_n11##y,z,v)), \
 (I[11] = (img)(_n1##x,_p10##y,z,v)), \
 (I[33] = (img)(_n1##x,_p9##y,z,v)), \
 (I[55] = (img)(_n1##x,_p8##y,z,v)), \
 (I[77] = (img)(_n1##x,_p7##y,z,v)), \
 (I[99] = (img)(_n1##x,_p6##y,z,v)), \
 (I[121] = (img)(_n1##x,_p5##y,z,v)), \
 (I[143] = (img)(_n1##x,_p4##y,z,v)), \
 (I[165] = (img)(_n1##x,_p3##y,z,v)), \
 (I[187] = (img)(_n1##x,_p2##y,z,v)), \
 (I[209] = (img)(_n1##x,_p1##y,z,v)), \
 (I[231] = (img)(_n1##x,y,z,v)), \
 (I[253] = (img)(_n1##x,_n1##y,z,v)), \
 (I[275] = (img)(_n1##x,_n2##y,z,v)), \
 (I[297] = (img)(_n1##x,_n3##y,z,v)), \
 (I[319] = (img)(_n1##x,_n4##y,z,v)), \
 (I[341] = (img)(_n1##x,_n5##y,z,v)), \
 (I[363] = (img)(_n1##x,_n6##y,z,v)), \
 (I[385] = (img)(_n1##x,_n7##y,z,v)), \
 (I[407] = (img)(_n1##x,_n8##y,z,v)), \
 (I[429] = (img)(_n1##x,_n9##y,z,v)), \
 (I[451] = (img)(_n1##x,_n10##y,z,v)), \
 (I[473] = (img)(_n1##x,_n11##y,z,v)), \
 (I[12] = (img)(_n2##x,_p10##y,z,v)), \
 (I[34] = (img)(_n2##x,_p9##y,z,v)), \
 (I[56] = (img)(_n2##x,_p8##y,z,v)), \
 (I[78] = (img)(_n2##x,_p7##y,z,v)), \
 (I[100] = (img)(_n2##x,_p6##y,z,v)), \
 (I[122] = (img)(_n2##x,_p5##y,z,v)), \
 (I[144] = (img)(_n2##x,_p4##y,z,v)), \
 (I[166] = (img)(_n2##x,_p3##y,z,v)), \
 (I[188] = (img)(_n2##x,_p2##y,z,v)), \
 (I[210] = (img)(_n2##x,_p1##y,z,v)), \
 (I[232] = (img)(_n2##x,y,z,v)), \
 (I[254] = (img)(_n2##x,_n1##y,z,v)), \
 (I[276] = (img)(_n2##x,_n2##y,z,v)), \
 (I[298] = (img)(_n2##x,_n3##y,z,v)), \
 (I[320] = (img)(_n2##x,_n4##y,z,v)), \
 (I[342] = (img)(_n2##x,_n5##y,z,v)), \
 (I[364] = (img)(_n2##x,_n6##y,z,v)), \
 (I[386] = (img)(_n2##x,_n7##y,z,v)), \
 (I[408] = (img)(_n2##x,_n8##y,z,v)), \
 (I[430] = (img)(_n2##x,_n9##y,z,v)), \
 (I[452] = (img)(_n2##x,_n10##y,z,v)), \
 (I[474] = (img)(_n2##x,_n11##y,z,v)), \
 (I[13] = (img)(_n3##x,_p10##y,z,v)), \
 (I[35] = (img)(_n3##x,_p9##y,z,v)), \
 (I[57] = (img)(_n3##x,_p8##y,z,v)), \
 (I[79] = (img)(_n3##x,_p7##y,z,v)), \
 (I[101] = (img)(_n3##x,_p6##y,z,v)), \
 (I[123] = (img)(_n3##x,_p5##y,z,v)), \
 (I[145] = (img)(_n3##x,_p4##y,z,v)), \
 (I[167] = (img)(_n3##x,_p3##y,z,v)), \
 (I[189] = (img)(_n3##x,_p2##y,z,v)), \
 (I[211] = (img)(_n3##x,_p1##y,z,v)), \
 (I[233] = (img)(_n3##x,y,z,v)), \
 (I[255] = (img)(_n3##x,_n1##y,z,v)), \
 (I[277] = (img)(_n3##x,_n2##y,z,v)), \
 (I[299] = (img)(_n3##x,_n3##y,z,v)), \
 (I[321] = (img)(_n3##x,_n4##y,z,v)), \
 (I[343] = (img)(_n3##x,_n5##y,z,v)), \
 (I[365] = (img)(_n3##x,_n6##y,z,v)), \
 (I[387] = (img)(_n3##x,_n7##y,z,v)), \
 (I[409] = (img)(_n3##x,_n8##y,z,v)), \
 (I[431] = (img)(_n3##x,_n9##y,z,v)), \
 (I[453] = (img)(_n3##x,_n10##y,z,v)), \
 (I[475] = (img)(_n3##x,_n11##y,z,v)), \
 (I[14] = (img)(_n4##x,_p10##y,z,v)), \
 (I[36] = (img)(_n4##x,_p9##y,z,v)), \
 (I[58] = (img)(_n4##x,_p8##y,z,v)), \
 (I[80] = (img)(_n4##x,_p7##y,z,v)), \
 (I[102] = (img)(_n4##x,_p6##y,z,v)), \
 (I[124] = (img)(_n4##x,_p5##y,z,v)), \
 (I[146] = (img)(_n4##x,_p4##y,z,v)), \
 (I[168] = (img)(_n4##x,_p3##y,z,v)), \
 (I[190] = (img)(_n4##x,_p2##y,z,v)), \
 (I[212] = (img)(_n4##x,_p1##y,z,v)), \
 (I[234] = (img)(_n4##x,y,z,v)), \
 (I[256] = (img)(_n4##x,_n1##y,z,v)), \
 (I[278] = (img)(_n4##x,_n2##y,z,v)), \
 (I[300] = (img)(_n4##x,_n3##y,z,v)), \
 (I[322] = (img)(_n4##x,_n4##y,z,v)), \
 (I[344] = (img)(_n4##x,_n5##y,z,v)), \
 (I[366] = (img)(_n4##x,_n6##y,z,v)), \
 (I[388] = (img)(_n4##x,_n7##y,z,v)), \
 (I[410] = (img)(_n4##x,_n8##y,z,v)), \
 (I[432] = (img)(_n4##x,_n9##y,z,v)), \
 (I[454] = (img)(_n4##x,_n10##y,z,v)), \
 (I[476] = (img)(_n4##x,_n11##y,z,v)), \
 (I[15] = (img)(_n5##x,_p10##y,z,v)), \
 (I[37] = (img)(_n5##x,_p9##y,z,v)), \
 (I[59] = (img)(_n5##x,_p8##y,z,v)), \
 (I[81] = (img)(_n5##x,_p7##y,z,v)), \
 (I[103] = (img)(_n5##x,_p6##y,z,v)), \
 (I[125] = (img)(_n5##x,_p5##y,z,v)), \
 (I[147] = (img)(_n5##x,_p4##y,z,v)), \
 (I[169] = (img)(_n5##x,_p3##y,z,v)), \
 (I[191] = (img)(_n5##x,_p2##y,z,v)), \
 (I[213] = (img)(_n5##x,_p1##y,z,v)), \
 (I[235] = (img)(_n5##x,y,z,v)), \
 (I[257] = (img)(_n5##x,_n1##y,z,v)), \
 (I[279] = (img)(_n5##x,_n2##y,z,v)), \
 (I[301] = (img)(_n5##x,_n3##y,z,v)), \
 (I[323] = (img)(_n5##x,_n4##y,z,v)), \
 (I[345] = (img)(_n5##x,_n5##y,z,v)), \
 (I[367] = (img)(_n5##x,_n6##y,z,v)), \
 (I[389] = (img)(_n5##x,_n7##y,z,v)), \
 (I[411] = (img)(_n5##x,_n8##y,z,v)), \
 (I[433] = (img)(_n5##x,_n9##y,z,v)), \
 (I[455] = (img)(_n5##x,_n10##y,z,v)), \
 (I[477] = (img)(_n5##x,_n11##y,z,v)), \
 (I[16] = (img)(_n6##x,_p10##y,z,v)), \
 (I[38] = (img)(_n6##x,_p9##y,z,v)), \
 (I[60] = (img)(_n6##x,_p8##y,z,v)), \
 (I[82] = (img)(_n6##x,_p7##y,z,v)), \
 (I[104] = (img)(_n6##x,_p6##y,z,v)), \
 (I[126] = (img)(_n6##x,_p5##y,z,v)), \
 (I[148] = (img)(_n6##x,_p4##y,z,v)), \
 (I[170] = (img)(_n6##x,_p3##y,z,v)), \
 (I[192] = (img)(_n6##x,_p2##y,z,v)), \
 (I[214] = (img)(_n6##x,_p1##y,z,v)), \
 (I[236] = (img)(_n6##x,y,z,v)), \
 (I[258] = (img)(_n6##x,_n1##y,z,v)), \
 (I[280] = (img)(_n6##x,_n2##y,z,v)), \
 (I[302] = (img)(_n6##x,_n3##y,z,v)), \
 (I[324] = (img)(_n6##x,_n4##y,z,v)), \
 (I[346] = (img)(_n6##x,_n5##y,z,v)), \
 (I[368] = (img)(_n6##x,_n6##y,z,v)), \
 (I[390] = (img)(_n6##x,_n7##y,z,v)), \
 (I[412] = (img)(_n6##x,_n8##y,z,v)), \
 (I[434] = (img)(_n6##x,_n9##y,z,v)), \
 (I[456] = (img)(_n6##x,_n10##y,z,v)), \
 (I[478] = (img)(_n6##x,_n11##y,z,v)), \
 (I[17] = (img)(_n7##x,_p10##y,z,v)), \
 (I[39] = (img)(_n7##x,_p9##y,z,v)), \
 (I[61] = (img)(_n7##x,_p8##y,z,v)), \
 (I[83] = (img)(_n7##x,_p7##y,z,v)), \
 (I[105] = (img)(_n7##x,_p6##y,z,v)), \
 (I[127] = (img)(_n7##x,_p5##y,z,v)), \
 (I[149] = (img)(_n7##x,_p4##y,z,v)), \
 (I[171] = (img)(_n7##x,_p3##y,z,v)), \
 (I[193] = (img)(_n7##x,_p2##y,z,v)), \
 (I[215] = (img)(_n7##x,_p1##y,z,v)), \
 (I[237] = (img)(_n7##x,y,z,v)), \
 (I[259] = (img)(_n7##x,_n1##y,z,v)), \
 (I[281] = (img)(_n7##x,_n2##y,z,v)), \
 (I[303] = (img)(_n7##x,_n3##y,z,v)), \
 (I[325] = (img)(_n7##x,_n4##y,z,v)), \
 (I[347] = (img)(_n7##x,_n5##y,z,v)), \
 (I[369] = (img)(_n7##x,_n6##y,z,v)), \
 (I[391] = (img)(_n7##x,_n7##y,z,v)), \
 (I[413] = (img)(_n7##x,_n8##y,z,v)), \
 (I[435] = (img)(_n7##x,_n9##y,z,v)), \
 (I[457] = (img)(_n7##x,_n10##y,z,v)), \
 (I[479] = (img)(_n7##x,_n11##y,z,v)), \
 (I[18] = (img)(_n8##x,_p10##y,z,v)), \
 (I[40] = (img)(_n8##x,_p9##y,z,v)), \
 (I[62] = (img)(_n8##x,_p8##y,z,v)), \
 (I[84] = (img)(_n8##x,_p7##y,z,v)), \
 (I[106] = (img)(_n8##x,_p6##y,z,v)), \
 (I[128] = (img)(_n8##x,_p5##y,z,v)), \
 (I[150] = (img)(_n8##x,_p4##y,z,v)), \
 (I[172] = (img)(_n8##x,_p3##y,z,v)), \
 (I[194] = (img)(_n8##x,_p2##y,z,v)), \
 (I[216] = (img)(_n8##x,_p1##y,z,v)), \
 (I[238] = (img)(_n8##x,y,z,v)), \
 (I[260] = (img)(_n8##x,_n1##y,z,v)), \
 (I[282] = (img)(_n8##x,_n2##y,z,v)), \
 (I[304] = (img)(_n8##x,_n3##y,z,v)), \
 (I[326] = (img)(_n8##x,_n4##y,z,v)), \
 (I[348] = (img)(_n8##x,_n5##y,z,v)), \
 (I[370] = (img)(_n8##x,_n6##y,z,v)), \
 (I[392] = (img)(_n8##x,_n7##y,z,v)), \
 (I[414] = (img)(_n8##x,_n8##y,z,v)), \
 (I[436] = (img)(_n8##x,_n9##y,z,v)), \
 (I[458] = (img)(_n8##x,_n10##y,z,v)), \
 (I[480] = (img)(_n8##x,_n11##y,z,v)), \
 (I[19] = (img)(_n9##x,_p10##y,z,v)), \
 (I[41] = (img)(_n9##x,_p9##y,z,v)), \
 (I[63] = (img)(_n9##x,_p8##y,z,v)), \
 (I[85] = (img)(_n9##x,_p7##y,z,v)), \
 (I[107] = (img)(_n9##x,_p6##y,z,v)), \
 (I[129] = (img)(_n9##x,_p5##y,z,v)), \
 (I[151] = (img)(_n9##x,_p4##y,z,v)), \
 (I[173] = (img)(_n9##x,_p3##y,z,v)), \
 (I[195] = (img)(_n9##x,_p2##y,z,v)), \
 (I[217] = (img)(_n9##x,_p1##y,z,v)), \
 (I[239] = (img)(_n9##x,y,z,v)), \
 (I[261] = (img)(_n9##x,_n1##y,z,v)), \
 (I[283] = (img)(_n9##x,_n2##y,z,v)), \
 (I[305] = (img)(_n9##x,_n3##y,z,v)), \
 (I[327] = (img)(_n9##x,_n4##y,z,v)), \
 (I[349] = (img)(_n9##x,_n5##y,z,v)), \
 (I[371] = (img)(_n9##x,_n6##y,z,v)), \
 (I[393] = (img)(_n9##x,_n7##y,z,v)), \
 (I[415] = (img)(_n9##x,_n8##y,z,v)), \
 (I[437] = (img)(_n9##x,_n9##y,z,v)), \
 (I[459] = (img)(_n9##x,_n10##y,z,v)), \
 (I[481] = (img)(_n9##x,_n11##y,z,v)), \
 (I[20] = (img)(_n10##x,_p10##y,z,v)), \
 (I[42] = (img)(_n10##x,_p9##y,z,v)), \
 (I[64] = (img)(_n10##x,_p8##y,z,v)), \
 (I[86] = (img)(_n10##x,_p7##y,z,v)), \
 (I[108] = (img)(_n10##x,_p6##y,z,v)), \
 (I[130] = (img)(_n10##x,_p5##y,z,v)), \
 (I[152] = (img)(_n10##x,_p4##y,z,v)), \
 (I[174] = (img)(_n10##x,_p3##y,z,v)), \
 (I[196] = (img)(_n10##x,_p2##y,z,v)), \
 (I[218] = (img)(_n10##x,_p1##y,z,v)), \
 (I[240] = (img)(_n10##x,y,z,v)), \
 (I[262] = (img)(_n10##x,_n1##y,z,v)), \
 (I[284] = (img)(_n10##x,_n2##y,z,v)), \
 (I[306] = (img)(_n10##x,_n3##y,z,v)), \
 (I[328] = (img)(_n10##x,_n4##y,z,v)), \
 (I[350] = (img)(_n10##x,_n5##y,z,v)), \
 (I[372] = (img)(_n10##x,_n6##y,z,v)), \
 (I[394] = (img)(_n10##x,_n7##y,z,v)), \
 (I[416] = (img)(_n10##x,_n8##y,z,v)), \
 (I[438] = (img)(_n10##x,_n9##y,z,v)), \
 (I[460] = (img)(_n10##x,_n10##y,z,v)), \
 (I[482] = (img)(_n10##x,_n11##y,z,v)), \
 11>=((img).width)?(int)((img).width)-1:11); \
 (_n11##x<(int)((img).width) && ( \
 (I[21] = (img)(_n11##x,_p10##y,z,v)), \
 (I[43] = (img)(_n11##x,_p9##y,z,v)), \
 (I[65] = (img)(_n11##x,_p8##y,z,v)), \
 (I[87] = (img)(_n11##x,_p7##y,z,v)), \
 (I[109] = (img)(_n11##x,_p6##y,z,v)), \
 (I[131] = (img)(_n11##x,_p5##y,z,v)), \
 (I[153] = (img)(_n11##x,_p4##y,z,v)), \
 (I[175] = (img)(_n11##x,_p3##y,z,v)), \
 (I[197] = (img)(_n11##x,_p2##y,z,v)), \
 (I[219] = (img)(_n11##x,_p1##y,z,v)), \
 (I[241] = (img)(_n11##x,y,z,v)), \
 (I[263] = (img)(_n11##x,_n1##y,z,v)), \
 (I[285] = (img)(_n11##x,_n2##y,z,v)), \
 (I[307] = (img)(_n11##x,_n3##y,z,v)), \
 (I[329] = (img)(_n11##x,_n4##y,z,v)), \
 (I[351] = (img)(_n11##x,_n5##y,z,v)), \
 (I[373] = (img)(_n11##x,_n6##y,z,v)), \
 (I[395] = (img)(_n11##x,_n7##y,z,v)), \
 (I[417] = (img)(_n11##x,_n8##y,z,v)), \
 (I[439] = (img)(_n11##x,_n9##y,z,v)), \
 (I[461] = (img)(_n11##x,_n10##y,z,v)), \
 (I[483] = (img)(_n11##x,_n11##y,z,v)),1)) || \
 _n10##x==--_n11##x || _n9##x==--_n10##x || _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n11##x = _n10##x = _n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], \
 I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], \
 I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], \
 I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], \
 I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], \
 I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], \
 I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], \
 I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], \
 I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], \
 I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], \
 I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], I[239] = I[240], I[240] = I[241], \
 I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], \
 I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], \
 I[286] = I[287], I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], \
 I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], I[321] = I[322], I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], \
 I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], I[335] = I[336], I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], I[342] = I[343], I[343] = I[344], I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], \
 I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], I[359] = I[360], I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], I[367] = I[368], I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], \
 I[374] = I[375], I[375] = I[376], I[376] = I[377], I[377] = I[378], I[378] = I[379], I[379] = I[380], I[380] = I[381], I[381] = I[382], I[382] = I[383], I[383] = I[384], I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], I[390] = I[391], I[391] = I[392], I[392] = I[393], I[393] = I[394], I[394] = I[395], \
 I[396] = I[397], I[397] = I[398], I[398] = I[399], I[399] = I[400], I[400] = I[401], I[401] = I[402], I[402] = I[403], I[403] = I[404], I[404] = I[405], I[405] = I[406], I[406] = I[407], I[407] = I[408], I[408] = I[409], I[409] = I[410], I[410] = I[411], I[411] = I[412], I[412] = I[413], I[413] = I[414], I[414] = I[415], I[415] = I[416], I[416] = I[417], \
 I[418] = I[419], I[419] = I[420], I[420] = I[421], I[421] = I[422], I[422] = I[423], I[423] = I[424], I[424] = I[425], I[425] = I[426], I[426] = I[427], I[427] = I[428], I[428] = I[429], I[429] = I[430], I[430] = I[431], I[431] = I[432], I[432] = I[433], I[433] = I[434], I[434] = I[435], I[435] = I[436], I[436] = I[437], I[437] = I[438], I[438] = I[439], \
 I[440] = I[441], I[441] = I[442], I[442] = I[443], I[443] = I[444], I[444] = I[445], I[445] = I[446], I[446] = I[447], I[447] = I[448], I[448] = I[449], I[449] = I[450], I[450] = I[451], I[451] = I[452], I[452] = I[453], I[453] = I[454], I[454] = I[455], I[455] = I[456], I[456] = I[457], I[457] = I[458], I[458] = I[459], I[459] = I[460], I[460] = I[461], \
 I[462] = I[463], I[463] = I[464], I[464] = I[465], I[465] = I[466], I[466] = I[467], I[467] = I[468], I[468] = I[469], I[469] = I[470], I[470] = I[471], I[471] = I[472], I[472] = I[473], I[473] = I[474], I[474] = I[475], I[475] = I[476], I[476] = I[477], I[477] = I[478], I[478] = I[479], I[479] = I[480], I[480] = I[481], I[481] = I[482], I[482] = I[483], \
 _p10##x = _p9##x, _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x, ++_n10##x, ++_n11##x)

#define cimg_for_in22x22(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in22((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p10##x = x-10<0?0:x-10, \
 _p9##x = x-9<0?0:x-9, \
 _p8##x = x-8<0?0:x-8, \
 _p7##x = x-7<0?0:x-7, \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = x+7>=(int)((img).width)?(int)((img).width)-1:x+7, \
 _n8##x = x+8>=(int)((img).width)?(int)((img).width)-1:x+8, \
 _n9##x = x+9>=(int)((img).width)?(int)((img).width)-1:x+9, \
 _n10##x = x+10>=(int)((img).width)?(int)((img).width)-1:x+10, \
 _n11##x = (int)( \
 (I[0] = (img)(_p10##x,_p10##y,z,v)), \
 (I[22] = (img)(_p10##x,_p9##y,z,v)), \
 (I[44] = (img)(_p10##x,_p8##y,z,v)), \
 (I[66] = (img)(_p10##x,_p7##y,z,v)), \
 (I[88] = (img)(_p10##x,_p6##y,z,v)), \
 (I[110] = (img)(_p10##x,_p5##y,z,v)), \
 (I[132] = (img)(_p10##x,_p4##y,z,v)), \
 (I[154] = (img)(_p10##x,_p3##y,z,v)), \
 (I[176] = (img)(_p10##x,_p2##y,z,v)), \
 (I[198] = (img)(_p10##x,_p1##y,z,v)), \
 (I[220] = (img)(_p10##x,y,z,v)), \
 (I[242] = (img)(_p10##x,_n1##y,z,v)), \
 (I[264] = (img)(_p10##x,_n2##y,z,v)), \
 (I[286] = (img)(_p10##x,_n3##y,z,v)), \
 (I[308] = (img)(_p10##x,_n4##y,z,v)), \
 (I[330] = (img)(_p10##x,_n5##y,z,v)), \
 (I[352] = (img)(_p10##x,_n6##y,z,v)), \
 (I[374] = (img)(_p10##x,_n7##y,z,v)), \
 (I[396] = (img)(_p10##x,_n8##y,z,v)), \
 (I[418] = (img)(_p10##x,_n9##y,z,v)), \
 (I[440] = (img)(_p10##x,_n10##y,z,v)), \
 (I[462] = (img)(_p10##x,_n11##y,z,v)), \
 (I[1] = (img)(_p9##x,_p10##y,z,v)), \
 (I[23] = (img)(_p9##x,_p9##y,z,v)), \
 (I[45] = (img)(_p9##x,_p8##y,z,v)), \
 (I[67] = (img)(_p9##x,_p7##y,z,v)), \
 (I[89] = (img)(_p9##x,_p6##y,z,v)), \
 (I[111] = (img)(_p9##x,_p5##y,z,v)), \
 (I[133] = (img)(_p9##x,_p4##y,z,v)), \
 (I[155] = (img)(_p9##x,_p3##y,z,v)), \
 (I[177] = (img)(_p9##x,_p2##y,z,v)), \
 (I[199] = (img)(_p9##x,_p1##y,z,v)), \
 (I[221] = (img)(_p9##x,y,z,v)), \
 (I[243] = (img)(_p9##x,_n1##y,z,v)), \
 (I[265] = (img)(_p9##x,_n2##y,z,v)), \
 (I[287] = (img)(_p9##x,_n3##y,z,v)), \
 (I[309] = (img)(_p9##x,_n4##y,z,v)), \
 (I[331] = (img)(_p9##x,_n5##y,z,v)), \
 (I[353] = (img)(_p9##x,_n6##y,z,v)), \
 (I[375] = (img)(_p9##x,_n7##y,z,v)), \
 (I[397] = (img)(_p9##x,_n8##y,z,v)), \
 (I[419] = (img)(_p9##x,_n9##y,z,v)), \
 (I[441] = (img)(_p9##x,_n10##y,z,v)), \
 (I[463] = (img)(_p9##x,_n11##y,z,v)), \
 (I[2] = (img)(_p8##x,_p10##y,z,v)), \
 (I[24] = (img)(_p8##x,_p9##y,z,v)), \
 (I[46] = (img)(_p8##x,_p8##y,z,v)), \
 (I[68] = (img)(_p8##x,_p7##y,z,v)), \
 (I[90] = (img)(_p8##x,_p6##y,z,v)), \
 (I[112] = (img)(_p8##x,_p5##y,z,v)), \
 (I[134] = (img)(_p8##x,_p4##y,z,v)), \
 (I[156] = (img)(_p8##x,_p3##y,z,v)), \
 (I[178] = (img)(_p8##x,_p2##y,z,v)), \
 (I[200] = (img)(_p8##x,_p1##y,z,v)), \
 (I[222] = (img)(_p8##x,y,z,v)), \
 (I[244] = (img)(_p8##x,_n1##y,z,v)), \
 (I[266] = (img)(_p8##x,_n2##y,z,v)), \
 (I[288] = (img)(_p8##x,_n3##y,z,v)), \
 (I[310] = (img)(_p8##x,_n4##y,z,v)), \
 (I[332] = (img)(_p8##x,_n5##y,z,v)), \
 (I[354] = (img)(_p8##x,_n6##y,z,v)), \
 (I[376] = (img)(_p8##x,_n7##y,z,v)), \
 (I[398] = (img)(_p8##x,_n8##y,z,v)), \
 (I[420] = (img)(_p8##x,_n9##y,z,v)), \
 (I[442] = (img)(_p8##x,_n10##y,z,v)), \
 (I[464] = (img)(_p8##x,_n11##y,z,v)), \
 (I[3] = (img)(_p7##x,_p10##y,z,v)), \
 (I[25] = (img)(_p7##x,_p9##y,z,v)), \
 (I[47] = (img)(_p7##x,_p8##y,z,v)), \
 (I[69] = (img)(_p7##x,_p7##y,z,v)), \
 (I[91] = (img)(_p7##x,_p6##y,z,v)), \
 (I[113] = (img)(_p7##x,_p5##y,z,v)), \
 (I[135] = (img)(_p7##x,_p4##y,z,v)), \
 (I[157] = (img)(_p7##x,_p3##y,z,v)), \
 (I[179] = (img)(_p7##x,_p2##y,z,v)), \
 (I[201] = (img)(_p7##x,_p1##y,z,v)), \
 (I[223] = (img)(_p7##x,y,z,v)), \
 (I[245] = (img)(_p7##x,_n1##y,z,v)), \
 (I[267] = (img)(_p7##x,_n2##y,z,v)), \
 (I[289] = (img)(_p7##x,_n3##y,z,v)), \
 (I[311] = (img)(_p7##x,_n4##y,z,v)), \
 (I[333] = (img)(_p7##x,_n5##y,z,v)), \
 (I[355] = (img)(_p7##x,_n6##y,z,v)), \
 (I[377] = (img)(_p7##x,_n7##y,z,v)), \
 (I[399] = (img)(_p7##x,_n8##y,z,v)), \
 (I[421] = (img)(_p7##x,_n9##y,z,v)), \
 (I[443] = (img)(_p7##x,_n10##y,z,v)), \
 (I[465] = (img)(_p7##x,_n11##y,z,v)), \
 (I[4] = (img)(_p6##x,_p10##y,z,v)), \
 (I[26] = (img)(_p6##x,_p9##y,z,v)), \
 (I[48] = (img)(_p6##x,_p8##y,z,v)), \
 (I[70] = (img)(_p6##x,_p7##y,z,v)), \
 (I[92] = (img)(_p6##x,_p6##y,z,v)), \
 (I[114] = (img)(_p6##x,_p5##y,z,v)), \
 (I[136] = (img)(_p6##x,_p4##y,z,v)), \
 (I[158] = (img)(_p6##x,_p3##y,z,v)), \
 (I[180] = (img)(_p6##x,_p2##y,z,v)), \
 (I[202] = (img)(_p6##x,_p1##y,z,v)), \
 (I[224] = (img)(_p6##x,y,z,v)), \
 (I[246] = (img)(_p6##x,_n1##y,z,v)), \
 (I[268] = (img)(_p6##x,_n2##y,z,v)), \
 (I[290] = (img)(_p6##x,_n3##y,z,v)), \
 (I[312] = (img)(_p6##x,_n4##y,z,v)), \
 (I[334] = (img)(_p6##x,_n5##y,z,v)), \
 (I[356] = (img)(_p6##x,_n6##y,z,v)), \
 (I[378] = (img)(_p6##x,_n7##y,z,v)), \
 (I[400] = (img)(_p6##x,_n8##y,z,v)), \
 (I[422] = (img)(_p6##x,_n9##y,z,v)), \
 (I[444] = (img)(_p6##x,_n10##y,z,v)), \
 (I[466] = (img)(_p6##x,_n11##y,z,v)), \
 (I[5] = (img)(_p5##x,_p10##y,z,v)), \
 (I[27] = (img)(_p5##x,_p9##y,z,v)), \
 (I[49] = (img)(_p5##x,_p8##y,z,v)), \
 (I[71] = (img)(_p5##x,_p7##y,z,v)), \
 (I[93] = (img)(_p5##x,_p6##y,z,v)), \
 (I[115] = (img)(_p5##x,_p5##y,z,v)), \
 (I[137] = (img)(_p5##x,_p4##y,z,v)), \
 (I[159] = (img)(_p5##x,_p3##y,z,v)), \
 (I[181] = (img)(_p5##x,_p2##y,z,v)), \
 (I[203] = (img)(_p5##x,_p1##y,z,v)), \
 (I[225] = (img)(_p5##x,y,z,v)), \
 (I[247] = (img)(_p5##x,_n1##y,z,v)), \
 (I[269] = (img)(_p5##x,_n2##y,z,v)), \
 (I[291] = (img)(_p5##x,_n3##y,z,v)), \
 (I[313] = (img)(_p5##x,_n4##y,z,v)), \
 (I[335] = (img)(_p5##x,_n5##y,z,v)), \
 (I[357] = (img)(_p5##x,_n6##y,z,v)), \
 (I[379] = (img)(_p5##x,_n7##y,z,v)), \
 (I[401] = (img)(_p5##x,_n8##y,z,v)), \
 (I[423] = (img)(_p5##x,_n9##y,z,v)), \
 (I[445] = (img)(_p5##x,_n10##y,z,v)), \
 (I[467] = (img)(_p5##x,_n11##y,z,v)), \
 (I[6] = (img)(_p4##x,_p10##y,z,v)), \
 (I[28] = (img)(_p4##x,_p9##y,z,v)), \
 (I[50] = (img)(_p4##x,_p8##y,z,v)), \
 (I[72] = (img)(_p4##x,_p7##y,z,v)), \
 (I[94] = (img)(_p4##x,_p6##y,z,v)), \
 (I[116] = (img)(_p4##x,_p5##y,z,v)), \
 (I[138] = (img)(_p4##x,_p4##y,z,v)), \
 (I[160] = (img)(_p4##x,_p3##y,z,v)), \
 (I[182] = (img)(_p4##x,_p2##y,z,v)), \
 (I[204] = (img)(_p4##x,_p1##y,z,v)), \
 (I[226] = (img)(_p4##x,y,z,v)), \
 (I[248] = (img)(_p4##x,_n1##y,z,v)), \
 (I[270] = (img)(_p4##x,_n2##y,z,v)), \
 (I[292] = (img)(_p4##x,_n3##y,z,v)), \
 (I[314] = (img)(_p4##x,_n4##y,z,v)), \
 (I[336] = (img)(_p4##x,_n5##y,z,v)), \
 (I[358] = (img)(_p4##x,_n6##y,z,v)), \
 (I[380] = (img)(_p4##x,_n7##y,z,v)), \
 (I[402] = (img)(_p4##x,_n8##y,z,v)), \
 (I[424] = (img)(_p4##x,_n9##y,z,v)), \
 (I[446] = (img)(_p4##x,_n10##y,z,v)), \
 (I[468] = (img)(_p4##x,_n11##y,z,v)), \
 (I[7] = (img)(_p3##x,_p10##y,z,v)), \
 (I[29] = (img)(_p3##x,_p9##y,z,v)), \
 (I[51] = (img)(_p3##x,_p8##y,z,v)), \
 (I[73] = (img)(_p3##x,_p7##y,z,v)), \
 (I[95] = (img)(_p3##x,_p6##y,z,v)), \
 (I[117] = (img)(_p3##x,_p5##y,z,v)), \
 (I[139] = (img)(_p3##x,_p4##y,z,v)), \
 (I[161] = (img)(_p3##x,_p3##y,z,v)), \
 (I[183] = (img)(_p3##x,_p2##y,z,v)), \
 (I[205] = (img)(_p3##x,_p1##y,z,v)), \
 (I[227] = (img)(_p3##x,y,z,v)), \
 (I[249] = (img)(_p3##x,_n1##y,z,v)), \
 (I[271] = (img)(_p3##x,_n2##y,z,v)), \
 (I[293] = (img)(_p3##x,_n3##y,z,v)), \
 (I[315] = (img)(_p3##x,_n4##y,z,v)), \
 (I[337] = (img)(_p3##x,_n5##y,z,v)), \
 (I[359] = (img)(_p3##x,_n6##y,z,v)), \
 (I[381] = (img)(_p3##x,_n7##y,z,v)), \
 (I[403] = (img)(_p3##x,_n8##y,z,v)), \
 (I[425] = (img)(_p3##x,_n9##y,z,v)), \
 (I[447] = (img)(_p3##x,_n10##y,z,v)), \
 (I[469] = (img)(_p3##x,_n11##y,z,v)), \
 (I[8] = (img)(_p2##x,_p10##y,z,v)), \
 (I[30] = (img)(_p2##x,_p9##y,z,v)), \
 (I[52] = (img)(_p2##x,_p8##y,z,v)), \
 (I[74] = (img)(_p2##x,_p7##y,z,v)), \
 (I[96] = (img)(_p2##x,_p6##y,z,v)), \
 (I[118] = (img)(_p2##x,_p5##y,z,v)), \
 (I[140] = (img)(_p2##x,_p4##y,z,v)), \
 (I[162] = (img)(_p2##x,_p3##y,z,v)), \
 (I[184] = (img)(_p2##x,_p2##y,z,v)), \
 (I[206] = (img)(_p2##x,_p1##y,z,v)), \
 (I[228] = (img)(_p2##x,y,z,v)), \
 (I[250] = (img)(_p2##x,_n1##y,z,v)), \
 (I[272] = (img)(_p2##x,_n2##y,z,v)), \
 (I[294] = (img)(_p2##x,_n3##y,z,v)), \
 (I[316] = (img)(_p2##x,_n4##y,z,v)), \
 (I[338] = (img)(_p2##x,_n5##y,z,v)), \
 (I[360] = (img)(_p2##x,_n6##y,z,v)), \
 (I[382] = (img)(_p2##x,_n7##y,z,v)), \
 (I[404] = (img)(_p2##x,_n8##y,z,v)), \
 (I[426] = (img)(_p2##x,_n9##y,z,v)), \
 (I[448] = (img)(_p2##x,_n10##y,z,v)), \
 (I[470] = (img)(_p2##x,_n11##y,z,v)), \
 (I[9] = (img)(_p1##x,_p10##y,z,v)), \
 (I[31] = (img)(_p1##x,_p9##y,z,v)), \
 (I[53] = (img)(_p1##x,_p8##y,z,v)), \
 (I[75] = (img)(_p1##x,_p7##y,z,v)), \
 (I[97] = (img)(_p1##x,_p6##y,z,v)), \
 (I[119] = (img)(_p1##x,_p5##y,z,v)), \
 (I[141] = (img)(_p1##x,_p4##y,z,v)), \
 (I[163] = (img)(_p1##x,_p3##y,z,v)), \
 (I[185] = (img)(_p1##x,_p2##y,z,v)), \
 (I[207] = (img)(_p1##x,_p1##y,z,v)), \
 (I[229] = (img)(_p1##x,y,z,v)), \
 (I[251] = (img)(_p1##x,_n1##y,z,v)), \
 (I[273] = (img)(_p1##x,_n2##y,z,v)), \
 (I[295] = (img)(_p1##x,_n3##y,z,v)), \
 (I[317] = (img)(_p1##x,_n4##y,z,v)), \
 (I[339] = (img)(_p1##x,_n5##y,z,v)), \
 (I[361] = (img)(_p1##x,_n6##y,z,v)), \
 (I[383] = (img)(_p1##x,_n7##y,z,v)), \
 (I[405] = (img)(_p1##x,_n8##y,z,v)), \
 (I[427] = (img)(_p1##x,_n9##y,z,v)), \
 (I[449] = (img)(_p1##x,_n10##y,z,v)), \
 (I[471] = (img)(_p1##x,_n11##y,z,v)), \
 (I[10] = (img)(x,_p10##y,z,v)), \
 (I[32] = (img)(x,_p9##y,z,v)), \
 (I[54] = (img)(x,_p8##y,z,v)), \
 (I[76] = (img)(x,_p7##y,z,v)), \
 (I[98] = (img)(x,_p6##y,z,v)), \
 (I[120] = (img)(x,_p5##y,z,v)), \
 (I[142] = (img)(x,_p4##y,z,v)), \
 (I[164] = (img)(x,_p3##y,z,v)), \
 (I[186] = (img)(x,_p2##y,z,v)), \
 (I[208] = (img)(x,_p1##y,z,v)), \
 (I[230] = (img)(x,y,z,v)), \
 (I[252] = (img)(x,_n1##y,z,v)), \
 (I[274] = (img)(x,_n2##y,z,v)), \
 (I[296] = (img)(x,_n3##y,z,v)), \
 (I[318] = (img)(x,_n4##y,z,v)), \
 (I[340] = (img)(x,_n5##y,z,v)), \
 (I[362] = (img)(x,_n6##y,z,v)), \
 (I[384] = (img)(x,_n7##y,z,v)), \
 (I[406] = (img)(x,_n8##y,z,v)), \
 (I[428] = (img)(x,_n9##y,z,v)), \
 (I[450] = (img)(x,_n10##y,z,v)), \
 (I[472] = (img)(x,_n11##y,z,v)), \
 (I[11] = (img)(_n1##x,_p10##y,z,v)), \
 (I[33] = (img)(_n1##x,_p9##y,z,v)), \
 (I[55] = (img)(_n1##x,_p8##y,z,v)), \
 (I[77] = (img)(_n1##x,_p7##y,z,v)), \
 (I[99] = (img)(_n1##x,_p6##y,z,v)), \
 (I[121] = (img)(_n1##x,_p5##y,z,v)), \
 (I[143] = (img)(_n1##x,_p4##y,z,v)), \
 (I[165] = (img)(_n1##x,_p3##y,z,v)), \
 (I[187] = (img)(_n1##x,_p2##y,z,v)), \
 (I[209] = (img)(_n1##x,_p1##y,z,v)), \
 (I[231] = (img)(_n1##x,y,z,v)), \
 (I[253] = (img)(_n1##x,_n1##y,z,v)), \
 (I[275] = (img)(_n1##x,_n2##y,z,v)), \
 (I[297] = (img)(_n1##x,_n3##y,z,v)), \
 (I[319] = (img)(_n1##x,_n4##y,z,v)), \
 (I[341] = (img)(_n1##x,_n5##y,z,v)), \
 (I[363] = (img)(_n1##x,_n6##y,z,v)), \
 (I[385] = (img)(_n1##x,_n7##y,z,v)), \
 (I[407] = (img)(_n1##x,_n8##y,z,v)), \
 (I[429] = (img)(_n1##x,_n9##y,z,v)), \
 (I[451] = (img)(_n1##x,_n10##y,z,v)), \
 (I[473] = (img)(_n1##x,_n11##y,z,v)), \
 (I[12] = (img)(_n2##x,_p10##y,z,v)), \
 (I[34] = (img)(_n2##x,_p9##y,z,v)), \
 (I[56] = (img)(_n2##x,_p8##y,z,v)), \
 (I[78] = (img)(_n2##x,_p7##y,z,v)), \
 (I[100] = (img)(_n2##x,_p6##y,z,v)), \
 (I[122] = (img)(_n2##x,_p5##y,z,v)), \
 (I[144] = (img)(_n2##x,_p4##y,z,v)), \
 (I[166] = (img)(_n2##x,_p3##y,z,v)), \
 (I[188] = (img)(_n2##x,_p2##y,z,v)), \
 (I[210] = (img)(_n2##x,_p1##y,z,v)), \
 (I[232] = (img)(_n2##x,y,z,v)), \
 (I[254] = (img)(_n2##x,_n1##y,z,v)), \
 (I[276] = (img)(_n2##x,_n2##y,z,v)), \
 (I[298] = (img)(_n2##x,_n3##y,z,v)), \
 (I[320] = (img)(_n2##x,_n4##y,z,v)), \
 (I[342] = (img)(_n2##x,_n5##y,z,v)), \
 (I[364] = (img)(_n2##x,_n6##y,z,v)), \
 (I[386] = (img)(_n2##x,_n7##y,z,v)), \
 (I[408] = (img)(_n2##x,_n8##y,z,v)), \
 (I[430] = (img)(_n2##x,_n9##y,z,v)), \
 (I[452] = (img)(_n2##x,_n10##y,z,v)), \
 (I[474] = (img)(_n2##x,_n11##y,z,v)), \
 (I[13] = (img)(_n3##x,_p10##y,z,v)), \
 (I[35] = (img)(_n3##x,_p9##y,z,v)), \
 (I[57] = (img)(_n3##x,_p8##y,z,v)), \
 (I[79] = (img)(_n3##x,_p7##y,z,v)), \
 (I[101] = (img)(_n3##x,_p6##y,z,v)), \
 (I[123] = (img)(_n3##x,_p5##y,z,v)), \
 (I[145] = (img)(_n3##x,_p4##y,z,v)), \
 (I[167] = (img)(_n3##x,_p3##y,z,v)), \
 (I[189] = (img)(_n3##x,_p2##y,z,v)), \
 (I[211] = (img)(_n3##x,_p1##y,z,v)), \
 (I[233] = (img)(_n3##x,y,z,v)), \
 (I[255] = (img)(_n3##x,_n1##y,z,v)), \
 (I[277] = (img)(_n3##x,_n2##y,z,v)), \
 (I[299] = (img)(_n3##x,_n3##y,z,v)), \
 (I[321] = (img)(_n3##x,_n4##y,z,v)), \
 (I[343] = (img)(_n3##x,_n5##y,z,v)), \
 (I[365] = (img)(_n3##x,_n6##y,z,v)), \
 (I[387] = (img)(_n3##x,_n7##y,z,v)), \
 (I[409] = (img)(_n3##x,_n8##y,z,v)), \
 (I[431] = (img)(_n3##x,_n9##y,z,v)), \
 (I[453] = (img)(_n3##x,_n10##y,z,v)), \
 (I[475] = (img)(_n3##x,_n11##y,z,v)), \
 (I[14] = (img)(_n4##x,_p10##y,z,v)), \
 (I[36] = (img)(_n4##x,_p9##y,z,v)), \
 (I[58] = (img)(_n4##x,_p8##y,z,v)), \
 (I[80] = (img)(_n4##x,_p7##y,z,v)), \
 (I[102] = (img)(_n4##x,_p6##y,z,v)), \
 (I[124] = (img)(_n4##x,_p5##y,z,v)), \
 (I[146] = (img)(_n4##x,_p4##y,z,v)), \
 (I[168] = (img)(_n4##x,_p3##y,z,v)), \
 (I[190] = (img)(_n4##x,_p2##y,z,v)), \
 (I[212] = (img)(_n4##x,_p1##y,z,v)), \
 (I[234] = (img)(_n4##x,y,z,v)), \
 (I[256] = (img)(_n4##x,_n1##y,z,v)), \
 (I[278] = (img)(_n4##x,_n2##y,z,v)), \
 (I[300] = (img)(_n4##x,_n3##y,z,v)), \
 (I[322] = (img)(_n4##x,_n4##y,z,v)), \
 (I[344] = (img)(_n4##x,_n5##y,z,v)), \
 (I[366] = (img)(_n4##x,_n6##y,z,v)), \
 (I[388] = (img)(_n4##x,_n7##y,z,v)), \
 (I[410] = (img)(_n4##x,_n8##y,z,v)), \
 (I[432] = (img)(_n4##x,_n9##y,z,v)), \
 (I[454] = (img)(_n4##x,_n10##y,z,v)), \
 (I[476] = (img)(_n4##x,_n11##y,z,v)), \
 (I[15] = (img)(_n5##x,_p10##y,z,v)), \
 (I[37] = (img)(_n5##x,_p9##y,z,v)), \
 (I[59] = (img)(_n5##x,_p8##y,z,v)), \
 (I[81] = (img)(_n5##x,_p7##y,z,v)), \
 (I[103] = (img)(_n5##x,_p6##y,z,v)), \
 (I[125] = (img)(_n5##x,_p5##y,z,v)), \
 (I[147] = (img)(_n5##x,_p4##y,z,v)), \
 (I[169] = (img)(_n5##x,_p3##y,z,v)), \
 (I[191] = (img)(_n5##x,_p2##y,z,v)), \
 (I[213] = (img)(_n5##x,_p1##y,z,v)), \
 (I[235] = (img)(_n5##x,y,z,v)), \
 (I[257] = (img)(_n5##x,_n1##y,z,v)), \
 (I[279] = (img)(_n5##x,_n2##y,z,v)), \
 (I[301] = (img)(_n5##x,_n3##y,z,v)), \
 (I[323] = (img)(_n5##x,_n4##y,z,v)), \
 (I[345] = (img)(_n5##x,_n5##y,z,v)), \
 (I[367] = (img)(_n5##x,_n6##y,z,v)), \
 (I[389] = (img)(_n5##x,_n7##y,z,v)), \
 (I[411] = (img)(_n5##x,_n8##y,z,v)), \
 (I[433] = (img)(_n5##x,_n9##y,z,v)), \
 (I[455] = (img)(_n5##x,_n10##y,z,v)), \
 (I[477] = (img)(_n5##x,_n11##y,z,v)), \
 (I[16] = (img)(_n6##x,_p10##y,z,v)), \
 (I[38] = (img)(_n6##x,_p9##y,z,v)), \
 (I[60] = (img)(_n6##x,_p8##y,z,v)), \
 (I[82] = (img)(_n6##x,_p7##y,z,v)), \
 (I[104] = (img)(_n6##x,_p6##y,z,v)), \
 (I[126] = (img)(_n6##x,_p5##y,z,v)), \
 (I[148] = (img)(_n6##x,_p4##y,z,v)), \
 (I[170] = (img)(_n6##x,_p3##y,z,v)), \
 (I[192] = (img)(_n6##x,_p2##y,z,v)), \
 (I[214] = (img)(_n6##x,_p1##y,z,v)), \
 (I[236] = (img)(_n6##x,y,z,v)), \
 (I[258] = (img)(_n6##x,_n1##y,z,v)), \
 (I[280] = (img)(_n6##x,_n2##y,z,v)), \
 (I[302] = (img)(_n6##x,_n3##y,z,v)), \
 (I[324] = (img)(_n6##x,_n4##y,z,v)), \
 (I[346] = (img)(_n6##x,_n5##y,z,v)), \
 (I[368] = (img)(_n6##x,_n6##y,z,v)), \
 (I[390] = (img)(_n6##x,_n7##y,z,v)), \
 (I[412] = (img)(_n6##x,_n8##y,z,v)), \
 (I[434] = (img)(_n6##x,_n9##y,z,v)), \
 (I[456] = (img)(_n6##x,_n10##y,z,v)), \
 (I[478] = (img)(_n6##x,_n11##y,z,v)), \
 (I[17] = (img)(_n7##x,_p10##y,z,v)), \
 (I[39] = (img)(_n7##x,_p9##y,z,v)), \
 (I[61] = (img)(_n7##x,_p8##y,z,v)), \
 (I[83] = (img)(_n7##x,_p7##y,z,v)), \
 (I[105] = (img)(_n7##x,_p6##y,z,v)), \
 (I[127] = (img)(_n7##x,_p5##y,z,v)), \
 (I[149] = (img)(_n7##x,_p4##y,z,v)), \
 (I[171] = (img)(_n7##x,_p3##y,z,v)), \
 (I[193] = (img)(_n7##x,_p2##y,z,v)), \
 (I[215] = (img)(_n7##x,_p1##y,z,v)), \
 (I[237] = (img)(_n7##x,y,z,v)), \
 (I[259] = (img)(_n7##x,_n1##y,z,v)), \
 (I[281] = (img)(_n7##x,_n2##y,z,v)), \
 (I[303] = (img)(_n7##x,_n3##y,z,v)), \
 (I[325] = (img)(_n7##x,_n4##y,z,v)), \
 (I[347] = (img)(_n7##x,_n5##y,z,v)), \
 (I[369] = (img)(_n7##x,_n6##y,z,v)), \
 (I[391] = (img)(_n7##x,_n7##y,z,v)), \
 (I[413] = (img)(_n7##x,_n8##y,z,v)), \
 (I[435] = (img)(_n7##x,_n9##y,z,v)), \
 (I[457] = (img)(_n7##x,_n10##y,z,v)), \
 (I[479] = (img)(_n7##x,_n11##y,z,v)), \
 (I[18] = (img)(_n8##x,_p10##y,z,v)), \
 (I[40] = (img)(_n8##x,_p9##y,z,v)), \
 (I[62] = (img)(_n8##x,_p8##y,z,v)), \
 (I[84] = (img)(_n8##x,_p7##y,z,v)), \
 (I[106] = (img)(_n8##x,_p6##y,z,v)), \
 (I[128] = (img)(_n8##x,_p5##y,z,v)), \
 (I[150] = (img)(_n8##x,_p4##y,z,v)), \
 (I[172] = (img)(_n8##x,_p3##y,z,v)), \
 (I[194] = (img)(_n8##x,_p2##y,z,v)), \
 (I[216] = (img)(_n8##x,_p1##y,z,v)), \
 (I[238] = (img)(_n8##x,y,z,v)), \
 (I[260] = (img)(_n8##x,_n1##y,z,v)), \
 (I[282] = (img)(_n8##x,_n2##y,z,v)), \
 (I[304] = (img)(_n8##x,_n3##y,z,v)), \
 (I[326] = (img)(_n8##x,_n4##y,z,v)), \
 (I[348] = (img)(_n8##x,_n5##y,z,v)), \
 (I[370] = (img)(_n8##x,_n6##y,z,v)), \
 (I[392] = (img)(_n8##x,_n7##y,z,v)), \
 (I[414] = (img)(_n8##x,_n8##y,z,v)), \
 (I[436] = (img)(_n8##x,_n9##y,z,v)), \
 (I[458] = (img)(_n8##x,_n10##y,z,v)), \
 (I[480] = (img)(_n8##x,_n11##y,z,v)), \
 (I[19] = (img)(_n9##x,_p10##y,z,v)), \
 (I[41] = (img)(_n9##x,_p9##y,z,v)), \
 (I[63] = (img)(_n9##x,_p8##y,z,v)), \
 (I[85] = (img)(_n9##x,_p7##y,z,v)), \
 (I[107] = (img)(_n9##x,_p6##y,z,v)), \
 (I[129] = (img)(_n9##x,_p5##y,z,v)), \
 (I[151] = (img)(_n9##x,_p4##y,z,v)), \
 (I[173] = (img)(_n9##x,_p3##y,z,v)), \
 (I[195] = (img)(_n9##x,_p2##y,z,v)), \
 (I[217] = (img)(_n9##x,_p1##y,z,v)), \
 (I[239] = (img)(_n9##x,y,z,v)), \
 (I[261] = (img)(_n9##x,_n1##y,z,v)), \
 (I[283] = (img)(_n9##x,_n2##y,z,v)), \
 (I[305] = (img)(_n9##x,_n3##y,z,v)), \
 (I[327] = (img)(_n9##x,_n4##y,z,v)), \
 (I[349] = (img)(_n9##x,_n5##y,z,v)), \
 (I[371] = (img)(_n9##x,_n6##y,z,v)), \
 (I[393] = (img)(_n9##x,_n7##y,z,v)), \
 (I[415] = (img)(_n9##x,_n8##y,z,v)), \
 (I[437] = (img)(_n9##x,_n9##y,z,v)), \
 (I[459] = (img)(_n9##x,_n10##y,z,v)), \
 (I[481] = (img)(_n9##x,_n11##y,z,v)), \
 (I[20] = (img)(_n10##x,_p10##y,z,v)), \
 (I[42] = (img)(_n10##x,_p9##y,z,v)), \
 (I[64] = (img)(_n10##x,_p8##y,z,v)), \
 (I[86] = (img)(_n10##x,_p7##y,z,v)), \
 (I[108] = (img)(_n10##x,_p6##y,z,v)), \
 (I[130] = (img)(_n10##x,_p5##y,z,v)), \
 (I[152] = (img)(_n10##x,_p4##y,z,v)), \
 (I[174] = (img)(_n10##x,_p3##y,z,v)), \
 (I[196] = (img)(_n10##x,_p2##y,z,v)), \
 (I[218] = (img)(_n10##x,_p1##y,z,v)), \
 (I[240] = (img)(_n10##x,y,z,v)), \
 (I[262] = (img)(_n10##x,_n1##y,z,v)), \
 (I[284] = (img)(_n10##x,_n2##y,z,v)), \
 (I[306] = (img)(_n10##x,_n3##y,z,v)), \
 (I[328] = (img)(_n10##x,_n4##y,z,v)), \
 (I[350] = (img)(_n10##x,_n5##y,z,v)), \
 (I[372] = (img)(_n10##x,_n6##y,z,v)), \
 (I[394] = (img)(_n10##x,_n7##y,z,v)), \
 (I[416] = (img)(_n10##x,_n8##y,z,v)), \
 (I[438] = (img)(_n10##x,_n9##y,z,v)), \
 (I[460] = (img)(_n10##x,_n10##y,z,v)), \
 (I[482] = (img)(_n10##x,_n11##y,z,v)), \
 x+11>=(int)((img).width)?(int)((img).width)-1:x+11); \
 x<=(int)(x1) && ((_n11##x<(int)((img).width) && ( \
 (I[21] = (img)(_n11##x,_p10##y,z,v)), \
 (I[43] = (img)(_n11##x,_p9##y,z,v)), \
 (I[65] = (img)(_n11##x,_p8##y,z,v)), \
 (I[87] = (img)(_n11##x,_p7##y,z,v)), \
 (I[109] = (img)(_n11##x,_p6##y,z,v)), \
 (I[131] = (img)(_n11##x,_p5##y,z,v)), \
 (I[153] = (img)(_n11##x,_p4##y,z,v)), \
 (I[175] = (img)(_n11##x,_p3##y,z,v)), \
 (I[197] = (img)(_n11##x,_p2##y,z,v)), \
 (I[219] = (img)(_n11##x,_p1##y,z,v)), \
 (I[241] = (img)(_n11##x,y,z,v)), \
 (I[263] = (img)(_n11##x,_n1##y,z,v)), \
 (I[285] = (img)(_n11##x,_n2##y,z,v)), \
 (I[307] = (img)(_n11##x,_n3##y,z,v)), \
 (I[329] = (img)(_n11##x,_n4##y,z,v)), \
 (I[351] = (img)(_n11##x,_n5##y,z,v)), \
 (I[373] = (img)(_n11##x,_n6##y,z,v)), \
 (I[395] = (img)(_n11##x,_n7##y,z,v)), \
 (I[417] = (img)(_n11##x,_n8##y,z,v)), \
 (I[439] = (img)(_n11##x,_n9##y,z,v)), \
 (I[461] = (img)(_n11##x,_n10##y,z,v)), \
 (I[483] = (img)(_n11##x,_n11##y,z,v)),1)) || \
 _n10##x==--_n11##x || _n9##x==--_n10##x || _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n11##x = _n10##x = _n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], \
 I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], \
 I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], \
 I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], \
 I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], \
 I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], \
 I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], \
 I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], \
 I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], \
 I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], \
 I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], I[239] = I[240], I[240] = I[241], \
 I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], \
 I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], \
 I[286] = I[287], I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], \
 I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], I[321] = I[322], I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], \
 I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], I[335] = I[336], I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], I[342] = I[343], I[343] = I[344], I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], \
 I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], I[359] = I[360], I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], I[367] = I[368], I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], \
 I[374] = I[375], I[375] = I[376], I[376] = I[377], I[377] = I[378], I[378] = I[379], I[379] = I[380], I[380] = I[381], I[381] = I[382], I[382] = I[383], I[383] = I[384], I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], I[390] = I[391], I[391] = I[392], I[392] = I[393], I[393] = I[394], I[394] = I[395], \
 I[396] = I[397], I[397] = I[398], I[398] = I[399], I[399] = I[400], I[400] = I[401], I[401] = I[402], I[402] = I[403], I[403] = I[404], I[404] = I[405], I[405] = I[406], I[406] = I[407], I[407] = I[408], I[408] = I[409], I[409] = I[410], I[410] = I[411], I[411] = I[412], I[412] = I[413], I[413] = I[414], I[414] = I[415], I[415] = I[416], I[416] = I[417], \
 I[418] = I[419], I[419] = I[420], I[420] = I[421], I[421] = I[422], I[422] = I[423], I[423] = I[424], I[424] = I[425], I[425] = I[426], I[426] = I[427], I[427] = I[428], I[428] = I[429], I[429] = I[430], I[430] = I[431], I[431] = I[432], I[432] = I[433], I[433] = I[434], I[434] = I[435], I[435] = I[436], I[436] = I[437], I[437] = I[438], I[438] = I[439], \
 I[440] = I[441], I[441] = I[442], I[442] = I[443], I[443] = I[444], I[444] = I[445], I[445] = I[446], I[446] = I[447], I[447] = I[448], I[448] = I[449], I[449] = I[450], I[450] = I[451], I[451] = I[452], I[452] = I[453], I[453] = I[454], I[454] = I[455], I[455] = I[456], I[456] = I[457], I[457] = I[458], I[458] = I[459], I[459] = I[460], I[460] = I[461], \
 I[462] = I[463], I[463] = I[464], I[464] = I[465], I[465] = I[466], I[466] = I[467], I[467] = I[468], I[468] = I[469], I[469] = I[470], I[470] = I[471], I[471] = I[472], I[472] = I[473], I[473] = I[474], I[474] = I[475], I[475] = I[476], I[476] = I[477], I[477] = I[478], I[478] = I[479], I[479] = I[480], I[480] = I[481], I[481] = I[482], I[482] = I[483], \
 _p10##x = _p9##x, _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x, ++_n10##x, ++_n11##x)

#define cimg_get22x22(img,x,y,z,v,I) \
 I[0] = (img)(_p10##x,_p10##y,z,v), I[1] = (img)(_p9##x,_p10##y,z,v), I[2] = (img)(_p8##x,_p10##y,z,v), I[3] = (img)(_p7##x,_p10##y,z,v), I[4] = (img)(_p6##x,_p10##y,z,v), I[5] = (img)(_p5##x,_p10##y,z,v), I[6] = (img)(_p4##x,_p10##y,z,v), I[7] = (img)(_p3##x,_p10##y,z,v), I[8] = (img)(_p2##x,_p10##y,z,v), I[9] = (img)(_p1##x,_p10##y,z,v), I[10] = (img)(x,_p10##y,z,v), I[11] = (img)(_n1##x,_p10##y,z,v), I[12] = (img)(_n2##x,_p10##y,z,v), I[13] = (img)(_n3##x,_p10##y,z,v), I[14] = (img)(_n4##x,_p10##y,z,v), I[15] = (img)(_n5##x,_p10##y,z,v), I[16] = (img)(_n6##x,_p10##y,z,v), I[17] = (img)(_n7##x,_p10##y,z,v), I[18] = (img)(_n8##x,_p10##y,z,v), I[19] = (img)(_n9##x,_p10##y,z,v), I[20] = (img)(_n10##x,_p10##y,z,v), I[21] = (img)(_n11##x,_p10##y,z,v), \
 I[22] = (img)(_p10##x,_p9##y,z,v), I[23] = (img)(_p9##x,_p9##y,z,v), I[24] = (img)(_p8##x,_p9##y,z,v), I[25] = (img)(_p7##x,_p9##y,z,v), I[26] = (img)(_p6##x,_p9##y,z,v), I[27] = (img)(_p5##x,_p9##y,z,v), I[28] = (img)(_p4##x,_p9##y,z,v), I[29] = (img)(_p3##x,_p9##y,z,v), I[30] = (img)(_p2##x,_p9##y,z,v), I[31] = (img)(_p1##x,_p9##y,z,v), I[32] = (img)(x,_p9##y,z,v), I[33] = (img)(_n1##x,_p9##y,z,v), I[34] = (img)(_n2##x,_p9##y,z,v), I[35] = (img)(_n3##x,_p9##y,z,v), I[36] = (img)(_n4##x,_p9##y,z,v), I[37] = (img)(_n5##x,_p9##y,z,v), I[38] = (img)(_n6##x,_p9##y,z,v), I[39] = (img)(_n7##x,_p9##y,z,v), I[40] = (img)(_n8##x,_p9##y,z,v), I[41] = (img)(_n9##x,_p9##y,z,v), I[42] = (img)(_n10##x,_p9##y,z,v), I[43] = (img)(_n11##x,_p9##y,z,v), \
 I[44] = (img)(_p10##x,_p8##y,z,v), I[45] = (img)(_p9##x,_p8##y,z,v), I[46] = (img)(_p8##x,_p8##y,z,v), I[47] = (img)(_p7##x,_p8##y,z,v), I[48] = (img)(_p6##x,_p8##y,z,v), I[49] = (img)(_p5##x,_p8##y,z,v), I[50] = (img)(_p4##x,_p8##y,z,v), I[51] = (img)(_p3##x,_p8##y,z,v), I[52] = (img)(_p2##x,_p8##y,z,v), I[53] = (img)(_p1##x,_p8##y,z,v), I[54] = (img)(x,_p8##y,z,v), I[55] = (img)(_n1##x,_p8##y,z,v), I[56] = (img)(_n2##x,_p8##y,z,v), I[57] = (img)(_n3##x,_p8##y,z,v), I[58] = (img)(_n4##x,_p8##y,z,v), I[59] = (img)(_n5##x,_p8##y,z,v), I[60] = (img)(_n6##x,_p8##y,z,v), I[61] = (img)(_n7##x,_p8##y,z,v), I[62] = (img)(_n8##x,_p8##y,z,v), I[63] = (img)(_n9##x,_p8##y,z,v), I[64] = (img)(_n10##x,_p8##y,z,v), I[65] = (img)(_n11##x,_p8##y,z,v), \
 I[66] = (img)(_p10##x,_p7##y,z,v), I[67] = (img)(_p9##x,_p7##y,z,v), I[68] = (img)(_p8##x,_p7##y,z,v), I[69] = (img)(_p7##x,_p7##y,z,v), I[70] = (img)(_p6##x,_p7##y,z,v), I[71] = (img)(_p5##x,_p7##y,z,v), I[72] = (img)(_p4##x,_p7##y,z,v), I[73] = (img)(_p3##x,_p7##y,z,v), I[74] = (img)(_p2##x,_p7##y,z,v), I[75] = (img)(_p1##x,_p7##y,z,v), I[76] = (img)(x,_p7##y,z,v), I[77] = (img)(_n1##x,_p7##y,z,v), I[78] = (img)(_n2##x,_p7##y,z,v), I[79] = (img)(_n3##x,_p7##y,z,v), I[80] = (img)(_n4##x,_p7##y,z,v), I[81] = (img)(_n5##x,_p7##y,z,v), I[82] = (img)(_n6##x,_p7##y,z,v), I[83] = (img)(_n7##x,_p7##y,z,v), I[84] = (img)(_n8##x,_p7##y,z,v), I[85] = (img)(_n9##x,_p7##y,z,v), I[86] = (img)(_n10##x,_p7##y,z,v), I[87] = (img)(_n11##x,_p7##y,z,v), \
 I[88] = (img)(_p10##x,_p6##y,z,v), I[89] = (img)(_p9##x,_p6##y,z,v), I[90] = (img)(_p8##x,_p6##y,z,v), I[91] = (img)(_p7##x,_p6##y,z,v), I[92] = (img)(_p6##x,_p6##y,z,v), I[93] = (img)(_p5##x,_p6##y,z,v), I[94] = (img)(_p4##x,_p6##y,z,v), I[95] = (img)(_p3##x,_p6##y,z,v), I[96] = (img)(_p2##x,_p6##y,z,v), I[97] = (img)(_p1##x,_p6##y,z,v), I[98] = (img)(x,_p6##y,z,v), I[99] = (img)(_n1##x,_p6##y,z,v), I[100] = (img)(_n2##x,_p6##y,z,v), I[101] = (img)(_n3##x,_p6##y,z,v), I[102] = (img)(_n4##x,_p6##y,z,v), I[103] = (img)(_n5##x,_p6##y,z,v), I[104] = (img)(_n6##x,_p6##y,z,v), I[105] = (img)(_n7##x,_p6##y,z,v), I[106] = (img)(_n8##x,_p6##y,z,v), I[107] = (img)(_n9##x,_p6##y,z,v), I[108] = (img)(_n10##x,_p6##y,z,v), I[109] = (img)(_n11##x,_p6##y,z,v), \
 I[110] = (img)(_p10##x,_p5##y,z,v), I[111] = (img)(_p9##x,_p5##y,z,v), I[112] = (img)(_p8##x,_p5##y,z,v), I[113] = (img)(_p7##x,_p5##y,z,v), I[114] = (img)(_p6##x,_p5##y,z,v), I[115] = (img)(_p5##x,_p5##y,z,v), I[116] = (img)(_p4##x,_p5##y,z,v), I[117] = (img)(_p3##x,_p5##y,z,v), I[118] = (img)(_p2##x,_p5##y,z,v), I[119] = (img)(_p1##x,_p5##y,z,v), I[120] = (img)(x,_p5##y,z,v), I[121] = (img)(_n1##x,_p5##y,z,v), I[122] = (img)(_n2##x,_p5##y,z,v), I[123] = (img)(_n3##x,_p5##y,z,v), I[124] = (img)(_n4##x,_p5##y,z,v), I[125] = (img)(_n5##x,_p5##y,z,v), I[126] = (img)(_n6##x,_p5##y,z,v), I[127] = (img)(_n7##x,_p5##y,z,v), I[128] = (img)(_n8##x,_p5##y,z,v), I[129] = (img)(_n9##x,_p5##y,z,v), I[130] = (img)(_n10##x,_p5##y,z,v), I[131] = (img)(_n11##x,_p5##y,z,v), \
 I[132] = (img)(_p10##x,_p4##y,z,v), I[133] = (img)(_p9##x,_p4##y,z,v), I[134] = (img)(_p8##x,_p4##y,z,v), I[135] = (img)(_p7##x,_p4##y,z,v), I[136] = (img)(_p6##x,_p4##y,z,v), I[137] = (img)(_p5##x,_p4##y,z,v), I[138] = (img)(_p4##x,_p4##y,z,v), I[139] = (img)(_p3##x,_p4##y,z,v), I[140] = (img)(_p2##x,_p4##y,z,v), I[141] = (img)(_p1##x,_p4##y,z,v), I[142] = (img)(x,_p4##y,z,v), I[143] = (img)(_n1##x,_p4##y,z,v), I[144] = (img)(_n2##x,_p4##y,z,v), I[145] = (img)(_n3##x,_p4##y,z,v), I[146] = (img)(_n4##x,_p4##y,z,v), I[147] = (img)(_n5##x,_p4##y,z,v), I[148] = (img)(_n6##x,_p4##y,z,v), I[149] = (img)(_n7##x,_p4##y,z,v), I[150] = (img)(_n8##x,_p4##y,z,v), I[151] = (img)(_n9##x,_p4##y,z,v), I[152] = (img)(_n10##x,_p4##y,z,v), I[153] = (img)(_n11##x,_p4##y,z,v), \
 I[154] = (img)(_p10##x,_p3##y,z,v), I[155] = (img)(_p9##x,_p3##y,z,v), I[156] = (img)(_p8##x,_p3##y,z,v), I[157] = (img)(_p7##x,_p3##y,z,v), I[158] = (img)(_p6##x,_p3##y,z,v), I[159] = (img)(_p5##x,_p3##y,z,v), I[160] = (img)(_p4##x,_p3##y,z,v), I[161] = (img)(_p3##x,_p3##y,z,v), I[162] = (img)(_p2##x,_p3##y,z,v), I[163] = (img)(_p1##x,_p3##y,z,v), I[164] = (img)(x,_p3##y,z,v), I[165] = (img)(_n1##x,_p3##y,z,v), I[166] = (img)(_n2##x,_p3##y,z,v), I[167] = (img)(_n3##x,_p3##y,z,v), I[168] = (img)(_n4##x,_p3##y,z,v), I[169] = (img)(_n5##x,_p3##y,z,v), I[170] = (img)(_n6##x,_p3##y,z,v), I[171] = (img)(_n7##x,_p3##y,z,v), I[172] = (img)(_n8##x,_p3##y,z,v), I[173] = (img)(_n9##x,_p3##y,z,v), I[174] = (img)(_n10##x,_p3##y,z,v), I[175] = (img)(_n11##x,_p3##y,z,v), \
 I[176] = (img)(_p10##x,_p2##y,z,v), I[177] = (img)(_p9##x,_p2##y,z,v), I[178] = (img)(_p8##x,_p2##y,z,v), I[179] = (img)(_p7##x,_p2##y,z,v), I[180] = (img)(_p6##x,_p2##y,z,v), I[181] = (img)(_p5##x,_p2##y,z,v), I[182] = (img)(_p4##x,_p2##y,z,v), I[183] = (img)(_p3##x,_p2##y,z,v), I[184] = (img)(_p2##x,_p2##y,z,v), I[185] = (img)(_p1##x,_p2##y,z,v), I[186] = (img)(x,_p2##y,z,v), I[187] = (img)(_n1##x,_p2##y,z,v), I[188] = (img)(_n2##x,_p2##y,z,v), I[189] = (img)(_n3##x,_p2##y,z,v), I[190] = (img)(_n4##x,_p2##y,z,v), I[191] = (img)(_n5##x,_p2##y,z,v), I[192] = (img)(_n6##x,_p2##y,z,v), I[193] = (img)(_n7##x,_p2##y,z,v), I[194] = (img)(_n8##x,_p2##y,z,v), I[195] = (img)(_n9##x,_p2##y,z,v), I[196] = (img)(_n10##x,_p2##y,z,v), I[197] = (img)(_n11##x,_p2##y,z,v), \
 I[198] = (img)(_p10##x,_p1##y,z,v), I[199] = (img)(_p9##x,_p1##y,z,v), I[200] = (img)(_p8##x,_p1##y,z,v), I[201] = (img)(_p7##x,_p1##y,z,v), I[202] = (img)(_p6##x,_p1##y,z,v), I[203] = (img)(_p5##x,_p1##y,z,v), I[204] = (img)(_p4##x,_p1##y,z,v), I[205] = (img)(_p3##x,_p1##y,z,v), I[206] = (img)(_p2##x,_p1##y,z,v), I[207] = (img)(_p1##x,_p1##y,z,v), I[208] = (img)(x,_p1##y,z,v), I[209] = (img)(_n1##x,_p1##y,z,v), I[210] = (img)(_n2##x,_p1##y,z,v), I[211] = (img)(_n3##x,_p1##y,z,v), I[212] = (img)(_n4##x,_p1##y,z,v), I[213] = (img)(_n5##x,_p1##y,z,v), I[214] = (img)(_n6##x,_p1##y,z,v), I[215] = (img)(_n7##x,_p1##y,z,v), I[216] = (img)(_n8##x,_p1##y,z,v), I[217] = (img)(_n9##x,_p1##y,z,v), I[218] = (img)(_n10##x,_p1##y,z,v), I[219] = (img)(_n11##x,_p1##y,z,v), \
 I[220] = (img)(_p10##x,y,z,v), I[221] = (img)(_p9##x,y,z,v), I[222] = (img)(_p8##x,y,z,v), I[223] = (img)(_p7##x,y,z,v), I[224] = (img)(_p6##x,y,z,v), I[225] = (img)(_p5##x,y,z,v), I[226] = (img)(_p4##x,y,z,v), I[227] = (img)(_p3##x,y,z,v), I[228] = (img)(_p2##x,y,z,v), I[229] = (img)(_p1##x,y,z,v), I[230] = (img)(x,y,z,v), I[231] = (img)(_n1##x,y,z,v), I[232] = (img)(_n2##x,y,z,v), I[233] = (img)(_n3##x,y,z,v), I[234] = (img)(_n4##x,y,z,v), I[235] = (img)(_n5##x,y,z,v), I[236] = (img)(_n6##x,y,z,v), I[237] = (img)(_n7##x,y,z,v), I[238] = (img)(_n8##x,y,z,v), I[239] = (img)(_n9##x,y,z,v), I[240] = (img)(_n10##x,y,z,v), I[241] = (img)(_n11##x,y,z,v), \
 I[242] = (img)(_p10##x,_n1##y,z,v), I[243] = (img)(_p9##x,_n1##y,z,v), I[244] = (img)(_p8##x,_n1##y,z,v), I[245] = (img)(_p7##x,_n1##y,z,v), I[246] = (img)(_p6##x,_n1##y,z,v), I[247] = (img)(_p5##x,_n1##y,z,v), I[248] = (img)(_p4##x,_n1##y,z,v), I[249] = (img)(_p3##x,_n1##y,z,v), I[250] = (img)(_p2##x,_n1##y,z,v), I[251] = (img)(_p1##x,_n1##y,z,v), I[252] = (img)(x,_n1##y,z,v), I[253] = (img)(_n1##x,_n1##y,z,v), I[254] = (img)(_n2##x,_n1##y,z,v), I[255] = (img)(_n3##x,_n1##y,z,v), I[256] = (img)(_n4##x,_n1##y,z,v), I[257] = (img)(_n5##x,_n1##y,z,v), I[258] = (img)(_n6##x,_n1##y,z,v), I[259] = (img)(_n7##x,_n1##y,z,v), I[260] = (img)(_n8##x,_n1##y,z,v), I[261] = (img)(_n9##x,_n1##y,z,v), I[262] = (img)(_n10##x,_n1##y,z,v), I[263] = (img)(_n11##x,_n1##y,z,v), \
 I[264] = (img)(_p10##x,_n2##y,z,v), I[265] = (img)(_p9##x,_n2##y,z,v), I[266] = (img)(_p8##x,_n2##y,z,v), I[267] = (img)(_p7##x,_n2##y,z,v), I[268] = (img)(_p6##x,_n2##y,z,v), I[269] = (img)(_p5##x,_n2##y,z,v), I[270] = (img)(_p4##x,_n2##y,z,v), I[271] = (img)(_p3##x,_n2##y,z,v), I[272] = (img)(_p2##x,_n2##y,z,v), I[273] = (img)(_p1##x,_n2##y,z,v), I[274] = (img)(x,_n2##y,z,v), I[275] = (img)(_n1##x,_n2##y,z,v), I[276] = (img)(_n2##x,_n2##y,z,v), I[277] = (img)(_n3##x,_n2##y,z,v), I[278] = (img)(_n4##x,_n2##y,z,v), I[279] = (img)(_n5##x,_n2##y,z,v), I[280] = (img)(_n6##x,_n2##y,z,v), I[281] = (img)(_n7##x,_n2##y,z,v), I[282] = (img)(_n8##x,_n2##y,z,v), I[283] = (img)(_n9##x,_n2##y,z,v), I[284] = (img)(_n10##x,_n2##y,z,v), I[285] = (img)(_n11##x,_n2##y,z,v), \
 I[286] = (img)(_p10##x,_n3##y,z,v), I[287] = (img)(_p9##x,_n3##y,z,v), I[288] = (img)(_p8##x,_n3##y,z,v), I[289] = (img)(_p7##x,_n3##y,z,v), I[290] = (img)(_p6##x,_n3##y,z,v), I[291] = (img)(_p5##x,_n3##y,z,v), I[292] = (img)(_p4##x,_n3##y,z,v), I[293] = (img)(_p3##x,_n3##y,z,v), I[294] = (img)(_p2##x,_n3##y,z,v), I[295] = (img)(_p1##x,_n3##y,z,v), I[296] = (img)(x,_n3##y,z,v), I[297] = (img)(_n1##x,_n3##y,z,v), I[298] = (img)(_n2##x,_n3##y,z,v), I[299] = (img)(_n3##x,_n3##y,z,v), I[300] = (img)(_n4##x,_n3##y,z,v), I[301] = (img)(_n5##x,_n3##y,z,v), I[302] = (img)(_n6##x,_n3##y,z,v), I[303] = (img)(_n7##x,_n3##y,z,v), I[304] = (img)(_n8##x,_n3##y,z,v), I[305] = (img)(_n9##x,_n3##y,z,v), I[306] = (img)(_n10##x,_n3##y,z,v), I[307] = (img)(_n11##x,_n3##y,z,v), \
 I[308] = (img)(_p10##x,_n4##y,z,v), I[309] = (img)(_p9##x,_n4##y,z,v), I[310] = (img)(_p8##x,_n4##y,z,v), I[311] = (img)(_p7##x,_n4##y,z,v), I[312] = (img)(_p6##x,_n4##y,z,v), I[313] = (img)(_p5##x,_n4##y,z,v), I[314] = (img)(_p4##x,_n4##y,z,v), I[315] = (img)(_p3##x,_n4##y,z,v), I[316] = (img)(_p2##x,_n4##y,z,v), I[317] = (img)(_p1##x,_n4##y,z,v), I[318] = (img)(x,_n4##y,z,v), I[319] = (img)(_n1##x,_n4##y,z,v), I[320] = (img)(_n2##x,_n4##y,z,v), I[321] = (img)(_n3##x,_n4##y,z,v), I[322] = (img)(_n4##x,_n4##y,z,v), I[323] = (img)(_n5##x,_n4##y,z,v), I[324] = (img)(_n6##x,_n4##y,z,v), I[325] = (img)(_n7##x,_n4##y,z,v), I[326] = (img)(_n8##x,_n4##y,z,v), I[327] = (img)(_n9##x,_n4##y,z,v), I[328] = (img)(_n10##x,_n4##y,z,v), I[329] = (img)(_n11##x,_n4##y,z,v), \
 I[330] = (img)(_p10##x,_n5##y,z,v), I[331] = (img)(_p9##x,_n5##y,z,v), I[332] = (img)(_p8##x,_n5##y,z,v), I[333] = (img)(_p7##x,_n5##y,z,v), I[334] = (img)(_p6##x,_n5##y,z,v), I[335] = (img)(_p5##x,_n5##y,z,v), I[336] = (img)(_p4##x,_n5##y,z,v), I[337] = (img)(_p3##x,_n5##y,z,v), I[338] = (img)(_p2##x,_n5##y,z,v), I[339] = (img)(_p1##x,_n5##y,z,v), I[340] = (img)(x,_n5##y,z,v), I[341] = (img)(_n1##x,_n5##y,z,v), I[342] = (img)(_n2##x,_n5##y,z,v), I[343] = (img)(_n3##x,_n5##y,z,v), I[344] = (img)(_n4##x,_n5##y,z,v), I[345] = (img)(_n5##x,_n5##y,z,v), I[346] = (img)(_n6##x,_n5##y,z,v), I[347] = (img)(_n7##x,_n5##y,z,v), I[348] = (img)(_n8##x,_n5##y,z,v), I[349] = (img)(_n9##x,_n5##y,z,v), I[350] = (img)(_n10##x,_n5##y,z,v), I[351] = (img)(_n11##x,_n5##y,z,v), \
 I[352] = (img)(_p10##x,_n6##y,z,v), I[353] = (img)(_p9##x,_n6##y,z,v), I[354] = (img)(_p8##x,_n6##y,z,v), I[355] = (img)(_p7##x,_n6##y,z,v), I[356] = (img)(_p6##x,_n6##y,z,v), I[357] = (img)(_p5##x,_n6##y,z,v), I[358] = (img)(_p4##x,_n6##y,z,v), I[359] = (img)(_p3##x,_n6##y,z,v), I[360] = (img)(_p2##x,_n6##y,z,v), I[361] = (img)(_p1##x,_n6##y,z,v), I[362] = (img)(x,_n6##y,z,v), I[363] = (img)(_n1##x,_n6##y,z,v), I[364] = (img)(_n2##x,_n6##y,z,v), I[365] = (img)(_n3##x,_n6##y,z,v), I[366] = (img)(_n4##x,_n6##y,z,v), I[367] = (img)(_n5##x,_n6##y,z,v), I[368] = (img)(_n6##x,_n6##y,z,v), I[369] = (img)(_n7##x,_n6##y,z,v), I[370] = (img)(_n8##x,_n6##y,z,v), I[371] = (img)(_n9##x,_n6##y,z,v), I[372] = (img)(_n10##x,_n6##y,z,v), I[373] = (img)(_n11##x,_n6##y,z,v), \
 I[374] = (img)(_p10##x,_n7##y,z,v), I[375] = (img)(_p9##x,_n7##y,z,v), I[376] = (img)(_p8##x,_n7##y,z,v), I[377] = (img)(_p7##x,_n7##y,z,v), I[378] = (img)(_p6##x,_n7##y,z,v), I[379] = (img)(_p5##x,_n7##y,z,v), I[380] = (img)(_p4##x,_n7##y,z,v), I[381] = (img)(_p3##x,_n7##y,z,v), I[382] = (img)(_p2##x,_n7##y,z,v), I[383] = (img)(_p1##x,_n7##y,z,v), I[384] = (img)(x,_n7##y,z,v), I[385] = (img)(_n1##x,_n7##y,z,v), I[386] = (img)(_n2##x,_n7##y,z,v), I[387] = (img)(_n3##x,_n7##y,z,v), I[388] = (img)(_n4##x,_n7##y,z,v), I[389] = (img)(_n5##x,_n7##y,z,v), I[390] = (img)(_n6##x,_n7##y,z,v), I[391] = (img)(_n7##x,_n7##y,z,v), I[392] = (img)(_n8##x,_n7##y,z,v), I[393] = (img)(_n9##x,_n7##y,z,v), I[394] = (img)(_n10##x,_n7##y,z,v), I[395] = (img)(_n11##x,_n7##y,z,v), \
 I[396] = (img)(_p10##x,_n8##y,z,v), I[397] = (img)(_p9##x,_n8##y,z,v), I[398] = (img)(_p8##x,_n8##y,z,v), I[399] = (img)(_p7##x,_n8##y,z,v), I[400] = (img)(_p6##x,_n8##y,z,v), I[401] = (img)(_p5##x,_n8##y,z,v), I[402] = (img)(_p4##x,_n8##y,z,v), I[403] = (img)(_p3##x,_n8##y,z,v), I[404] = (img)(_p2##x,_n8##y,z,v), I[405] = (img)(_p1##x,_n8##y,z,v), I[406] = (img)(x,_n8##y,z,v), I[407] = (img)(_n1##x,_n8##y,z,v), I[408] = (img)(_n2##x,_n8##y,z,v), I[409] = (img)(_n3##x,_n8##y,z,v), I[410] = (img)(_n4##x,_n8##y,z,v), I[411] = (img)(_n5##x,_n8##y,z,v), I[412] = (img)(_n6##x,_n8##y,z,v), I[413] = (img)(_n7##x,_n8##y,z,v), I[414] = (img)(_n8##x,_n8##y,z,v), I[415] = (img)(_n9##x,_n8##y,z,v), I[416] = (img)(_n10##x,_n8##y,z,v), I[417] = (img)(_n11##x,_n8##y,z,v), \
 I[418] = (img)(_p10##x,_n9##y,z,v), I[419] = (img)(_p9##x,_n9##y,z,v), I[420] = (img)(_p8##x,_n9##y,z,v), I[421] = (img)(_p7##x,_n9##y,z,v), I[422] = (img)(_p6##x,_n9##y,z,v), I[423] = (img)(_p5##x,_n9##y,z,v), I[424] = (img)(_p4##x,_n9##y,z,v), I[425] = (img)(_p3##x,_n9##y,z,v), I[426] = (img)(_p2##x,_n9##y,z,v), I[427] = (img)(_p1##x,_n9##y,z,v), I[428] = (img)(x,_n9##y,z,v), I[429] = (img)(_n1##x,_n9##y,z,v), I[430] = (img)(_n2##x,_n9##y,z,v), I[431] = (img)(_n3##x,_n9##y,z,v), I[432] = (img)(_n4##x,_n9##y,z,v), I[433] = (img)(_n5##x,_n9##y,z,v), I[434] = (img)(_n6##x,_n9##y,z,v), I[435] = (img)(_n7##x,_n9##y,z,v), I[436] = (img)(_n8##x,_n9##y,z,v), I[437] = (img)(_n9##x,_n9##y,z,v), I[438] = (img)(_n10##x,_n9##y,z,v), I[439] = (img)(_n11##x,_n9##y,z,v), \
 I[440] = (img)(_p10##x,_n10##y,z,v), I[441] = (img)(_p9##x,_n10##y,z,v), I[442] = (img)(_p8##x,_n10##y,z,v), I[443] = (img)(_p7##x,_n10##y,z,v), I[444] = (img)(_p6##x,_n10##y,z,v), I[445] = (img)(_p5##x,_n10##y,z,v), I[446] = (img)(_p4##x,_n10##y,z,v), I[447] = (img)(_p3##x,_n10##y,z,v), I[448] = (img)(_p2##x,_n10##y,z,v), I[449] = (img)(_p1##x,_n10##y,z,v), I[450] = (img)(x,_n10##y,z,v), I[451] = (img)(_n1##x,_n10##y,z,v), I[452] = (img)(_n2##x,_n10##y,z,v), I[453] = (img)(_n3##x,_n10##y,z,v), I[454] = (img)(_n4##x,_n10##y,z,v), I[455] = (img)(_n5##x,_n10##y,z,v), I[456] = (img)(_n6##x,_n10##y,z,v), I[457] = (img)(_n7##x,_n10##y,z,v), I[458] = (img)(_n8##x,_n10##y,z,v), I[459] = (img)(_n9##x,_n10##y,z,v), I[460] = (img)(_n10##x,_n10##y,z,v), I[461] = (img)(_n11##x,_n10##y,z,v), \
 I[462] = (img)(_p10##x,_n11##y,z,v), I[463] = (img)(_p9##x,_n11##y,z,v), I[464] = (img)(_p8##x,_n11##y,z,v), I[465] = (img)(_p7##x,_n11##y,z,v), I[466] = (img)(_p6##x,_n11##y,z,v), I[467] = (img)(_p5##x,_n11##y,z,v), I[468] = (img)(_p4##x,_n11##y,z,v), I[469] = (img)(_p3##x,_n11##y,z,v), I[470] = (img)(_p2##x,_n11##y,z,v), I[471] = (img)(_p1##x,_n11##y,z,v), I[472] = (img)(x,_n11##y,z,v), I[473] = (img)(_n1##x,_n11##y,z,v), I[474] = (img)(_n2##x,_n11##y,z,v), I[475] = (img)(_n3##x,_n11##y,z,v), I[476] = (img)(_n4##x,_n11##y,z,v), I[477] = (img)(_n5##x,_n11##y,z,v), I[478] = (img)(_n6##x,_n11##y,z,v), I[479] = (img)(_n7##x,_n11##y,z,v), I[480] = (img)(_n8##x,_n11##y,z,v), I[481] = (img)(_n9##x,_n11##y,z,v), I[482] = (img)(_n10##x,_n11##y,z,v), I[483] = (img)(_n11##x,_n11##y,z,v);

// Define 23x23 loop macros for CImg
//----------------------------------
#define cimg_for23(bound,i) for (int i = 0, \
 _p11##i = 0, _p10##i = 0, _p9##i = 0, _p8##i = 0, _p7##i = 0, _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7, \
 _n8##i = 8>=(int)(bound)?(int)(bound)-1:8, \
 _n9##i = 9>=(int)(bound)?(int)(bound)-1:9, \
 _n10##i = 10>=(int)(bound)?(int)(bound)-1:10, \
 _n11##i = 11>=(int)(bound)?(int)(bound)-1:11; \
 _n11##i<(int)(bound) || _n10##i==--_n11##i || _n9##i==--_n10##i || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n11##i = _n10##i = _n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p11##i = _p10##i, _p10##i = _p9##i, _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i, ++_n10##i, ++_n11##i)

#define cimg_for23X(img,x) cimg_for23((img).width,x)
#define cimg_for23Y(img,y) cimg_for23((img).height,y)
#define cimg_for23Z(img,z) cimg_for23((img).depth,z)
#define cimg_for23V(img,v) cimg_for23((img).dim,v)
#define cimg_for23XY(img,x,y) cimg_for23Y(img,y) cimg_for23X(img,x)
#define cimg_for23XZ(img,x,z) cimg_for23Z(img,z) cimg_for23X(img,x)
#define cimg_for23XV(img,x,v) cimg_for23V(img,v) cimg_for23X(img,x)
#define cimg_for23YZ(img,y,z) cimg_for23Z(img,z) cimg_for23Y(img,y)
#define cimg_for23YV(img,y,v) cimg_for23V(img,v) cimg_for23Y(img,y)
#define cimg_for23ZV(img,z,v) cimg_for23V(img,v) cimg_for23Z(img,z)
#define cimg_for23XYZ(img,x,y,z) cimg_for23Z(img,z) cimg_for23XY(img,x,y)
#define cimg_for23XZV(img,x,z,v) cimg_for23V(img,v) cimg_for23XZ(img,x,z)
#define cimg_for23YZV(img,y,z,v) cimg_for23V(img,v) cimg_for23YZ(img,y,z)
#define cimg_for23XYZV(img,x,y,z,v) cimg_for23V(img,v) cimg_for23XYZ(img,x,y,z)

#define cimg_for_in23(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p11##i = i-11<0?0:i-11, \
 _p10##i = i-10<0?0:i-10, \
 _p9##i = i-9<0?0:i-9, \
 _p8##i = i-8<0?0:i-8, \
 _p7##i = i-7<0?0:i-7, \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7, \
 _n8##i = i+8>=(int)(bound)?(int)(bound)-1:i+8, \
 _n9##i = i+9>=(int)(bound)?(int)(bound)-1:i+9, \
 _n10##i = i+10>=(int)(bound)?(int)(bound)-1:i+10, \
 _n11##i = i+11>=(int)(bound)?(int)(bound)-1:i+11; \
 i<=(int)(i1) && (_n11##i<(int)(bound) || _n10##i==--_n11##i || _n9##i==--_n10##i || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n11##i = _n10##i = _n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p11##i = _p10##i, _p10##i = _p9##i, _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i, ++_n10##i, ++_n11##i)

#define cimg_for_in23X(img,x0,x1,x) cimg_for_in23((img).width,x0,x1,x)
#define cimg_for_in23Y(img,y0,y1,y) cimg_for_in23((img).height,y0,y1,y)
#define cimg_for_in23Z(img,z0,z1,z) cimg_for_in23((img).depth,z0,z1,z)
#define cimg_for_in23V(img,v0,v1,v) cimg_for_in23((img).dim,v0,v1,v)
#define cimg_for_in23XY(img,x0,y0,x1,y1,x,y) cimg_for_in23Y(img,y0,y1,y) cimg_for_in23X(img,x0,x1,x)
#define cimg_for_in23XZ(img,x0,z0,x1,z1,x,z) cimg_for_in23Z(img,z0,z1,z) cimg_for_in23X(img,x0,x1,x)
#define cimg_for_in23XV(img,x0,v0,x1,v1,x,v) cimg_for_in23V(img,v0,v1,v) cimg_for_in23X(img,x0,x1,x)
#define cimg_for_in23YZ(img,y0,z0,y1,z1,y,z) cimg_for_in23Z(img,z0,z1,z) cimg_for_in23Y(img,y0,y1,y)
#define cimg_for_in23YV(img,y0,v0,y1,v1,y,v) cimg_for_in23V(img,v0,v1,v) cimg_for_in23Y(img,y0,y1,y)
#define cimg_for_in23ZV(img,z0,v0,z1,v1,z,v) cimg_for_in23V(img,v0,v1,v) cimg_for_in23Z(img,z0,z1,z)
#define cimg_for_in23XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in23Z(img,z0,z1,z) cimg_for_in23XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in23XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in23V(img,v0,v1,v) cimg_for_in23XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in23YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in23V(img,v0,v1,v) cimg_for_in23YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in23XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in23V(img,v0,v1,v) cimg_for_in23XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for23x23(img,x,y,z,v,I) \
 cimg_for23((img).height,y) for (int x = 0, \
 _p11##x = 0, _p10##x = 0, _p9##x = 0, _p8##x = 0, _p7##x = 0, _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = 7>=((img).width)?(int)((img).width)-1:7, \
 _n8##x = 8>=((img).width)?(int)((img).width)-1:8, \
 _n9##x = 9>=((img).width)?(int)((img).width)-1:9, \
 _n10##x = 10>=((img).width)?(int)((img).width)-1:10, \
 _n11##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = I[7] = I[8] = I[9] = I[10] = I[11] = (img)(0,_p11##y,z,v)), \
 (I[23] = I[24] = I[25] = I[26] = I[27] = I[28] = I[29] = I[30] = I[31] = I[32] = I[33] = I[34] = (img)(0,_p10##y,z,v)), \
 (I[46] = I[47] = I[48] = I[49] = I[50] = I[51] = I[52] = I[53] = I[54] = I[55] = I[56] = I[57] = (img)(0,_p9##y,z,v)), \
 (I[69] = I[70] = I[71] = I[72] = I[73] = I[74] = I[75] = I[76] = I[77] = I[78] = I[79] = I[80] = (img)(0,_p8##y,z,v)), \
 (I[92] = I[93] = I[94] = I[95] = I[96] = I[97] = I[98] = I[99] = I[100] = I[101] = I[102] = I[103] = (img)(0,_p7##y,z,v)), \
 (I[115] = I[116] = I[117] = I[118] = I[119] = I[120] = I[121] = I[122] = I[123] = I[124] = I[125] = I[126] = (img)(0,_p6##y,z,v)), \
 (I[138] = I[139] = I[140] = I[141] = I[142] = I[143] = I[144] = I[145] = I[146] = I[147] = I[148] = I[149] = (img)(0,_p5##y,z,v)), \
 (I[161] = I[162] = I[163] = I[164] = I[165] = I[166] = I[167] = I[168] = I[169] = I[170] = I[171] = I[172] = (img)(0,_p4##y,z,v)), \
 (I[184] = I[185] = I[186] = I[187] = I[188] = I[189] = I[190] = I[191] = I[192] = I[193] = I[194] = I[195] = (img)(0,_p3##y,z,v)), \
 (I[207] = I[208] = I[209] = I[210] = I[211] = I[212] = I[213] = I[214] = I[215] = I[216] = I[217] = I[218] = (img)(0,_p2##y,z,v)), \
 (I[230] = I[231] = I[232] = I[233] = I[234] = I[235] = I[236] = I[237] = I[238] = I[239] = I[240] = I[241] = (img)(0,_p1##y,z,v)), \
 (I[253] = I[254] = I[255] = I[256] = I[257] = I[258] = I[259] = I[260] = I[261] = I[262] = I[263] = I[264] = (img)(0,y,z,v)), \
 (I[276] = I[277] = I[278] = I[279] = I[280] = I[281] = I[282] = I[283] = I[284] = I[285] = I[286] = I[287] = (img)(0,_n1##y,z,v)), \
 (I[299] = I[300] = I[301] = I[302] = I[303] = I[304] = I[305] = I[306] = I[307] = I[308] = I[309] = I[310] = (img)(0,_n2##y,z,v)), \
 (I[322] = I[323] = I[324] = I[325] = I[326] = I[327] = I[328] = I[329] = I[330] = I[331] = I[332] = I[333] = (img)(0,_n3##y,z,v)), \
 (I[345] = I[346] = I[347] = I[348] = I[349] = I[350] = I[351] = I[352] = I[353] = I[354] = I[355] = I[356] = (img)(0,_n4##y,z,v)), \
 (I[368] = I[369] = I[370] = I[371] = I[372] = I[373] = I[374] = I[375] = I[376] = I[377] = I[378] = I[379] = (img)(0,_n5##y,z,v)), \
 (I[391] = I[392] = I[393] = I[394] = I[395] = I[396] = I[397] = I[398] = I[399] = I[400] = I[401] = I[402] = (img)(0,_n6##y,z,v)), \
 (I[414] = I[415] = I[416] = I[417] = I[418] = I[419] = I[420] = I[421] = I[422] = I[423] = I[424] = I[425] = (img)(0,_n7##y,z,v)), \
 (I[437] = I[438] = I[439] = I[440] = I[441] = I[442] = I[443] = I[444] = I[445] = I[446] = I[447] = I[448] = (img)(0,_n8##y,z,v)), \
 (I[460] = I[461] = I[462] = I[463] = I[464] = I[465] = I[466] = I[467] = I[468] = I[469] = I[470] = I[471] = (img)(0,_n9##y,z,v)), \
 (I[483] = I[484] = I[485] = I[486] = I[487] = I[488] = I[489] = I[490] = I[491] = I[492] = I[493] = I[494] = (img)(0,_n10##y,z,v)), \
 (I[506] = I[507] = I[508] = I[509] = I[510] = I[511] = I[512] = I[513] = I[514] = I[515] = I[516] = I[517] = (img)(0,_n11##y,z,v)), \
 (I[12] = (img)(_n1##x,_p11##y,z,v)), \
 (I[35] = (img)(_n1##x,_p10##y,z,v)), \
 (I[58] = (img)(_n1##x,_p9##y,z,v)), \
 (I[81] = (img)(_n1##x,_p8##y,z,v)), \
 (I[104] = (img)(_n1##x,_p7##y,z,v)), \
 (I[127] = (img)(_n1##x,_p6##y,z,v)), \
 (I[150] = (img)(_n1##x,_p5##y,z,v)), \
 (I[173] = (img)(_n1##x,_p4##y,z,v)), \
 (I[196] = (img)(_n1##x,_p3##y,z,v)), \
 (I[219] = (img)(_n1##x,_p2##y,z,v)), \
 (I[242] = (img)(_n1##x,_p1##y,z,v)), \
 (I[265] = (img)(_n1##x,y,z,v)), \
 (I[288] = (img)(_n1##x,_n1##y,z,v)), \
 (I[311] = (img)(_n1##x,_n2##y,z,v)), \
 (I[334] = (img)(_n1##x,_n3##y,z,v)), \
 (I[357] = (img)(_n1##x,_n4##y,z,v)), \
 (I[380] = (img)(_n1##x,_n5##y,z,v)), \
 (I[403] = (img)(_n1##x,_n6##y,z,v)), \
 (I[426] = (img)(_n1##x,_n7##y,z,v)), \
 (I[449] = (img)(_n1##x,_n8##y,z,v)), \
 (I[472] = (img)(_n1##x,_n9##y,z,v)), \
 (I[495] = (img)(_n1##x,_n10##y,z,v)), \
 (I[518] = (img)(_n1##x,_n11##y,z,v)), \
 (I[13] = (img)(_n2##x,_p11##y,z,v)), \
 (I[36] = (img)(_n2##x,_p10##y,z,v)), \
 (I[59] = (img)(_n2##x,_p9##y,z,v)), \
 (I[82] = (img)(_n2##x,_p8##y,z,v)), \
 (I[105] = (img)(_n2##x,_p7##y,z,v)), \
 (I[128] = (img)(_n2##x,_p6##y,z,v)), \
 (I[151] = (img)(_n2##x,_p5##y,z,v)), \
 (I[174] = (img)(_n2##x,_p4##y,z,v)), \
 (I[197] = (img)(_n2##x,_p3##y,z,v)), \
 (I[220] = (img)(_n2##x,_p2##y,z,v)), \
 (I[243] = (img)(_n2##x,_p1##y,z,v)), \
 (I[266] = (img)(_n2##x,y,z,v)), \
 (I[289] = (img)(_n2##x,_n1##y,z,v)), \
 (I[312] = (img)(_n2##x,_n2##y,z,v)), \
 (I[335] = (img)(_n2##x,_n3##y,z,v)), \
 (I[358] = (img)(_n2##x,_n4##y,z,v)), \
 (I[381] = (img)(_n2##x,_n5##y,z,v)), \
 (I[404] = (img)(_n2##x,_n6##y,z,v)), \
 (I[427] = (img)(_n2##x,_n7##y,z,v)), \
 (I[450] = (img)(_n2##x,_n8##y,z,v)), \
 (I[473] = (img)(_n2##x,_n9##y,z,v)), \
 (I[496] = (img)(_n2##x,_n10##y,z,v)), \
 (I[519] = (img)(_n2##x,_n11##y,z,v)), \
 (I[14] = (img)(_n3##x,_p11##y,z,v)), \
 (I[37] = (img)(_n3##x,_p10##y,z,v)), \
 (I[60] = (img)(_n3##x,_p9##y,z,v)), \
 (I[83] = (img)(_n3##x,_p8##y,z,v)), \
 (I[106] = (img)(_n3##x,_p7##y,z,v)), \
 (I[129] = (img)(_n3##x,_p6##y,z,v)), \
 (I[152] = (img)(_n3##x,_p5##y,z,v)), \
 (I[175] = (img)(_n3##x,_p4##y,z,v)), \
 (I[198] = (img)(_n3##x,_p3##y,z,v)), \
 (I[221] = (img)(_n3##x,_p2##y,z,v)), \
 (I[244] = (img)(_n3##x,_p1##y,z,v)), \
 (I[267] = (img)(_n3##x,y,z,v)), \
 (I[290] = (img)(_n3##x,_n1##y,z,v)), \
 (I[313] = (img)(_n3##x,_n2##y,z,v)), \
 (I[336] = (img)(_n3##x,_n3##y,z,v)), \
 (I[359] = (img)(_n3##x,_n4##y,z,v)), \
 (I[382] = (img)(_n3##x,_n5##y,z,v)), \
 (I[405] = (img)(_n3##x,_n6##y,z,v)), \
 (I[428] = (img)(_n3##x,_n7##y,z,v)), \
 (I[451] = (img)(_n3##x,_n8##y,z,v)), \
 (I[474] = (img)(_n3##x,_n9##y,z,v)), \
 (I[497] = (img)(_n3##x,_n10##y,z,v)), \
 (I[520] = (img)(_n3##x,_n11##y,z,v)), \
 (I[15] = (img)(_n4##x,_p11##y,z,v)), \
 (I[38] = (img)(_n4##x,_p10##y,z,v)), \
 (I[61] = (img)(_n4##x,_p9##y,z,v)), \
 (I[84] = (img)(_n4##x,_p8##y,z,v)), \
 (I[107] = (img)(_n4##x,_p7##y,z,v)), \
 (I[130] = (img)(_n4##x,_p6##y,z,v)), \
 (I[153] = (img)(_n4##x,_p5##y,z,v)), \
 (I[176] = (img)(_n4##x,_p4##y,z,v)), \
 (I[199] = (img)(_n4##x,_p3##y,z,v)), \
 (I[222] = (img)(_n4##x,_p2##y,z,v)), \
 (I[245] = (img)(_n4##x,_p1##y,z,v)), \
 (I[268] = (img)(_n4##x,y,z,v)), \
 (I[291] = (img)(_n4##x,_n1##y,z,v)), \
 (I[314] = (img)(_n4##x,_n2##y,z,v)), \
 (I[337] = (img)(_n4##x,_n3##y,z,v)), \
 (I[360] = (img)(_n4##x,_n4##y,z,v)), \
 (I[383] = (img)(_n4##x,_n5##y,z,v)), \
 (I[406] = (img)(_n4##x,_n6##y,z,v)), \
 (I[429] = (img)(_n4##x,_n7##y,z,v)), \
 (I[452] = (img)(_n4##x,_n8##y,z,v)), \
 (I[475] = (img)(_n4##x,_n9##y,z,v)), \
 (I[498] = (img)(_n4##x,_n10##y,z,v)), \
 (I[521] = (img)(_n4##x,_n11##y,z,v)), \
 (I[16] = (img)(_n5##x,_p11##y,z,v)), \
 (I[39] = (img)(_n5##x,_p10##y,z,v)), \
 (I[62] = (img)(_n5##x,_p9##y,z,v)), \
 (I[85] = (img)(_n5##x,_p8##y,z,v)), \
 (I[108] = (img)(_n5##x,_p7##y,z,v)), \
 (I[131] = (img)(_n5##x,_p6##y,z,v)), \
 (I[154] = (img)(_n5##x,_p5##y,z,v)), \
 (I[177] = (img)(_n5##x,_p4##y,z,v)), \
 (I[200] = (img)(_n5##x,_p3##y,z,v)), \
 (I[223] = (img)(_n5##x,_p2##y,z,v)), \
 (I[246] = (img)(_n5##x,_p1##y,z,v)), \
 (I[269] = (img)(_n5##x,y,z,v)), \
 (I[292] = (img)(_n5##x,_n1##y,z,v)), \
 (I[315] = (img)(_n5##x,_n2##y,z,v)), \
 (I[338] = (img)(_n5##x,_n3##y,z,v)), \
 (I[361] = (img)(_n5##x,_n4##y,z,v)), \
 (I[384] = (img)(_n5##x,_n5##y,z,v)), \
 (I[407] = (img)(_n5##x,_n6##y,z,v)), \
 (I[430] = (img)(_n5##x,_n7##y,z,v)), \
 (I[453] = (img)(_n5##x,_n8##y,z,v)), \
 (I[476] = (img)(_n5##x,_n9##y,z,v)), \
 (I[499] = (img)(_n5##x,_n10##y,z,v)), \
 (I[522] = (img)(_n5##x,_n11##y,z,v)), \
 (I[17] = (img)(_n6##x,_p11##y,z,v)), \
 (I[40] = (img)(_n6##x,_p10##y,z,v)), \
 (I[63] = (img)(_n6##x,_p9##y,z,v)), \
 (I[86] = (img)(_n6##x,_p8##y,z,v)), \
 (I[109] = (img)(_n6##x,_p7##y,z,v)), \
 (I[132] = (img)(_n6##x,_p6##y,z,v)), \
 (I[155] = (img)(_n6##x,_p5##y,z,v)), \
 (I[178] = (img)(_n6##x,_p4##y,z,v)), \
 (I[201] = (img)(_n6##x,_p3##y,z,v)), \
 (I[224] = (img)(_n6##x,_p2##y,z,v)), \
 (I[247] = (img)(_n6##x,_p1##y,z,v)), \
 (I[270] = (img)(_n6##x,y,z,v)), \
 (I[293] = (img)(_n6##x,_n1##y,z,v)), \
 (I[316] = (img)(_n6##x,_n2##y,z,v)), \
 (I[339] = (img)(_n6##x,_n3##y,z,v)), \
 (I[362] = (img)(_n6##x,_n4##y,z,v)), \
 (I[385] = (img)(_n6##x,_n5##y,z,v)), \
 (I[408] = (img)(_n6##x,_n6##y,z,v)), \
 (I[431] = (img)(_n6##x,_n7##y,z,v)), \
 (I[454] = (img)(_n6##x,_n8##y,z,v)), \
 (I[477] = (img)(_n6##x,_n9##y,z,v)), \
 (I[500] = (img)(_n6##x,_n10##y,z,v)), \
 (I[523] = (img)(_n6##x,_n11##y,z,v)), \
 (I[18] = (img)(_n7##x,_p11##y,z,v)), \
 (I[41] = (img)(_n7##x,_p10##y,z,v)), \
 (I[64] = (img)(_n7##x,_p9##y,z,v)), \
 (I[87] = (img)(_n7##x,_p8##y,z,v)), \
 (I[110] = (img)(_n7##x,_p7##y,z,v)), \
 (I[133] = (img)(_n7##x,_p6##y,z,v)), \
 (I[156] = (img)(_n7##x,_p5##y,z,v)), \
 (I[179] = (img)(_n7##x,_p4##y,z,v)), \
 (I[202] = (img)(_n7##x,_p3##y,z,v)), \
 (I[225] = (img)(_n7##x,_p2##y,z,v)), \
 (I[248] = (img)(_n7##x,_p1##y,z,v)), \
 (I[271] = (img)(_n7##x,y,z,v)), \
 (I[294] = (img)(_n7##x,_n1##y,z,v)), \
 (I[317] = (img)(_n7##x,_n2##y,z,v)), \
 (I[340] = (img)(_n7##x,_n3##y,z,v)), \
 (I[363] = (img)(_n7##x,_n4##y,z,v)), \
 (I[386] = (img)(_n7##x,_n5##y,z,v)), \
 (I[409] = (img)(_n7##x,_n6##y,z,v)), \
 (I[432] = (img)(_n7##x,_n7##y,z,v)), \
 (I[455] = (img)(_n7##x,_n8##y,z,v)), \
 (I[478] = (img)(_n7##x,_n9##y,z,v)), \
 (I[501] = (img)(_n7##x,_n10##y,z,v)), \
 (I[524] = (img)(_n7##x,_n11##y,z,v)), \
 (I[19] = (img)(_n8##x,_p11##y,z,v)), \
 (I[42] = (img)(_n8##x,_p10##y,z,v)), \
 (I[65] = (img)(_n8##x,_p9##y,z,v)), \
 (I[88] = (img)(_n8##x,_p8##y,z,v)), \
 (I[111] = (img)(_n8##x,_p7##y,z,v)), \
 (I[134] = (img)(_n8##x,_p6##y,z,v)), \
 (I[157] = (img)(_n8##x,_p5##y,z,v)), \
 (I[180] = (img)(_n8##x,_p4##y,z,v)), \
 (I[203] = (img)(_n8##x,_p3##y,z,v)), \
 (I[226] = (img)(_n8##x,_p2##y,z,v)), \
 (I[249] = (img)(_n8##x,_p1##y,z,v)), \
 (I[272] = (img)(_n8##x,y,z,v)), \
 (I[295] = (img)(_n8##x,_n1##y,z,v)), \
 (I[318] = (img)(_n8##x,_n2##y,z,v)), \
 (I[341] = (img)(_n8##x,_n3##y,z,v)), \
 (I[364] = (img)(_n8##x,_n4##y,z,v)), \
 (I[387] = (img)(_n8##x,_n5##y,z,v)), \
 (I[410] = (img)(_n8##x,_n6##y,z,v)), \
 (I[433] = (img)(_n8##x,_n7##y,z,v)), \
 (I[456] = (img)(_n8##x,_n8##y,z,v)), \
 (I[479] = (img)(_n8##x,_n9##y,z,v)), \
 (I[502] = (img)(_n8##x,_n10##y,z,v)), \
 (I[525] = (img)(_n8##x,_n11##y,z,v)), \
 (I[20] = (img)(_n9##x,_p11##y,z,v)), \
 (I[43] = (img)(_n9##x,_p10##y,z,v)), \
 (I[66] = (img)(_n9##x,_p9##y,z,v)), \
 (I[89] = (img)(_n9##x,_p8##y,z,v)), \
 (I[112] = (img)(_n9##x,_p7##y,z,v)), \
 (I[135] = (img)(_n9##x,_p6##y,z,v)), \
 (I[158] = (img)(_n9##x,_p5##y,z,v)), \
 (I[181] = (img)(_n9##x,_p4##y,z,v)), \
 (I[204] = (img)(_n9##x,_p3##y,z,v)), \
 (I[227] = (img)(_n9##x,_p2##y,z,v)), \
 (I[250] = (img)(_n9##x,_p1##y,z,v)), \
 (I[273] = (img)(_n9##x,y,z,v)), \
 (I[296] = (img)(_n9##x,_n1##y,z,v)), \
 (I[319] = (img)(_n9##x,_n2##y,z,v)), \
 (I[342] = (img)(_n9##x,_n3##y,z,v)), \
 (I[365] = (img)(_n9##x,_n4##y,z,v)), \
 (I[388] = (img)(_n9##x,_n5##y,z,v)), \
 (I[411] = (img)(_n9##x,_n6##y,z,v)), \
 (I[434] = (img)(_n9##x,_n7##y,z,v)), \
 (I[457] = (img)(_n9##x,_n8##y,z,v)), \
 (I[480] = (img)(_n9##x,_n9##y,z,v)), \
 (I[503] = (img)(_n9##x,_n10##y,z,v)), \
 (I[526] = (img)(_n9##x,_n11##y,z,v)), \
 (I[21] = (img)(_n10##x,_p11##y,z,v)), \
 (I[44] = (img)(_n10##x,_p10##y,z,v)), \
 (I[67] = (img)(_n10##x,_p9##y,z,v)), \
 (I[90] = (img)(_n10##x,_p8##y,z,v)), \
 (I[113] = (img)(_n10##x,_p7##y,z,v)), \
 (I[136] = (img)(_n10##x,_p6##y,z,v)), \
 (I[159] = (img)(_n10##x,_p5##y,z,v)), \
 (I[182] = (img)(_n10##x,_p4##y,z,v)), \
 (I[205] = (img)(_n10##x,_p3##y,z,v)), \
 (I[228] = (img)(_n10##x,_p2##y,z,v)), \
 (I[251] = (img)(_n10##x,_p1##y,z,v)), \
 (I[274] = (img)(_n10##x,y,z,v)), \
 (I[297] = (img)(_n10##x,_n1##y,z,v)), \
 (I[320] = (img)(_n10##x,_n2##y,z,v)), \
 (I[343] = (img)(_n10##x,_n3##y,z,v)), \
 (I[366] = (img)(_n10##x,_n4##y,z,v)), \
 (I[389] = (img)(_n10##x,_n5##y,z,v)), \
 (I[412] = (img)(_n10##x,_n6##y,z,v)), \
 (I[435] = (img)(_n10##x,_n7##y,z,v)), \
 (I[458] = (img)(_n10##x,_n8##y,z,v)), \
 (I[481] = (img)(_n10##x,_n9##y,z,v)), \
 (I[504] = (img)(_n10##x,_n10##y,z,v)), \
 (I[527] = (img)(_n10##x,_n11##y,z,v)), \
 11>=((img).width)?(int)((img).width)-1:11); \
 (_n11##x<(int)((img).width) && ( \
 (I[22] = (img)(_n11##x,_p11##y,z,v)), \
 (I[45] = (img)(_n11##x,_p10##y,z,v)), \
 (I[68] = (img)(_n11##x,_p9##y,z,v)), \
 (I[91] = (img)(_n11##x,_p8##y,z,v)), \
 (I[114] = (img)(_n11##x,_p7##y,z,v)), \
 (I[137] = (img)(_n11##x,_p6##y,z,v)), \
 (I[160] = (img)(_n11##x,_p5##y,z,v)), \
 (I[183] = (img)(_n11##x,_p4##y,z,v)), \
 (I[206] = (img)(_n11##x,_p3##y,z,v)), \
 (I[229] = (img)(_n11##x,_p2##y,z,v)), \
 (I[252] = (img)(_n11##x,_p1##y,z,v)), \
 (I[275] = (img)(_n11##x,y,z,v)), \
 (I[298] = (img)(_n11##x,_n1##y,z,v)), \
 (I[321] = (img)(_n11##x,_n2##y,z,v)), \
 (I[344] = (img)(_n11##x,_n3##y,z,v)), \
 (I[367] = (img)(_n11##x,_n4##y,z,v)), \
 (I[390] = (img)(_n11##x,_n5##y,z,v)), \
 (I[413] = (img)(_n11##x,_n6##y,z,v)), \
 (I[436] = (img)(_n11##x,_n7##y,z,v)), \
 (I[459] = (img)(_n11##x,_n8##y,z,v)), \
 (I[482] = (img)(_n11##x,_n9##y,z,v)), \
 (I[505] = (img)(_n11##x,_n10##y,z,v)), \
 (I[528] = (img)(_n11##x,_n11##y,z,v)),1)) || \
 _n10##x==--_n11##x || _n9##x==--_n10##x || _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n11##x = _n10##x = _n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], \
 I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], \
 I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], \
 I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], \
 I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], \
 I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], \
 I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], \
 I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], \
 I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], \
 I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], \
 I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], \
 I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], \
 I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], \
 I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], \
 I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], I[335] = I[336], I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], I[342] = I[343], I[343] = I[344], \
 I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], I[351] = I[352], I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], I[359] = I[360], I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], \
 I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], I[373] = I[374], I[374] = I[375], I[375] = I[376], I[376] = I[377], I[377] = I[378], I[378] = I[379], I[379] = I[380], I[380] = I[381], I[381] = I[382], I[382] = I[383], I[383] = I[384], I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], \
 I[391] = I[392], I[392] = I[393], I[393] = I[394], I[394] = I[395], I[395] = I[396], I[396] = I[397], I[397] = I[398], I[398] = I[399], I[399] = I[400], I[400] = I[401], I[401] = I[402], I[402] = I[403], I[403] = I[404], I[404] = I[405], I[405] = I[406], I[406] = I[407], I[407] = I[408], I[408] = I[409], I[409] = I[410], I[410] = I[411], I[411] = I[412], I[412] = I[413], \
 I[414] = I[415], I[415] = I[416], I[416] = I[417], I[417] = I[418], I[418] = I[419], I[419] = I[420], I[420] = I[421], I[421] = I[422], I[422] = I[423], I[423] = I[424], I[424] = I[425], I[425] = I[426], I[426] = I[427], I[427] = I[428], I[428] = I[429], I[429] = I[430], I[430] = I[431], I[431] = I[432], I[432] = I[433], I[433] = I[434], I[434] = I[435], I[435] = I[436], \
 I[437] = I[438], I[438] = I[439], I[439] = I[440], I[440] = I[441], I[441] = I[442], I[442] = I[443], I[443] = I[444], I[444] = I[445], I[445] = I[446], I[446] = I[447], I[447] = I[448], I[448] = I[449], I[449] = I[450], I[450] = I[451], I[451] = I[452], I[452] = I[453], I[453] = I[454], I[454] = I[455], I[455] = I[456], I[456] = I[457], I[457] = I[458], I[458] = I[459], \
 I[460] = I[461], I[461] = I[462], I[462] = I[463], I[463] = I[464], I[464] = I[465], I[465] = I[466], I[466] = I[467], I[467] = I[468], I[468] = I[469], I[469] = I[470], I[470] = I[471], I[471] = I[472], I[472] = I[473], I[473] = I[474], I[474] = I[475], I[475] = I[476], I[476] = I[477], I[477] = I[478], I[478] = I[479], I[479] = I[480], I[480] = I[481], I[481] = I[482], \
 I[483] = I[484], I[484] = I[485], I[485] = I[486], I[486] = I[487], I[487] = I[488], I[488] = I[489], I[489] = I[490], I[490] = I[491], I[491] = I[492], I[492] = I[493], I[493] = I[494], I[494] = I[495], I[495] = I[496], I[496] = I[497], I[497] = I[498], I[498] = I[499], I[499] = I[500], I[500] = I[501], I[501] = I[502], I[502] = I[503], I[503] = I[504], I[504] = I[505], \
 I[506] = I[507], I[507] = I[508], I[508] = I[509], I[509] = I[510], I[510] = I[511], I[511] = I[512], I[512] = I[513], I[513] = I[514], I[514] = I[515], I[515] = I[516], I[516] = I[517], I[517] = I[518], I[518] = I[519], I[519] = I[520], I[520] = I[521], I[521] = I[522], I[522] = I[523], I[523] = I[524], I[524] = I[525], I[525] = I[526], I[526] = I[527], I[527] = I[528], \
 _p11##x = _p10##x, _p10##x = _p9##x, _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x, ++_n10##x, ++_n11##x)

#define cimg_for_in23x23(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in23((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p11##x = x-11<0?0:x-11, \
 _p10##x = x-10<0?0:x-10, \
 _p9##x = x-9<0?0:x-9, \
 _p8##x = x-8<0?0:x-8, \
 _p7##x = x-7<0?0:x-7, \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = x+7>=(int)((img).width)?(int)((img).width)-1:x+7, \
 _n8##x = x+8>=(int)((img).width)?(int)((img).width)-1:x+8, \
 _n9##x = x+9>=(int)((img).width)?(int)((img).width)-1:x+9, \
 _n10##x = x+10>=(int)((img).width)?(int)((img).width)-1:x+10, \
 _n11##x = (int)( \
 (I[0] = (img)(_p11##x,_p11##y,z,v)), \
 (I[23] = (img)(_p11##x,_p10##y,z,v)), \
 (I[46] = (img)(_p11##x,_p9##y,z,v)), \
 (I[69] = (img)(_p11##x,_p8##y,z,v)), \
 (I[92] = (img)(_p11##x,_p7##y,z,v)), \
 (I[115] = (img)(_p11##x,_p6##y,z,v)), \
 (I[138] = (img)(_p11##x,_p5##y,z,v)), \
 (I[161] = (img)(_p11##x,_p4##y,z,v)), \
 (I[184] = (img)(_p11##x,_p3##y,z,v)), \
 (I[207] = (img)(_p11##x,_p2##y,z,v)), \
 (I[230] = (img)(_p11##x,_p1##y,z,v)), \
 (I[253] = (img)(_p11##x,y,z,v)), \
 (I[276] = (img)(_p11##x,_n1##y,z,v)), \
 (I[299] = (img)(_p11##x,_n2##y,z,v)), \
 (I[322] = (img)(_p11##x,_n3##y,z,v)), \
 (I[345] = (img)(_p11##x,_n4##y,z,v)), \
 (I[368] = (img)(_p11##x,_n5##y,z,v)), \
 (I[391] = (img)(_p11##x,_n6##y,z,v)), \
 (I[414] = (img)(_p11##x,_n7##y,z,v)), \
 (I[437] = (img)(_p11##x,_n8##y,z,v)), \
 (I[460] = (img)(_p11##x,_n9##y,z,v)), \
 (I[483] = (img)(_p11##x,_n10##y,z,v)), \
 (I[506] = (img)(_p11##x,_n11##y,z,v)), \
 (I[1] = (img)(_p10##x,_p11##y,z,v)), \
 (I[24] = (img)(_p10##x,_p10##y,z,v)), \
 (I[47] = (img)(_p10##x,_p9##y,z,v)), \
 (I[70] = (img)(_p10##x,_p8##y,z,v)), \
 (I[93] = (img)(_p10##x,_p7##y,z,v)), \
 (I[116] = (img)(_p10##x,_p6##y,z,v)), \
 (I[139] = (img)(_p10##x,_p5##y,z,v)), \
 (I[162] = (img)(_p10##x,_p4##y,z,v)), \
 (I[185] = (img)(_p10##x,_p3##y,z,v)), \
 (I[208] = (img)(_p10##x,_p2##y,z,v)), \
 (I[231] = (img)(_p10##x,_p1##y,z,v)), \
 (I[254] = (img)(_p10##x,y,z,v)), \
 (I[277] = (img)(_p10##x,_n1##y,z,v)), \
 (I[300] = (img)(_p10##x,_n2##y,z,v)), \
 (I[323] = (img)(_p10##x,_n3##y,z,v)), \
 (I[346] = (img)(_p10##x,_n4##y,z,v)), \
 (I[369] = (img)(_p10##x,_n5##y,z,v)), \
 (I[392] = (img)(_p10##x,_n6##y,z,v)), \
 (I[415] = (img)(_p10##x,_n7##y,z,v)), \
 (I[438] = (img)(_p10##x,_n8##y,z,v)), \
 (I[461] = (img)(_p10##x,_n9##y,z,v)), \
 (I[484] = (img)(_p10##x,_n10##y,z,v)), \
 (I[507] = (img)(_p10##x,_n11##y,z,v)), \
 (I[2] = (img)(_p9##x,_p11##y,z,v)), \
 (I[25] = (img)(_p9##x,_p10##y,z,v)), \
 (I[48] = (img)(_p9##x,_p9##y,z,v)), \
 (I[71] = (img)(_p9##x,_p8##y,z,v)), \
 (I[94] = (img)(_p9##x,_p7##y,z,v)), \
 (I[117] = (img)(_p9##x,_p6##y,z,v)), \
 (I[140] = (img)(_p9##x,_p5##y,z,v)), \
 (I[163] = (img)(_p9##x,_p4##y,z,v)), \
 (I[186] = (img)(_p9##x,_p3##y,z,v)), \
 (I[209] = (img)(_p9##x,_p2##y,z,v)), \
 (I[232] = (img)(_p9##x,_p1##y,z,v)), \
 (I[255] = (img)(_p9##x,y,z,v)), \
 (I[278] = (img)(_p9##x,_n1##y,z,v)), \
 (I[301] = (img)(_p9##x,_n2##y,z,v)), \
 (I[324] = (img)(_p9##x,_n3##y,z,v)), \
 (I[347] = (img)(_p9##x,_n4##y,z,v)), \
 (I[370] = (img)(_p9##x,_n5##y,z,v)), \
 (I[393] = (img)(_p9##x,_n6##y,z,v)), \
 (I[416] = (img)(_p9##x,_n7##y,z,v)), \
 (I[439] = (img)(_p9##x,_n8##y,z,v)), \
 (I[462] = (img)(_p9##x,_n9##y,z,v)), \
 (I[485] = (img)(_p9##x,_n10##y,z,v)), \
 (I[508] = (img)(_p9##x,_n11##y,z,v)), \
 (I[3] = (img)(_p8##x,_p11##y,z,v)), \
 (I[26] = (img)(_p8##x,_p10##y,z,v)), \
 (I[49] = (img)(_p8##x,_p9##y,z,v)), \
 (I[72] = (img)(_p8##x,_p8##y,z,v)), \
 (I[95] = (img)(_p8##x,_p7##y,z,v)), \
 (I[118] = (img)(_p8##x,_p6##y,z,v)), \
 (I[141] = (img)(_p8##x,_p5##y,z,v)), \
 (I[164] = (img)(_p8##x,_p4##y,z,v)), \
 (I[187] = (img)(_p8##x,_p3##y,z,v)), \
 (I[210] = (img)(_p8##x,_p2##y,z,v)), \
 (I[233] = (img)(_p8##x,_p1##y,z,v)), \
 (I[256] = (img)(_p8##x,y,z,v)), \
 (I[279] = (img)(_p8##x,_n1##y,z,v)), \
 (I[302] = (img)(_p8##x,_n2##y,z,v)), \
 (I[325] = (img)(_p8##x,_n3##y,z,v)), \
 (I[348] = (img)(_p8##x,_n4##y,z,v)), \
 (I[371] = (img)(_p8##x,_n5##y,z,v)), \
 (I[394] = (img)(_p8##x,_n6##y,z,v)), \
 (I[417] = (img)(_p8##x,_n7##y,z,v)), \
 (I[440] = (img)(_p8##x,_n8##y,z,v)), \
 (I[463] = (img)(_p8##x,_n9##y,z,v)), \
 (I[486] = (img)(_p8##x,_n10##y,z,v)), \
 (I[509] = (img)(_p8##x,_n11##y,z,v)), \
 (I[4] = (img)(_p7##x,_p11##y,z,v)), \
 (I[27] = (img)(_p7##x,_p10##y,z,v)), \
 (I[50] = (img)(_p7##x,_p9##y,z,v)), \
 (I[73] = (img)(_p7##x,_p8##y,z,v)), \
 (I[96] = (img)(_p7##x,_p7##y,z,v)), \
 (I[119] = (img)(_p7##x,_p6##y,z,v)), \
 (I[142] = (img)(_p7##x,_p5##y,z,v)), \
 (I[165] = (img)(_p7##x,_p4##y,z,v)), \
 (I[188] = (img)(_p7##x,_p3##y,z,v)), \
 (I[211] = (img)(_p7##x,_p2##y,z,v)), \
 (I[234] = (img)(_p7##x,_p1##y,z,v)), \
 (I[257] = (img)(_p7##x,y,z,v)), \
 (I[280] = (img)(_p7##x,_n1##y,z,v)), \
 (I[303] = (img)(_p7##x,_n2##y,z,v)), \
 (I[326] = (img)(_p7##x,_n3##y,z,v)), \
 (I[349] = (img)(_p7##x,_n4##y,z,v)), \
 (I[372] = (img)(_p7##x,_n5##y,z,v)), \
 (I[395] = (img)(_p7##x,_n6##y,z,v)), \
 (I[418] = (img)(_p7##x,_n7##y,z,v)), \
 (I[441] = (img)(_p7##x,_n8##y,z,v)), \
 (I[464] = (img)(_p7##x,_n9##y,z,v)), \
 (I[487] = (img)(_p7##x,_n10##y,z,v)), \
 (I[510] = (img)(_p7##x,_n11##y,z,v)), \
 (I[5] = (img)(_p6##x,_p11##y,z,v)), \
 (I[28] = (img)(_p6##x,_p10##y,z,v)), \
 (I[51] = (img)(_p6##x,_p9##y,z,v)), \
 (I[74] = (img)(_p6##x,_p8##y,z,v)), \
 (I[97] = (img)(_p6##x,_p7##y,z,v)), \
 (I[120] = (img)(_p6##x,_p6##y,z,v)), \
 (I[143] = (img)(_p6##x,_p5##y,z,v)), \
 (I[166] = (img)(_p6##x,_p4##y,z,v)), \
 (I[189] = (img)(_p6##x,_p3##y,z,v)), \
 (I[212] = (img)(_p6##x,_p2##y,z,v)), \
 (I[235] = (img)(_p6##x,_p1##y,z,v)), \
 (I[258] = (img)(_p6##x,y,z,v)), \
 (I[281] = (img)(_p6##x,_n1##y,z,v)), \
 (I[304] = (img)(_p6##x,_n2##y,z,v)), \
 (I[327] = (img)(_p6##x,_n3##y,z,v)), \
 (I[350] = (img)(_p6##x,_n4##y,z,v)), \
 (I[373] = (img)(_p6##x,_n5##y,z,v)), \
 (I[396] = (img)(_p6##x,_n6##y,z,v)), \
 (I[419] = (img)(_p6##x,_n7##y,z,v)), \
 (I[442] = (img)(_p6##x,_n8##y,z,v)), \
 (I[465] = (img)(_p6##x,_n9##y,z,v)), \
 (I[488] = (img)(_p6##x,_n10##y,z,v)), \
 (I[511] = (img)(_p6##x,_n11##y,z,v)), \
 (I[6] = (img)(_p5##x,_p11##y,z,v)), \
 (I[29] = (img)(_p5##x,_p10##y,z,v)), \
 (I[52] = (img)(_p5##x,_p9##y,z,v)), \
 (I[75] = (img)(_p5##x,_p8##y,z,v)), \
 (I[98] = (img)(_p5##x,_p7##y,z,v)), \
 (I[121] = (img)(_p5##x,_p6##y,z,v)), \
 (I[144] = (img)(_p5##x,_p5##y,z,v)), \
 (I[167] = (img)(_p5##x,_p4##y,z,v)), \
 (I[190] = (img)(_p5##x,_p3##y,z,v)), \
 (I[213] = (img)(_p5##x,_p2##y,z,v)), \
 (I[236] = (img)(_p5##x,_p1##y,z,v)), \
 (I[259] = (img)(_p5##x,y,z,v)), \
 (I[282] = (img)(_p5##x,_n1##y,z,v)), \
 (I[305] = (img)(_p5##x,_n2##y,z,v)), \
 (I[328] = (img)(_p5##x,_n3##y,z,v)), \
 (I[351] = (img)(_p5##x,_n4##y,z,v)), \
 (I[374] = (img)(_p5##x,_n5##y,z,v)), \
 (I[397] = (img)(_p5##x,_n6##y,z,v)), \
 (I[420] = (img)(_p5##x,_n7##y,z,v)), \
 (I[443] = (img)(_p5##x,_n8##y,z,v)), \
 (I[466] = (img)(_p5##x,_n9##y,z,v)), \
 (I[489] = (img)(_p5##x,_n10##y,z,v)), \
 (I[512] = (img)(_p5##x,_n11##y,z,v)), \
 (I[7] = (img)(_p4##x,_p11##y,z,v)), \
 (I[30] = (img)(_p4##x,_p10##y,z,v)), \
 (I[53] = (img)(_p4##x,_p9##y,z,v)), \
 (I[76] = (img)(_p4##x,_p8##y,z,v)), \
 (I[99] = (img)(_p4##x,_p7##y,z,v)), \
 (I[122] = (img)(_p4##x,_p6##y,z,v)), \
 (I[145] = (img)(_p4##x,_p5##y,z,v)), \
 (I[168] = (img)(_p4##x,_p4##y,z,v)), \
 (I[191] = (img)(_p4##x,_p3##y,z,v)), \
 (I[214] = (img)(_p4##x,_p2##y,z,v)), \
 (I[237] = (img)(_p4##x,_p1##y,z,v)), \
 (I[260] = (img)(_p4##x,y,z,v)), \
 (I[283] = (img)(_p4##x,_n1##y,z,v)), \
 (I[306] = (img)(_p4##x,_n2##y,z,v)), \
 (I[329] = (img)(_p4##x,_n3##y,z,v)), \
 (I[352] = (img)(_p4##x,_n4##y,z,v)), \
 (I[375] = (img)(_p4##x,_n5##y,z,v)), \
 (I[398] = (img)(_p4##x,_n6##y,z,v)), \
 (I[421] = (img)(_p4##x,_n7##y,z,v)), \
 (I[444] = (img)(_p4##x,_n8##y,z,v)), \
 (I[467] = (img)(_p4##x,_n9##y,z,v)), \
 (I[490] = (img)(_p4##x,_n10##y,z,v)), \
 (I[513] = (img)(_p4##x,_n11##y,z,v)), \
 (I[8] = (img)(_p3##x,_p11##y,z,v)), \
 (I[31] = (img)(_p3##x,_p10##y,z,v)), \
 (I[54] = (img)(_p3##x,_p9##y,z,v)), \
 (I[77] = (img)(_p3##x,_p8##y,z,v)), \
 (I[100] = (img)(_p3##x,_p7##y,z,v)), \
 (I[123] = (img)(_p3##x,_p6##y,z,v)), \
 (I[146] = (img)(_p3##x,_p5##y,z,v)), \
 (I[169] = (img)(_p3##x,_p4##y,z,v)), \
 (I[192] = (img)(_p3##x,_p3##y,z,v)), \
 (I[215] = (img)(_p3##x,_p2##y,z,v)), \
 (I[238] = (img)(_p3##x,_p1##y,z,v)), \
 (I[261] = (img)(_p3##x,y,z,v)), \
 (I[284] = (img)(_p3##x,_n1##y,z,v)), \
 (I[307] = (img)(_p3##x,_n2##y,z,v)), \
 (I[330] = (img)(_p3##x,_n3##y,z,v)), \
 (I[353] = (img)(_p3##x,_n4##y,z,v)), \
 (I[376] = (img)(_p3##x,_n5##y,z,v)), \
 (I[399] = (img)(_p3##x,_n6##y,z,v)), \
 (I[422] = (img)(_p3##x,_n7##y,z,v)), \
 (I[445] = (img)(_p3##x,_n8##y,z,v)), \
 (I[468] = (img)(_p3##x,_n9##y,z,v)), \
 (I[491] = (img)(_p3##x,_n10##y,z,v)), \
 (I[514] = (img)(_p3##x,_n11##y,z,v)), \
 (I[9] = (img)(_p2##x,_p11##y,z,v)), \
 (I[32] = (img)(_p2##x,_p10##y,z,v)), \
 (I[55] = (img)(_p2##x,_p9##y,z,v)), \
 (I[78] = (img)(_p2##x,_p8##y,z,v)), \
 (I[101] = (img)(_p2##x,_p7##y,z,v)), \
 (I[124] = (img)(_p2##x,_p6##y,z,v)), \
 (I[147] = (img)(_p2##x,_p5##y,z,v)), \
 (I[170] = (img)(_p2##x,_p4##y,z,v)), \
 (I[193] = (img)(_p2##x,_p3##y,z,v)), \
 (I[216] = (img)(_p2##x,_p2##y,z,v)), \
 (I[239] = (img)(_p2##x,_p1##y,z,v)), \
 (I[262] = (img)(_p2##x,y,z,v)), \
 (I[285] = (img)(_p2##x,_n1##y,z,v)), \
 (I[308] = (img)(_p2##x,_n2##y,z,v)), \
 (I[331] = (img)(_p2##x,_n3##y,z,v)), \
 (I[354] = (img)(_p2##x,_n4##y,z,v)), \
 (I[377] = (img)(_p2##x,_n5##y,z,v)), \
 (I[400] = (img)(_p2##x,_n6##y,z,v)), \
 (I[423] = (img)(_p2##x,_n7##y,z,v)), \
 (I[446] = (img)(_p2##x,_n8##y,z,v)), \
 (I[469] = (img)(_p2##x,_n9##y,z,v)), \
 (I[492] = (img)(_p2##x,_n10##y,z,v)), \
 (I[515] = (img)(_p2##x,_n11##y,z,v)), \
 (I[10] = (img)(_p1##x,_p11##y,z,v)), \
 (I[33] = (img)(_p1##x,_p10##y,z,v)), \
 (I[56] = (img)(_p1##x,_p9##y,z,v)), \
 (I[79] = (img)(_p1##x,_p8##y,z,v)), \
 (I[102] = (img)(_p1##x,_p7##y,z,v)), \
 (I[125] = (img)(_p1##x,_p6##y,z,v)), \
 (I[148] = (img)(_p1##x,_p5##y,z,v)), \
 (I[171] = (img)(_p1##x,_p4##y,z,v)), \
 (I[194] = (img)(_p1##x,_p3##y,z,v)), \
 (I[217] = (img)(_p1##x,_p2##y,z,v)), \
 (I[240] = (img)(_p1##x,_p1##y,z,v)), \
 (I[263] = (img)(_p1##x,y,z,v)), \
 (I[286] = (img)(_p1##x,_n1##y,z,v)), \
 (I[309] = (img)(_p1##x,_n2##y,z,v)), \
 (I[332] = (img)(_p1##x,_n3##y,z,v)), \
 (I[355] = (img)(_p1##x,_n4##y,z,v)), \
 (I[378] = (img)(_p1##x,_n5##y,z,v)), \
 (I[401] = (img)(_p1##x,_n6##y,z,v)), \
 (I[424] = (img)(_p1##x,_n7##y,z,v)), \
 (I[447] = (img)(_p1##x,_n8##y,z,v)), \
 (I[470] = (img)(_p1##x,_n9##y,z,v)), \
 (I[493] = (img)(_p1##x,_n10##y,z,v)), \
 (I[516] = (img)(_p1##x,_n11##y,z,v)), \
 (I[11] = (img)(x,_p11##y,z,v)), \
 (I[34] = (img)(x,_p10##y,z,v)), \
 (I[57] = (img)(x,_p9##y,z,v)), \
 (I[80] = (img)(x,_p8##y,z,v)), \
 (I[103] = (img)(x,_p7##y,z,v)), \
 (I[126] = (img)(x,_p6##y,z,v)), \
 (I[149] = (img)(x,_p5##y,z,v)), \
 (I[172] = (img)(x,_p4##y,z,v)), \
 (I[195] = (img)(x,_p3##y,z,v)), \
 (I[218] = (img)(x,_p2##y,z,v)), \
 (I[241] = (img)(x,_p1##y,z,v)), \
 (I[264] = (img)(x,y,z,v)), \
 (I[287] = (img)(x,_n1##y,z,v)), \
 (I[310] = (img)(x,_n2##y,z,v)), \
 (I[333] = (img)(x,_n3##y,z,v)), \
 (I[356] = (img)(x,_n4##y,z,v)), \
 (I[379] = (img)(x,_n5##y,z,v)), \
 (I[402] = (img)(x,_n6##y,z,v)), \
 (I[425] = (img)(x,_n7##y,z,v)), \
 (I[448] = (img)(x,_n8##y,z,v)), \
 (I[471] = (img)(x,_n9##y,z,v)), \
 (I[494] = (img)(x,_n10##y,z,v)), \
 (I[517] = (img)(x,_n11##y,z,v)), \
 (I[12] = (img)(_n1##x,_p11##y,z,v)), \
 (I[35] = (img)(_n1##x,_p10##y,z,v)), \
 (I[58] = (img)(_n1##x,_p9##y,z,v)), \
 (I[81] = (img)(_n1##x,_p8##y,z,v)), \
 (I[104] = (img)(_n1##x,_p7##y,z,v)), \
 (I[127] = (img)(_n1##x,_p6##y,z,v)), \
 (I[150] = (img)(_n1##x,_p5##y,z,v)), \
 (I[173] = (img)(_n1##x,_p4##y,z,v)), \
 (I[196] = (img)(_n1##x,_p3##y,z,v)), \
 (I[219] = (img)(_n1##x,_p2##y,z,v)), \
 (I[242] = (img)(_n1##x,_p1##y,z,v)), \
 (I[265] = (img)(_n1##x,y,z,v)), \
 (I[288] = (img)(_n1##x,_n1##y,z,v)), \
 (I[311] = (img)(_n1##x,_n2##y,z,v)), \
 (I[334] = (img)(_n1##x,_n3##y,z,v)), \
 (I[357] = (img)(_n1##x,_n4##y,z,v)), \
 (I[380] = (img)(_n1##x,_n5##y,z,v)), \
 (I[403] = (img)(_n1##x,_n6##y,z,v)), \
 (I[426] = (img)(_n1##x,_n7##y,z,v)), \
 (I[449] = (img)(_n1##x,_n8##y,z,v)), \
 (I[472] = (img)(_n1##x,_n9##y,z,v)), \
 (I[495] = (img)(_n1##x,_n10##y,z,v)), \
 (I[518] = (img)(_n1##x,_n11##y,z,v)), \
 (I[13] = (img)(_n2##x,_p11##y,z,v)), \
 (I[36] = (img)(_n2##x,_p10##y,z,v)), \
 (I[59] = (img)(_n2##x,_p9##y,z,v)), \
 (I[82] = (img)(_n2##x,_p8##y,z,v)), \
 (I[105] = (img)(_n2##x,_p7##y,z,v)), \
 (I[128] = (img)(_n2##x,_p6##y,z,v)), \
 (I[151] = (img)(_n2##x,_p5##y,z,v)), \
 (I[174] = (img)(_n2##x,_p4##y,z,v)), \
 (I[197] = (img)(_n2##x,_p3##y,z,v)), \
 (I[220] = (img)(_n2##x,_p2##y,z,v)), \
 (I[243] = (img)(_n2##x,_p1##y,z,v)), \
 (I[266] = (img)(_n2##x,y,z,v)), \
 (I[289] = (img)(_n2##x,_n1##y,z,v)), \
 (I[312] = (img)(_n2##x,_n2##y,z,v)), \
 (I[335] = (img)(_n2##x,_n3##y,z,v)), \
 (I[358] = (img)(_n2##x,_n4##y,z,v)), \
 (I[381] = (img)(_n2##x,_n5##y,z,v)), \
 (I[404] = (img)(_n2##x,_n6##y,z,v)), \
 (I[427] = (img)(_n2##x,_n7##y,z,v)), \
 (I[450] = (img)(_n2##x,_n8##y,z,v)), \
 (I[473] = (img)(_n2##x,_n9##y,z,v)), \
 (I[496] = (img)(_n2##x,_n10##y,z,v)), \
 (I[519] = (img)(_n2##x,_n11##y,z,v)), \
 (I[14] = (img)(_n3##x,_p11##y,z,v)), \
 (I[37] = (img)(_n3##x,_p10##y,z,v)), \
 (I[60] = (img)(_n3##x,_p9##y,z,v)), \
 (I[83] = (img)(_n3##x,_p8##y,z,v)), \
 (I[106] = (img)(_n3##x,_p7##y,z,v)), \
 (I[129] = (img)(_n3##x,_p6##y,z,v)), \
 (I[152] = (img)(_n3##x,_p5##y,z,v)), \
 (I[175] = (img)(_n3##x,_p4##y,z,v)), \
 (I[198] = (img)(_n3##x,_p3##y,z,v)), \
 (I[221] = (img)(_n3##x,_p2##y,z,v)), \
 (I[244] = (img)(_n3##x,_p1##y,z,v)), \
 (I[267] = (img)(_n3##x,y,z,v)), \
 (I[290] = (img)(_n3##x,_n1##y,z,v)), \
 (I[313] = (img)(_n3##x,_n2##y,z,v)), \
 (I[336] = (img)(_n3##x,_n3##y,z,v)), \
 (I[359] = (img)(_n3##x,_n4##y,z,v)), \
 (I[382] = (img)(_n3##x,_n5##y,z,v)), \
 (I[405] = (img)(_n3##x,_n6##y,z,v)), \
 (I[428] = (img)(_n3##x,_n7##y,z,v)), \
 (I[451] = (img)(_n3##x,_n8##y,z,v)), \
 (I[474] = (img)(_n3##x,_n9##y,z,v)), \
 (I[497] = (img)(_n3##x,_n10##y,z,v)), \
 (I[520] = (img)(_n3##x,_n11##y,z,v)), \
 (I[15] = (img)(_n4##x,_p11##y,z,v)), \
 (I[38] = (img)(_n4##x,_p10##y,z,v)), \
 (I[61] = (img)(_n4##x,_p9##y,z,v)), \
 (I[84] = (img)(_n4##x,_p8##y,z,v)), \
 (I[107] = (img)(_n4##x,_p7##y,z,v)), \
 (I[130] = (img)(_n4##x,_p6##y,z,v)), \
 (I[153] = (img)(_n4##x,_p5##y,z,v)), \
 (I[176] = (img)(_n4##x,_p4##y,z,v)), \
 (I[199] = (img)(_n4##x,_p3##y,z,v)), \
 (I[222] = (img)(_n4##x,_p2##y,z,v)), \
 (I[245] = (img)(_n4##x,_p1##y,z,v)), \
 (I[268] = (img)(_n4##x,y,z,v)), \
 (I[291] = (img)(_n4##x,_n1##y,z,v)), \
 (I[314] = (img)(_n4##x,_n2##y,z,v)), \
 (I[337] = (img)(_n4##x,_n3##y,z,v)), \
 (I[360] = (img)(_n4##x,_n4##y,z,v)), \
 (I[383] = (img)(_n4##x,_n5##y,z,v)), \
 (I[406] = (img)(_n4##x,_n6##y,z,v)), \
 (I[429] = (img)(_n4##x,_n7##y,z,v)), \
 (I[452] = (img)(_n4##x,_n8##y,z,v)), \
 (I[475] = (img)(_n4##x,_n9##y,z,v)), \
 (I[498] = (img)(_n4##x,_n10##y,z,v)), \
 (I[521] = (img)(_n4##x,_n11##y,z,v)), \
 (I[16] = (img)(_n5##x,_p11##y,z,v)), \
 (I[39] = (img)(_n5##x,_p10##y,z,v)), \
 (I[62] = (img)(_n5##x,_p9##y,z,v)), \
 (I[85] = (img)(_n5##x,_p8##y,z,v)), \
 (I[108] = (img)(_n5##x,_p7##y,z,v)), \
 (I[131] = (img)(_n5##x,_p6##y,z,v)), \
 (I[154] = (img)(_n5##x,_p5##y,z,v)), \
 (I[177] = (img)(_n5##x,_p4##y,z,v)), \
 (I[200] = (img)(_n5##x,_p3##y,z,v)), \
 (I[223] = (img)(_n5##x,_p2##y,z,v)), \
 (I[246] = (img)(_n5##x,_p1##y,z,v)), \
 (I[269] = (img)(_n5##x,y,z,v)), \
 (I[292] = (img)(_n5##x,_n1##y,z,v)), \
 (I[315] = (img)(_n5##x,_n2##y,z,v)), \
 (I[338] = (img)(_n5##x,_n3##y,z,v)), \
 (I[361] = (img)(_n5##x,_n4##y,z,v)), \
 (I[384] = (img)(_n5##x,_n5##y,z,v)), \
 (I[407] = (img)(_n5##x,_n6##y,z,v)), \
 (I[430] = (img)(_n5##x,_n7##y,z,v)), \
 (I[453] = (img)(_n5##x,_n8##y,z,v)), \
 (I[476] = (img)(_n5##x,_n9##y,z,v)), \
 (I[499] = (img)(_n5##x,_n10##y,z,v)), \
 (I[522] = (img)(_n5##x,_n11##y,z,v)), \
 (I[17] = (img)(_n6##x,_p11##y,z,v)), \
 (I[40] = (img)(_n6##x,_p10##y,z,v)), \
 (I[63] = (img)(_n6##x,_p9##y,z,v)), \
 (I[86] = (img)(_n6##x,_p8##y,z,v)), \
 (I[109] = (img)(_n6##x,_p7##y,z,v)), \
 (I[132] = (img)(_n6##x,_p6##y,z,v)), \
 (I[155] = (img)(_n6##x,_p5##y,z,v)), \
 (I[178] = (img)(_n6##x,_p4##y,z,v)), \
 (I[201] = (img)(_n6##x,_p3##y,z,v)), \
 (I[224] = (img)(_n6##x,_p2##y,z,v)), \
 (I[247] = (img)(_n6##x,_p1##y,z,v)), \
 (I[270] = (img)(_n6##x,y,z,v)), \
 (I[293] = (img)(_n6##x,_n1##y,z,v)), \
 (I[316] = (img)(_n6##x,_n2##y,z,v)), \
 (I[339] = (img)(_n6##x,_n3##y,z,v)), \
 (I[362] = (img)(_n6##x,_n4##y,z,v)), \
 (I[385] = (img)(_n6##x,_n5##y,z,v)), \
 (I[408] = (img)(_n6##x,_n6##y,z,v)), \
 (I[431] = (img)(_n6##x,_n7##y,z,v)), \
 (I[454] = (img)(_n6##x,_n8##y,z,v)), \
 (I[477] = (img)(_n6##x,_n9##y,z,v)), \
 (I[500] = (img)(_n6##x,_n10##y,z,v)), \
 (I[523] = (img)(_n6##x,_n11##y,z,v)), \
 (I[18] = (img)(_n7##x,_p11##y,z,v)), \
 (I[41] = (img)(_n7##x,_p10##y,z,v)), \
 (I[64] = (img)(_n7##x,_p9##y,z,v)), \
 (I[87] = (img)(_n7##x,_p8##y,z,v)), \
 (I[110] = (img)(_n7##x,_p7##y,z,v)), \
 (I[133] = (img)(_n7##x,_p6##y,z,v)), \
 (I[156] = (img)(_n7##x,_p5##y,z,v)), \
 (I[179] = (img)(_n7##x,_p4##y,z,v)), \
 (I[202] = (img)(_n7##x,_p3##y,z,v)), \
 (I[225] = (img)(_n7##x,_p2##y,z,v)), \
 (I[248] = (img)(_n7##x,_p1##y,z,v)), \
 (I[271] = (img)(_n7##x,y,z,v)), \
 (I[294] = (img)(_n7##x,_n1##y,z,v)), \
 (I[317] = (img)(_n7##x,_n2##y,z,v)), \
 (I[340] = (img)(_n7##x,_n3##y,z,v)), \
 (I[363] = (img)(_n7##x,_n4##y,z,v)), \
 (I[386] = (img)(_n7##x,_n5##y,z,v)), \
 (I[409] = (img)(_n7##x,_n6##y,z,v)), \
 (I[432] = (img)(_n7##x,_n7##y,z,v)), \
 (I[455] = (img)(_n7##x,_n8##y,z,v)), \
 (I[478] = (img)(_n7##x,_n9##y,z,v)), \
 (I[501] = (img)(_n7##x,_n10##y,z,v)), \
 (I[524] = (img)(_n7##x,_n11##y,z,v)), \
 (I[19] = (img)(_n8##x,_p11##y,z,v)), \
 (I[42] = (img)(_n8##x,_p10##y,z,v)), \
 (I[65] = (img)(_n8##x,_p9##y,z,v)), \
 (I[88] = (img)(_n8##x,_p8##y,z,v)), \
 (I[111] = (img)(_n8##x,_p7##y,z,v)), \
 (I[134] = (img)(_n8##x,_p6##y,z,v)), \
 (I[157] = (img)(_n8##x,_p5##y,z,v)), \
 (I[180] = (img)(_n8##x,_p4##y,z,v)), \
 (I[203] = (img)(_n8##x,_p3##y,z,v)), \
 (I[226] = (img)(_n8##x,_p2##y,z,v)), \
 (I[249] = (img)(_n8##x,_p1##y,z,v)), \
 (I[272] = (img)(_n8##x,y,z,v)), \
 (I[295] = (img)(_n8##x,_n1##y,z,v)), \
 (I[318] = (img)(_n8##x,_n2##y,z,v)), \
 (I[341] = (img)(_n8##x,_n3##y,z,v)), \
 (I[364] = (img)(_n8##x,_n4##y,z,v)), \
 (I[387] = (img)(_n8##x,_n5##y,z,v)), \
 (I[410] = (img)(_n8##x,_n6##y,z,v)), \
 (I[433] = (img)(_n8##x,_n7##y,z,v)), \
 (I[456] = (img)(_n8##x,_n8##y,z,v)), \
 (I[479] = (img)(_n8##x,_n9##y,z,v)), \
 (I[502] = (img)(_n8##x,_n10##y,z,v)), \
 (I[525] = (img)(_n8##x,_n11##y,z,v)), \
 (I[20] = (img)(_n9##x,_p11##y,z,v)), \
 (I[43] = (img)(_n9##x,_p10##y,z,v)), \
 (I[66] = (img)(_n9##x,_p9##y,z,v)), \
 (I[89] = (img)(_n9##x,_p8##y,z,v)), \
 (I[112] = (img)(_n9##x,_p7##y,z,v)), \
 (I[135] = (img)(_n9##x,_p6##y,z,v)), \
 (I[158] = (img)(_n9##x,_p5##y,z,v)), \
 (I[181] = (img)(_n9##x,_p4##y,z,v)), \
 (I[204] = (img)(_n9##x,_p3##y,z,v)), \
 (I[227] = (img)(_n9##x,_p2##y,z,v)), \
 (I[250] = (img)(_n9##x,_p1##y,z,v)), \
 (I[273] = (img)(_n9##x,y,z,v)), \
 (I[296] = (img)(_n9##x,_n1##y,z,v)), \
 (I[319] = (img)(_n9##x,_n2##y,z,v)), \
 (I[342] = (img)(_n9##x,_n3##y,z,v)), \
 (I[365] = (img)(_n9##x,_n4##y,z,v)), \
 (I[388] = (img)(_n9##x,_n5##y,z,v)), \
 (I[411] = (img)(_n9##x,_n6##y,z,v)), \
 (I[434] = (img)(_n9##x,_n7##y,z,v)), \
 (I[457] = (img)(_n9##x,_n8##y,z,v)), \
 (I[480] = (img)(_n9##x,_n9##y,z,v)), \
 (I[503] = (img)(_n9##x,_n10##y,z,v)), \
 (I[526] = (img)(_n9##x,_n11##y,z,v)), \
 (I[21] = (img)(_n10##x,_p11##y,z,v)), \
 (I[44] = (img)(_n10##x,_p10##y,z,v)), \
 (I[67] = (img)(_n10##x,_p9##y,z,v)), \
 (I[90] = (img)(_n10##x,_p8##y,z,v)), \
 (I[113] = (img)(_n10##x,_p7##y,z,v)), \
 (I[136] = (img)(_n10##x,_p6##y,z,v)), \
 (I[159] = (img)(_n10##x,_p5##y,z,v)), \
 (I[182] = (img)(_n10##x,_p4##y,z,v)), \
 (I[205] = (img)(_n10##x,_p3##y,z,v)), \
 (I[228] = (img)(_n10##x,_p2##y,z,v)), \
 (I[251] = (img)(_n10##x,_p1##y,z,v)), \
 (I[274] = (img)(_n10##x,y,z,v)), \
 (I[297] = (img)(_n10##x,_n1##y,z,v)), \
 (I[320] = (img)(_n10##x,_n2##y,z,v)), \
 (I[343] = (img)(_n10##x,_n3##y,z,v)), \
 (I[366] = (img)(_n10##x,_n4##y,z,v)), \
 (I[389] = (img)(_n10##x,_n5##y,z,v)), \
 (I[412] = (img)(_n10##x,_n6##y,z,v)), \
 (I[435] = (img)(_n10##x,_n7##y,z,v)), \
 (I[458] = (img)(_n10##x,_n8##y,z,v)), \
 (I[481] = (img)(_n10##x,_n9##y,z,v)), \
 (I[504] = (img)(_n10##x,_n10##y,z,v)), \
 (I[527] = (img)(_n10##x,_n11##y,z,v)), \
 x+11>=(int)((img).width)?(int)((img).width)-1:x+11); \
 x<=(int)(x1) && ((_n11##x<(int)((img).width) && ( \
 (I[22] = (img)(_n11##x,_p11##y,z,v)), \
 (I[45] = (img)(_n11##x,_p10##y,z,v)), \
 (I[68] = (img)(_n11##x,_p9##y,z,v)), \
 (I[91] = (img)(_n11##x,_p8##y,z,v)), \
 (I[114] = (img)(_n11##x,_p7##y,z,v)), \
 (I[137] = (img)(_n11##x,_p6##y,z,v)), \
 (I[160] = (img)(_n11##x,_p5##y,z,v)), \
 (I[183] = (img)(_n11##x,_p4##y,z,v)), \
 (I[206] = (img)(_n11##x,_p3##y,z,v)), \
 (I[229] = (img)(_n11##x,_p2##y,z,v)), \
 (I[252] = (img)(_n11##x,_p1##y,z,v)), \
 (I[275] = (img)(_n11##x,y,z,v)), \
 (I[298] = (img)(_n11##x,_n1##y,z,v)), \
 (I[321] = (img)(_n11##x,_n2##y,z,v)), \
 (I[344] = (img)(_n11##x,_n3##y,z,v)), \
 (I[367] = (img)(_n11##x,_n4##y,z,v)), \
 (I[390] = (img)(_n11##x,_n5##y,z,v)), \
 (I[413] = (img)(_n11##x,_n6##y,z,v)), \
 (I[436] = (img)(_n11##x,_n7##y,z,v)), \
 (I[459] = (img)(_n11##x,_n8##y,z,v)), \
 (I[482] = (img)(_n11##x,_n9##y,z,v)), \
 (I[505] = (img)(_n11##x,_n10##y,z,v)), \
 (I[528] = (img)(_n11##x,_n11##y,z,v)),1)) || \
 _n10##x==--_n11##x || _n9##x==--_n10##x || _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n11##x = _n10##x = _n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], \
 I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], \
 I[46] = I[47], I[47] = I[48], I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], \
 I[69] = I[70], I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], \
 I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], \
 I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], \
 I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], \
 I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], I[167] = I[168], I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], \
 I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], \
 I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], \
 I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], \
 I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], \
 I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], \
 I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], \
 I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], I[335] = I[336], I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], I[342] = I[343], I[343] = I[344], \
 I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], I[351] = I[352], I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], I[359] = I[360], I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], \
 I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], I[373] = I[374], I[374] = I[375], I[375] = I[376], I[376] = I[377], I[377] = I[378], I[378] = I[379], I[379] = I[380], I[380] = I[381], I[381] = I[382], I[382] = I[383], I[383] = I[384], I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], \
 I[391] = I[392], I[392] = I[393], I[393] = I[394], I[394] = I[395], I[395] = I[396], I[396] = I[397], I[397] = I[398], I[398] = I[399], I[399] = I[400], I[400] = I[401], I[401] = I[402], I[402] = I[403], I[403] = I[404], I[404] = I[405], I[405] = I[406], I[406] = I[407], I[407] = I[408], I[408] = I[409], I[409] = I[410], I[410] = I[411], I[411] = I[412], I[412] = I[413], \
 I[414] = I[415], I[415] = I[416], I[416] = I[417], I[417] = I[418], I[418] = I[419], I[419] = I[420], I[420] = I[421], I[421] = I[422], I[422] = I[423], I[423] = I[424], I[424] = I[425], I[425] = I[426], I[426] = I[427], I[427] = I[428], I[428] = I[429], I[429] = I[430], I[430] = I[431], I[431] = I[432], I[432] = I[433], I[433] = I[434], I[434] = I[435], I[435] = I[436], \
 I[437] = I[438], I[438] = I[439], I[439] = I[440], I[440] = I[441], I[441] = I[442], I[442] = I[443], I[443] = I[444], I[444] = I[445], I[445] = I[446], I[446] = I[447], I[447] = I[448], I[448] = I[449], I[449] = I[450], I[450] = I[451], I[451] = I[452], I[452] = I[453], I[453] = I[454], I[454] = I[455], I[455] = I[456], I[456] = I[457], I[457] = I[458], I[458] = I[459], \
 I[460] = I[461], I[461] = I[462], I[462] = I[463], I[463] = I[464], I[464] = I[465], I[465] = I[466], I[466] = I[467], I[467] = I[468], I[468] = I[469], I[469] = I[470], I[470] = I[471], I[471] = I[472], I[472] = I[473], I[473] = I[474], I[474] = I[475], I[475] = I[476], I[476] = I[477], I[477] = I[478], I[478] = I[479], I[479] = I[480], I[480] = I[481], I[481] = I[482], \
 I[483] = I[484], I[484] = I[485], I[485] = I[486], I[486] = I[487], I[487] = I[488], I[488] = I[489], I[489] = I[490], I[490] = I[491], I[491] = I[492], I[492] = I[493], I[493] = I[494], I[494] = I[495], I[495] = I[496], I[496] = I[497], I[497] = I[498], I[498] = I[499], I[499] = I[500], I[500] = I[501], I[501] = I[502], I[502] = I[503], I[503] = I[504], I[504] = I[505], \
 I[506] = I[507], I[507] = I[508], I[508] = I[509], I[509] = I[510], I[510] = I[511], I[511] = I[512], I[512] = I[513], I[513] = I[514], I[514] = I[515], I[515] = I[516], I[516] = I[517], I[517] = I[518], I[518] = I[519], I[519] = I[520], I[520] = I[521], I[521] = I[522], I[522] = I[523], I[523] = I[524], I[524] = I[525], I[525] = I[526], I[526] = I[527], I[527] = I[528], \
 _p11##x = _p10##x, _p10##x = _p9##x, _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x, ++_n10##x, ++_n11##x)

#define cimg_get23x23(img,x,y,z,v,I) \
 I[0] = (img)(_p11##x,_p11##y,z,v), I[1] = (img)(_p10##x,_p11##y,z,v), I[2] = (img)(_p9##x,_p11##y,z,v), I[3] = (img)(_p8##x,_p11##y,z,v), I[4] = (img)(_p7##x,_p11##y,z,v), I[5] = (img)(_p6##x,_p11##y,z,v), I[6] = (img)(_p5##x,_p11##y,z,v), I[7] = (img)(_p4##x,_p11##y,z,v), I[8] = (img)(_p3##x,_p11##y,z,v), I[9] = (img)(_p2##x,_p11##y,z,v), I[10] = (img)(_p1##x,_p11##y,z,v), I[11] = (img)(x,_p11##y,z,v), I[12] = (img)(_n1##x,_p11##y,z,v), I[13] = (img)(_n2##x,_p11##y,z,v), I[14] = (img)(_n3##x,_p11##y,z,v), I[15] = (img)(_n4##x,_p11##y,z,v), I[16] = (img)(_n5##x,_p11##y,z,v), I[17] = (img)(_n6##x,_p11##y,z,v), I[18] = (img)(_n7##x,_p11##y,z,v), I[19] = (img)(_n8##x,_p11##y,z,v), I[20] = (img)(_n9##x,_p11##y,z,v), I[21] = (img)(_n10##x,_p11##y,z,v), I[22] = (img)(_n11##x,_p11##y,z,v), \
 I[23] = (img)(_p11##x,_p10##y,z,v), I[24] = (img)(_p10##x,_p10##y,z,v), I[25] = (img)(_p9##x,_p10##y,z,v), I[26] = (img)(_p8##x,_p10##y,z,v), I[27] = (img)(_p7##x,_p10##y,z,v), I[28] = (img)(_p6##x,_p10##y,z,v), I[29] = (img)(_p5##x,_p10##y,z,v), I[30] = (img)(_p4##x,_p10##y,z,v), I[31] = (img)(_p3##x,_p10##y,z,v), I[32] = (img)(_p2##x,_p10##y,z,v), I[33] = (img)(_p1##x,_p10##y,z,v), I[34] = (img)(x,_p10##y,z,v), I[35] = (img)(_n1##x,_p10##y,z,v), I[36] = (img)(_n2##x,_p10##y,z,v), I[37] = (img)(_n3##x,_p10##y,z,v), I[38] = (img)(_n4##x,_p10##y,z,v), I[39] = (img)(_n5##x,_p10##y,z,v), I[40] = (img)(_n6##x,_p10##y,z,v), I[41] = (img)(_n7##x,_p10##y,z,v), I[42] = (img)(_n8##x,_p10##y,z,v), I[43] = (img)(_n9##x,_p10##y,z,v), I[44] = (img)(_n10##x,_p10##y,z,v), I[45] = (img)(_n11##x,_p10##y,z,v), \
 I[46] = (img)(_p11##x,_p9##y,z,v), I[47] = (img)(_p10##x,_p9##y,z,v), I[48] = (img)(_p9##x,_p9##y,z,v), I[49] = (img)(_p8##x,_p9##y,z,v), I[50] = (img)(_p7##x,_p9##y,z,v), I[51] = (img)(_p6##x,_p9##y,z,v), I[52] = (img)(_p5##x,_p9##y,z,v), I[53] = (img)(_p4##x,_p9##y,z,v), I[54] = (img)(_p3##x,_p9##y,z,v), I[55] = (img)(_p2##x,_p9##y,z,v), I[56] = (img)(_p1##x,_p9##y,z,v), I[57] = (img)(x,_p9##y,z,v), I[58] = (img)(_n1##x,_p9##y,z,v), I[59] = (img)(_n2##x,_p9##y,z,v), I[60] = (img)(_n3##x,_p9##y,z,v), I[61] = (img)(_n4##x,_p9##y,z,v), I[62] = (img)(_n5##x,_p9##y,z,v), I[63] = (img)(_n6##x,_p9##y,z,v), I[64] = (img)(_n7##x,_p9##y,z,v), I[65] = (img)(_n8##x,_p9##y,z,v), I[66] = (img)(_n9##x,_p9##y,z,v), I[67] = (img)(_n10##x,_p9##y,z,v), I[68] = (img)(_n11##x,_p9##y,z,v), \
 I[69] = (img)(_p11##x,_p8##y,z,v), I[70] = (img)(_p10##x,_p8##y,z,v), I[71] = (img)(_p9##x,_p8##y,z,v), I[72] = (img)(_p8##x,_p8##y,z,v), I[73] = (img)(_p7##x,_p8##y,z,v), I[74] = (img)(_p6##x,_p8##y,z,v), I[75] = (img)(_p5##x,_p8##y,z,v), I[76] = (img)(_p4##x,_p8##y,z,v), I[77] = (img)(_p3##x,_p8##y,z,v), I[78] = (img)(_p2##x,_p8##y,z,v), I[79] = (img)(_p1##x,_p8##y,z,v), I[80] = (img)(x,_p8##y,z,v), I[81] = (img)(_n1##x,_p8##y,z,v), I[82] = (img)(_n2##x,_p8##y,z,v), I[83] = (img)(_n3##x,_p8##y,z,v), I[84] = (img)(_n4##x,_p8##y,z,v), I[85] = (img)(_n5##x,_p8##y,z,v), I[86] = (img)(_n6##x,_p8##y,z,v), I[87] = (img)(_n7##x,_p8##y,z,v), I[88] = (img)(_n8##x,_p8##y,z,v), I[89] = (img)(_n9##x,_p8##y,z,v), I[90] = (img)(_n10##x,_p8##y,z,v), I[91] = (img)(_n11##x,_p8##y,z,v), \
 I[92] = (img)(_p11##x,_p7##y,z,v), I[93] = (img)(_p10##x,_p7##y,z,v), I[94] = (img)(_p9##x,_p7##y,z,v), I[95] = (img)(_p8##x,_p7##y,z,v), I[96] = (img)(_p7##x,_p7##y,z,v), I[97] = (img)(_p6##x,_p7##y,z,v), I[98] = (img)(_p5##x,_p7##y,z,v), I[99] = (img)(_p4##x,_p7##y,z,v), I[100] = (img)(_p3##x,_p7##y,z,v), I[101] = (img)(_p2##x,_p7##y,z,v), I[102] = (img)(_p1##x,_p7##y,z,v), I[103] = (img)(x,_p7##y,z,v), I[104] = (img)(_n1##x,_p7##y,z,v), I[105] = (img)(_n2##x,_p7##y,z,v), I[106] = (img)(_n3##x,_p7##y,z,v), I[107] = (img)(_n4##x,_p7##y,z,v), I[108] = (img)(_n5##x,_p7##y,z,v), I[109] = (img)(_n6##x,_p7##y,z,v), I[110] = (img)(_n7##x,_p7##y,z,v), I[111] = (img)(_n8##x,_p7##y,z,v), I[112] = (img)(_n9##x,_p7##y,z,v), I[113] = (img)(_n10##x,_p7##y,z,v), I[114] = (img)(_n11##x,_p7##y,z,v), \
 I[115] = (img)(_p11##x,_p6##y,z,v), I[116] = (img)(_p10##x,_p6##y,z,v), I[117] = (img)(_p9##x,_p6##y,z,v), I[118] = (img)(_p8##x,_p6##y,z,v), I[119] = (img)(_p7##x,_p6##y,z,v), I[120] = (img)(_p6##x,_p6##y,z,v), I[121] = (img)(_p5##x,_p6##y,z,v), I[122] = (img)(_p4##x,_p6##y,z,v), I[123] = (img)(_p3##x,_p6##y,z,v), I[124] = (img)(_p2##x,_p6##y,z,v), I[125] = (img)(_p1##x,_p6##y,z,v), I[126] = (img)(x,_p6##y,z,v), I[127] = (img)(_n1##x,_p6##y,z,v), I[128] = (img)(_n2##x,_p6##y,z,v), I[129] = (img)(_n3##x,_p6##y,z,v), I[130] = (img)(_n4##x,_p6##y,z,v), I[131] = (img)(_n5##x,_p6##y,z,v), I[132] = (img)(_n6##x,_p6##y,z,v), I[133] = (img)(_n7##x,_p6##y,z,v), I[134] = (img)(_n8##x,_p6##y,z,v), I[135] = (img)(_n9##x,_p6##y,z,v), I[136] = (img)(_n10##x,_p6##y,z,v), I[137] = (img)(_n11##x,_p6##y,z,v), \
 I[138] = (img)(_p11##x,_p5##y,z,v), I[139] = (img)(_p10##x,_p5##y,z,v), I[140] = (img)(_p9##x,_p5##y,z,v), I[141] = (img)(_p8##x,_p5##y,z,v), I[142] = (img)(_p7##x,_p5##y,z,v), I[143] = (img)(_p6##x,_p5##y,z,v), I[144] = (img)(_p5##x,_p5##y,z,v), I[145] = (img)(_p4##x,_p5##y,z,v), I[146] = (img)(_p3##x,_p5##y,z,v), I[147] = (img)(_p2##x,_p5##y,z,v), I[148] = (img)(_p1##x,_p5##y,z,v), I[149] = (img)(x,_p5##y,z,v), I[150] = (img)(_n1##x,_p5##y,z,v), I[151] = (img)(_n2##x,_p5##y,z,v), I[152] = (img)(_n3##x,_p5##y,z,v), I[153] = (img)(_n4##x,_p5##y,z,v), I[154] = (img)(_n5##x,_p5##y,z,v), I[155] = (img)(_n6##x,_p5##y,z,v), I[156] = (img)(_n7##x,_p5##y,z,v), I[157] = (img)(_n8##x,_p5##y,z,v), I[158] = (img)(_n9##x,_p5##y,z,v), I[159] = (img)(_n10##x,_p5##y,z,v), I[160] = (img)(_n11##x,_p5##y,z,v), \
 I[161] = (img)(_p11##x,_p4##y,z,v), I[162] = (img)(_p10##x,_p4##y,z,v), I[163] = (img)(_p9##x,_p4##y,z,v), I[164] = (img)(_p8##x,_p4##y,z,v), I[165] = (img)(_p7##x,_p4##y,z,v), I[166] = (img)(_p6##x,_p4##y,z,v), I[167] = (img)(_p5##x,_p4##y,z,v), I[168] = (img)(_p4##x,_p4##y,z,v), I[169] = (img)(_p3##x,_p4##y,z,v), I[170] = (img)(_p2##x,_p4##y,z,v), I[171] = (img)(_p1##x,_p4##y,z,v), I[172] = (img)(x,_p4##y,z,v), I[173] = (img)(_n1##x,_p4##y,z,v), I[174] = (img)(_n2##x,_p4##y,z,v), I[175] = (img)(_n3##x,_p4##y,z,v), I[176] = (img)(_n4##x,_p4##y,z,v), I[177] = (img)(_n5##x,_p4##y,z,v), I[178] = (img)(_n6##x,_p4##y,z,v), I[179] = (img)(_n7##x,_p4##y,z,v), I[180] = (img)(_n8##x,_p4##y,z,v), I[181] = (img)(_n9##x,_p4##y,z,v), I[182] = (img)(_n10##x,_p4##y,z,v), I[183] = (img)(_n11##x,_p4##y,z,v), \
 I[184] = (img)(_p11##x,_p3##y,z,v), I[185] = (img)(_p10##x,_p3##y,z,v), I[186] = (img)(_p9##x,_p3##y,z,v), I[187] = (img)(_p8##x,_p3##y,z,v), I[188] = (img)(_p7##x,_p3##y,z,v), I[189] = (img)(_p6##x,_p3##y,z,v), I[190] = (img)(_p5##x,_p3##y,z,v), I[191] = (img)(_p4##x,_p3##y,z,v), I[192] = (img)(_p3##x,_p3##y,z,v), I[193] = (img)(_p2##x,_p3##y,z,v), I[194] = (img)(_p1##x,_p3##y,z,v), I[195] = (img)(x,_p3##y,z,v), I[196] = (img)(_n1##x,_p3##y,z,v), I[197] = (img)(_n2##x,_p3##y,z,v), I[198] = (img)(_n3##x,_p3##y,z,v), I[199] = (img)(_n4##x,_p3##y,z,v), I[200] = (img)(_n5##x,_p3##y,z,v), I[201] = (img)(_n6##x,_p3##y,z,v), I[202] = (img)(_n7##x,_p3##y,z,v), I[203] = (img)(_n8##x,_p3##y,z,v), I[204] = (img)(_n9##x,_p3##y,z,v), I[205] = (img)(_n10##x,_p3##y,z,v), I[206] = (img)(_n11##x,_p3##y,z,v), \
 I[207] = (img)(_p11##x,_p2##y,z,v), I[208] = (img)(_p10##x,_p2##y,z,v), I[209] = (img)(_p9##x,_p2##y,z,v), I[210] = (img)(_p8##x,_p2##y,z,v), I[211] = (img)(_p7##x,_p2##y,z,v), I[212] = (img)(_p6##x,_p2##y,z,v), I[213] = (img)(_p5##x,_p2##y,z,v), I[214] = (img)(_p4##x,_p2##y,z,v), I[215] = (img)(_p3##x,_p2##y,z,v), I[216] = (img)(_p2##x,_p2##y,z,v), I[217] = (img)(_p1##x,_p2##y,z,v), I[218] = (img)(x,_p2##y,z,v), I[219] = (img)(_n1##x,_p2##y,z,v), I[220] = (img)(_n2##x,_p2##y,z,v), I[221] = (img)(_n3##x,_p2##y,z,v), I[222] = (img)(_n4##x,_p2##y,z,v), I[223] = (img)(_n5##x,_p2##y,z,v), I[224] = (img)(_n6##x,_p2##y,z,v), I[225] = (img)(_n7##x,_p2##y,z,v), I[226] = (img)(_n8##x,_p2##y,z,v), I[227] = (img)(_n9##x,_p2##y,z,v), I[228] = (img)(_n10##x,_p2##y,z,v), I[229] = (img)(_n11##x,_p2##y,z,v), \
 I[230] = (img)(_p11##x,_p1##y,z,v), I[231] = (img)(_p10##x,_p1##y,z,v), I[232] = (img)(_p9##x,_p1##y,z,v), I[233] = (img)(_p8##x,_p1##y,z,v), I[234] = (img)(_p7##x,_p1##y,z,v), I[235] = (img)(_p6##x,_p1##y,z,v), I[236] = (img)(_p5##x,_p1##y,z,v), I[237] = (img)(_p4##x,_p1##y,z,v), I[238] = (img)(_p3##x,_p1##y,z,v), I[239] = (img)(_p2##x,_p1##y,z,v), I[240] = (img)(_p1##x,_p1##y,z,v), I[241] = (img)(x,_p1##y,z,v), I[242] = (img)(_n1##x,_p1##y,z,v), I[243] = (img)(_n2##x,_p1##y,z,v), I[244] = (img)(_n3##x,_p1##y,z,v), I[245] = (img)(_n4##x,_p1##y,z,v), I[246] = (img)(_n5##x,_p1##y,z,v), I[247] = (img)(_n6##x,_p1##y,z,v), I[248] = (img)(_n7##x,_p1##y,z,v), I[249] = (img)(_n8##x,_p1##y,z,v), I[250] = (img)(_n9##x,_p1##y,z,v), I[251] = (img)(_n10##x,_p1##y,z,v), I[252] = (img)(_n11##x,_p1##y,z,v), \
 I[253] = (img)(_p11##x,y,z,v), I[254] = (img)(_p10##x,y,z,v), I[255] = (img)(_p9##x,y,z,v), I[256] = (img)(_p8##x,y,z,v), I[257] = (img)(_p7##x,y,z,v), I[258] = (img)(_p6##x,y,z,v), I[259] = (img)(_p5##x,y,z,v), I[260] = (img)(_p4##x,y,z,v), I[261] = (img)(_p3##x,y,z,v), I[262] = (img)(_p2##x,y,z,v), I[263] = (img)(_p1##x,y,z,v), I[264] = (img)(x,y,z,v), I[265] = (img)(_n1##x,y,z,v), I[266] = (img)(_n2##x,y,z,v), I[267] = (img)(_n3##x,y,z,v), I[268] = (img)(_n4##x,y,z,v), I[269] = (img)(_n5##x,y,z,v), I[270] = (img)(_n6##x,y,z,v), I[271] = (img)(_n7##x,y,z,v), I[272] = (img)(_n8##x,y,z,v), I[273] = (img)(_n9##x,y,z,v), I[274] = (img)(_n10##x,y,z,v), I[275] = (img)(_n11##x,y,z,v), \
 I[276] = (img)(_p11##x,_n1##y,z,v), I[277] = (img)(_p10##x,_n1##y,z,v), I[278] = (img)(_p9##x,_n1##y,z,v), I[279] = (img)(_p8##x,_n1##y,z,v), I[280] = (img)(_p7##x,_n1##y,z,v), I[281] = (img)(_p6##x,_n1##y,z,v), I[282] = (img)(_p5##x,_n1##y,z,v), I[283] = (img)(_p4##x,_n1##y,z,v), I[284] = (img)(_p3##x,_n1##y,z,v), I[285] = (img)(_p2##x,_n1##y,z,v), I[286] = (img)(_p1##x,_n1##y,z,v), I[287] = (img)(x,_n1##y,z,v), I[288] = (img)(_n1##x,_n1##y,z,v), I[289] = (img)(_n2##x,_n1##y,z,v), I[290] = (img)(_n3##x,_n1##y,z,v), I[291] = (img)(_n4##x,_n1##y,z,v), I[292] = (img)(_n5##x,_n1##y,z,v), I[293] = (img)(_n6##x,_n1##y,z,v), I[294] = (img)(_n7##x,_n1##y,z,v), I[295] = (img)(_n8##x,_n1##y,z,v), I[296] = (img)(_n9##x,_n1##y,z,v), I[297] = (img)(_n10##x,_n1##y,z,v), I[298] = (img)(_n11##x,_n1##y,z,v), \
 I[299] = (img)(_p11##x,_n2##y,z,v), I[300] = (img)(_p10##x,_n2##y,z,v), I[301] = (img)(_p9##x,_n2##y,z,v), I[302] = (img)(_p8##x,_n2##y,z,v), I[303] = (img)(_p7##x,_n2##y,z,v), I[304] = (img)(_p6##x,_n2##y,z,v), I[305] = (img)(_p5##x,_n2##y,z,v), I[306] = (img)(_p4##x,_n2##y,z,v), I[307] = (img)(_p3##x,_n2##y,z,v), I[308] = (img)(_p2##x,_n2##y,z,v), I[309] = (img)(_p1##x,_n2##y,z,v), I[310] = (img)(x,_n2##y,z,v), I[311] = (img)(_n1##x,_n2##y,z,v), I[312] = (img)(_n2##x,_n2##y,z,v), I[313] = (img)(_n3##x,_n2##y,z,v), I[314] = (img)(_n4##x,_n2##y,z,v), I[315] = (img)(_n5##x,_n2##y,z,v), I[316] = (img)(_n6##x,_n2##y,z,v), I[317] = (img)(_n7##x,_n2##y,z,v), I[318] = (img)(_n8##x,_n2##y,z,v), I[319] = (img)(_n9##x,_n2##y,z,v), I[320] = (img)(_n10##x,_n2##y,z,v), I[321] = (img)(_n11##x,_n2##y,z,v), \
 I[322] = (img)(_p11##x,_n3##y,z,v), I[323] = (img)(_p10##x,_n3##y,z,v), I[324] = (img)(_p9##x,_n3##y,z,v), I[325] = (img)(_p8##x,_n3##y,z,v), I[326] = (img)(_p7##x,_n3##y,z,v), I[327] = (img)(_p6##x,_n3##y,z,v), I[328] = (img)(_p5##x,_n3##y,z,v), I[329] = (img)(_p4##x,_n3##y,z,v), I[330] = (img)(_p3##x,_n3##y,z,v), I[331] = (img)(_p2##x,_n3##y,z,v), I[332] = (img)(_p1##x,_n3##y,z,v), I[333] = (img)(x,_n3##y,z,v), I[334] = (img)(_n1##x,_n3##y,z,v), I[335] = (img)(_n2##x,_n3##y,z,v), I[336] = (img)(_n3##x,_n3##y,z,v), I[337] = (img)(_n4##x,_n3##y,z,v), I[338] = (img)(_n5##x,_n3##y,z,v), I[339] = (img)(_n6##x,_n3##y,z,v), I[340] = (img)(_n7##x,_n3##y,z,v), I[341] = (img)(_n8##x,_n3##y,z,v), I[342] = (img)(_n9##x,_n3##y,z,v), I[343] = (img)(_n10##x,_n3##y,z,v), I[344] = (img)(_n11##x,_n3##y,z,v), \
 I[345] = (img)(_p11##x,_n4##y,z,v), I[346] = (img)(_p10##x,_n4##y,z,v), I[347] = (img)(_p9##x,_n4##y,z,v), I[348] = (img)(_p8##x,_n4##y,z,v), I[349] = (img)(_p7##x,_n4##y,z,v), I[350] = (img)(_p6##x,_n4##y,z,v), I[351] = (img)(_p5##x,_n4##y,z,v), I[352] = (img)(_p4##x,_n4##y,z,v), I[353] = (img)(_p3##x,_n4##y,z,v), I[354] = (img)(_p2##x,_n4##y,z,v), I[355] = (img)(_p1##x,_n4##y,z,v), I[356] = (img)(x,_n4##y,z,v), I[357] = (img)(_n1##x,_n4##y,z,v), I[358] = (img)(_n2##x,_n4##y,z,v), I[359] = (img)(_n3##x,_n4##y,z,v), I[360] = (img)(_n4##x,_n4##y,z,v), I[361] = (img)(_n5##x,_n4##y,z,v), I[362] = (img)(_n6##x,_n4##y,z,v), I[363] = (img)(_n7##x,_n4##y,z,v), I[364] = (img)(_n8##x,_n4##y,z,v), I[365] = (img)(_n9##x,_n4##y,z,v), I[366] = (img)(_n10##x,_n4##y,z,v), I[367] = (img)(_n11##x,_n4##y,z,v), \
 I[368] = (img)(_p11##x,_n5##y,z,v), I[369] = (img)(_p10##x,_n5##y,z,v), I[370] = (img)(_p9##x,_n5##y,z,v), I[371] = (img)(_p8##x,_n5##y,z,v), I[372] = (img)(_p7##x,_n5##y,z,v), I[373] = (img)(_p6##x,_n5##y,z,v), I[374] = (img)(_p5##x,_n5##y,z,v), I[375] = (img)(_p4##x,_n5##y,z,v), I[376] = (img)(_p3##x,_n5##y,z,v), I[377] = (img)(_p2##x,_n5##y,z,v), I[378] = (img)(_p1##x,_n5##y,z,v), I[379] = (img)(x,_n5##y,z,v), I[380] = (img)(_n1##x,_n5##y,z,v), I[381] = (img)(_n2##x,_n5##y,z,v), I[382] = (img)(_n3##x,_n5##y,z,v), I[383] = (img)(_n4##x,_n5##y,z,v), I[384] = (img)(_n5##x,_n5##y,z,v), I[385] = (img)(_n6##x,_n5##y,z,v), I[386] = (img)(_n7##x,_n5##y,z,v), I[387] = (img)(_n8##x,_n5##y,z,v), I[388] = (img)(_n9##x,_n5##y,z,v), I[389] = (img)(_n10##x,_n5##y,z,v), I[390] = (img)(_n11##x,_n5##y,z,v), \
 I[391] = (img)(_p11##x,_n6##y,z,v), I[392] = (img)(_p10##x,_n6##y,z,v), I[393] = (img)(_p9##x,_n6##y,z,v), I[394] = (img)(_p8##x,_n6##y,z,v), I[395] = (img)(_p7##x,_n6##y,z,v), I[396] = (img)(_p6##x,_n6##y,z,v), I[397] = (img)(_p5##x,_n6##y,z,v), I[398] = (img)(_p4##x,_n6##y,z,v), I[399] = (img)(_p3##x,_n6##y,z,v), I[400] = (img)(_p2##x,_n6##y,z,v), I[401] = (img)(_p1##x,_n6##y,z,v), I[402] = (img)(x,_n6##y,z,v), I[403] = (img)(_n1##x,_n6##y,z,v), I[404] = (img)(_n2##x,_n6##y,z,v), I[405] = (img)(_n3##x,_n6##y,z,v), I[406] = (img)(_n4##x,_n6##y,z,v), I[407] = (img)(_n5##x,_n6##y,z,v), I[408] = (img)(_n6##x,_n6##y,z,v), I[409] = (img)(_n7##x,_n6##y,z,v), I[410] = (img)(_n8##x,_n6##y,z,v), I[411] = (img)(_n9##x,_n6##y,z,v), I[412] = (img)(_n10##x,_n6##y,z,v), I[413] = (img)(_n11##x,_n6##y,z,v), \
 I[414] = (img)(_p11##x,_n7##y,z,v), I[415] = (img)(_p10##x,_n7##y,z,v), I[416] = (img)(_p9##x,_n7##y,z,v), I[417] = (img)(_p8##x,_n7##y,z,v), I[418] = (img)(_p7##x,_n7##y,z,v), I[419] = (img)(_p6##x,_n7##y,z,v), I[420] = (img)(_p5##x,_n7##y,z,v), I[421] = (img)(_p4##x,_n7##y,z,v), I[422] = (img)(_p3##x,_n7##y,z,v), I[423] = (img)(_p2##x,_n7##y,z,v), I[424] = (img)(_p1##x,_n7##y,z,v), I[425] = (img)(x,_n7##y,z,v), I[426] = (img)(_n1##x,_n7##y,z,v), I[427] = (img)(_n2##x,_n7##y,z,v), I[428] = (img)(_n3##x,_n7##y,z,v), I[429] = (img)(_n4##x,_n7##y,z,v), I[430] = (img)(_n5##x,_n7##y,z,v), I[431] = (img)(_n6##x,_n7##y,z,v), I[432] = (img)(_n7##x,_n7##y,z,v), I[433] = (img)(_n8##x,_n7##y,z,v), I[434] = (img)(_n9##x,_n7##y,z,v), I[435] = (img)(_n10##x,_n7##y,z,v), I[436] = (img)(_n11##x,_n7##y,z,v), \
 I[437] = (img)(_p11##x,_n8##y,z,v), I[438] = (img)(_p10##x,_n8##y,z,v), I[439] = (img)(_p9##x,_n8##y,z,v), I[440] = (img)(_p8##x,_n8##y,z,v), I[441] = (img)(_p7##x,_n8##y,z,v), I[442] = (img)(_p6##x,_n8##y,z,v), I[443] = (img)(_p5##x,_n8##y,z,v), I[444] = (img)(_p4##x,_n8##y,z,v), I[445] = (img)(_p3##x,_n8##y,z,v), I[446] = (img)(_p2##x,_n8##y,z,v), I[447] = (img)(_p1##x,_n8##y,z,v), I[448] = (img)(x,_n8##y,z,v), I[449] = (img)(_n1##x,_n8##y,z,v), I[450] = (img)(_n2##x,_n8##y,z,v), I[451] = (img)(_n3##x,_n8##y,z,v), I[452] = (img)(_n4##x,_n8##y,z,v), I[453] = (img)(_n5##x,_n8##y,z,v), I[454] = (img)(_n6##x,_n8##y,z,v), I[455] = (img)(_n7##x,_n8##y,z,v), I[456] = (img)(_n8##x,_n8##y,z,v), I[457] = (img)(_n9##x,_n8##y,z,v), I[458] = (img)(_n10##x,_n8##y,z,v), I[459] = (img)(_n11##x,_n8##y,z,v), \
 I[460] = (img)(_p11##x,_n9##y,z,v), I[461] = (img)(_p10##x,_n9##y,z,v), I[462] = (img)(_p9##x,_n9##y,z,v), I[463] = (img)(_p8##x,_n9##y,z,v), I[464] = (img)(_p7##x,_n9##y,z,v), I[465] = (img)(_p6##x,_n9##y,z,v), I[466] = (img)(_p5##x,_n9##y,z,v), I[467] = (img)(_p4##x,_n9##y,z,v), I[468] = (img)(_p3##x,_n9##y,z,v), I[469] = (img)(_p2##x,_n9##y,z,v), I[470] = (img)(_p1##x,_n9##y,z,v), I[471] = (img)(x,_n9##y,z,v), I[472] = (img)(_n1##x,_n9##y,z,v), I[473] = (img)(_n2##x,_n9##y,z,v), I[474] = (img)(_n3##x,_n9##y,z,v), I[475] = (img)(_n4##x,_n9##y,z,v), I[476] = (img)(_n5##x,_n9##y,z,v), I[477] = (img)(_n6##x,_n9##y,z,v), I[478] = (img)(_n7##x,_n9##y,z,v), I[479] = (img)(_n8##x,_n9##y,z,v), I[480] = (img)(_n9##x,_n9##y,z,v), I[481] = (img)(_n10##x,_n9##y,z,v), I[482] = (img)(_n11##x,_n9##y,z,v), \
 I[483] = (img)(_p11##x,_n10##y,z,v), I[484] = (img)(_p10##x,_n10##y,z,v), I[485] = (img)(_p9##x,_n10##y,z,v), I[486] = (img)(_p8##x,_n10##y,z,v), I[487] = (img)(_p7##x,_n10##y,z,v), I[488] = (img)(_p6##x,_n10##y,z,v), I[489] = (img)(_p5##x,_n10##y,z,v), I[490] = (img)(_p4##x,_n10##y,z,v), I[491] = (img)(_p3##x,_n10##y,z,v), I[492] = (img)(_p2##x,_n10##y,z,v), I[493] = (img)(_p1##x,_n10##y,z,v), I[494] = (img)(x,_n10##y,z,v), I[495] = (img)(_n1##x,_n10##y,z,v), I[496] = (img)(_n2##x,_n10##y,z,v), I[497] = (img)(_n3##x,_n10##y,z,v), I[498] = (img)(_n4##x,_n10##y,z,v), I[499] = (img)(_n5##x,_n10##y,z,v), I[500] = (img)(_n6##x,_n10##y,z,v), I[501] = (img)(_n7##x,_n10##y,z,v), I[502] = (img)(_n8##x,_n10##y,z,v), I[503] = (img)(_n9##x,_n10##y,z,v), I[504] = (img)(_n10##x,_n10##y,z,v), I[505] = (img)(_n11##x,_n10##y,z,v), \
 I[506] = (img)(_p11##x,_n11##y,z,v), I[507] = (img)(_p10##x,_n11##y,z,v), I[508] = (img)(_p9##x,_n11##y,z,v), I[509] = (img)(_p8##x,_n11##y,z,v), I[510] = (img)(_p7##x,_n11##y,z,v), I[511] = (img)(_p6##x,_n11##y,z,v), I[512] = (img)(_p5##x,_n11##y,z,v), I[513] = (img)(_p4##x,_n11##y,z,v), I[514] = (img)(_p3##x,_n11##y,z,v), I[515] = (img)(_p2##x,_n11##y,z,v), I[516] = (img)(_p1##x,_n11##y,z,v), I[517] = (img)(x,_n11##y,z,v), I[518] = (img)(_n1##x,_n11##y,z,v), I[519] = (img)(_n2##x,_n11##y,z,v), I[520] = (img)(_n3##x,_n11##y,z,v), I[521] = (img)(_n4##x,_n11##y,z,v), I[522] = (img)(_n5##x,_n11##y,z,v), I[523] = (img)(_n6##x,_n11##y,z,v), I[524] = (img)(_n7##x,_n11##y,z,v), I[525] = (img)(_n8##x,_n11##y,z,v), I[526] = (img)(_n9##x,_n11##y,z,v), I[527] = (img)(_n10##x,_n11##y,z,v), I[528] = (img)(_n11##x,_n11##y,z,v);

// Define 24x24 loop macros for CImg
//----------------------------------
#define cimg_for24(bound,i) for (int i = 0, \
 _p11##i = 0, _p10##i = 0, _p9##i = 0, _p8##i = 0, _p7##i = 0, _p6##i = 0, _p5##i = 0, _p4##i = 0, _p3##i = 0, _p2##i = 0, _p1##i = 0, \
 _n1##i = 1>=(int)(bound)?(int)(bound)-1:1, \
 _n2##i = 2>=(int)(bound)?(int)(bound)-1:2, \
 _n3##i = 3>=(int)(bound)?(int)(bound)-1:3, \
 _n4##i = 4>=(int)(bound)?(int)(bound)-1:4, \
 _n5##i = 5>=(int)(bound)?(int)(bound)-1:5, \
 _n6##i = 6>=(int)(bound)?(int)(bound)-1:6, \
 _n7##i = 7>=(int)(bound)?(int)(bound)-1:7, \
 _n8##i = 8>=(int)(bound)?(int)(bound)-1:8, \
 _n9##i = 9>=(int)(bound)?(int)(bound)-1:9, \
 _n10##i = 10>=(int)(bound)?(int)(bound)-1:10, \
 _n11##i = 11>=(int)(bound)?(int)(bound)-1:11, \
 _n12##i = 12>=(int)(bound)?(int)(bound)-1:12; \
 _n12##i<(int)(bound) || _n11##i==--_n12##i || _n10##i==--_n11##i || _n9##i==--_n10##i || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n12##i = _n11##i = _n10##i = _n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i); \
 _p11##i = _p10##i, _p10##i = _p9##i, _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i, ++_n10##i, ++_n11##i, ++_n12##i)

#define cimg_for24X(img,x) cimg_for24((img).width,x)
#define cimg_for24Y(img,y) cimg_for24((img).height,y)
#define cimg_for24Z(img,z) cimg_for24((img).depth,z)
#define cimg_for24V(img,v) cimg_for24((img).dim,v)
#define cimg_for24XY(img,x,y) cimg_for24Y(img,y) cimg_for24X(img,x)
#define cimg_for24XZ(img,x,z) cimg_for24Z(img,z) cimg_for24X(img,x)
#define cimg_for24XV(img,x,v) cimg_for24V(img,v) cimg_for24X(img,x)
#define cimg_for24YZ(img,y,z) cimg_for24Z(img,z) cimg_for24Y(img,y)
#define cimg_for24YV(img,y,v) cimg_for24V(img,v) cimg_for24Y(img,y)
#define cimg_for24ZV(img,z,v) cimg_for24V(img,v) cimg_for24Z(img,z)
#define cimg_for24XYZ(img,x,y,z) cimg_for24Z(img,z) cimg_for24XY(img,x,y)
#define cimg_for24XZV(img,x,z,v) cimg_for24V(img,v) cimg_for24XZ(img,x,z)
#define cimg_for24YZV(img,y,z,v) cimg_for24V(img,v) cimg_for24YZ(img,y,z)
#define cimg_for24XYZV(img,x,y,z,v) cimg_for24V(img,v) cimg_for24XYZ(img,x,y,z)

#define cimg_for_in24(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p11##i = i-11<0?0:i-11, \
 _p10##i = i-10<0?0:i-10, \
 _p9##i = i-9<0?0:i-9, \
 _p8##i = i-8<0?0:i-8, \
 _p7##i = i-7<0?0:i-7, \
 _p6##i = i-6<0?0:i-6, \
 _p5##i = i-5<0?0:i-5, \
 _p4##i = i-4<0?0:i-4, \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4, \
 _n5##i = i+5>=(int)(bound)?(int)(bound)-1:i+5, \
 _n6##i = i+6>=(int)(bound)?(int)(bound)-1:i+6, \
 _n7##i = i+7>=(int)(bound)?(int)(bound)-1:i+7, \
 _n8##i = i+8>=(int)(bound)?(int)(bound)-1:i+8, \
 _n9##i = i+9>=(int)(bound)?(int)(bound)-1:i+9, \
 _n10##i = i+10>=(int)(bound)?(int)(bound)-1:i+10, \
 _n11##i = i+11>=(int)(bound)?(int)(bound)-1:i+11, \
 _n12##i = i+12>=(int)(bound)?(int)(bound)-1:i+12; \
 i<=(int)(i1) && (_n12##i<(int)(bound) || _n11##i==--_n12##i || _n10##i==--_n11##i || _n9##i==--_n10##i || _n8##i==--_n9##i || _n7##i==--_n8##i || _n6##i==--_n7##i || _n5##i==--_n6##i || _n4##i==--_n5##i || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n12##i = _n11##i = _n10##i = _n9##i = _n8##i = _n7##i = _n6##i = _n5##i = _n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p11##i = _p10##i, _p10##i = _p9##i, _p9##i = _p8##i, _p8##i = _p7##i, _p7##i = _p6##i, _p6##i = _p5##i, _p5##i = _p4##i, _p4##i = _p3##i, _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i, ++_n5##i, ++_n6##i, ++_n7##i, ++_n8##i, ++_n9##i, ++_n10##i, ++_n11##i, ++_n12##i)

#define cimg_for_in24X(img,x0,x1,x) cimg_for_in24((img).width,x0,x1,x)
#define cimg_for_in24Y(img,y0,y1,y) cimg_for_in24((img).height,y0,y1,y)
#define cimg_for_in24Z(img,z0,z1,z) cimg_for_in24((img).depth,z0,z1,z)
#define cimg_for_in24V(img,v0,v1,v) cimg_for_in24((img).dim,v0,v1,v)
#define cimg_for_in24XY(img,x0,y0,x1,y1,x,y) cimg_for_in24Y(img,y0,y1,y) cimg_for_in24X(img,x0,x1,x)
#define cimg_for_in24XZ(img,x0,z0,x1,z1,x,z) cimg_for_in24Z(img,z0,z1,z) cimg_for_in24X(img,x0,x1,x)
#define cimg_for_in24XV(img,x0,v0,x1,v1,x,v) cimg_for_in24V(img,v0,v1,v) cimg_for_in24X(img,x0,x1,x)
#define cimg_for_in24YZ(img,y0,z0,y1,z1,y,z) cimg_for_in24Z(img,z0,z1,z) cimg_for_in24Y(img,y0,y1,y)
#define cimg_for_in24YV(img,y0,v0,y1,v1,y,v) cimg_for_in24V(img,v0,v1,v) cimg_for_in24Y(img,y0,y1,y)
#define cimg_for_in24ZV(img,z0,v0,z1,v1,z,v) cimg_for_in24V(img,v0,v1,v) cimg_for_in24Z(img,z0,z1,z)
#define cimg_for_in24XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in24Z(img,z0,z1,z) cimg_for_in24XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in24XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in24V(img,v0,v1,v) cimg_for_in24XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in24YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in24V(img,v0,v1,v) cimg_for_in24YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in24XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in24V(img,v0,v1,v) cimg_for_in24XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for24x24(img,x,y,z,v,I) \
 cimg_for24((img).height,y) for (int x = 0, \
 _p11##x = 0, _p10##x = 0, _p9##x = 0, _p8##x = 0, _p7##x = 0, _p6##x = 0, _p5##x = 0, _p4##x = 0, _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = 4>=((img).width)?(int)((img).width)-1:4, \
 _n5##x = 5>=((img).width)?(int)((img).width)-1:5, \
 _n6##x = 6>=((img).width)?(int)((img).width)-1:6, \
 _n7##x = 7>=((img).width)?(int)((img).width)-1:7, \
 _n8##x = 8>=((img).width)?(int)((img).width)-1:8, \
 _n9##x = 9>=((img).width)?(int)((img).width)-1:9, \
 _n10##x = 10>=((img).width)?(int)((img).width)-1:10, \
 _n11##x = 11>=((img).width)?(int)((img).width)-1:11, \
 _n12##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = I[4] = I[5] = I[6] = I[7] = I[8] = I[9] = I[10] = I[11] = (img)(0,_p11##y,z,v)), \
 (I[24] = I[25] = I[26] = I[27] = I[28] = I[29] = I[30] = I[31] = I[32] = I[33] = I[34] = I[35] = (img)(0,_p10##y,z,v)), \
 (I[48] = I[49] = I[50] = I[51] = I[52] = I[53] = I[54] = I[55] = I[56] = I[57] = I[58] = I[59] = (img)(0,_p9##y,z,v)), \
 (I[72] = I[73] = I[74] = I[75] = I[76] = I[77] = I[78] = I[79] = I[80] = I[81] = I[82] = I[83] = (img)(0,_p8##y,z,v)), \
 (I[96] = I[97] = I[98] = I[99] = I[100] = I[101] = I[102] = I[103] = I[104] = I[105] = I[106] = I[107] = (img)(0,_p7##y,z,v)), \
 (I[120] = I[121] = I[122] = I[123] = I[124] = I[125] = I[126] = I[127] = I[128] = I[129] = I[130] = I[131] = (img)(0,_p6##y,z,v)), \
 (I[144] = I[145] = I[146] = I[147] = I[148] = I[149] = I[150] = I[151] = I[152] = I[153] = I[154] = I[155] = (img)(0,_p5##y,z,v)), \
 (I[168] = I[169] = I[170] = I[171] = I[172] = I[173] = I[174] = I[175] = I[176] = I[177] = I[178] = I[179] = (img)(0,_p4##y,z,v)), \
 (I[192] = I[193] = I[194] = I[195] = I[196] = I[197] = I[198] = I[199] = I[200] = I[201] = I[202] = I[203] = (img)(0,_p3##y,z,v)), \
 (I[216] = I[217] = I[218] = I[219] = I[220] = I[221] = I[222] = I[223] = I[224] = I[225] = I[226] = I[227] = (img)(0,_p2##y,z,v)), \
 (I[240] = I[241] = I[242] = I[243] = I[244] = I[245] = I[246] = I[247] = I[248] = I[249] = I[250] = I[251] = (img)(0,_p1##y,z,v)), \
 (I[264] = I[265] = I[266] = I[267] = I[268] = I[269] = I[270] = I[271] = I[272] = I[273] = I[274] = I[275] = (img)(0,y,z,v)), \
 (I[288] = I[289] = I[290] = I[291] = I[292] = I[293] = I[294] = I[295] = I[296] = I[297] = I[298] = I[299] = (img)(0,_n1##y,z,v)), \
 (I[312] = I[313] = I[314] = I[315] = I[316] = I[317] = I[318] = I[319] = I[320] = I[321] = I[322] = I[323] = (img)(0,_n2##y,z,v)), \
 (I[336] = I[337] = I[338] = I[339] = I[340] = I[341] = I[342] = I[343] = I[344] = I[345] = I[346] = I[347] = (img)(0,_n3##y,z,v)), \
 (I[360] = I[361] = I[362] = I[363] = I[364] = I[365] = I[366] = I[367] = I[368] = I[369] = I[370] = I[371] = (img)(0,_n4##y,z,v)), \
 (I[384] = I[385] = I[386] = I[387] = I[388] = I[389] = I[390] = I[391] = I[392] = I[393] = I[394] = I[395] = (img)(0,_n5##y,z,v)), \
 (I[408] = I[409] = I[410] = I[411] = I[412] = I[413] = I[414] = I[415] = I[416] = I[417] = I[418] = I[419] = (img)(0,_n6##y,z,v)), \
 (I[432] = I[433] = I[434] = I[435] = I[436] = I[437] = I[438] = I[439] = I[440] = I[441] = I[442] = I[443] = (img)(0,_n7##y,z,v)), \
 (I[456] = I[457] = I[458] = I[459] = I[460] = I[461] = I[462] = I[463] = I[464] = I[465] = I[466] = I[467] = (img)(0,_n8##y,z,v)), \
 (I[480] = I[481] = I[482] = I[483] = I[484] = I[485] = I[486] = I[487] = I[488] = I[489] = I[490] = I[491] = (img)(0,_n9##y,z,v)), \
 (I[504] = I[505] = I[506] = I[507] = I[508] = I[509] = I[510] = I[511] = I[512] = I[513] = I[514] = I[515] = (img)(0,_n10##y,z,v)), \
 (I[528] = I[529] = I[530] = I[531] = I[532] = I[533] = I[534] = I[535] = I[536] = I[537] = I[538] = I[539] = (img)(0,_n11##y,z,v)), \
 (I[552] = I[553] = I[554] = I[555] = I[556] = I[557] = I[558] = I[559] = I[560] = I[561] = I[562] = I[563] = (img)(0,_n12##y,z,v)), \
 (I[12] = (img)(_n1##x,_p11##y,z,v)), \
 (I[36] = (img)(_n1##x,_p10##y,z,v)), \
 (I[60] = (img)(_n1##x,_p9##y,z,v)), \
 (I[84] = (img)(_n1##x,_p8##y,z,v)), \
 (I[108] = (img)(_n1##x,_p7##y,z,v)), \
 (I[132] = (img)(_n1##x,_p6##y,z,v)), \
 (I[156] = (img)(_n1##x,_p5##y,z,v)), \
 (I[180] = (img)(_n1##x,_p4##y,z,v)), \
 (I[204] = (img)(_n1##x,_p3##y,z,v)), \
 (I[228] = (img)(_n1##x,_p2##y,z,v)), \
 (I[252] = (img)(_n1##x,_p1##y,z,v)), \
 (I[276] = (img)(_n1##x,y,z,v)), \
 (I[300] = (img)(_n1##x,_n1##y,z,v)), \
 (I[324] = (img)(_n1##x,_n2##y,z,v)), \
 (I[348] = (img)(_n1##x,_n3##y,z,v)), \
 (I[372] = (img)(_n1##x,_n4##y,z,v)), \
 (I[396] = (img)(_n1##x,_n5##y,z,v)), \
 (I[420] = (img)(_n1##x,_n6##y,z,v)), \
 (I[444] = (img)(_n1##x,_n7##y,z,v)), \
 (I[468] = (img)(_n1##x,_n8##y,z,v)), \
 (I[492] = (img)(_n1##x,_n9##y,z,v)), \
 (I[516] = (img)(_n1##x,_n10##y,z,v)), \
 (I[540] = (img)(_n1##x,_n11##y,z,v)), \
 (I[564] = (img)(_n1##x,_n12##y,z,v)), \
 (I[13] = (img)(_n2##x,_p11##y,z,v)), \
 (I[37] = (img)(_n2##x,_p10##y,z,v)), \
 (I[61] = (img)(_n2##x,_p9##y,z,v)), \
 (I[85] = (img)(_n2##x,_p8##y,z,v)), \
 (I[109] = (img)(_n2##x,_p7##y,z,v)), \
 (I[133] = (img)(_n2##x,_p6##y,z,v)), \
 (I[157] = (img)(_n2##x,_p5##y,z,v)), \
 (I[181] = (img)(_n2##x,_p4##y,z,v)), \
 (I[205] = (img)(_n2##x,_p3##y,z,v)), \
 (I[229] = (img)(_n2##x,_p2##y,z,v)), \
 (I[253] = (img)(_n2##x,_p1##y,z,v)), \
 (I[277] = (img)(_n2##x,y,z,v)), \
 (I[301] = (img)(_n2##x,_n1##y,z,v)), \
 (I[325] = (img)(_n2##x,_n2##y,z,v)), \
 (I[349] = (img)(_n2##x,_n3##y,z,v)), \
 (I[373] = (img)(_n2##x,_n4##y,z,v)), \
 (I[397] = (img)(_n2##x,_n5##y,z,v)), \
 (I[421] = (img)(_n2##x,_n6##y,z,v)), \
 (I[445] = (img)(_n2##x,_n7##y,z,v)), \
 (I[469] = (img)(_n2##x,_n8##y,z,v)), \
 (I[493] = (img)(_n2##x,_n9##y,z,v)), \
 (I[517] = (img)(_n2##x,_n10##y,z,v)), \
 (I[541] = (img)(_n2##x,_n11##y,z,v)), \
 (I[565] = (img)(_n2##x,_n12##y,z,v)), \
 (I[14] = (img)(_n3##x,_p11##y,z,v)), \
 (I[38] = (img)(_n3##x,_p10##y,z,v)), \
 (I[62] = (img)(_n3##x,_p9##y,z,v)), \
 (I[86] = (img)(_n3##x,_p8##y,z,v)), \
 (I[110] = (img)(_n3##x,_p7##y,z,v)), \
 (I[134] = (img)(_n3##x,_p6##y,z,v)), \
 (I[158] = (img)(_n3##x,_p5##y,z,v)), \
 (I[182] = (img)(_n3##x,_p4##y,z,v)), \
 (I[206] = (img)(_n3##x,_p3##y,z,v)), \
 (I[230] = (img)(_n3##x,_p2##y,z,v)), \
 (I[254] = (img)(_n3##x,_p1##y,z,v)), \
 (I[278] = (img)(_n3##x,y,z,v)), \
 (I[302] = (img)(_n3##x,_n1##y,z,v)), \
 (I[326] = (img)(_n3##x,_n2##y,z,v)), \
 (I[350] = (img)(_n3##x,_n3##y,z,v)), \
 (I[374] = (img)(_n3##x,_n4##y,z,v)), \
 (I[398] = (img)(_n3##x,_n5##y,z,v)), \
 (I[422] = (img)(_n3##x,_n6##y,z,v)), \
 (I[446] = (img)(_n3##x,_n7##y,z,v)), \
 (I[470] = (img)(_n3##x,_n8##y,z,v)), \
 (I[494] = (img)(_n3##x,_n9##y,z,v)), \
 (I[518] = (img)(_n3##x,_n10##y,z,v)), \
 (I[542] = (img)(_n3##x,_n11##y,z,v)), \
 (I[566] = (img)(_n3##x,_n12##y,z,v)), \
 (I[15] = (img)(_n4##x,_p11##y,z,v)), \
 (I[39] = (img)(_n4##x,_p10##y,z,v)), \
 (I[63] = (img)(_n4##x,_p9##y,z,v)), \
 (I[87] = (img)(_n4##x,_p8##y,z,v)), \
 (I[111] = (img)(_n4##x,_p7##y,z,v)), \
 (I[135] = (img)(_n4##x,_p6##y,z,v)), \
 (I[159] = (img)(_n4##x,_p5##y,z,v)), \
 (I[183] = (img)(_n4##x,_p4##y,z,v)), \
 (I[207] = (img)(_n4##x,_p3##y,z,v)), \
 (I[231] = (img)(_n4##x,_p2##y,z,v)), \
 (I[255] = (img)(_n4##x,_p1##y,z,v)), \
 (I[279] = (img)(_n4##x,y,z,v)), \
 (I[303] = (img)(_n4##x,_n1##y,z,v)), \
 (I[327] = (img)(_n4##x,_n2##y,z,v)), \
 (I[351] = (img)(_n4##x,_n3##y,z,v)), \
 (I[375] = (img)(_n4##x,_n4##y,z,v)), \
 (I[399] = (img)(_n4##x,_n5##y,z,v)), \
 (I[423] = (img)(_n4##x,_n6##y,z,v)), \
 (I[447] = (img)(_n4##x,_n7##y,z,v)), \
 (I[471] = (img)(_n4##x,_n8##y,z,v)), \
 (I[495] = (img)(_n4##x,_n9##y,z,v)), \
 (I[519] = (img)(_n4##x,_n10##y,z,v)), \
 (I[543] = (img)(_n4##x,_n11##y,z,v)), \
 (I[567] = (img)(_n4##x,_n12##y,z,v)), \
 (I[16] = (img)(_n5##x,_p11##y,z,v)), \
 (I[40] = (img)(_n5##x,_p10##y,z,v)), \
 (I[64] = (img)(_n5##x,_p9##y,z,v)), \
 (I[88] = (img)(_n5##x,_p8##y,z,v)), \
 (I[112] = (img)(_n5##x,_p7##y,z,v)), \
 (I[136] = (img)(_n5##x,_p6##y,z,v)), \
 (I[160] = (img)(_n5##x,_p5##y,z,v)), \
 (I[184] = (img)(_n5##x,_p4##y,z,v)), \
 (I[208] = (img)(_n5##x,_p3##y,z,v)), \
 (I[232] = (img)(_n5##x,_p2##y,z,v)), \
 (I[256] = (img)(_n5##x,_p1##y,z,v)), \
 (I[280] = (img)(_n5##x,y,z,v)), \
 (I[304] = (img)(_n5##x,_n1##y,z,v)), \
 (I[328] = (img)(_n5##x,_n2##y,z,v)), \
 (I[352] = (img)(_n5##x,_n3##y,z,v)), \
 (I[376] = (img)(_n5##x,_n4##y,z,v)), \
 (I[400] = (img)(_n5##x,_n5##y,z,v)), \
 (I[424] = (img)(_n5##x,_n6##y,z,v)), \
 (I[448] = (img)(_n5##x,_n7##y,z,v)), \
 (I[472] = (img)(_n5##x,_n8##y,z,v)), \
 (I[496] = (img)(_n5##x,_n9##y,z,v)), \
 (I[520] = (img)(_n5##x,_n10##y,z,v)), \
 (I[544] = (img)(_n5##x,_n11##y,z,v)), \
 (I[568] = (img)(_n5##x,_n12##y,z,v)), \
 (I[17] = (img)(_n6##x,_p11##y,z,v)), \
 (I[41] = (img)(_n6##x,_p10##y,z,v)), \
 (I[65] = (img)(_n6##x,_p9##y,z,v)), \
 (I[89] = (img)(_n6##x,_p8##y,z,v)), \
 (I[113] = (img)(_n6##x,_p7##y,z,v)), \
 (I[137] = (img)(_n6##x,_p6##y,z,v)), \
 (I[161] = (img)(_n6##x,_p5##y,z,v)), \
 (I[185] = (img)(_n6##x,_p4##y,z,v)), \
 (I[209] = (img)(_n6##x,_p3##y,z,v)), \
 (I[233] = (img)(_n6##x,_p2##y,z,v)), \
 (I[257] = (img)(_n6##x,_p1##y,z,v)), \
 (I[281] = (img)(_n6##x,y,z,v)), \
 (I[305] = (img)(_n6##x,_n1##y,z,v)), \
 (I[329] = (img)(_n6##x,_n2##y,z,v)), \
 (I[353] = (img)(_n6##x,_n3##y,z,v)), \
 (I[377] = (img)(_n6##x,_n4##y,z,v)), \
 (I[401] = (img)(_n6##x,_n5##y,z,v)), \
 (I[425] = (img)(_n6##x,_n6##y,z,v)), \
 (I[449] = (img)(_n6##x,_n7##y,z,v)), \
 (I[473] = (img)(_n6##x,_n8##y,z,v)), \
 (I[497] = (img)(_n6##x,_n9##y,z,v)), \
 (I[521] = (img)(_n6##x,_n10##y,z,v)), \
 (I[545] = (img)(_n6##x,_n11##y,z,v)), \
 (I[569] = (img)(_n6##x,_n12##y,z,v)), \
 (I[18] = (img)(_n7##x,_p11##y,z,v)), \
 (I[42] = (img)(_n7##x,_p10##y,z,v)), \
 (I[66] = (img)(_n7##x,_p9##y,z,v)), \
 (I[90] = (img)(_n7##x,_p8##y,z,v)), \
 (I[114] = (img)(_n7##x,_p7##y,z,v)), \
 (I[138] = (img)(_n7##x,_p6##y,z,v)), \
 (I[162] = (img)(_n7##x,_p5##y,z,v)), \
 (I[186] = (img)(_n7##x,_p4##y,z,v)), \
 (I[210] = (img)(_n7##x,_p3##y,z,v)), \
 (I[234] = (img)(_n7##x,_p2##y,z,v)), \
 (I[258] = (img)(_n7##x,_p1##y,z,v)), \
 (I[282] = (img)(_n7##x,y,z,v)), \
 (I[306] = (img)(_n7##x,_n1##y,z,v)), \
 (I[330] = (img)(_n7##x,_n2##y,z,v)), \
 (I[354] = (img)(_n7##x,_n3##y,z,v)), \
 (I[378] = (img)(_n7##x,_n4##y,z,v)), \
 (I[402] = (img)(_n7##x,_n5##y,z,v)), \
 (I[426] = (img)(_n7##x,_n6##y,z,v)), \
 (I[450] = (img)(_n7##x,_n7##y,z,v)), \
 (I[474] = (img)(_n7##x,_n8##y,z,v)), \
 (I[498] = (img)(_n7##x,_n9##y,z,v)), \
 (I[522] = (img)(_n7##x,_n10##y,z,v)), \
 (I[546] = (img)(_n7##x,_n11##y,z,v)), \
 (I[570] = (img)(_n7##x,_n12##y,z,v)), \
 (I[19] = (img)(_n8##x,_p11##y,z,v)), \
 (I[43] = (img)(_n8##x,_p10##y,z,v)), \
 (I[67] = (img)(_n8##x,_p9##y,z,v)), \
 (I[91] = (img)(_n8##x,_p8##y,z,v)), \
 (I[115] = (img)(_n8##x,_p7##y,z,v)), \
 (I[139] = (img)(_n8##x,_p6##y,z,v)), \
 (I[163] = (img)(_n8##x,_p5##y,z,v)), \
 (I[187] = (img)(_n8##x,_p4##y,z,v)), \
 (I[211] = (img)(_n8##x,_p3##y,z,v)), \
 (I[235] = (img)(_n8##x,_p2##y,z,v)), \
 (I[259] = (img)(_n8##x,_p1##y,z,v)), \
 (I[283] = (img)(_n8##x,y,z,v)), \
 (I[307] = (img)(_n8##x,_n1##y,z,v)), \
 (I[331] = (img)(_n8##x,_n2##y,z,v)), \
 (I[355] = (img)(_n8##x,_n3##y,z,v)), \
 (I[379] = (img)(_n8##x,_n4##y,z,v)), \
 (I[403] = (img)(_n8##x,_n5##y,z,v)), \
 (I[427] = (img)(_n8##x,_n6##y,z,v)), \
 (I[451] = (img)(_n8##x,_n7##y,z,v)), \
 (I[475] = (img)(_n8##x,_n8##y,z,v)), \
 (I[499] = (img)(_n8##x,_n9##y,z,v)), \
 (I[523] = (img)(_n8##x,_n10##y,z,v)), \
 (I[547] = (img)(_n8##x,_n11##y,z,v)), \
 (I[571] = (img)(_n8##x,_n12##y,z,v)), \
 (I[20] = (img)(_n9##x,_p11##y,z,v)), \
 (I[44] = (img)(_n9##x,_p10##y,z,v)), \
 (I[68] = (img)(_n9##x,_p9##y,z,v)), \
 (I[92] = (img)(_n9##x,_p8##y,z,v)), \
 (I[116] = (img)(_n9##x,_p7##y,z,v)), \
 (I[140] = (img)(_n9##x,_p6##y,z,v)), \
 (I[164] = (img)(_n9##x,_p5##y,z,v)), \
 (I[188] = (img)(_n9##x,_p4##y,z,v)), \
 (I[212] = (img)(_n9##x,_p3##y,z,v)), \
 (I[236] = (img)(_n9##x,_p2##y,z,v)), \
 (I[260] = (img)(_n9##x,_p1##y,z,v)), \
 (I[284] = (img)(_n9##x,y,z,v)), \
 (I[308] = (img)(_n9##x,_n1##y,z,v)), \
 (I[332] = (img)(_n9##x,_n2##y,z,v)), \
 (I[356] = (img)(_n9##x,_n3##y,z,v)), \
 (I[380] = (img)(_n9##x,_n4##y,z,v)), \
 (I[404] = (img)(_n9##x,_n5##y,z,v)), \
 (I[428] = (img)(_n9##x,_n6##y,z,v)), \
 (I[452] = (img)(_n9##x,_n7##y,z,v)), \
 (I[476] = (img)(_n9##x,_n8##y,z,v)), \
 (I[500] = (img)(_n9##x,_n9##y,z,v)), \
 (I[524] = (img)(_n9##x,_n10##y,z,v)), \
 (I[548] = (img)(_n9##x,_n11##y,z,v)), \
 (I[572] = (img)(_n9##x,_n12##y,z,v)), \
 (I[21] = (img)(_n10##x,_p11##y,z,v)), \
 (I[45] = (img)(_n10##x,_p10##y,z,v)), \
 (I[69] = (img)(_n10##x,_p9##y,z,v)), \
 (I[93] = (img)(_n10##x,_p8##y,z,v)), \
 (I[117] = (img)(_n10##x,_p7##y,z,v)), \
 (I[141] = (img)(_n10##x,_p6##y,z,v)), \
 (I[165] = (img)(_n10##x,_p5##y,z,v)), \
 (I[189] = (img)(_n10##x,_p4##y,z,v)), \
 (I[213] = (img)(_n10##x,_p3##y,z,v)), \
 (I[237] = (img)(_n10##x,_p2##y,z,v)), \
 (I[261] = (img)(_n10##x,_p1##y,z,v)), \
 (I[285] = (img)(_n10##x,y,z,v)), \
 (I[309] = (img)(_n10##x,_n1##y,z,v)), \
 (I[333] = (img)(_n10##x,_n2##y,z,v)), \
 (I[357] = (img)(_n10##x,_n3##y,z,v)), \
 (I[381] = (img)(_n10##x,_n4##y,z,v)), \
 (I[405] = (img)(_n10##x,_n5##y,z,v)), \
 (I[429] = (img)(_n10##x,_n6##y,z,v)), \
 (I[453] = (img)(_n10##x,_n7##y,z,v)), \
 (I[477] = (img)(_n10##x,_n8##y,z,v)), \
 (I[501] = (img)(_n10##x,_n9##y,z,v)), \
 (I[525] = (img)(_n10##x,_n10##y,z,v)), \
 (I[549] = (img)(_n10##x,_n11##y,z,v)), \
 (I[573] = (img)(_n10##x,_n12##y,z,v)), \
 (I[22] = (img)(_n11##x,_p11##y,z,v)), \
 (I[46] = (img)(_n11##x,_p10##y,z,v)), \
 (I[70] = (img)(_n11##x,_p9##y,z,v)), \
 (I[94] = (img)(_n11##x,_p8##y,z,v)), \
 (I[118] = (img)(_n11##x,_p7##y,z,v)), \
 (I[142] = (img)(_n11##x,_p6##y,z,v)), \
 (I[166] = (img)(_n11##x,_p5##y,z,v)), \
 (I[190] = (img)(_n11##x,_p4##y,z,v)), \
 (I[214] = (img)(_n11##x,_p3##y,z,v)), \
 (I[238] = (img)(_n11##x,_p2##y,z,v)), \
 (I[262] = (img)(_n11##x,_p1##y,z,v)), \
 (I[286] = (img)(_n11##x,y,z,v)), \
 (I[310] = (img)(_n11##x,_n1##y,z,v)), \
 (I[334] = (img)(_n11##x,_n2##y,z,v)), \
 (I[358] = (img)(_n11##x,_n3##y,z,v)), \
 (I[382] = (img)(_n11##x,_n4##y,z,v)), \
 (I[406] = (img)(_n11##x,_n5##y,z,v)), \
 (I[430] = (img)(_n11##x,_n6##y,z,v)), \
 (I[454] = (img)(_n11##x,_n7##y,z,v)), \
 (I[478] = (img)(_n11##x,_n8##y,z,v)), \
 (I[502] = (img)(_n11##x,_n9##y,z,v)), \
 (I[526] = (img)(_n11##x,_n10##y,z,v)), \
 (I[550] = (img)(_n11##x,_n11##y,z,v)), \
 (I[574] = (img)(_n11##x,_n12##y,z,v)), \
 12>=((img).width)?(int)((img).width)-1:12); \
 (_n12##x<(int)((img).width) && ( \
 (I[23] = (img)(_n12##x,_p11##y,z,v)), \
 (I[47] = (img)(_n12##x,_p10##y,z,v)), \
 (I[71] = (img)(_n12##x,_p9##y,z,v)), \
 (I[95] = (img)(_n12##x,_p8##y,z,v)), \
 (I[119] = (img)(_n12##x,_p7##y,z,v)), \
 (I[143] = (img)(_n12##x,_p6##y,z,v)), \
 (I[167] = (img)(_n12##x,_p5##y,z,v)), \
 (I[191] = (img)(_n12##x,_p4##y,z,v)), \
 (I[215] = (img)(_n12##x,_p3##y,z,v)), \
 (I[239] = (img)(_n12##x,_p2##y,z,v)), \
 (I[263] = (img)(_n12##x,_p1##y,z,v)), \
 (I[287] = (img)(_n12##x,y,z,v)), \
 (I[311] = (img)(_n12##x,_n1##y,z,v)), \
 (I[335] = (img)(_n12##x,_n2##y,z,v)), \
 (I[359] = (img)(_n12##x,_n3##y,z,v)), \
 (I[383] = (img)(_n12##x,_n4##y,z,v)), \
 (I[407] = (img)(_n12##x,_n5##y,z,v)), \
 (I[431] = (img)(_n12##x,_n6##y,z,v)), \
 (I[455] = (img)(_n12##x,_n7##y,z,v)), \
 (I[479] = (img)(_n12##x,_n8##y,z,v)), \
 (I[503] = (img)(_n12##x,_n9##y,z,v)), \
 (I[527] = (img)(_n12##x,_n10##y,z,v)), \
 (I[551] = (img)(_n12##x,_n11##y,z,v)), \
 (I[575] = (img)(_n12##x,_n12##y,z,v)),1)) || \
 _n11##x==--_n12##x || _n10##x==--_n11##x || _n9##x==--_n10##x || _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n12##x = _n11##x = _n10##x = _n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], \
 I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], \
 I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], \
 I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], \
 I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], \
 I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], \
 I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], \
 I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], \
 I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], \
 I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], I[321] = I[322], I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], \
 I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], I[342] = I[343], I[343] = I[344], I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], I[351] = I[352], I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], \
 I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], I[367] = I[368], I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], I[373] = I[374], I[374] = I[375], I[375] = I[376], I[376] = I[377], I[377] = I[378], I[378] = I[379], I[379] = I[380], I[380] = I[381], I[381] = I[382], I[382] = I[383], \
 I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], I[390] = I[391], I[391] = I[392], I[392] = I[393], I[393] = I[394], I[394] = I[395], I[395] = I[396], I[396] = I[397], I[397] = I[398], I[398] = I[399], I[399] = I[400], I[400] = I[401], I[401] = I[402], I[402] = I[403], I[403] = I[404], I[404] = I[405], I[405] = I[406], I[406] = I[407], \
 I[408] = I[409], I[409] = I[410], I[410] = I[411], I[411] = I[412], I[412] = I[413], I[413] = I[414], I[414] = I[415], I[415] = I[416], I[416] = I[417], I[417] = I[418], I[418] = I[419], I[419] = I[420], I[420] = I[421], I[421] = I[422], I[422] = I[423], I[423] = I[424], I[424] = I[425], I[425] = I[426], I[426] = I[427], I[427] = I[428], I[428] = I[429], I[429] = I[430], I[430] = I[431], \
 I[432] = I[433], I[433] = I[434], I[434] = I[435], I[435] = I[436], I[436] = I[437], I[437] = I[438], I[438] = I[439], I[439] = I[440], I[440] = I[441], I[441] = I[442], I[442] = I[443], I[443] = I[444], I[444] = I[445], I[445] = I[446], I[446] = I[447], I[447] = I[448], I[448] = I[449], I[449] = I[450], I[450] = I[451], I[451] = I[452], I[452] = I[453], I[453] = I[454], I[454] = I[455], \
 I[456] = I[457], I[457] = I[458], I[458] = I[459], I[459] = I[460], I[460] = I[461], I[461] = I[462], I[462] = I[463], I[463] = I[464], I[464] = I[465], I[465] = I[466], I[466] = I[467], I[467] = I[468], I[468] = I[469], I[469] = I[470], I[470] = I[471], I[471] = I[472], I[472] = I[473], I[473] = I[474], I[474] = I[475], I[475] = I[476], I[476] = I[477], I[477] = I[478], I[478] = I[479], \
 I[480] = I[481], I[481] = I[482], I[482] = I[483], I[483] = I[484], I[484] = I[485], I[485] = I[486], I[486] = I[487], I[487] = I[488], I[488] = I[489], I[489] = I[490], I[490] = I[491], I[491] = I[492], I[492] = I[493], I[493] = I[494], I[494] = I[495], I[495] = I[496], I[496] = I[497], I[497] = I[498], I[498] = I[499], I[499] = I[500], I[500] = I[501], I[501] = I[502], I[502] = I[503], \
 I[504] = I[505], I[505] = I[506], I[506] = I[507], I[507] = I[508], I[508] = I[509], I[509] = I[510], I[510] = I[511], I[511] = I[512], I[512] = I[513], I[513] = I[514], I[514] = I[515], I[515] = I[516], I[516] = I[517], I[517] = I[518], I[518] = I[519], I[519] = I[520], I[520] = I[521], I[521] = I[522], I[522] = I[523], I[523] = I[524], I[524] = I[525], I[525] = I[526], I[526] = I[527], \
 I[528] = I[529], I[529] = I[530], I[530] = I[531], I[531] = I[532], I[532] = I[533], I[533] = I[534], I[534] = I[535], I[535] = I[536], I[536] = I[537], I[537] = I[538], I[538] = I[539], I[539] = I[540], I[540] = I[541], I[541] = I[542], I[542] = I[543], I[543] = I[544], I[544] = I[545], I[545] = I[546], I[546] = I[547], I[547] = I[548], I[548] = I[549], I[549] = I[550], I[550] = I[551], \
 I[552] = I[553], I[553] = I[554], I[554] = I[555], I[555] = I[556], I[556] = I[557], I[557] = I[558], I[558] = I[559], I[559] = I[560], I[560] = I[561], I[561] = I[562], I[562] = I[563], I[563] = I[564], I[564] = I[565], I[565] = I[566], I[566] = I[567], I[567] = I[568], I[568] = I[569], I[569] = I[570], I[570] = I[571], I[571] = I[572], I[572] = I[573], I[573] = I[574], I[574] = I[575], \
 _p11##x = _p10##x, _p10##x = _p9##x, _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x, ++_n10##x, ++_n11##x, ++_n12##x)

#define cimg_for_in24x24(img,x0,y0,x1,y1,x,y,z,v,I) \
 cimg_for_in24((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p11##x = x-11<0?0:x-11, \
 _p10##x = x-10<0?0:x-10, \
 _p9##x = x-9<0?0:x-9, \
 _p8##x = x-8<0?0:x-8, \
 _p7##x = x-7<0?0:x-7, \
 _p6##x = x-6<0?0:x-6, \
 _p5##x = x-5<0?0:x-5, \
 _p4##x = x-4<0?0:x-4, \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = x+4>=(int)((img).width)?(int)((img).width)-1:x+4, \
 _n5##x = x+5>=(int)((img).width)?(int)((img).width)-1:x+5, \
 _n6##x = x+6>=(int)((img).width)?(int)((img).width)-1:x+6, \
 _n7##x = x+7>=(int)((img).width)?(int)((img).width)-1:x+7, \
 _n8##x = x+8>=(int)((img).width)?(int)((img).width)-1:x+8, \
 _n9##x = x+9>=(int)((img).width)?(int)((img).width)-1:x+9, \
 _n10##x = x+10>=(int)((img).width)?(int)((img).width)-1:x+10, \
 _n11##x = x+11>=(int)((img).width)?(int)((img).width)-1:x+11, \
 _n12##x = (int)( \
 (I[0] = (img)(_p11##x,_p11##y,z,v)), \
 (I[24] = (img)(_p11##x,_p10##y,z,v)), \
 (I[48] = (img)(_p11##x,_p9##y,z,v)), \
 (I[72] = (img)(_p11##x,_p8##y,z,v)), \
 (I[96] = (img)(_p11##x,_p7##y,z,v)), \
 (I[120] = (img)(_p11##x,_p6##y,z,v)), \
 (I[144] = (img)(_p11##x,_p5##y,z,v)), \
 (I[168] = (img)(_p11##x,_p4##y,z,v)), \
 (I[192] = (img)(_p11##x,_p3##y,z,v)), \
 (I[216] = (img)(_p11##x,_p2##y,z,v)), \
 (I[240] = (img)(_p11##x,_p1##y,z,v)), \
 (I[264] = (img)(_p11##x,y,z,v)), \
 (I[288] = (img)(_p11##x,_n1##y,z,v)), \
 (I[312] = (img)(_p11##x,_n2##y,z,v)), \
 (I[336] = (img)(_p11##x,_n3##y,z,v)), \
 (I[360] = (img)(_p11##x,_n4##y,z,v)), \
 (I[384] = (img)(_p11##x,_n5##y,z,v)), \
 (I[408] = (img)(_p11##x,_n6##y,z,v)), \
 (I[432] = (img)(_p11##x,_n7##y,z,v)), \
 (I[456] = (img)(_p11##x,_n8##y,z,v)), \
 (I[480] = (img)(_p11##x,_n9##y,z,v)), \
 (I[504] = (img)(_p11##x,_n10##y,z,v)), \
 (I[528] = (img)(_p11##x,_n11##y,z,v)), \
 (I[552] = (img)(_p11##x,_n12##y,z,v)), \
 (I[1] = (img)(_p10##x,_p11##y,z,v)), \
 (I[25] = (img)(_p10##x,_p10##y,z,v)), \
 (I[49] = (img)(_p10##x,_p9##y,z,v)), \
 (I[73] = (img)(_p10##x,_p8##y,z,v)), \
 (I[97] = (img)(_p10##x,_p7##y,z,v)), \
 (I[121] = (img)(_p10##x,_p6##y,z,v)), \
 (I[145] = (img)(_p10##x,_p5##y,z,v)), \
 (I[169] = (img)(_p10##x,_p4##y,z,v)), \
 (I[193] = (img)(_p10##x,_p3##y,z,v)), \
 (I[217] = (img)(_p10##x,_p2##y,z,v)), \
 (I[241] = (img)(_p10##x,_p1##y,z,v)), \
 (I[265] = (img)(_p10##x,y,z,v)), \
 (I[289] = (img)(_p10##x,_n1##y,z,v)), \
 (I[313] = (img)(_p10##x,_n2##y,z,v)), \
 (I[337] = (img)(_p10##x,_n3##y,z,v)), \
 (I[361] = (img)(_p10##x,_n4##y,z,v)), \
 (I[385] = (img)(_p10##x,_n5##y,z,v)), \
 (I[409] = (img)(_p10##x,_n6##y,z,v)), \
 (I[433] = (img)(_p10##x,_n7##y,z,v)), \
 (I[457] = (img)(_p10##x,_n8##y,z,v)), \
 (I[481] = (img)(_p10##x,_n9##y,z,v)), \
 (I[505] = (img)(_p10##x,_n10##y,z,v)), \
 (I[529] = (img)(_p10##x,_n11##y,z,v)), \
 (I[553] = (img)(_p10##x,_n12##y,z,v)), \
 (I[2] = (img)(_p9##x,_p11##y,z,v)), \
 (I[26] = (img)(_p9##x,_p10##y,z,v)), \
 (I[50] = (img)(_p9##x,_p9##y,z,v)), \
 (I[74] = (img)(_p9##x,_p8##y,z,v)), \
 (I[98] = (img)(_p9##x,_p7##y,z,v)), \
 (I[122] = (img)(_p9##x,_p6##y,z,v)), \
 (I[146] = (img)(_p9##x,_p5##y,z,v)), \
 (I[170] = (img)(_p9##x,_p4##y,z,v)), \
 (I[194] = (img)(_p9##x,_p3##y,z,v)), \
 (I[218] = (img)(_p9##x,_p2##y,z,v)), \
 (I[242] = (img)(_p9##x,_p1##y,z,v)), \
 (I[266] = (img)(_p9##x,y,z,v)), \
 (I[290] = (img)(_p9##x,_n1##y,z,v)), \
 (I[314] = (img)(_p9##x,_n2##y,z,v)), \
 (I[338] = (img)(_p9##x,_n3##y,z,v)), \
 (I[362] = (img)(_p9##x,_n4##y,z,v)), \
 (I[386] = (img)(_p9##x,_n5##y,z,v)), \
 (I[410] = (img)(_p9##x,_n6##y,z,v)), \
 (I[434] = (img)(_p9##x,_n7##y,z,v)), \
 (I[458] = (img)(_p9##x,_n8##y,z,v)), \
 (I[482] = (img)(_p9##x,_n9##y,z,v)), \
 (I[506] = (img)(_p9##x,_n10##y,z,v)), \
 (I[530] = (img)(_p9##x,_n11##y,z,v)), \
 (I[554] = (img)(_p9##x,_n12##y,z,v)), \
 (I[3] = (img)(_p8##x,_p11##y,z,v)), \
 (I[27] = (img)(_p8##x,_p10##y,z,v)), \
 (I[51] = (img)(_p8##x,_p9##y,z,v)), \
 (I[75] = (img)(_p8##x,_p8##y,z,v)), \
 (I[99] = (img)(_p8##x,_p7##y,z,v)), \
 (I[123] = (img)(_p8##x,_p6##y,z,v)), \
 (I[147] = (img)(_p8##x,_p5##y,z,v)), \
 (I[171] = (img)(_p8##x,_p4##y,z,v)), \
 (I[195] = (img)(_p8##x,_p3##y,z,v)), \
 (I[219] = (img)(_p8##x,_p2##y,z,v)), \
 (I[243] = (img)(_p8##x,_p1##y,z,v)), \
 (I[267] = (img)(_p8##x,y,z,v)), \
 (I[291] = (img)(_p8##x,_n1##y,z,v)), \
 (I[315] = (img)(_p8##x,_n2##y,z,v)), \
 (I[339] = (img)(_p8##x,_n3##y,z,v)), \
 (I[363] = (img)(_p8##x,_n4##y,z,v)), \
 (I[387] = (img)(_p8##x,_n5##y,z,v)), \
 (I[411] = (img)(_p8##x,_n6##y,z,v)), \
 (I[435] = (img)(_p8##x,_n7##y,z,v)), \
 (I[459] = (img)(_p8##x,_n8##y,z,v)), \
 (I[483] = (img)(_p8##x,_n9##y,z,v)), \
 (I[507] = (img)(_p8##x,_n10##y,z,v)), \
 (I[531] = (img)(_p8##x,_n11##y,z,v)), \
 (I[555] = (img)(_p8##x,_n12##y,z,v)), \
 (I[4] = (img)(_p7##x,_p11##y,z,v)), \
 (I[28] = (img)(_p7##x,_p10##y,z,v)), \
 (I[52] = (img)(_p7##x,_p9##y,z,v)), \
 (I[76] = (img)(_p7##x,_p8##y,z,v)), \
 (I[100] = (img)(_p7##x,_p7##y,z,v)), \
 (I[124] = (img)(_p7##x,_p6##y,z,v)), \
 (I[148] = (img)(_p7##x,_p5##y,z,v)), \
 (I[172] = (img)(_p7##x,_p4##y,z,v)), \
 (I[196] = (img)(_p7##x,_p3##y,z,v)), \
 (I[220] = (img)(_p7##x,_p2##y,z,v)), \
 (I[244] = (img)(_p7##x,_p1##y,z,v)), \
 (I[268] = (img)(_p7##x,y,z,v)), \
 (I[292] = (img)(_p7##x,_n1##y,z,v)), \
 (I[316] = (img)(_p7##x,_n2##y,z,v)), \
 (I[340] = (img)(_p7##x,_n3##y,z,v)), \
 (I[364] = (img)(_p7##x,_n4##y,z,v)), \
 (I[388] = (img)(_p7##x,_n5##y,z,v)), \
 (I[412] = (img)(_p7##x,_n6##y,z,v)), \
 (I[436] = (img)(_p7##x,_n7##y,z,v)), \
 (I[460] = (img)(_p7##x,_n8##y,z,v)), \
 (I[484] = (img)(_p7##x,_n9##y,z,v)), \
 (I[508] = (img)(_p7##x,_n10##y,z,v)), \
 (I[532] = (img)(_p7##x,_n11##y,z,v)), \
 (I[556] = (img)(_p7##x,_n12##y,z,v)), \
 (I[5] = (img)(_p6##x,_p11##y,z,v)), \
 (I[29] = (img)(_p6##x,_p10##y,z,v)), \
 (I[53] = (img)(_p6##x,_p9##y,z,v)), \
 (I[77] = (img)(_p6##x,_p8##y,z,v)), \
 (I[101] = (img)(_p6##x,_p7##y,z,v)), \
 (I[125] = (img)(_p6##x,_p6##y,z,v)), \
 (I[149] = (img)(_p6##x,_p5##y,z,v)), \
 (I[173] = (img)(_p6##x,_p4##y,z,v)), \
 (I[197] = (img)(_p6##x,_p3##y,z,v)), \
 (I[221] = (img)(_p6##x,_p2##y,z,v)), \
 (I[245] = (img)(_p6##x,_p1##y,z,v)), \
 (I[269] = (img)(_p6##x,y,z,v)), \
 (I[293] = (img)(_p6##x,_n1##y,z,v)), \
 (I[317] = (img)(_p6##x,_n2##y,z,v)), \
 (I[341] = (img)(_p6##x,_n3##y,z,v)), \
 (I[365] = (img)(_p6##x,_n4##y,z,v)), \
 (I[389] = (img)(_p6##x,_n5##y,z,v)), \
 (I[413] = (img)(_p6##x,_n6##y,z,v)), \
 (I[437] = (img)(_p6##x,_n7##y,z,v)), \
 (I[461] = (img)(_p6##x,_n8##y,z,v)), \
 (I[485] = (img)(_p6##x,_n9##y,z,v)), \
 (I[509] = (img)(_p6##x,_n10##y,z,v)), \
 (I[533] = (img)(_p6##x,_n11##y,z,v)), \
 (I[557] = (img)(_p6##x,_n12##y,z,v)), \
 (I[6] = (img)(_p5##x,_p11##y,z,v)), \
 (I[30] = (img)(_p5##x,_p10##y,z,v)), \
 (I[54] = (img)(_p5##x,_p9##y,z,v)), \
 (I[78] = (img)(_p5##x,_p8##y,z,v)), \
 (I[102] = (img)(_p5##x,_p7##y,z,v)), \
 (I[126] = (img)(_p5##x,_p6##y,z,v)), \
 (I[150] = (img)(_p5##x,_p5##y,z,v)), \
 (I[174] = (img)(_p5##x,_p4##y,z,v)), \
 (I[198] = (img)(_p5##x,_p3##y,z,v)), \
 (I[222] = (img)(_p5##x,_p2##y,z,v)), \
 (I[246] = (img)(_p5##x,_p1##y,z,v)), \
 (I[270] = (img)(_p5##x,y,z,v)), \
 (I[294] = (img)(_p5##x,_n1##y,z,v)), \
 (I[318] = (img)(_p5##x,_n2##y,z,v)), \
 (I[342] = (img)(_p5##x,_n3##y,z,v)), \
 (I[366] = (img)(_p5##x,_n4##y,z,v)), \
 (I[390] = (img)(_p5##x,_n5##y,z,v)), \
 (I[414] = (img)(_p5##x,_n6##y,z,v)), \
 (I[438] = (img)(_p5##x,_n7##y,z,v)), \
 (I[462] = (img)(_p5##x,_n8##y,z,v)), \
 (I[486] = (img)(_p5##x,_n9##y,z,v)), \
 (I[510] = (img)(_p5##x,_n10##y,z,v)), \
 (I[534] = (img)(_p5##x,_n11##y,z,v)), \
 (I[558] = (img)(_p5##x,_n12##y,z,v)), \
 (I[7] = (img)(_p4##x,_p11##y,z,v)), \
 (I[31] = (img)(_p4##x,_p10##y,z,v)), \
 (I[55] = (img)(_p4##x,_p9##y,z,v)), \
 (I[79] = (img)(_p4##x,_p8##y,z,v)), \
 (I[103] = (img)(_p4##x,_p7##y,z,v)), \
 (I[127] = (img)(_p4##x,_p6##y,z,v)), \
 (I[151] = (img)(_p4##x,_p5##y,z,v)), \
 (I[175] = (img)(_p4##x,_p4##y,z,v)), \
 (I[199] = (img)(_p4##x,_p3##y,z,v)), \
 (I[223] = (img)(_p4##x,_p2##y,z,v)), \
 (I[247] = (img)(_p4##x,_p1##y,z,v)), \
 (I[271] = (img)(_p4##x,y,z,v)), \
 (I[295] = (img)(_p4##x,_n1##y,z,v)), \
 (I[319] = (img)(_p4##x,_n2##y,z,v)), \
 (I[343] = (img)(_p4##x,_n3##y,z,v)), \
 (I[367] = (img)(_p4##x,_n4##y,z,v)), \
 (I[391] = (img)(_p4##x,_n5##y,z,v)), \
 (I[415] = (img)(_p4##x,_n6##y,z,v)), \
 (I[439] = (img)(_p4##x,_n7##y,z,v)), \
 (I[463] = (img)(_p4##x,_n8##y,z,v)), \
 (I[487] = (img)(_p4##x,_n9##y,z,v)), \
 (I[511] = (img)(_p4##x,_n10##y,z,v)), \
 (I[535] = (img)(_p4##x,_n11##y,z,v)), \
 (I[559] = (img)(_p4##x,_n12##y,z,v)), \
 (I[8] = (img)(_p3##x,_p11##y,z,v)), \
 (I[32] = (img)(_p3##x,_p10##y,z,v)), \
 (I[56] = (img)(_p3##x,_p9##y,z,v)), \
 (I[80] = (img)(_p3##x,_p8##y,z,v)), \
 (I[104] = (img)(_p3##x,_p7##y,z,v)), \
 (I[128] = (img)(_p3##x,_p6##y,z,v)), \
 (I[152] = (img)(_p3##x,_p5##y,z,v)), \
 (I[176] = (img)(_p3##x,_p4##y,z,v)), \
 (I[200] = (img)(_p3##x,_p3##y,z,v)), \
 (I[224] = (img)(_p3##x,_p2##y,z,v)), \
 (I[248] = (img)(_p3##x,_p1##y,z,v)), \
 (I[272] = (img)(_p3##x,y,z,v)), \
 (I[296] = (img)(_p3##x,_n1##y,z,v)), \
 (I[320] = (img)(_p3##x,_n2##y,z,v)), \
 (I[344] = (img)(_p3##x,_n3##y,z,v)), \
 (I[368] = (img)(_p3##x,_n4##y,z,v)), \
 (I[392] = (img)(_p3##x,_n5##y,z,v)), \
 (I[416] = (img)(_p3##x,_n6##y,z,v)), \
 (I[440] = (img)(_p3##x,_n7##y,z,v)), \
 (I[464] = (img)(_p3##x,_n8##y,z,v)), \
 (I[488] = (img)(_p3##x,_n9##y,z,v)), \
 (I[512] = (img)(_p3##x,_n10##y,z,v)), \
 (I[536] = (img)(_p3##x,_n11##y,z,v)), \
 (I[560] = (img)(_p3##x,_n12##y,z,v)), \
 (I[9] = (img)(_p2##x,_p11##y,z,v)), \
 (I[33] = (img)(_p2##x,_p10##y,z,v)), \
 (I[57] = (img)(_p2##x,_p9##y,z,v)), \
 (I[81] = (img)(_p2##x,_p8##y,z,v)), \
 (I[105] = (img)(_p2##x,_p7##y,z,v)), \
 (I[129] = (img)(_p2##x,_p6##y,z,v)), \
 (I[153] = (img)(_p2##x,_p5##y,z,v)), \
 (I[177] = (img)(_p2##x,_p4##y,z,v)), \
 (I[201] = (img)(_p2##x,_p3##y,z,v)), \
 (I[225] = (img)(_p2##x,_p2##y,z,v)), \
 (I[249] = (img)(_p2##x,_p1##y,z,v)), \
 (I[273] = (img)(_p2##x,y,z,v)), \
 (I[297] = (img)(_p2##x,_n1##y,z,v)), \
 (I[321] = (img)(_p2##x,_n2##y,z,v)), \
 (I[345] = (img)(_p2##x,_n3##y,z,v)), \
 (I[369] = (img)(_p2##x,_n4##y,z,v)), \
 (I[393] = (img)(_p2##x,_n5##y,z,v)), \
 (I[417] = (img)(_p2##x,_n6##y,z,v)), \
 (I[441] = (img)(_p2##x,_n7##y,z,v)), \
 (I[465] = (img)(_p2##x,_n8##y,z,v)), \
 (I[489] = (img)(_p2##x,_n9##y,z,v)), \
 (I[513] = (img)(_p2##x,_n10##y,z,v)), \
 (I[537] = (img)(_p2##x,_n11##y,z,v)), \
 (I[561] = (img)(_p2##x,_n12##y,z,v)), \
 (I[10] = (img)(_p1##x,_p11##y,z,v)), \
 (I[34] = (img)(_p1##x,_p10##y,z,v)), \
 (I[58] = (img)(_p1##x,_p9##y,z,v)), \
 (I[82] = (img)(_p1##x,_p8##y,z,v)), \
 (I[106] = (img)(_p1##x,_p7##y,z,v)), \
 (I[130] = (img)(_p1##x,_p6##y,z,v)), \
 (I[154] = (img)(_p1##x,_p5##y,z,v)), \
 (I[178] = (img)(_p1##x,_p4##y,z,v)), \
 (I[202] = (img)(_p1##x,_p3##y,z,v)), \
 (I[226] = (img)(_p1##x,_p2##y,z,v)), \
 (I[250] = (img)(_p1##x,_p1##y,z,v)), \
 (I[274] = (img)(_p1##x,y,z,v)), \
 (I[298] = (img)(_p1##x,_n1##y,z,v)), \
 (I[322] = (img)(_p1##x,_n2##y,z,v)), \
 (I[346] = (img)(_p1##x,_n3##y,z,v)), \
 (I[370] = (img)(_p1##x,_n4##y,z,v)), \
 (I[394] = (img)(_p1##x,_n5##y,z,v)), \
 (I[418] = (img)(_p1##x,_n6##y,z,v)), \
 (I[442] = (img)(_p1##x,_n7##y,z,v)), \
 (I[466] = (img)(_p1##x,_n8##y,z,v)), \
 (I[490] = (img)(_p1##x,_n9##y,z,v)), \
 (I[514] = (img)(_p1##x,_n10##y,z,v)), \
 (I[538] = (img)(_p1##x,_n11##y,z,v)), \
 (I[562] = (img)(_p1##x,_n12##y,z,v)), \
 (I[11] = (img)(x,_p11##y,z,v)), \
 (I[35] = (img)(x,_p10##y,z,v)), \
 (I[59] = (img)(x,_p9##y,z,v)), \
 (I[83] = (img)(x,_p8##y,z,v)), \
 (I[107] = (img)(x,_p7##y,z,v)), \
 (I[131] = (img)(x,_p6##y,z,v)), \
 (I[155] = (img)(x,_p5##y,z,v)), \
 (I[179] = (img)(x,_p4##y,z,v)), \
 (I[203] = (img)(x,_p3##y,z,v)), \
 (I[227] = (img)(x,_p2##y,z,v)), \
 (I[251] = (img)(x,_p1##y,z,v)), \
 (I[275] = (img)(x,y,z,v)), \
 (I[299] = (img)(x,_n1##y,z,v)), \
 (I[323] = (img)(x,_n2##y,z,v)), \
 (I[347] = (img)(x,_n3##y,z,v)), \
 (I[371] = (img)(x,_n4##y,z,v)), \
 (I[395] = (img)(x,_n5##y,z,v)), \
 (I[419] = (img)(x,_n6##y,z,v)), \
 (I[443] = (img)(x,_n7##y,z,v)), \
 (I[467] = (img)(x,_n8##y,z,v)), \
 (I[491] = (img)(x,_n9##y,z,v)), \
 (I[515] = (img)(x,_n10##y,z,v)), \
 (I[539] = (img)(x,_n11##y,z,v)), \
 (I[563] = (img)(x,_n12##y,z,v)), \
 (I[12] = (img)(_n1##x,_p11##y,z,v)), \
 (I[36] = (img)(_n1##x,_p10##y,z,v)), \
 (I[60] = (img)(_n1##x,_p9##y,z,v)), \
 (I[84] = (img)(_n1##x,_p8##y,z,v)), \
 (I[108] = (img)(_n1##x,_p7##y,z,v)), \
 (I[132] = (img)(_n1##x,_p6##y,z,v)), \
 (I[156] = (img)(_n1##x,_p5##y,z,v)), \
 (I[180] = (img)(_n1##x,_p4##y,z,v)), \
 (I[204] = (img)(_n1##x,_p3##y,z,v)), \
 (I[228] = (img)(_n1##x,_p2##y,z,v)), \
 (I[252] = (img)(_n1##x,_p1##y,z,v)), \
 (I[276] = (img)(_n1##x,y,z,v)), \
 (I[300] = (img)(_n1##x,_n1##y,z,v)), \
 (I[324] = (img)(_n1##x,_n2##y,z,v)), \
 (I[348] = (img)(_n1##x,_n3##y,z,v)), \
 (I[372] = (img)(_n1##x,_n4##y,z,v)), \
 (I[396] = (img)(_n1##x,_n5##y,z,v)), \
 (I[420] = (img)(_n1##x,_n6##y,z,v)), \
 (I[444] = (img)(_n1##x,_n7##y,z,v)), \
 (I[468] = (img)(_n1##x,_n8##y,z,v)), \
 (I[492] = (img)(_n1##x,_n9##y,z,v)), \
 (I[516] = (img)(_n1##x,_n10##y,z,v)), \
 (I[540] = (img)(_n1##x,_n11##y,z,v)), \
 (I[564] = (img)(_n1##x,_n12##y,z,v)), \
 (I[13] = (img)(_n2##x,_p11##y,z,v)), \
 (I[37] = (img)(_n2##x,_p10##y,z,v)), \
 (I[61] = (img)(_n2##x,_p9##y,z,v)), \
 (I[85] = (img)(_n2##x,_p8##y,z,v)), \
 (I[109] = (img)(_n2##x,_p7##y,z,v)), \
 (I[133] = (img)(_n2##x,_p6##y,z,v)), \
 (I[157] = (img)(_n2##x,_p5##y,z,v)), \
 (I[181] = (img)(_n2##x,_p4##y,z,v)), \
 (I[205] = (img)(_n2##x,_p3##y,z,v)), \
 (I[229] = (img)(_n2##x,_p2##y,z,v)), \
 (I[253] = (img)(_n2##x,_p1##y,z,v)), \
 (I[277] = (img)(_n2##x,y,z,v)), \
 (I[301] = (img)(_n2##x,_n1##y,z,v)), \
 (I[325] = (img)(_n2##x,_n2##y,z,v)), \
 (I[349] = (img)(_n2##x,_n3##y,z,v)), \
 (I[373] = (img)(_n2##x,_n4##y,z,v)), \
 (I[397] = (img)(_n2##x,_n5##y,z,v)), \
 (I[421] = (img)(_n2##x,_n6##y,z,v)), \
 (I[445] = (img)(_n2##x,_n7##y,z,v)), \
 (I[469] = (img)(_n2##x,_n8##y,z,v)), \
 (I[493] = (img)(_n2##x,_n9##y,z,v)), \
 (I[517] = (img)(_n2##x,_n10##y,z,v)), \
 (I[541] = (img)(_n2##x,_n11##y,z,v)), \
 (I[565] = (img)(_n2##x,_n12##y,z,v)), \
 (I[14] = (img)(_n3##x,_p11##y,z,v)), \
 (I[38] = (img)(_n3##x,_p10##y,z,v)), \
 (I[62] = (img)(_n3##x,_p9##y,z,v)), \
 (I[86] = (img)(_n3##x,_p8##y,z,v)), \
 (I[110] = (img)(_n3##x,_p7##y,z,v)), \
 (I[134] = (img)(_n3##x,_p6##y,z,v)), \
 (I[158] = (img)(_n3##x,_p5##y,z,v)), \
 (I[182] = (img)(_n3##x,_p4##y,z,v)), \
 (I[206] = (img)(_n3##x,_p3##y,z,v)), \
 (I[230] = (img)(_n3##x,_p2##y,z,v)), \
 (I[254] = (img)(_n3##x,_p1##y,z,v)), \
 (I[278] = (img)(_n3##x,y,z,v)), \
 (I[302] = (img)(_n3##x,_n1##y,z,v)), \
 (I[326] = (img)(_n3##x,_n2##y,z,v)), \
 (I[350] = (img)(_n3##x,_n3##y,z,v)), \
 (I[374] = (img)(_n3##x,_n4##y,z,v)), \
 (I[398] = (img)(_n3##x,_n5##y,z,v)), \
 (I[422] = (img)(_n3##x,_n6##y,z,v)), \
 (I[446] = (img)(_n3##x,_n7##y,z,v)), \
 (I[470] = (img)(_n3##x,_n8##y,z,v)), \
 (I[494] = (img)(_n3##x,_n9##y,z,v)), \
 (I[518] = (img)(_n3##x,_n10##y,z,v)), \
 (I[542] = (img)(_n3##x,_n11##y,z,v)), \
 (I[566] = (img)(_n3##x,_n12##y,z,v)), \
 (I[15] = (img)(_n4##x,_p11##y,z,v)), \
 (I[39] = (img)(_n4##x,_p10##y,z,v)), \
 (I[63] = (img)(_n4##x,_p9##y,z,v)), \
 (I[87] = (img)(_n4##x,_p8##y,z,v)), \
 (I[111] = (img)(_n4##x,_p7##y,z,v)), \
 (I[135] = (img)(_n4##x,_p6##y,z,v)), \
 (I[159] = (img)(_n4##x,_p5##y,z,v)), \
 (I[183] = (img)(_n4##x,_p4##y,z,v)), \
 (I[207] = (img)(_n4##x,_p3##y,z,v)), \
 (I[231] = (img)(_n4##x,_p2##y,z,v)), \
 (I[255] = (img)(_n4##x,_p1##y,z,v)), \
 (I[279] = (img)(_n4##x,y,z,v)), \
 (I[303] = (img)(_n4##x,_n1##y,z,v)), \
 (I[327] = (img)(_n4##x,_n2##y,z,v)), \
 (I[351] = (img)(_n4##x,_n3##y,z,v)), \
 (I[375] = (img)(_n4##x,_n4##y,z,v)), \
 (I[399] = (img)(_n4##x,_n5##y,z,v)), \
 (I[423] = (img)(_n4##x,_n6##y,z,v)), \
 (I[447] = (img)(_n4##x,_n7##y,z,v)), \
 (I[471] = (img)(_n4##x,_n8##y,z,v)), \
 (I[495] = (img)(_n4##x,_n9##y,z,v)), \
 (I[519] = (img)(_n4##x,_n10##y,z,v)), \
 (I[543] = (img)(_n4##x,_n11##y,z,v)), \
 (I[567] = (img)(_n4##x,_n12##y,z,v)), \
 (I[16] = (img)(_n5##x,_p11##y,z,v)), \
 (I[40] = (img)(_n5##x,_p10##y,z,v)), \
 (I[64] = (img)(_n5##x,_p9##y,z,v)), \
 (I[88] = (img)(_n5##x,_p8##y,z,v)), \
 (I[112] = (img)(_n5##x,_p7##y,z,v)), \
 (I[136] = (img)(_n5##x,_p6##y,z,v)), \
 (I[160] = (img)(_n5##x,_p5##y,z,v)), \
 (I[184] = (img)(_n5##x,_p4##y,z,v)), \
 (I[208] = (img)(_n5##x,_p3##y,z,v)), \
 (I[232] = (img)(_n5##x,_p2##y,z,v)), \
 (I[256] = (img)(_n5##x,_p1##y,z,v)), \
 (I[280] = (img)(_n5##x,y,z,v)), \
 (I[304] = (img)(_n5##x,_n1##y,z,v)), \
 (I[328] = (img)(_n5##x,_n2##y,z,v)), \
 (I[352] = (img)(_n5##x,_n3##y,z,v)), \
 (I[376] = (img)(_n5##x,_n4##y,z,v)), \
 (I[400] = (img)(_n5##x,_n5##y,z,v)), \
 (I[424] = (img)(_n5##x,_n6##y,z,v)), \
 (I[448] = (img)(_n5##x,_n7##y,z,v)), \
 (I[472] = (img)(_n5##x,_n8##y,z,v)), \
 (I[496] = (img)(_n5##x,_n9##y,z,v)), \
 (I[520] = (img)(_n5##x,_n10##y,z,v)), \
 (I[544] = (img)(_n5##x,_n11##y,z,v)), \
 (I[568] = (img)(_n5##x,_n12##y,z,v)), \
 (I[17] = (img)(_n6##x,_p11##y,z,v)), \
 (I[41] = (img)(_n6##x,_p10##y,z,v)), \
 (I[65] = (img)(_n6##x,_p9##y,z,v)), \
 (I[89] = (img)(_n6##x,_p8##y,z,v)), \
 (I[113] = (img)(_n6##x,_p7##y,z,v)), \
 (I[137] = (img)(_n6##x,_p6##y,z,v)), \
 (I[161] = (img)(_n6##x,_p5##y,z,v)), \
 (I[185] = (img)(_n6##x,_p4##y,z,v)), \
 (I[209] = (img)(_n6##x,_p3##y,z,v)), \
 (I[233] = (img)(_n6##x,_p2##y,z,v)), \
 (I[257] = (img)(_n6##x,_p1##y,z,v)), \
 (I[281] = (img)(_n6##x,y,z,v)), \
 (I[305] = (img)(_n6##x,_n1##y,z,v)), \
 (I[329] = (img)(_n6##x,_n2##y,z,v)), \
 (I[353] = (img)(_n6##x,_n3##y,z,v)), \
 (I[377] = (img)(_n6##x,_n4##y,z,v)), \
 (I[401] = (img)(_n6##x,_n5##y,z,v)), \
 (I[425] = (img)(_n6##x,_n6##y,z,v)), \
 (I[449] = (img)(_n6##x,_n7##y,z,v)), \
 (I[473] = (img)(_n6##x,_n8##y,z,v)), \
 (I[497] = (img)(_n6##x,_n9##y,z,v)), \
 (I[521] = (img)(_n6##x,_n10##y,z,v)), \
 (I[545] = (img)(_n6##x,_n11##y,z,v)), \
 (I[569] = (img)(_n6##x,_n12##y,z,v)), \
 (I[18] = (img)(_n7##x,_p11##y,z,v)), \
 (I[42] = (img)(_n7##x,_p10##y,z,v)), \
 (I[66] = (img)(_n7##x,_p9##y,z,v)), \
 (I[90] = (img)(_n7##x,_p8##y,z,v)), \
 (I[114] = (img)(_n7##x,_p7##y,z,v)), \
 (I[138] = (img)(_n7##x,_p6##y,z,v)), \
 (I[162] = (img)(_n7##x,_p5##y,z,v)), \
 (I[186] = (img)(_n7##x,_p4##y,z,v)), \
 (I[210] = (img)(_n7##x,_p3##y,z,v)), \
 (I[234] = (img)(_n7##x,_p2##y,z,v)), \
 (I[258] = (img)(_n7##x,_p1##y,z,v)), \
 (I[282] = (img)(_n7##x,y,z,v)), \
 (I[306] = (img)(_n7##x,_n1##y,z,v)), \
 (I[330] = (img)(_n7##x,_n2##y,z,v)), \
 (I[354] = (img)(_n7##x,_n3##y,z,v)), \
 (I[378] = (img)(_n7##x,_n4##y,z,v)), \
 (I[402] = (img)(_n7##x,_n5##y,z,v)), \
 (I[426] = (img)(_n7##x,_n6##y,z,v)), \
 (I[450] = (img)(_n7##x,_n7##y,z,v)), \
 (I[474] = (img)(_n7##x,_n8##y,z,v)), \
 (I[498] = (img)(_n7##x,_n9##y,z,v)), \
 (I[522] = (img)(_n7##x,_n10##y,z,v)), \
 (I[546] = (img)(_n7##x,_n11##y,z,v)), \
 (I[570] = (img)(_n7##x,_n12##y,z,v)), \
 (I[19] = (img)(_n8##x,_p11##y,z,v)), \
 (I[43] = (img)(_n8##x,_p10##y,z,v)), \
 (I[67] = (img)(_n8##x,_p9##y,z,v)), \
 (I[91] = (img)(_n8##x,_p8##y,z,v)), \
 (I[115] = (img)(_n8##x,_p7##y,z,v)), \
 (I[139] = (img)(_n8##x,_p6##y,z,v)), \
 (I[163] = (img)(_n8##x,_p5##y,z,v)), \
 (I[187] = (img)(_n8##x,_p4##y,z,v)), \
 (I[211] = (img)(_n8##x,_p3##y,z,v)), \
 (I[235] = (img)(_n8##x,_p2##y,z,v)), \
 (I[259] = (img)(_n8##x,_p1##y,z,v)), \
 (I[283] = (img)(_n8##x,y,z,v)), \
 (I[307] = (img)(_n8##x,_n1##y,z,v)), \
 (I[331] = (img)(_n8##x,_n2##y,z,v)), \
 (I[355] = (img)(_n8##x,_n3##y,z,v)), \
 (I[379] = (img)(_n8##x,_n4##y,z,v)), \
 (I[403] = (img)(_n8##x,_n5##y,z,v)), \
 (I[427] = (img)(_n8##x,_n6##y,z,v)), \
 (I[451] = (img)(_n8##x,_n7##y,z,v)), \
 (I[475] = (img)(_n8##x,_n8##y,z,v)), \
 (I[499] = (img)(_n8##x,_n9##y,z,v)), \
 (I[523] = (img)(_n8##x,_n10##y,z,v)), \
 (I[547] = (img)(_n8##x,_n11##y,z,v)), \
 (I[571] = (img)(_n8##x,_n12##y,z,v)), \
 (I[20] = (img)(_n9##x,_p11##y,z,v)), \
 (I[44] = (img)(_n9##x,_p10##y,z,v)), \
 (I[68] = (img)(_n9##x,_p9##y,z,v)), \
 (I[92] = (img)(_n9##x,_p8##y,z,v)), \
 (I[116] = (img)(_n9##x,_p7##y,z,v)), \
 (I[140] = (img)(_n9##x,_p6##y,z,v)), \
 (I[164] = (img)(_n9##x,_p5##y,z,v)), \
 (I[188] = (img)(_n9##x,_p4##y,z,v)), \
 (I[212] = (img)(_n9##x,_p3##y,z,v)), \
 (I[236] = (img)(_n9##x,_p2##y,z,v)), \
 (I[260] = (img)(_n9##x,_p1##y,z,v)), \
 (I[284] = (img)(_n9##x,y,z,v)), \
 (I[308] = (img)(_n9##x,_n1##y,z,v)), \
 (I[332] = (img)(_n9##x,_n2##y,z,v)), \
 (I[356] = (img)(_n9##x,_n3##y,z,v)), \
 (I[380] = (img)(_n9##x,_n4##y,z,v)), \
 (I[404] = (img)(_n9##x,_n5##y,z,v)), \
 (I[428] = (img)(_n9##x,_n6##y,z,v)), \
 (I[452] = (img)(_n9##x,_n7##y,z,v)), \
 (I[476] = (img)(_n9##x,_n8##y,z,v)), \
 (I[500] = (img)(_n9##x,_n9##y,z,v)), \
 (I[524] = (img)(_n9##x,_n10##y,z,v)), \
 (I[548] = (img)(_n9##x,_n11##y,z,v)), \
 (I[572] = (img)(_n9##x,_n12##y,z,v)), \
 (I[21] = (img)(_n10##x,_p11##y,z,v)), \
 (I[45] = (img)(_n10##x,_p10##y,z,v)), \
 (I[69] = (img)(_n10##x,_p9##y,z,v)), \
 (I[93] = (img)(_n10##x,_p8##y,z,v)), \
 (I[117] = (img)(_n10##x,_p7##y,z,v)), \
 (I[141] = (img)(_n10##x,_p6##y,z,v)), \
 (I[165] = (img)(_n10##x,_p5##y,z,v)), \
 (I[189] = (img)(_n10##x,_p4##y,z,v)), \
 (I[213] = (img)(_n10##x,_p3##y,z,v)), \
 (I[237] = (img)(_n10##x,_p2##y,z,v)), \
 (I[261] = (img)(_n10##x,_p1##y,z,v)), \
 (I[285] = (img)(_n10##x,y,z,v)), \
 (I[309] = (img)(_n10##x,_n1##y,z,v)), \
 (I[333] = (img)(_n10##x,_n2##y,z,v)), \
 (I[357] = (img)(_n10##x,_n3##y,z,v)), \
 (I[381] = (img)(_n10##x,_n4##y,z,v)), \
 (I[405] = (img)(_n10##x,_n5##y,z,v)), \
 (I[429] = (img)(_n10##x,_n6##y,z,v)), \
 (I[453] = (img)(_n10##x,_n7##y,z,v)), \
 (I[477] = (img)(_n10##x,_n8##y,z,v)), \
 (I[501] = (img)(_n10##x,_n9##y,z,v)), \
 (I[525] = (img)(_n10##x,_n10##y,z,v)), \
 (I[549] = (img)(_n10##x,_n11##y,z,v)), \
 (I[573] = (img)(_n10##x,_n12##y,z,v)), \
 (I[22] = (img)(_n11##x,_p11##y,z,v)), \
 (I[46] = (img)(_n11##x,_p10##y,z,v)), \
 (I[70] = (img)(_n11##x,_p9##y,z,v)), \
 (I[94] = (img)(_n11##x,_p8##y,z,v)), \
 (I[118] = (img)(_n11##x,_p7##y,z,v)), \
 (I[142] = (img)(_n11##x,_p6##y,z,v)), \
 (I[166] = (img)(_n11##x,_p5##y,z,v)), \
 (I[190] = (img)(_n11##x,_p4##y,z,v)), \
 (I[214] = (img)(_n11##x,_p3##y,z,v)), \
 (I[238] = (img)(_n11##x,_p2##y,z,v)), \
 (I[262] = (img)(_n11##x,_p1##y,z,v)), \
 (I[286] = (img)(_n11##x,y,z,v)), \
 (I[310] = (img)(_n11##x,_n1##y,z,v)), \
 (I[334] = (img)(_n11##x,_n2##y,z,v)), \
 (I[358] = (img)(_n11##x,_n3##y,z,v)), \
 (I[382] = (img)(_n11##x,_n4##y,z,v)), \
 (I[406] = (img)(_n11##x,_n5##y,z,v)), \
 (I[430] = (img)(_n11##x,_n6##y,z,v)), \
 (I[454] = (img)(_n11##x,_n7##y,z,v)), \
 (I[478] = (img)(_n11##x,_n8##y,z,v)), \
 (I[502] = (img)(_n11##x,_n9##y,z,v)), \
 (I[526] = (img)(_n11##x,_n10##y,z,v)), \
 (I[550] = (img)(_n11##x,_n11##y,z,v)), \
 (I[574] = (img)(_n11##x,_n12##y,z,v)), \
 x+12>=(int)((img).width)?(int)((img).width)-1:x+12); \
 x<=(int)(x1) && ((_n12##x<(int)((img).width) && ( \
 (I[23] = (img)(_n12##x,_p11##y,z,v)), \
 (I[47] = (img)(_n12##x,_p10##y,z,v)), \
 (I[71] = (img)(_n12##x,_p9##y,z,v)), \
 (I[95] = (img)(_n12##x,_p8##y,z,v)), \
 (I[119] = (img)(_n12##x,_p7##y,z,v)), \
 (I[143] = (img)(_n12##x,_p6##y,z,v)), \
 (I[167] = (img)(_n12##x,_p5##y,z,v)), \
 (I[191] = (img)(_n12##x,_p4##y,z,v)), \
 (I[215] = (img)(_n12##x,_p3##y,z,v)), \
 (I[239] = (img)(_n12##x,_p2##y,z,v)), \
 (I[263] = (img)(_n12##x,_p1##y,z,v)), \
 (I[287] = (img)(_n12##x,y,z,v)), \
 (I[311] = (img)(_n12##x,_n1##y,z,v)), \
 (I[335] = (img)(_n12##x,_n2##y,z,v)), \
 (I[359] = (img)(_n12##x,_n3##y,z,v)), \
 (I[383] = (img)(_n12##x,_n4##y,z,v)), \
 (I[407] = (img)(_n12##x,_n5##y,z,v)), \
 (I[431] = (img)(_n12##x,_n6##y,z,v)), \
 (I[455] = (img)(_n12##x,_n7##y,z,v)), \
 (I[479] = (img)(_n12##x,_n8##y,z,v)), \
 (I[503] = (img)(_n12##x,_n9##y,z,v)), \
 (I[527] = (img)(_n12##x,_n10##y,z,v)), \
 (I[551] = (img)(_n12##x,_n11##y,z,v)), \
 (I[575] = (img)(_n12##x,_n12##y,z,v)),1)) || \
 _n11##x==--_n12##x || _n10##x==--_n11##x || _n9##x==--_n10##x || _n8##x==--_n9##x || _n7##x==--_n8##x || _n6##x==--_n7##x || _n5##x==--_n6##x || _n4##x==--_n5##x || _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n12##x = _n11##x = _n10##x = _n9##x = _n8##x = _n7##x = _n6##x = _n5##x = _n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], \
 I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], \
 I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], \
 I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], \
 I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], \
 I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], I[223] = I[224], I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], \
 I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], \
 I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], I[279] = I[280], I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], \
 I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], \
 I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], I[321] = I[322], I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], \
 I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], I[342] = I[343], I[343] = I[344], I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], I[351] = I[352], I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], \
 I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], I[367] = I[368], I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], I[373] = I[374], I[374] = I[375], I[375] = I[376], I[376] = I[377], I[377] = I[378], I[378] = I[379], I[379] = I[380], I[380] = I[381], I[381] = I[382], I[382] = I[383], \
 I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], I[390] = I[391], I[391] = I[392], I[392] = I[393], I[393] = I[394], I[394] = I[395], I[395] = I[396], I[396] = I[397], I[397] = I[398], I[398] = I[399], I[399] = I[400], I[400] = I[401], I[401] = I[402], I[402] = I[403], I[403] = I[404], I[404] = I[405], I[405] = I[406], I[406] = I[407], \
 I[408] = I[409], I[409] = I[410], I[410] = I[411], I[411] = I[412], I[412] = I[413], I[413] = I[414], I[414] = I[415], I[415] = I[416], I[416] = I[417], I[417] = I[418], I[418] = I[419], I[419] = I[420], I[420] = I[421], I[421] = I[422], I[422] = I[423], I[423] = I[424], I[424] = I[425], I[425] = I[426], I[426] = I[427], I[427] = I[428], I[428] = I[429], I[429] = I[430], I[430] = I[431], \
 I[432] = I[433], I[433] = I[434], I[434] = I[435], I[435] = I[436], I[436] = I[437], I[437] = I[438], I[438] = I[439], I[439] = I[440], I[440] = I[441], I[441] = I[442], I[442] = I[443], I[443] = I[444], I[444] = I[445], I[445] = I[446], I[446] = I[447], I[447] = I[448], I[448] = I[449], I[449] = I[450], I[450] = I[451], I[451] = I[452], I[452] = I[453], I[453] = I[454], I[454] = I[455], \
 I[456] = I[457], I[457] = I[458], I[458] = I[459], I[459] = I[460], I[460] = I[461], I[461] = I[462], I[462] = I[463], I[463] = I[464], I[464] = I[465], I[465] = I[466], I[466] = I[467], I[467] = I[468], I[468] = I[469], I[469] = I[470], I[470] = I[471], I[471] = I[472], I[472] = I[473], I[473] = I[474], I[474] = I[475], I[475] = I[476], I[476] = I[477], I[477] = I[478], I[478] = I[479], \
 I[480] = I[481], I[481] = I[482], I[482] = I[483], I[483] = I[484], I[484] = I[485], I[485] = I[486], I[486] = I[487], I[487] = I[488], I[488] = I[489], I[489] = I[490], I[490] = I[491], I[491] = I[492], I[492] = I[493], I[493] = I[494], I[494] = I[495], I[495] = I[496], I[496] = I[497], I[497] = I[498], I[498] = I[499], I[499] = I[500], I[500] = I[501], I[501] = I[502], I[502] = I[503], \
 I[504] = I[505], I[505] = I[506], I[506] = I[507], I[507] = I[508], I[508] = I[509], I[509] = I[510], I[510] = I[511], I[511] = I[512], I[512] = I[513], I[513] = I[514], I[514] = I[515], I[515] = I[516], I[516] = I[517], I[517] = I[518], I[518] = I[519], I[519] = I[520], I[520] = I[521], I[521] = I[522], I[522] = I[523], I[523] = I[524], I[524] = I[525], I[525] = I[526], I[526] = I[527], \
 I[528] = I[529], I[529] = I[530], I[530] = I[531], I[531] = I[532], I[532] = I[533], I[533] = I[534], I[534] = I[535], I[535] = I[536], I[536] = I[537], I[537] = I[538], I[538] = I[539], I[539] = I[540], I[540] = I[541], I[541] = I[542], I[542] = I[543], I[543] = I[544], I[544] = I[545], I[545] = I[546], I[546] = I[547], I[547] = I[548], I[548] = I[549], I[549] = I[550], I[550] = I[551], \
 I[552] = I[553], I[553] = I[554], I[554] = I[555], I[555] = I[556], I[556] = I[557], I[557] = I[558], I[558] = I[559], I[559] = I[560], I[560] = I[561], I[561] = I[562], I[562] = I[563], I[563] = I[564], I[564] = I[565], I[565] = I[566], I[566] = I[567], I[567] = I[568], I[568] = I[569], I[569] = I[570], I[570] = I[571], I[571] = I[572], I[572] = I[573], I[573] = I[574], I[574] = I[575], \
 _p11##x = _p10##x, _p10##x = _p9##x, _p9##x = _p8##x, _p8##x = _p7##x, _p7##x = _p6##x, _p6##x = _p5##x, _p5##x = _p4##x, _p4##x = _p3##x, _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x, ++_n5##x, ++_n6##x, ++_n7##x, ++_n8##x, ++_n9##x, ++_n10##x, ++_n11##x, ++_n12##x)

#define cimg_get24x24(img,x,y,z,v,I) \
 I[0] = (img)(_p11##x,_p11##y,z,v), I[1] = (img)(_p10##x,_p11##y,z,v), I[2] = (img)(_p9##x,_p11##y,z,v), I[3] = (img)(_p8##x,_p11##y,z,v), I[4] = (img)(_p7##x,_p11##y,z,v), I[5] = (img)(_p6##x,_p11##y,z,v), I[6] = (img)(_p5##x,_p11##y,z,v), I[7] = (img)(_p4##x,_p11##y,z,v), I[8] = (img)(_p3##x,_p11##y,z,v), I[9] = (img)(_p2##x,_p11##y,z,v), I[10] = (img)(_p1##x,_p11##y,z,v), I[11] = (img)(x,_p11##y,z,v), I[12] = (img)(_n1##x,_p11##y,z,v), I[13] = (img)(_n2##x,_p11##y,z,v), I[14] = (img)(_n3##x,_p11##y,z,v), I[15] = (img)(_n4##x,_p11##y,z,v), I[16] = (img)(_n5##x,_p11##y,z,v), I[17] = (img)(_n6##x,_p11##y,z,v), I[18] = (img)(_n7##x,_p11##y,z,v), I[19] = (img)(_n8##x,_p11##y,z,v), I[20] = (img)(_n9##x,_p11##y,z,v), I[21] = (img)(_n10##x,_p11##y,z,v), I[22] = (img)(_n11##x,_p11##y,z,v), I[23] = (img)(_n12##x,_p11##y,z,v), \
 I[24] = (img)(_p11##x,_p10##y,z,v), I[25] = (img)(_p10##x,_p10##y,z,v), I[26] = (img)(_p9##x,_p10##y,z,v), I[27] = (img)(_p8##x,_p10##y,z,v), I[28] = (img)(_p7##x,_p10##y,z,v), I[29] = (img)(_p6##x,_p10##y,z,v), I[30] = (img)(_p5##x,_p10##y,z,v), I[31] = (img)(_p4##x,_p10##y,z,v), I[32] = (img)(_p3##x,_p10##y,z,v), I[33] = (img)(_p2##x,_p10##y,z,v), I[34] = (img)(_p1##x,_p10##y,z,v), I[35] = (img)(x,_p10##y,z,v), I[36] = (img)(_n1##x,_p10##y,z,v), I[37] = (img)(_n2##x,_p10##y,z,v), I[38] = (img)(_n3##x,_p10##y,z,v), I[39] = (img)(_n4##x,_p10##y,z,v), I[40] = (img)(_n5##x,_p10##y,z,v), I[41] = (img)(_n6##x,_p10##y,z,v), I[42] = (img)(_n7##x,_p10##y,z,v), I[43] = (img)(_n8##x,_p10##y,z,v), I[44] = (img)(_n9##x,_p10##y,z,v), I[45] = (img)(_n10##x,_p10##y,z,v), I[46] = (img)(_n11##x,_p10##y,z,v), I[47] = (img)(_n12##x,_p10##y,z,v), \
 I[48] = (img)(_p11##x,_p9##y,z,v), I[49] = (img)(_p10##x,_p9##y,z,v), I[50] = (img)(_p9##x,_p9##y,z,v), I[51] = (img)(_p8##x,_p9##y,z,v), I[52] = (img)(_p7##x,_p9##y,z,v), I[53] = (img)(_p6##x,_p9##y,z,v), I[54] = (img)(_p5##x,_p9##y,z,v), I[55] = (img)(_p4##x,_p9##y,z,v), I[56] = (img)(_p3##x,_p9##y,z,v), I[57] = (img)(_p2##x,_p9##y,z,v), I[58] = (img)(_p1##x,_p9##y,z,v), I[59] = (img)(x,_p9##y,z,v), I[60] = (img)(_n1##x,_p9##y,z,v), I[61] = (img)(_n2##x,_p9##y,z,v), I[62] = (img)(_n3##x,_p9##y,z,v), I[63] = (img)(_n4##x,_p9##y,z,v), I[64] = (img)(_n5##x,_p9##y,z,v), I[65] = (img)(_n6##x,_p9##y,z,v), I[66] = (img)(_n7##x,_p9##y,z,v), I[67] = (img)(_n8##x,_p9##y,z,v), I[68] = (img)(_n9##x,_p9##y,z,v), I[69] = (img)(_n10##x,_p9##y,z,v), I[70] = (img)(_n11##x,_p9##y,z,v), I[71] = (img)(_n12##x,_p9##y,z,v), \
 I[72] = (img)(_p11##x,_p8##y,z,v), I[73] = (img)(_p10##x,_p8##y,z,v), I[74] = (img)(_p9##x,_p8##y,z,v), I[75] = (img)(_p8##x,_p8##y,z,v), I[76] = (img)(_p7##x,_p8##y,z,v), I[77] = (img)(_p6##x,_p8##y,z,v), I[78] = (img)(_p5##x,_p8##y,z,v), I[79] = (img)(_p4##x,_p8##y,z,v), I[80] = (img)(_p3##x,_p8##y,z,v), I[81] = (img)(_p2##x,_p8##y,z,v), I[82] = (img)(_p1##x,_p8##y,z,v), I[83] = (img)(x,_p8##y,z,v), I[84] = (img)(_n1##x,_p8##y,z,v), I[85] = (img)(_n2##x,_p8##y,z,v), I[86] = (img)(_n3##x,_p8##y,z,v), I[87] = (img)(_n4##x,_p8##y,z,v), I[88] = (img)(_n5##x,_p8##y,z,v), I[89] = (img)(_n6##x,_p8##y,z,v), I[90] = (img)(_n7##x,_p8##y,z,v), I[91] = (img)(_n8##x,_p8##y,z,v), I[92] = (img)(_n9##x,_p8##y,z,v), I[93] = (img)(_n10##x,_p8##y,z,v), I[94] = (img)(_n11##x,_p8##y,z,v), I[95] = (img)(_n12##x,_p8##y,z,v), \
 I[96] = (img)(_p11##x,_p7##y,z,v), I[97] = (img)(_p10##x,_p7##y,z,v), I[98] = (img)(_p9##x,_p7##y,z,v), I[99] = (img)(_p8##x,_p7##y,z,v), I[100] = (img)(_p7##x,_p7##y,z,v), I[101] = (img)(_p6##x,_p7##y,z,v), I[102] = (img)(_p5##x,_p7##y,z,v), I[103] = (img)(_p4##x,_p7##y,z,v), I[104] = (img)(_p3##x,_p7##y,z,v), I[105] = (img)(_p2##x,_p7##y,z,v), I[106] = (img)(_p1##x,_p7##y,z,v), I[107] = (img)(x,_p7##y,z,v), I[108] = (img)(_n1##x,_p7##y,z,v), I[109] = (img)(_n2##x,_p7##y,z,v), I[110] = (img)(_n3##x,_p7##y,z,v), I[111] = (img)(_n4##x,_p7##y,z,v), I[112] = (img)(_n5##x,_p7##y,z,v), I[113] = (img)(_n6##x,_p7##y,z,v), I[114] = (img)(_n7##x,_p7##y,z,v), I[115] = (img)(_n8##x,_p7##y,z,v), I[116] = (img)(_n9##x,_p7##y,z,v), I[117] = (img)(_n10##x,_p7##y,z,v), I[118] = (img)(_n11##x,_p7##y,z,v), I[119] = (img)(_n12##x,_p7##y,z,v), \
 I[120] = (img)(_p11##x,_p6##y,z,v), I[121] = (img)(_p10##x,_p6##y,z,v), I[122] = (img)(_p9##x,_p6##y,z,v), I[123] = (img)(_p8##x,_p6##y,z,v), I[124] = (img)(_p7##x,_p6##y,z,v), I[125] = (img)(_p6##x,_p6##y,z,v), I[126] = (img)(_p5##x,_p6##y,z,v), I[127] = (img)(_p4##x,_p6##y,z,v), I[128] = (img)(_p3##x,_p6##y,z,v), I[129] = (img)(_p2##x,_p6##y,z,v), I[130] = (img)(_p1##x,_p6##y,z,v), I[131] = (img)(x,_p6##y,z,v), I[132] = (img)(_n1##x,_p6##y,z,v), I[133] = (img)(_n2##x,_p6##y,z,v), I[134] = (img)(_n3##x,_p6##y,z,v), I[135] = (img)(_n4##x,_p6##y,z,v), I[136] = (img)(_n5##x,_p6##y,z,v), I[137] = (img)(_n6##x,_p6##y,z,v), I[138] = (img)(_n7##x,_p6##y,z,v), I[139] = (img)(_n8##x,_p6##y,z,v), I[140] = (img)(_n9##x,_p6##y,z,v), I[141] = (img)(_n10##x,_p6##y,z,v), I[142] = (img)(_n11##x,_p6##y,z,v), I[143] = (img)(_n12##x,_p6##y,z,v), \
 I[144] = (img)(_p11##x,_p5##y,z,v), I[145] = (img)(_p10##x,_p5##y,z,v), I[146] = (img)(_p9##x,_p5##y,z,v), I[147] = (img)(_p8##x,_p5##y,z,v), I[148] = (img)(_p7##x,_p5##y,z,v), I[149] = (img)(_p6##x,_p5##y,z,v), I[150] = (img)(_p5##x,_p5##y,z,v), I[151] = (img)(_p4##x,_p5##y,z,v), I[152] = (img)(_p3##x,_p5##y,z,v), I[153] = (img)(_p2##x,_p5##y,z,v), I[154] = (img)(_p1##x,_p5##y,z,v), I[155] = (img)(x,_p5##y,z,v), I[156] = (img)(_n1##x,_p5##y,z,v), I[157] = (img)(_n2##x,_p5##y,z,v), I[158] = (img)(_n3##x,_p5##y,z,v), I[159] = (img)(_n4##x,_p5##y,z,v), I[160] = (img)(_n5##x,_p5##y,z,v), I[161] = (img)(_n6##x,_p5##y,z,v), I[162] = (img)(_n7##x,_p5##y,z,v), I[163] = (img)(_n8##x,_p5##y,z,v), I[164] = (img)(_n9##x,_p5##y,z,v), I[165] = (img)(_n10##x,_p5##y,z,v), I[166] = (img)(_n11##x,_p5##y,z,v), I[167] = (img)(_n12##x,_p5##y,z,v), \
 I[168] = (img)(_p11##x,_p4##y,z,v), I[169] = (img)(_p10##x,_p4##y,z,v), I[170] = (img)(_p9##x,_p4##y,z,v), I[171] = (img)(_p8##x,_p4##y,z,v), I[172] = (img)(_p7##x,_p4##y,z,v), I[173] = (img)(_p6##x,_p4##y,z,v), I[174] = (img)(_p5##x,_p4##y,z,v), I[175] = (img)(_p4##x,_p4##y,z,v), I[176] = (img)(_p3##x,_p4##y,z,v), I[177] = (img)(_p2##x,_p4##y,z,v), I[178] = (img)(_p1##x,_p4##y,z,v), I[179] = (img)(x,_p4##y,z,v), I[180] = (img)(_n1##x,_p4##y,z,v), I[181] = (img)(_n2##x,_p4##y,z,v), I[182] = (img)(_n3##x,_p4##y,z,v), I[183] = (img)(_n4##x,_p4##y,z,v), I[184] = (img)(_n5##x,_p4##y,z,v), I[185] = (img)(_n6##x,_p4##y,z,v), I[186] = (img)(_n7##x,_p4##y,z,v), I[187] = (img)(_n8##x,_p4##y,z,v), I[188] = (img)(_n9##x,_p4##y,z,v), I[189] = (img)(_n10##x,_p4##y,z,v), I[190] = (img)(_n11##x,_p4##y,z,v), I[191] = (img)(_n12##x,_p4##y,z,v), \
 I[192] = (img)(_p11##x,_p3##y,z,v), I[193] = (img)(_p10##x,_p3##y,z,v), I[194] = (img)(_p9##x,_p3##y,z,v), I[195] = (img)(_p8##x,_p3##y,z,v), I[196] = (img)(_p7##x,_p3##y,z,v), I[197] = (img)(_p6##x,_p3##y,z,v), I[198] = (img)(_p5##x,_p3##y,z,v), I[199] = (img)(_p4##x,_p3##y,z,v), I[200] = (img)(_p3##x,_p3##y,z,v), I[201] = (img)(_p2##x,_p3##y,z,v), I[202] = (img)(_p1##x,_p3##y,z,v), I[203] = (img)(x,_p3##y,z,v), I[204] = (img)(_n1##x,_p3##y,z,v), I[205] = (img)(_n2##x,_p3##y,z,v), I[206] = (img)(_n3##x,_p3##y,z,v), I[207] = (img)(_n4##x,_p3##y,z,v), I[208] = (img)(_n5##x,_p3##y,z,v), I[209] = (img)(_n6##x,_p3##y,z,v), I[210] = (img)(_n7##x,_p3##y,z,v), I[211] = (img)(_n8##x,_p3##y,z,v), I[212] = (img)(_n9##x,_p3##y,z,v), I[213] = (img)(_n10##x,_p3##y,z,v), I[214] = (img)(_n11##x,_p3##y,z,v), I[215] = (img)(_n12##x,_p3##y,z,v), \
 I[216] = (img)(_p11##x,_p2##y,z,v), I[217] = (img)(_p10##x,_p2##y,z,v), I[218] = (img)(_p9##x,_p2##y,z,v), I[219] = (img)(_p8##x,_p2##y,z,v), I[220] = (img)(_p7##x,_p2##y,z,v), I[221] = (img)(_p6##x,_p2##y,z,v), I[222] = (img)(_p5##x,_p2##y,z,v), I[223] = (img)(_p4##x,_p2##y,z,v), I[224] = (img)(_p3##x,_p2##y,z,v), I[225] = (img)(_p2##x,_p2##y,z,v), I[226] = (img)(_p1##x,_p2##y,z,v), I[227] = (img)(x,_p2##y,z,v), I[228] = (img)(_n1##x,_p2##y,z,v), I[229] = (img)(_n2##x,_p2##y,z,v), I[230] = (img)(_n3##x,_p2##y,z,v), I[231] = (img)(_n4##x,_p2##y,z,v), I[232] = (img)(_n5##x,_p2##y,z,v), I[233] = (img)(_n6##x,_p2##y,z,v), I[234] = (img)(_n7##x,_p2##y,z,v), I[235] = (img)(_n8##x,_p2##y,z,v), I[236] = (img)(_n9##x,_p2##y,z,v), I[237] = (img)(_n10##x,_p2##y,z,v), I[238] = (img)(_n11##x,_p2##y,z,v), I[239] = (img)(_n12##x,_p2##y,z,v), \
 I[240] = (img)(_p11##x,_p1##y,z,v), I[241] = (img)(_p10##x,_p1##y,z,v), I[242] = (img)(_p9##x,_p1##y,z,v), I[243] = (img)(_p8##x,_p1##y,z,v), I[244] = (img)(_p7##x,_p1##y,z,v), I[245] = (img)(_p6##x,_p1##y,z,v), I[246] = (img)(_p5##x,_p1##y,z,v), I[247] = (img)(_p4##x,_p1##y,z,v), I[248] = (img)(_p3##x,_p1##y,z,v), I[249] = (img)(_p2##x,_p1##y,z,v), I[250] = (img)(_p1##x,_p1##y,z,v), I[251] = (img)(x,_p1##y,z,v), I[252] = (img)(_n1##x,_p1##y,z,v), I[253] = (img)(_n2##x,_p1##y,z,v), I[254] = (img)(_n3##x,_p1##y,z,v), I[255] = (img)(_n4##x,_p1##y,z,v), I[256] = (img)(_n5##x,_p1##y,z,v), I[257] = (img)(_n6##x,_p1##y,z,v), I[258] = (img)(_n7##x,_p1##y,z,v), I[259] = (img)(_n8##x,_p1##y,z,v), I[260] = (img)(_n9##x,_p1##y,z,v), I[261] = (img)(_n10##x,_p1##y,z,v), I[262] = (img)(_n11##x,_p1##y,z,v), I[263] = (img)(_n12##x,_p1##y,z,v), \
 I[264] = (img)(_p11##x,y,z,v), I[265] = (img)(_p10##x,y,z,v), I[266] = (img)(_p9##x,y,z,v), I[267] = (img)(_p8##x,y,z,v), I[268] = (img)(_p7##x,y,z,v), I[269] = (img)(_p6##x,y,z,v), I[270] = (img)(_p5##x,y,z,v), I[271] = (img)(_p4##x,y,z,v), I[272] = (img)(_p3##x,y,z,v), I[273] = (img)(_p2##x,y,z,v), I[274] = (img)(_p1##x,y,z,v), I[275] = (img)(x,y,z,v), I[276] = (img)(_n1##x,y,z,v), I[277] = (img)(_n2##x,y,z,v), I[278] = (img)(_n3##x,y,z,v), I[279] = (img)(_n4##x,y,z,v), I[280] = (img)(_n5##x,y,z,v), I[281] = (img)(_n6##x,y,z,v), I[282] = (img)(_n7##x,y,z,v), I[283] = (img)(_n8##x,y,z,v), I[284] = (img)(_n9##x,y,z,v), I[285] = (img)(_n10##x,y,z,v), I[286] = (img)(_n11##x,y,z,v), I[287] = (img)(_n12##x,y,z,v), \
 I[288] = (img)(_p11##x,_n1##y,z,v), I[289] = (img)(_p10##x,_n1##y,z,v), I[290] = (img)(_p9##x,_n1##y,z,v), I[291] = (img)(_p8##x,_n1##y,z,v), I[292] = (img)(_p7##x,_n1##y,z,v), I[293] = (img)(_p6##x,_n1##y,z,v), I[294] = (img)(_p5##x,_n1##y,z,v), I[295] = (img)(_p4##x,_n1##y,z,v), I[296] = (img)(_p3##x,_n1##y,z,v), I[297] = (img)(_p2##x,_n1##y,z,v), I[298] = (img)(_p1##x,_n1##y,z,v), I[299] = (img)(x,_n1##y,z,v), I[300] = (img)(_n1##x,_n1##y,z,v), I[301] = (img)(_n2##x,_n1##y,z,v), I[302] = (img)(_n3##x,_n1##y,z,v), I[303] = (img)(_n4##x,_n1##y,z,v), I[304] = (img)(_n5##x,_n1##y,z,v), I[305] = (img)(_n6##x,_n1##y,z,v), I[306] = (img)(_n7##x,_n1##y,z,v), I[307] = (img)(_n8##x,_n1##y,z,v), I[308] = (img)(_n9##x,_n1##y,z,v), I[309] = (img)(_n10##x,_n1##y,z,v), I[310] = (img)(_n11##x,_n1##y,z,v), I[311] = (img)(_n12##x,_n1##y,z,v), \
 I[312] = (img)(_p11##x,_n2##y,z,v), I[313] = (img)(_p10##x,_n2##y,z,v), I[314] = (img)(_p9##x,_n2##y,z,v), I[315] = (img)(_p8##x,_n2##y,z,v), I[316] = (img)(_p7##x,_n2##y,z,v), I[317] = (img)(_p6##x,_n2##y,z,v), I[318] = (img)(_p5##x,_n2##y,z,v), I[319] = (img)(_p4##x,_n2##y,z,v), I[320] = (img)(_p3##x,_n2##y,z,v), I[321] = (img)(_p2##x,_n2##y,z,v), I[322] = (img)(_p1##x,_n2##y,z,v), I[323] = (img)(x,_n2##y,z,v), I[324] = (img)(_n1##x,_n2##y,z,v), I[325] = (img)(_n2##x,_n2##y,z,v), I[326] = (img)(_n3##x,_n2##y,z,v), I[327] = (img)(_n4##x,_n2##y,z,v), I[328] = (img)(_n5##x,_n2##y,z,v), I[329] = (img)(_n6##x,_n2##y,z,v), I[330] = (img)(_n7##x,_n2##y,z,v), I[331] = (img)(_n8##x,_n2##y,z,v), I[332] = (img)(_n9##x,_n2##y,z,v), I[333] = (img)(_n10##x,_n2##y,z,v), I[334] = (img)(_n11##x,_n2##y,z,v), I[335] = (img)(_n12##x,_n2##y,z,v), \
 I[336] = (img)(_p11##x,_n3##y,z,v), I[337] = (img)(_p10##x,_n3##y,z,v), I[338] = (img)(_p9##x,_n3##y,z,v), I[339] = (img)(_p8##x,_n3##y,z,v), I[340] = (img)(_p7##x,_n3##y,z,v), I[341] = (img)(_p6##x,_n3##y,z,v), I[342] = (img)(_p5##x,_n3##y,z,v), I[343] = (img)(_p4##x,_n3##y,z,v), I[344] = (img)(_p3##x,_n3##y,z,v), I[345] = (img)(_p2##x,_n3##y,z,v), I[346] = (img)(_p1##x,_n3##y,z,v), I[347] = (img)(x,_n3##y,z,v), I[348] = (img)(_n1##x,_n3##y,z,v), I[349] = (img)(_n2##x,_n3##y,z,v), I[350] = (img)(_n3##x,_n3##y,z,v), I[351] = (img)(_n4##x,_n3##y,z,v), I[352] = (img)(_n5##x,_n3##y,z,v), I[353] = (img)(_n6##x,_n3##y,z,v), I[354] = (img)(_n7##x,_n3##y,z,v), I[355] = (img)(_n8##x,_n3##y,z,v), I[356] = (img)(_n9##x,_n3##y,z,v), I[357] = (img)(_n10##x,_n3##y,z,v), I[358] = (img)(_n11##x,_n3##y,z,v), I[359] = (img)(_n12##x,_n3##y,z,v), \
 I[360] = (img)(_p11##x,_n4##y,z,v), I[361] = (img)(_p10##x,_n4##y,z,v), I[362] = (img)(_p9##x,_n4##y,z,v), I[363] = (img)(_p8##x,_n4##y,z,v), I[364] = (img)(_p7##x,_n4##y,z,v), I[365] = (img)(_p6##x,_n4##y,z,v), I[366] = (img)(_p5##x,_n4##y,z,v), I[367] = (img)(_p4##x,_n4##y,z,v), I[368] = (img)(_p3##x,_n4##y,z,v), I[369] = (img)(_p2##x,_n4##y,z,v), I[370] = (img)(_p1##x,_n4##y,z,v), I[371] = (img)(x,_n4##y,z,v), I[372] = (img)(_n1##x,_n4##y,z,v), I[373] = (img)(_n2##x,_n4##y,z,v), I[374] = (img)(_n3##x,_n4##y,z,v), I[375] = (img)(_n4##x,_n4##y,z,v), I[376] = (img)(_n5##x,_n4##y,z,v), I[377] = (img)(_n6##x,_n4##y,z,v), I[378] = (img)(_n7##x,_n4##y,z,v), I[379] = (img)(_n8##x,_n4##y,z,v), I[380] = (img)(_n9##x,_n4##y,z,v), I[381] = (img)(_n10##x,_n4##y,z,v), I[382] = (img)(_n11##x,_n4##y,z,v), I[383] = (img)(_n12##x,_n4##y,z,v), \
 I[384] = (img)(_p11##x,_n5##y,z,v), I[385] = (img)(_p10##x,_n5##y,z,v), I[386] = (img)(_p9##x,_n5##y,z,v), I[387] = (img)(_p8##x,_n5##y,z,v), I[388] = (img)(_p7##x,_n5##y,z,v), I[389] = (img)(_p6##x,_n5##y,z,v), I[390] = (img)(_p5##x,_n5##y,z,v), I[391] = (img)(_p4##x,_n5##y,z,v), I[392] = (img)(_p3##x,_n5##y,z,v), I[393] = (img)(_p2##x,_n5##y,z,v), I[394] = (img)(_p1##x,_n5##y,z,v), I[395] = (img)(x,_n5##y,z,v), I[396] = (img)(_n1##x,_n5##y,z,v), I[397] = (img)(_n2##x,_n5##y,z,v), I[398] = (img)(_n3##x,_n5##y,z,v), I[399] = (img)(_n4##x,_n5##y,z,v), I[400] = (img)(_n5##x,_n5##y,z,v), I[401] = (img)(_n6##x,_n5##y,z,v), I[402] = (img)(_n7##x,_n5##y,z,v), I[403] = (img)(_n8##x,_n5##y,z,v), I[404] = (img)(_n9##x,_n5##y,z,v), I[405] = (img)(_n10##x,_n5##y,z,v), I[406] = (img)(_n11##x,_n5##y,z,v), I[407] = (img)(_n12##x,_n5##y,z,v), \
 I[408] = (img)(_p11##x,_n6##y,z,v), I[409] = (img)(_p10##x,_n6##y,z,v), I[410] = (img)(_p9##x,_n6##y,z,v), I[411] = (img)(_p8##x,_n6##y,z,v), I[412] = (img)(_p7##x,_n6##y,z,v), I[413] = (img)(_p6##x,_n6##y,z,v), I[414] = (img)(_p5##x,_n6##y,z,v), I[415] = (img)(_p4##x,_n6##y,z,v), I[416] = (img)(_p3##x,_n6##y,z,v), I[417] = (img)(_p2##x,_n6##y,z,v), I[418] = (img)(_p1##x,_n6##y,z,v), I[419] = (img)(x,_n6##y,z,v), I[420] = (img)(_n1##x,_n6##y,z,v), I[421] = (img)(_n2##x,_n6##y,z,v), I[422] = (img)(_n3##x,_n6##y,z,v), I[423] = (img)(_n4##x,_n6##y,z,v), I[424] = (img)(_n5##x,_n6##y,z,v), I[425] = (img)(_n6##x,_n6##y,z,v), I[426] = (img)(_n7##x,_n6##y,z,v), I[427] = (img)(_n8##x,_n6##y,z,v), I[428] = (img)(_n9##x,_n6##y,z,v), I[429] = (img)(_n10##x,_n6##y,z,v), I[430] = (img)(_n11##x,_n6##y,z,v), I[431] = (img)(_n12##x,_n6##y,z,v), \
 I[432] = (img)(_p11##x,_n7##y,z,v), I[433] = (img)(_p10##x,_n7##y,z,v), I[434] = (img)(_p9##x,_n7##y,z,v), I[435] = (img)(_p8##x,_n7##y,z,v), I[436] = (img)(_p7##x,_n7##y,z,v), I[437] = (img)(_p6##x,_n7##y,z,v), I[438] = (img)(_p5##x,_n7##y,z,v), I[439] = (img)(_p4##x,_n7##y,z,v), I[440] = (img)(_p3##x,_n7##y,z,v), I[441] = (img)(_p2##x,_n7##y,z,v), I[442] = (img)(_p1##x,_n7##y,z,v), I[443] = (img)(x,_n7##y,z,v), I[444] = (img)(_n1##x,_n7##y,z,v), I[445] = (img)(_n2##x,_n7##y,z,v), I[446] = (img)(_n3##x,_n7##y,z,v), I[447] = (img)(_n4##x,_n7##y,z,v), I[448] = (img)(_n5##x,_n7##y,z,v), I[449] = (img)(_n6##x,_n7##y,z,v), I[450] = (img)(_n7##x,_n7##y,z,v), I[451] = (img)(_n8##x,_n7##y,z,v), I[452] = (img)(_n9##x,_n7##y,z,v), I[453] = (img)(_n10##x,_n7##y,z,v), I[454] = (img)(_n11##x,_n7##y,z,v), I[455] = (img)(_n12##x,_n7##y,z,v), \
 I[456] = (img)(_p11##x,_n8##y,z,v), I[457] = (img)(_p10##x,_n8##y,z,v), I[458] = (img)(_p9##x,_n8##y,z,v), I[459] = (img)(_p8##x,_n8##y,z,v), I[460] = (img)(_p7##x,_n8##y,z,v), I[461] = (img)(_p6##x,_n8##y,z,v), I[462] = (img)(_p5##x,_n8##y,z,v), I[463] = (img)(_p4##x,_n8##y,z,v), I[464] = (img)(_p3##x,_n8##y,z,v), I[465] = (img)(_p2##x,_n8##y,z,v), I[466] = (img)(_p1##x,_n8##y,z,v), I[467] = (img)(x,_n8##y,z,v), I[468] = (img)(_n1##x,_n8##y,z,v), I[469] = (img)(_n2##x,_n8##y,z,v), I[470] = (img)(_n3##x,_n8##y,z,v), I[471] = (img)(_n4##x,_n8##y,z,v), I[472] = (img)(_n5##x,_n8##y,z,v), I[473] = (img)(_n6##x,_n8##y,z,v), I[474] = (img)(_n7##x,_n8##y,z,v), I[475] = (img)(_n8##x,_n8##y,z,v), I[476] = (img)(_n9##x,_n8##y,z,v), I[477] = (img)(_n10##x,_n8##y,z,v), I[478] = (img)(_n11##x,_n8##y,z,v), I[479] = (img)(_n12##x,_n8##y,z,v), \
 I[480] = (img)(_p11##x,_n9##y,z,v), I[481] = (img)(_p10##x,_n9##y,z,v), I[482] = (img)(_p9##x,_n9##y,z,v), I[483] = (img)(_p8##x,_n9##y,z,v), I[484] = (img)(_p7##x,_n9##y,z,v), I[485] = (img)(_p6##x,_n9##y,z,v), I[486] = (img)(_p5##x,_n9##y,z,v), I[487] = (img)(_p4##x,_n9##y,z,v), I[488] = (img)(_p3##x,_n9##y,z,v), I[489] = (img)(_p2##x,_n9##y,z,v), I[490] = (img)(_p1##x,_n9##y,z,v), I[491] = (img)(x,_n9##y,z,v), I[492] = (img)(_n1##x,_n9##y,z,v), I[493] = (img)(_n2##x,_n9##y,z,v), I[494] = (img)(_n3##x,_n9##y,z,v), I[495] = (img)(_n4##x,_n9##y,z,v), I[496] = (img)(_n5##x,_n9##y,z,v), I[497] = (img)(_n6##x,_n9##y,z,v), I[498] = (img)(_n7##x,_n9##y,z,v), I[499] = (img)(_n8##x,_n9##y,z,v), I[500] = (img)(_n9##x,_n9##y,z,v), I[501] = (img)(_n10##x,_n9##y,z,v), I[502] = (img)(_n11##x,_n9##y,z,v), I[503] = (img)(_n12##x,_n9##y,z,v), \
 I[504] = (img)(_p11##x,_n10##y,z,v), I[505] = (img)(_p10##x,_n10##y,z,v), I[506] = (img)(_p9##x,_n10##y,z,v), I[507] = (img)(_p8##x,_n10##y,z,v), I[508] = (img)(_p7##x,_n10##y,z,v), I[509] = (img)(_p6##x,_n10##y,z,v), I[510] = (img)(_p5##x,_n10##y,z,v), I[511] = (img)(_p4##x,_n10##y,z,v), I[512] = (img)(_p3##x,_n10##y,z,v), I[513] = (img)(_p2##x,_n10##y,z,v), I[514] = (img)(_p1##x,_n10##y,z,v), I[515] = (img)(x,_n10##y,z,v), I[516] = (img)(_n1##x,_n10##y,z,v), I[517] = (img)(_n2##x,_n10##y,z,v), I[518] = (img)(_n3##x,_n10##y,z,v), I[519] = (img)(_n4##x,_n10##y,z,v), I[520] = (img)(_n5##x,_n10##y,z,v), I[521] = (img)(_n6##x,_n10##y,z,v), I[522] = (img)(_n7##x,_n10##y,z,v), I[523] = (img)(_n8##x,_n10##y,z,v), I[524] = (img)(_n9##x,_n10##y,z,v), I[525] = (img)(_n10##x,_n10##y,z,v), I[526] = (img)(_n11##x,_n10##y,z,v), I[527] = (img)(_n12##x,_n10##y,z,v), \
 I[528] = (img)(_p11##x,_n11##y,z,v), I[529] = (img)(_p10##x,_n11##y,z,v), I[530] = (img)(_p9##x,_n11##y,z,v), I[531] = (img)(_p8##x,_n11##y,z,v), I[532] = (img)(_p7##x,_n11##y,z,v), I[533] = (img)(_p6##x,_n11##y,z,v), I[534] = (img)(_p5##x,_n11##y,z,v), I[535] = (img)(_p4##x,_n11##y,z,v), I[536] = (img)(_p3##x,_n11##y,z,v), I[537] = (img)(_p2##x,_n11##y,z,v), I[538] = (img)(_p1##x,_n11##y,z,v), I[539] = (img)(x,_n11##y,z,v), I[540] = (img)(_n1##x,_n11##y,z,v), I[541] = (img)(_n2##x,_n11##y,z,v), I[542] = (img)(_n3##x,_n11##y,z,v), I[543] = (img)(_n4##x,_n11##y,z,v), I[544] = (img)(_n5##x,_n11##y,z,v), I[545] = (img)(_n6##x,_n11##y,z,v), I[546] = (img)(_n7##x,_n11##y,z,v), I[547] = (img)(_n8##x,_n11##y,z,v), I[548] = (img)(_n9##x,_n11##y,z,v), I[549] = (img)(_n10##x,_n11##y,z,v), I[550] = (img)(_n11##x,_n11##y,z,v), I[551] = (img)(_n12##x,_n11##y,z,v), \
 I[552] = (img)(_p11##x,_n12##y,z,v), I[553] = (img)(_p10##x,_n12##y,z,v), I[554] = (img)(_p9##x,_n12##y,z,v), I[555] = (img)(_p8##x,_n12##y,z,v), I[556] = (img)(_p7##x,_n12##y,z,v), I[557] = (img)(_p6##x,_n12##y,z,v), I[558] = (img)(_p5##x,_n12##y,z,v), I[559] = (img)(_p4##x,_n12##y,z,v), I[560] = (img)(_p3##x,_n12##y,z,v), I[561] = (img)(_p2##x,_n12##y,z,v), I[562] = (img)(_p1##x,_n12##y,z,v), I[563] = (img)(x,_n12##y,z,v), I[564] = (img)(_n1##x,_n12##y,z,v), I[565] = (img)(_n2##x,_n12##y,z,v), I[566] = (img)(_n3##x,_n12##y,z,v), I[567] = (img)(_n4##x,_n12##y,z,v), I[568] = (img)(_n5##x,_n12##y,z,v), I[569] = (img)(_n6##x,_n12##y,z,v), I[570] = (img)(_n7##x,_n12##y,z,v), I[571] = (img)(_n8##x,_n12##y,z,v), I[572] = (img)(_n9##x,_n12##y,z,v), I[573] = (img)(_n10##x,_n12##y,z,v), I[574] = (img)(_n11##x,_n12##y,z,v), I[575] = (img)(_n12##x,_n12##y,z,v);

// Define 4x4x4 loop macros for CImg
//-------------------------------------
#define cimg_for_in4(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2; \
 i<=(int)(i1) && (_n2##i<(int)(bound) || _n1##i==--_n2##i || \
 i==(_n2##i = --_n1##i)); \
 _p1##i = i++, \
 ++_n1##i, ++_n2##i)

#define cimg_for_in4X(img,x0,x1,x) cimg_for_in4((img).width,x0,x1,x)
#define cimg_for_in4Y(img,y0,y1,y) cimg_for_in4((img).height,y0,y1,y)
#define cimg_for_in4Z(img,z0,z1,z) cimg_for_in4((img).depth,z0,z1,z)
#define cimg_for_in4V(img,v0,v1,v) cimg_for_in4((img).dim,v0,v1,v)
#define cimg_for_in4XY(img,x0,y0,x1,y1,x,y) cimg_for_in4Y(img,y0,y1,y) cimg_for_in4X(img,x0,x1,x)
#define cimg_for_in4XZ(img,x0,z0,x1,z1,x,z) cimg_for_in4Z(img,z0,z1,z) cimg_for_in4X(img,x0,x1,x)
#define cimg_for_in4XV(img,x0,v0,x1,v1,x,v) cimg_for_in4V(img,v0,v1,v) cimg_for_in4X(img,x0,x1,x)
#define cimg_for_in4YZ(img,y0,z0,y1,z1,y,z) cimg_for_in4Z(img,z0,z1,z) cimg_for_in4Y(img,y0,y1,y)
#define cimg_for_in4YV(img,y0,v0,y1,v1,y,v) cimg_for_in4V(img,v0,v1,v) cimg_for_in4Y(img,y0,y1,y)
#define cimg_for_in4ZV(img,z0,v0,z1,v1,z,v) cimg_for_in4V(img,v0,v1,v) cimg_for_in4Z(img,z0,z1,z)
#define cimg_for_in4XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in4Z(img,z0,z1,z) cimg_for_in4XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in4XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in4V(img,v0,v1,v) cimg_for_in4XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in4YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in4V(img,v0,v1,v) cimg_for_in4YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in4XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in4V(img,v0,v1,v) cimg_for_in4XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for4x4x4(img,x,y,z,v,I) \
 cimg_for4((img).depth,z) cimg_for4((img).height,y) for (int x = 0, \
 _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = (int)( \
 (I[0] = I[1] = (img)(0,_p1##y,_p1##z,v)), \
 (I[4] = I[5] = (img)(0,y,_p1##z,v)), \
 (I[8] = I[9] = (img)(0,_n1##y,_p1##z,v)), \
 (I[12] = I[13] = (img)(0,_n2##y,_p1##z,v)), \
 (I[16] = I[17] = (img)(0,_p1##y,z,v)), \
 (I[20] = I[21] = (img)(0,y,z,v)), \
 (I[24] = I[25] = (img)(0,_n1##y,z,v)), \
 (I[28] = I[29] = (img)(0,_n2##y,z,v)), \
 (I[32] = I[33] = (img)(0,_p1##y,_n1##z,v)), \
 (I[36] = I[37] = (img)(0,y,_n1##z,v)), \
 (I[40] = I[41] = (img)(0,_n1##y,_n1##z,v)), \
 (I[44] = I[45] = (img)(0,_n2##y,_n1##z,v)), \
 (I[48] = I[49] = (img)(0,_p1##y,_n2##z,v)), \
 (I[52] = I[53] = (img)(0,y,_n2##z,v)), \
 (I[56] = I[57] = (img)(0,_n1##y,_n2##z,v)), \
 (I[60] = I[61] = (img)(0,_n2##y,_n2##z,v)), \
 (I[2] = (img)(_n1##x,_p1##y,_p1##z,v)), \
 (I[6] = (img)(_n1##x,y,_p1##z,v)), \
 (I[10] = (img)(_n1##x,_n1##y,_p1##z,v)), \
 (I[14] = (img)(_n1##x,_n2##y,_p1##z,v)), \
 (I[18] = (img)(_n1##x,_p1##y,z,v)), \
 (I[22] = (img)(_n1##x,y,z,v)), \
 (I[26] = (img)(_n1##x,_n1##y,z,v)), \
 (I[30] = (img)(_n1##x,_n2##y,z,v)), \
 (I[34] = (img)(_n1##x,_p1##y,_n1##z,v)), \
 (I[38] = (img)(_n1##x,y,_n1##z,v)), \
 (I[42] = (img)(_n1##x,_n1##y,_n1##z,v)), \
 (I[46] = (img)(_n1##x,_n2##y,_n1##z,v)), \
 (I[50] = (img)(_n1##x,_p1##y,_n2##z,v)), \
 (I[54] = (img)(_n1##x,y,_n2##z,v)), \
 (I[58] = (img)(_n1##x,_n1##y,_n2##z,v)), \
 (I[62] = (img)(_n1##x,_n2##y,_n2##z,v)), \
 2>=((img).width)?(int)((img).width)-1:2); \
 (_n2##x<(int)((img).width) && ( \
 (I[3] = (img)(_n2##x,_p1##y,_p1##z,v)), \
 (I[7] = (img)(_n2##x,y,_p1##z,v)), \
 (I[11] = (img)(_n2##x,_n1##y,_p1##z,v)), \
 (I[15] = (img)(_n2##x,_n2##y,_p1##z,v)), \
 (I[19] = (img)(_n2##x,_p1##y,z,v)), \
 (I[23] = (img)(_n2##x,y,z,v)), \
 (I[27] = (img)(_n2##x,_n1##y,z,v)), \
 (I[31] = (img)(_n2##x,_n2##y,z,v)), \
 (I[35] = (img)(_n2##x,_p1##y,_n1##z,v)), \
 (I[39] = (img)(_n2##x,y,_n1##z,v)), \
 (I[43] = (img)(_n2##x,_n1##y,_n1##z,v)), \
 (I[47] = (img)(_n2##x,_n2##y,_n1##z,v)), \
 (I[51] = (img)(_n2##x,_p1##y,_n2##z,v)), \
 (I[55] = (img)(_n2##x,y,_n2##z,v)), \
 (I[59] = (img)(_n2##x,_n1##y,_n2##z,v)), \
 (I[63] = (img)(_n2##x,_n2##y,_n2##z,v)),1)) || \
 _n1##x==--_n2##x || x==(_n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], \
 I[4] = I[5], I[5] = I[6], I[6] = I[7], \
 I[8] = I[9], I[9] = I[10], I[10] = I[11], \
 I[12] = I[13], I[13] = I[14], I[14] = I[15], \
 I[16] = I[17], I[17] = I[18], I[18] = I[19], \
 I[20] = I[21], I[21] = I[22], I[22] = I[23], \
 I[24] = I[25], I[25] = I[26], I[26] = I[27], \
 I[28] = I[29], I[29] = I[30], I[30] = I[31], \
 I[32] = I[33], I[33] = I[34], I[34] = I[35], \
 I[36] = I[37], I[37] = I[38], I[38] = I[39], \
 I[40] = I[41], I[41] = I[42], I[42] = I[43], \
 I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], \
 I[52] = I[53], I[53] = I[54], I[54] = I[55], \
 I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], \
 _p1##x = x++, ++_n1##x, ++_n2##x)

#define cimg_for_in4x4x4(img,x0,y0,z0,x1,y1,z1,x,y,z,v,I) \
 cimg_for_in4((img).depth,z0,z1,z) cimg_for_in4((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = (int)( \
 (I[0] = (img)(_p1##x,_p1##y,_p1##z,v)), \
 (I[4] = (img)(_p1##x,y,_p1##z,v)), \
 (I[8] = (img)(_p1##x,_n1##y,_p1##z,v)), \
 (I[12] = (img)(_p1##x,_n2##y,_p1##z,v)), \
 (I[16] = (img)(_p1##x,_p1##y,z,v)), \
 (I[20] = (img)(_p1##x,y,z,v)), \
 (I[24] = (img)(_p1##x,_n1##y,z,v)), \
 (I[28] = (img)(_p1##x,_n2##y,z,v)), \
 (I[32] = (img)(_p1##x,_p1##y,_n1##z,v)), \
 (I[36] = (img)(_p1##x,y,_n1##z,v)), \
 (I[40] = (img)(_p1##x,_n1##y,_n1##z,v)), \
 (I[44] = (img)(_p1##x,_n2##y,_n1##z,v)), \
 (I[48] = (img)(_p1##x,_p1##y,_n2##z,v)), \
 (I[52] = (img)(_p1##x,y,_n2##z,v)), \
 (I[56] = (img)(_p1##x,_n1##y,_n2##z,v)), \
 (I[60] = (img)(_p1##x,_n2##y,_n2##z,v)), \
 (I[1] = (img)(x,_p1##y,_p1##z,v)), \
 (I[5] = (img)(x,y,_p1##z,v)), \
 (I[9] = (img)(x,_n1##y,_p1##z,v)), \
 (I[13] = (img)(x,_n2##y,_p1##z,v)), \
 (I[17] = (img)(x,_p1##y,z,v)), \
 (I[21] = (img)(x,y,z,v)), \
 (I[25] = (img)(x,_n1##y,z,v)), \
 (I[29] = (img)(x,_n2##y,z,v)), \
 (I[33] = (img)(x,_p1##y,_n1##z,v)), \
 (I[37] = (img)(x,y,_n1##z,v)), \
 (I[41] = (img)(x,_n1##y,_n1##z,v)), \
 (I[45] = (img)(x,_n2##y,_n1##z,v)), \
 (I[49] = (img)(x,_p1##y,_n2##z,v)), \
 (I[53] = (img)(x,y,_n2##z,v)), \
 (I[57] = (img)(x,_n1##y,_n2##z,v)), \
 (I[61] = (img)(x,_n2##y,_n2##z,v)), \
 (I[2] = (img)(_n1##x,_p1##y,_p1##z,v)), \
 (I[6] = (img)(_n1##x,y,_p1##z,v)), \
 (I[10] = (img)(_n1##x,_n1##y,_p1##z,v)), \
 (I[14] = (img)(_n1##x,_n2##y,_p1##z,v)), \
 (I[18] = (img)(_n1##x,_p1##y,z,v)), \
 (I[22] = (img)(_n1##x,y,z,v)), \
 (I[26] = (img)(_n1##x,_n1##y,z,v)), \
 (I[30] = (img)(_n1##x,_n2##y,z,v)), \
 (I[34] = (img)(_n1##x,_p1##y,_n1##z,v)), \
 (I[38] = (img)(_n1##x,y,_n1##z,v)), \
 (I[42] = (img)(_n1##x,_n1##y,_n1##z,v)), \
 (I[46] = (img)(_n1##x,_n2##y,_n1##z,v)), \
 (I[50] = (img)(_n1##x,_p1##y,_n2##z,v)), \
 (I[54] = (img)(_n1##x,y,_n2##z,v)), \
 (I[58] = (img)(_n1##x,_n1##y,_n2##z,v)), \
 (I[62] = (img)(_n1##x,_n2##y,_n2##z,v)), \
 x+2>=(int)((img).width)?(int)((img).width)-1:x+2); \
 x<=(int)(x1) && ((_n2##x<(int)((img).width) && ( \
 (I[3] = (img)(_n2##x,_p1##y,_p1##z,v)), \
 (I[7] = (img)(_n2##x,y,_p1##z,v)), \
 (I[11] = (img)(_n2##x,_n1##y,_p1##z,v)), \
 (I[15] = (img)(_n2##x,_n2##y,_p1##z,v)), \
 (I[19] = (img)(_n2##x,_p1##y,z,v)), \
 (I[23] = (img)(_n2##x,y,z,v)), \
 (I[27] = (img)(_n2##x,_n1##y,z,v)), \
 (I[31] = (img)(_n2##x,_n2##y,z,v)), \
 (I[35] = (img)(_n2##x,_p1##y,_n1##z,v)), \
 (I[39] = (img)(_n2##x,y,_n1##z,v)), \
 (I[43] = (img)(_n2##x,_n1##y,_n1##z,v)), \
 (I[47] = (img)(_n2##x,_n2##y,_n1##z,v)), \
 (I[51] = (img)(_n2##x,_p1##y,_n2##z,v)), \
 (I[55] = (img)(_n2##x,y,_n2##z,v)), \
 (I[59] = (img)(_n2##x,_n1##y,_n2##z,v)), \
 (I[63] = (img)(_n2##x,_n2##y,_n2##z,v)),1)) || \
 _n1##x==--_n2##x || x==(_n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], \
 I[4] = I[5], I[5] = I[6], I[6] = I[7], \
 I[8] = I[9], I[9] = I[10], I[10] = I[11], \
 I[12] = I[13], I[13] = I[14], I[14] = I[15], \
 I[16] = I[17], I[17] = I[18], I[18] = I[19], \
 I[20] = I[21], I[21] = I[22], I[22] = I[23], \
 I[24] = I[25], I[25] = I[26], I[26] = I[27], \
 I[28] = I[29], I[29] = I[30], I[30] = I[31], \
 I[32] = I[33], I[33] = I[34], I[34] = I[35], \
 I[36] = I[37], I[37] = I[38], I[38] = I[39], \
 I[40] = I[41], I[41] = I[42], I[42] = I[43], \
 I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], \
 I[52] = I[53], I[53] = I[54], I[54] = I[55], \
 I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], \
 _p1##x = x++, ++_n1##x, ++_n2##x)

#define cimg_get4x4x4(img,x,y,z,v,I) \
 I[0] = (img)(_p1##x,_p1##y,_p1##z,v), I[1] = (img)(x,_p1##y,_p1##z,v), I[2] = (img)(_n1##x,_p1##y,_p1##z,v), I[3] = (img)(_n2##x,_p1##y,_p1##z,v), \
 I[4] = (img)(_p1##x,y,_p1##z,v), I[5] = (img)(x,y,_p1##z,v), I[6] = (img)(_n1##x,y,_p1##z,v), I[7] = (img)(_n2##x,y,_p1##z,v), \
 I[8] = (img)(_p1##x,_n1##y,_p1##z,v), I[9] = (img)(x,_n1##y,_p1##z,v), I[10] = (img)(_n1##x,_n1##y,_p1##z,v), I[11] = (img)(_n2##x,_n1##y,_p1##z,v), \
 I[12] = (img)(_p1##x,_n2##y,_p1##z,v), I[13] = (img)(x,_n2##y,_p1##z,v), I[14] = (img)(_n1##x,_n2##y,_p1##z,v), I[15] = (img)(_n2##x,_n2##y,_p1##z,v), \
 I[16] = (img)(_p1##x,_p1##y,z,v), I[17] = (img)(x,_p1##y,z,v), I[18] = (img)(_n1##x,_p1##y,z,v), I[19] = (img)(_n2##x,_p1##y,z,v), \
 I[20] = (img)(_p1##x,y,z,v), I[21] = (img)(x,y,z,v), I[22] = (img)(_n1##x,y,z,v), I[23] = (img)(_n2##x,y,z,v), \
 I[24] = (img)(_p1##x,_n1##y,z,v), I[25] = (img)(x,_n1##y,z,v), I[26] = (img)(_n1##x,_n1##y,z,v), I[27] = (img)(_n2##x,_n1##y,z,v), \
 I[28] = (img)(_p1##x,_n2##y,z,v), I[29] = (img)(x,_n2##y,z,v), I[30] = (img)(_n1##x,_n2##y,z,v), I[31] = (img)(_n2##x,_n2##y,z,v), \
 I[32] = (img)(_p1##x,_p1##y,_n1##z,v), I[33] = (img)(x,_p1##y,_n1##z,v), I[34] = (img)(_n1##x,_p1##y,_n1##z,v), I[35] = (img)(_n2##x,_p1##y,_n1##z,v), \
 I[36] = (img)(_p1##x,y,_n1##z,v), I[37] = (img)(x,y,_n1##z,v), I[38] = (img)(_n1##x,y,_n1##z,v), I[39] = (img)(_n2##x,y,_n1##z,v), \
 I[40] = (img)(_p1##x,_n1##y,_n1##z,v), I[41] = (img)(x,_n1##y,_n1##z,v), I[42] = (img)(_n1##x,_n1##y,_n1##z,v), I[43] = (img)(_n2##x,_n1##y,_n1##z,v), \
 I[44] = (img)(_p1##x,_n2##y,_n1##z,v), I[45] = (img)(x,_n2##y,_n1##z,v), I[46] = (img)(_n1##x,_n2##y,_n1##z,v), I[47] = (img)(_n2##x,_n2##y,_n1##z,v), \
 I[48] = (img)(_p1##x,_p1##y,_n2##z,v), I[49] = (img)(x,_p1##y,_n2##z,v), I[50] = (img)(_n1##x,_p1##y,_n2##z,v), I[51] = (img)(_n2##x,_p1##y,_n2##z,v), \
 I[52] = (img)(_p1##x,y,_n2##z,v), I[53] = (img)(x,y,_n2##z,v), I[54] = (img)(_n1##x,y,_n2##z,v), I[55] = (img)(_n2##x,y,_n2##z,v), \
 I[56] = (img)(_p1##x,_n1##y,_n2##z,v), I[57] = (img)(x,_n1##y,_n2##z,v), I[58] = (img)(_n1##x,_n1##y,_n2##z,v), I[59] = (img)(_n2##x,_n1##y,_n2##z,v), \
 I[60] = (img)(_p1##x,_n2##y,_n2##z,v), I[61] = (img)(x,_n2##y,_n2##z,v), I[62] = (img)(_n1##x,_n2##y,_n2##z,v), I[63] = (img)(_n2##x,_n2##y,_n2##z,v);

// Define 5x5x5 loop macros for CImg
//-------------------------------------
#define cimg_for_in5(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2; \
 i<=(int)(i1) && (_n2##i<(int)(bound) || _n1##i==--_n2##i || \
 i==(_n2##i = --_n1##i)); \
 _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i)

#define cimg_for_in5X(img,x0,x1,x) cimg_for_in5((img).width,x0,x1,x)
#define cimg_for_in5Y(img,y0,y1,y) cimg_for_in5((img).height,y0,y1,y)
#define cimg_for_in5Z(img,z0,z1,z) cimg_for_in5((img).depth,z0,z1,z)
#define cimg_for_in5V(img,v0,v1,v) cimg_for_in5((img).dim,v0,v1,v)
#define cimg_for_in5XY(img,x0,y0,x1,y1,x,y) cimg_for_in5Y(img,y0,y1,y) cimg_for_in5X(img,x0,x1,x)
#define cimg_for_in5XZ(img,x0,z0,x1,z1,x,z) cimg_for_in5Z(img,z0,z1,z) cimg_for_in5X(img,x0,x1,x)
#define cimg_for_in5XV(img,x0,v0,x1,v1,x,v) cimg_for_in5V(img,v0,v1,v) cimg_for_in5X(img,x0,x1,x)
#define cimg_for_in5YZ(img,y0,z0,y1,z1,y,z) cimg_for_in5Z(img,z0,z1,z) cimg_for_in5Y(img,y0,y1,y)
#define cimg_for_in5YV(img,y0,v0,y1,v1,y,v) cimg_for_in5V(img,v0,v1,v) cimg_for_in5Y(img,y0,y1,y)
#define cimg_for_in5ZV(img,z0,v0,z1,v1,z,v) cimg_for_in5V(img,v0,v1,v) cimg_for_in5Z(img,z0,z1,z)
#define cimg_for_in5XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in5Z(img,z0,z1,z) cimg_for_in5XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in5XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in5V(img,v0,v1,v) cimg_for_in5XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in5YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in5V(img,v0,v1,v) cimg_for_in5YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in5XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in5V(img,v0,v1,v) cimg_for_in5XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for5x5x5(img,x,y,z,v,I) \
 cimg_for5((img).depth,z) cimg_for5((img).height,y) for (int x = 0, \
 _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = (int)( \
 (I[0] = I[1] = I[2] = (img)(0,_p2##y,_p2##z,v)), \
 (I[5] = I[6] = I[7] = (img)(0,_p1##y,_p2##z,v)), \
 (I[10] = I[11] = I[12] = (img)(0,y,_p2##z,v)), \
 (I[15] = I[16] = I[17] = (img)(0,_n1##y,_p2##z,v)), \
 (I[20] = I[21] = I[22] = (img)(0,_n2##y,_p2##z,v)), \
 (I[25] = I[26] = I[27] = (img)(0,_p2##y,_p1##z,v)), \
 (I[30] = I[31] = I[32] = (img)(0,_p1##y,_p1##z,v)), \
 (I[35] = I[36] = I[37] = (img)(0,y,_p1##z,v)), \
 (I[40] = I[41] = I[42] = (img)(0,_n1##y,_p1##z,v)), \
 (I[45] = I[46] = I[47] = (img)(0,_n2##y,_p1##z,v)), \
 (I[50] = I[51] = I[52] = (img)(0,_p2##y,z,v)), \
 (I[55] = I[56] = I[57] = (img)(0,_p1##y,z,v)), \
 (I[60] = I[61] = I[62] = (img)(0,y,z,v)), \
 (I[65] = I[66] = I[67] = (img)(0,_n1##y,z,v)), \
 (I[70] = I[71] = I[72] = (img)(0,_n2##y,z,v)), \
 (I[75] = I[76] = I[77] = (img)(0,_p2##y,_n1##z,v)), \
 (I[80] = I[81] = I[82] = (img)(0,_p1##y,_n1##z,v)), \
 (I[85] = I[86] = I[87] = (img)(0,y,_n1##z,v)), \
 (I[90] = I[91] = I[92] = (img)(0,_n1##y,_n1##z,v)), \
 (I[95] = I[96] = I[97] = (img)(0,_n2##y,_n1##z,v)), \
 (I[100] = I[101] = I[102] = (img)(0,_p2##y,_n2##z,v)), \
 (I[105] = I[106] = I[107] = (img)(0,_p1##y,_n2##z,v)), \
 (I[110] = I[111] = I[112] = (img)(0,y,_n2##z,v)), \
 (I[115] = I[116] = I[117] = (img)(0,_n1##y,_n2##z,v)), \
 (I[120] = I[121] = I[122] = (img)(0,_n2##y,_n2##z,v)), \
 (I[3] = (img)(_n1##x,_p2##y,_p2##z,v)), \
 (I[8] = (img)(_n1##x,_p1##y,_p2##z,v)), \
 (I[13] = (img)(_n1##x,y,_p2##z,v)), \
 (I[18] = (img)(_n1##x,_n1##y,_p2##z,v)), \
 (I[23] = (img)(_n1##x,_n2##y,_p2##z,v)), \
 (I[28] = (img)(_n1##x,_p2##y,_p1##z,v)), \
 (I[33] = (img)(_n1##x,_p1##y,_p1##z,v)), \
 (I[38] = (img)(_n1##x,y,_p1##z,v)), \
 (I[43] = (img)(_n1##x,_n1##y,_p1##z,v)), \
 (I[48] = (img)(_n1##x,_n2##y,_p1##z,v)), \
 (I[53] = (img)(_n1##x,_p2##y,z,v)), \
 (I[58] = (img)(_n1##x,_p1##y,z,v)), \
 (I[63] = (img)(_n1##x,y,z,v)), \
 (I[68] = (img)(_n1##x,_n1##y,z,v)), \
 (I[73] = (img)(_n1##x,_n2##y,z,v)), \
 (I[78] = (img)(_n1##x,_p2##y,_n1##z,v)), \
 (I[83] = (img)(_n1##x,_p1##y,_n1##z,v)), \
 (I[88] = (img)(_n1##x,y,_n1##z,v)), \
 (I[93] = (img)(_n1##x,_n1##y,_n1##z,v)), \
 (I[98] = (img)(_n1##x,_n2##y,_n1##z,v)), \
 (I[103] = (img)(_n1##x,_p2##y,_n2##z,v)), \
 (I[108] = (img)(_n1##x,_p1##y,_n2##z,v)), \
 (I[113] = (img)(_n1##x,y,_n2##z,v)), \
 (I[118] = (img)(_n1##x,_n1##y,_n2##z,v)), \
 (I[123] = (img)(_n1##x,_n2##y,_n2##z,v)), \
 2>=((img).width)?(int)((img).width)-1:2); \
 (_n2##x<(int)((img).width) && ( \
 (I[4] = (img)(_n2##x,_p2##y,_p2##z,v)), \
 (I[9] = (img)(_n2##x,_p1##y,_p2##z,v)), \
 (I[14] = (img)(_n2##x,y,_p2##z,v)), \
 (I[19] = (img)(_n2##x,_n1##y,_p2##z,v)), \
 (I[24] = (img)(_n2##x,_n2##y,_p2##z,v)), \
 (I[29] = (img)(_n2##x,_p2##y,_p1##z,v)), \
 (I[34] = (img)(_n2##x,_p1##y,_p1##z,v)), \
 (I[39] = (img)(_n2##x,y,_p1##z,v)), \
 (I[44] = (img)(_n2##x,_n1##y,_p1##z,v)), \
 (I[49] = (img)(_n2##x,_n2##y,_p1##z,v)), \
 (I[54] = (img)(_n2##x,_p2##y,z,v)), \
 (I[59] = (img)(_n2##x,_p1##y,z,v)), \
 (I[64] = (img)(_n2##x,y,z,v)), \
 (I[69] = (img)(_n2##x,_n1##y,z,v)), \
 (I[74] = (img)(_n2##x,_n2##y,z,v)), \
 (I[79] = (img)(_n2##x,_p2##y,_n1##z,v)), \
 (I[84] = (img)(_n2##x,_p1##y,_n1##z,v)), \
 (I[89] = (img)(_n2##x,y,_n1##z,v)), \
 (I[94] = (img)(_n2##x,_n1##y,_n1##z,v)), \
 (I[99] = (img)(_n2##x,_n2##y,_n1##z,v)), \
 (I[104] = (img)(_n2##x,_p2##y,_n2##z,v)), \
 (I[109] = (img)(_n2##x,_p1##y,_n2##z,v)), \
 (I[114] = (img)(_n2##x,y,_n2##z,v)), \
 (I[119] = (img)(_n2##x,_n1##y,_n2##z,v)), \
 (I[124] = (img)(_n2##x,_n2##y,_n2##z,v)),1)) || \
 _n1##x==--_n2##x || x==(_n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], \
 I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], \
 I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], \
 I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], \
 I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], \
 I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], \
 I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], \
 I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], \
 I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], \
 I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], \
 I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], \
 I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], \
 I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], \
 I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], \
 I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], \
 I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], \
 I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], \
 I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], \
 I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], \
 I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], \
 I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], \
 I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], \
 I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], \
 _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x)

#define cimg_for_in5x5x5(img,x0,y0,z0,x1,y1,z1,x,y,z,v,I) \
 cimg_for_in5((img).depth,z0,z1,z) cimg_for_in5((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = (int)( \
 (I[0] = (img)(_p2##x,_p2##y,_p2##z,v)), \
 (I[5] = (img)(_p2##x,_p1##y,_p2##z,v)), \
 (I[10] = (img)(_p2##x,y,_p2##z,v)), \
 (I[15] = (img)(_p2##x,_n1##y,_p2##z,v)), \
 (I[20] = (img)(_p2##x,_n2##y,_p2##z,v)), \
 (I[25] = (img)(_p2##x,_p2##y,_p1##z,v)), \
 (I[30] = (img)(_p2##x,_p1##y,_p1##z,v)), \
 (I[35] = (img)(_p2##x,y,_p1##z,v)), \
 (I[40] = (img)(_p2##x,_n1##y,_p1##z,v)), \
 (I[45] = (img)(_p2##x,_n2##y,_p1##z,v)), \
 (I[50] = (img)(_p2##x,_p2##y,z,v)), \
 (I[55] = (img)(_p2##x,_p1##y,z,v)), \
 (I[60] = (img)(_p2##x,y,z,v)), \
 (I[65] = (img)(_p2##x,_n1##y,z,v)), \
 (I[70] = (img)(_p2##x,_n2##y,z,v)), \
 (I[75] = (img)(_p2##x,_p2##y,_n1##z,v)), \
 (I[80] = (img)(_p2##x,_p1##y,_n1##z,v)), \
 (I[85] = (img)(_p2##x,y,_n1##z,v)), \
 (I[90] = (img)(_p2##x,_n1##y,_n1##z,v)), \
 (I[95] = (img)(_p2##x,_n2##y,_n1##z,v)), \
 (I[100] = (img)(_p2##x,_p2##y,_n2##z,v)), \
 (I[105] = (img)(_p2##x,_p1##y,_n2##z,v)), \
 (I[110] = (img)(_p2##x,y,_n2##z,v)), \
 (I[115] = (img)(_p2##x,_n1##y,_n2##z,v)), \
 (I[120] = (img)(_p2##x,_n2##y,_n2##z,v)), \
 (I[1] = (img)(_p1##x,_p2##y,_p2##z,v)), \
 (I[6] = (img)(_p1##x,_p1##y,_p2##z,v)), \
 (I[11] = (img)(_p1##x,y,_p2##z,v)), \
 (I[16] = (img)(_p1##x,_n1##y,_p2##z,v)), \
 (I[21] = (img)(_p1##x,_n2##y,_p2##z,v)), \
 (I[26] = (img)(_p1##x,_p2##y,_p1##z,v)), \
 (I[31] = (img)(_p1##x,_p1##y,_p1##z,v)), \
 (I[36] = (img)(_p1##x,y,_p1##z,v)), \
 (I[41] = (img)(_p1##x,_n1##y,_p1##z,v)), \
 (I[46] = (img)(_p1##x,_n2##y,_p1##z,v)), \
 (I[51] = (img)(_p1##x,_p2##y,z,v)), \
 (I[56] = (img)(_p1##x,_p1##y,z,v)), \
 (I[61] = (img)(_p1##x,y,z,v)), \
 (I[66] = (img)(_p1##x,_n1##y,z,v)), \
 (I[71] = (img)(_p1##x,_n2##y,z,v)), \
 (I[76] = (img)(_p1##x,_p2##y,_n1##z,v)), \
 (I[81] = (img)(_p1##x,_p1##y,_n1##z,v)), \
 (I[86] = (img)(_p1##x,y,_n1##z,v)), \
 (I[91] = (img)(_p1##x,_n1##y,_n1##z,v)), \
 (I[96] = (img)(_p1##x,_n2##y,_n1##z,v)), \
 (I[101] = (img)(_p1##x,_p2##y,_n2##z,v)), \
 (I[106] = (img)(_p1##x,_p1##y,_n2##z,v)), \
 (I[111] = (img)(_p1##x,y,_n2##z,v)), \
 (I[116] = (img)(_p1##x,_n1##y,_n2##z,v)), \
 (I[121] = (img)(_p1##x,_n2##y,_n2##z,v)), \
 (I[2] = (img)(x,_p2##y,_p2##z,v)), \
 (I[7] = (img)(x,_p1##y,_p2##z,v)), \
 (I[12] = (img)(x,y,_p2##z,v)), \
 (I[17] = (img)(x,_n1##y,_p2##z,v)), \
 (I[22] = (img)(x,_n2##y,_p2##z,v)), \
 (I[27] = (img)(x,_p2##y,_p1##z,v)), \
 (I[32] = (img)(x,_p1##y,_p1##z,v)), \
 (I[37] = (img)(x,y,_p1##z,v)), \
 (I[42] = (img)(x,_n1##y,_p1##z,v)), \
 (I[47] = (img)(x,_n2##y,_p1##z,v)), \
 (I[52] = (img)(x,_p2##y,z,v)), \
 (I[57] = (img)(x,_p1##y,z,v)), \
 (I[62] = (img)(x,y,z,v)), \
 (I[67] = (img)(x,_n1##y,z,v)), \
 (I[72] = (img)(x,_n2##y,z,v)), \
 (I[77] = (img)(x,_p2##y,_n1##z,v)), \
 (I[82] = (img)(x,_p1##y,_n1##z,v)), \
 (I[87] = (img)(x,y,_n1##z,v)), \
 (I[92] = (img)(x,_n1##y,_n1##z,v)), \
 (I[97] = (img)(x,_n2##y,_n1##z,v)), \
 (I[102] = (img)(x,_p2##y,_n2##z,v)), \
 (I[107] = (img)(x,_p1##y,_n2##z,v)), \
 (I[112] = (img)(x,y,_n2##z,v)), \
 (I[117] = (img)(x,_n1##y,_n2##z,v)), \
 (I[122] = (img)(x,_n2##y,_n2##z,v)), \
 (I[3] = (img)(_n1##x,_p2##y,_p2##z,v)), \
 (I[8] = (img)(_n1##x,_p1##y,_p2##z,v)), \
 (I[13] = (img)(_n1##x,y,_p2##z,v)), \
 (I[18] = (img)(_n1##x,_n1##y,_p2##z,v)), \
 (I[23] = (img)(_n1##x,_n2##y,_p2##z,v)), \
 (I[28] = (img)(_n1##x,_p2##y,_p1##z,v)), \
 (I[33] = (img)(_n1##x,_p1##y,_p1##z,v)), \
 (I[38] = (img)(_n1##x,y,_p1##z,v)), \
 (I[43] = (img)(_n1##x,_n1##y,_p1##z,v)), \
 (I[48] = (img)(_n1##x,_n2##y,_p1##z,v)), \
 (I[53] = (img)(_n1##x,_p2##y,z,v)), \
 (I[58] = (img)(_n1##x,_p1##y,z,v)), \
 (I[63] = (img)(_n1##x,y,z,v)), \
 (I[68] = (img)(_n1##x,_n1##y,z,v)), \
 (I[73] = (img)(_n1##x,_n2##y,z,v)), \
 (I[78] = (img)(_n1##x,_p2##y,_n1##z,v)), \
 (I[83] = (img)(_n1##x,_p1##y,_n1##z,v)), \
 (I[88] = (img)(_n1##x,y,_n1##z,v)), \
 (I[93] = (img)(_n1##x,_n1##y,_n1##z,v)), \
 (I[98] = (img)(_n1##x,_n2##y,_n1##z,v)), \
 (I[103] = (img)(_n1##x,_p2##y,_n2##z,v)), \
 (I[108] = (img)(_n1##x,_p1##y,_n2##z,v)), \
 (I[113] = (img)(_n1##x,y,_n2##z,v)), \
 (I[118] = (img)(_n1##x,_n1##y,_n2##z,v)), \
 (I[123] = (img)(_n1##x,_n2##y,_n2##z,v)), \
 x+2>=(int)((img).width)?(int)((img).width)-1:x+2); \
 x<=(int)(x1) && ((_n2##x<(int)((img).width) && ( \
 (I[4] = (img)(_n2##x,_p2##y,_p2##z,v)), \
 (I[9] = (img)(_n2##x,_p1##y,_p2##z,v)), \
 (I[14] = (img)(_n2##x,y,_p2##z,v)), \
 (I[19] = (img)(_n2##x,_n1##y,_p2##z,v)), \
 (I[24] = (img)(_n2##x,_n2##y,_p2##z,v)), \
 (I[29] = (img)(_n2##x,_p2##y,_p1##z,v)), \
 (I[34] = (img)(_n2##x,_p1##y,_p1##z,v)), \
 (I[39] = (img)(_n2##x,y,_p1##z,v)), \
 (I[44] = (img)(_n2##x,_n1##y,_p1##z,v)), \
 (I[49] = (img)(_n2##x,_n2##y,_p1##z,v)), \
 (I[54] = (img)(_n2##x,_p2##y,z,v)), \
 (I[59] = (img)(_n2##x,_p1##y,z,v)), \
 (I[64] = (img)(_n2##x,y,z,v)), \
 (I[69] = (img)(_n2##x,_n1##y,z,v)), \
 (I[74] = (img)(_n2##x,_n2##y,z,v)), \
 (I[79] = (img)(_n2##x,_p2##y,_n1##z,v)), \
 (I[84] = (img)(_n2##x,_p1##y,_n1##z,v)), \
 (I[89] = (img)(_n2##x,y,_n1##z,v)), \
 (I[94] = (img)(_n2##x,_n1##y,_n1##z,v)), \
 (I[99] = (img)(_n2##x,_n2##y,_n1##z,v)), \
 (I[104] = (img)(_n2##x,_p2##y,_n2##z,v)), \
 (I[109] = (img)(_n2##x,_p1##y,_n2##z,v)), \
 (I[114] = (img)(_n2##x,y,_n2##z,v)), \
 (I[119] = (img)(_n2##x,_n1##y,_n2##z,v)), \
 (I[124] = (img)(_n2##x,_n2##y,_n2##z,v)),1)) || \
 _n1##x==--_n2##x || x==(_n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], \
 I[5] = I[6], I[6] = I[7], I[7] = I[8], I[8] = I[9], \
 I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], \
 I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], \
 I[20] = I[21], I[21] = I[22], I[22] = I[23], I[23] = I[24], \
 I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], \
 I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], \
 I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], \
 I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], \
 I[45] = I[46], I[46] = I[47], I[47] = I[48], I[48] = I[49], \
 I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], \
 I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], \
 I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], \
 I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], \
 I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], \
 I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], \
 I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], \
 I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], \
 I[95] = I[96], I[96] = I[97], I[97] = I[98], I[98] = I[99], \
 I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], \
 I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], \
 I[110] = I[111], I[111] = I[112], I[112] = I[113], I[113] = I[114], \
 I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], \
 _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x)

#define cimg_get5x5x5(img,x,y,z,v,I) \
 I[0] = (img)(_p2##x,_p2##y,_p2##z,v), I[1] = (img)(_p1##x,_p2##y,_p2##z,v), I[2] = (img)(x,_p2##y,_p2##z,v), I[3] = (img)(_n1##x,_p2##y,_p2##z,v), I[4] = (img)(_n2##x,_p2##y,_p2##z,v), \
 I[5] = (img)(_p2##x,_p1##y,_p2##z,v), I[6] = (img)(_p1##x,_p1##y,_p2##z,v), I[7] = (img)(x,_p1##y,_p2##z,v), I[8] = (img)(_n1##x,_p1##y,_p2##z,v), I[9] = (img)(_n2##x,_p1##y,_p2##z,v), \
 I[10] = (img)(_p2##x,y,_p2##z,v), I[11] = (img)(_p1##x,y,_p2##z,v), I[12] = (img)(x,y,_p2##z,v), I[13] = (img)(_n1##x,y,_p2##z,v), I[14] = (img)(_n2##x,y,_p2##z,v), \
 I[15] = (img)(_p2##x,_n1##y,_p2##z,v), I[16] = (img)(_p1##x,_n1##y,_p2##z,v), I[17] = (img)(x,_n1##y,_p2##z,v), I[18] = (img)(_n1##x,_n1##y,_p2##z,v), I[19] = (img)(_n2##x,_n1##y,_p2##z,v), \
 I[20] = (img)(_p2##x,_n2##y,_p2##z,v), I[21] = (img)(_p1##x,_n2##y,_p2##z,v), I[22] = (img)(x,_n2##y,_p2##z,v), I[23] = (img)(_n1##x,_n2##y,_p2##z,v), I[24] = (img)(_n2##x,_n2##y,_p2##z,v), \
 I[25] = (img)(_p2##x,_p2##y,_p1##z,v), I[26] = (img)(_p1##x,_p2##y,_p1##z,v), I[27] = (img)(x,_p2##y,_p1##z,v), I[28] = (img)(_n1##x,_p2##y,_p1##z,v), I[29] = (img)(_n2##x,_p2##y,_p1##z,v), \
 I[30] = (img)(_p2##x,_p1##y,_p1##z,v), I[31] = (img)(_p1##x,_p1##y,_p1##z,v), I[32] = (img)(x,_p1##y,_p1##z,v), I[33] = (img)(_n1##x,_p1##y,_p1##z,v), I[34] = (img)(_n2##x,_p1##y,_p1##z,v), \
 I[35] = (img)(_p2##x,y,_p1##z,v), I[36] = (img)(_p1##x,y,_p1##z,v), I[37] = (img)(x,y,_p1##z,v), I[38] = (img)(_n1##x,y,_p1##z,v), I[39] = (img)(_n2##x,y,_p1##z,v), \
 I[40] = (img)(_p2##x,_n1##y,_p1##z,v), I[41] = (img)(_p1##x,_n1##y,_p1##z,v), I[42] = (img)(x,_n1##y,_p1##z,v), I[43] = (img)(_n1##x,_n1##y,_p1##z,v), I[44] = (img)(_n2##x,_n1##y,_p1##z,v), \
 I[45] = (img)(_p2##x,_n2##y,_p1##z,v), I[46] = (img)(_p1##x,_n2##y,_p1##z,v), I[47] = (img)(x,_n2##y,_p1##z,v), I[48] = (img)(_n1##x,_n2##y,_p1##z,v), I[49] = (img)(_n2##x,_n2##y,_p1##z,v), \
 I[50] = (img)(_p2##x,_p2##y,z,v), I[51] = (img)(_p1##x,_p2##y,z,v), I[52] = (img)(x,_p2##y,z,v), I[53] = (img)(_n1##x,_p2##y,z,v), I[54] = (img)(_n2##x,_p2##y,z,v), \
 I[55] = (img)(_p2##x,_p1##y,z,v), I[56] = (img)(_p1##x,_p1##y,z,v), I[57] = (img)(x,_p1##y,z,v), I[58] = (img)(_n1##x,_p1##y,z,v), I[59] = (img)(_n2##x,_p1##y,z,v), \
 I[60] = (img)(_p2##x,y,z,v), I[61] = (img)(_p1##x,y,z,v), I[62] = (img)(x,y,z,v), I[63] = (img)(_n1##x,y,z,v), I[64] = (img)(_n2##x,y,z,v), \
 I[65] = (img)(_p2##x,_n1##y,z,v), I[66] = (img)(_p1##x,_n1##y,z,v), I[67] = (img)(x,_n1##y,z,v), I[68] = (img)(_n1##x,_n1##y,z,v), I[69] = (img)(_n2##x,_n1##y,z,v), \
 I[70] = (img)(_p2##x,_n2##y,z,v), I[71] = (img)(_p1##x,_n2##y,z,v), I[72] = (img)(x,_n2##y,z,v), I[73] = (img)(_n1##x,_n2##y,z,v), I[74] = (img)(_n2##x,_n2##y,z,v), \
 I[75] = (img)(_p2##x,_p2##y,_n1##z,v), I[76] = (img)(_p1##x,_p2##y,_n1##z,v), I[77] = (img)(x,_p2##y,_n1##z,v), I[78] = (img)(_n1##x,_p2##y,_n1##z,v), I[79] = (img)(_n2##x,_p2##y,_n1##z,v), \
 I[80] = (img)(_p2##x,_p1##y,_n1##z,v), I[81] = (img)(_p1##x,_p1##y,_n1##z,v), I[82] = (img)(x,_p1##y,_n1##z,v), I[83] = (img)(_n1##x,_p1##y,_n1##z,v), I[84] = (img)(_n2##x,_p1##y,_n1##z,v), \
 I[85] = (img)(_p2##x,y,_n1##z,v), I[86] = (img)(_p1##x,y,_n1##z,v), I[87] = (img)(x,y,_n1##z,v), I[88] = (img)(_n1##x,y,_n1##z,v), I[89] = (img)(_n2##x,y,_n1##z,v), \
 I[90] = (img)(_p2##x,_n1##y,_n1##z,v), I[91] = (img)(_p1##x,_n1##y,_n1##z,v), I[92] = (img)(x,_n1##y,_n1##z,v), I[93] = (img)(_n1##x,_n1##y,_n1##z,v), I[94] = (img)(_n2##x,_n1##y,_n1##z,v), \
 I[95] = (img)(_p2##x,_n2##y,_n1##z,v), I[96] = (img)(_p1##x,_n2##y,_n1##z,v), I[97] = (img)(x,_n2##y,_n1##z,v), I[98] = (img)(_n1##x,_n2##y,_n1##z,v), I[99] = (img)(_n2##x,_n2##y,_n1##z,v), \
 I[100] = (img)(_p2##x,_p2##y,_n2##z,v), I[101] = (img)(_p1##x,_p2##y,_n2##z,v), I[102] = (img)(x,_p2##y,_n2##z,v), I[103] = (img)(_n1##x,_p2##y,_n2##z,v), I[104] = (img)(_n2##x,_p2##y,_n2##z,v), \
 I[105] = (img)(_p2##x,_p1##y,_n2##z,v), I[106] = (img)(_p1##x,_p1##y,_n2##z,v), I[107] = (img)(x,_p1##y,_n2##z,v), I[108] = (img)(_n1##x,_p1##y,_n2##z,v), I[109] = (img)(_n2##x,_p1##y,_n2##z,v), \
 I[110] = (img)(_p2##x,y,_n2##z,v), I[111] = (img)(_p1##x,y,_n2##z,v), I[112] = (img)(x,y,_n2##z,v), I[113] = (img)(_n1##x,y,_n2##z,v), I[114] = (img)(_n2##x,y,_n2##z,v), \
 I[115] = (img)(_p2##x,_n1##y,_n2##z,v), I[116] = (img)(_p1##x,_n1##y,_n2##z,v), I[117] = (img)(x,_n1##y,_n2##z,v), I[118] = (img)(_n1##x,_n1##y,_n2##z,v), I[119] = (img)(_n2##x,_n1##y,_n2##z,v), \
 I[120] = (img)(_p2##x,_n2##y,_n2##z,v), I[121] = (img)(_p1##x,_n2##y,_n2##z,v), I[122] = (img)(x,_n2##y,_n2##z,v), I[123] = (img)(_n1##x,_n2##y,_n2##z,v), I[124] = (img)(_n2##x,_n2##y,_n2##z,v);

// Define 6x6x6 loop macros for CImg
//-------------------------------------
#define cimg_for_in6(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3; \
 i<=(int)(i1) && (_n3##i<(int)(bound) || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n3##i = _n2##i = --_n1##i)); \
 _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i)

#define cimg_for_in6X(img,x0,x1,x) cimg_for_in6((img).width,x0,x1,x)
#define cimg_for_in6Y(img,y0,y1,y) cimg_for_in6((img).height,y0,y1,y)
#define cimg_for_in6Z(img,z0,z1,z) cimg_for_in6((img).depth,z0,z1,z)
#define cimg_for_in6V(img,v0,v1,v) cimg_for_in6((img).dim,v0,v1,v)
#define cimg_for_in6XY(img,x0,y0,x1,y1,x,y) cimg_for_in6Y(img,y0,y1,y) cimg_for_in6X(img,x0,x1,x)
#define cimg_for_in6XZ(img,x0,z0,x1,z1,x,z) cimg_for_in6Z(img,z0,z1,z) cimg_for_in6X(img,x0,x1,x)
#define cimg_for_in6XV(img,x0,v0,x1,v1,x,v) cimg_for_in6V(img,v0,v1,v) cimg_for_in6X(img,x0,x1,x)
#define cimg_for_in6YZ(img,y0,z0,y1,z1,y,z) cimg_for_in6Z(img,z0,z1,z) cimg_for_in6Y(img,y0,y1,y)
#define cimg_for_in6YV(img,y0,v0,y1,v1,y,v) cimg_for_in6V(img,v0,v1,v) cimg_for_in6Y(img,y0,y1,y)
#define cimg_for_in6ZV(img,z0,v0,z1,v1,z,v) cimg_for_in6V(img,v0,v1,v) cimg_for_in6Z(img,z0,z1,z)
#define cimg_for_in6XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in6Z(img,z0,z1,z) cimg_for_in6XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in6XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in6V(img,v0,v1,v) cimg_for_in6XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in6YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in6V(img,v0,v1,v) cimg_for_in6YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in6XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in6V(img,v0,v1,v) cimg_for_in6XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for6x6x6(img,x,y,z,v,I) \
 cimg_for6((img).depth,z) cimg_for6((img).height,y) for (int x = 0, \
 _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = (int)( \
 (I[0] = I[1] = I[2] = (img)(0,_p2##y,_p2##z,v)), \
 (I[6] = I[7] = I[8] = (img)(0,_p1##y,_p2##z,v)), \
 (I[12] = I[13] = I[14] = (img)(0,y,_p2##z,v)), \
 (I[18] = I[19] = I[20] = (img)(0,_n1##y,_p2##z,v)), \
 (I[24] = I[25] = I[26] = (img)(0,_n2##y,_p2##z,v)), \
 (I[30] = I[31] = I[32] = (img)(0,_n3##y,_p2##z,v)), \
 (I[36] = I[37] = I[38] = (img)(0,_p2##y,_p1##z,v)), \
 (I[42] = I[43] = I[44] = (img)(0,_p1##y,_p1##z,v)), \
 (I[48] = I[49] = I[50] = (img)(0,y,_p1##z,v)), \
 (I[54] = I[55] = I[56] = (img)(0,_n1##y,_p1##z,v)), \
 (I[60] = I[61] = I[62] = (img)(0,_n2##y,_p1##z,v)), \
 (I[66] = I[67] = I[68] = (img)(0,_n3##y,_p1##z,v)), \
 (I[72] = I[73] = I[74] = (img)(0,_p2##y,z,v)), \
 (I[78] = I[79] = I[80] = (img)(0,_p1##y,z,v)), \
 (I[84] = I[85] = I[86] = (img)(0,y,z,v)), \
 (I[90] = I[91] = I[92] = (img)(0,_n1##y,z,v)), \
 (I[96] = I[97] = I[98] = (img)(0,_n2##y,z,v)), \
 (I[102] = I[103] = I[104] = (img)(0,_n3##y,z,v)), \
 (I[108] = I[109] = I[110] = (img)(0,_p2##y,_n1##z,v)), \
 (I[114] = I[115] = I[116] = (img)(0,_p1##y,_n1##z,v)), \
 (I[120] = I[121] = I[122] = (img)(0,y,_n1##z,v)), \
 (I[126] = I[127] = I[128] = (img)(0,_n1##y,_n1##z,v)), \
 (I[132] = I[133] = I[134] = (img)(0,_n2##y,_n1##z,v)), \
 (I[138] = I[139] = I[140] = (img)(0,_n3##y,_n1##z,v)), \
 (I[144] = I[145] = I[146] = (img)(0,_p2##y,_n2##z,v)), \
 (I[150] = I[151] = I[152] = (img)(0,_p1##y,_n2##z,v)), \
 (I[156] = I[157] = I[158] = (img)(0,y,_n2##z,v)), \
 (I[162] = I[163] = I[164] = (img)(0,_n1##y,_n2##z,v)), \
 (I[168] = I[169] = I[170] = (img)(0,_n2##y,_n2##z,v)), \
 (I[174] = I[175] = I[176] = (img)(0,_n3##y,_n2##z,v)), \
 (I[180] = I[181] = I[182] = (img)(0,_p2##y,_n3##z,v)), \
 (I[186] = I[187] = I[188] = (img)(0,_p1##y,_n3##z,v)), \
 (I[192] = I[193] = I[194] = (img)(0,y,_n3##z,v)), \
 (I[198] = I[199] = I[200] = (img)(0,_n1##y,_n3##z,v)), \
 (I[204] = I[205] = I[206] = (img)(0,_n2##y,_n3##z,v)), \
 (I[210] = I[211] = I[212] = (img)(0,_n3##y,_n3##z,v)), \
 (I[3] = (img)(_n1##x,_p2##y,_p2##z,v)), \
 (I[9] = (img)(_n1##x,_p1##y,_p2##z,v)), \
 (I[15] = (img)(_n1##x,y,_p2##z,v)), \
 (I[21] = (img)(_n1##x,_n1##y,_p2##z,v)), \
 (I[27] = (img)(_n1##x,_n2##y,_p2##z,v)), \
 (I[33] = (img)(_n1##x,_n3##y,_p2##z,v)), \
 (I[39] = (img)(_n1##x,_p2##y,_p1##z,v)), \
 (I[45] = (img)(_n1##x,_p1##y,_p1##z,v)), \
 (I[51] = (img)(_n1##x,y,_p1##z,v)), \
 (I[57] = (img)(_n1##x,_n1##y,_p1##z,v)), \
 (I[63] = (img)(_n1##x,_n2##y,_p1##z,v)), \
 (I[69] = (img)(_n1##x,_n3##y,_p1##z,v)), \
 (I[75] = (img)(_n1##x,_p2##y,z,v)), \
 (I[81] = (img)(_n1##x,_p1##y,z,v)), \
 (I[87] = (img)(_n1##x,y,z,v)), \
 (I[93] = (img)(_n1##x,_n1##y,z,v)), \
 (I[99] = (img)(_n1##x,_n2##y,z,v)), \
 (I[105] = (img)(_n1##x,_n3##y,z,v)), \
 (I[111] = (img)(_n1##x,_p2##y,_n1##z,v)), \
 (I[117] = (img)(_n1##x,_p1##y,_n1##z,v)), \
 (I[123] = (img)(_n1##x,y,_n1##z,v)), \
 (I[129] = (img)(_n1##x,_n1##y,_n1##z,v)), \
 (I[135] = (img)(_n1##x,_n2##y,_n1##z,v)), \
 (I[141] = (img)(_n1##x,_n3##y,_n1##z,v)), \
 (I[147] = (img)(_n1##x,_p2##y,_n2##z,v)), \
 (I[153] = (img)(_n1##x,_p1##y,_n2##z,v)), \
 (I[159] = (img)(_n1##x,y,_n2##z,v)), \
 (I[165] = (img)(_n1##x,_n1##y,_n2##z,v)), \
 (I[171] = (img)(_n1##x,_n2##y,_n2##z,v)), \
 (I[177] = (img)(_n1##x,_n3##y,_n2##z,v)), \
 (I[183] = (img)(_n1##x,_p2##y,_n3##z,v)), \
 (I[189] = (img)(_n1##x,_p1##y,_n3##z,v)), \
 (I[195] = (img)(_n1##x,y,_n3##z,v)), \
 (I[201] = (img)(_n1##x,_n1##y,_n3##z,v)), \
 (I[207] = (img)(_n1##x,_n2##y,_n3##z,v)), \
 (I[213] = (img)(_n1##x,_n3##y,_n3##z,v)), \
 (I[4] = (img)(_n2##x,_p2##y,_p2##z,v)), \
 (I[10] = (img)(_n2##x,_p1##y,_p2##z,v)), \
 (I[16] = (img)(_n2##x,y,_p2##z,v)), \
 (I[22] = (img)(_n2##x,_n1##y,_p2##z,v)), \
 (I[28] = (img)(_n2##x,_n2##y,_p2##z,v)), \
 (I[34] = (img)(_n2##x,_n3##y,_p2##z,v)), \
 (I[40] = (img)(_n2##x,_p2##y,_p1##z,v)), \
 (I[46] = (img)(_n2##x,_p1##y,_p1##z,v)), \
 (I[52] = (img)(_n2##x,y,_p1##z,v)), \
 (I[58] = (img)(_n2##x,_n1##y,_p1##z,v)), \
 (I[64] = (img)(_n2##x,_n2##y,_p1##z,v)), \
 (I[70] = (img)(_n2##x,_n3##y,_p1##z,v)), \
 (I[76] = (img)(_n2##x,_p2##y,z,v)), \
 (I[82] = (img)(_n2##x,_p1##y,z,v)), \
 (I[88] = (img)(_n2##x,y,z,v)), \
 (I[94] = (img)(_n2##x,_n1##y,z,v)), \
 (I[100] = (img)(_n2##x,_n2##y,z,v)), \
 (I[106] = (img)(_n2##x,_n3##y,z,v)), \
 (I[112] = (img)(_n2##x,_p2##y,_n1##z,v)), \
 (I[118] = (img)(_n2##x,_p1##y,_n1##z,v)), \
 (I[124] = (img)(_n2##x,y,_n1##z,v)), \
 (I[130] = (img)(_n2##x,_n1##y,_n1##z,v)), \
 (I[136] = (img)(_n2##x,_n2##y,_n1##z,v)), \
 (I[142] = (img)(_n2##x,_n3##y,_n1##z,v)), \
 (I[148] = (img)(_n2##x,_p2##y,_n2##z,v)), \
 (I[154] = (img)(_n2##x,_p1##y,_n2##z,v)), \
 (I[160] = (img)(_n2##x,y,_n2##z,v)), \
 (I[166] = (img)(_n2##x,_n1##y,_n2##z,v)), \
 (I[172] = (img)(_n2##x,_n2##y,_n2##z,v)), \
 (I[178] = (img)(_n2##x,_n3##y,_n2##z,v)), \
 (I[184] = (img)(_n2##x,_p2##y,_n3##z,v)), \
 (I[190] = (img)(_n2##x,_p1##y,_n3##z,v)), \
 (I[196] = (img)(_n2##x,y,_n3##z,v)), \
 (I[202] = (img)(_n2##x,_n1##y,_n3##z,v)), \
 (I[208] = (img)(_n2##x,_n2##y,_n3##z,v)), \
 (I[214] = (img)(_n2##x,_n3##y,_n3##z,v)), \
 3>=((img).width)?(int)((img).width)-1:3); \
 (_n3##x<(int)((img).width) && ( \
 (I[5] = (img)(_n3##x,_p2##y,_p2##z,v)), \
 (I[11] = (img)(_n3##x,_p1##y,_p2##z,v)), \
 (I[17] = (img)(_n3##x,y,_p2##z,v)), \
 (I[23] = (img)(_n3##x,_n1##y,_p2##z,v)), \
 (I[29] = (img)(_n3##x,_n2##y,_p2##z,v)), \
 (I[35] = (img)(_n3##x,_n3##y,_p2##z,v)), \
 (I[41] = (img)(_n3##x,_p2##y,_p1##z,v)), \
 (I[47] = (img)(_n3##x,_p1##y,_p1##z,v)), \
 (I[53] = (img)(_n3##x,y,_p1##z,v)), \
 (I[59] = (img)(_n3##x,_n1##y,_p1##z,v)), \
 (I[65] = (img)(_n3##x,_n2##y,_p1##z,v)), \
 (I[71] = (img)(_n3##x,_n3##y,_p1##z,v)), \
 (I[77] = (img)(_n3##x,_p2##y,z,v)), \
 (I[83] = (img)(_n3##x,_p1##y,z,v)), \
 (I[89] = (img)(_n3##x,y,z,v)), \
 (I[95] = (img)(_n3##x,_n1##y,z,v)), \
 (I[101] = (img)(_n3##x,_n2##y,z,v)), \
 (I[107] = (img)(_n3##x,_n3##y,z,v)), \
 (I[113] = (img)(_n3##x,_p2##y,_n1##z,v)), \
 (I[119] = (img)(_n3##x,_p1##y,_n1##z,v)), \
 (I[125] = (img)(_n3##x,y,_n1##z,v)), \
 (I[131] = (img)(_n3##x,_n1##y,_n1##z,v)), \
 (I[137] = (img)(_n3##x,_n2##y,_n1##z,v)), \
 (I[143] = (img)(_n3##x,_n3##y,_n1##z,v)), \
 (I[149] = (img)(_n3##x,_p2##y,_n2##z,v)), \
 (I[155] = (img)(_n3##x,_p1##y,_n2##z,v)), \
 (I[161] = (img)(_n3##x,y,_n2##z,v)), \
 (I[167] = (img)(_n3##x,_n1##y,_n2##z,v)), \
 (I[173] = (img)(_n3##x,_n2##y,_n2##z,v)), \
 (I[179] = (img)(_n3##x,_n3##y,_n2##z,v)), \
 (I[185] = (img)(_n3##x,_p2##y,_n3##z,v)), \
 (I[191] = (img)(_n3##x,_p1##y,_n3##z,v)), \
 (I[197] = (img)(_n3##x,y,_n3##z,v)), \
 (I[203] = (img)(_n3##x,_n1##y,_n3##z,v)), \
 (I[209] = (img)(_n3##x,_n2##y,_n3##z,v)), \
 (I[215] = (img)(_n3##x,_n3##y,_n3##z,v)),1)) || \
 _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], \
 I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], \
 I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], \
 I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], \
 I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], \
 I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], \
 I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], \
 I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], \
 I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], \
 I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], \
 I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], \
 I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], \
 I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], \
 I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], \
 I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], \
 I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], \
 I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], \
 I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], \
 I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], \
 I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], \
 I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], \
 I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], \
 I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], \
 I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], \
 I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], \
 I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], \
 I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], \
 I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], \
 I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], \
 I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], \
 I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], \
 _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x)

#define cimg_for_in6x6x6(img,x0,y0,z0,x1,y1,z1,x,y,z,v,I) \
 cimg_for_in6((img).depth,z0,z1,z) cimg_for_in6((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = (int)( \
 (I[0] = (img)(_p2##x,_p2##y,_p2##z,v)), \
 (I[6] = (img)(_p2##x,_p1##y,_p2##z,v)), \
 (I[12] = (img)(_p2##x,y,_p2##z,v)), \
 (I[18] = (img)(_p2##x,_n1##y,_p2##z,v)), \
 (I[24] = (img)(_p2##x,_n2##y,_p2##z,v)), \
 (I[30] = (img)(_p2##x,_n3##y,_p2##z,v)), \
 (I[36] = (img)(_p2##x,_p2##y,_p1##z,v)), \
 (I[42] = (img)(_p2##x,_p1##y,_p1##z,v)), \
 (I[48] = (img)(_p2##x,y,_p1##z,v)), \
 (I[54] = (img)(_p2##x,_n1##y,_p1##z,v)), \
 (I[60] = (img)(_p2##x,_n2##y,_p1##z,v)), \
 (I[66] = (img)(_p2##x,_n3##y,_p1##z,v)), \
 (I[72] = (img)(_p2##x,_p2##y,z,v)), \
 (I[78] = (img)(_p2##x,_p1##y,z,v)), \
 (I[84] = (img)(_p2##x,y,z,v)), \
 (I[90] = (img)(_p2##x,_n1##y,z,v)), \
 (I[96] = (img)(_p2##x,_n2##y,z,v)), \
 (I[102] = (img)(_p2##x,_n3##y,z,v)), \
 (I[108] = (img)(_p2##x,_p2##y,_n1##z,v)), \
 (I[114] = (img)(_p2##x,_p1##y,_n1##z,v)), \
 (I[120] = (img)(_p2##x,y,_n1##z,v)), \
 (I[126] = (img)(_p2##x,_n1##y,_n1##z,v)), \
 (I[132] = (img)(_p2##x,_n2##y,_n1##z,v)), \
 (I[138] = (img)(_p2##x,_n3##y,_n1##z,v)), \
 (I[144] = (img)(_p2##x,_p2##y,_n2##z,v)), \
 (I[150] = (img)(_p2##x,_p1##y,_n2##z,v)), \
 (I[156] = (img)(_p2##x,y,_n2##z,v)), \
 (I[162] = (img)(_p2##x,_n1##y,_n2##z,v)), \
 (I[168] = (img)(_p2##x,_n2##y,_n2##z,v)), \
 (I[174] = (img)(_p2##x,_n3##y,_n2##z,v)), \
 (I[180] = (img)(_p2##x,_p2##y,_n3##z,v)), \
 (I[186] = (img)(_p2##x,_p1##y,_n3##z,v)), \
 (I[192] = (img)(_p2##x,y,_n3##z,v)), \
 (I[198] = (img)(_p2##x,_n1##y,_n3##z,v)), \
 (I[204] = (img)(_p2##x,_n2##y,_n3##z,v)), \
 (I[210] = (img)(_p2##x,_n3##y,_n3##z,v)), \
 (I[1] = (img)(_p1##x,_p2##y,_p2##z,v)), \
 (I[7] = (img)(_p1##x,_p1##y,_p2##z,v)), \
 (I[13] = (img)(_p1##x,y,_p2##z,v)), \
 (I[19] = (img)(_p1##x,_n1##y,_p2##z,v)), \
 (I[25] = (img)(_p1##x,_n2##y,_p2##z,v)), \
 (I[31] = (img)(_p1##x,_n3##y,_p2##z,v)), \
 (I[37] = (img)(_p1##x,_p2##y,_p1##z,v)), \
 (I[43] = (img)(_p1##x,_p1##y,_p1##z,v)), \
 (I[49] = (img)(_p1##x,y,_p1##z,v)), \
 (I[55] = (img)(_p1##x,_n1##y,_p1##z,v)), \
 (I[61] = (img)(_p1##x,_n2##y,_p1##z,v)), \
 (I[67] = (img)(_p1##x,_n3##y,_p1##z,v)), \
 (I[73] = (img)(_p1##x,_p2##y,z,v)), \
 (I[79] = (img)(_p1##x,_p1##y,z,v)), \
 (I[85] = (img)(_p1##x,y,z,v)), \
 (I[91] = (img)(_p1##x,_n1##y,z,v)), \
 (I[97] = (img)(_p1##x,_n2##y,z,v)), \
 (I[103] = (img)(_p1##x,_n3##y,z,v)), \
 (I[109] = (img)(_p1##x,_p2##y,_n1##z,v)), \
 (I[115] = (img)(_p1##x,_p1##y,_n1##z,v)), \
 (I[121] = (img)(_p1##x,y,_n1##z,v)), \
 (I[127] = (img)(_p1##x,_n1##y,_n1##z,v)), \
 (I[133] = (img)(_p1##x,_n2##y,_n1##z,v)), \
 (I[139] = (img)(_p1##x,_n3##y,_n1##z,v)), \
 (I[145] = (img)(_p1##x,_p2##y,_n2##z,v)), \
 (I[151] = (img)(_p1##x,_p1##y,_n2##z,v)), \
 (I[157] = (img)(_p1##x,y,_n2##z,v)), \
 (I[163] = (img)(_p1##x,_n1##y,_n2##z,v)), \
 (I[169] = (img)(_p1##x,_n2##y,_n2##z,v)), \
 (I[175] = (img)(_p1##x,_n3##y,_n2##z,v)), \
 (I[181] = (img)(_p1##x,_p2##y,_n3##z,v)), \
 (I[187] = (img)(_p1##x,_p1##y,_n3##z,v)), \
 (I[193] = (img)(_p1##x,y,_n3##z,v)), \
 (I[199] = (img)(_p1##x,_n1##y,_n3##z,v)), \
 (I[205] = (img)(_p1##x,_n2##y,_n3##z,v)), \
 (I[211] = (img)(_p1##x,_n3##y,_n3##z,v)), \
 (I[2] = (img)(x,_p2##y,_p2##z,v)), \
 (I[8] = (img)(x,_p1##y,_p2##z,v)), \
 (I[14] = (img)(x,y,_p2##z,v)), \
 (I[20] = (img)(x,_n1##y,_p2##z,v)), \
 (I[26] = (img)(x,_n2##y,_p2##z,v)), \
 (I[32] = (img)(x,_n3##y,_p2##z,v)), \
 (I[38] = (img)(x,_p2##y,_p1##z,v)), \
 (I[44] = (img)(x,_p1##y,_p1##z,v)), \
 (I[50] = (img)(x,y,_p1##z,v)), \
 (I[56] = (img)(x,_n1##y,_p1##z,v)), \
 (I[62] = (img)(x,_n2##y,_p1##z,v)), \
 (I[68] = (img)(x,_n3##y,_p1##z,v)), \
 (I[74] = (img)(x,_p2##y,z,v)), \
 (I[80] = (img)(x,_p1##y,z,v)), \
 (I[86] = (img)(x,y,z,v)), \
 (I[92] = (img)(x,_n1##y,z,v)), \
 (I[98] = (img)(x,_n2##y,z,v)), \
 (I[104] = (img)(x,_n3##y,z,v)), \
 (I[110] = (img)(x,_p2##y,_n1##z,v)), \
 (I[116] = (img)(x,_p1##y,_n1##z,v)), \
 (I[122] = (img)(x,y,_n1##z,v)), \
 (I[128] = (img)(x,_n1##y,_n1##z,v)), \
 (I[134] = (img)(x,_n2##y,_n1##z,v)), \
 (I[140] = (img)(x,_n3##y,_n1##z,v)), \
 (I[146] = (img)(x,_p2##y,_n2##z,v)), \
 (I[152] = (img)(x,_p1##y,_n2##z,v)), \
 (I[158] = (img)(x,y,_n2##z,v)), \
 (I[164] = (img)(x,_n1##y,_n2##z,v)), \
 (I[170] = (img)(x,_n2##y,_n2##z,v)), \
 (I[176] = (img)(x,_n3##y,_n2##z,v)), \
 (I[182] = (img)(x,_p2##y,_n3##z,v)), \
 (I[188] = (img)(x,_p1##y,_n3##z,v)), \
 (I[194] = (img)(x,y,_n3##z,v)), \
 (I[200] = (img)(x,_n1##y,_n3##z,v)), \
 (I[206] = (img)(x,_n2##y,_n3##z,v)), \
 (I[212] = (img)(x,_n3##y,_n3##z,v)), \
 (I[3] = (img)(_n1##x,_p2##y,_p2##z,v)), \
 (I[9] = (img)(_n1##x,_p1##y,_p2##z,v)), \
 (I[15] = (img)(_n1##x,y,_p2##z,v)), \
 (I[21] = (img)(_n1##x,_n1##y,_p2##z,v)), \
 (I[27] = (img)(_n1##x,_n2##y,_p2##z,v)), \
 (I[33] = (img)(_n1##x,_n3##y,_p2##z,v)), \
 (I[39] = (img)(_n1##x,_p2##y,_p1##z,v)), \
 (I[45] = (img)(_n1##x,_p1##y,_p1##z,v)), \
 (I[51] = (img)(_n1##x,y,_p1##z,v)), \
 (I[57] = (img)(_n1##x,_n1##y,_p1##z,v)), \
 (I[63] = (img)(_n1##x,_n2##y,_p1##z,v)), \
 (I[69] = (img)(_n1##x,_n3##y,_p1##z,v)), \
 (I[75] = (img)(_n1##x,_p2##y,z,v)), \
 (I[81] = (img)(_n1##x,_p1##y,z,v)), \
 (I[87] = (img)(_n1##x,y,z,v)), \
 (I[93] = (img)(_n1##x,_n1##y,z,v)), \
 (I[99] = (img)(_n1##x,_n2##y,z,v)), \
 (I[105] = (img)(_n1##x,_n3##y,z,v)), \
 (I[111] = (img)(_n1##x,_p2##y,_n1##z,v)), \
 (I[117] = (img)(_n1##x,_p1##y,_n1##z,v)), \
 (I[123] = (img)(_n1##x,y,_n1##z,v)), \
 (I[129] = (img)(_n1##x,_n1##y,_n1##z,v)), \
 (I[135] = (img)(_n1##x,_n2##y,_n1##z,v)), \
 (I[141] = (img)(_n1##x,_n3##y,_n1##z,v)), \
 (I[147] = (img)(_n1##x,_p2##y,_n2##z,v)), \
 (I[153] = (img)(_n1##x,_p1##y,_n2##z,v)), \
 (I[159] = (img)(_n1##x,y,_n2##z,v)), \
 (I[165] = (img)(_n1##x,_n1##y,_n2##z,v)), \
 (I[171] = (img)(_n1##x,_n2##y,_n2##z,v)), \
 (I[177] = (img)(_n1##x,_n3##y,_n2##z,v)), \
 (I[183] = (img)(_n1##x,_p2##y,_n3##z,v)), \
 (I[189] = (img)(_n1##x,_p1##y,_n3##z,v)), \
 (I[195] = (img)(_n1##x,y,_n3##z,v)), \
 (I[201] = (img)(_n1##x,_n1##y,_n3##z,v)), \
 (I[207] = (img)(_n1##x,_n2##y,_n3##z,v)), \
 (I[213] = (img)(_n1##x,_n3##y,_n3##z,v)), \
 (I[4] = (img)(_n2##x,_p2##y,_p2##z,v)), \
 (I[10] = (img)(_n2##x,_p1##y,_p2##z,v)), \
 (I[16] = (img)(_n2##x,y,_p2##z,v)), \
 (I[22] = (img)(_n2##x,_n1##y,_p2##z,v)), \
 (I[28] = (img)(_n2##x,_n2##y,_p2##z,v)), \
 (I[34] = (img)(_n2##x,_n3##y,_p2##z,v)), \
 (I[40] = (img)(_n2##x,_p2##y,_p1##z,v)), \
 (I[46] = (img)(_n2##x,_p1##y,_p1##z,v)), \
 (I[52] = (img)(_n2##x,y,_p1##z,v)), \
 (I[58] = (img)(_n2##x,_n1##y,_p1##z,v)), \
 (I[64] = (img)(_n2##x,_n2##y,_p1##z,v)), \
 (I[70] = (img)(_n2##x,_n3##y,_p1##z,v)), \
 (I[76] = (img)(_n2##x,_p2##y,z,v)), \
 (I[82] = (img)(_n2##x,_p1##y,z,v)), \
 (I[88] = (img)(_n2##x,y,z,v)), \
 (I[94] = (img)(_n2##x,_n1##y,z,v)), \
 (I[100] = (img)(_n2##x,_n2##y,z,v)), \
 (I[106] = (img)(_n2##x,_n3##y,z,v)), \
 (I[112] = (img)(_n2##x,_p2##y,_n1##z,v)), \
 (I[118] = (img)(_n2##x,_p1##y,_n1##z,v)), \
 (I[124] = (img)(_n2##x,y,_n1##z,v)), \
 (I[130] = (img)(_n2##x,_n1##y,_n1##z,v)), \
 (I[136] = (img)(_n2##x,_n2##y,_n1##z,v)), \
 (I[142] = (img)(_n2##x,_n3##y,_n1##z,v)), \
 (I[148] = (img)(_n2##x,_p2##y,_n2##z,v)), \
 (I[154] = (img)(_n2##x,_p1##y,_n2##z,v)), \
 (I[160] = (img)(_n2##x,y,_n2##z,v)), \
 (I[166] = (img)(_n2##x,_n1##y,_n2##z,v)), \
 (I[172] = (img)(_n2##x,_n2##y,_n2##z,v)), \
 (I[178] = (img)(_n2##x,_n3##y,_n2##z,v)), \
 (I[184] = (img)(_n2##x,_p2##y,_n3##z,v)), \
 (I[190] = (img)(_n2##x,_p1##y,_n3##z,v)), \
 (I[196] = (img)(_n2##x,y,_n3##z,v)), \
 (I[202] = (img)(_n2##x,_n1##y,_n3##z,v)), \
 (I[208] = (img)(_n2##x,_n2##y,_n3##z,v)), \
 (I[214] = (img)(_n2##x,_n3##y,_n3##z,v)), \
 x+3>=(int)((img).width)?(int)((img).width)-1:x+3); \
 x<=(int)(x1) && ((_n3##x<(int)((img).width) && ( \
 (I[5] = (img)(_n3##x,_p2##y,_p2##z,v)), \
 (I[11] = (img)(_n3##x,_p1##y,_p2##z,v)), \
 (I[17] = (img)(_n3##x,y,_p2##z,v)), \
 (I[23] = (img)(_n3##x,_n1##y,_p2##z,v)), \
 (I[29] = (img)(_n3##x,_n2##y,_p2##z,v)), \
 (I[35] = (img)(_n3##x,_n3##y,_p2##z,v)), \
 (I[41] = (img)(_n3##x,_p2##y,_p1##z,v)), \
 (I[47] = (img)(_n3##x,_p1##y,_p1##z,v)), \
 (I[53] = (img)(_n3##x,y,_p1##z,v)), \
 (I[59] = (img)(_n3##x,_n1##y,_p1##z,v)), \
 (I[65] = (img)(_n3##x,_n2##y,_p1##z,v)), \
 (I[71] = (img)(_n3##x,_n3##y,_p1##z,v)), \
 (I[77] = (img)(_n3##x,_p2##y,z,v)), \
 (I[83] = (img)(_n3##x,_p1##y,z,v)), \
 (I[89] = (img)(_n3##x,y,z,v)), \
 (I[95] = (img)(_n3##x,_n1##y,z,v)), \
 (I[101] = (img)(_n3##x,_n2##y,z,v)), \
 (I[107] = (img)(_n3##x,_n3##y,z,v)), \
 (I[113] = (img)(_n3##x,_p2##y,_n1##z,v)), \
 (I[119] = (img)(_n3##x,_p1##y,_n1##z,v)), \
 (I[125] = (img)(_n3##x,y,_n1##z,v)), \
 (I[131] = (img)(_n3##x,_n1##y,_n1##z,v)), \
 (I[137] = (img)(_n3##x,_n2##y,_n1##z,v)), \
 (I[143] = (img)(_n3##x,_n3##y,_n1##z,v)), \
 (I[149] = (img)(_n3##x,_p2##y,_n2##z,v)), \
 (I[155] = (img)(_n3##x,_p1##y,_n2##z,v)), \
 (I[161] = (img)(_n3##x,y,_n2##z,v)), \
 (I[167] = (img)(_n3##x,_n1##y,_n2##z,v)), \
 (I[173] = (img)(_n3##x,_n2##y,_n2##z,v)), \
 (I[179] = (img)(_n3##x,_n3##y,_n2##z,v)), \
 (I[185] = (img)(_n3##x,_p2##y,_n3##z,v)), \
 (I[191] = (img)(_n3##x,_p1##y,_n3##z,v)), \
 (I[197] = (img)(_n3##x,y,_n3##z,v)), \
 (I[203] = (img)(_n3##x,_n1##y,_n3##z,v)), \
 (I[209] = (img)(_n3##x,_n2##y,_n3##z,v)), \
 (I[215] = (img)(_n3##x,_n3##y,_n3##z,v)),1)) || \
 _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], \
 I[6] = I[7], I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], \
 I[12] = I[13], I[13] = I[14], I[14] = I[15], I[15] = I[16], I[16] = I[17], \
 I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], \
 I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], \
 I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], I[34] = I[35], \
 I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], \
 I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], \
 I[54] = I[55], I[55] = I[56], I[56] = I[57], I[57] = I[58], I[58] = I[59], \
 I[60] = I[61], I[61] = I[62], I[62] = I[63], I[63] = I[64], I[64] = I[65], \
 I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], \
 I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], \
 I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], \
 I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], \
 I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], \
 I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], \
 I[102] = I[103], I[103] = I[104], I[104] = I[105], I[105] = I[106], I[106] = I[107], \
 I[108] = I[109], I[109] = I[110], I[110] = I[111], I[111] = I[112], I[112] = I[113], \
 I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], \
 I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], \
 I[132] = I[133], I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], \
 I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], \
 I[150] = I[151], I[151] = I[152], I[152] = I[153], I[153] = I[154], I[154] = I[155], \
 I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], I[160] = I[161], \
 I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], \
 I[174] = I[175], I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], \
 I[180] = I[181], I[181] = I[182], I[182] = I[183], I[183] = I[184], I[184] = I[185], \
 I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], \
 I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], \
 I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], I[202] = I[203], \
 I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], \
 I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], \
 _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x)

#define cimg_get6x6x6(img,x,y,z,v,I) \
 I[0] = (img)(_p2##x,_p2##y,_p2##z,v), I[1] = (img)(_p1##x,_p2##y,_p2##z,v), I[2] = (img)(x,_p2##y,_p2##z,v), I[3] = (img)(_n1##x,_p2##y,_p2##z,v), I[4] = (img)(_n2##x,_p2##y,_p2##z,v), I[5] = (img)(_n3##x,_p2##y,_p2##z,v), \
 I[6] = (img)(_p2##x,_p1##y,_p2##z,v), I[7] = (img)(_p1##x,_p1##y,_p2##z,v), I[8] = (img)(x,_p1##y,_p2##z,v), I[9] = (img)(_n1##x,_p1##y,_p2##z,v), I[10] = (img)(_n2##x,_p1##y,_p2##z,v), I[11] = (img)(_n3##x,_p1##y,_p2##z,v), \
 I[12] = (img)(_p2##x,y,_p2##z,v), I[13] = (img)(_p1##x,y,_p2##z,v), I[14] = (img)(x,y,_p2##z,v), I[15] = (img)(_n1##x,y,_p2##z,v), I[16] = (img)(_n2##x,y,_p2##z,v), I[17] = (img)(_n3##x,y,_p2##z,v), \
 I[18] = (img)(_p2##x,_n1##y,_p2##z,v), I[19] = (img)(_p1##x,_n1##y,_p2##z,v), I[20] = (img)(x,_n1##y,_p2##z,v), I[21] = (img)(_n1##x,_n1##y,_p2##z,v), I[22] = (img)(_n2##x,_n1##y,_p2##z,v), I[23] = (img)(_n3##x,_n1##y,_p2##z,v), \
 I[24] = (img)(_p2##x,_n2##y,_p2##z,v), I[25] = (img)(_p1##x,_n2##y,_p2##z,v), I[26] = (img)(x,_n2##y,_p2##z,v), I[27] = (img)(_n1##x,_n2##y,_p2##z,v), I[28] = (img)(_n2##x,_n2##y,_p2##z,v), I[29] = (img)(_n3##x,_n2##y,_p2##z,v), \
 I[30] = (img)(_p2##x,_n3##y,_p2##z,v), I[31] = (img)(_p1##x,_n3##y,_p2##z,v), I[32] = (img)(x,_n3##y,_p2##z,v), I[33] = (img)(_n1##x,_n3##y,_p2##z,v), I[34] = (img)(_n2##x,_n3##y,_p2##z,v), I[35] = (img)(_n3##x,_n3##y,_p2##z,v), \
 I[36] = (img)(_p2##x,_p2##y,_p1##z,v), I[37] = (img)(_p1##x,_p2##y,_p1##z,v), I[38] = (img)(x,_p2##y,_p1##z,v), I[39] = (img)(_n1##x,_p2##y,_p1##z,v), I[40] = (img)(_n2##x,_p2##y,_p1##z,v), I[41] = (img)(_n3##x,_p2##y,_p1##z,v), \
 I[42] = (img)(_p2##x,_p1##y,_p1##z,v), I[43] = (img)(_p1##x,_p1##y,_p1##z,v), I[44] = (img)(x,_p1##y,_p1##z,v), I[45] = (img)(_n1##x,_p1##y,_p1##z,v), I[46] = (img)(_n2##x,_p1##y,_p1##z,v), I[47] = (img)(_n3##x,_p1##y,_p1##z,v), \
 I[48] = (img)(_p2##x,y,_p1##z,v), I[49] = (img)(_p1##x,y,_p1##z,v), I[50] = (img)(x,y,_p1##z,v), I[51] = (img)(_n1##x,y,_p1##z,v), I[52] = (img)(_n2##x,y,_p1##z,v), I[53] = (img)(_n3##x,y,_p1##z,v), \
 I[54] = (img)(_p2##x,_n1##y,_p1##z,v), I[55] = (img)(_p1##x,_n1##y,_p1##z,v), I[56] = (img)(x,_n1##y,_p1##z,v), I[57] = (img)(_n1##x,_n1##y,_p1##z,v), I[58] = (img)(_n2##x,_n1##y,_p1##z,v), I[59] = (img)(_n3##x,_n1##y,_p1##z,v), \
 I[60] = (img)(_p2##x,_n2##y,_p1##z,v), I[61] = (img)(_p1##x,_n2##y,_p1##z,v), I[62] = (img)(x,_n2##y,_p1##z,v), I[63] = (img)(_n1##x,_n2##y,_p1##z,v), I[64] = (img)(_n2##x,_n2##y,_p1##z,v), I[65] = (img)(_n3##x,_n2##y,_p1##z,v), \
 I[66] = (img)(_p2##x,_n3##y,_p1##z,v), I[67] = (img)(_p1##x,_n3##y,_p1##z,v), I[68] = (img)(x,_n3##y,_p1##z,v), I[69] = (img)(_n1##x,_n3##y,_p1##z,v), I[70] = (img)(_n2##x,_n3##y,_p1##z,v), I[71] = (img)(_n3##x,_n3##y,_p1##z,v), \
 I[72] = (img)(_p2##x,_p2##y,z,v), I[73] = (img)(_p1##x,_p2##y,z,v), I[74] = (img)(x,_p2##y,z,v), I[75] = (img)(_n1##x,_p2##y,z,v), I[76] = (img)(_n2##x,_p2##y,z,v), I[77] = (img)(_n3##x,_p2##y,z,v), \
 I[78] = (img)(_p2##x,_p1##y,z,v), I[79] = (img)(_p1##x,_p1##y,z,v), I[80] = (img)(x,_p1##y,z,v), I[81] = (img)(_n1##x,_p1##y,z,v), I[82] = (img)(_n2##x,_p1##y,z,v), I[83] = (img)(_n3##x,_p1##y,z,v), \
 I[84] = (img)(_p2##x,y,z,v), I[85] = (img)(_p1##x,y,z,v), I[86] = (img)(x,y,z,v), I[87] = (img)(_n1##x,y,z,v), I[88] = (img)(_n2##x,y,z,v), I[89] = (img)(_n3##x,y,z,v), \
 I[90] = (img)(_p2##x,_n1##y,z,v), I[91] = (img)(_p1##x,_n1##y,z,v), I[92] = (img)(x,_n1##y,z,v), I[93] = (img)(_n1##x,_n1##y,z,v), I[94] = (img)(_n2##x,_n1##y,z,v), I[95] = (img)(_n3##x,_n1##y,z,v), \
 I[96] = (img)(_p2##x,_n2##y,z,v), I[97] = (img)(_p1##x,_n2##y,z,v), I[98] = (img)(x,_n2##y,z,v), I[99] = (img)(_n1##x,_n2##y,z,v), I[100] = (img)(_n2##x,_n2##y,z,v), I[101] = (img)(_n3##x,_n2##y,z,v), \
 I[102] = (img)(_p2##x,_n3##y,z,v), I[103] = (img)(_p1##x,_n3##y,z,v), I[104] = (img)(x,_n3##y,z,v), I[105] = (img)(_n1##x,_n3##y,z,v), I[106] = (img)(_n2##x,_n3##y,z,v), I[107] = (img)(_n3##x,_n3##y,z,v), \
 I[108] = (img)(_p2##x,_p2##y,_n1##z,v), I[109] = (img)(_p1##x,_p2##y,_n1##z,v), I[110] = (img)(x,_p2##y,_n1##z,v), I[111] = (img)(_n1##x,_p2##y,_n1##z,v), I[112] = (img)(_n2##x,_p2##y,_n1##z,v), I[113] = (img)(_n3##x,_p2##y,_n1##z,v), \
 I[114] = (img)(_p2##x,_p1##y,_n1##z,v), I[115] = (img)(_p1##x,_p1##y,_n1##z,v), I[116] = (img)(x,_p1##y,_n1##z,v), I[117] = (img)(_n1##x,_p1##y,_n1##z,v), I[118] = (img)(_n2##x,_p1##y,_n1##z,v), I[119] = (img)(_n3##x,_p1##y,_n1##z,v), \
 I[120] = (img)(_p2##x,y,_n1##z,v), I[121] = (img)(_p1##x,y,_n1##z,v), I[122] = (img)(x,y,_n1##z,v), I[123] = (img)(_n1##x,y,_n1##z,v), I[124] = (img)(_n2##x,y,_n1##z,v), I[125] = (img)(_n3##x,y,_n1##z,v), \
 I[126] = (img)(_p2##x,_n1##y,_n1##z,v), I[127] = (img)(_p1##x,_n1##y,_n1##z,v), I[128] = (img)(x,_n1##y,_n1##z,v), I[129] = (img)(_n1##x,_n1##y,_n1##z,v), I[130] = (img)(_n2##x,_n1##y,_n1##z,v), I[131] = (img)(_n3##x,_n1##y,_n1##z,v), \
 I[132] = (img)(_p2##x,_n2##y,_n1##z,v), I[133] = (img)(_p1##x,_n2##y,_n1##z,v), I[134] = (img)(x,_n2##y,_n1##z,v), I[135] = (img)(_n1##x,_n2##y,_n1##z,v), I[136] = (img)(_n2##x,_n2##y,_n1##z,v), I[137] = (img)(_n3##x,_n2##y,_n1##z,v), \
 I[138] = (img)(_p2##x,_n3##y,_n1##z,v), I[139] = (img)(_p1##x,_n3##y,_n1##z,v), I[140] = (img)(x,_n3##y,_n1##z,v), I[141] = (img)(_n1##x,_n3##y,_n1##z,v), I[142] = (img)(_n2##x,_n3##y,_n1##z,v), I[143] = (img)(_n3##x,_n3##y,_n1##z,v), \
 I[144] = (img)(_p2##x,_p2##y,_n2##z,v), I[145] = (img)(_p1##x,_p2##y,_n2##z,v), I[146] = (img)(x,_p2##y,_n2##z,v), I[147] = (img)(_n1##x,_p2##y,_n2##z,v), I[148] = (img)(_n2##x,_p2##y,_n2##z,v), I[149] = (img)(_n3##x,_p2##y,_n2##z,v), \
 I[150] = (img)(_p2##x,_p1##y,_n2##z,v), I[151] = (img)(_p1##x,_p1##y,_n2##z,v), I[152] = (img)(x,_p1##y,_n2##z,v), I[153] = (img)(_n1##x,_p1##y,_n2##z,v), I[154] = (img)(_n2##x,_p1##y,_n2##z,v), I[155] = (img)(_n3##x,_p1##y,_n2##z,v), \
 I[156] = (img)(_p2##x,y,_n2##z,v), I[157] = (img)(_p1##x,y,_n2##z,v), I[158] = (img)(x,y,_n2##z,v), I[159] = (img)(_n1##x,y,_n2##z,v), I[160] = (img)(_n2##x,y,_n2##z,v), I[161] = (img)(_n3##x,y,_n2##z,v), \
 I[162] = (img)(_p2##x,_n1##y,_n2##z,v), I[163] = (img)(_p1##x,_n1##y,_n2##z,v), I[164] = (img)(x,_n1##y,_n2##z,v), I[165] = (img)(_n1##x,_n1##y,_n2##z,v), I[166] = (img)(_n2##x,_n1##y,_n2##z,v), I[167] = (img)(_n3##x,_n1##y,_n2##z,v), \
 I[168] = (img)(_p2##x,_n2##y,_n2##z,v), I[169] = (img)(_p1##x,_n2##y,_n2##z,v), I[170] = (img)(x,_n2##y,_n2##z,v), I[171] = (img)(_n1##x,_n2##y,_n2##z,v), I[172] = (img)(_n2##x,_n2##y,_n2##z,v), I[173] = (img)(_n3##x,_n2##y,_n2##z,v), \
 I[174] = (img)(_p2##x,_n3##y,_n2##z,v), I[175] = (img)(_p1##x,_n3##y,_n2##z,v), I[176] = (img)(x,_n3##y,_n2##z,v), I[177] = (img)(_n1##x,_n3##y,_n2##z,v), I[178] = (img)(_n2##x,_n3##y,_n2##z,v), I[179] = (img)(_n3##x,_n3##y,_n2##z,v), \
 I[180] = (img)(_p2##x,_p2##y,_n3##z,v), I[181] = (img)(_p1##x,_p2##y,_n3##z,v), I[182] = (img)(x,_p2##y,_n3##z,v), I[183] = (img)(_n1##x,_p2##y,_n3##z,v), I[184] = (img)(_n2##x,_p2##y,_n3##z,v), I[185] = (img)(_n3##x,_p2##y,_n3##z,v), \
 I[186] = (img)(_p2##x,_p1##y,_n3##z,v), I[187] = (img)(_p1##x,_p1##y,_n3##z,v), I[188] = (img)(x,_p1##y,_n3##z,v), I[189] = (img)(_n1##x,_p1##y,_n3##z,v), I[190] = (img)(_n2##x,_p1##y,_n3##z,v), I[191] = (img)(_n3##x,_p1##y,_n3##z,v), \
 I[192] = (img)(_p2##x,y,_n3##z,v), I[193] = (img)(_p1##x,y,_n3##z,v), I[194] = (img)(x,y,_n3##z,v), I[195] = (img)(_n1##x,y,_n3##z,v), I[196] = (img)(_n2##x,y,_n3##z,v), I[197] = (img)(_n3##x,y,_n3##z,v), \
 I[198] = (img)(_p2##x,_n1##y,_n3##z,v), I[199] = (img)(_p1##x,_n1##y,_n3##z,v), I[200] = (img)(x,_n1##y,_n3##z,v), I[201] = (img)(_n1##x,_n1##y,_n3##z,v), I[202] = (img)(_n2##x,_n1##y,_n3##z,v), I[203] = (img)(_n3##x,_n1##y,_n3##z,v), \
 I[204] = (img)(_p2##x,_n2##y,_n3##z,v), I[205] = (img)(_p1##x,_n2##y,_n3##z,v), I[206] = (img)(x,_n2##y,_n3##z,v), I[207] = (img)(_n1##x,_n2##y,_n3##z,v), I[208] = (img)(_n2##x,_n2##y,_n3##z,v), I[209] = (img)(_n3##x,_n2##y,_n3##z,v), \
 I[210] = (img)(_p2##x,_n3##y,_n3##z,v), I[211] = (img)(_p1##x,_n3##y,_n3##z,v), I[212] = (img)(x,_n3##y,_n3##z,v), I[213] = (img)(_n1##x,_n3##y,_n3##z,v), I[214] = (img)(_n2##x,_n3##y,_n3##z,v), I[215] = (img)(_n3##x,_n3##y,_n3##z,v);

// Define 7x7x7 loop macros for CImg
//-------------------------------------
#define cimg_for_in7(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3; \
 i<=(int)(i1) && (_n3##i<(int)(bound) || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n3##i = _n2##i = --_n1##i)); \
 _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i)

#define cimg_for_in7X(img,x0,x1,x) cimg_for_in7((img).width,x0,x1,x)
#define cimg_for_in7Y(img,y0,y1,y) cimg_for_in7((img).height,y0,y1,y)
#define cimg_for_in7Z(img,z0,z1,z) cimg_for_in7((img).depth,z0,z1,z)
#define cimg_for_in7V(img,v0,v1,v) cimg_for_in7((img).dim,v0,v1,v)
#define cimg_for_in7XY(img,x0,y0,x1,y1,x,y) cimg_for_in7Y(img,y0,y1,y) cimg_for_in7X(img,x0,x1,x)
#define cimg_for_in7XZ(img,x0,z0,x1,z1,x,z) cimg_for_in7Z(img,z0,z1,z) cimg_for_in7X(img,x0,x1,x)
#define cimg_for_in7XV(img,x0,v0,x1,v1,x,v) cimg_for_in7V(img,v0,v1,v) cimg_for_in7X(img,x0,x1,x)
#define cimg_for_in7YZ(img,y0,z0,y1,z1,y,z) cimg_for_in7Z(img,z0,z1,z) cimg_for_in7Y(img,y0,y1,y)
#define cimg_for_in7YV(img,y0,v0,y1,v1,y,v) cimg_for_in7V(img,v0,v1,v) cimg_for_in7Y(img,y0,y1,y)
#define cimg_for_in7ZV(img,z0,v0,z1,v1,z,v) cimg_for_in7V(img,v0,v1,v) cimg_for_in7Z(img,z0,z1,z)
#define cimg_for_in7XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in7Z(img,z0,z1,z) cimg_for_in7XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in7XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in7V(img,v0,v1,v) cimg_for_in7XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in7YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in7V(img,v0,v1,v) cimg_for_in7YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in7XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in7V(img,v0,v1,v) cimg_for_in7XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for7x7x7(img,x,y,z,v,I) \
 cimg_for7((img).depth,z) cimg_for7((img).height,y) for (int x = 0, \
 _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = (img)(0,_p3##y,_p3##z,v)), \
 (I[7] = I[8] = I[9] = I[10] = (img)(0,_p2##y,_p3##z,v)), \
 (I[14] = I[15] = I[16] = I[17] = (img)(0,_p1##y,_p3##z,v)), \
 (I[21] = I[22] = I[23] = I[24] = (img)(0,y,_p3##z,v)), \
 (I[28] = I[29] = I[30] = I[31] = (img)(0,_n1##y,_p3##z,v)), \
 (I[35] = I[36] = I[37] = I[38] = (img)(0,_n2##y,_p3##z,v)), \
 (I[42] = I[43] = I[44] = I[45] = (img)(0,_n3##y,_p3##z,v)), \
 (I[49] = I[50] = I[51] = I[52] = (img)(0,_p3##y,_p2##z,v)), \
 (I[56] = I[57] = I[58] = I[59] = (img)(0,_p2##y,_p2##z,v)), \
 (I[63] = I[64] = I[65] = I[66] = (img)(0,_p1##y,_p2##z,v)), \
 (I[70] = I[71] = I[72] = I[73] = (img)(0,y,_p2##z,v)), \
 (I[77] = I[78] = I[79] = I[80] = (img)(0,_n1##y,_p2##z,v)), \
 (I[84] = I[85] = I[86] = I[87] = (img)(0,_n2##y,_p2##z,v)), \
 (I[91] = I[92] = I[93] = I[94] = (img)(0,_n3##y,_p2##z,v)), \
 (I[98] = I[99] = I[100] = I[101] = (img)(0,_p3##y,_p1##z,v)), \
 (I[105] = I[106] = I[107] = I[108] = (img)(0,_p2##y,_p1##z,v)), \
 (I[112] = I[113] = I[114] = I[115] = (img)(0,_p1##y,_p1##z,v)), \
 (I[119] = I[120] = I[121] = I[122] = (img)(0,y,_p1##z,v)), \
 (I[126] = I[127] = I[128] = I[129] = (img)(0,_n1##y,_p1##z,v)), \
 (I[133] = I[134] = I[135] = I[136] = (img)(0,_n2##y,_p1##z,v)), \
 (I[140] = I[141] = I[142] = I[143] = (img)(0,_n3##y,_p1##z,v)), \
 (I[147] = I[148] = I[149] = I[150] = (img)(0,_p3##y,z,v)), \
 (I[154] = I[155] = I[156] = I[157] = (img)(0,_p2##y,z,v)), \
 (I[161] = I[162] = I[163] = I[164] = (img)(0,_p1##y,z,v)), \
 (I[168] = I[169] = I[170] = I[171] = (img)(0,y,z,v)), \
 (I[175] = I[176] = I[177] = I[178] = (img)(0,_n1##y,z,v)), \
 (I[182] = I[183] = I[184] = I[185] = (img)(0,_n2##y,z,v)), \
 (I[189] = I[190] = I[191] = I[192] = (img)(0,_n3##y,z,v)), \
 (I[196] = I[197] = I[198] = I[199] = (img)(0,_p3##y,_n1##z,v)), \
 (I[203] = I[204] = I[205] = I[206] = (img)(0,_p2##y,_n1##z,v)), \
 (I[210] = I[211] = I[212] = I[213] = (img)(0,_p1##y,_n1##z,v)), \
 (I[217] = I[218] = I[219] = I[220] = (img)(0,y,_n1##z,v)), \
 (I[224] = I[225] = I[226] = I[227] = (img)(0,_n1##y,_n1##z,v)), \
 (I[231] = I[232] = I[233] = I[234] = (img)(0,_n2##y,_n1##z,v)), \
 (I[238] = I[239] = I[240] = I[241] = (img)(0,_n3##y,_n1##z,v)), \
 (I[245] = I[246] = I[247] = I[248] = (img)(0,_p3##y,_n2##z,v)), \
 (I[252] = I[253] = I[254] = I[255] = (img)(0,_p2##y,_n2##z,v)), \
 (I[259] = I[260] = I[261] = I[262] = (img)(0,_p1##y,_n2##z,v)), \
 (I[266] = I[267] = I[268] = I[269] = (img)(0,y,_n2##z,v)), \
 (I[273] = I[274] = I[275] = I[276] = (img)(0,_n1##y,_n2##z,v)), \
 (I[280] = I[281] = I[282] = I[283] = (img)(0,_n2##y,_n2##z,v)), \
 (I[287] = I[288] = I[289] = I[290] = (img)(0,_n3##y,_n2##z,v)), \
 (I[294] = I[295] = I[296] = I[297] = (img)(0,_p3##y,_n3##z,v)), \
 (I[301] = I[302] = I[303] = I[304] = (img)(0,_p2##y,_n3##z,v)), \
 (I[308] = I[309] = I[310] = I[311] = (img)(0,_p1##y,_n3##z,v)), \
 (I[315] = I[316] = I[317] = I[318] = (img)(0,y,_n3##z,v)), \
 (I[322] = I[323] = I[324] = I[325] = (img)(0,_n1##y,_n3##z,v)), \
 (I[329] = I[330] = I[331] = I[332] = (img)(0,_n2##y,_n3##z,v)), \
 (I[336] = I[337] = I[338] = I[339] = (img)(0,_n3##y,_n3##z,v)), \
 (I[4] = (img)(_n1##x,_p3##y,_p3##z,v)), \
 (I[11] = (img)(_n1##x,_p2##y,_p3##z,v)), \
 (I[18] = (img)(_n1##x,_p1##y,_p3##z,v)), \
 (I[25] = (img)(_n1##x,y,_p3##z,v)), \
 (I[32] = (img)(_n1##x,_n1##y,_p3##z,v)), \
 (I[39] = (img)(_n1##x,_n2##y,_p3##z,v)), \
 (I[46] = (img)(_n1##x,_n3##y,_p3##z,v)), \
 (I[53] = (img)(_n1##x,_p3##y,_p2##z,v)), \
 (I[60] = (img)(_n1##x,_p2##y,_p2##z,v)), \
 (I[67] = (img)(_n1##x,_p1##y,_p2##z,v)), \
 (I[74] = (img)(_n1##x,y,_p2##z,v)), \
 (I[81] = (img)(_n1##x,_n1##y,_p2##z,v)), \
 (I[88] = (img)(_n1##x,_n2##y,_p2##z,v)), \
 (I[95] = (img)(_n1##x,_n3##y,_p2##z,v)), \
 (I[102] = (img)(_n1##x,_p3##y,_p1##z,v)), \
 (I[109] = (img)(_n1##x,_p2##y,_p1##z,v)), \
 (I[116] = (img)(_n1##x,_p1##y,_p1##z,v)), \
 (I[123] = (img)(_n1##x,y,_p1##z,v)), \
 (I[130] = (img)(_n1##x,_n1##y,_p1##z,v)), \
 (I[137] = (img)(_n1##x,_n2##y,_p1##z,v)), \
 (I[144] = (img)(_n1##x,_n3##y,_p1##z,v)), \
 (I[151] = (img)(_n1##x,_p3##y,z,v)), \
 (I[158] = (img)(_n1##x,_p2##y,z,v)), \
 (I[165] = (img)(_n1##x,_p1##y,z,v)), \
 (I[172] = (img)(_n1##x,y,z,v)), \
 (I[179] = (img)(_n1##x,_n1##y,z,v)), \
 (I[186] = (img)(_n1##x,_n2##y,z,v)), \
 (I[193] = (img)(_n1##x,_n3##y,z,v)), \
 (I[200] = (img)(_n1##x,_p3##y,_n1##z,v)), \
 (I[207] = (img)(_n1##x,_p2##y,_n1##z,v)), \
 (I[214] = (img)(_n1##x,_p1##y,_n1##z,v)), \
 (I[221] = (img)(_n1##x,y,_n1##z,v)), \
 (I[228] = (img)(_n1##x,_n1##y,_n1##z,v)), \
 (I[235] = (img)(_n1##x,_n2##y,_n1##z,v)), \
 (I[242] = (img)(_n1##x,_n3##y,_n1##z,v)), \
 (I[249] = (img)(_n1##x,_p3##y,_n2##z,v)), \
 (I[256] = (img)(_n1##x,_p2##y,_n2##z,v)), \
 (I[263] = (img)(_n1##x,_p1##y,_n2##z,v)), \
 (I[270] = (img)(_n1##x,y,_n2##z,v)), \
 (I[277] = (img)(_n1##x,_n1##y,_n2##z,v)), \
 (I[284] = (img)(_n1##x,_n2##y,_n2##z,v)), \
 (I[291] = (img)(_n1##x,_n3##y,_n2##z,v)), \
 (I[298] = (img)(_n1##x,_p3##y,_n3##z,v)), \
 (I[305] = (img)(_n1##x,_p2##y,_n3##z,v)), \
 (I[312] = (img)(_n1##x,_p1##y,_n3##z,v)), \
 (I[319] = (img)(_n1##x,y,_n3##z,v)), \
 (I[326] = (img)(_n1##x,_n1##y,_n3##z,v)), \
 (I[333] = (img)(_n1##x,_n2##y,_n3##z,v)), \
 (I[340] = (img)(_n1##x,_n3##y,_n3##z,v)), \
 (I[5] = (img)(_n2##x,_p3##y,_p3##z,v)), \
 (I[12] = (img)(_n2##x,_p2##y,_p3##z,v)), \
 (I[19] = (img)(_n2##x,_p1##y,_p3##z,v)), \
 (I[26] = (img)(_n2##x,y,_p3##z,v)), \
 (I[33] = (img)(_n2##x,_n1##y,_p3##z,v)), \
 (I[40] = (img)(_n2##x,_n2##y,_p3##z,v)), \
 (I[47] = (img)(_n2##x,_n3##y,_p3##z,v)), \
 (I[54] = (img)(_n2##x,_p3##y,_p2##z,v)), \
 (I[61] = (img)(_n2##x,_p2##y,_p2##z,v)), \
 (I[68] = (img)(_n2##x,_p1##y,_p2##z,v)), \
 (I[75] = (img)(_n2##x,y,_p2##z,v)), \
 (I[82] = (img)(_n2##x,_n1##y,_p2##z,v)), \
 (I[89] = (img)(_n2##x,_n2##y,_p2##z,v)), \
 (I[96] = (img)(_n2##x,_n3##y,_p2##z,v)), \
 (I[103] = (img)(_n2##x,_p3##y,_p1##z,v)), \
 (I[110] = (img)(_n2##x,_p2##y,_p1##z,v)), \
 (I[117] = (img)(_n2##x,_p1##y,_p1##z,v)), \
 (I[124] = (img)(_n2##x,y,_p1##z,v)), \
 (I[131] = (img)(_n2##x,_n1##y,_p1##z,v)), \
 (I[138] = (img)(_n2##x,_n2##y,_p1##z,v)), \
 (I[145] = (img)(_n2##x,_n3##y,_p1##z,v)), \
 (I[152] = (img)(_n2##x,_p3##y,z,v)), \
 (I[159] = (img)(_n2##x,_p2##y,z,v)), \
 (I[166] = (img)(_n2##x,_p1##y,z,v)), \
 (I[173] = (img)(_n2##x,y,z,v)), \
 (I[180] = (img)(_n2##x,_n1##y,z,v)), \
 (I[187] = (img)(_n2##x,_n2##y,z,v)), \
 (I[194] = (img)(_n2##x,_n3##y,z,v)), \
 (I[201] = (img)(_n2##x,_p3##y,_n1##z,v)), \
 (I[208] = (img)(_n2##x,_p2##y,_n1##z,v)), \
 (I[215] = (img)(_n2##x,_p1##y,_n1##z,v)), \
 (I[222] = (img)(_n2##x,y,_n1##z,v)), \
 (I[229] = (img)(_n2##x,_n1##y,_n1##z,v)), \
 (I[236] = (img)(_n2##x,_n2##y,_n1##z,v)), \
 (I[243] = (img)(_n2##x,_n3##y,_n1##z,v)), \
 (I[250] = (img)(_n2##x,_p3##y,_n2##z,v)), \
 (I[257] = (img)(_n2##x,_p2##y,_n2##z,v)), \
 (I[264] = (img)(_n2##x,_p1##y,_n2##z,v)), \
 (I[271] = (img)(_n2##x,y,_n2##z,v)), \
 (I[278] = (img)(_n2##x,_n1##y,_n2##z,v)), \
 (I[285] = (img)(_n2##x,_n2##y,_n2##z,v)), \
 (I[292] = (img)(_n2##x,_n3##y,_n2##z,v)), \
 (I[299] = (img)(_n2##x,_p3##y,_n3##z,v)), \
 (I[306] = (img)(_n2##x,_p2##y,_n3##z,v)), \
 (I[313] = (img)(_n2##x,_p1##y,_n3##z,v)), \
 (I[320] = (img)(_n2##x,y,_n3##z,v)), \
 (I[327] = (img)(_n2##x,_n1##y,_n3##z,v)), \
 (I[334] = (img)(_n2##x,_n2##y,_n3##z,v)), \
 (I[341] = (img)(_n2##x,_n3##y,_n3##z,v)), \
 3>=((img).width)?(int)((img).width)-1:3); \
 (_n3##x<(int)((img).width) && ( \
 (I[6] = (img)(_n3##x,_p3##y,_p3##z,v)), \
 (I[13] = (img)(_n3##x,_p2##y,_p3##z,v)), \
 (I[20] = (img)(_n3##x,_p1##y,_p3##z,v)), \
 (I[27] = (img)(_n3##x,y,_p3##z,v)), \
 (I[34] = (img)(_n3##x,_n1##y,_p3##z,v)), \
 (I[41] = (img)(_n3##x,_n2##y,_p3##z,v)), \
 (I[48] = (img)(_n3##x,_n3##y,_p3##z,v)), \
 (I[55] = (img)(_n3##x,_p3##y,_p2##z,v)), \
 (I[62] = (img)(_n3##x,_p2##y,_p2##z,v)), \
 (I[69] = (img)(_n3##x,_p1##y,_p2##z,v)), \
 (I[76] = (img)(_n3##x,y,_p2##z,v)), \
 (I[83] = (img)(_n3##x,_n1##y,_p2##z,v)), \
 (I[90] = (img)(_n3##x,_n2##y,_p2##z,v)), \
 (I[97] = (img)(_n3##x,_n3##y,_p2##z,v)), \
 (I[104] = (img)(_n3##x,_p3##y,_p1##z,v)), \
 (I[111] = (img)(_n3##x,_p2##y,_p1##z,v)), \
 (I[118] = (img)(_n3##x,_p1##y,_p1##z,v)), \
 (I[125] = (img)(_n3##x,y,_p1##z,v)), \
 (I[132] = (img)(_n3##x,_n1##y,_p1##z,v)), \
 (I[139] = (img)(_n3##x,_n2##y,_p1##z,v)), \
 (I[146] = (img)(_n3##x,_n3##y,_p1##z,v)), \
 (I[153] = (img)(_n3##x,_p3##y,z,v)), \
 (I[160] = (img)(_n3##x,_p2##y,z,v)), \
 (I[167] = (img)(_n3##x,_p1##y,z,v)), \
 (I[174] = (img)(_n3##x,y,z,v)), \
 (I[181] = (img)(_n3##x,_n1##y,z,v)), \
 (I[188] = (img)(_n3##x,_n2##y,z,v)), \
 (I[195] = (img)(_n3##x,_n3##y,z,v)), \
 (I[202] = (img)(_n3##x,_p3##y,_n1##z,v)), \
 (I[209] = (img)(_n3##x,_p2##y,_n1##z,v)), \
 (I[216] = (img)(_n3##x,_p1##y,_n1##z,v)), \
 (I[223] = (img)(_n3##x,y,_n1##z,v)), \
 (I[230] = (img)(_n3##x,_n1##y,_n1##z,v)), \
 (I[237] = (img)(_n3##x,_n2##y,_n1##z,v)), \
 (I[244] = (img)(_n3##x,_n3##y,_n1##z,v)), \
 (I[251] = (img)(_n3##x,_p3##y,_n2##z,v)), \
 (I[258] = (img)(_n3##x,_p2##y,_n2##z,v)), \
 (I[265] = (img)(_n3##x,_p1##y,_n2##z,v)), \
 (I[272] = (img)(_n3##x,y,_n2##z,v)), \
 (I[279] = (img)(_n3##x,_n1##y,_n2##z,v)), \
 (I[286] = (img)(_n3##x,_n2##y,_n2##z,v)), \
 (I[293] = (img)(_n3##x,_n3##y,_n2##z,v)), \
 (I[300] = (img)(_n3##x,_p3##y,_n3##z,v)), \
 (I[307] = (img)(_n3##x,_p2##y,_n3##z,v)), \
 (I[314] = (img)(_n3##x,_p1##y,_n3##z,v)), \
 (I[321] = (img)(_n3##x,y,_n3##z,v)), \
 (I[328] = (img)(_n3##x,_n1##y,_n3##z,v)), \
 (I[335] = (img)(_n3##x,_n2##y,_n3##z,v)), \
 (I[342] = (img)(_n3##x,_n3##y,_n3##z,v)),1)) || \
 _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], \
 I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], \
 I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], \
 I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], \
 I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], \
 I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], \
 I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], \
 I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], \
 I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], \
 I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], \
 I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], \
 I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], \
 I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], \
 I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], \
 I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], \
 I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], \
 I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], \
 I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], \
 I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], \
 I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], \
 I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], \
 I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], \
 I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], \
 I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], \
 I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], \
 I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], \
 I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], \
 I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], \
 I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], \
 I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], \
 I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], \
 I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], \
 I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], \
 I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], \
 I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], \
 I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], \
 I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], \
 I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], \
 I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], \
 I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], \
 I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], \
 I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], \
 I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], \
 I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], \
 I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], \
 I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], \
 I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], \
 I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], \
 _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x)

#define cimg_for_in7x7x7(img,x0,y0,z0,x1,y1,z1,x,y,z,v,I) \
 cimg_for_in7((img).depth,z0,z1,z) cimg_for_in7((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = (int)( \
 (I[0] = (img)(_p3##x,_p3##y,_p3##z,v)), \
 (I[7] = (img)(_p3##x,_p2##y,_p3##z,v)), \
 (I[14] = (img)(_p3##x,_p1##y,_p3##z,v)), \
 (I[21] = (img)(_p3##x,y,_p3##z,v)), \
 (I[28] = (img)(_p3##x,_n1##y,_p3##z,v)), \
 (I[35] = (img)(_p3##x,_n2##y,_p3##z,v)), \
 (I[42] = (img)(_p3##x,_n3##y,_p3##z,v)), \
 (I[49] = (img)(_p3##x,_p3##y,_p2##z,v)), \
 (I[56] = (img)(_p3##x,_p2##y,_p2##z,v)), \
 (I[63] = (img)(_p3##x,_p1##y,_p2##z,v)), \
 (I[70] = (img)(_p3##x,y,_p2##z,v)), \
 (I[77] = (img)(_p3##x,_n1##y,_p2##z,v)), \
 (I[84] = (img)(_p3##x,_n2##y,_p2##z,v)), \
 (I[91] = (img)(_p3##x,_n3##y,_p2##z,v)), \
 (I[98] = (img)(_p3##x,_p3##y,_p1##z,v)), \
 (I[105] = (img)(_p3##x,_p2##y,_p1##z,v)), \
 (I[112] = (img)(_p3##x,_p1##y,_p1##z,v)), \
 (I[119] = (img)(_p3##x,y,_p1##z,v)), \
 (I[126] = (img)(_p3##x,_n1##y,_p1##z,v)), \
 (I[133] = (img)(_p3##x,_n2##y,_p1##z,v)), \
 (I[140] = (img)(_p3##x,_n3##y,_p1##z,v)), \
 (I[147] = (img)(_p3##x,_p3##y,z,v)), \
 (I[154] = (img)(_p3##x,_p2##y,z,v)), \
 (I[161] = (img)(_p3##x,_p1##y,z,v)), \
 (I[168] = (img)(_p3##x,y,z,v)), \
 (I[175] = (img)(_p3##x,_n1##y,z,v)), \
 (I[182] = (img)(_p3##x,_n2##y,z,v)), \
 (I[189] = (img)(_p3##x,_n3##y,z,v)), \
 (I[196] = (img)(_p3##x,_p3##y,_n1##z,v)), \
 (I[203] = (img)(_p3##x,_p2##y,_n1##z,v)), \
 (I[210] = (img)(_p3##x,_p1##y,_n1##z,v)), \
 (I[217] = (img)(_p3##x,y,_n1##z,v)), \
 (I[224] = (img)(_p3##x,_n1##y,_n1##z,v)), \
 (I[231] = (img)(_p3##x,_n2##y,_n1##z,v)), \
 (I[238] = (img)(_p3##x,_n3##y,_n1##z,v)), \
 (I[245] = (img)(_p3##x,_p3##y,_n2##z,v)), \
 (I[252] = (img)(_p3##x,_p2##y,_n2##z,v)), \
 (I[259] = (img)(_p3##x,_p1##y,_n2##z,v)), \
 (I[266] = (img)(_p3##x,y,_n2##z,v)), \
 (I[273] = (img)(_p3##x,_n1##y,_n2##z,v)), \
 (I[280] = (img)(_p3##x,_n2##y,_n2##z,v)), \
 (I[287] = (img)(_p3##x,_n3##y,_n2##z,v)), \
 (I[294] = (img)(_p3##x,_p3##y,_n3##z,v)), \
 (I[301] = (img)(_p3##x,_p2##y,_n3##z,v)), \
 (I[308] = (img)(_p3##x,_p1##y,_n3##z,v)), \
 (I[315] = (img)(_p3##x,y,_n3##z,v)), \
 (I[322] = (img)(_p3##x,_n1##y,_n3##z,v)), \
 (I[329] = (img)(_p3##x,_n2##y,_n3##z,v)), \
 (I[336] = (img)(_p3##x,_n3##y,_n3##z,v)), \
 (I[1] = (img)(_p2##x,_p3##y,_p3##z,v)), \
 (I[8] = (img)(_p2##x,_p2##y,_p3##z,v)), \
 (I[15] = (img)(_p2##x,_p1##y,_p3##z,v)), \
 (I[22] = (img)(_p2##x,y,_p3##z,v)), \
 (I[29] = (img)(_p2##x,_n1##y,_p3##z,v)), \
 (I[36] = (img)(_p2##x,_n2##y,_p3##z,v)), \
 (I[43] = (img)(_p2##x,_n3##y,_p3##z,v)), \
 (I[50] = (img)(_p2##x,_p3##y,_p2##z,v)), \
 (I[57] = (img)(_p2##x,_p2##y,_p2##z,v)), \
 (I[64] = (img)(_p2##x,_p1##y,_p2##z,v)), \
 (I[71] = (img)(_p2##x,y,_p2##z,v)), \
 (I[78] = (img)(_p2##x,_n1##y,_p2##z,v)), \
 (I[85] = (img)(_p2##x,_n2##y,_p2##z,v)), \
 (I[92] = (img)(_p2##x,_n3##y,_p2##z,v)), \
 (I[99] = (img)(_p2##x,_p3##y,_p1##z,v)), \
 (I[106] = (img)(_p2##x,_p2##y,_p1##z,v)), \
 (I[113] = (img)(_p2##x,_p1##y,_p1##z,v)), \
 (I[120] = (img)(_p2##x,y,_p1##z,v)), \
 (I[127] = (img)(_p2##x,_n1##y,_p1##z,v)), \
 (I[134] = (img)(_p2##x,_n2##y,_p1##z,v)), \
 (I[141] = (img)(_p2##x,_n3##y,_p1##z,v)), \
 (I[148] = (img)(_p2##x,_p3##y,z,v)), \
 (I[155] = (img)(_p2##x,_p2##y,z,v)), \
 (I[162] = (img)(_p2##x,_p1##y,z,v)), \
 (I[169] = (img)(_p2##x,y,z,v)), \
 (I[176] = (img)(_p2##x,_n1##y,z,v)), \
 (I[183] = (img)(_p2##x,_n2##y,z,v)), \
 (I[190] = (img)(_p2##x,_n3##y,z,v)), \
 (I[197] = (img)(_p2##x,_p3##y,_n1##z,v)), \
 (I[204] = (img)(_p2##x,_p2##y,_n1##z,v)), \
 (I[211] = (img)(_p2##x,_p1##y,_n1##z,v)), \
 (I[218] = (img)(_p2##x,y,_n1##z,v)), \
 (I[225] = (img)(_p2##x,_n1##y,_n1##z,v)), \
 (I[232] = (img)(_p2##x,_n2##y,_n1##z,v)), \
 (I[239] = (img)(_p2##x,_n3##y,_n1##z,v)), \
 (I[246] = (img)(_p2##x,_p3##y,_n2##z,v)), \
 (I[253] = (img)(_p2##x,_p2##y,_n2##z,v)), \
 (I[260] = (img)(_p2##x,_p1##y,_n2##z,v)), \
 (I[267] = (img)(_p2##x,y,_n2##z,v)), \
 (I[274] = (img)(_p2##x,_n1##y,_n2##z,v)), \
 (I[281] = (img)(_p2##x,_n2##y,_n2##z,v)), \
 (I[288] = (img)(_p2##x,_n3##y,_n2##z,v)), \
 (I[295] = (img)(_p2##x,_p3##y,_n3##z,v)), \
 (I[302] = (img)(_p2##x,_p2##y,_n3##z,v)), \
 (I[309] = (img)(_p2##x,_p1##y,_n3##z,v)), \
 (I[316] = (img)(_p2##x,y,_n3##z,v)), \
 (I[323] = (img)(_p2##x,_n1##y,_n3##z,v)), \
 (I[330] = (img)(_p2##x,_n2##y,_n3##z,v)), \
 (I[337] = (img)(_p2##x,_n3##y,_n3##z,v)), \
 (I[2] = (img)(_p1##x,_p3##y,_p3##z,v)), \
 (I[9] = (img)(_p1##x,_p2##y,_p3##z,v)), \
 (I[16] = (img)(_p1##x,_p1##y,_p3##z,v)), \
 (I[23] = (img)(_p1##x,y,_p3##z,v)), \
 (I[30] = (img)(_p1##x,_n1##y,_p3##z,v)), \
 (I[37] = (img)(_p1##x,_n2##y,_p3##z,v)), \
 (I[44] = (img)(_p1##x,_n3##y,_p3##z,v)), \
 (I[51] = (img)(_p1##x,_p3##y,_p2##z,v)), \
 (I[58] = (img)(_p1##x,_p2##y,_p2##z,v)), \
 (I[65] = (img)(_p1##x,_p1##y,_p2##z,v)), \
 (I[72] = (img)(_p1##x,y,_p2##z,v)), \
 (I[79] = (img)(_p1##x,_n1##y,_p2##z,v)), \
 (I[86] = (img)(_p1##x,_n2##y,_p2##z,v)), \
 (I[93] = (img)(_p1##x,_n3##y,_p2##z,v)), \
 (I[100] = (img)(_p1##x,_p3##y,_p1##z,v)), \
 (I[107] = (img)(_p1##x,_p2##y,_p1##z,v)), \
 (I[114] = (img)(_p1##x,_p1##y,_p1##z,v)), \
 (I[121] = (img)(_p1##x,y,_p1##z,v)), \
 (I[128] = (img)(_p1##x,_n1##y,_p1##z,v)), \
 (I[135] = (img)(_p1##x,_n2##y,_p1##z,v)), \
 (I[142] = (img)(_p1##x,_n3##y,_p1##z,v)), \
 (I[149] = (img)(_p1##x,_p3##y,z,v)), \
 (I[156] = (img)(_p1##x,_p2##y,z,v)), \
 (I[163] = (img)(_p1##x,_p1##y,z,v)), \
 (I[170] = (img)(_p1##x,y,z,v)), \
 (I[177] = (img)(_p1##x,_n1##y,z,v)), \
 (I[184] = (img)(_p1##x,_n2##y,z,v)), \
 (I[191] = (img)(_p1##x,_n3##y,z,v)), \
 (I[198] = (img)(_p1##x,_p3##y,_n1##z,v)), \
 (I[205] = (img)(_p1##x,_p2##y,_n1##z,v)), \
 (I[212] = (img)(_p1##x,_p1##y,_n1##z,v)), \
 (I[219] = (img)(_p1##x,y,_n1##z,v)), \
 (I[226] = (img)(_p1##x,_n1##y,_n1##z,v)), \
 (I[233] = (img)(_p1##x,_n2##y,_n1##z,v)), \
 (I[240] = (img)(_p1##x,_n3##y,_n1##z,v)), \
 (I[247] = (img)(_p1##x,_p3##y,_n2##z,v)), \
 (I[254] = (img)(_p1##x,_p2##y,_n2##z,v)), \
 (I[261] = (img)(_p1##x,_p1##y,_n2##z,v)), \
 (I[268] = (img)(_p1##x,y,_n2##z,v)), \
 (I[275] = (img)(_p1##x,_n1##y,_n2##z,v)), \
 (I[282] = (img)(_p1##x,_n2##y,_n2##z,v)), \
 (I[289] = (img)(_p1##x,_n3##y,_n2##z,v)), \
 (I[296] = (img)(_p1##x,_p3##y,_n3##z,v)), \
 (I[303] = (img)(_p1##x,_p2##y,_n3##z,v)), \
 (I[310] = (img)(_p1##x,_p1##y,_n3##z,v)), \
 (I[317] = (img)(_p1##x,y,_n3##z,v)), \
 (I[324] = (img)(_p1##x,_n1##y,_n3##z,v)), \
 (I[331] = (img)(_p1##x,_n2##y,_n3##z,v)), \
 (I[338] = (img)(_p1##x,_n3##y,_n3##z,v)), \
 (I[3] = (img)(x,_p3##y,_p3##z,v)), \
 (I[10] = (img)(x,_p2##y,_p3##z,v)), \
 (I[17] = (img)(x,_p1##y,_p3##z,v)), \
 (I[24] = (img)(x,y,_p3##z,v)), \
 (I[31] = (img)(x,_n1##y,_p3##z,v)), \
 (I[38] = (img)(x,_n2##y,_p3##z,v)), \
 (I[45] = (img)(x,_n3##y,_p3##z,v)), \
 (I[52] = (img)(x,_p3##y,_p2##z,v)), \
 (I[59] = (img)(x,_p2##y,_p2##z,v)), \
 (I[66] = (img)(x,_p1##y,_p2##z,v)), \
 (I[73] = (img)(x,y,_p2##z,v)), \
 (I[80] = (img)(x,_n1##y,_p2##z,v)), \
 (I[87] = (img)(x,_n2##y,_p2##z,v)), \
 (I[94] = (img)(x,_n3##y,_p2##z,v)), \
 (I[101] = (img)(x,_p3##y,_p1##z,v)), \
 (I[108] = (img)(x,_p2##y,_p1##z,v)), \
 (I[115] = (img)(x,_p1##y,_p1##z,v)), \
 (I[122] = (img)(x,y,_p1##z,v)), \
 (I[129] = (img)(x,_n1##y,_p1##z,v)), \
 (I[136] = (img)(x,_n2##y,_p1##z,v)), \
 (I[143] = (img)(x,_n3##y,_p1##z,v)), \
 (I[150] = (img)(x,_p3##y,z,v)), \
 (I[157] = (img)(x,_p2##y,z,v)), \
 (I[164] = (img)(x,_p1##y,z,v)), \
 (I[171] = (img)(x,y,z,v)), \
 (I[178] = (img)(x,_n1##y,z,v)), \
 (I[185] = (img)(x,_n2##y,z,v)), \
 (I[192] = (img)(x,_n3##y,z,v)), \
 (I[199] = (img)(x,_p3##y,_n1##z,v)), \
 (I[206] = (img)(x,_p2##y,_n1##z,v)), \
 (I[213] = (img)(x,_p1##y,_n1##z,v)), \
 (I[220] = (img)(x,y,_n1##z,v)), \
 (I[227] = (img)(x,_n1##y,_n1##z,v)), \
 (I[234] = (img)(x,_n2##y,_n1##z,v)), \
 (I[241] = (img)(x,_n3##y,_n1##z,v)), \
 (I[248] = (img)(x,_p3##y,_n2##z,v)), \
 (I[255] = (img)(x,_p2##y,_n2##z,v)), \
 (I[262] = (img)(x,_p1##y,_n2##z,v)), \
 (I[269] = (img)(x,y,_n2##z,v)), \
 (I[276] = (img)(x,_n1##y,_n2##z,v)), \
 (I[283] = (img)(x,_n2##y,_n2##z,v)), \
 (I[290] = (img)(x,_n3##y,_n2##z,v)), \
 (I[297] = (img)(x,_p3##y,_n3##z,v)), \
 (I[304] = (img)(x,_p2##y,_n3##z,v)), \
 (I[311] = (img)(x,_p1##y,_n3##z,v)), \
 (I[318] = (img)(x,y,_n3##z,v)), \
 (I[325] = (img)(x,_n1##y,_n3##z,v)), \
 (I[332] = (img)(x,_n2##y,_n3##z,v)), \
 (I[339] = (img)(x,_n3##y,_n3##z,v)), \
 (I[4] = (img)(_n1##x,_p3##y,_p3##z,v)), \
 (I[11] = (img)(_n1##x,_p2##y,_p3##z,v)), \
 (I[18] = (img)(_n1##x,_p1##y,_p3##z,v)), \
 (I[25] = (img)(_n1##x,y,_p3##z,v)), \
 (I[32] = (img)(_n1##x,_n1##y,_p3##z,v)), \
 (I[39] = (img)(_n1##x,_n2##y,_p3##z,v)), \
 (I[46] = (img)(_n1##x,_n3##y,_p3##z,v)), \
 (I[53] = (img)(_n1##x,_p3##y,_p2##z,v)), \
 (I[60] = (img)(_n1##x,_p2##y,_p2##z,v)), \
 (I[67] = (img)(_n1##x,_p1##y,_p2##z,v)), \
 (I[74] = (img)(_n1##x,y,_p2##z,v)), \
 (I[81] = (img)(_n1##x,_n1##y,_p2##z,v)), \
 (I[88] = (img)(_n1##x,_n2##y,_p2##z,v)), \
 (I[95] = (img)(_n1##x,_n3##y,_p2##z,v)), \
 (I[102] = (img)(_n1##x,_p3##y,_p1##z,v)), \
 (I[109] = (img)(_n1##x,_p2##y,_p1##z,v)), \
 (I[116] = (img)(_n1##x,_p1##y,_p1##z,v)), \
 (I[123] = (img)(_n1##x,y,_p1##z,v)), \
 (I[130] = (img)(_n1##x,_n1##y,_p1##z,v)), \
 (I[137] = (img)(_n1##x,_n2##y,_p1##z,v)), \
 (I[144] = (img)(_n1##x,_n3##y,_p1##z,v)), \
 (I[151] = (img)(_n1##x,_p3##y,z,v)), \
 (I[158] = (img)(_n1##x,_p2##y,z,v)), \
 (I[165] = (img)(_n1##x,_p1##y,z,v)), \
 (I[172] = (img)(_n1##x,y,z,v)), \
 (I[179] = (img)(_n1##x,_n1##y,z,v)), \
 (I[186] = (img)(_n1##x,_n2##y,z,v)), \
 (I[193] = (img)(_n1##x,_n3##y,z,v)), \
 (I[200] = (img)(_n1##x,_p3##y,_n1##z,v)), \
 (I[207] = (img)(_n1##x,_p2##y,_n1##z,v)), \
 (I[214] = (img)(_n1##x,_p1##y,_n1##z,v)), \
 (I[221] = (img)(_n1##x,y,_n1##z,v)), \
 (I[228] = (img)(_n1##x,_n1##y,_n1##z,v)), \
 (I[235] = (img)(_n1##x,_n2##y,_n1##z,v)), \
 (I[242] = (img)(_n1##x,_n3##y,_n1##z,v)), \
 (I[249] = (img)(_n1##x,_p3##y,_n2##z,v)), \
 (I[256] = (img)(_n1##x,_p2##y,_n2##z,v)), \
 (I[263] = (img)(_n1##x,_p1##y,_n2##z,v)), \
 (I[270] = (img)(_n1##x,y,_n2##z,v)), \
 (I[277] = (img)(_n1##x,_n1##y,_n2##z,v)), \
 (I[284] = (img)(_n1##x,_n2##y,_n2##z,v)), \
 (I[291] = (img)(_n1##x,_n3##y,_n2##z,v)), \
 (I[298] = (img)(_n1##x,_p3##y,_n3##z,v)), \
 (I[305] = (img)(_n1##x,_p2##y,_n3##z,v)), \
 (I[312] = (img)(_n1##x,_p1##y,_n3##z,v)), \
 (I[319] = (img)(_n1##x,y,_n3##z,v)), \
 (I[326] = (img)(_n1##x,_n1##y,_n3##z,v)), \
 (I[333] = (img)(_n1##x,_n2##y,_n3##z,v)), \
 (I[340] = (img)(_n1##x,_n3##y,_n3##z,v)), \
 (I[5] = (img)(_n2##x,_p3##y,_p3##z,v)), \
 (I[12] = (img)(_n2##x,_p2##y,_p3##z,v)), \
 (I[19] = (img)(_n2##x,_p1##y,_p3##z,v)), \
 (I[26] = (img)(_n2##x,y,_p3##z,v)), \
 (I[33] = (img)(_n2##x,_n1##y,_p3##z,v)), \
 (I[40] = (img)(_n2##x,_n2##y,_p3##z,v)), \
 (I[47] = (img)(_n2##x,_n3##y,_p3##z,v)), \
 (I[54] = (img)(_n2##x,_p3##y,_p2##z,v)), \
 (I[61] = (img)(_n2##x,_p2##y,_p2##z,v)), \
 (I[68] = (img)(_n2##x,_p1##y,_p2##z,v)), \
 (I[75] = (img)(_n2##x,y,_p2##z,v)), \
 (I[82] = (img)(_n2##x,_n1##y,_p2##z,v)), \
 (I[89] = (img)(_n2##x,_n2##y,_p2##z,v)), \
 (I[96] = (img)(_n2##x,_n3##y,_p2##z,v)), \
 (I[103] = (img)(_n2##x,_p3##y,_p1##z,v)), \
 (I[110] = (img)(_n2##x,_p2##y,_p1##z,v)), \
 (I[117] = (img)(_n2##x,_p1##y,_p1##z,v)), \
 (I[124] = (img)(_n2##x,y,_p1##z,v)), \
 (I[131] = (img)(_n2##x,_n1##y,_p1##z,v)), \
 (I[138] = (img)(_n2##x,_n2##y,_p1##z,v)), \
 (I[145] = (img)(_n2##x,_n3##y,_p1##z,v)), \
 (I[152] = (img)(_n2##x,_p3##y,z,v)), \
 (I[159] = (img)(_n2##x,_p2##y,z,v)), \
 (I[166] = (img)(_n2##x,_p1##y,z,v)), \
 (I[173] = (img)(_n2##x,y,z,v)), \
 (I[180] = (img)(_n2##x,_n1##y,z,v)), \
 (I[187] = (img)(_n2##x,_n2##y,z,v)), \
 (I[194] = (img)(_n2##x,_n3##y,z,v)), \
 (I[201] = (img)(_n2##x,_p3##y,_n1##z,v)), \
 (I[208] = (img)(_n2##x,_p2##y,_n1##z,v)), \
 (I[215] = (img)(_n2##x,_p1##y,_n1##z,v)), \
 (I[222] = (img)(_n2##x,y,_n1##z,v)), \
 (I[229] = (img)(_n2##x,_n1##y,_n1##z,v)), \
 (I[236] = (img)(_n2##x,_n2##y,_n1##z,v)), \
 (I[243] = (img)(_n2##x,_n3##y,_n1##z,v)), \
 (I[250] = (img)(_n2##x,_p3##y,_n2##z,v)), \
 (I[257] = (img)(_n2##x,_p2##y,_n2##z,v)), \
 (I[264] = (img)(_n2##x,_p1##y,_n2##z,v)), \
 (I[271] = (img)(_n2##x,y,_n2##z,v)), \
 (I[278] = (img)(_n2##x,_n1##y,_n2##z,v)), \
 (I[285] = (img)(_n2##x,_n2##y,_n2##z,v)), \
 (I[292] = (img)(_n2##x,_n3##y,_n2##z,v)), \
 (I[299] = (img)(_n2##x,_p3##y,_n3##z,v)), \
 (I[306] = (img)(_n2##x,_p2##y,_n3##z,v)), \
 (I[313] = (img)(_n2##x,_p1##y,_n3##z,v)), \
 (I[320] = (img)(_n2##x,y,_n3##z,v)), \
 (I[327] = (img)(_n2##x,_n1##y,_n3##z,v)), \
 (I[334] = (img)(_n2##x,_n2##y,_n3##z,v)), \
 (I[341] = (img)(_n2##x,_n3##y,_n3##z,v)), \
 x+3>=(int)((img).width)?(int)((img).width)-1:x+3); \
 x<=(int)(x1) && ((_n3##x<(int)((img).width) && ( \
 (I[6] = (img)(_n3##x,_p3##y,_p3##z,v)), \
 (I[13] = (img)(_n3##x,_p2##y,_p3##z,v)), \
 (I[20] = (img)(_n3##x,_p1##y,_p3##z,v)), \
 (I[27] = (img)(_n3##x,y,_p3##z,v)), \
 (I[34] = (img)(_n3##x,_n1##y,_p3##z,v)), \
 (I[41] = (img)(_n3##x,_n2##y,_p3##z,v)), \
 (I[48] = (img)(_n3##x,_n3##y,_p3##z,v)), \
 (I[55] = (img)(_n3##x,_p3##y,_p2##z,v)), \
 (I[62] = (img)(_n3##x,_p2##y,_p2##z,v)), \
 (I[69] = (img)(_n3##x,_p1##y,_p2##z,v)), \
 (I[76] = (img)(_n3##x,y,_p2##z,v)), \
 (I[83] = (img)(_n3##x,_n1##y,_p2##z,v)), \
 (I[90] = (img)(_n3##x,_n2##y,_p2##z,v)), \
 (I[97] = (img)(_n3##x,_n3##y,_p2##z,v)), \
 (I[104] = (img)(_n3##x,_p3##y,_p1##z,v)), \
 (I[111] = (img)(_n3##x,_p2##y,_p1##z,v)), \
 (I[118] = (img)(_n3##x,_p1##y,_p1##z,v)), \
 (I[125] = (img)(_n3##x,y,_p1##z,v)), \
 (I[132] = (img)(_n3##x,_n1##y,_p1##z,v)), \
 (I[139] = (img)(_n3##x,_n2##y,_p1##z,v)), \
 (I[146] = (img)(_n3##x,_n3##y,_p1##z,v)), \
 (I[153] = (img)(_n3##x,_p3##y,z,v)), \
 (I[160] = (img)(_n3##x,_p2##y,z,v)), \
 (I[167] = (img)(_n3##x,_p1##y,z,v)), \
 (I[174] = (img)(_n3##x,y,z,v)), \
 (I[181] = (img)(_n3##x,_n1##y,z,v)), \
 (I[188] = (img)(_n3##x,_n2##y,z,v)), \
 (I[195] = (img)(_n3##x,_n3##y,z,v)), \
 (I[202] = (img)(_n3##x,_p3##y,_n1##z,v)), \
 (I[209] = (img)(_n3##x,_p2##y,_n1##z,v)), \
 (I[216] = (img)(_n3##x,_p1##y,_n1##z,v)), \
 (I[223] = (img)(_n3##x,y,_n1##z,v)), \
 (I[230] = (img)(_n3##x,_n1##y,_n1##z,v)), \
 (I[237] = (img)(_n3##x,_n2##y,_n1##z,v)), \
 (I[244] = (img)(_n3##x,_n3##y,_n1##z,v)), \
 (I[251] = (img)(_n3##x,_p3##y,_n2##z,v)), \
 (I[258] = (img)(_n3##x,_p2##y,_n2##z,v)), \
 (I[265] = (img)(_n3##x,_p1##y,_n2##z,v)), \
 (I[272] = (img)(_n3##x,y,_n2##z,v)), \
 (I[279] = (img)(_n3##x,_n1##y,_n2##z,v)), \
 (I[286] = (img)(_n3##x,_n2##y,_n2##z,v)), \
 (I[293] = (img)(_n3##x,_n3##y,_n2##z,v)), \
 (I[300] = (img)(_n3##x,_p3##y,_n3##z,v)), \
 (I[307] = (img)(_n3##x,_p2##y,_n3##z,v)), \
 (I[314] = (img)(_n3##x,_p1##y,_n3##z,v)), \
 (I[321] = (img)(_n3##x,y,_n3##z,v)), \
 (I[328] = (img)(_n3##x,_n1##y,_n3##z,v)), \
 (I[335] = (img)(_n3##x,_n2##y,_n3##z,v)), \
 (I[342] = (img)(_n3##x,_n3##y,_n3##z,v)),1)) || \
 _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], \
 I[7] = I[8], I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], \
 I[14] = I[15], I[15] = I[16], I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], \
 I[21] = I[22], I[22] = I[23], I[23] = I[24], I[24] = I[25], I[25] = I[26], I[26] = I[27], \
 I[28] = I[29], I[29] = I[30], I[30] = I[31], I[31] = I[32], I[32] = I[33], I[33] = I[34], \
 I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], I[39] = I[40], I[40] = I[41], \
 I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], I[47] = I[48], \
 I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], \
 I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], \
 I[63] = I[64], I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], \
 I[70] = I[71], I[71] = I[72], I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], \
 I[77] = I[78], I[78] = I[79], I[79] = I[80], I[80] = I[81], I[81] = I[82], I[82] = I[83], \
 I[84] = I[85], I[85] = I[86], I[86] = I[87], I[87] = I[88], I[88] = I[89], I[89] = I[90], \
 I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], I[95] = I[96], I[96] = I[97], \
 I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], I[103] = I[104], \
 I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], \
 I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], \
 I[119] = I[120], I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], \
 I[126] = I[127], I[127] = I[128], I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], \
 I[133] = I[134], I[134] = I[135], I[135] = I[136], I[136] = I[137], I[137] = I[138], I[138] = I[139], \
 I[140] = I[141], I[141] = I[142], I[142] = I[143], I[143] = I[144], I[144] = I[145], I[145] = I[146], \
 I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], I[151] = I[152], I[152] = I[153], \
 I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], I[159] = I[160], \
 I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], \
 I[175] = I[176], I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], \
 I[182] = I[183], I[183] = I[184], I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], \
 I[189] = I[190], I[190] = I[191], I[191] = I[192], I[192] = I[193], I[193] = I[194], I[194] = I[195], \
 I[196] = I[197], I[197] = I[198], I[198] = I[199], I[199] = I[200], I[200] = I[201], I[201] = I[202], \
 I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], I[207] = I[208], I[208] = I[209], \
 I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], I[215] = I[216], \
 I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], \
 I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], \
 I[231] = I[232], I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], \
 I[238] = I[239], I[239] = I[240], I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], \
 I[245] = I[246], I[246] = I[247], I[247] = I[248], I[248] = I[249], I[249] = I[250], I[250] = I[251], \
 I[252] = I[253], I[253] = I[254], I[254] = I[255], I[255] = I[256], I[256] = I[257], I[257] = I[258], \
 I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], I[263] = I[264], I[264] = I[265], \
 I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], I[271] = I[272], \
 I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], \
 I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], \
 I[287] = I[288], I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], \
 I[294] = I[295], I[295] = I[296], I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], \
 I[301] = I[302], I[302] = I[303], I[303] = I[304], I[304] = I[305], I[305] = I[306], I[306] = I[307], \
 I[308] = I[309], I[309] = I[310], I[310] = I[311], I[311] = I[312], I[312] = I[313], I[313] = I[314], \
 I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], I[319] = I[320], I[320] = I[321], \
 I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], I[327] = I[328], \
 I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], \
 I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], \
 _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x)

#define cimg_get7x7x7(img,x,y,z,v,I) \
 I[0] = (img)(_p3##x,_p3##y,_p3##z,v), I[1] = (img)(_p2##x,_p3##y,_p3##z,v), I[2] = (img)(_p1##x,_p3##y,_p3##z,v), I[3] = (img)(x,_p3##y,_p3##z,v), I[4] = (img)(_n1##x,_p3##y,_p3##z,v), I[5] = (img)(_n2##x,_p3##y,_p3##z,v), I[6] = (img)(_n3##x,_p3##y,_p3##z,v), \
 I[7] = (img)(_p3##x,_p2##y,_p3##z,v), I[8] = (img)(_p2##x,_p2##y,_p3##z,v), I[9] = (img)(_p1##x,_p2##y,_p3##z,v), I[10] = (img)(x,_p2##y,_p3##z,v), I[11] = (img)(_n1##x,_p2##y,_p3##z,v), I[12] = (img)(_n2##x,_p2##y,_p3##z,v), I[13] = (img)(_n3##x,_p2##y,_p3##z,v), \
 I[14] = (img)(_p3##x,_p1##y,_p3##z,v), I[15] = (img)(_p2##x,_p1##y,_p3##z,v), I[16] = (img)(_p1##x,_p1##y,_p3##z,v), I[17] = (img)(x,_p1##y,_p3##z,v), I[18] = (img)(_n1##x,_p1##y,_p3##z,v), I[19] = (img)(_n2##x,_p1##y,_p3##z,v), I[20] = (img)(_n3##x,_p1##y,_p3##z,v), \
 I[21] = (img)(_p3##x,y,_p3##z,v), I[22] = (img)(_p2##x,y,_p3##z,v), I[23] = (img)(_p1##x,y,_p3##z,v), I[24] = (img)(x,y,_p3##z,v), I[25] = (img)(_n1##x,y,_p3##z,v), I[26] = (img)(_n2##x,y,_p3##z,v), I[27] = (img)(_n3##x,y,_p3##z,v), \
 I[28] = (img)(_p3##x,_n1##y,_p3##z,v), I[29] = (img)(_p2##x,_n1##y,_p3##z,v), I[30] = (img)(_p1##x,_n1##y,_p3##z,v), I[31] = (img)(x,_n1##y,_p3##z,v), I[32] = (img)(_n1##x,_n1##y,_p3##z,v), I[33] = (img)(_n2##x,_n1##y,_p3##z,v), I[34] = (img)(_n3##x,_n1##y,_p3##z,v), \
 I[35] = (img)(_p3##x,_n2##y,_p3##z,v), I[36] = (img)(_p2##x,_n2##y,_p3##z,v), I[37] = (img)(_p1##x,_n2##y,_p3##z,v), I[38] = (img)(x,_n2##y,_p3##z,v), I[39] = (img)(_n1##x,_n2##y,_p3##z,v), I[40] = (img)(_n2##x,_n2##y,_p3##z,v), I[41] = (img)(_n3##x,_n2##y,_p3##z,v), \
 I[42] = (img)(_p3##x,_n3##y,_p3##z,v), I[43] = (img)(_p2##x,_n3##y,_p3##z,v), I[44] = (img)(_p1##x,_n3##y,_p3##z,v), I[45] = (img)(x,_n3##y,_p3##z,v), I[46] = (img)(_n1##x,_n3##y,_p3##z,v), I[47] = (img)(_n2##x,_n3##y,_p3##z,v), I[48] = (img)(_n3##x,_n3##y,_p3##z,v), \
 I[49] = (img)(_p3##x,_p3##y,_p2##z,v), I[50] = (img)(_p2##x,_p3##y,_p2##z,v), I[51] = (img)(_p1##x,_p3##y,_p2##z,v), I[52] = (img)(x,_p3##y,_p2##z,v), I[53] = (img)(_n1##x,_p3##y,_p2##z,v), I[54] = (img)(_n2##x,_p3##y,_p2##z,v), I[55] = (img)(_n3##x,_p3##y,_p2##z,v), \
 I[56] = (img)(_p3##x,_p2##y,_p2##z,v), I[57] = (img)(_p2##x,_p2##y,_p2##z,v), I[58] = (img)(_p1##x,_p2##y,_p2##z,v), I[59] = (img)(x,_p2##y,_p2##z,v), I[60] = (img)(_n1##x,_p2##y,_p2##z,v), I[61] = (img)(_n2##x,_p2##y,_p2##z,v), I[62] = (img)(_n3##x,_p2##y,_p2##z,v), \
 I[63] = (img)(_p3##x,_p1##y,_p2##z,v), I[64] = (img)(_p2##x,_p1##y,_p2##z,v), I[65] = (img)(_p1##x,_p1##y,_p2##z,v), I[66] = (img)(x,_p1##y,_p2##z,v), I[67] = (img)(_n1##x,_p1##y,_p2##z,v), I[68] = (img)(_n2##x,_p1##y,_p2##z,v), I[69] = (img)(_n3##x,_p1##y,_p2##z,v), \
 I[70] = (img)(_p3##x,y,_p2##z,v), I[71] = (img)(_p2##x,y,_p2##z,v), I[72] = (img)(_p1##x,y,_p2##z,v), I[73] = (img)(x,y,_p2##z,v), I[74] = (img)(_n1##x,y,_p2##z,v), I[75] = (img)(_n2##x,y,_p2##z,v), I[76] = (img)(_n3##x,y,_p2##z,v), \
 I[77] = (img)(_p3##x,_n1##y,_p2##z,v), I[78] = (img)(_p2##x,_n1##y,_p2##z,v), I[79] = (img)(_p1##x,_n1##y,_p2##z,v), I[80] = (img)(x,_n1##y,_p2##z,v), I[81] = (img)(_n1##x,_n1##y,_p2##z,v), I[82] = (img)(_n2##x,_n1##y,_p2##z,v), I[83] = (img)(_n3##x,_n1##y,_p2##z,v), \
 I[84] = (img)(_p3##x,_n2##y,_p2##z,v), I[85] = (img)(_p2##x,_n2##y,_p2##z,v), I[86] = (img)(_p1##x,_n2##y,_p2##z,v), I[87] = (img)(x,_n2##y,_p2##z,v), I[88] = (img)(_n1##x,_n2##y,_p2##z,v), I[89] = (img)(_n2##x,_n2##y,_p2##z,v), I[90] = (img)(_n3##x,_n2##y,_p2##z,v), \
 I[91] = (img)(_p3##x,_n3##y,_p2##z,v), I[92] = (img)(_p2##x,_n3##y,_p2##z,v), I[93] = (img)(_p1##x,_n3##y,_p2##z,v), I[94] = (img)(x,_n3##y,_p2##z,v), I[95] = (img)(_n1##x,_n3##y,_p2##z,v), I[96] = (img)(_n2##x,_n3##y,_p2##z,v), I[97] = (img)(_n3##x,_n3##y,_p2##z,v), \
 I[98] = (img)(_p3##x,_p3##y,_p1##z,v), I[99] = (img)(_p2##x,_p3##y,_p1##z,v), I[100] = (img)(_p1##x,_p3##y,_p1##z,v), I[101] = (img)(x,_p3##y,_p1##z,v), I[102] = (img)(_n1##x,_p3##y,_p1##z,v), I[103] = (img)(_n2##x,_p3##y,_p1##z,v), I[104] = (img)(_n3##x,_p3##y,_p1##z,v), \
 I[105] = (img)(_p3##x,_p2##y,_p1##z,v), I[106] = (img)(_p2##x,_p2##y,_p1##z,v), I[107] = (img)(_p1##x,_p2##y,_p1##z,v), I[108] = (img)(x,_p2##y,_p1##z,v), I[109] = (img)(_n1##x,_p2##y,_p1##z,v), I[110] = (img)(_n2##x,_p2##y,_p1##z,v), I[111] = (img)(_n3##x,_p2##y,_p1##z,v), \
 I[112] = (img)(_p3##x,_p1##y,_p1##z,v), I[113] = (img)(_p2##x,_p1##y,_p1##z,v), I[114] = (img)(_p1##x,_p1##y,_p1##z,v), I[115] = (img)(x,_p1##y,_p1##z,v), I[116] = (img)(_n1##x,_p1##y,_p1##z,v), I[117] = (img)(_n2##x,_p1##y,_p1##z,v), I[118] = (img)(_n3##x,_p1##y,_p1##z,v), \
 I[119] = (img)(_p3##x,y,_p1##z,v), I[120] = (img)(_p2##x,y,_p1##z,v), I[121] = (img)(_p1##x,y,_p1##z,v), I[122] = (img)(x,y,_p1##z,v), I[123] = (img)(_n1##x,y,_p1##z,v), I[124] = (img)(_n2##x,y,_p1##z,v), I[125] = (img)(_n3##x,y,_p1##z,v), \
 I[126] = (img)(_p3##x,_n1##y,_p1##z,v), I[127] = (img)(_p2##x,_n1##y,_p1##z,v), I[128] = (img)(_p1##x,_n1##y,_p1##z,v), I[129] = (img)(x,_n1##y,_p1##z,v), I[130] = (img)(_n1##x,_n1##y,_p1##z,v), I[131] = (img)(_n2##x,_n1##y,_p1##z,v), I[132] = (img)(_n3##x,_n1##y,_p1##z,v), \
 I[133] = (img)(_p3##x,_n2##y,_p1##z,v), I[134] = (img)(_p2##x,_n2##y,_p1##z,v), I[135] = (img)(_p1##x,_n2##y,_p1##z,v), I[136] = (img)(x,_n2##y,_p1##z,v), I[137] = (img)(_n1##x,_n2##y,_p1##z,v), I[138] = (img)(_n2##x,_n2##y,_p1##z,v), I[139] = (img)(_n3##x,_n2##y,_p1##z,v), \
 I[140] = (img)(_p3##x,_n3##y,_p1##z,v), I[141] = (img)(_p2##x,_n3##y,_p1##z,v), I[142] = (img)(_p1##x,_n3##y,_p1##z,v), I[143] = (img)(x,_n3##y,_p1##z,v), I[144] = (img)(_n1##x,_n3##y,_p1##z,v), I[145] = (img)(_n2##x,_n3##y,_p1##z,v), I[146] = (img)(_n3##x,_n3##y,_p1##z,v), \
 I[147] = (img)(_p3##x,_p3##y,z,v), I[148] = (img)(_p2##x,_p3##y,z,v), I[149] = (img)(_p1##x,_p3##y,z,v), I[150] = (img)(x,_p3##y,z,v), I[151] = (img)(_n1##x,_p3##y,z,v), I[152] = (img)(_n2##x,_p3##y,z,v), I[153] = (img)(_n3##x,_p3##y,z,v), \
 I[154] = (img)(_p3##x,_p2##y,z,v), I[155] = (img)(_p2##x,_p2##y,z,v), I[156] = (img)(_p1##x,_p2##y,z,v), I[157] = (img)(x,_p2##y,z,v), I[158] = (img)(_n1##x,_p2##y,z,v), I[159] = (img)(_n2##x,_p2##y,z,v), I[160] = (img)(_n3##x,_p2##y,z,v), \
 I[161] = (img)(_p3##x,_p1##y,z,v), I[162] = (img)(_p2##x,_p1##y,z,v), I[163] = (img)(_p1##x,_p1##y,z,v), I[164] = (img)(x,_p1##y,z,v), I[165] = (img)(_n1##x,_p1##y,z,v), I[166] = (img)(_n2##x,_p1##y,z,v), I[167] = (img)(_n3##x,_p1##y,z,v), \
 I[168] = (img)(_p3##x,y,z,v), I[169] = (img)(_p2##x,y,z,v), I[170] = (img)(_p1##x,y,z,v), I[171] = (img)(x,y,z,v), I[172] = (img)(_n1##x,y,z,v), I[173] = (img)(_n2##x,y,z,v), I[174] = (img)(_n3##x,y,z,v), \
 I[175] = (img)(_p3##x,_n1##y,z,v), I[176] = (img)(_p2##x,_n1##y,z,v), I[177] = (img)(_p1##x,_n1##y,z,v), I[178] = (img)(x,_n1##y,z,v), I[179] = (img)(_n1##x,_n1##y,z,v), I[180] = (img)(_n2##x,_n1##y,z,v), I[181] = (img)(_n3##x,_n1##y,z,v), \
 I[182] = (img)(_p3##x,_n2##y,z,v), I[183] = (img)(_p2##x,_n2##y,z,v), I[184] = (img)(_p1##x,_n2##y,z,v), I[185] = (img)(x,_n2##y,z,v), I[186] = (img)(_n1##x,_n2##y,z,v), I[187] = (img)(_n2##x,_n2##y,z,v), I[188] = (img)(_n3##x,_n2##y,z,v), \
 I[189] = (img)(_p3##x,_n3##y,z,v), I[190] = (img)(_p2##x,_n3##y,z,v), I[191] = (img)(_p1##x,_n3##y,z,v), I[192] = (img)(x,_n3##y,z,v), I[193] = (img)(_n1##x,_n3##y,z,v), I[194] = (img)(_n2##x,_n3##y,z,v), I[195] = (img)(_n3##x,_n3##y,z,v), \
 I[196] = (img)(_p3##x,_p3##y,_n1##z,v), I[197] = (img)(_p2##x,_p3##y,_n1##z,v), I[198] = (img)(_p1##x,_p3##y,_n1##z,v), I[199] = (img)(x,_p3##y,_n1##z,v), I[200] = (img)(_n1##x,_p3##y,_n1##z,v), I[201] = (img)(_n2##x,_p3##y,_n1##z,v), I[202] = (img)(_n3##x,_p3##y,_n1##z,v), \
 I[203] = (img)(_p3##x,_p2##y,_n1##z,v), I[204] = (img)(_p2##x,_p2##y,_n1##z,v), I[205] = (img)(_p1##x,_p2##y,_n1##z,v), I[206] = (img)(x,_p2##y,_n1##z,v), I[207] = (img)(_n1##x,_p2##y,_n1##z,v), I[208] = (img)(_n2##x,_p2##y,_n1##z,v), I[209] = (img)(_n3##x,_p2##y,_n1##z,v), \
 I[210] = (img)(_p3##x,_p1##y,_n1##z,v), I[211] = (img)(_p2##x,_p1##y,_n1##z,v), I[212] = (img)(_p1##x,_p1##y,_n1##z,v), I[213] = (img)(x,_p1##y,_n1##z,v), I[214] = (img)(_n1##x,_p1##y,_n1##z,v), I[215] = (img)(_n2##x,_p1##y,_n1##z,v), I[216] = (img)(_n3##x,_p1##y,_n1##z,v), \
 I[217] = (img)(_p3##x,y,_n1##z,v), I[218] = (img)(_p2##x,y,_n1##z,v), I[219] = (img)(_p1##x,y,_n1##z,v), I[220] = (img)(x,y,_n1##z,v), I[221] = (img)(_n1##x,y,_n1##z,v), I[222] = (img)(_n2##x,y,_n1##z,v), I[223] = (img)(_n3##x,y,_n1##z,v), \
 I[224] = (img)(_p3##x,_n1##y,_n1##z,v), I[225] = (img)(_p2##x,_n1##y,_n1##z,v), I[226] = (img)(_p1##x,_n1##y,_n1##z,v), I[227] = (img)(x,_n1##y,_n1##z,v), I[228] = (img)(_n1##x,_n1##y,_n1##z,v), I[229] = (img)(_n2##x,_n1##y,_n1##z,v), I[230] = (img)(_n3##x,_n1##y,_n1##z,v), \
 I[231] = (img)(_p3##x,_n2##y,_n1##z,v), I[232] = (img)(_p2##x,_n2##y,_n1##z,v), I[233] = (img)(_p1##x,_n2##y,_n1##z,v), I[234] = (img)(x,_n2##y,_n1##z,v), I[235] = (img)(_n1##x,_n2##y,_n1##z,v), I[236] = (img)(_n2##x,_n2##y,_n1##z,v), I[237] = (img)(_n3##x,_n2##y,_n1##z,v), \
 I[238] = (img)(_p3##x,_n3##y,_n1##z,v), I[239] = (img)(_p2##x,_n3##y,_n1##z,v), I[240] = (img)(_p1##x,_n3##y,_n1##z,v), I[241] = (img)(x,_n3##y,_n1##z,v), I[242] = (img)(_n1##x,_n3##y,_n1##z,v), I[243] = (img)(_n2##x,_n3##y,_n1##z,v), I[244] = (img)(_n3##x,_n3##y,_n1##z,v), \
 I[245] = (img)(_p3##x,_p3##y,_n2##z,v), I[246] = (img)(_p2##x,_p3##y,_n2##z,v), I[247] = (img)(_p1##x,_p3##y,_n2##z,v), I[248] = (img)(x,_p3##y,_n2##z,v), I[249] = (img)(_n1##x,_p3##y,_n2##z,v), I[250] = (img)(_n2##x,_p3##y,_n2##z,v), I[251] = (img)(_n3##x,_p3##y,_n2##z,v), \
 I[252] = (img)(_p3##x,_p2##y,_n2##z,v), I[253] = (img)(_p2##x,_p2##y,_n2##z,v), I[254] = (img)(_p1##x,_p2##y,_n2##z,v), I[255] = (img)(x,_p2##y,_n2##z,v), I[256] = (img)(_n1##x,_p2##y,_n2##z,v), I[257] = (img)(_n2##x,_p2##y,_n2##z,v), I[258] = (img)(_n3##x,_p2##y,_n2##z,v), \
 I[259] = (img)(_p3##x,_p1##y,_n2##z,v), I[260] = (img)(_p2##x,_p1##y,_n2##z,v), I[261] = (img)(_p1##x,_p1##y,_n2##z,v), I[262] = (img)(x,_p1##y,_n2##z,v), I[263] = (img)(_n1##x,_p1##y,_n2##z,v), I[264] = (img)(_n2##x,_p1##y,_n2##z,v), I[265] = (img)(_n3##x,_p1##y,_n2##z,v), \
 I[266] = (img)(_p3##x,y,_n2##z,v), I[267] = (img)(_p2##x,y,_n2##z,v), I[268] = (img)(_p1##x,y,_n2##z,v), I[269] = (img)(x,y,_n2##z,v), I[270] = (img)(_n1##x,y,_n2##z,v), I[271] = (img)(_n2##x,y,_n2##z,v), I[272] = (img)(_n3##x,y,_n2##z,v), \
 I[273] = (img)(_p3##x,_n1##y,_n2##z,v), I[274] = (img)(_p2##x,_n1##y,_n2##z,v), I[275] = (img)(_p1##x,_n1##y,_n2##z,v), I[276] = (img)(x,_n1##y,_n2##z,v), I[277] = (img)(_n1##x,_n1##y,_n2##z,v), I[278] = (img)(_n2##x,_n1##y,_n2##z,v), I[279] = (img)(_n3##x,_n1##y,_n2##z,v), \
 I[280] = (img)(_p3##x,_n2##y,_n2##z,v), I[281] = (img)(_p2##x,_n2##y,_n2##z,v), I[282] = (img)(_p1##x,_n2##y,_n2##z,v), I[283] = (img)(x,_n2##y,_n2##z,v), I[284] = (img)(_n1##x,_n2##y,_n2##z,v), I[285] = (img)(_n2##x,_n2##y,_n2##z,v), I[286] = (img)(_n3##x,_n2##y,_n2##z,v), \
 I[287] = (img)(_p3##x,_n3##y,_n2##z,v), I[288] = (img)(_p2##x,_n3##y,_n2##z,v), I[289] = (img)(_p1##x,_n3##y,_n2##z,v), I[290] = (img)(x,_n3##y,_n2##z,v), I[291] = (img)(_n1##x,_n3##y,_n2##z,v), I[292] = (img)(_n2##x,_n3##y,_n2##z,v), I[293] = (img)(_n3##x,_n3##y,_n2##z,v), \
 I[294] = (img)(_p3##x,_p3##y,_n3##z,v), I[295] = (img)(_p2##x,_p3##y,_n3##z,v), I[296] = (img)(_p1##x,_p3##y,_n3##z,v), I[297] = (img)(x,_p3##y,_n3##z,v), I[298] = (img)(_n1##x,_p3##y,_n3##z,v), I[299] = (img)(_n2##x,_p3##y,_n3##z,v), I[300] = (img)(_n3##x,_p3##y,_n3##z,v), \
 I[301] = (img)(_p3##x,_p2##y,_n3##z,v), I[302] = (img)(_p2##x,_p2##y,_n3##z,v), I[303] = (img)(_p1##x,_p2##y,_n3##z,v), I[304] = (img)(x,_p2##y,_n3##z,v), I[305] = (img)(_n1##x,_p2##y,_n3##z,v), I[306] = (img)(_n2##x,_p2##y,_n3##z,v), I[307] = (img)(_n3##x,_p2##y,_n3##z,v), \
 I[308] = (img)(_p3##x,_p1##y,_n3##z,v), I[309] = (img)(_p2##x,_p1##y,_n3##z,v), I[310] = (img)(_p1##x,_p1##y,_n3##z,v), I[311] = (img)(x,_p1##y,_n3##z,v), I[312] = (img)(_n1##x,_p1##y,_n3##z,v), I[313] = (img)(_n2##x,_p1##y,_n3##z,v), I[314] = (img)(_n3##x,_p1##y,_n3##z,v), \
 I[315] = (img)(_p3##x,y,_n3##z,v), I[316] = (img)(_p2##x,y,_n3##z,v), I[317] = (img)(_p1##x,y,_n3##z,v), I[318] = (img)(x,y,_n3##z,v), I[319] = (img)(_n1##x,y,_n3##z,v), I[320] = (img)(_n2##x,y,_n3##z,v), I[321] = (img)(_n3##x,y,_n3##z,v), \
 I[322] = (img)(_p3##x,_n1##y,_n3##z,v), I[323] = (img)(_p2##x,_n1##y,_n3##z,v), I[324] = (img)(_p1##x,_n1##y,_n3##z,v), I[325] = (img)(x,_n1##y,_n3##z,v), I[326] = (img)(_n1##x,_n1##y,_n3##z,v), I[327] = (img)(_n2##x,_n1##y,_n3##z,v), I[328] = (img)(_n3##x,_n1##y,_n3##z,v), \
 I[329] = (img)(_p3##x,_n2##y,_n3##z,v), I[330] = (img)(_p2##x,_n2##y,_n3##z,v), I[331] = (img)(_p1##x,_n2##y,_n3##z,v), I[332] = (img)(x,_n2##y,_n3##z,v), I[333] = (img)(_n1##x,_n2##y,_n3##z,v), I[334] = (img)(_n2##x,_n2##y,_n3##z,v), I[335] = (img)(_n3##x,_n2##y,_n3##z,v), \
 I[336] = (img)(_p3##x,_n3##y,_n3##z,v), I[337] = (img)(_p2##x,_n3##y,_n3##z,v), I[338] = (img)(_p1##x,_n3##y,_n3##z,v), I[339] = (img)(x,_n3##y,_n3##z,v), I[340] = (img)(_n1##x,_n3##y,_n3##z,v), I[341] = (img)(_n2##x,_n3##y,_n3##z,v), I[342] = (img)(_n3##x,_n3##y,_n3##z,v);

// Define 8x8x8 loop macros for CImg
//-------------------------------------
#define cimg_for_in8(bound,i0,i1,i) for (int i = (int)(i0)<0?0:(int)(i0), \
 _p3##i = i-3<0?0:i-3, \
 _p2##i = i-2<0?0:i-2, \
 _p1##i = i-1<0?0:i-1, \
 _n1##i = i+1>=(int)(bound)?(int)(bound)-1:i+1, \
 _n2##i = i+2>=(int)(bound)?(int)(bound)-1:i+2, \
 _n3##i = i+3>=(int)(bound)?(int)(bound)-1:i+3, \
 _n4##i = i+4>=(int)(bound)?(int)(bound)-1:i+4; \
 i<=(int)(i1) && (_n4##i<(int)(bound) || _n3##i==--_n4##i || _n2##i==--_n3##i || _n1##i==--_n2##i || \
 i==(_n4##i = _n3##i = _n2##i = --_n1##i)); \
 _p3##i = _p2##i, _p2##i = _p1##i, _p1##i = i++, \
 ++_n1##i, ++_n2##i, ++_n3##i, ++_n4##i)

#define cimg_for_in8X(img,x0,x1,x) cimg_for_in8((img).width,x0,x1,x)
#define cimg_for_in8Y(img,y0,y1,y) cimg_for_in8((img).height,y0,y1,y)
#define cimg_for_in8Z(img,z0,z1,z) cimg_for_in8((img).depth,z0,z1,z)
#define cimg_for_in8V(img,v0,v1,v) cimg_for_in8((img).dim,v0,v1,v)
#define cimg_for_in8XY(img,x0,y0,x1,y1,x,y) cimg_for_in8Y(img,y0,y1,y) cimg_for_in8X(img,x0,x1,x)
#define cimg_for_in8XZ(img,x0,z0,x1,z1,x,z) cimg_for_in8Z(img,z0,z1,z) cimg_for_in8X(img,x0,x1,x)
#define cimg_for_in8XV(img,x0,v0,x1,v1,x,v) cimg_for_in8V(img,v0,v1,v) cimg_for_in8X(img,x0,x1,x)
#define cimg_for_in8YZ(img,y0,z0,y1,z1,y,z) cimg_for_in8Z(img,z0,z1,z) cimg_for_in8Y(img,y0,y1,y)
#define cimg_for_in8YV(img,y0,v0,y1,v1,y,v) cimg_for_in8V(img,v0,v1,v) cimg_for_in8Y(img,y0,y1,y)
#define cimg_for_in8ZV(img,z0,v0,z1,v1,z,v) cimg_for_in8V(img,v0,v1,v) cimg_for_in8Z(img,z0,z1,z)
#define cimg_for_in8XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z) cimg_for_in8Z(img,z0,z1,z) cimg_for_in8XY(img,x0,y0,x1,y1,x,y)
#define cimg_for_in8XZV(img,x0,z0,v0,x1,y1,v1,x,z,v) cimg_for_in8V(img,v0,v1,v) cimg_for_in8XZ(img,x0,y0,x1,y1,x,z)
#define cimg_for_in8YZV(img,y0,z0,v0,y1,z1,v1,y,z,v) cimg_for_in8V(img,v0,v1,v) cimg_for_in8YZ(img,y0,z0,y1,z1,y,z)
#define cimg_for_in8XYZV(img,x0,y0,z0,v0,x1,y1,z1,v1,x,y,z,v) cimg_for_in8V(img,v0,v1,v) cimg_for_in8XYZ(img,x0,y0,z0,x1,y1,z1,x,y,z)

#define cimg_for8x8x8(img,x,y,z,v,I) \
 cimg_for8((img).depth,z) cimg_for8((img).height,y) for (int x = 0, \
 _p3##x = 0, _p2##x = 0, _p1##x = 0, \
 _n1##x = 1>=((img).width)?(int)((img).width)-1:1, \
 _n2##x = 2>=((img).width)?(int)((img).width)-1:2, \
 _n3##x = 3>=((img).width)?(int)((img).width)-1:3, \
 _n4##x = (int)( \
 (I[0] = I[1] = I[2] = I[3] = (img)(0,_p3##y,_p3##z,v)), \
 (I[8] = I[9] = I[10] = I[11] = (img)(0,_p2##y,_p3##z,v)), \
 (I[16] = I[17] = I[18] = I[19] = (img)(0,_p1##y,_p3##z,v)), \
 (I[24] = I[25] = I[26] = I[27] = (img)(0,y,_p3##z,v)), \
 (I[32] = I[33] = I[34] = I[35] = (img)(0,_n1##y,_p3##z,v)), \
 (I[40] = I[41] = I[42] = I[43] = (img)(0,_n2##y,_p3##z,v)), \
 (I[48] = I[49] = I[50] = I[51] = (img)(0,_n3##y,_p3##z,v)), \
 (I[56] = I[57] = I[58] = I[59] = (img)(0,_n4##y,_p3##z,v)), \
 (I[64] = I[65] = I[66] = I[67] = (img)(0,_p3##y,_p2##z,v)), \
 (I[72] = I[73] = I[74] = I[75] = (img)(0,_p2##y,_p2##z,v)), \
 (I[80] = I[81] = I[82] = I[83] = (img)(0,_p1##y,_p2##z,v)), \
 (I[88] = I[89] = I[90] = I[91] = (img)(0,y,_p2##z,v)), \
 (I[96] = I[97] = I[98] = I[99] = (img)(0,_n1##y,_p2##z,v)), \
 (I[104] = I[105] = I[106] = I[107] = (img)(0,_n2##y,_p2##z,v)), \
 (I[112] = I[113] = I[114] = I[115] = (img)(0,_n3##y,_p2##z,v)), \
 (I[120] = I[121] = I[122] = I[123] = (img)(0,_n4##y,_p2##z,v)), \
 (I[128] = I[129] = I[130] = I[131] = (img)(0,_p3##y,_p1##z,v)), \
 (I[136] = I[137] = I[138] = I[139] = (img)(0,_p2##y,_p1##z,v)), \
 (I[144] = I[145] = I[146] = I[147] = (img)(0,_p1##y,_p1##z,v)), \
 (I[152] = I[153] = I[154] = I[155] = (img)(0,y,_p1##z,v)), \
 (I[160] = I[161] = I[162] = I[163] = (img)(0,_n1##y,_p1##z,v)), \
 (I[168] = I[169] = I[170] = I[171] = (img)(0,_n2##y,_p1##z,v)), \
 (I[176] = I[177] = I[178] = I[179] = (img)(0,_n3##y,_p1##z,v)), \
 (I[184] = I[185] = I[186] = I[187] = (img)(0,_n4##y,_p1##z,v)), \
 (I[192] = I[193] = I[194] = I[195] = (img)(0,_p3##y,z,v)), \
 (I[200] = I[201] = I[202] = I[203] = (img)(0,_p2##y,z,v)), \
 (I[208] = I[209] = I[210] = I[211] = (img)(0,_p1##y,z,v)), \
 (I[216] = I[217] = I[218] = I[219] = (img)(0,y,z,v)), \
 (I[224] = I[225] = I[226] = I[227] = (img)(0,_n1##y,z,v)), \
 (I[232] = I[233] = I[234] = I[235] = (img)(0,_n2##y,z,v)), \
 (I[240] = I[241] = I[242] = I[243] = (img)(0,_n3##y,z,v)), \
 (I[248] = I[249] = I[250] = I[251] = (img)(0,_n4##y,z,v)), \
 (I[256] = I[257] = I[258] = I[259] = (img)(0,_p3##y,_n1##z,v)), \
 (I[264] = I[265] = I[266] = I[267] = (img)(0,_p2##y,_n1##z,v)), \
 (I[272] = I[273] = I[274] = I[275] = (img)(0,_p1##y,_n1##z,v)), \
 (I[280] = I[281] = I[282] = I[283] = (img)(0,y,_n1##z,v)), \
 (I[288] = I[289] = I[290] = I[291] = (img)(0,_n1##y,_n1##z,v)), \
 (I[296] = I[297] = I[298] = I[299] = (img)(0,_n2##y,_n1##z,v)), \
 (I[304] = I[305] = I[306] = I[307] = (img)(0,_n3##y,_n1##z,v)), \
 (I[312] = I[313] = I[314] = I[315] = (img)(0,_n4##y,_n1##z,v)), \
 (I[320] = I[321] = I[322] = I[323] = (img)(0,_p3##y,_n2##z,v)), \
 (I[328] = I[329] = I[330] = I[331] = (img)(0,_p2##y,_n2##z,v)), \
 (I[336] = I[337] = I[338] = I[339] = (img)(0,_p1##y,_n2##z,v)), \
 (I[344] = I[345] = I[346] = I[347] = (img)(0,y,_n2##z,v)), \
 (I[352] = I[353] = I[354] = I[355] = (img)(0,_n1##y,_n2##z,v)), \
 (I[360] = I[361] = I[362] = I[363] = (img)(0,_n2##y,_n2##z,v)), \
 (I[368] = I[369] = I[370] = I[371] = (img)(0,_n3##y,_n2##z,v)), \
 (I[376] = I[377] = I[378] = I[379] = (img)(0,_n4##y,_n2##z,v)), \
 (I[384] = I[385] = I[386] = I[387] = (img)(0,_p3##y,_n3##z,v)), \
 (I[392] = I[393] = I[394] = I[395] = (img)(0,_p2##y,_n3##z,v)), \
 (I[400] = I[401] = I[402] = I[403] = (img)(0,_p1##y,_n3##z,v)), \
 (I[408] = I[409] = I[410] = I[411] = (img)(0,y,_n3##z,v)), \
 (I[416] = I[417] = I[418] = I[419] = (img)(0,_n1##y,_n3##z,v)), \
 (I[424] = I[425] = I[426] = I[427] = (img)(0,_n2##y,_n3##z,v)), \
 (I[432] = I[433] = I[434] = I[435] = (img)(0,_n3##y,_n3##z,v)), \
 (I[440] = I[441] = I[442] = I[443] = (img)(0,_n4##y,_n3##z,v)), \
 (I[448] = I[449] = I[450] = I[451] = (img)(0,_p3##y,_n4##z,v)), \
 (I[456] = I[457] = I[458] = I[459] = (img)(0,_p2##y,_n4##z,v)), \
 (I[464] = I[465] = I[466] = I[467] = (img)(0,_p1##y,_n4##z,v)), \
 (I[472] = I[473] = I[474] = I[475] = (img)(0,y,_n4##z,v)), \
 (I[480] = I[481] = I[482] = I[483] = (img)(0,_n1##y,_n4##z,v)), \
 (I[488] = I[489] = I[490] = I[491] = (img)(0,_n2##y,_n4##z,v)), \
 (I[496] = I[497] = I[498] = I[499] = (img)(0,_n3##y,_n4##z,v)), \
 (I[504] = I[505] = I[506] = I[507] = (img)(0,_n4##y,_n4##z,v)), \
 (I[4] = (img)(_n1##x,_p3##y,_p3##z,v)), \
 (I[12] = (img)(_n1##x,_p2##y,_p3##z,v)), \
 (I[20] = (img)(_n1##x,_p1##y,_p3##z,v)), \
 (I[28] = (img)(_n1##x,y,_p3##z,v)), \
 (I[36] = (img)(_n1##x,_n1##y,_p3##z,v)), \
 (I[44] = (img)(_n1##x,_n2##y,_p3##z,v)), \
 (I[52] = (img)(_n1##x,_n3##y,_p3##z,v)), \
 (I[60] = (img)(_n1##x,_n4##y,_p3##z,v)), \
 (I[68] = (img)(_n1##x,_p3##y,_p2##z,v)), \
 (I[76] = (img)(_n1##x,_p2##y,_p2##z,v)), \
 (I[84] = (img)(_n1##x,_p1##y,_p2##z,v)), \
 (I[92] = (img)(_n1##x,y,_p2##z,v)), \
 (I[100] = (img)(_n1##x,_n1##y,_p2##z,v)), \
 (I[108] = (img)(_n1##x,_n2##y,_p2##z,v)), \
 (I[116] = (img)(_n1##x,_n3##y,_p2##z,v)), \
 (I[124] = (img)(_n1##x,_n4##y,_p2##z,v)), \
 (I[132] = (img)(_n1##x,_p3##y,_p1##z,v)), \
 (I[140] = (img)(_n1##x,_p2##y,_p1##z,v)), \
 (I[148] = (img)(_n1##x,_p1##y,_p1##z,v)), \
 (I[156] = (img)(_n1##x,y,_p1##z,v)), \
 (I[164] = (img)(_n1##x,_n1##y,_p1##z,v)), \
 (I[172] = (img)(_n1##x,_n2##y,_p1##z,v)), \
 (I[180] = (img)(_n1##x,_n3##y,_p1##z,v)), \
 (I[188] = (img)(_n1##x,_n4##y,_p1##z,v)), \
 (I[196] = (img)(_n1##x,_p3##y,z,v)), \
 (I[204] = (img)(_n1##x,_p2##y,z,v)), \
 (I[212] = (img)(_n1##x,_p1##y,z,v)), \
 (I[220] = (img)(_n1##x,y,z,v)), \
 (I[228] = (img)(_n1##x,_n1##y,z,v)), \
 (I[236] = (img)(_n1##x,_n2##y,z,v)), \
 (I[244] = (img)(_n1##x,_n3##y,z,v)), \
 (I[252] = (img)(_n1##x,_n4##y,z,v)), \
 (I[260] = (img)(_n1##x,_p3##y,_n1##z,v)), \
 (I[268] = (img)(_n1##x,_p2##y,_n1##z,v)), \
 (I[276] = (img)(_n1##x,_p1##y,_n1##z,v)), \
 (I[284] = (img)(_n1##x,y,_n1##z,v)), \
 (I[292] = (img)(_n1##x,_n1##y,_n1##z,v)), \
 (I[300] = (img)(_n1##x,_n2##y,_n1##z,v)), \
 (I[308] = (img)(_n1##x,_n3##y,_n1##z,v)), \
 (I[316] = (img)(_n1##x,_n4##y,_n1##z,v)), \
 (I[324] = (img)(_n1##x,_p3##y,_n2##z,v)), \
 (I[332] = (img)(_n1##x,_p2##y,_n2##z,v)), \
 (I[340] = (img)(_n1##x,_p1##y,_n2##z,v)), \
 (I[348] = (img)(_n1##x,y,_n2##z,v)), \
 (I[356] = (img)(_n1##x,_n1##y,_n2##z,v)), \
 (I[364] = (img)(_n1##x,_n2##y,_n2##z,v)), \
 (I[372] = (img)(_n1##x,_n3##y,_n2##z,v)), \
 (I[380] = (img)(_n1##x,_n4##y,_n2##z,v)), \
 (I[388] = (img)(_n1##x,_p3##y,_n3##z,v)), \
 (I[396] = (img)(_n1##x,_p2##y,_n3##z,v)), \
 (I[404] = (img)(_n1##x,_p1##y,_n3##z,v)), \
 (I[412] = (img)(_n1##x,y,_n3##z,v)), \
 (I[420] = (img)(_n1##x,_n1##y,_n3##z,v)), \
 (I[428] = (img)(_n1##x,_n2##y,_n3##z,v)), \
 (I[436] = (img)(_n1##x,_n3##y,_n3##z,v)), \
 (I[444] = (img)(_n1##x,_n4##y,_n3##z,v)), \
 (I[452] = (img)(_n1##x,_p3##y,_n4##z,v)), \
 (I[460] = (img)(_n1##x,_p2##y,_n4##z,v)), \
 (I[468] = (img)(_n1##x,_p1##y,_n4##z,v)), \
 (I[476] = (img)(_n1##x,y,_n4##z,v)), \
 (I[484] = (img)(_n1##x,_n1##y,_n4##z,v)), \
 (I[492] = (img)(_n1##x,_n2##y,_n4##z,v)), \
 (I[500] = (img)(_n1##x,_n3##y,_n4##z,v)), \
 (I[508] = (img)(_n1##x,_n4##y,_n4##z,v)), \
 (I[5] = (img)(_n2##x,_p3##y,_p3##z,v)), \
 (I[13] = (img)(_n2##x,_p2##y,_p3##z,v)), \
 (I[21] = (img)(_n2##x,_p1##y,_p3##z,v)), \
 (I[29] = (img)(_n2##x,y,_p3##z,v)), \
 (I[37] = (img)(_n2##x,_n1##y,_p3##z,v)), \
 (I[45] = (img)(_n2##x,_n2##y,_p3##z,v)), \
 (I[53] = (img)(_n2##x,_n3##y,_p3##z,v)), \
 (I[61] = (img)(_n2##x,_n4##y,_p3##z,v)), \
 (I[69] = (img)(_n2##x,_p3##y,_p2##z,v)), \
 (I[77] = (img)(_n2##x,_p2##y,_p2##z,v)), \
 (I[85] = (img)(_n2##x,_p1##y,_p2##z,v)), \
 (I[93] = (img)(_n2##x,y,_p2##z,v)), \
 (I[101] = (img)(_n2##x,_n1##y,_p2##z,v)), \
 (I[109] = (img)(_n2##x,_n2##y,_p2##z,v)), \
 (I[117] = (img)(_n2##x,_n3##y,_p2##z,v)), \
 (I[125] = (img)(_n2##x,_n4##y,_p2##z,v)), \
 (I[133] = (img)(_n2##x,_p3##y,_p1##z,v)), \
 (I[141] = (img)(_n2##x,_p2##y,_p1##z,v)), \
 (I[149] = (img)(_n2##x,_p1##y,_p1##z,v)), \
 (I[157] = (img)(_n2##x,y,_p1##z,v)), \
 (I[165] = (img)(_n2##x,_n1##y,_p1##z,v)), \
 (I[173] = (img)(_n2##x,_n2##y,_p1##z,v)), \
 (I[181] = (img)(_n2##x,_n3##y,_p1##z,v)), \
 (I[189] = (img)(_n2##x,_n4##y,_p1##z,v)), \
 (I[197] = (img)(_n2##x,_p3##y,z,v)), \
 (I[205] = (img)(_n2##x,_p2##y,z,v)), \
 (I[213] = (img)(_n2##x,_p1##y,z,v)), \
 (I[221] = (img)(_n2##x,y,z,v)), \
 (I[229] = (img)(_n2##x,_n1##y,z,v)), \
 (I[237] = (img)(_n2##x,_n2##y,z,v)), \
 (I[245] = (img)(_n2##x,_n3##y,z,v)), \
 (I[253] = (img)(_n2##x,_n4##y,z,v)), \
 (I[261] = (img)(_n2##x,_p3##y,_n1##z,v)), \
 (I[269] = (img)(_n2##x,_p2##y,_n1##z,v)), \
 (I[277] = (img)(_n2##x,_p1##y,_n1##z,v)), \
 (I[285] = (img)(_n2##x,y,_n1##z,v)), \
 (I[293] = (img)(_n2##x,_n1##y,_n1##z,v)), \
 (I[301] = (img)(_n2##x,_n2##y,_n1##z,v)), \
 (I[309] = (img)(_n2##x,_n3##y,_n1##z,v)), \
 (I[317] = (img)(_n2##x,_n4##y,_n1##z,v)), \
 (I[325] = (img)(_n2##x,_p3##y,_n2##z,v)), \
 (I[333] = (img)(_n2##x,_p2##y,_n2##z,v)), \
 (I[341] = (img)(_n2##x,_p1##y,_n2##z,v)), \
 (I[349] = (img)(_n2##x,y,_n2##z,v)), \
 (I[357] = (img)(_n2##x,_n1##y,_n2##z,v)), \
 (I[365] = (img)(_n2##x,_n2##y,_n2##z,v)), \
 (I[373] = (img)(_n2##x,_n3##y,_n2##z,v)), \
 (I[381] = (img)(_n2##x,_n4##y,_n2##z,v)), \
 (I[389] = (img)(_n2##x,_p3##y,_n3##z,v)), \
 (I[397] = (img)(_n2##x,_p2##y,_n3##z,v)), \
 (I[405] = (img)(_n2##x,_p1##y,_n3##z,v)), \
 (I[413] = (img)(_n2##x,y,_n3##z,v)), \
 (I[421] = (img)(_n2##x,_n1##y,_n3##z,v)), \
 (I[429] = (img)(_n2##x,_n2##y,_n3##z,v)), \
 (I[437] = (img)(_n2##x,_n3##y,_n3##z,v)), \
 (I[445] = (img)(_n2##x,_n4##y,_n3##z,v)), \
 (I[453] = (img)(_n2##x,_p3##y,_n4##z,v)), \
 (I[461] = (img)(_n2##x,_p2##y,_n4##z,v)), \
 (I[469] = (img)(_n2##x,_p1##y,_n4##z,v)), \
 (I[477] = (img)(_n2##x,y,_n4##z,v)), \
 (I[485] = (img)(_n2##x,_n1##y,_n4##z,v)), \
 (I[493] = (img)(_n2##x,_n2##y,_n4##z,v)), \
 (I[501] = (img)(_n2##x,_n3##y,_n4##z,v)), \
 (I[509] = (img)(_n2##x,_n4##y,_n4##z,v)), \
 (I[6] = (img)(_n3##x,_p3##y,_p3##z,v)), \
 (I[14] = (img)(_n3##x,_p2##y,_p3##z,v)), \
 (I[22] = (img)(_n3##x,_p1##y,_p3##z,v)), \
 (I[30] = (img)(_n3##x,y,_p3##z,v)), \
 (I[38] = (img)(_n3##x,_n1##y,_p3##z,v)), \
 (I[46] = (img)(_n3##x,_n2##y,_p3##z,v)), \
 (I[54] = (img)(_n3##x,_n3##y,_p3##z,v)), \
 (I[62] = (img)(_n3##x,_n4##y,_p3##z,v)), \
 (I[70] = (img)(_n3##x,_p3##y,_p2##z,v)), \
 (I[78] = (img)(_n3##x,_p2##y,_p2##z,v)), \
 (I[86] = (img)(_n3##x,_p1##y,_p2##z,v)), \
 (I[94] = (img)(_n3##x,y,_p2##z,v)), \
 (I[102] = (img)(_n3##x,_n1##y,_p2##z,v)), \
 (I[110] = (img)(_n3##x,_n2##y,_p2##z,v)), \
 (I[118] = (img)(_n3##x,_n3##y,_p2##z,v)), \
 (I[126] = (img)(_n3##x,_n4##y,_p2##z,v)), \
 (I[134] = (img)(_n3##x,_p3##y,_p1##z,v)), \
 (I[142] = (img)(_n3##x,_p2##y,_p1##z,v)), \
 (I[150] = (img)(_n3##x,_p1##y,_p1##z,v)), \
 (I[158] = (img)(_n3##x,y,_p1##z,v)), \
 (I[166] = (img)(_n3##x,_n1##y,_p1##z,v)), \
 (I[174] = (img)(_n3##x,_n2##y,_p1##z,v)), \
 (I[182] = (img)(_n3##x,_n3##y,_p1##z,v)), \
 (I[190] = (img)(_n3##x,_n4##y,_p1##z,v)), \
 (I[198] = (img)(_n3##x,_p3##y,z,v)), \
 (I[206] = (img)(_n3##x,_p2##y,z,v)), \
 (I[214] = (img)(_n3##x,_p1##y,z,v)), \
 (I[222] = (img)(_n3##x,y,z,v)), \
 (I[230] = (img)(_n3##x,_n1##y,z,v)), \
 (I[238] = (img)(_n3##x,_n2##y,z,v)), \
 (I[246] = (img)(_n3##x,_n3##y,z,v)), \
 (I[254] = (img)(_n3##x,_n4##y,z,v)), \
 (I[262] = (img)(_n3##x,_p3##y,_n1##z,v)), \
 (I[270] = (img)(_n3##x,_p2##y,_n1##z,v)), \
 (I[278] = (img)(_n3##x,_p1##y,_n1##z,v)), \
 (I[286] = (img)(_n3##x,y,_n1##z,v)), \
 (I[294] = (img)(_n3##x,_n1##y,_n1##z,v)), \
 (I[302] = (img)(_n3##x,_n2##y,_n1##z,v)), \
 (I[310] = (img)(_n3##x,_n3##y,_n1##z,v)), \
 (I[318] = (img)(_n3##x,_n4##y,_n1##z,v)), \
 (I[326] = (img)(_n3##x,_p3##y,_n2##z,v)), \
 (I[334] = (img)(_n3##x,_p2##y,_n2##z,v)), \
 (I[342] = (img)(_n3##x,_p1##y,_n2##z,v)), \
 (I[350] = (img)(_n3##x,y,_n2##z,v)), \
 (I[358] = (img)(_n3##x,_n1##y,_n2##z,v)), \
 (I[366] = (img)(_n3##x,_n2##y,_n2##z,v)), \
 (I[374] = (img)(_n3##x,_n3##y,_n2##z,v)), \
 (I[382] = (img)(_n3##x,_n4##y,_n2##z,v)), \
 (I[390] = (img)(_n3##x,_p3##y,_n3##z,v)), \
 (I[398] = (img)(_n3##x,_p2##y,_n3##z,v)), \
 (I[406] = (img)(_n3##x,_p1##y,_n3##z,v)), \
 (I[414] = (img)(_n3##x,y,_n3##z,v)), \
 (I[422] = (img)(_n3##x,_n1##y,_n3##z,v)), \
 (I[430] = (img)(_n3##x,_n2##y,_n3##z,v)), \
 (I[438] = (img)(_n3##x,_n3##y,_n3##z,v)), \
 (I[446] = (img)(_n3##x,_n4##y,_n3##z,v)), \
 (I[454] = (img)(_n3##x,_p3##y,_n4##z,v)), \
 (I[462] = (img)(_n3##x,_p2##y,_n4##z,v)), \
 (I[470] = (img)(_n3##x,_p1##y,_n4##z,v)), \
 (I[478] = (img)(_n3##x,y,_n4##z,v)), \
 (I[486] = (img)(_n3##x,_n1##y,_n4##z,v)), \
 (I[494] = (img)(_n3##x,_n2##y,_n4##z,v)), \
 (I[502] = (img)(_n3##x,_n3##y,_n4##z,v)), \
 (I[510] = (img)(_n3##x,_n4##y,_n4##z,v)), \
 4>=((img).width)?(int)((img).width)-1:4); \
 (_n4##x<(int)((img).width) && ( \
 (I[7] = (img)(_n4##x,_p3##y,_p3##z,v)), \
 (I[15] = (img)(_n4##x,_p2##y,_p3##z,v)), \
 (I[23] = (img)(_n4##x,_p1##y,_p3##z,v)), \
 (I[31] = (img)(_n4##x,y,_p3##z,v)), \
 (I[39] = (img)(_n4##x,_n1##y,_p3##z,v)), \
 (I[47] = (img)(_n4##x,_n2##y,_p3##z,v)), \
 (I[55] = (img)(_n4##x,_n3##y,_p3##z,v)), \
 (I[63] = (img)(_n4##x,_n4##y,_p3##z,v)), \
 (I[71] = (img)(_n4##x,_p3##y,_p2##z,v)), \
 (I[79] = (img)(_n4##x,_p2##y,_p2##z,v)), \
 (I[87] = (img)(_n4##x,_p1##y,_p2##z,v)), \
 (I[95] = (img)(_n4##x,y,_p2##z,v)), \
 (I[103] = (img)(_n4##x,_n1##y,_p2##z,v)), \
 (I[111] = (img)(_n4##x,_n2##y,_p2##z,v)), \
 (I[119] = (img)(_n4##x,_n3##y,_p2##z,v)), \
 (I[127] = (img)(_n4##x,_n4##y,_p2##z,v)), \
 (I[135] = (img)(_n4##x,_p3##y,_p1##z,v)), \
 (I[143] = (img)(_n4##x,_p2##y,_p1##z,v)), \
 (I[151] = (img)(_n4##x,_p1##y,_p1##z,v)), \
 (I[159] = (img)(_n4##x,y,_p1##z,v)), \
 (I[167] = (img)(_n4##x,_n1##y,_p1##z,v)), \
 (I[175] = (img)(_n4##x,_n2##y,_p1##z,v)), \
 (I[183] = (img)(_n4##x,_n3##y,_p1##z,v)), \
 (I[191] = (img)(_n4##x,_n4##y,_p1##z,v)), \
 (I[199] = (img)(_n4##x,_p3##y,z,v)), \
 (I[207] = (img)(_n4##x,_p2##y,z,v)), \
 (I[215] = (img)(_n4##x,_p1##y,z,v)), \
 (I[223] = (img)(_n4##x,y,z,v)), \
 (I[231] = (img)(_n4##x,_n1##y,z,v)), \
 (I[239] = (img)(_n4##x,_n2##y,z,v)), \
 (I[247] = (img)(_n4##x,_n3##y,z,v)), \
 (I[255] = (img)(_n4##x,_n4##y,z,v)), \
 (I[263] = (img)(_n4##x,_p3##y,_n1##z,v)), \
 (I[271] = (img)(_n4##x,_p2##y,_n1##z,v)), \
 (I[279] = (img)(_n4##x,_p1##y,_n1##z,v)), \
 (I[287] = (img)(_n4##x,y,_n1##z,v)), \
 (I[295] = (img)(_n4##x,_n1##y,_n1##z,v)), \
 (I[303] = (img)(_n4##x,_n2##y,_n1##z,v)), \
 (I[311] = (img)(_n4##x,_n3##y,_n1##z,v)), \
 (I[319] = (img)(_n4##x,_n4##y,_n1##z,v)), \
 (I[327] = (img)(_n4##x,_p3##y,_n2##z,v)), \
 (I[335] = (img)(_n4##x,_p2##y,_n2##z,v)), \
 (I[343] = (img)(_n4##x,_p1##y,_n2##z,v)), \
 (I[351] = (img)(_n4##x,y,_n2##z,v)), \
 (I[359] = (img)(_n4##x,_n1##y,_n2##z,v)), \
 (I[367] = (img)(_n4##x,_n2##y,_n2##z,v)), \
 (I[375] = (img)(_n4##x,_n3##y,_n2##z,v)), \
 (I[383] = (img)(_n4##x,_n4##y,_n2##z,v)), \
 (I[391] = (img)(_n4##x,_p3##y,_n3##z,v)), \
 (I[399] = (img)(_n4##x,_p2##y,_n3##z,v)), \
 (I[407] = (img)(_n4##x,_p1##y,_n3##z,v)), \
 (I[415] = (img)(_n4##x,y,_n3##z,v)), \
 (I[423] = (img)(_n4##x,_n1##y,_n3##z,v)), \
 (I[431] = (img)(_n4##x,_n2##y,_n3##z,v)), \
 (I[439] = (img)(_n4##x,_n3##y,_n3##z,v)), \
 (I[447] = (img)(_n4##x,_n4##y,_n3##z,v)), \
 (I[455] = (img)(_n4##x,_p3##y,_n4##z,v)), \
 (I[463] = (img)(_n4##x,_p2##y,_n4##z,v)), \
 (I[471] = (img)(_n4##x,_p1##y,_n4##z,v)), \
 (I[479] = (img)(_n4##x,y,_n4##z,v)), \
 (I[487] = (img)(_n4##x,_n1##y,_n4##z,v)), \
 (I[495] = (img)(_n4##x,_n2##y,_n4##z,v)), \
 (I[503] = (img)(_n4##x,_n3##y,_n4##z,v)), \
 (I[511] = (img)(_n4##x,_n4##y,_n4##z,v)),1)) || \
 _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n4##x = _n3##x = _n2##x = --_n1##x); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], \
 I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], \
 I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], \
 I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], \
 I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], \
 I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], \
 I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], \
 I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], \
 I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], \
 I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], \
 I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], \
 I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], \
 I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], \
 I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], \
 I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], \
 I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], \
 I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], \
 I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], \
 I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], \
 I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], \
 I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], \
 I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], \
 I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], \
 I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], \
 I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], \
 I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], \
 I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], \
 I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], \
 I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], \
 I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], \
 I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], \
 I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], \
 I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], \
 I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], \
 I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], \
 I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], \
 I[320] = I[321], I[321] = I[322], I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], \
 I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], \
 I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], I[342] = I[343], \
 I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], \
 I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], \
 I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], \
 I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], I[373] = I[374], I[374] = I[375], \
 I[376] = I[377], I[377] = I[378], I[378] = I[379], I[379] = I[380], I[380] = I[381], I[381] = I[382], I[382] = I[383], \
 I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], I[390] = I[391], \
 I[392] = I[393], I[393] = I[394], I[394] = I[395], I[395] = I[396], I[396] = I[397], I[397] = I[398], I[398] = I[399], \
 I[400] = I[401], I[401] = I[402], I[402] = I[403], I[403] = I[404], I[404] = I[405], I[405] = I[406], I[406] = I[407], \
 I[408] = I[409], I[409] = I[410], I[410] = I[411], I[411] = I[412], I[412] = I[413], I[413] = I[414], I[414] = I[415], \
 I[416] = I[417], I[417] = I[418], I[418] = I[419], I[419] = I[420], I[420] = I[421], I[421] = I[422], I[422] = I[423], \
 I[424] = I[425], I[425] = I[426], I[426] = I[427], I[427] = I[428], I[428] = I[429], I[429] = I[430], I[430] = I[431], \
 I[432] = I[433], I[433] = I[434], I[434] = I[435], I[435] = I[436], I[436] = I[437], I[437] = I[438], I[438] = I[439], \
 I[440] = I[441], I[441] = I[442], I[442] = I[443], I[443] = I[444], I[444] = I[445], I[445] = I[446], I[446] = I[447], \
 I[448] = I[449], I[449] = I[450], I[450] = I[451], I[451] = I[452], I[452] = I[453], I[453] = I[454], I[454] = I[455], \
 I[456] = I[457], I[457] = I[458], I[458] = I[459], I[459] = I[460], I[460] = I[461], I[461] = I[462], I[462] = I[463], \
 I[464] = I[465], I[465] = I[466], I[466] = I[467], I[467] = I[468], I[468] = I[469], I[469] = I[470], I[470] = I[471], \
 I[472] = I[473], I[473] = I[474], I[474] = I[475], I[475] = I[476], I[476] = I[477], I[477] = I[478], I[478] = I[479], \
 I[480] = I[481], I[481] = I[482], I[482] = I[483], I[483] = I[484], I[484] = I[485], I[485] = I[486], I[486] = I[487], \
 I[488] = I[489], I[489] = I[490], I[490] = I[491], I[491] = I[492], I[492] = I[493], I[493] = I[494], I[494] = I[495], \
 I[496] = I[497], I[497] = I[498], I[498] = I[499], I[499] = I[500], I[500] = I[501], I[501] = I[502], I[502] = I[503], \
 I[504] = I[505], I[505] = I[506], I[506] = I[507], I[507] = I[508], I[508] = I[509], I[509] = I[510], I[510] = I[511], \
 _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x)

#define cimg_for_in8x8x8(img,x0,y0,z0,x1,y1,z1,x,y,z,v,I) \
 cimg_for_in8((img).depth,z0,z1,z) cimg_for_in8((img).height,y0,y1,y) for (int x = (int)(x0)<0?0:(int)(x0), \
 _p3##x = x-3<0?0:x-3, \
 _p2##x = x-2<0?0:x-2, \
 _p1##x = x-1<0?0:x-1, \
 _n1##x = x+1>=(int)((img).width)?(int)((img).width)-1:x+1, \
 _n2##x = x+2>=(int)((img).width)?(int)((img).width)-1:x+2, \
 _n3##x = x+3>=(int)((img).width)?(int)((img).width)-1:x+3, \
 _n4##x = (int)( \
 (I[0] = (img)(_p3##x,_p3##y,_p3##z,v)), \
 (I[8] = (img)(_p3##x,_p2##y,_p3##z,v)), \
 (I[16] = (img)(_p3##x,_p1##y,_p3##z,v)), \
 (I[24] = (img)(_p3##x,y,_p3##z,v)), \
 (I[32] = (img)(_p3##x,_n1##y,_p3##z,v)), \
 (I[40] = (img)(_p3##x,_n2##y,_p3##z,v)), \
 (I[48] = (img)(_p3##x,_n3##y,_p3##z,v)), \
 (I[56] = (img)(_p3##x,_n4##y,_p3##z,v)), \
 (I[64] = (img)(_p3##x,_p3##y,_p2##z,v)), \
 (I[72] = (img)(_p3##x,_p2##y,_p2##z,v)), \
 (I[80] = (img)(_p3##x,_p1##y,_p2##z,v)), \
 (I[88] = (img)(_p3##x,y,_p2##z,v)), \
 (I[96] = (img)(_p3##x,_n1##y,_p2##z,v)), \
 (I[104] = (img)(_p3##x,_n2##y,_p2##z,v)), \
 (I[112] = (img)(_p3##x,_n3##y,_p2##z,v)), \
 (I[120] = (img)(_p3##x,_n4##y,_p2##z,v)), \
 (I[128] = (img)(_p3##x,_p3##y,_p1##z,v)), \
 (I[136] = (img)(_p3##x,_p2##y,_p1##z,v)), \
 (I[144] = (img)(_p3##x,_p1##y,_p1##z,v)), \
 (I[152] = (img)(_p3##x,y,_p1##z,v)), \
 (I[160] = (img)(_p3##x,_n1##y,_p1##z,v)), \
 (I[168] = (img)(_p3##x,_n2##y,_p1##z,v)), \
 (I[176] = (img)(_p3##x,_n3##y,_p1##z,v)), \
 (I[184] = (img)(_p3##x,_n4##y,_p1##z,v)), \
 (I[192] = (img)(_p3##x,_p3##y,z,v)), \
 (I[200] = (img)(_p3##x,_p2##y,z,v)), \
 (I[208] = (img)(_p3##x,_p1##y,z,v)), \
 (I[216] = (img)(_p3##x,y,z,v)), \
 (I[224] = (img)(_p3##x,_n1##y,z,v)), \
 (I[232] = (img)(_p3##x,_n2##y,z,v)), \
 (I[240] = (img)(_p3##x,_n3##y,z,v)), \
 (I[248] = (img)(_p3##x,_n4##y,z,v)), \
 (I[256] = (img)(_p3##x,_p3##y,_n1##z,v)), \
 (I[264] = (img)(_p3##x,_p2##y,_n1##z,v)), \
 (I[272] = (img)(_p3##x,_p1##y,_n1##z,v)), \
 (I[280] = (img)(_p3##x,y,_n1##z,v)), \
 (I[288] = (img)(_p3##x,_n1##y,_n1##z,v)), \
 (I[296] = (img)(_p3##x,_n2##y,_n1##z,v)), \
 (I[304] = (img)(_p3##x,_n3##y,_n1##z,v)), \
 (I[312] = (img)(_p3##x,_n4##y,_n1##z,v)), \
 (I[320] = (img)(_p3##x,_p3##y,_n2##z,v)), \
 (I[328] = (img)(_p3##x,_p2##y,_n2##z,v)), \
 (I[336] = (img)(_p3##x,_p1##y,_n2##z,v)), \
 (I[344] = (img)(_p3##x,y,_n2##z,v)), \
 (I[352] = (img)(_p3##x,_n1##y,_n2##z,v)), \
 (I[360] = (img)(_p3##x,_n2##y,_n2##z,v)), \
 (I[368] = (img)(_p3##x,_n3##y,_n2##z,v)), \
 (I[376] = (img)(_p3##x,_n4##y,_n2##z,v)), \
 (I[384] = (img)(_p3##x,_p3##y,_n3##z,v)), \
 (I[392] = (img)(_p3##x,_p2##y,_n3##z,v)), \
 (I[400] = (img)(_p3##x,_p1##y,_n3##z,v)), \
 (I[408] = (img)(_p3##x,y,_n3##z,v)), \
 (I[416] = (img)(_p3##x,_n1##y,_n3##z,v)), \
 (I[424] = (img)(_p3##x,_n2##y,_n3##z,v)), \
 (I[432] = (img)(_p3##x,_n3##y,_n3##z,v)), \
 (I[440] = (img)(_p3##x,_n4##y,_n3##z,v)), \
 (I[448] = (img)(_p3##x,_p3##y,_n4##z,v)), \
 (I[456] = (img)(_p3##x,_p2##y,_n4##z,v)), \
 (I[464] = (img)(_p3##x,_p1##y,_n4##z,v)), \
 (I[472] = (img)(_p3##x,y,_n4##z,v)), \
 (I[480] = (img)(_p3##x,_n1##y,_n4##z,v)), \
 (I[488] = (img)(_p3##x,_n2##y,_n4##z,v)), \
 (I[496] = (img)(_p3##x,_n3##y,_n4##z,v)), \
 (I[504] = (img)(_p3##x,_n4##y,_n4##z,v)), \
 (I[1] = (img)(_p2##x,_p3##y,_p3##z,v)), \
 (I[9] = (img)(_p2##x,_p2##y,_p3##z,v)), \
 (I[17] = (img)(_p2##x,_p1##y,_p3##z,v)), \
 (I[25] = (img)(_p2##x,y,_p3##z,v)), \
 (I[33] = (img)(_p2##x,_n1##y,_p3##z,v)), \
 (I[41] = (img)(_p2##x,_n2##y,_p3##z,v)), \
 (I[49] = (img)(_p2##x,_n3##y,_p3##z,v)), \
 (I[57] = (img)(_p2##x,_n4##y,_p3##z,v)), \
 (I[65] = (img)(_p2##x,_p3##y,_p2##z,v)), \
 (I[73] = (img)(_p2##x,_p2##y,_p2##z,v)), \
 (I[81] = (img)(_p2##x,_p1##y,_p2##z,v)), \
 (I[89] = (img)(_p2##x,y,_p2##z,v)), \
 (I[97] = (img)(_p2##x,_n1##y,_p2##z,v)), \
 (I[105] = (img)(_p2##x,_n2##y,_p2##z,v)), \
 (I[113] = (img)(_p2##x,_n3##y,_p2##z,v)), \
 (I[121] = (img)(_p2##x,_n4##y,_p2##z,v)), \
 (I[129] = (img)(_p2##x,_p3##y,_p1##z,v)), \
 (I[137] = (img)(_p2##x,_p2##y,_p1##z,v)), \
 (I[145] = (img)(_p2##x,_p1##y,_p1##z,v)), \
 (I[153] = (img)(_p2##x,y,_p1##z,v)), \
 (I[161] = (img)(_p2##x,_n1##y,_p1##z,v)), \
 (I[169] = (img)(_p2##x,_n2##y,_p1##z,v)), \
 (I[177] = (img)(_p2##x,_n3##y,_p1##z,v)), \
 (I[185] = (img)(_p2##x,_n4##y,_p1##z,v)), \
 (I[193] = (img)(_p2##x,_p3##y,z,v)), \
 (I[201] = (img)(_p2##x,_p2##y,z,v)), \
 (I[209] = (img)(_p2##x,_p1##y,z,v)), \
 (I[217] = (img)(_p2##x,y,z,v)), \
 (I[225] = (img)(_p2##x,_n1##y,z,v)), \
 (I[233] = (img)(_p2##x,_n2##y,z,v)), \
 (I[241] = (img)(_p2##x,_n3##y,z,v)), \
 (I[249] = (img)(_p2##x,_n4##y,z,v)), \
 (I[257] = (img)(_p2##x,_p3##y,_n1##z,v)), \
 (I[265] = (img)(_p2##x,_p2##y,_n1##z,v)), \
 (I[273] = (img)(_p2##x,_p1##y,_n1##z,v)), \
 (I[281] = (img)(_p2##x,y,_n1##z,v)), \
 (I[289] = (img)(_p2##x,_n1##y,_n1##z,v)), \
 (I[297] = (img)(_p2##x,_n2##y,_n1##z,v)), \
 (I[305] = (img)(_p2##x,_n3##y,_n1##z,v)), \
 (I[313] = (img)(_p2##x,_n4##y,_n1##z,v)), \
 (I[321] = (img)(_p2##x,_p3##y,_n2##z,v)), \
 (I[329] = (img)(_p2##x,_p2##y,_n2##z,v)), \
 (I[337] = (img)(_p2##x,_p1##y,_n2##z,v)), \
 (I[345] = (img)(_p2##x,y,_n2##z,v)), \
 (I[353] = (img)(_p2##x,_n1##y,_n2##z,v)), \
 (I[361] = (img)(_p2##x,_n2##y,_n2##z,v)), \
 (I[369] = (img)(_p2##x,_n3##y,_n2##z,v)), \
 (I[377] = (img)(_p2##x,_n4##y,_n2##z,v)), \
 (I[385] = (img)(_p2##x,_p3##y,_n3##z,v)), \
 (I[393] = (img)(_p2##x,_p2##y,_n3##z,v)), \
 (I[401] = (img)(_p2##x,_p1##y,_n3##z,v)), \
 (I[409] = (img)(_p2##x,y,_n3##z,v)), \
 (I[417] = (img)(_p2##x,_n1##y,_n3##z,v)), \
 (I[425] = (img)(_p2##x,_n2##y,_n3##z,v)), \
 (I[433] = (img)(_p2##x,_n3##y,_n3##z,v)), \
 (I[441] = (img)(_p2##x,_n4##y,_n3##z,v)), \
 (I[449] = (img)(_p2##x,_p3##y,_n4##z,v)), \
 (I[457] = (img)(_p2##x,_p2##y,_n4##z,v)), \
 (I[465] = (img)(_p2##x,_p1##y,_n4##z,v)), \
 (I[473] = (img)(_p2##x,y,_n4##z,v)), \
 (I[481] = (img)(_p2##x,_n1##y,_n4##z,v)), \
 (I[489] = (img)(_p2##x,_n2##y,_n4##z,v)), \
 (I[497] = (img)(_p2##x,_n3##y,_n4##z,v)), \
 (I[505] = (img)(_p2##x,_n4##y,_n4##z,v)), \
 (I[2] = (img)(_p1##x,_p3##y,_p3##z,v)), \
 (I[10] = (img)(_p1##x,_p2##y,_p3##z,v)), \
 (I[18] = (img)(_p1##x,_p1##y,_p3##z,v)), \
 (I[26] = (img)(_p1##x,y,_p3##z,v)), \
 (I[34] = (img)(_p1##x,_n1##y,_p3##z,v)), \
 (I[42] = (img)(_p1##x,_n2##y,_p3##z,v)), \
 (I[50] = (img)(_p1##x,_n3##y,_p3##z,v)), \
 (I[58] = (img)(_p1##x,_n4##y,_p3##z,v)), \
 (I[66] = (img)(_p1##x,_p3##y,_p2##z,v)), \
 (I[74] = (img)(_p1##x,_p2##y,_p2##z,v)), \
 (I[82] = (img)(_p1##x,_p1##y,_p2##z,v)), \
 (I[90] = (img)(_p1##x,y,_p2##z,v)), \
 (I[98] = (img)(_p1##x,_n1##y,_p2##z,v)), \
 (I[106] = (img)(_p1##x,_n2##y,_p2##z,v)), \
 (I[114] = (img)(_p1##x,_n3##y,_p2##z,v)), \
 (I[122] = (img)(_p1##x,_n4##y,_p2##z,v)), \
 (I[130] = (img)(_p1##x,_p3##y,_p1##z,v)), \
 (I[138] = (img)(_p1##x,_p2##y,_p1##z,v)), \
 (I[146] = (img)(_p1##x,_p1##y,_p1##z,v)), \
 (I[154] = (img)(_p1##x,y,_p1##z,v)), \
 (I[162] = (img)(_p1##x,_n1##y,_p1##z,v)), \
 (I[170] = (img)(_p1##x,_n2##y,_p1##z,v)), \
 (I[178] = (img)(_p1##x,_n3##y,_p1##z,v)), \
 (I[186] = (img)(_p1##x,_n4##y,_p1##z,v)), \
 (I[194] = (img)(_p1##x,_p3##y,z,v)), \
 (I[202] = (img)(_p1##x,_p2##y,z,v)), \
 (I[210] = (img)(_p1##x,_p1##y,z,v)), \
 (I[218] = (img)(_p1##x,y,z,v)), \
 (I[226] = (img)(_p1##x,_n1##y,z,v)), \
 (I[234] = (img)(_p1##x,_n2##y,z,v)), \
 (I[242] = (img)(_p1##x,_n3##y,z,v)), \
 (I[250] = (img)(_p1##x,_n4##y,z,v)), \
 (I[258] = (img)(_p1##x,_p3##y,_n1##z,v)), \
 (I[266] = (img)(_p1##x,_p2##y,_n1##z,v)), \
 (I[274] = (img)(_p1##x,_p1##y,_n1##z,v)), \
 (I[282] = (img)(_p1##x,y,_n1##z,v)), \
 (I[290] = (img)(_p1##x,_n1##y,_n1##z,v)), \
 (I[298] = (img)(_p1##x,_n2##y,_n1##z,v)), \
 (I[306] = (img)(_p1##x,_n3##y,_n1##z,v)), \
 (I[314] = (img)(_p1##x,_n4##y,_n1##z,v)), \
 (I[322] = (img)(_p1##x,_p3##y,_n2##z,v)), \
 (I[330] = (img)(_p1##x,_p2##y,_n2##z,v)), \
 (I[338] = (img)(_p1##x,_p1##y,_n2##z,v)), \
 (I[346] = (img)(_p1##x,y,_n2##z,v)), \
 (I[354] = (img)(_p1##x,_n1##y,_n2##z,v)), \
 (I[362] = (img)(_p1##x,_n2##y,_n2##z,v)), \
 (I[370] = (img)(_p1##x,_n3##y,_n2##z,v)), \
 (I[378] = (img)(_p1##x,_n4##y,_n2##z,v)), \
 (I[386] = (img)(_p1##x,_p3##y,_n3##z,v)), \
 (I[394] = (img)(_p1##x,_p2##y,_n3##z,v)), \
 (I[402] = (img)(_p1##x,_p1##y,_n3##z,v)), \
 (I[410] = (img)(_p1##x,y,_n3##z,v)), \
 (I[418] = (img)(_p1##x,_n1##y,_n3##z,v)), \
 (I[426] = (img)(_p1##x,_n2##y,_n3##z,v)), \
 (I[434] = (img)(_p1##x,_n3##y,_n3##z,v)), \
 (I[442] = (img)(_p1##x,_n4##y,_n3##z,v)), \
 (I[450] = (img)(_p1##x,_p3##y,_n4##z,v)), \
 (I[458] = (img)(_p1##x,_p2##y,_n4##z,v)), \
 (I[466] = (img)(_p1##x,_p1##y,_n4##z,v)), \
 (I[474] = (img)(_p1##x,y,_n4##z,v)), \
 (I[482] = (img)(_p1##x,_n1##y,_n4##z,v)), \
 (I[490] = (img)(_p1##x,_n2##y,_n4##z,v)), \
 (I[498] = (img)(_p1##x,_n3##y,_n4##z,v)), \
 (I[506] = (img)(_p1##x,_n4##y,_n4##z,v)), \
 (I[3] = (img)(x,_p3##y,_p3##z,v)), \
 (I[11] = (img)(x,_p2##y,_p3##z,v)), \
 (I[19] = (img)(x,_p1##y,_p3##z,v)), \
 (I[27] = (img)(x,y,_p3##z,v)), \
 (I[35] = (img)(x,_n1##y,_p3##z,v)), \
 (I[43] = (img)(x,_n2##y,_p3##z,v)), \
 (I[51] = (img)(x,_n3##y,_p3##z,v)), \
 (I[59] = (img)(x,_n4##y,_p3##z,v)), \
 (I[67] = (img)(x,_p3##y,_p2##z,v)), \
 (I[75] = (img)(x,_p2##y,_p2##z,v)), \
 (I[83] = (img)(x,_p1##y,_p2##z,v)), \
 (I[91] = (img)(x,y,_p2##z,v)), \
 (I[99] = (img)(x,_n1##y,_p2##z,v)), \
 (I[107] = (img)(x,_n2##y,_p2##z,v)), \
 (I[115] = (img)(x,_n3##y,_p2##z,v)), \
 (I[123] = (img)(x,_n4##y,_p2##z,v)), \
 (I[131] = (img)(x,_p3##y,_p1##z,v)), \
 (I[139] = (img)(x,_p2##y,_p1##z,v)), \
 (I[147] = (img)(x,_p1##y,_p1##z,v)), \
 (I[155] = (img)(x,y,_p1##z,v)), \
 (I[163] = (img)(x,_n1##y,_p1##z,v)), \
 (I[171] = (img)(x,_n2##y,_p1##z,v)), \
 (I[179] = (img)(x,_n3##y,_p1##z,v)), \
 (I[187] = (img)(x,_n4##y,_p1##z,v)), \
 (I[195] = (img)(x,_p3##y,z,v)), \
 (I[203] = (img)(x,_p2##y,z,v)), \
 (I[211] = (img)(x,_p1##y,z,v)), \
 (I[219] = (img)(x,y,z,v)), \
 (I[227] = (img)(x,_n1##y,z,v)), \
 (I[235] = (img)(x,_n2##y,z,v)), \
 (I[243] = (img)(x,_n3##y,z,v)), \
 (I[251] = (img)(x,_n4##y,z,v)), \
 (I[259] = (img)(x,_p3##y,_n1##z,v)), \
 (I[267] = (img)(x,_p2##y,_n1##z,v)), \
 (I[275] = (img)(x,_p1##y,_n1##z,v)), \
 (I[283] = (img)(x,y,_n1##z,v)), \
 (I[291] = (img)(x,_n1##y,_n1##z,v)), \
 (I[299] = (img)(x,_n2##y,_n1##z,v)), \
 (I[307] = (img)(x,_n3##y,_n1##z,v)), \
 (I[315] = (img)(x,_n4##y,_n1##z,v)), \
 (I[323] = (img)(x,_p3##y,_n2##z,v)), \
 (I[331] = (img)(x,_p2##y,_n2##z,v)), \
 (I[339] = (img)(x,_p1##y,_n2##z,v)), \
 (I[347] = (img)(x,y,_n2##z,v)), \
 (I[355] = (img)(x,_n1##y,_n2##z,v)), \
 (I[363] = (img)(x,_n2##y,_n2##z,v)), \
 (I[371] = (img)(x,_n3##y,_n2##z,v)), \
 (I[379] = (img)(x,_n4##y,_n2##z,v)), \
 (I[387] = (img)(x,_p3##y,_n3##z,v)), \
 (I[395] = (img)(x,_p2##y,_n3##z,v)), \
 (I[403] = (img)(x,_p1##y,_n3##z,v)), \
 (I[411] = (img)(x,y,_n3##z,v)), \
 (I[419] = (img)(x,_n1##y,_n3##z,v)), \
 (I[427] = (img)(x,_n2##y,_n3##z,v)), \
 (I[435] = (img)(x,_n3##y,_n3##z,v)), \
 (I[443] = (img)(x,_n4##y,_n3##z,v)), \
 (I[451] = (img)(x,_p3##y,_n4##z,v)), \
 (I[459] = (img)(x,_p2##y,_n4##z,v)), \
 (I[467] = (img)(x,_p1##y,_n4##z,v)), \
 (I[475] = (img)(x,y,_n4##z,v)), \
 (I[483] = (img)(x,_n1##y,_n4##z,v)), \
 (I[491] = (img)(x,_n2##y,_n4##z,v)), \
 (I[499] = (img)(x,_n3##y,_n4##z,v)), \
 (I[507] = (img)(x,_n4##y,_n4##z,v)), \
 (I[4] = (img)(_n1##x,_p3##y,_p3##z,v)), \
 (I[12] = (img)(_n1##x,_p2##y,_p3##z,v)), \
 (I[20] = (img)(_n1##x,_p1##y,_p3##z,v)), \
 (I[28] = (img)(_n1##x,y,_p3##z,v)), \
 (I[36] = (img)(_n1##x,_n1##y,_p3##z,v)), \
 (I[44] = (img)(_n1##x,_n2##y,_p3##z,v)), \
 (I[52] = (img)(_n1##x,_n3##y,_p3##z,v)), \
 (I[60] = (img)(_n1##x,_n4##y,_p3##z,v)), \
 (I[68] = (img)(_n1##x,_p3##y,_p2##z,v)), \
 (I[76] = (img)(_n1##x,_p2##y,_p2##z,v)), \
 (I[84] = (img)(_n1##x,_p1##y,_p2##z,v)), \
 (I[92] = (img)(_n1##x,y,_p2##z,v)), \
 (I[100] = (img)(_n1##x,_n1##y,_p2##z,v)), \
 (I[108] = (img)(_n1##x,_n2##y,_p2##z,v)), \
 (I[116] = (img)(_n1##x,_n3##y,_p2##z,v)), \
 (I[124] = (img)(_n1##x,_n4##y,_p2##z,v)), \
 (I[132] = (img)(_n1##x,_p3##y,_p1##z,v)), \
 (I[140] = (img)(_n1##x,_p2##y,_p1##z,v)), \
 (I[148] = (img)(_n1##x,_p1##y,_p1##z,v)), \
 (I[156] = (img)(_n1##x,y,_p1##z,v)), \
 (I[164] = (img)(_n1##x,_n1##y,_p1##z,v)), \
 (I[172] = (img)(_n1##x,_n2##y,_p1##z,v)), \
 (I[180] = (img)(_n1##x,_n3##y,_p1##z,v)), \
 (I[188] = (img)(_n1##x,_n4##y,_p1##z,v)), \
 (I[196] = (img)(_n1##x,_p3##y,z,v)), \
 (I[204] = (img)(_n1##x,_p2##y,z,v)), \
 (I[212] = (img)(_n1##x,_p1##y,z,v)), \
 (I[220] = (img)(_n1##x,y,z,v)), \
 (I[228] = (img)(_n1##x,_n1##y,z,v)), \
 (I[236] = (img)(_n1##x,_n2##y,z,v)), \
 (I[244] = (img)(_n1##x,_n3##y,z,v)), \
 (I[252] = (img)(_n1##x,_n4##y,z,v)), \
 (I[260] = (img)(_n1##x,_p3##y,_n1##z,v)), \
 (I[268] = (img)(_n1##x,_p2##y,_n1##z,v)), \
 (I[276] = (img)(_n1##x,_p1##y,_n1##z,v)), \
 (I[284] = (img)(_n1##x,y,_n1##z,v)), \
 (I[292] = (img)(_n1##x,_n1##y,_n1##z,v)), \
 (I[300] = (img)(_n1##x,_n2##y,_n1##z,v)), \
 (I[308] = (img)(_n1##x,_n3##y,_n1##z,v)), \
 (I[316] = (img)(_n1##x,_n4##y,_n1##z,v)), \
 (I[324] = (img)(_n1##x,_p3##y,_n2##z,v)), \
 (I[332] = (img)(_n1##x,_p2##y,_n2##z,v)), \
 (I[340] = (img)(_n1##x,_p1##y,_n2##z,v)), \
 (I[348] = (img)(_n1##x,y,_n2##z,v)), \
 (I[356] = (img)(_n1##x,_n1##y,_n2##z,v)), \
 (I[364] = (img)(_n1##x,_n2##y,_n2##z,v)), \
 (I[372] = (img)(_n1##x,_n3##y,_n2##z,v)), \
 (I[380] = (img)(_n1##x,_n4##y,_n2##z,v)), \
 (I[388] = (img)(_n1##x,_p3##y,_n3##z,v)), \
 (I[396] = (img)(_n1##x,_p2##y,_n3##z,v)), \
 (I[404] = (img)(_n1##x,_p1##y,_n3##z,v)), \
 (I[412] = (img)(_n1##x,y,_n3##z,v)), \
 (I[420] = (img)(_n1##x,_n1##y,_n3##z,v)), \
 (I[428] = (img)(_n1##x,_n2##y,_n3##z,v)), \
 (I[436] = (img)(_n1##x,_n3##y,_n3##z,v)), \
 (I[444] = (img)(_n1##x,_n4##y,_n3##z,v)), \
 (I[452] = (img)(_n1##x,_p3##y,_n4##z,v)), \
 (I[460] = (img)(_n1##x,_p2##y,_n4##z,v)), \
 (I[468] = (img)(_n1##x,_p1##y,_n4##z,v)), \
 (I[476] = (img)(_n1##x,y,_n4##z,v)), \
 (I[484] = (img)(_n1##x,_n1##y,_n4##z,v)), \
 (I[492] = (img)(_n1##x,_n2##y,_n4##z,v)), \
 (I[500] = (img)(_n1##x,_n3##y,_n4##z,v)), \
 (I[508] = (img)(_n1##x,_n4##y,_n4##z,v)), \
 (I[5] = (img)(_n2##x,_p3##y,_p3##z,v)), \
 (I[13] = (img)(_n2##x,_p2##y,_p3##z,v)), \
 (I[21] = (img)(_n2##x,_p1##y,_p3##z,v)), \
 (I[29] = (img)(_n2##x,y,_p3##z,v)), \
 (I[37] = (img)(_n2##x,_n1##y,_p3##z,v)), \
 (I[45] = (img)(_n2##x,_n2##y,_p3##z,v)), \
 (I[53] = (img)(_n2##x,_n3##y,_p3##z,v)), \
 (I[61] = (img)(_n2##x,_n4##y,_p3##z,v)), \
 (I[69] = (img)(_n2##x,_p3##y,_p2##z,v)), \
 (I[77] = (img)(_n2##x,_p2##y,_p2##z,v)), \
 (I[85] = (img)(_n2##x,_p1##y,_p2##z,v)), \
 (I[93] = (img)(_n2##x,y,_p2##z,v)), \
 (I[101] = (img)(_n2##x,_n1##y,_p2##z,v)), \
 (I[109] = (img)(_n2##x,_n2##y,_p2##z,v)), \
 (I[117] = (img)(_n2##x,_n3##y,_p2##z,v)), \
 (I[125] = (img)(_n2##x,_n4##y,_p2##z,v)), \
 (I[133] = (img)(_n2##x,_p3##y,_p1##z,v)), \
 (I[141] = (img)(_n2##x,_p2##y,_p1##z,v)), \
 (I[149] = (img)(_n2##x,_p1##y,_p1##z,v)), \
 (I[157] = (img)(_n2##x,y,_p1##z,v)), \
 (I[165] = (img)(_n2##x,_n1##y,_p1##z,v)), \
 (I[173] = (img)(_n2##x,_n2##y,_p1##z,v)), \
 (I[181] = (img)(_n2##x,_n3##y,_p1##z,v)), \
 (I[189] = (img)(_n2##x,_n4##y,_p1##z,v)), \
 (I[197] = (img)(_n2##x,_p3##y,z,v)), \
 (I[205] = (img)(_n2##x,_p2##y,z,v)), \
 (I[213] = (img)(_n2##x,_p1##y,z,v)), \
 (I[221] = (img)(_n2##x,y,z,v)), \
 (I[229] = (img)(_n2##x,_n1##y,z,v)), \
 (I[237] = (img)(_n2##x,_n2##y,z,v)), \
 (I[245] = (img)(_n2##x,_n3##y,z,v)), \
 (I[253] = (img)(_n2##x,_n4##y,z,v)), \
 (I[261] = (img)(_n2##x,_p3##y,_n1##z,v)), \
 (I[269] = (img)(_n2##x,_p2##y,_n1##z,v)), \
 (I[277] = (img)(_n2##x,_p1##y,_n1##z,v)), \
 (I[285] = (img)(_n2##x,y,_n1##z,v)), \
 (I[293] = (img)(_n2##x,_n1##y,_n1##z,v)), \
 (I[301] = (img)(_n2##x,_n2##y,_n1##z,v)), \
 (I[309] = (img)(_n2##x,_n3##y,_n1##z,v)), \
 (I[317] = (img)(_n2##x,_n4##y,_n1##z,v)), \
 (I[325] = (img)(_n2##x,_p3##y,_n2##z,v)), \
 (I[333] = (img)(_n2##x,_p2##y,_n2##z,v)), \
 (I[341] = (img)(_n2##x,_p1##y,_n2##z,v)), \
 (I[349] = (img)(_n2##x,y,_n2##z,v)), \
 (I[357] = (img)(_n2##x,_n1##y,_n2##z,v)), \
 (I[365] = (img)(_n2##x,_n2##y,_n2##z,v)), \
 (I[373] = (img)(_n2##x,_n3##y,_n2##z,v)), \
 (I[381] = (img)(_n2##x,_n4##y,_n2##z,v)), \
 (I[389] = (img)(_n2##x,_p3##y,_n3##z,v)), \
 (I[397] = (img)(_n2##x,_p2##y,_n3##z,v)), \
 (I[405] = (img)(_n2##x,_p1##y,_n3##z,v)), \
 (I[413] = (img)(_n2##x,y,_n3##z,v)), \
 (I[421] = (img)(_n2##x,_n1##y,_n3##z,v)), \
 (I[429] = (img)(_n2##x,_n2##y,_n3##z,v)), \
 (I[437] = (img)(_n2##x,_n3##y,_n3##z,v)), \
 (I[445] = (img)(_n2##x,_n4##y,_n3##z,v)), \
 (I[453] = (img)(_n2##x,_p3##y,_n4##z,v)), \
 (I[461] = (img)(_n2##x,_p2##y,_n4##z,v)), \
 (I[469] = (img)(_n2##x,_p1##y,_n4##z,v)), \
 (I[477] = (img)(_n2##x,y,_n4##z,v)), \
 (I[485] = (img)(_n2##x,_n1##y,_n4##z,v)), \
 (I[493] = (img)(_n2##x,_n2##y,_n4##z,v)), \
 (I[501] = (img)(_n2##x,_n3##y,_n4##z,v)), \
 (I[509] = (img)(_n2##x,_n4##y,_n4##z,v)), \
 (I[6] = (img)(_n3##x,_p3##y,_p3##z,v)), \
 (I[14] = (img)(_n3##x,_p2##y,_p3##z,v)), \
 (I[22] = (img)(_n3##x,_p1##y,_p3##z,v)), \
 (I[30] = (img)(_n3##x,y,_p3##z,v)), \
 (I[38] = (img)(_n3##x,_n1##y,_p3##z,v)), \
 (I[46] = (img)(_n3##x,_n2##y,_p3##z,v)), \
 (I[54] = (img)(_n3##x,_n3##y,_p3##z,v)), \
 (I[62] = (img)(_n3##x,_n4##y,_p3##z,v)), \
 (I[70] = (img)(_n3##x,_p3##y,_p2##z,v)), \
 (I[78] = (img)(_n3##x,_p2##y,_p2##z,v)), \
 (I[86] = (img)(_n3##x,_p1##y,_p2##z,v)), \
 (I[94] = (img)(_n3##x,y,_p2##z,v)), \
 (I[102] = (img)(_n3##x,_n1##y,_p2##z,v)), \
 (I[110] = (img)(_n3##x,_n2##y,_p2##z,v)), \
 (I[118] = (img)(_n3##x,_n3##y,_p2##z,v)), \
 (I[126] = (img)(_n3##x,_n4##y,_p2##z,v)), \
 (I[134] = (img)(_n3##x,_p3##y,_p1##z,v)), \
 (I[142] = (img)(_n3##x,_p2##y,_p1##z,v)), \
 (I[150] = (img)(_n3##x,_p1##y,_p1##z,v)), \
 (I[158] = (img)(_n3##x,y,_p1##z,v)), \
 (I[166] = (img)(_n3##x,_n1##y,_p1##z,v)), \
 (I[174] = (img)(_n3##x,_n2##y,_p1##z,v)), \
 (I[182] = (img)(_n3##x,_n3##y,_p1##z,v)), \
 (I[190] = (img)(_n3##x,_n4##y,_p1##z,v)), \
 (I[198] = (img)(_n3##x,_p3##y,z,v)), \
 (I[206] = (img)(_n3##x,_p2##y,z,v)), \
 (I[214] = (img)(_n3##x,_p1##y,z,v)), \
 (I[222] = (img)(_n3##x,y,z,v)), \
 (I[230] = (img)(_n3##x,_n1##y,z,v)), \
 (I[238] = (img)(_n3##x,_n2##y,z,v)), \
 (I[246] = (img)(_n3##x,_n3##y,z,v)), \
 (I[254] = (img)(_n3##x,_n4##y,z,v)), \
 (I[262] = (img)(_n3##x,_p3##y,_n1##z,v)), \
 (I[270] = (img)(_n3##x,_p2##y,_n1##z,v)), \
 (I[278] = (img)(_n3##x,_p1##y,_n1##z,v)), \
 (I[286] = (img)(_n3##x,y,_n1##z,v)), \
 (I[294] = (img)(_n3##x,_n1##y,_n1##z,v)), \
 (I[302] = (img)(_n3##x,_n2##y,_n1##z,v)), \
 (I[310] = (img)(_n3##x,_n3##y,_n1##z,v)), \
 (I[318] = (img)(_n3##x,_n4##y,_n1##z,v)), \
 (I[326] = (img)(_n3##x,_p3##y,_n2##z,v)), \
 (I[334] = (img)(_n3##x,_p2##y,_n2##z,v)), \
 (I[342] = (img)(_n3##x,_p1##y,_n2##z,v)), \
 (I[350] = (img)(_n3##x,y,_n2##z,v)), \
 (I[358] = (img)(_n3##x,_n1##y,_n2##z,v)), \
 (I[366] = (img)(_n3##x,_n2##y,_n2##z,v)), \
 (I[374] = (img)(_n3##x,_n3##y,_n2##z,v)), \
 (I[382] = (img)(_n3##x,_n4##y,_n2##z,v)), \
 (I[390] = (img)(_n3##x,_p3##y,_n3##z,v)), \
 (I[398] = (img)(_n3##x,_p2##y,_n3##z,v)), \
 (I[406] = (img)(_n3##x,_p1##y,_n3##z,v)), \
 (I[414] = (img)(_n3##x,y,_n3##z,v)), \
 (I[422] = (img)(_n3##x,_n1##y,_n3##z,v)), \
 (I[430] = (img)(_n3##x,_n2##y,_n3##z,v)), \
 (I[438] = (img)(_n3##x,_n3##y,_n3##z,v)), \
 (I[446] = (img)(_n3##x,_n4##y,_n3##z,v)), \
 (I[454] = (img)(_n3##x,_p3##y,_n4##z,v)), \
 (I[462] = (img)(_n3##x,_p2##y,_n4##z,v)), \
 (I[470] = (img)(_n3##x,_p1##y,_n4##z,v)), \
 (I[478] = (img)(_n3##x,y,_n4##z,v)), \
 (I[486] = (img)(_n3##x,_n1##y,_n4##z,v)), \
 (I[494] = (img)(_n3##x,_n2##y,_n4##z,v)), \
 (I[502] = (img)(_n3##x,_n3##y,_n4##z,v)), \
 (I[510] = (img)(_n3##x,_n4##y,_n4##z,v)), \
 x+4>=(int)((img).width)?(int)((img).width)-1:x+4); \
 x<=(int)(x1) && ((_n4##x<(int)((img).width) && ( \
 (I[7] = (img)(_n4##x,_p3##y,_p3##z,v)), \
 (I[15] = (img)(_n4##x,_p2##y,_p3##z,v)), \
 (I[23] = (img)(_n4##x,_p1##y,_p3##z,v)), \
 (I[31] = (img)(_n4##x,y,_p3##z,v)), \
 (I[39] = (img)(_n4##x,_n1##y,_p3##z,v)), \
 (I[47] = (img)(_n4##x,_n2##y,_p3##z,v)), \
 (I[55] = (img)(_n4##x,_n3##y,_p3##z,v)), \
 (I[63] = (img)(_n4##x,_n4##y,_p3##z,v)), \
 (I[71] = (img)(_n4##x,_p3##y,_p2##z,v)), \
 (I[79] = (img)(_n4##x,_p2##y,_p2##z,v)), \
 (I[87] = (img)(_n4##x,_p1##y,_p2##z,v)), \
 (I[95] = (img)(_n4##x,y,_p2##z,v)), \
 (I[103] = (img)(_n4##x,_n1##y,_p2##z,v)), \
 (I[111] = (img)(_n4##x,_n2##y,_p2##z,v)), \
 (I[119] = (img)(_n4##x,_n3##y,_p2##z,v)), \
 (I[127] = (img)(_n4##x,_n4##y,_p2##z,v)), \
 (I[135] = (img)(_n4##x,_p3##y,_p1##z,v)), \
 (I[143] = (img)(_n4##x,_p2##y,_p1##z,v)), \
 (I[151] = (img)(_n4##x,_p1##y,_p1##z,v)), \
 (I[159] = (img)(_n4##x,y,_p1##z,v)), \
 (I[167] = (img)(_n4##x,_n1##y,_p1##z,v)), \
 (I[175] = (img)(_n4##x,_n2##y,_p1##z,v)), \
 (I[183] = (img)(_n4##x,_n3##y,_p1##z,v)), \
 (I[191] = (img)(_n4##x,_n4##y,_p1##z,v)), \
 (I[199] = (img)(_n4##x,_p3##y,z,v)), \
 (I[207] = (img)(_n4##x,_p2##y,z,v)), \
 (I[215] = (img)(_n4##x,_p1##y,z,v)), \
 (I[223] = (img)(_n4##x,y,z,v)), \
 (I[231] = (img)(_n4##x,_n1##y,z,v)), \
 (I[239] = (img)(_n4##x,_n2##y,z,v)), \
 (I[247] = (img)(_n4##x,_n3##y,z,v)), \
 (I[255] = (img)(_n4##x,_n4##y,z,v)), \
 (I[263] = (img)(_n4##x,_p3##y,_n1##z,v)), \
 (I[271] = (img)(_n4##x,_p2##y,_n1##z,v)), \
 (I[279] = (img)(_n4##x,_p1##y,_n1##z,v)), \
 (I[287] = (img)(_n4##x,y,_n1##z,v)), \
 (I[295] = (img)(_n4##x,_n1##y,_n1##z,v)), \
 (I[303] = (img)(_n4##x,_n2##y,_n1##z,v)), \
 (I[311] = (img)(_n4##x,_n3##y,_n1##z,v)), \
 (I[319] = (img)(_n4##x,_n4##y,_n1##z,v)), \
 (I[327] = (img)(_n4##x,_p3##y,_n2##z,v)), \
 (I[335] = (img)(_n4##x,_p2##y,_n2##z,v)), \
 (I[343] = (img)(_n4##x,_p1##y,_n2##z,v)), \
 (I[351] = (img)(_n4##x,y,_n2##z,v)), \
 (I[359] = (img)(_n4##x,_n1##y,_n2##z,v)), \
 (I[367] = (img)(_n4##x,_n2##y,_n2##z,v)), \
 (I[375] = (img)(_n4##x,_n3##y,_n2##z,v)), \
 (I[383] = (img)(_n4##x,_n4##y,_n2##z,v)), \
 (I[391] = (img)(_n4##x,_p3##y,_n3##z,v)), \
 (I[399] = (img)(_n4##x,_p2##y,_n3##z,v)), \
 (I[407] = (img)(_n4##x,_p1##y,_n3##z,v)), \
 (I[415] = (img)(_n4##x,y,_n3##z,v)), \
 (I[423] = (img)(_n4##x,_n1##y,_n3##z,v)), \
 (I[431] = (img)(_n4##x,_n2##y,_n3##z,v)), \
 (I[439] = (img)(_n4##x,_n3##y,_n3##z,v)), \
 (I[447] = (img)(_n4##x,_n4##y,_n3##z,v)), \
 (I[455] = (img)(_n4##x,_p3##y,_n4##z,v)), \
 (I[463] = (img)(_n4##x,_p2##y,_n4##z,v)), \
 (I[471] = (img)(_n4##x,_p1##y,_n4##z,v)), \
 (I[479] = (img)(_n4##x,y,_n4##z,v)), \
 (I[487] = (img)(_n4##x,_n1##y,_n4##z,v)), \
 (I[495] = (img)(_n4##x,_n2##y,_n4##z,v)), \
 (I[503] = (img)(_n4##x,_n3##y,_n4##z,v)), \
 (I[511] = (img)(_n4##x,_n4##y,_n4##z,v)),1)) || \
 _n3##x==--_n4##x || _n2##x==--_n3##x || _n1##x==--_n2##x || x==(_n4##x = _n3##x = _n2##x = --_n1##x)); \
 I[0] = I[1], I[1] = I[2], I[2] = I[3], I[3] = I[4], I[4] = I[5], I[5] = I[6], I[6] = I[7], \
 I[8] = I[9], I[9] = I[10], I[10] = I[11], I[11] = I[12], I[12] = I[13], I[13] = I[14], I[14] = I[15], \
 I[16] = I[17], I[17] = I[18], I[18] = I[19], I[19] = I[20], I[20] = I[21], I[21] = I[22], I[22] = I[23], \
 I[24] = I[25], I[25] = I[26], I[26] = I[27], I[27] = I[28], I[28] = I[29], I[29] = I[30], I[30] = I[31], \
 I[32] = I[33], I[33] = I[34], I[34] = I[35], I[35] = I[36], I[36] = I[37], I[37] = I[38], I[38] = I[39], \
 I[40] = I[41], I[41] = I[42], I[42] = I[43], I[43] = I[44], I[44] = I[45], I[45] = I[46], I[46] = I[47], \
 I[48] = I[49], I[49] = I[50], I[50] = I[51], I[51] = I[52], I[52] = I[53], I[53] = I[54], I[54] = I[55], \
 I[56] = I[57], I[57] = I[58], I[58] = I[59], I[59] = I[60], I[60] = I[61], I[61] = I[62], I[62] = I[63], \
 I[64] = I[65], I[65] = I[66], I[66] = I[67], I[67] = I[68], I[68] = I[69], I[69] = I[70], I[70] = I[71], \
 I[72] = I[73], I[73] = I[74], I[74] = I[75], I[75] = I[76], I[76] = I[77], I[77] = I[78], I[78] = I[79], \
 I[80] = I[81], I[81] = I[82], I[82] = I[83], I[83] = I[84], I[84] = I[85], I[85] = I[86], I[86] = I[87], \
 I[88] = I[89], I[89] = I[90], I[90] = I[91], I[91] = I[92], I[92] = I[93], I[93] = I[94], I[94] = I[95], \
 I[96] = I[97], I[97] = I[98], I[98] = I[99], I[99] = I[100], I[100] = I[101], I[101] = I[102], I[102] = I[103], \
 I[104] = I[105], I[105] = I[106], I[106] = I[107], I[107] = I[108], I[108] = I[109], I[109] = I[110], I[110] = I[111], \
 I[112] = I[113], I[113] = I[114], I[114] = I[115], I[115] = I[116], I[116] = I[117], I[117] = I[118], I[118] = I[119], \
 I[120] = I[121], I[121] = I[122], I[122] = I[123], I[123] = I[124], I[124] = I[125], I[125] = I[126], I[126] = I[127], \
 I[128] = I[129], I[129] = I[130], I[130] = I[131], I[131] = I[132], I[132] = I[133], I[133] = I[134], I[134] = I[135], \
 I[136] = I[137], I[137] = I[138], I[138] = I[139], I[139] = I[140], I[140] = I[141], I[141] = I[142], I[142] = I[143], \
 I[144] = I[145], I[145] = I[146], I[146] = I[147], I[147] = I[148], I[148] = I[149], I[149] = I[150], I[150] = I[151], \
 I[152] = I[153], I[153] = I[154], I[154] = I[155], I[155] = I[156], I[156] = I[157], I[157] = I[158], I[158] = I[159], \
 I[160] = I[161], I[161] = I[162], I[162] = I[163], I[163] = I[164], I[164] = I[165], I[165] = I[166], I[166] = I[167], \
 I[168] = I[169], I[169] = I[170], I[170] = I[171], I[171] = I[172], I[172] = I[173], I[173] = I[174], I[174] = I[175], \
 I[176] = I[177], I[177] = I[178], I[178] = I[179], I[179] = I[180], I[180] = I[181], I[181] = I[182], I[182] = I[183], \
 I[184] = I[185], I[185] = I[186], I[186] = I[187], I[187] = I[188], I[188] = I[189], I[189] = I[190], I[190] = I[191], \
 I[192] = I[193], I[193] = I[194], I[194] = I[195], I[195] = I[196], I[196] = I[197], I[197] = I[198], I[198] = I[199], \
 I[200] = I[201], I[201] = I[202], I[202] = I[203], I[203] = I[204], I[204] = I[205], I[205] = I[206], I[206] = I[207], \
 I[208] = I[209], I[209] = I[210], I[210] = I[211], I[211] = I[212], I[212] = I[213], I[213] = I[214], I[214] = I[215], \
 I[216] = I[217], I[217] = I[218], I[218] = I[219], I[219] = I[220], I[220] = I[221], I[221] = I[222], I[222] = I[223], \
 I[224] = I[225], I[225] = I[226], I[226] = I[227], I[227] = I[228], I[228] = I[229], I[229] = I[230], I[230] = I[231], \
 I[232] = I[233], I[233] = I[234], I[234] = I[235], I[235] = I[236], I[236] = I[237], I[237] = I[238], I[238] = I[239], \
 I[240] = I[241], I[241] = I[242], I[242] = I[243], I[243] = I[244], I[244] = I[245], I[245] = I[246], I[246] = I[247], \
 I[248] = I[249], I[249] = I[250], I[250] = I[251], I[251] = I[252], I[252] = I[253], I[253] = I[254], I[254] = I[255], \
 I[256] = I[257], I[257] = I[258], I[258] = I[259], I[259] = I[260], I[260] = I[261], I[261] = I[262], I[262] = I[263], \
 I[264] = I[265], I[265] = I[266], I[266] = I[267], I[267] = I[268], I[268] = I[269], I[269] = I[270], I[270] = I[271], \
 I[272] = I[273], I[273] = I[274], I[274] = I[275], I[275] = I[276], I[276] = I[277], I[277] = I[278], I[278] = I[279], \
 I[280] = I[281], I[281] = I[282], I[282] = I[283], I[283] = I[284], I[284] = I[285], I[285] = I[286], I[286] = I[287], \
 I[288] = I[289], I[289] = I[290], I[290] = I[291], I[291] = I[292], I[292] = I[293], I[293] = I[294], I[294] = I[295], \
 I[296] = I[297], I[297] = I[298], I[298] = I[299], I[299] = I[300], I[300] = I[301], I[301] = I[302], I[302] = I[303], \
 I[304] = I[305], I[305] = I[306], I[306] = I[307], I[307] = I[308], I[308] = I[309], I[309] = I[310], I[310] = I[311], \
 I[312] = I[313], I[313] = I[314], I[314] = I[315], I[315] = I[316], I[316] = I[317], I[317] = I[318], I[318] = I[319], \
 I[320] = I[321], I[321] = I[322], I[322] = I[323], I[323] = I[324], I[324] = I[325], I[325] = I[326], I[326] = I[327], \
 I[328] = I[329], I[329] = I[330], I[330] = I[331], I[331] = I[332], I[332] = I[333], I[333] = I[334], I[334] = I[335], \
 I[336] = I[337], I[337] = I[338], I[338] = I[339], I[339] = I[340], I[340] = I[341], I[341] = I[342], I[342] = I[343], \
 I[344] = I[345], I[345] = I[346], I[346] = I[347], I[347] = I[348], I[348] = I[349], I[349] = I[350], I[350] = I[351], \
 I[352] = I[353], I[353] = I[354], I[354] = I[355], I[355] = I[356], I[356] = I[357], I[357] = I[358], I[358] = I[359], \
 I[360] = I[361], I[361] = I[362], I[362] = I[363], I[363] = I[364], I[364] = I[365], I[365] = I[366], I[366] = I[367], \
 I[368] = I[369], I[369] = I[370], I[370] = I[371], I[371] = I[372], I[372] = I[373], I[373] = I[374], I[374] = I[375], \
 I[376] = I[377], I[377] = I[378], I[378] = I[379], I[379] = I[380], I[380] = I[381], I[381] = I[382], I[382] = I[383], \
 I[384] = I[385], I[385] = I[386], I[386] = I[387], I[387] = I[388], I[388] = I[389], I[389] = I[390], I[390] = I[391], \
 I[392] = I[393], I[393] = I[394], I[394] = I[395], I[395] = I[396], I[396] = I[397], I[397] = I[398], I[398] = I[399], \
 I[400] = I[401], I[401] = I[402], I[402] = I[403], I[403] = I[404], I[404] = I[405], I[405] = I[406], I[406] = I[407], \
 I[408] = I[409], I[409] = I[410], I[410] = I[411], I[411] = I[412], I[412] = I[413], I[413] = I[414], I[414] = I[415], \
 I[416] = I[417], I[417] = I[418], I[418] = I[419], I[419] = I[420], I[420] = I[421], I[421] = I[422], I[422] = I[423], \
 I[424] = I[425], I[425] = I[426], I[426] = I[427], I[427] = I[428], I[428] = I[429], I[429] = I[430], I[430] = I[431], \
 I[432] = I[433], I[433] = I[434], I[434] = I[435], I[435] = I[436], I[436] = I[437], I[437] = I[438], I[438] = I[439], \
 I[440] = I[441], I[441] = I[442], I[442] = I[443], I[443] = I[444], I[444] = I[445], I[445] = I[446], I[446] = I[447], \
 I[448] = I[449], I[449] = I[450], I[450] = I[451], I[451] = I[452], I[452] = I[453], I[453] = I[454], I[454] = I[455], \
 I[456] = I[457], I[457] = I[458], I[458] = I[459], I[459] = I[460], I[460] = I[461], I[461] = I[462], I[462] = I[463], \
 I[464] = I[465], I[465] = I[466], I[466] = I[467], I[467] = I[468], I[468] = I[469], I[469] = I[470], I[470] = I[471], \
 I[472] = I[473], I[473] = I[474], I[474] = I[475], I[475] = I[476], I[476] = I[477], I[477] = I[478], I[478] = I[479], \
 I[480] = I[481], I[481] = I[482], I[482] = I[483], I[483] = I[484], I[484] = I[485], I[485] = I[486], I[486] = I[487], \
 I[488] = I[489], I[489] = I[490], I[490] = I[491], I[491] = I[492], I[492] = I[493], I[493] = I[494], I[494] = I[495], \
 I[496] = I[497], I[497] = I[498], I[498] = I[499], I[499] = I[500], I[500] = I[501], I[501] = I[502], I[502] = I[503], \
 I[504] = I[505], I[505] = I[506], I[506] = I[507], I[507] = I[508], I[508] = I[509], I[509] = I[510], I[510] = I[511], \
 _p3##x = _p2##x, _p2##x = _p1##x, _p1##x = x++, ++_n1##x, ++_n2##x, ++_n3##x, ++_n4##x)

#define cimg_get8x8x8(img,x,y,z,v,I) \
 I[0] = (img)(_p3##x,_p3##y,_p3##z,v), I[1] = (img)(_p2##x,_p3##y,_p3##z,v), I[2] = (img)(_p1##x,_p3##y,_p3##z,v), I[3] = (img)(x,_p3##y,_p3##z,v), I[4] = (img)(_n1##x,_p3##y,_p3##z,v), I[5] = (img)(_n2##x,_p3##y,_p3##z,v), I[6] = (img)(_n3##x,_p3##y,_p3##z,v), I[7] = (img)(_n4##x,_p3##y,_p3##z,v), \
 I[8] = (img)(_p3##x,_p2##y,_p3##z,v), I[9] = (img)(_p2##x,_p2##y,_p3##z,v), I[10] = (img)(_p1##x,_p2##y,_p3##z,v), I[11] = (img)(x,_p2##y,_p3##z,v), I[12] = (img)(_n1##x,_p2##y,_p3##z,v), I[13] = (img)(_n2##x,_p2##y,_p3##z,v), I[14] = (img)(_n3##x,_p2##y,_p3##z,v), I[15] = (img)(_n4##x,_p2##y,_p3##z,v), \
 I[16] = (img)(_p3##x,_p1##y,_p3##z,v), I[17] = (img)(_p2##x,_p1##y,_p3##z,v), I[18] = (img)(_p1##x,_p1##y,_p3##z,v), I[19] = (img)(x,_p1##y,_p3##z,v), I[20] = (img)(_n1##x,_p1##y,_p3##z,v), I[21] = (img)(_n2##x,_p1##y,_p3##z,v), I[22] = (img)(_n3##x,_p1##y,_p3##z,v), I[23] = (img)(_n4##x,_p1##y,_p3##z,v), \
 I[24] = (img)(_p3##x,y,_p3##z,v), I[25] = (img)(_p2##x,y,_p3##z,v), I[26] = (img)(_p1##x,y,_p3##z,v), I[27] = (img)(x,y,_p3##z,v), I[28] = (img)(_n1##x,y,_p3##z,v), I[29] = (img)(_n2##x,y,_p3##z,v), I[30] = (img)(_n3##x,y,_p3##z,v), I[31] = (img)(_n4##x,y,_p3##z,v), \
 I[32] = (img)(_p3##x,_n1##y,_p3##z,v), I[33] = (img)(_p2##x,_n1##y,_p3##z,v), I[34] = (img)(_p1##x,_n1##y,_p3##z,v), I[35] = (img)(x,_n1##y,_p3##z,v), I[36] = (img)(_n1##x,_n1##y,_p3##z,v), I[37] = (img)(_n2##x,_n1##y,_p3##z,v), I[38] = (img)(_n3##x,_n1##y,_p3##z,v), I[39] = (img)(_n4##x,_n1##y,_p3##z,v), \
 I[40] = (img)(_p3##x,_n2##y,_p3##z,v), I[41] = (img)(_p2##x,_n2##y,_p3##z,v), I[42] = (img)(_p1##x,_n2##y,_p3##z,v), I[43] = (img)(x,_n2##y,_p3##z,v), I[44] = (img)(_n1##x,_n2##y,_p3##z,v), I[45] = (img)(_n2##x,_n2##y,_p3##z,v), I[46] = (img)(_n3##x,_n2##y,_p3##z,v), I[47] = (img)(_n4##x,_n2##y,_p3##z,v), \
 I[48] = (img)(_p3##x,_n3##y,_p3##z,v), I[49] = (img)(_p2##x,_n3##y,_p3##z,v), I[50] = (img)(_p1##x,_n3##y,_p3##z,v), I[51] = (img)(x,_n3##y,_p3##z,v), I[52] = (img)(_n1##x,_n3##y,_p3##z,v), I[53] = (img)(_n2##x,_n3##y,_p3##z,v), I[54] = (img)(_n3##x,_n3##y,_p3##z,v), I[55] = (img)(_n4##x,_n3##y,_p3##z,v), \
 I[56] = (img)(_p3##x,_n4##y,_p3##z,v), I[57] = (img)(_p2##x,_n4##y,_p3##z,v), I[58] = (img)(_p1##x,_n4##y,_p3##z,v), I[59] = (img)(x,_n4##y,_p3##z,v), I[60] = (img)(_n1##x,_n4##y,_p3##z,v), I[61] = (img)(_n2##x,_n4##y,_p3##z,v), I[62] = (img)(_n3##x,_n4##y,_p3##z,v), I[63] = (img)(_n4##x,_n4##y,_p3##z,v), \
 I[64] = (img)(_p3##x,_p3##y,_p2##z,v), I[65] = (img)(_p2##x,_p3##y,_p2##z,v), I[66] = (img)(_p1##x,_p3##y,_p2##z,v), I[67] = (img)(x,_p3##y,_p2##z,v), I[68] = (img)(_n1##x,_p3##y,_p2##z,v), I[69] = (img)(_n2##x,_p3##y,_p2##z,v), I[70] = (img)(_n3##x,_p3##y,_p2##z,v), I[71] = (img)(_n4##x,_p3##y,_p2##z,v), \
 I[72] = (img)(_p3##x,_p2##y,_p2##z,v), I[73] = (img)(_p2##x,_p2##y,_p2##z,v), I[74] = (img)(_p1##x,_p2##y,_p2##z,v), I[75] = (img)(x,_p2##y,_p2##z,v), I[76] = (img)(_n1##x,_p2##y,_p2##z,v), I[77] = (img)(_n2##x,_p2##y,_p2##z,v), I[78] = (img)(_n3##x,_p2##y,_p2##z,v), I[79] = (img)(_n4##x,_p2##y,_p2##z,v), \
 I[80] = (img)(_p3##x,_p1##y,_p2##z,v), I[81] = (img)(_p2##x,_p1##y,_p2##z,v), I[82] = (img)(_p1##x,_p1##y,_p2##z,v), I[83] = (img)(x,_p1##y,_p2##z,v), I[84] = (img)(_n1##x,_p1##y,_p2##z,v), I[85] = (img)(_n2##x,_p1##y,_p2##z,v), I[86] = (img)(_n3##x,_p1##y,_p2##z,v), I[87] = (img)(_n4##x,_p1##y,_p2##z,v), \
 I[88] = (img)(_p3##x,y,_p2##z,v), I[89] = (img)(_p2##x,y,_p2##z,v), I[90] = (img)(_p1##x,y,_p2##z,v), I[91] = (img)(x,y,_p2##z,v), I[92] = (img)(_n1##x,y,_p2##z,v), I[93] = (img)(_n2##x,y,_p2##z,v), I[94] = (img)(_n3##x,y,_p2##z,v), I[95] = (img)(_n4##x,y,_p2##z,v), \
 I[96] = (img)(_p3##x,_n1##y,_p2##z,v), I[97] = (img)(_p2##x,_n1##y,_p2##z,v), I[98] = (img)(_p1##x,_n1##y,_p2##z,v), I[99] = (img)(x,_n1##y,_p2##z,v), I[100] = (img)(_n1##x,_n1##y,_p2##z,v), I[101] = (img)(_n2##x,_n1##y,_p2##z,v), I[102] = (img)(_n3##x,_n1##y,_p2##z,v), I[103] = (img)(_n4##x,_n1##y,_p2##z,v), \
 I[104] = (img)(_p3##x,_n2##y,_p2##z,v), I[105] = (img)(_p2##x,_n2##y,_p2##z,v), I[106] = (img)(_p1##x,_n2##y,_p2##z,v), I[107] = (img)(x,_n2##y,_p2##z,v), I[108] = (img)(_n1##x,_n2##y,_p2##z,v), I[109] = (img)(_n2##x,_n2##y,_p2##z,v), I[110] = (img)(_n3##x,_n2##y,_p2##z,v), I[111] = (img)(_n4##x,_n2##y,_p2##z,v), \
 I[112] = (img)(_p3##x,_n3##y,_p2##z,v), I[113] = (img)(_p2##x,_n3##y,_p2##z,v), I[114] = (img)(_p1##x,_n3##y,_p2##z,v), I[115] = (img)(x,_n3##y,_p2##z,v), I[116] = (img)(_n1##x,_n3##y,_p2##z,v), I[117] = (img)(_n2##x,_n3##y,_p2##z,v), I[118] = (img)(_n3##x,_n3##y,_p2##z,v), I[119] = (img)(_n4##x,_n3##y,_p2##z,v), \
 I[120] = (img)(_p3##x,_n4##y,_p2##z,v), I[121] = (img)(_p2##x,_n4##y,_p2##z,v), I[122] = (img)(_p1##x,_n4##y,_p2##z,v), I[123] = (img)(x,_n4##y,_p2##z,v), I[124] = (img)(_n1##x,_n4##y,_p2##z,v), I[125] = (img)(_n2##x,_n4##y,_p2##z,v), I[126] = (img)(_n3##x,_n4##y,_p2##z,v), I[127] = (img)(_n4##x,_n4##y,_p2##z,v), \
 I[128] = (img)(_p3##x,_p3##y,_p1##z,v), I[129] = (img)(_p2##x,_p3##y,_p1##z,v), I[130] = (img)(_p1##x,_p3##y,_p1##z,v), I[131] = (img)(x,_p3##y,_p1##z,v), I[132] = (img)(_n1##x,_p3##y,_p1##z,v), I[133] = (img)(_n2##x,_p3##y,_p1##z,v), I[134] = (img)(_n3##x,_p3##y,_p1##z,v), I[135] = (img)(_n4##x,_p3##y,_p1##z,v), \
 I[136] = (img)(_p3##x,_p2##y,_p1##z,v), I[137] = (img)(_p2##x,_p2##y,_p1##z,v), I[138] = (img)(_p1##x,_p2##y,_p1##z,v), I[139] = (img)(x,_p2##y,_p1##z,v), I[140] = (img)(_n1##x,_p2##y,_p1##z,v), I[141] = (img)(_n2##x,_p2##y,_p1##z,v), I[142] = (img)(_n3##x,_p2##y,_p1##z,v), I[143] = (img)(_n4##x,_p2##y,_p1##z,v), \
 I[144] = (img)(_p3##x,_p1##y,_p1##z,v), I[145] = (img)(_p2##x,_p1##y,_p1##z,v), I[146] = (img)(_p1##x,_p1##y,_p1##z,v), I[147] = (img)(x,_p1##y,_p1##z,v), I[148] = (img)(_n1##x,_p1##y,_p1##z,v), I[149] = (img)(_n2##x,_p1##y,_p1##z,v), I[150] = (img)(_n3##x,_p1##y,_p1##z,v), I[151] = (img)(_n4##x,_p1##y,_p1##z,v), \
 I[152] = (img)(_p3##x,y,_p1##z,v), I[153] = (img)(_p2##x,y,_p1##z,v), I[154] = (img)(_p1##x,y,_p1##z,v), I[155] = (img)(x,y,_p1##z,v), I[156] = (img)(_n1##x,y,_p1##z,v), I[157] = (img)(_n2##x,y,_p1##z,v), I[158] = (img)(_n3##x,y,_p1##z,v), I[159] = (img)(_n4##x,y,_p1##z,v), \
 I[160] = (img)(_p3##x,_n1##y,_p1##z,v), I[161] = (img)(_p2##x,_n1##y,_p1##z,v), I[162] = (img)(_p1##x,_n1##y,_p1##z,v), I[163] = (img)(x,_n1##y,_p1##z,v), I[164] = (img)(_n1##x,_n1##y,_p1##z,v), I[165] = (img)(_n2##x,_n1##y,_p1##z,v), I[166] = (img)(_n3##x,_n1##y,_p1##z,v), I[167] = (img)(_n4##x,_n1##y,_p1##z,v), \
 I[168] = (img)(_p3##x,_n2##y,_p1##z,v), I[169] = (img)(_p2##x,_n2##y,_p1##z,v), I[170] = (img)(_p1##x,_n2##y,_p1##z,v), I[171] = (img)(x,_n2##y,_p1##z,v), I[172] = (img)(_n1##x,_n2##y,_p1##z,v), I[173] = (img)(_n2##x,_n2##y,_p1##z,v), I[174] = (img)(_n3##x,_n2##y,_p1##z,v), I[175] = (img)(_n4##x,_n2##y,_p1##z,v), \
 I[176] = (img)(_p3##x,_n3##y,_p1##z,v), I[177] = (img)(_p2##x,_n3##y,_p1##z,v), I[178] = (img)(_p1##x,_n3##y,_p1##z,v), I[179] = (img)(x,_n3##y,_p1##z,v), I[180] = (img)(_n1##x,_n3##y,_p1##z,v), I[181] = (img)(_n2##x,_n3##y,_p1##z,v), I[182] = (img)(_n3##x,_n3##y,_p1##z,v), I[183] = (img)(_n4##x,_n3##y,_p1##z,v), \
 I[184] = (img)(_p3##x,_n4##y,_p1##z,v), I[185] = (img)(_p2##x,_n4##y,_p1##z,v), I[186] = (img)(_p1##x,_n4##y,_p1##z,v), I[187] = (img)(x,_n4##y,_p1##z,v), I[188] = (img)(_n1##x,_n4##y,_p1##z,v), I[189] = (img)(_n2##x,_n4##y,_p1##z,v), I[190] = (img)(_n3##x,_n4##y,_p1##z,v), I[191] = (img)(_n4##x,_n4##y,_p1##z,v), \
 I[192] = (img)(_p3##x,_p3##y,z,v), I[193] = (img)(_p2##x,_p3##y,z,v), I[194] = (img)(_p1##x,_p3##y,z,v), I[195] = (img)(x,_p3##y,z,v), I[196] = (img)(_n1##x,_p3##y,z,v), I[197] = (img)(_n2##x,_p3##y,z,v), I[198] = (img)(_n3##x,_p3##y,z,v), I[199] = (img)(_n4##x,_p3##y,z,v), \
 I[200] = (img)(_p3##x,_p2##y,z,v), I[201] = (img)(_p2##x,_p2##y,z,v), I[202] = (img)(_p1##x,_p2##y,z,v), I[203] = (img)(x,_p2##y,z,v), I[204] = (img)(_n1##x,_p2##y,z,v), I[205] = (img)(_n2##x,_p2##y,z,v), I[206] = (img)(_n3##x,_p2##y,z,v), I[207] = (img)(_n4##x,_p2##y,z,v), \
 I[208] = (img)(_p3##x,_p1##y,z,v), I[209] = (img)(_p2##x,_p1##y,z,v), I[210] = (img)(_p1##x,_p1##y,z,v), I[211] = (img)(x,_p1##y,z,v), I[212] = (img)(_n1##x,_p1##y,z,v), I[213] = (img)(_n2##x,_p1##y,z,v), I[214] = (img)(_n3##x,_p1##y,z,v), I[215] = (img)(_n4##x,_p1##y,z,v), \
 I[216] = (img)(_p3##x,y,z,v), I[217] = (img)(_p2##x,y,z,v), I[218] = (img)(_p1##x,y,z,v), I[219] = (img)(x,y,z,v), I[220] = (img)(_n1##x,y,z,v), I[221] = (img)(_n2##x,y,z,v), I[222] = (img)(_n3##x,y,z,v), I[223] = (img)(_n4##x,y,z,v), \
 I[224] = (img)(_p3##x,_n1##y,z,v), I[225] = (img)(_p2##x,_n1##y,z,v), I[226] = (img)(_p1##x,_n1##y,z,v), I[227] = (img)(x,_n1##y,z,v), I[228] = (img)(_n1##x,_n1##y,z,v), I[229] = (img)(_n2##x,_n1##y,z,v), I[230] = (img)(_n3##x,_n1##y,z,v), I[231] = (img)(_n4##x,_n1##y,z,v), \
 I[232] = (img)(_p3##x,_n2##y,z,v), I[233] = (img)(_p2##x,_n2##y,z,v), I[234] = (img)(_p1##x,_n2##y,z,v), I[235] = (img)(x,_n2##y,z,v), I[236] = (img)(_n1##x,_n2##y,z,v), I[237] = (img)(_n2##x,_n2##y,z,v), I[238] = (img)(_n3##x,_n2##y,z,v), I[239] = (img)(_n4##x,_n2##y,z,v), \
 I[240] = (img)(_p3##x,_n3##y,z,v), I[241] = (img)(_p2##x,_n3##y,z,v), I[242] = (img)(_p1##x,_n3##y,z,v), I[243] = (img)(x,_n3##y,z,v), I[244] = (img)(_n1##x,_n3##y,z,v), I[245] = (img)(_n2##x,_n3##y,z,v), I[246] = (img)(_n3##x,_n3##y,z,v), I[247] = (img)(_n4##x,_n3##y,z,v), \
 I[248] = (img)(_p3##x,_n4##y,z,v), I[249] = (img)(_p2##x,_n4##y,z,v), I[250] = (img)(_p1##x,_n4##y,z,v), I[251] = (img)(x,_n4##y,z,v), I[252] = (img)(_n1##x,_n4##y,z,v), I[253] = (img)(_n2##x,_n4##y,z,v), I[254] = (img)(_n3##x,_n4##y,z,v), I[255] = (img)(_n4##x,_n4##y,z,v), \
 I[256] = (img)(_p3##x,_p3##y,_n1##z,v), I[257] = (img)(_p2##x,_p3##y,_n1##z,v), I[258] = (img)(_p1##x,_p3##y,_n1##z,v), I[259] = (img)(x,_p3##y,_n1##z,v), I[260] = (img)(_n1##x,_p3##y,_n1##z,v), I[261] = (img)(_n2##x,_p3##y,_n1##z,v), I[262] = (img)(_n3##x,_p3##y,_n1##z,v), I[263] = (img)(_n4##x,_p3##y,_n1##z,v), \
 I[264] = (img)(_p3##x,_p2##y,_n1##z,v), I[265] = (img)(_p2##x,_p2##y,_n1##z,v), I[266] = (img)(_p1##x,_p2##y,_n1##z,v), I[267] = (img)(x,_p2##y,_n1##z,v), I[268] = (img)(_n1##x,_p2##y,_n1##z,v), I[269] = (img)(_n2##x,_p2##y,_n1##z,v), I[270] = (img)(_n3##x,_p2##y,_n1##z,v), I[271] = (img)(_n4##x,_p2##y,_n1##z,v), \
 I[272] = (img)(_p3##x,_p1##y,_n1##z,v), I[273] = (img)(_p2##x,_p1##y,_n1##z,v), I[274] = (img)(_p1##x,_p1##y,_n1##z,v), I[275] = (img)(x,_p1##y,_n1##z,v), I[276] = (img)(_n1##x,_p1##y,_n1##z,v), I[277] = (img)(_n2##x,_p1##y,_n1##z,v), I[278] = (img)(_n3##x,_p1##y,_n1##z,v), I[279] = (img)(_n4##x,_p1##y,_n1##z,v), \
 I[280] = (img)(_p3##x,y,_n1##z,v), I[281] = (img)(_p2##x,y,_n1##z,v), I[282] = (img)(_p1##x,y,_n1##z,v), I[283] = (img)(x,y,_n1##z,v), I[284] = (img)(_n1##x,y,_n1##z,v), I[285] = (img)(_n2##x,y,_n1##z,v), I[286] = (img)(_n3##x,y,_n1##z,v), I[287] = (img)(_n4##x,y,_n1##z,v), \
 I[288] = (img)(_p3##x,_n1##y,_n1##z,v), I[289] = (img)(_p2##x,_n1##y,_n1##z,v), I[290] = (img)(_p1##x,_n1##y,_n1##z,v), I[291] = (img)(x,_n1##y,_n1##z,v), I[292] = (img)(_n1##x,_n1##y,_n1##z,v), I[293] = (img)(_n2##x,_n1##y,_n1##z,v), I[294] = (img)(_n3##x,_n1##y,_n1##z,v), I[295] = (img)(_n4##x,_n1##y,_n1##z,v), \
 I[296] = (img)(_p3##x,_n2##y,_n1##z,v), I[297] = (img)(_p2##x,_n2##y,_n1##z,v), I[298] = (img)(_p1##x,_n2##y,_n1##z,v), I[299] = (img)(x,_n2##y,_n1##z,v), I[300] = (img)(_n1##x,_n2##y,_n1##z,v), I[301] = (img)(_n2##x,_n2##y,_n1##z,v), I[302] = (img)(_n3##x,_n2##y,_n1##z,v), I[303] = (img)(_n4##x,_n2##y,_n1##z,v), \
 I[304] = (img)(_p3##x,_n3##y,_n1##z,v), I[305] = (img)(_p2##x,_n3##y,_n1##z,v), I[306] = (img)(_p1##x,_n3##y,_n1##z,v), I[307] = (img)(x,_n3##y,_n1##z,v), I[308] = (img)(_n1##x,_n3##y,_n1##z,v), I[309] = (img)(_n2##x,_n3##y,_n1##z,v), I[310] = (img)(_n3##x,_n3##y,_n1##z,v), I[311] = (img)(_n4##x,_n3##y,_n1##z,v), \
 I[312] = (img)(_p3##x,_n4##y,_n1##z,v), I[313] = (img)(_p2##x,_n4##y,_n1##z,v), I[314] = (img)(_p1##x,_n4##y,_n1##z,v), I[315] = (img)(x,_n4##y,_n1##z,v), I[316] = (img)(_n1##x,_n4##y,_n1##z,v), I[317] = (img)(_n2##x,_n4##y,_n1##z,v), I[318] = (img)(_n3##x,_n4##y,_n1##z,v), I[319] = (img)(_n4##x,_n4##y,_n1##z,v), \
 I[320] = (img)(_p3##x,_p3##y,_n2##z,v), I[321] = (img)(_p2##x,_p3##y,_n2##z,v), I[322] = (img)(_p1##x,_p3##y,_n2##z,v), I[323] = (img)(x,_p3##y,_n2##z,v), I[324] = (img)(_n1##x,_p3##y,_n2##z,v), I[325] = (img)(_n2##x,_p3##y,_n2##z,v), I[326] = (img)(_n3##x,_p3##y,_n2##z,v), I[327] = (img)(_n4##x,_p3##y,_n2##z,v), \
 I[328] = (img)(_p3##x,_p2##y,_n2##z,v), I[329] = (img)(_p2##x,_p2##y,_n2##z,v), I[330] = (img)(_p1##x,_p2##y,_n2##z,v), I[331] = (img)(x,_p2##y,_n2##z,v), I[332] = (img)(_n1##x,_p2##y,_n2##z,v), I[333] = (img)(_n2##x,_p2##y,_n2##z,v), I[334] = (img)(_n3##x,_p2##y,_n2##z,v), I[335] = (img)(_n4##x,_p2##y,_n2##z,v), \
 I[336] = (img)(_p3##x,_p1##y,_n2##z,v), I[337] = (img)(_p2##x,_p1##y,_n2##z,v), I[338] = (img)(_p1##x,_p1##y,_n2##z,v), I[339] = (img)(x,_p1##y,_n2##z,v), I[340] = (img)(_n1##x,_p1##y,_n2##z,v), I[341] = (img)(_n2##x,_p1##y,_n2##z,v), I[342] = (img)(_n3##x,_p1##y,_n2##z,v), I[343] = (img)(_n4##x,_p1##y,_n2##z,v), \
 I[344] = (img)(_p3##x,y,_n2##z,v), I[345] = (img)(_p2##x,y,_n2##z,v), I[346] = (img)(_p1##x,y,_n2##z,v), I[347] = (img)(x,y,_n2##z,v), I[348] = (img)(_n1##x,y,_n2##z,v), I[349] = (img)(_n2##x,y,_n2##z,v), I[350] = (img)(_n3##x,y,_n2##z,v), I[351] = (img)(_n4##x,y,_n2##z,v), \
 I[352] = (img)(_p3##x,_n1##y,_n2##z,v), I[353] = (img)(_p2##x,_n1##y,_n2##z,v), I[354] = (img)(_p1##x,_n1##y,_n2##z,v), I[355] = (img)(x,_n1##y,_n2##z,v), I[356] = (img)(_n1##x,_n1##y,_n2##z,v), I[357] = (img)(_n2##x,_n1##y,_n2##z,v), I[358] = (img)(_n3##x,_n1##y,_n2##z,v), I[359] = (img)(_n4##x,_n1##y,_n2##z,v), \
 I[360] = (img)(_p3##x,_n2##y,_n2##z,v), I[361] = (img)(_p2##x,_n2##y,_n2##z,v), I[362] = (img)(_p1##x,_n2##y,_n2##z,v), I[363] = (img)(x,_n2##y,_n2##z,v), I[364] = (img)(_n1##x,_n2##y,_n2##z,v), I[365] = (img)(_n2##x,_n2##y,_n2##z,v), I[366] = (img)(_n3##x,_n2##y,_n2##z,v), I[367] = (img)(_n4##x,_n2##y,_n2##z,v), \
 I[368] = (img)(_p3##x,_n3##y,_n2##z,v), I[369] = (img)(_p2##x,_n3##y,_n2##z,v), I[370] = (img)(_p1##x,_n3##y,_n2##z,v), I[371] = (img)(x,_n3##y,_n2##z,v), I[372] = (img)(_n1##x,_n3##y,_n2##z,v), I[373] = (img)(_n2##x,_n3##y,_n2##z,v), I[374] = (img)(_n3##x,_n3##y,_n2##z,v), I[375] = (img)(_n4##x,_n3##y,_n2##z,v), \
 I[376] = (img)(_p3##x,_n4##y,_n2##z,v), I[377] = (img)(_p2##x,_n4##y,_n2##z,v), I[378] = (img)(_p1##x,_n4##y,_n2##z,v), I[379] = (img)(x,_n4##y,_n2##z,v), I[380] = (img)(_n1##x,_n4##y,_n2##z,v), I[381] = (img)(_n2##x,_n4##y,_n2##z,v), I[382] = (img)(_n3##x,_n4##y,_n2##z,v), I[383] = (img)(_n4##x,_n4##y,_n2##z,v), \
 I[384] = (img)(_p3##x,_p3##y,_n3##z,v), I[385] = (img)(_p2##x,_p3##y,_n3##z,v), I[386] = (img)(_p1##x,_p3##y,_n3##z,v), I[387] = (img)(x,_p3##y,_n3##z,v), I[388] = (img)(_n1##x,_p3##y,_n3##z,v), I[389] = (img)(_n2##x,_p3##y,_n3##z,v), I[390] = (img)(_n3##x,_p3##y,_n3##z,v), I[391] = (img)(_n4##x,_p3##y,_n3##z,v), \
 I[392] = (img)(_p3##x,_p2##y,_n3##z,v), I[393] = (img)(_p2##x,_p2##y,_n3##z,v), I[394] = (img)(_p1##x,_p2##y,_n3##z,v), I[395] = (img)(x,_p2##y,_n3##z,v), I[396] = (img)(_n1##x,_p2##y,_n3##z,v), I[397] = (img)(_n2##x,_p2##y,_n3##z,v), I[398] = (img)(_n3##x,_p2##y,_n3##z,v), I[399] = (img)(_n4##x,_p2##y,_n3##z,v), \
 I[400] = (img)(_p3##x,_p1##y,_n3##z,v), I[401] = (img)(_p2##x,_p1##y,_n3##z,v), I[402] = (img)(_p1##x,_p1##y,_n3##z,v), I[403] = (img)(x,_p1##y,_n3##z,v), I[404] = (img)(_n1##x,_p1##y,_n3##z,v), I[405] = (img)(_n2##x,_p1##y,_n3##z,v), I[406] = (img)(_n3##x,_p1##y,_n3##z,v), I[407] = (img)(_n4##x,_p1##y,_n3##z,v), \
 I[408] = (img)(_p3##x,y,_n3##z,v), I[409] = (img)(_p2##x,y,_n3##z,v), I[410] = (img)(_p1##x,y,_n3##z,v), I[411] = (img)(x,y,_n3##z,v), I[412] = (img)(_n1##x,y,_n3##z,v), I[413] = (img)(_n2##x,y,_n3##z,v), I[414] = (img)(_n3##x,y,_n3##z,v), I[415] = (img)(_n4##x,y,_n3##z,v), \
 I[416] = (img)(_p3##x,_n1##y,_n3##z,v), I[417] = (img)(_p2##x,_n1##y,_n3##z,v), I[418] = (img)(_p1##x,_n1##y,_n3##z,v), I[419] = (img)(x,_n1##y,_n3##z,v), I[420] = (img)(_n1##x,_n1##y,_n3##z,v), I[421] = (img)(_n2##x,_n1##y,_n3##z,v), I[422] = (img)(_n3##x,_n1##y,_n3##z,v), I[423] = (img)(_n4##x,_n1##y,_n3##z,v), \
 I[424] = (img)(_p3##x,_n2##y,_n3##z,v), I[425] = (img)(_p2##x,_n2##y,_n3##z,v), I[426] = (img)(_p1##x,_n2##y,_n3##z,v), I[427] = (img)(x,_n2##y,_n3##z,v), I[428] = (img)(_n1##x,_n2##y,_n3##z,v), I[429] = (img)(_n2##x,_n2##y,_n3##z,v), I[430] = (img)(_n3##x,_n2##y,_n3##z,v), I[431] = (img)(_n4##x,_n2##y,_n3##z,v), \
 I[432] = (img)(_p3##x,_n3##y,_n3##z,v), I[433] = (img)(_p2##x,_n3##y,_n3##z,v), I[434] = (img)(_p1##x,_n3##y,_n3##z,v), I[435] = (img)(x,_n3##y,_n3##z,v), I[436] = (img)(_n1##x,_n3##y,_n3##z,v), I[437] = (img)(_n2##x,_n3##y,_n3##z,v), I[438] = (img)(_n3##x,_n3##y,_n3##z,v), I[439] = (img)(_n4##x,_n3##y,_n3##z,v), \
 I[440] = (img)(_p3##x,_n4##y,_n3##z,v), I[441] = (img)(_p2##x,_n4##y,_n3##z,v), I[442] = (img)(_p1##x,_n4##y,_n3##z,v), I[443] = (img)(x,_n4##y,_n3##z,v), I[444] = (img)(_n1##x,_n4##y,_n3##z,v), I[445] = (img)(_n2##x,_n4##y,_n3##z,v), I[446] = (img)(_n3##x,_n4##y,_n3##z,v), I[447] = (img)(_n4##x,_n4##y,_n3##z,v), \
 I[448] = (img)(_p3##x,_p3##y,_n4##z,v), I[449] = (img)(_p2##x,_p3##y,_n4##z,v), I[450] = (img)(_p1##x,_p3##y,_n4##z,v), I[451] = (img)(x,_p3##y,_n4##z,v), I[452] = (img)(_n1##x,_p3##y,_n4##z,v), I[453] = (img)(_n2##x,_p3##y,_n4##z,v), I[454] = (img)(_n3##x,_p3##y,_n4##z,v), I[455] = (img)(_n4##x,_p3##y,_n4##z,v), \
 I[456] = (img)(_p3##x,_p2##y,_n4##z,v), I[457] = (img)(_p2##x,_p2##y,_n4##z,v), I[458] = (img)(_p1##x,_p2##y,_n4##z,v), I[459] = (img)(x,_p2##y,_n4##z,v), I[460] = (img)(_n1##x,_p2##y,_n4##z,v), I[461] = (img)(_n2##x,_p2##y,_n4##z,v), I[462] = (img)(_n3##x,_p2##y,_n4##z,v), I[463] = (img)(_n4##x,_p2##y,_n4##z,v), \
 I[464] = (img)(_p3##x,_p1##y,_n4##z,v), I[465] = (img)(_p2##x,_p1##y,_n4##z,v), I[466] = (img)(_p1##x,_p1##y,_n4##z,v), I[467] = (img)(x,_p1##y,_n4##z,v), I[468] = (img)(_n1##x,_p1##y,_n4##z,v), I[469] = (img)(_n2##x,_p1##y,_n4##z,v), I[470] = (img)(_n3##x,_p1##y,_n4##z,v), I[471] = (img)(_n4##x,_p1##y,_n4##z,v), \
 I[472] = (img)(_p3##x,y,_n4##z,v), I[473] = (img)(_p2##x,y,_n4##z,v), I[474] = (img)(_p1##x,y,_n4##z,v), I[475] = (img)(x,y,_n4##z,v), I[476] = (img)(_n1##x,y,_n4##z,v), I[477] = (img)(_n2##x,y,_n4##z,v), I[478] = (img)(_n3##x,y,_n4##z,v), I[479] = (img)(_n4##x,y,_n4##z,v), \
 I[480] = (img)(_p3##x,_n1##y,_n4##z,v), I[481] = (img)(_p2##x,_n1##y,_n4##z,v), I[482] = (img)(_p1##x,_n1##y,_n4##z,v), I[483] = (img)(x,_n1##y,_n4##z,v), I[484] = (img)(_n1##x,_n1##y,_n4##z,v), I[485] = (img)(_n2##x,_n1##y,_n4##z,v), I[486] = (img)(_n3##x,_n1##y,_n4##z,v), I[487] = (img)(_n4##x,_n1##y,_n4##z,v), \
 I[488] = (img)(_p3##x,_n2##y,_n4##z,v), I[489] = (img)(_p2##x,_n2##y,_n4##z,v), I[490] = (img)(_p1##x,_n2##y,_n4##z,v), I[491] = (img)(x,_n2##y,_n4##z,v), I[492] = (img)(_n1##x,_n2##y,_n4##z,v), I[493] = (img)(_n2##x,_n2##y,_n4##z,v), I[494] = (img)(_n3##x,_n2##y,_n4##z,v), I[495] = (img)(_n4##x,_n2##y,_n4##z,v), \
 I[496] = (img)(_p3##x,_n3##y,_n4##z,v), I[497] = (img)(_p2##x,_n3##y,_n4##z,v), I[498] = (img)(_p1##x,_n3##y,_n4##z,v), I[499] = (img)(x,_n3##y,_n4##z,v), I[500] = (img)(_n1##x,_n3##y,_n4##z,v), I[501] = (img)(_n2##x,_n3##y,_n4##z,v), I[502] = (img)(_n3##x,_n3##y,_n4##z,v), I[503] = (img)(_n4##x,_n3##y,_n4##z,v), \
 I[504] = (img)(_p3##x,_n4##y,_n4##z,v), I[505] = (img)(_p2##x,_n4##y,_n4##z,v), I[506] = (img)(_p1##x,_n4##y,_n4##z,v), I[507] = (img)(x,_n4##y,_n4##z,v), I[508] = (img)(_n1##x,_n4##y,_n4##z,v), I[509] = (img)(_n2##x,_n4##y,_n4##z,v), I[510] = (img)(_n3##x,_n4##y,_n4##z,v), I[511] = (img)(_n4##x,_n4##y,_n4##z,v);

#endif
