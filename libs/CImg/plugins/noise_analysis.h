/*
 #
 #  File        : noise_analysis.h
 #                ( C++ header file - CImg plug-in )
 #
 #  Description : CImg plug-in that estimates noise standard deviation.
 #                This file is a part of the CImg Library project.
 #                ( http://cimg.sourceforge.net )
 #
 #  Copyright   : Jerome Boulanger
 #                ( http://www.irisa.fr/vista/Equipe/People/Jerome.Boulanger.html )
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

#ifndef cimg_plugin_noise_analysis
#define cimg_plugin_noise_analysis

//! Compute somme pseudo-residuals
/*
  The pseudo residual r_i of the image Y_i are so thar E[r_i^2] = E[Y_i^2].
  This is the 2D pseudo-implementation.
*/
CImg<float> get_pseudo_residuals() const {
  CImg<float> residu(dimx(),dimy(),dimz(),dim);
  if (!is_empty()){
    cimg_forXYZV(*this,x,y,z,v) {
      double t2 = 0;
      if (x==0) t2+=(*this)(x+1,y,z,v);
      else t2+=(*this)(x-1,y,z,v);
      if ((unsigned int)x==(unsigned int)(dimx()-1)) t2+=(*this)(x-1,y,z,v);
      else t2+=(*this)(x+1,y,z,v);
      if (y==0) t2+=(*this)(x,y+1,z,v);
      else t2+=(*this)(x,y-1,z,v);
      if ((unsigned int)y==(unsigned int)(dimy()-1)) t2+=(*this)(x,y-1,z,v);
      else t2+=(*this)(x,y+1,z,v);
      residu(x,y,z,v) = (float)(0.223606798*(4.*(double)(*this)(x,y,z,v)-t2));
    }
  }
  return residu;
}

//! Estimate the noise variance
/*
  \param method = 0 : Least Median of Square,
                  1 : Least Trimmed of Square,
                  2 : Least Mean of Square.
   Robustly estimatate the variance of a the noise using the pseudo-residuals.
   \see variance_estimation()
*/
double noise_variance(const unsigned int method=0) const {
  return (*this).get_pseudo_residuals().variance(method);
}

#endif
