#ifndef PSEUDO_INVERSE_HPP
#define PSEUDO_INVERSE_HPP

#include <Eigen/Eigen>
#include <limits>
#include <cassert>
#include <cstdio>

namespace doppia
{
	/*
	 * Compute the Moore-Penrose-Pseudoinverse of a matrix M by SVD.
	 * The threshold was taken from Matlab, where it's defined as
	 * MAX(SIZE(A)) * NORM(A) * EPS(class(A))
	 */

        template<typename ScalarType, int R, int C>
        Eigen::Matrix< ScalarType, C, R > pseudo_inverse(const Eigen::Matrix< ScalarType, R, C>& M)
	{            
		// Eigen SVD requires MxN matrix with M >= N
		// But pinv(M) = (pinv(M^T))^T
		if(M.rows() < M.cols())
		{
                        const Eigen::Matrix< ScalarType, C, R > Mt = M.transpose();
			return pseudo_inverse(Mt).transpose();
		}

		// Compute the SVD of a MxN matrix with M >= N
		assert(M.rows() >= M.cols());

                Eigen::JacobiSVD<Eigen::Matrix< ScalarType, R, C> > svd(M);

		// Compute the norm of M
                ScalarType normM = 0;

		for(int i=0; i<svd.singularValues().size(); i+=1)
		{
			normM = std::max(normM, svd.singularValues()(i));
		}
		
		// Compute the zero threshold
                const ScalarType threshold = std::max(M.cols(), M.rows()) * normM * std::numeric_limits<ScalarType>::epsilon();
		
		// Compute the pseudo inverse of the diagonal singular value matrix
                Eigen::Matrix< ScalarType, C, R > tmp(M.cols(), M.rows());
		tmp.setZero();
		for(int i=0; i<svd.singularValues().size(); i+=1)
		{
			tmp(i,i) = 
				  (svd.singularValues()(i) < threshold)
				? 0
				: (1 / svd.singularValues()(i));
		}

		// Re-compose V * S^+ * U^T
		return svd.matrixV() * tmp * svd.matrixU().transpose();
	}

        template<typename MatrixType>
        Eigen::Matrix< typename MatrixType::Scalar, MatrixType::ColsAtCompileTime, MatrixType::RowsAtCompileTime > pseudo_inverse(const MatrixType& M)
        {
            Eigen::Matrix< typename MatrixType::Scalar, MatrixType::RowsAtCompileTime, MatrixType::ColsAtCompileTime > M2(M);
            return pseudo_inverse(M2);
        }
}

#endif
