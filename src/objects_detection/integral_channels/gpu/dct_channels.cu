//#include <cuda_runtime.h>
#include "dct_channels.cu.hpp"
#include "helpers/gpu/cuda_safe_call.hpp"

//#include <cudatemplates/devicememorypitched.hpp>
//#include <cudatemplates/devicememoryreference.hpp>
//#include <boost/cstdint.hpp>
//#include "cudatemplates/copy.hpp"





#define BLOCK_SIZE          8


/**
 *  Square of dimension of pixels block
 */
#define BLOCK_SIZE2         64


/**
 *  log_2{BLOCK_SIZE), used for quick multiplication or division by the
 *  pixels block dimension via shifting
 */
//#define BLOCK_SIZE_LOG2     3


/**
 *  log_2{BLOCK_SIZE*BLOCK_SIZE), used for quick multiplication or division by the
 *  square of pixels block via shifting
 */
//#define BLOCK_SIZE2_LOG2    6


/**
 *  This macro states that __mul24 operation is performed faster that traditional
 *  multiplication for two integers on CUDA. Please undefine if it appears to be
 *  wrong on your system
 */
//#define __MUL24_FASTER_THAN_ASTERIX


/**
 *  Wrapper to the fastest integer multiplication function on CUDA
 */
//#ifdef __MUL24_FASTER_THAN_ASTERIX
//#define FMUL(x,y)   (__mul24(x,y))
//#else
//#define FMUL(x,y)   ((x)*(y))
//#endif


/**
 *  This unitary matrix performs discrete cosine transform of rows of the matrix to the left
 */
__constant__ float DCTv8matrix[] =
{
		0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,  0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f,
		0.3535533905932738f,  0.4157348061512726f,  0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f,
		0.3535533905932738f,  0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f,  0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f,
		0.3535533905932738f,  0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f,  0.3535533905932737f,  0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f,
		0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f,  0.2777851165098009f,  0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f,  0.4903926402016152f,
		0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f,  0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f,  0.4619397662556437f, -0.4157348061512720f,
		0.3535533905932738f, -0.4157348061512727f,  0.1913417161825450f,  0.0975451610080640f, -0.3535533905932736f,  0.4903926402016152f, -0.4619397662556435f,  0.2777851165098022f,
		0.3535533905932738f, -0.4903926402016152f,  0.4619397662556433f, -0.4157348061512721f,  0.3535533905932733f, -0.2777851165098008f,  0.1913417161825431f, -0.0975451610080625f
};


// Temporary blocks
__shared__ float CurBlockLocal1[BLOCK_SIZE2];
__shared__ float CurBlockLocal2[BLOCK_SIZE2];

namespace doppia {

namespace integral_channels {
/**
 **************************************************************************
 *  Performs 1st implementation of 8x8 block-wise Forward Discrete Cosine Transform of the given
 *  image plane and outputs result to the array of coefficients.
 *
 * \param Dst            [OUT] - Coefficients plane
 * \param ImgWidth       [IN] - Stride of Dst
 * \param OffsetXBlocks  [IN] - Offset along X in blocks from which to perform processing
 * \param OffsetYBlocks  [IN] - Offset along Y in blocks from which to perform processing
 *
 * \return None
 */
#define STOPATXPOSTION -1
#define STOPATYPOSTION -1
__global__ void dct_channels_kernel(gpu_channels_t::KernelData input_channel)
{
	//gpu_channels_t::KernelData &input_channel = input_channelParam;
	//the real input annd output address of this kernel processing
	const size_t input_channel_stride = input_channel.stride[1];
	const size_t input_row_stride = input_channel.stride[0];
	const size_t input_width = input_channel.size[0];
	const size_t input_height = input_channel.size[1] * (input_channel.size[2]/2);
	const size_t num_input_channels = input_channel.size[2];

	gpu_channels_t::KernelData Src = input_channel;
	int dctChannelsIndexStart = input_channel_stride * (num_input_channels/2);
	//gpu_channels_t::KernelData Dst = &input_channel[input_channel_stride * (num_input_channels/2)];
	// Block index
	const int bx = blockIdx.x;
	const int by = blockIdx.y;

	// Thread index (current coefficient)
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;



	// global image index for the  src and dst global memory
	//int may exceed the limit?
	int img_x = bx * BLOCK_SIZE + tx;
	int img_y = by * BLOCK_SIZE + ty;
	if(img_x >=input_width ){
		//exceed the boundary of the image, use the border pixel to fill the temp shared block
		//		printf("out of bourndary x: bx=%d,by=%d,tx=%d,ty=%d,img_x=%d, img_y=%d, input_width=%d\n",
		//						bx,by,tx,ty,img_x,img_y,input_width);
		img_x = input_width - 1;
	}
	if(img_y >= input_height){
		//exceed the boundary of the image, use the border pixel to fill the temp shared block
		//		printf("out of bourndary y: bx=%d,by=%d,tx=%d,ty=%d,img_x=%d, img_y=%d, input_height=%d\n",
		//								bx,by,tx,ty,img_x,img_y,input_height);
		img_y = input_height - 1;

	}
	const int img_index = img_y * input_row_stride + img_x;
	//the block index for the shared memory block
	//const int blockIndex = ty<< BLOCK_SIZE_LOG2 + tx;
	const int blockIndex = ty* BLOCK_SIZE+ tx;

	//copy current image pixel to the first block
	//CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = tex2D(TexSrc, tex_x, tex_y);
	//printf("bx=%d,by=%d,tx=%d,ty=%d,blockIndex=%d, img_index=%d\n", bx,by,tx,ty,blockIndex,img_index);
	CurBlockLocal1[blockIndex] = (Src.data[img_index] -128.0f);

	if(img_x==STOPATXPOSTION && img_y==STOPATYPOSTION){
		printf("data in kernel:x=%d,y=%d, value=%f \n",
				img_x,img_y, (Src.data[img_index] -128.0f));
	}
	//synchronize threads to make sure the block is copied
	__syncthreads();

	//Now the temp block has been properly padded, we can ignore the image pixel that is outside
	//of the image boundary
	img_x = bx * BLOCK_SIZE + tx;
	img_y = by * BLOCK_SIZE + ty;
	if(img_x >=input_width || img_y >= input_height){
		//		printf("out of bourndary: bx=%d,by=%d,tx=%d,ty=%d,img_x=%d, img_y=%d\n", bx,by,tx,ty,img_x,img_y);
		//out of the boundary and do not process them at all
		return;
	}


	//calculate the multiplication of DCTv8matrixT * A and place it in the second block
	float curelem = 0;
	int DCTv8matrixIndex = 0 * BLOCK_SIZE + ty;
	int CurBlockLocal1Index = 0 * BLOCK_SIZE + tx;
#pragma unroll

	for (int i=0; i<BLOCK_SIZE; i++)
	{
		curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
		DCTv8matrixIndex += BLOCK_SIZE;
		CurBlockLocal1Index += BLOCK_SIZE;
	}

	//CurBlockLocal2[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;
	CurBlockLocal2[blockIndex] = curelem;

	if(img_x==STOPATXPOSTION && img_y==STOPATYPOSTION){
		printf("data in kernel:x=%d,y=%d, value=%f \n",
				img_x,img_y, curelem);
	}

	//synchronize threads to make sure the first 2 matrices are multiplied and the result is stored in the second block
	__syncthreads();

	//calculate the multiplication of (DCTv8matrixT * A) * DCTv8matrix and place it in the first block
	curelem = 0;
	int CurBlockLocal2Index = ty *BLOCK_SIZE + 0;
	DCTv8matrixIndex = 0 * BLOCK_SIZE + tx;
#pragma unroll

	for (int i=0; i<BLOCK_SIZE; i++)
	{
		curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
		CurBlockLocal2Index += 1;
		DCTv8matrixIndex += BLOCK_SIZE;
	}

	//CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;
	CurBlockLocal1[blockIndex ] = curelem;

	if(img_x==STOPATXPOSTION && img_y==STOPATYPOSTION){
		printf("data in kernel:x=%d,y=%d, value=%f \n",
				img_x,img_y, curelem);
	}
	//synchronize threads to make sure the matrices are multiplied and the result is stored back in the first block
	__syncthreads();

	//copy current coefficient to its place in the result array
	//Dst[ FMUL(((by << BLOCK_SIZE_LOG2) + ty), ImgWidth) + ((bx << BLOCK_SIZE_LOG2) + tx) ] = CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ];
	//Dst[img_index] = CurBlockLocal1[blockIndex];
	//scale the dct value so that they can be accomodated by the byte arrary
	int val = abs(CurBlockLocal1[blockIndex]) * 0.1f;
	if(val>255){
		val = 255;
	}
	//const int x=bx*8 + tx;
	//const int y=by*8 + ty;
//	if(x==72 && y==1352){
//		printf("dct value in kenerl: x=%d,y=%d,img_index=%d,transformeddctvalue=%d, dctvalue=%f, originalvalue =%d\n",
//				tx,ty,img_index,val,CurBlockLocal1[blockIndex],Src.data[img_index]);
//	}

	Src.data[img_index + dctChannelsIndexStart] = val;




	//	if(bx==9 && by==169 && tx==0 && ty==0){
	//		printf("dct value in kenerl: x=%d,y=%d,img_index=%d,value=%d, originalvalue =%d\n",
	//				tx,ty,img_index,val,Src.data[img_index + dctChannelsIndexStart]);
	//	}

}

void compute_dct_channels(gpu_channels_t &input_channel){

	const int image_width = input_channel.size[0];
	//treat the whole original channels as one big image, with each image lay vertically
	//in sequence
	const int image_height = input_channel.size[1] * (input_channel.size[2]/2);
	//The grid and block layout is as below
	//(width/blocksize X height/blocksize)    (blocksize X blocksize)
	//this is so arranged so that we can allocate conveniently allocate shared memory for each
	//block
	const dim3 block_dimensions(BLOCK_SIZE, BLOCK_SIZE);
	const dim3 grid_dimensions(image_width/BLOCK_SIZE + 1, image_height/BLOCK_SIZE + 1);
	//assert(image_width%BLOCK_SIZE == 0 && image_height%BLOCK_SIZE == 0);

	dct_channels_kernel<<<grid_dimensions, block_dimensions>>>(input_channel);

	cuda_safe_call( cudaGetLastError() );
	cuda_safe_call( cudaDeviceSynchronize() ); // make sure all GPU computations are finished

}

} // end of namespace integral_channels

} // end of namespace doppia



