/*#include <stdio.h>
#include <cuda.h>

#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

#define DIM 500

struct cuComplex
{

	float r;
	float i;

	__device__ cuComplex( float a, float b ) : 	r(a), i(b) {}
	__device__ float magnitude2( void )
	{
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& a)
	{
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a)
	{
		return cuComplex(r+a.r, i+a.i);
	}
};

__device__ int julia( int x, int y )
{
	const float scale = 1.5;
	float jx = scale * (float)( x - DIM/2)/(DIM/2);
	float jy = scale * (float)(y - DIM/2)/(DIM/2);
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);

	int i = 0;
	for (i=0; i<200; i++)
	{
		a = a * a + c;
		if (a.magnitude2() > 1000)
			return 0;
	}
	return 1;
}

__global__ void kernel( unsigned char *ptr )
{
	// map from blockIdx to pixel position
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * blockDim.x * gridDim.x;
	// now calculate the value at that position
	int juliaValue = julia( x, y );
	ptr[offset] = 255 * juliaValue;
}


int main( void )
{
	unsigned char *gpu_bitmap;
	unsigned char *cpu_bitmap=(unsigned char*)malloc(sizeof(unsigned char)*DIM*DIM*4);
	cudaMalloc( (void**)&gpu_bitmap, DIM*DIM*4 );
	dim3 grid(DIM,DIM);
	kernel<<<grid,1>>>( gpu_bitmap );
	cudaMemcpy( cpu_bitmap, gpu_bitmap,DIM*DIM*4 , cudaMemcpyDeviceToHost );
	cudaFree( gpu_bitmap );

	printf("hola \n");

	Mat imgJulia = Mat::zeros(DIM, DIM, CV_8UC3);
	for(int x=0; x<DIM; x++)
    {
        for(int y=0; y<DIM; y++)
        {
        	int offset = x + y * DIM;
        	//imgJulia.at<Vec3b>(y,x)[2]= cpu_bitmap [offset + 0];
        	imgJulia.at<Vec3b>(y,x)[1]= cpu_bitmap [offset + 1];
        	//imgJulia.at<Vec3b>(y,x)[0]= cpu_bitmap [offset + 2];
	    }
    }
    imshow( "Imagen Julia", imgJulia );

	waitKey(0);
	return 0;
}
*/
