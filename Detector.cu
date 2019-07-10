#include <cuda.h>

#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <math.h>

using namespace cv;

String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;


/**
 * @brief Convertir IMG a escala de grices.
 * 
 * @param char Imagen de entrada
 * @param char Imagen de salida
 */
/*__global__ void kernel_Gris( unsigned char *imgColor, unsigned char *imgGris)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	imgGris[offset] = (0.3*imgColor[offset*3 + 0] + 0.59*imgColor[offset*3 + 1] + 0.11*imgColor[offset*3 + 2]);
}*/

/**
 * @brief Ordenar elementos para el uso del filtro de la Media
 * 
 * @param char Valores a ordenar
 * @param length Tamaño
 */
__device__ void insertion_sort (unsigned char arr[], int length)
{
	int j;
	unsigned char temp;

	for (int i = 0; i < length; i++)
	{
		j = i;

		while (j > 0 && arr[j] < arr[j-1])
		{
			temp = arr[j];
			arr[j] = arr[j-1];
			arr[j-1] = temp;
			j--;
		}
	}
}

/**
 * @brief Filtro de la media utilizando una ventana de 3x3
 * 
 * @param char Imagen de entrada
 * @param char Imagen de salida
 * @param ancho Ancho de la imagen
 * @param alto Alto de la imagen
 */
/*__global__ void kernel_Media(unsigned char *imgI, unsigned char *imgO, int ancho, int alto )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (x <= 0 || x >= ancho-1 || y <= 0 || y >= alto-1)
	{
		imgO[offset] = imgI[offset];
		return;
	}

	//-----------------Pixeles Vecinos--------------------
	int leftTop = offset - ancho - 1;
	int top = offset - ancho;
	int rightTop = offset - ancho + 1;

	int left = offset - 1;
	int right = offset + 1;


	int leftBottom = offset + ancho - 1;
	int bottom = offset + ancho;
	int rightBottom = offset + ancho + 1;

	unsigned char arr[9] = {imgI[leftTop], imgI[top], imgI[rightTop],
			imgI[left], imgI[offset], imgI[right],
			imgI[leftBottom], imgI[bottom], imgI[rightBottom]};

	insertion_sort(arr, 9);
	imgO[offset] = arr[4];
}*/
/**
 * @brief Filtro de la media utilizando una ventana de 5x5
 * 
 * @param char Imagen de entrada
 * @param char Imagen de salida
 * @param ancho Ancho de la imagen
 * @param alto Alto de la imagen
 */
__global__ void kernel_Media2(unsigned char *imgI, unsigned char *imgO, int ancho, int alto)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (x <= 1 || x >= ancho-2 || y <= 1 || y >= alto-2)
	{
		imgO[offset] = imgI[offset];
		return;
	}

	//-----------------Pixeles Vecinos--------------------
	int p11 = offset - 2*ancho - 2;
	int p12 = offset - 2*ancho - 1;
	int p13 = offset - 2*ancho;
	int p14 = offset - 2*ancho + 1;
	int p15 = offset - 2*ancho + 2;

	int p21 = offset - ancho - 2;
	int p22 = offset - ancho - 1;
	int p23 = offset - ancho;
	int p24 = offset - ancho + 1;
	int p25 = offset - ancho + 2;

	int p31 = offset - 2;
	int p32 = offset - 1;
	int p33 = offset;
	int p34 = offset + 1;
	int p35 = offset + 2;

	int p41 = offset + ancho - 2;
	int p42 = offset + ancho - 1;
	int p43 = offset + ancho;
	int p44 = offset + ancho + 1;
	int p45 = offset + ancho + 2;

	int p51 = offset + 2*ancho - 2;
	int p52 = offset + 2*ancho - 1;
	int p53 = offset + 2*ancho;
	int p54 = offset + 2*ancho + 1;
	int p55 = offset + 2*ancho + 2;

	unsigned char arr[25] = {imgI[p11], imgI[p12], imgI[p13], imgI[p14], imgI[p15],
			imgI[p21], imgI[p22], imgI[p23], imgI[p24], imgI[p25],
			imgI[p31], imgI[p32], imgI[p33], imgI[p34], imgI[p35],
			imgI[p41], imgI[p42], imgI[p43], imgI[p44], imgI[p45],
			imgI[p51], imgI[p52], imgI[p53], imgI[p54], imgI[p55]};

	insertion_sort(arr, 25);
	imgO[offset] = arr[12];
}
/**
 * @brief Suavizar imagen para mejorar deteccion de bordes, utiliza una ventana 3x3
 * 
 * @param char Imagen de entrada
 * @param char Imagen de salida
 * @param ancho Ancho de la imagen
 * @param alto Alto de la imagen
 */
/*__global__ void kernel_Suavizar(unsigned char *imgI, unsigned char *imgO, int ancho, int alto )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (x <= 0 || x >= ancho-1 || y <= 0 || y >= alto-1)
	{
		imgO[offset] = imgI[offset];
		return;
	}

	//-----------------Pixeles Vecinos--------------------
	int leftTop = offset - ancho - 1;
	int top = offset - ancho;
	int rightTop = offset - ancho + 1;

	int left = offset - 1;
	int right = offset + 1;


	int leftBottom = offset + ancho - 1;
	int bottom = offset + ancho;
	int rightBottom = offset + ancho + 1;

	//--------------------Suavisar----------------------
	imgO[offset] = (imgI[leftTop] + imgI[top] + imgI[rightTop] +
			imgI[left] + imgI[offset] + imgI[right]  +
			imgI[leftBottom] + imgI[bottom] + imgI[rightBottom])/9;
}*/
/**
 * @brief Suavizar imagen utilizando un filtro gaussiano para mejorar deteccion de bordes, utiliza una ventana 5x5
 * 
 * @param char Imagen de entrada
 * @param char Imagen de salida
 * @param ancho Ancho de la imagen
 * @param alto Alto de la imagen
 */
__global__ void kernel_Gaussiano(unsigned char *imgI, unsigned char *imgO, int ancho, int alto)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (x <= 1 || x >= ancho-2 || y <= 1 || y >= alto-2)
	{
		imgO[offset] = imgI[offset];
		return;
	}

	//-----------------Pixeles Vecinos--------------------
	int p11 = offset - 2*ancho - 2;
	int p12 = offset - 2*ancho - 1;
	int p13 = offset - 2*ancho;
	int p14 = offset - 2*ancho + 1;
	int p15 = offset - 2*ancho + 2;

	int p21 = offset - ancho - 2;
	int p22 = offset - ancho - 1;
	int p23 = offset - ancho;
	int p24 = offset - ancho + 1;
	int p25 = offset - ancho + 2;

	int p31 = offset - 2;
	int p32 = offset - 1;
	int p33 = offset;
	int p34 = offset + 1;
	int p35 = offset + 2;

	int p41 = offset + ancho - 2;
	int p42 = offset + ancho - 1;
	int p43 = offset + ancho;
	int p44 = offset + ancho + 1;
	int p45 = offset + ancho + 2;

	int p51 = offset + 2*ancho - 2;
	int p52 = offset + 2*ancho - 1;
	int p53 = offset + 2*ancho;
	int p54 = offset + 2*ancho + 1;
	int p55 = offset + 2*ancho + 2;


	//-----------------Borde-----------------
	imgO[offset] = (2*imgI[p11] + 4*imgI[p12]  + 5*imgI[p13]  + 4*imgI[p14]  + 2*imgI[p15] +
			4*imgI[p21] + 9*imgI[p22]  + 12*imgI[p23] + 9*imgI[p24]  + 4*imgI[p25] +
			5*imgI[p31] + 12*imgI[p32] + 15*imgI[p33] + 12*imgI[p34] + 5*imgI[p35] +
			4*imgI[p41] + 9*imgI[p42]  + 12*imgI[p43] + 9*imgI[p44]  + 4*imgI[p45] +
			2*imgI[p51] + 4*imgI[p52]  + 5*imgI[p53]  + 4*imgI[p54]  + 2*imgI[p55] )/159;
}


/**
 * @brief Binarizar imagen utilizando un umbral
 * 
 * @param char Imagen
 * @param threshold umbral
 */
__global__ void kernel_Binarizar(unsigned char *img,int threshold)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	//----------------Binarizar----------------
	if(img[offset]>threshold)
		img[offset]=255;
	else
		img[offset] =0;

}

/**
 * @brief Deteccion de bordes utilizando Sobel
 * 
 * @param char Imagen de entrada
 * @param char Imagen de salida
 * @param ancho Ancho de la imagen
 * @param alto Alto de la imagen
 */
/*__global__ void kernel_Sobel(unsigned char *imgI, unsigned char *imgO, int ancho, int alto )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (x <= 0 || x >= ancho-1 || y <= 0 || y >= alto-1)
	{
		imgO[offset] = 0;
		return;
	}

	//-----------------Pixeles Vecinos--------------------
	int leftTop = offset - ancho - 1;
	int top = offset - ancho;
	int rightTop = offset - ancho + 1;

	int left = offset - 1;
	int right = offset + 1;


	int leftBottom = offset + ancho - 1;
	int bottom = offset + ancho;
	int rightBottom = offset + ancho + 1;


	//-----------------Por formula-----------------
	int auxInte1 = (imgI[leftTop] - imgI[rightTop] +
			2*imgI[left] - 2*imgI[right] +
			imgI[leftBottom] - imgI[rightBottom])/4;


	int auxInte2 = (-imgI[leftTop]+ -2*imgI[top] - imgI[rightTop] +
			imgI[leftBottom] + 2*imgI[bottom] + imgI[rightBottom])/4;

	auxInte1 =  sqrtf((auxInte1*auxInte1)+(auxInte2*auxInte2))  ;

	if(auxInte1>255 )
		imgO[offset] = 255;
	else if(auxInte1<0 )
		imgO[offset] = 0;
	else
		imgO[offset] = auxInte1;
}*/

/**
 * @brief Deteccion de bordes utilizando Laplace
 * @details [long description]
 * 
 * @param char Imagen de entrada
 * @param char Imagen de salida
 * @param ancho Ancho de la imagen
 * @param alto Alto de la imagen
 */
__global__ void kernel_Laplace(unsigned char *imgI, unsigned char *imgO, int ancho, int alto)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if (x <= 0 || x >= ancho-1 || y <= 0 || y >= alto-1)
	{
		imgO[offset] = imgI[offset];
		return;
	}

	//-----------------Pixeles Vecinos--------------------

	int top = offset - ancho;
	int left = offset - 1;
	int right = offset + 1;
	int bottom = offset + ancho;


	//-----------------Borde-----------------
	imgO[offset] = imgI[top] + imgI[left] + imgI[right] + imgI[bottom] -4*imgI[offset];
}
__global__ void kernel_ebu(unsigned char *imgI, unsigned char *imgO, int ancho, int alto)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * blockDim.x * gridDim.x;

		if (x <= 1 || x >= ancho-2 || y <= 1 || y >= alto-2)
		{
			imgO[offset] = imgI[offset];
			return;
		}

		//-----------------Pixeles Vecinos--------------------
		int p11 = offset - 2*ancho - 2;
		int p12 = offset - 2*ancho - 1;
		int p13 = offset - 2*ancho;
		int p14 = offset - 2*ancho + 1;
		int p15 = offset - 2*ancho + 2;

		int p21 = offset - ancho - 2;
		int p22 = offset - ancho - 1;
		int p23 = offset - ancho;
		int p24 = offset - ancho + 1;
		int p25 = offset - ancho + 2;

		int p31 = offset - 2;
		int p32 = offset - 1;
		int p33 = offset;
		int p34 = offset + 1;
		int p35 = offset + 2;

		int p41 = offset + ancho - 2;
		int p42 = offset + ancho - 1;
		int p43 = offset + ancho;
		int p44 = offset + ancho + 1;
		int p45 = offset + ancho + 2;

		int p51 = offset + 2*ancho - 2;
		int p52 = offset + 2*ancho - 1;
		int p53 = offset + 2*ancho;
		int p54 = offset + 2*ancho + 1;
		int p55 = offset + 2*ancho + 2;


		//-----------------Borde-----------------
		imgO[offset] = (imgI[p11] + imgI[p12]  + imgI[p13]  + imgI[p14]  + imgI[p15] +
				imgI[p21] + imgI[p22]  + imgI[p23] + imgI[p24]  + imgI[p25] +
				imgI[p31] + imgI[p32]  + imgI[p33] + imgI[p34]  + imgI[p35] +
				imgI[p41] + imgI[p42]  + imgI[p43] + imgI[p44]  + imgI[p45] +
				imgI[p51] + imgI[p52]  + imgI[p53] + imgI[p54]  + imgI[p55] )/25;
}

/**
 * @brief En cada pixel que sea borde, lo usa de centro, y luego hace un circulo
 * 
 * @param char Imagen de entrada
 * @param char Imagen de salida
 * @param int Imagen con circulos
 * @param radio Radio del circulo
 * @param ancho Ancho de la imagen
 * @param alto Alto de la imagen
 */
__global__ void kernel_Circulos(unsigned char *imgI, unsigned char *imgO, unsigned int *circulos, int radio, int ancho, int alto )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	imgO[offset] = 0;

	int limite = radio;
	if (x < limite || x+limite > ancho || y < limite || y+limite > alto)
		return;
	int xAux, yAux;

	if(imgI[offset]>128)
	{
		for(int angulo=0; angulo<360; angulo++){
			xAux = (x) + (radio)* cosf(angulo);
			yAux = (y) + (radio)* sinf (angulo);
			if(0<=xAux && xAux<ancho && 0<=yAux && yAux<alto)
			{
				imgO[xAux+yAux*ancho] += 1;
				atomicAdd(&circulos[xAux+yAux*ancho], 1);
			}

			xAux = (x) + (radio+1)* cosf(angulo);
			yAux = (y) + (radio+1)* sinf (angulo);
			if(0<=xAux && xAux<ancho && 0<=yAux && yAux<alto)
			{
				imgO[xAux+yAux*ancho] += 1;
				atomicAdd(&circulos[xAux+yAux*ancho], 1);
			}
		}
	}
}
/**
 * @brief Inicializa el buffer de circulos a 0
 * 
 * @param int Imagen buffer de circulos
 */
__global__ void kernel_Iniciar(unsigned int *circulos)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	circulos[offset] = 0;

}

int main()
{
	int BLOCK_SIZE=16;
	int filasY;
	int columnasX;
	int anchoImgCam, altoImgCam;

	int circleU = 120;
	int threshold = 70;

	//Vericacndo que la camara funcione
	VideoCapture cap;
	cap.open(0);
	if(!cap.isOpened())
	{
		printf("No hay camara web");
		return -1;
	}

	int maxIntensidad = 0; //maxima coincidencia
	int posxMax, posyMax, posxAux, posyAux;//posiciones de donde se encuentra la maxima intensidad

	Mat imagenColor, imagenGris, imgBola;//fotmato de imagen que usa opencv

	cap >> imagenColor;//captura de la camara web

	//dimensiones de la imagen capturada por la camara
	altoImgCam=imagenColor.rows;
	anchoImgCam=imagenColor.cols;
	filasY=altoImgCam;
	columnasX=anchoImgCam;

	int inicioX=0, finX=anchoImgCam, inicioY=0, finY=altoImgCam;

	unsigned char *gpu_imgGris1, *gpu_imgGris2, *gpu_imgColor;
	unsigned char *cpu_imgGris=(unsigned char*)malloc(sizeof(unsigned char)*altoImgCam*anchoImgCam);
	unsigned char *cpu_imgColor=(unsigned char*)malloc(sizeof(unsigned char)*altoImgCam*anchoImgCam*3);
	cudaMalloc( (void**)&gpu_imgGris1, altoImgCam*anchoImgCam );
	cudaMalloc( (void**)&gpu_imgGris2, altoImgCam*anchoImgCam );
	cudaMalloc( (void**)&gpu_imgColor, altoImgCam*anchoImgCam*3 );

	unsigned int *gpu_circulos;
	cudaMalloc( (void**)&gpu_circulos,altoImgCam*anchoImgCam * sizeof( int ) );
	cudaMemset( gpu_circulos, 0, altoImgCam*anchoImgCam * sizeof( int ) );

	unsigned int *cpu_circulos=(unsigned int*)malloc(sizeof(unsigned int)*altoImgCam*anchoImgCam);

	imagenGris = Mat::zeros(altoImgCam, anchoImgCam, CV_8UC1);

	int radio = anchoImgCam/40;//iris

	char tecla;

	std::vector<Rect> faces;
	Mat frame_gray;
	if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	int i;
	while(1)
	{
		filasY=altoImgCam;
		columnasX=anchoImgCam;

		cap >> imagenColor;//captura de imagen de la camara web

		cvtColor( imagenColor, imagenGris, CV_BGR2GRAY );
		face_cascade.detectMultiScale( imagenGris, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(150, 150) );

		if(faces.size()>=1)
		{
			Point center( faces[0].x + faces[0].width*0.5, faces[0].y + faces[0].height*0.5 );
			rectangle(imagenColor,Point(faces[0].x,faces[0].y),Point(faces[0].x+faces[0].width,faces[0].y+faces[0].height),Scalar(255,0,255),4);

			inicioX = faces[0].x;
			finX = faces[0].x + faces[0].width/2;

			inicioY = faces[0].y;
			finY = faces[0].y + faces[0].height;
		}else{
			inicioX=0;
			finX=columnasX;
			inicioY=0;
			finY=filasY;
		}

		filasY=finY-inicioY;
		columnasX=finX-inicioX;

		/* Grid multiplo de BLOCK_SIZE */
		filasY = ceil(filasY/BLOCK_SIZE)*BLOCK_SIZE;
		columnasX = ceil(columnasX/BLOCK_SIZE)*BLOCK_SIZE;

		/* Actualizamos finales para mandar bloque completo */
		finY = inicioY + filasY;
		finX = inicioX + columnasX;

		i = 0;
		//para colocar los datos de Mat a un buffer(array), para el GPU
		for(int y=inicioY; y<finY; y++)
		{
			for(int x=inicioX; x<finX; x++)
			{	//insertando cada uno de los colores
				cpu_imgGris[i] = imagenGris.at<uchar>(y,x);
				i++;
			}
		}

		cudaMemcpy( gpu_imgGris1, cpu_imgGris, filasY*columnasX, cudaMemcpyHostToDevice );
		dim3 grids(columnasX/BLOCK_SIZE,filasY/BLOCK_SIZE);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		//kernel_Media2<<<grids,threads>>>(gpu_imgGris1, gpu_imgGris2, columnasX, filasY);
		//kernel_Media2<<<grids,threads>>>(gpu_imgGris2, gpu_imgGris1, columnasX, filasY);
		kernel_Gaussiano<<<grids,threads>>>(gpu_imgGris1, gpu_imgGris2, columnasX, filasY);
		//kernel_Gaussiano<<<grids,threads>>>(gpu_imgGris2, gpu_imgGris1, columnasX, filasY);
		kernel_Binarizar<<<grids,threads>>>(gpu_imgGris2,threshold);
		//kernel_Media2<<<grids,threads>>>(gpu_imgGris1, gpu_imgGris2, columnasX, filasY);
		kernel_ebu<<<grids,threads>>>(gpu_imgGris2, gpu_imgGris1, columnasX, filasY);
		//kernel_Sobel<<<grids,threads>>>(gpu_imgGris1, gpu_imgGris2, columnasX, filasY);
		kernel_Laplace<<<grids,threads>>>(gpu_imgGris1, gpu_imgGris2, columnasX, filasY);
		kernel_Iniciar<<<grids,threads>>>(gpu_circulos);
		kernel_Circulos<<<grids,threads>>>(gpu_imgGris2, gpu_imgGris1, gpu_circulos, radio, columnasX, filasY);

		cudaMemcpy( cpu_circulos, gpu_circulos, filasY*columnasX * sizeof( int ) , cudaMemcpyDeviceToHost );
		cudaMemcpy( cpu_imgGris, gpu_imgGris2, filasY*columnasX  , cudaMemcpyDeviceToHost );

		i = 0;
		//mara mostrar lo que se hizo en el GPU, convertir del buffer a Mat (pero no es necesario tenerlo)
		for(int y=0; y<filasY; y++)
		{
			for(int x=0; x<columnasX; x++)
			{
				imagenGris.at<uchar>(y+inicioY,x+inicioX) = cpu_imgGris[i];
				i ++;
			}
		}
		imshow( "GPU_1", imagenGris );

		i = 0;
		//mara mostrar lo que se hizo en el GPU, convertir del buffer a Mat (pero no es necesario tenerlo)
		for(int y=0; y<filasY; y++)
		{
			for(int x=0; x<columnasX; x++)
			{
				if(cpu_circulos[i]>255)
					imagenGris.at<uchar>(y+inicioY,x+inicioX) = 255;
				else
					imagenGris.at<uchar>(y+inicioY,x+inicioX) = (char)cpu_circulos[i];
				i ++;
			}
		}
		imshow( "Circulos_GPU_1", imagenGris );

		//se busca donde tenga mas coicidencia, el valor maximo, con sus posiciones respectivas
		maxIntensidad = 0;
		for(int pos=(filasY * columnasX)-1; pos>=0; pos--)
		{
			if(cpu_circulos[pos]>=maxIntensidad)
			{
				posxAux = pos%columnasX;
				posyAux = pos/columnasX;
				if(radio<posxAux && posxAux<columnasX-radio && radio<posyAux && posyAux<filasY-radio)
				{
					maxIntensidad = cpu_circulos[pos];
					posxMax = posxAux;
					posyMax = posyAux;
				}
			}
		}
		/*for(int pos=(filasY * columnasX)-1; pos>=0; pos--)
		{
			if(cpu_circulos[pos]>circleU)
			{
				posxAux = pos%columnasX;
				posyAux = pos/columnasX;
				if(radio<posxAux && posxAux<columnasX-radio && radio<posyAux && posyAux<filasY-radio)
				{
					circle( imagenColor, Point(inicioX+posxAux,inicioY+posyAux), radio, Scalar(255*cpu_circulos[pos]/maxIntensidad,255*cpu_circulos[pos]/maxIntensidad,255*cpu_circulos[pos]/maxIntensidad), -1, 8, 0 );
				}
			}
		}*/
		//pintando el punto y circulo en la imagen original
		Point center2(posxMax+inicioX, posyMax+inicioY);
		circle( imagenColor, center2, 3, Scalar(0,255,0), -1, 8, 0 );
		circle( imagenColor, center2, radio, Scalar(0,0,255), 3, 8, 0 );

		imshow( "Final", imagenColor );

		tecla = waitKey(1);
		if(tecla == 27)//escape
			break;
		if(tecla == '+' )
		{
			radio++;
			printf("radio %d\n",radio);
		}else if(tecla == '-')
		{
			radio--;
			printf("radio %d\n",radio);
		}else if(tecla == 'w')
		{
			circleU++;
			printf("circleU %d\n",circleU);
		}
		else if(tecla == 's')
		{
			circleU--;
			printf("circleU %d\n",circleU);
		}else if(tecla == 'a' )
		{
			threshold++;
			printf("threshold %d\n",threshold);
		}else if(tecla == 'd')
		{
			threshold--;
			printf("threshold %d\n",threshold);
		}
	}
	cudaFree( gpu_imgGris1);
	cudaFree( gpu_imgGris2);
	cudaFree( gpu_imgColor);
	cudaFree( gpu_circulos);

	return 0;
}


