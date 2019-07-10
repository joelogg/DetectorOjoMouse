/*#include <cv.h>
#include <highgui.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;



int main()
{
	VideoCapture cap(0);

	if(!cap.isOpened())
	{
		printf("No hay camara web");
		return -1;
	}

	//namedWindow("webcam");


	Mat imagen, gray_image, blur_image, binaria_image, borde_image;
	cap >> imagen;
	int filasY=imagen.rows;
	int columnasX=imagen.cols;

	//imshow( "Original", imagen );
	int radio = columnasX/8;//iris
	//float menorR = 0.8, mayorR = 1.05;
	int menorR = 10, mayorR = 10;
	while(1)
	{
		cap >> imagen;
		//imshow( "Origal", imagen );
		//cvtColor( imagen, gray_image, CV_BGR2GRAY );
		gray_image = Mat::zeros(filasY, columnasX, CV_8UC1);
		for(int y=0; y<filasY; y++)
		{
			for(int x=0; x<columnasX; x++)
			{
				gray_image.at<uchar>(y,x) = (imagen.at<Vec3b>(y,x)[1]+imagen.at<Vec3b>(y,x)[2])/2;
				//gray_image.at<uchar>(y,x) = (imagen.at<Vec3b>(y,x)[0]+ imagen.at<Vec3b>(y,x)[1]+imagen.at<Vec3b>(y,x)[2])/3;
			}
		}
		//imshow( "Gray1", gray_image );

		//-------suavisar-------
		//GaussianBlur(gray_image,blur_image,cv::Size(7,7),1.5,1.5);
		blur_image = Mat::zeros(filasY, columnasX, CV_8UC1);
		int filasYAux = filasY-1, columnasXAux = columnasX-1;
		for(int y=1; y<filasYAux; y++)
		{
			for(int x=1; x<columnasXAux; x++)
			{
				blur_image.at<uchar>(y,x) = (gray_image.at<uchar>(y-1,x-1)+
						gray_image.at<uchar>(y-1,x) + gray_image.at<uchar>(y-1,x+1) +
						gray_image.at<uchar>(y,x-1) + gray_image.at<uchar>(y,x) +
						gray_image.at<uchar>(y,x+1) + gray_image.at<uchar>(y+1,x-1) +
						gray_image.at<uchar>(y+1,x) + gray_image.at<uchar>(y+1,x+1))/9;
			}
		}
		for(int y=0; y<filasY; y++)
		{
			blur_image.at<uchar>(y, 0) = gray_image.at<uchar>(y, 0);
			blur_image.at<uchar>(y, columnasXAux) = gray_image.at<uchar>(y, columnasXAux);

		}
		for(int x=1; x<columnasX; x++)
		{
			blur_image.at<uchar>(0,x) = gray_image.at<uchar>(0,x);;
			blur_image.at<uchar>(filasYAux,x) = gray_image.at<uchar>(filasYAux,x);;
		}
		//imshow( "blur", blur_image );
		//Canny(blur_image,borde_image,50,100,3);
		//imshow( "Borde", borde_image );

		//-----Binarizar----
		//threshold(blur_image, binaria_image, 50, 255, THRESH_BINARY);
		binaria_image = Mat::zeros(filasY, columnasX, CV_8UC1);
		for(int y=0; y<filasY; y++)
		{
			for(int x=0; x<columnasX; x++)
			{
				if(blur_image.at<uchar>(y,x)>70)
					binaria_image.at<uchar>(y,x) = 255;
				else
					binaria_image.at<uchar>(y,x) = 0;
			}
		}
		//imshow( "Binaria", binaria_image );

		Canny(binaria_image,borde_image,0,30,3);
		borde_image = Mat::zeros(filasY, columnasX, CV_8UC1);
        	int auxInte = 0;
        	for(int y=1; y<filasYAux; y++)
        	{
        		for(int x=1; x<columnasXAux; x++)
        		{
        			auxInte = (1*binaria_image.at<uchar>(y-1,x-1)+
        					0*binaria_image.at<uchar>(y-1,x) - 1*binaria_image.at<uchar>(y-1,x+1) +
        					2*binaria_image.at<uchar>(y,x-1) + 0*binaria_image.at<uchar>(y,x) -
        					2*binaria_image.at<uchar>(y,x+1) + 1*binaria_image.at<uchar>(y+1,x-1) +
        					0*binaria_image.at<uchar>(y+1,x) - 1*binaria_image.at<uchar>(y+1,x+1))/4;
        			borde_image.at<uchar>(y,x) = auxInte;
        			if(auxInte>255)
        				borde_image.at<uchar>(y,x) = 255;
        			else if(auxInte<0)
        				borde_image.at<uchar>(y,x) = 0;



        			auxInte = (-1*binaria_image.at<uchar>(y-1,x-1)+
        					-2*binaria_image.at<uchar>(y-1,x) - 1*binaria_image.at<uchar>(y-1,x+1) +
        					0*binaria_image.at<uchar>(y,x-1) + 0*binaria_image.at<uchar>(y,x) +
        					0*binaria_image.at<uchar>(y,x+1) + 1*binaria_image.at<uchar>(y+1,x-1) +
        					2*binaria_image.at<uchar>(y+1,x) + 1*binaria_image.at<uchar>(y+1,x+1))/4;
        			borde_image.at<uchar>(y,x) = auxInte;
        			if(auxInte>255)
        				borde_image.at<uchar>(y,x) = 255;
        			else if(auxInte<0)
        				borde_image.at<uchar>(y,x) = 0;
        		}
        	}
		//imshow( "Borde", borde_image );


		int xAux, yAux;
		//-----------------------------menor-----------------------------------
		Mat circulos1 = Mat::zeros(filasY, columnasX, CV_8UC1);
		for(int y=0; y<filasY; y++)
		{
			for(int x=0; x<columnasX; x++)
			{
				if(borde_image.at<uchar>(y,x)>128)
				{
					for(int angulo=0; angulo<360; angulo++)
					{
						xAux = x + (radio-menorR)*cos(angulo);
						yAux = y + (radio-menorR)*sin(angulo);
						if(0<=xAux && xAux<columnasX && 0<=yAux && yAux<filasY && circulos1.at<uchar>(yAux,xAux)<254)
							circulos1.at<uchar>(yAux,xAux) += 1;
					}
				}
			}
		}
		//imshow( "Circulos1", circulos1 );

		int maxIntensidad1 = 0;
		int posxMax1, posyMax1;
		for(int y=0; y<filasY; y++)
		{
			for(int x=0; x<columnasX; x++)
			{
				if(circulos1.at<uchar>(y,x)>maxIntensidad1)
				{
					maxIntensidad1 = circulos1.at<uchar>(y,x);
					posxMax1 = x;
					posyMax1 = y;
				}
			}
		}

		//------------------------------real----------------------------------

		Mat circulos2 = Mat::zeros(filasY, columnasX, CV_8UC1);
		for(int y=0; y<filasY; y++)
		{
			for(int x=0; x<columnasX; x++)
			{
				if(borde_image.at<uchar>(y,x)>128)
				{
					for(int angulo=0; angulo<360; angulo++)
					{
						xAux = x + (radio+mayorR)*cos(angulo);
						yAux = y + (radio+mayorR)*sin(angulo);
						if(0<=xAux && xAux<columnasX && 0<=yAux && yAux<filasY && circulos2.at<uchar>(yAux,xAux)<254)
							circulos2.at<uchar>(yAux,xAux) += 1;
					}
				}
			}
		}
		//imshow( "Circulos2", circulos2 );

		int maxIntensidad2 = 0;
		int posxMax2, posyMax2;
		for(int y=0; y<filasY; y++)
		{
			for(int x=0; x<columnasX; x++)
			{
				if(circulos2.at<uchar>(y,x)>maxIntensidad2)
				{
					maxIntensidad2 = circulos2.at<uchar>(y,x);
					posxMax2 = x;
					posyMax2 = y;
				}
			}
		}
		//---------------------------------Mayor-----------------
		Mat circulos3 = Mat::zeros(filasY, columnasX, CV_8UC1);
		for(int y=0; y<filasY; y++)
		{
			for(int x=0; x<columnasX; x++)
			{
				if(borde_image.at<uchar>(y,x)>128)
				{
					for(int angulo=0; angulo<360; angulo++)
					{
						xAux = x + (radio+10)*cos(angulo);
						yAux = y + (radio+10)*sin(angulo);
						if(0<=xAux && xAux<columnasX && 0<=yAux && yAux<filasY && circulos3.at<uchar>(yAux,xAux)<254)
							circulos3.at<uchar>(yAux,xAux) += 1;
					}
				}
			}
		}
		//imshow( "Circulos3", circulos3 );

		int maxIntensidad3 = 0;
		int posxMax3, posyMax3;
		for(int y=0; y<filasY; y++)
		{
			for(int x=0; x<columnasX; x++)
			{
				if(circulos3.at<uchar>(y,x)>maxIntensidad3)
				{
					maxIntensidad3 = circulos3.at<uchar>(y,x);
					posxMax3 = x;
					posyMax3 = y;
				}
			}
		}


		Mat imagen2 = imagen.clone();
		Mat imagen3 = imagen.clone();
		Mat imagenFinal = imagen.clone();

		/*Point center1(posxMax1, posyMax1);
		circle( imagen, center1, 3, Scalar(0,255,0), -1, 8, 0 );
		circle( imagen, center1, radio-10, Scalar(0,0,255), 3, 8, 0 );
		//imshow( "Centro1", imagen );

		Point center2(posxMax2, posyMax2);
		circle( imagen2, center2, 3, Scalar(0,255,0), -1, 8, 0 );
		circle( imagen2, center2, radio, Scalar(0,0,255), 3, 8, 0 );
		//imshow( "Centro2", imagen2 );

		Point center3(posxMax3, posyMax3);
		circle( imagen3, center3, 3, Scalar(0,255,0), -1, 8, 0 );
		circle( imagen3, center3, radio+10, Scalar(0,0,255), 3, 8, 0 );*/
		//imshow( "Centro3", imagen3 );

/*		int radioAux = radio;
		if(maxIntensidad2>=maxIntensidad1 && maxIntensidad2>=maxIntensidad3)
		{
			posxMax1 = posxMax2;
			posyMax1 = posyMax2;

		}
		else if(maxIntensidad3>=maxIntensidad2 && maxIntensidad3>=maxIntensidad1)
		{
			posxMax1 = posxMax3;
			posyMax1 = posyMax3;
			radioAux += mayorR;
		}
		else
		{
			radioAux -= menorR;
		}

		Point center(posxMax1, posyMax1);
		if(3<=posxMax1 && posxMax1< columnasX-3 && 3<=posyMax1 && posyMax1< filasY-3)
			circle( imagenFinal, center, 3, Scalar(0,255,0), -1, 8, 0 );
		if(radioAux<=posxMax1 && posxMax1< columnasX-radioAux && radioAux<=posyMax1 && posyMax1< filasY-radioAux)
			circle( imagenFinal, center, radio, Scalar(0,0,255), 3, 8, 0 );
		maxIntensidad1 = 0;
		maxIntensidad2 = 0;
		maxIntensidad3 = 0;

		imshow( "Final", imagenFinal );
		if(waitKey(50) >= 0) break;
	}

	return 0;
}
*/
