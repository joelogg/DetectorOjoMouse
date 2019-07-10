/*#include <GL/glut.h>
#include <stdio.h>
#include <cuda.h>

#define DIM 500

__global__ void kernel( unsigned char *ptr, int ticks )
{
	// map from threadIdx/BlockIdx to pixel position
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	// now calculate the value at that position
	float fx = x - DIM/2;
	float fy = y - DIM/2;
	float d = sqrtf( fx * fx + fy * fy );
	float fact=16.0f;
	if(d<fact)
		fact=d;
	unsigned char color = (unsigned char)((255.0f * cos(d/2.0f - ticks/8.0f))/(d/fact));
	ptr[offset*4 + 0] = 0;
	ptr[offset*4 + 1] = color*0.88f;
	ptr[offset*4 + 2] = color*0.92f;
	ptr[offset*4 + 3] = 255;
}

void display_cb()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glColor3f(1,1,0);

	unsigned char *gpu_bitmap;
	unsigned char *cpu_bitmap=(unsigned char*)malloc(sizeof(unsigned char)*DIM*DIM*4);
	cudaMalloc( (void**)&gpu_bitmap, DIM*DIM*4 );


	dim3 blocks(DIM/16,DIM/16);
	dim3 threads(16,16);

	int ticks =  0;
	while(ticks<50)
	{
		kernel<<<blocks,threads>>>( gpu_bitmap, ticks );
		ticks++;

		cudaMemcpy( cpu_bitmap, gpu_bitmap,DIM*DIM*4 , cudaMemcpyDeviceToHost );

		//visualize cpu_bitmap
		glBegin(GL_POINTS);
		for(int x=0; x<DIM; x++)
		{
			for(int y=0; y<DIM; y++)
			{
				int offset = x + y * DIM;

				glColor3f((cpu_bitmap [offset*4 + 0]/255.0f),
						(cpu_bitmap [offset*4 + 1]/255.0f),
						(cpu_bitmap [offset*4 + 1]/255.0f));
				glVertex2f(x,y);
			}
		}
		glEnd();
		glutSwapBuffers();

	}

	cudaFree( gpu_bitmap );
	glutPostRedisplay();

}

void reshape_cb (int w, int h)
{
	if (w==0||h==0) return;
	glViewport(0,0,w,h);
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();
	gluOrtho2D(0,w,0,h);
	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity ();
}



void initialize()
{
	glutInitDisplayMode (GLUT_RGBA|GLUT_DOUBLE);
	glutInitWindowSize (DIM,DIM);
	glutInitWindowPosition (100,100);
	glutCreateWindow ("Ventana OpenGL");
	glutDisplayFunc (display_cb);
	glutReshapeFunc (reshape_cb);
	glClearColor(0.f,0.f,0.f,1.f);
}

int main (int argc, char **argv)
{
	glutInit (&argc, argv);
	initialize();
	glutMainLoop();
	return 0;
}

*/
