#include <GL/glew.h>
#include <GL/freeglut.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "complexRange.h"
#include "mandel_GPU.cuh"


CRange range;

dim3 windowSize(1536,769);

static GLuint pbo_buffer = 0;
struct cudaGraphicsResource *cuda_pbo_resource;

/**
 * Bind OpenGL Buffer to CUDA
 */
void initGL(void) {
    // Generate OpenGL Pixel Buffer Object
    glGenBuffers(1, &pbo_buffer);
    // Bind this buffer to set the correct state
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo_buffer);
    // Allocate memory
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, sizeof(uchar4) * windowSize.x * windowSize.y, 0, GL_DYNAMIC_DRAW_ARB);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard));
}

void display(void) {
    uchar4 *devimg = NULL;
    size_t num_bytes;

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, NULL));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&devimg, &num_bytes, cuda_pbo_resource));

    if(num_bytes != windowSize.x*windowSize.y*sizeof(uchar4)) {
        printf("WRONG SIZE!!!\n");
        exit(-1);
    }

    render(devimg, range, 255);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, NULL));

    glDrawPixels(windowSize.x, windowSize.y, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glutSwapBuffers();
}

void destroyGL(void) {
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

    glDeleteBuffers(1, &pbo_buffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    float stepsize = 0.005;
    switch (key) {
        case 27:
        case 'q':
        case 'Q':
            destroyGL();
            glutDestroyWindow(glutGetWindow());
            break;
        case 'd':   // move left
            range.xmin += stepsize;
            break;
        case 'a':   // move right
            range.xmin -= stepsize;
            break;
        case 'w':   // move up
            range.ymin += stepsize;
            break;
        case 's':   // move down
            range.ymin -= stepsize;
            break;
        case '+':   // zoom in
            range.xstep *= 0.99;
            range.ystep *= 0.99;
            break;
        case '-':   // zoom out
            range.xstep *= 1.01;
            range.ystep *= 1.01;
            break;
        case 'r':   // reset
            range.set(-2.0, -1.0, 2.0, 1.0, windowSize);
            break;
        default:
            break;
    }
}

/**
 * Initialize window
 */
int initGlutDisplay(int argc, char* argv[]) {
    // Init window
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGB);
    glutInitWindowSize(windowSize.x, windowSize.y);
    glutInitWindowPosition(100, 100);
    glutCreateWindow("Jeah OpenGL");

    // Set perspective
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);

    // Initialize memory for the image
    glewInit();

    // Link OpenGL and CUDA
    initGL();

    // Calculate Grid Size
    calcGrid(windowSize);
    range.set(-2.0, -1.0, 2.0, 1.0, windowSize);

    // Initial drawing
    glutDisplayFunc(display);
    // Refresh image
    glutIdleFunc(display);
    // Set keyboard bindings
    glutKeyboardFunc(keyboard);

    glutMainLoop();

    return 0;
}

int main(int argc, char* argv[]) {
    initGlutDisplay(argc, argv);

    return 0;
}
