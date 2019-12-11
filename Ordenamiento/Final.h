#pragma once

#include <optixu/optixu_math_namespace.h>                                        

struct ParallelogramLight
{
    optix::float3 corner;
    optix::float3 v1, v2;
    optix::float3 normal;
    optix::float3 emission;
};


optix::Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext();
void loadGeometry();
void setupCamera();
void updateCamera();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);