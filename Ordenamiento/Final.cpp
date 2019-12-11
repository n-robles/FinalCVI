#include <GL/glew.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include "Final.h"
#include <sutil.h>
#include <Arcball.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <string>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <cstdlib>
#include <iomanip>

using namespace optix;

const char* const SAMPLE_NAME = "optixDenoiser";
constexpr const char* APP_NAME = "ProyectoFinal";


Context        context = nullptr;
int            width = 512;
int            height = 512;
bool           use_pbo = true;

bool           denoiser_perf_mode = false;
int            denoiser_perf_iter = 1;

int            frame_number = 1;
int            sqrt_num_samples = 4;
int            rr_begin_depth = 1;
Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;

// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
bool           camera_changed = true;
bool           postprocessing_needs_init = true;
sutil::Arcball arcball;

// Mouse state
int2           mouse_prev_pos;
int            mouse_button;

// Post-processing
CommandList commandListWithDenoiser;
CommandList commandListWithoutDenoiser;
PostprocessingStage tonemapStage;
PostprocessingStage denoiserStage;
Buffer denoisedBuffer;
Buffer emptyBuffer;
Buffer trainingDataBuffer;

int numNonDenoisedFrames = 5;

float denoiseBlend = 0.45f;

std::string training_file;


void loadTrainingFile(const std::string& path)
{
    if (path.length() == 0)
    {
        trainingDataBuffer->setSize(0);
        return;
    }

    using namespace std;
    ifstream fin(path.c_str(), ios::in | ios::ate | ios::binary);
    if (fin.fail())
    {
        fprintf(stderr, "Failed to load training file %s\n", path.c_str());
        return;
    }
    size_t size = static_cast<size_t>(fin.tellg());

    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp)
    {
        fprintf(stderr, "Failed to load training file %s\n", path.c_str());
        return;
    }

    trainingDataBuffer->setSize(size);

    char* data = reinterpret_cast<char*>(trainingDataBuffer->map());

    const bool ok = fread(data, 1, size, fp) == size;
    fclose(fp);

    trainingDataBuffer->unmap();

    if (!ok)
    {
        fprintf(stderr, "Failed to load training file %s\n", path.c_str());
        trainingDataBuffer->setSize(0);
    }
}


Buffer getOutputBuffer()
{
    return context["output_buffer"]->getBuffer();
}

Buffer getTonemappedBuffer()
{
    return context["tonemapped_buffer"]->getBuffer();
}

Buffer getAlbedoBuffer()
{
    return context["input_albedo_buffer"]->getBuffer();
}

Buffer getNormalBuffer()
{
    return context["input_normal_buffer"]->getBuffer();
}

void destroyContext()
{
    if (context)
    {
        context->destroy();
        context = 0;
    }
}


void registerExitHandler()
{
    glutCloseFunc(destroyContext);
}

void convertNormalsToColors(
    Buffer& normalBuffer)
{
    float* data = reinterpret_cast<float*>(normalBuffer->map());

    RTsize width, height;
    normalBuffer->getSize(width, height);

    RTsize size = width * height;
    for (size_t i = 0; i < size; ++i)
    {
        const float r = *(data + 3 * i);
        const float g = *(data + 3 * i + 1);
        const float b = *(data + 3 * i + 2);

        *(data + 3 * i) = std::abs(r);
        *(data + 3 * i + 1) = std::abs(g);
        *(data + 3 * i + 2) = std::abs(b);
    }

    normalBuffer->unmap();
}


void setMaterial(
    GeometryInstance& gi,
    Material material,
    const std::string& color_name,
    const float3& color)
{
    gi->addMaterial(material);
    gi[color_name]->setFloat(color);
}


GeometryInstance createParallelogram(
    const float3& anchor,
    const float3& offset1,
    const float3& offset2)
{
    Geometry parallelogram = context->createGeometry();
    parallelogram->setPrimitiveCount(1u);
    parallelogram->setIntersectionProgram(pgram_intersection);
    parallelogram->setBoundingBoxProgram(pgram_bounding_box);

    float3 normal = normalize(cross(offset1, offset2));
    float d = dot(normal, anchor);
    float4 plane = make_float4(normal, d);

    float3 v1 = offset1*(1.0 / dot(offset1, offset1));
    float3 v2 = offset2*(1.0 / dot(offset2, offset2));

    parallelogram["plane"]->setFloat(plane);
    parallelogram["anchor"]->setFloat(anchor);
    parallelogram["v1"]->setFloat(v1);
    parallelogram["v2"]->setFloat(v2);

    GeometryInstance gi = context->createGeometryInstance();
    gi->setGeometry(parallelogram);
    return gi;
}

void denoiserReportCallback(int lvl, const char* tag, const char* msg, void* cbdata)
{
    if (std::string("DLDENOISER") == tag)
        std::cout << "[" << std::left << std::setw(12) << tag << "] " << msg;
    else if (std::string("POSTPROCESSING") == tag && denoiser_perf_mode)
        std::cout << "[" << std::left << std::setw(12) << tag << "] " << msg;

}

void createContext()
{
    context = Context::create();
    context->setRayTypeCount(2);
    context->setEntryPointCount(1);
    context->setStackSize(1800);

    context["scene_epsilon"]->setFloat(1.e-3f);
    context["rr_begin_depth"]->setUint(rr_begin_depth);

    context->setUsageReportCallback(denoiserReportCallback, 2, NULL);

    Buffer renderBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["output_buffer"]->set(renderBuffer);
    Buffer tonemappedBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["tonemapped_buffer"]->set(tonemappedBuffer);
    Buffer albedoBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["input_albedo_buffer"]->set(albedoBuffer);

    // The normal buffer use float4 for performance reasons, the fourth channel will be ignored.
    Buffer normalBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    context["input_normal_buffer"]->set(normalBuffer);

    denoisedBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
    emptyBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, 0, 0);
    trainingDataBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, 0);

    // Setup programs
    const char* ptx = sutil::getPtxString(SAMPLE_NAME, "optixDenoiser.cu");
    context->setRayGenerationProgram(0, context->createProgramFromPTXString(ptx, "pathtrace_camera"));
    context->setExceptionProgram(0, context->createProgramFromPTXString(ptx, "exception"));
    context->setMissProgram(0, context->createProgramFromPTXString(ptx, "miss"));

    context["sqrt_num_samples"]->setUint(sqrt_num_samples);
    context["bad_color"]->setFloat(1000000.0f, 0.0f, 1000000.0f);
    context["bg_color"]->setFloat(make_float3(0.0f));
}


void loadGeometry()
{
    // Light buffer
    ParallelogramLight lights[] = {
        {make_float3(343.0f, 840.0f, 227.0f),make_float3(-130.0f, 0.0f, 0.0f),make_float3(0.0f, 0.0f, 105.0f),normalize(cross(make_float3(-130.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 105.0f))),make_float3(340.0f, 190.0f, 100.0f)},
        {make_float3(950.0f, 440.0f, 427.0f),make_float3(0.0f, -130.0f, 0.0f),make_float3(0.0f, 0.0f, 105.0f),normalize(cross(make_float3(0.0f, -130.0f, 0.0f), make_float3(0.0f, 0.0f, 105.0f))),make_float3(950.0f, 440.0f, 427.0f)}
    };

    Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
    light_buffer->setFormat(RT_FORMAT_USER);
    light_buffer->setElementSize(sizeof(ParallelogramLight));
    light_buffer->setSize(2u);
    memcpy(light_buffer->map(), lights, sizeof(lights));
    light_buffer->unmap();
    context["lights"]->setBuffer(light_buffer);


    // Set up material
    Material diffuse = context->createMaterial();
    const char* ptx = sutil::getPtxString(SAMPLE_NAME, "optixDenoiser.cu");
    Program diffuse_ch = context->createProgramFromPTXString(ptx, "diffuse");
    Program diffuse_ah = context->createProgramFromPTXString(ptx, "shadow");
    diffuse->setClosestHitProgram(0, diffuse_ch);
    diffuse->setAnyHitProgram(1, diffuse_ah);

    Material diffuse_light = context->createMaterial();
    Program diffuse_em = context->createProgramFromPTXString(ptx, "diffuseEmitter");
    diffuse_light->setClosestHitProgram(0, diffuse_em);

    // Set up parallelogram programs
    ptx = sutil::getPtxString(SAMPLE_NAME, "parallelogram.cu");
    pgram_bounding_box = context->createProgramFromPTXString(ptx, "bounds");
    pgram_intersection = context->createProgramFromPTXString(ptx, "intersect");

    // create geometry instances
    std::vector<GeometryInstance> gis;

    const float3 white = make_float3(0.8f, 0.8f, 0.8f);
    const float3 green = make_float3(0.05f, 0.8f, 0.05f);
    const float3 red = make_float3(0.8f, 0.05f, 0.05f);
    const float3 blue = make_float3(0.05f, 0.05f, 0.8f);
    const float3 light_em = make_float3(340.0f, 190.0f, 100.0f);
    const float3 light_em1 = make_float3(240.0f, 90.0f, 120.0f);

    // Floor
    gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 950.0f),
        make_float3(950.0f, 0.0f, 0.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    // Ceiling
    gis.push_back(createParallelogram(make_float3(0.0f, 840.0f, 0.0f),
        make_float3(950.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 950.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    // Back wall
    gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 950.0f),
        make_float3(0.0f, 840.0f, 0.0f),
        make_float3(950.0f, 0.0f, 0.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", blue);

    // Right wall
    gis.push_back(createParallelogram(make_float3(0.0f, 0.0f, 0.0f),
        make_float3(0.0f, 840.0f, 0.0f),
        make_float3(0.0f, 0.0f, 950.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", green);

    // Left wall
    gis.push_back(createParallelogram(make_float3(950.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 950.0f),
        make_float3(0.0f, 840.0f, 0.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", red);

    // Short block
    gis.push_back(createParallelogram(make_float3(130.0f, 165.0f, 65.0f),
        make_float3(-48.0f, 0.0f, 160.0f),
        make_float3(160.0f, 0.0f, 49.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(290.0f, 0.0f, 114.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(-50.0f, 0.0f, 158.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(130.0f, 0.0f, 65.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(160.0f, 0.0f, 49.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(82.0f, 0.0f, 225.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(48.0f, 0.0f, -160.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(240.0f, 0.0f, 272.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(-158.0f, 0.0f, -47.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    // Tall block
    gis.push_back(createParallelogram(make_float3(423.0f, 330.0f, 247.0f),
        make_float3(-158.0f, 0.0f, 49.0f),
        make_float3(49.0f, 0.0f, 159.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(423.0f, 0.0f, 247.0f),
        make_float3(0.0f, 330.0f, 0.0f),
        make_float3(49.0f, 0.0f, 159.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(472.0f, 0.0f, 406.0f),
        make_float3(0.0f, 330.0f, 0.0f),
        make_float3(-158.0f, 0.0f, 50.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(314.0f, 0.0f, 456.0f),
        make_float3(0.0f, 330.0f, 0.0f),
        make_float3(-49.0f, 0.0f, -160.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(265.0f, 0.0f, 296.0f),
        make_float3(0.0f, 330.0f, 0.0f),
        make_float3(158.0f, 0.0f, -49.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    gis.push_back(createParallelogram(make_float3(130.0f, 165.0f, 65.0f),
        make_float3(848.0f, 0.0f, 760.0f),
        make_float3(760.0f, 0.0f, 849.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(290.0f, 0.0f, 114.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(850.0f, 0.0f, 758.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(130.0f, 0.0f, 65.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(760.0f, 0.0f, 849.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(82.0f, 0.0f, 225.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(848.0f, 0.0f, 760.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);
    gis.push_back(createParallelogram(make_float3(240.0f, 0.0f, 272.0f),
        make_float3(0.0f, 165.0f, 0.0f),
        make_float3(758.0f, 0.0f, 847.0f)));
    setMaterial(gis.back(), diffuse, "diffuse_color", white);

    // Create shadow group (no light)
    GeometryGroup shadow_group = context->createGeometryGroup(gis.begin(), gis.end());
    shadow_group->setAcceleration(context->createAcceleration("Trbvh"));
    context["top_shadower"]->set(shadow_group);

    // Light
    gis.push_back(createParallelogram(make_float3(343.0f, 840.0f, 227.0f),
        make_float3(-130.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 105.0f)));
    setMaterial(gis.back(), diffuse_light, "emission_color", light_em);

    gis.push_back(createParallelogram(make_float3(843.0f, 840.0f, 427.0f),
        make_float3(130.0f, 0.0f, 0.0f),
        make_float3(0.0f, 0.0f, 105.0f)));
    setMaterial(gis.back(), diffuse_light, "emission_color", light_em1);

    // Create geometry group
    GeometryGroup geometry_group = context->createGeometryGroup(gis.begin(), gis.end());
    geometry_group->setAcceleration(context->createAcceleration("Trbvh"));
    context["top_object"]->set(geometry_group);
}


void setupCamera()
{
    camera_eye = make_float3(478.0f, 273.0f, -100.0f);
    camera_lookat = make_float3(978.0f, 273.0f, 910.0f);
    camera_up = make_float3(0.0f, 1.0f, 0.0f);

    camera_rotate = Matrix4x4::identity();
}


void updateCamera()
{
    const float fov = 35.0f;
    const float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);

    float3 camera_u, camera_v, camera_w;
    sutil::calculateCameraVariables(
        camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
        camera_u, camera_v, camera_w, /*fov_is_vertical*/ true);

    const Matrix4x4 frame = Matrix4x4::fromBasis(
        normalize(camera_u),
        normalize(camera_v),
        normalize(-camera_w),
        camera_lookat);
    const Matrix4x4 frame_inv = frame.inverse();
    // Apply camera rotation twice to match old SDK behavior
    const Matrix4x4 trans = frame * camera_rotate * camera_rotate * frame_inv;

    camera_eye = make_float3(trans * make_float4(camera_eye, 1.0f));
    camera_lookat = make_float3(trans * make_float4(camera_lookat, 1.0f));
    camera_up = make_float3(trans * make_float4(camera_up, 0.0f));

    sutil::calculateCameraVariables(
        camera_eye, camera_lookat, camera_up, fov, aspect_ratio,
        camera_u, camera_v, camera_w, true);

    camera_rotate = Matrix4x4::identity();

    if (camera_changed) // reset accumulation
        frame_number = 1;
    camera_changed = false;

    context["frame_number"]->setUint(frame_number);
    context["eye"]->setFloat(camera_eye);
    context["U"]->setFloat(camera_u);
    context["V"]->setFloat(camera_v);
    context["W"]->setFloat(camera_w);

    const Matrix4x4 current_frame_inv = Matrix4x4::fromBasis(
        normalize(camera_u),
        normalize(camera_v),
        normalize(-camera_w),
        camera_lookat).inverse();
    Matrix3x3 normal_matrix = make_matrix3x3(current_frame_inv);

    context["normal_matrix"]->setMatrix3x3fv(false, normal_matrix.getData());
}


void setupPostprocessing()
{

    if (!tonemapStage)
    {
        tonemapStage = context->createBuiltinPostProcessingStage("TonemapperSimple");
        denoiserStage = context->createBuiltinPostProcessingStage("DLDenoiser");
        if (trainingDataBuffer)
        {
            Variable trainingBuff = denoiserStage->declareVariable("training_data_buffer");
            trainingBuff->set(trainingDataBuffer);
        }

        tonemapStage->declareVariable("input_buffer")->set(getOutputBuffer());
        tonemapStage->declareVariable("output_buffer")->set(getTonemappedBuffer());
        tonemapStage->declareVariable("exposure")->setFloat(0.25f);
        tonemapStage->declareVariable("gamma")->setFloat(2.2f);

        denoiserStage->declareVariable("input_buffer")->set(getTonemappedBuffer());
        denoiserStage->declareVariable("output_buffer")->set(denoisedBuffer);
        denoiserStage->declareVariable("blend")->setFloat(denoiseBlend);
        denoiserStage->declareVariable("input_albedo_buffer");
        denoiserStage->declareVariable("input_normal_buffer");
    }

    if (commandListWithDenoiser)
    {
        commandListWithDenoiser->destroy();
        commandListWithoutDenoiser->destroy();
    }

    // Create two command lists with two postprocessing topologies we want:
    // One with the denoiser stage, one without. Note that both share the same
    // tonemap stage.

    commandListWithDenoiser = context->createCommandList();
    commandListWithDenoiser->appendLaunch(0, width, height);
    commandListWithDenoiser->appendPostprocessingStage(tonemapStage, width, height);
    commandListWithDenoiser->appendPostprocessingStage(denoiserStage, width, height);
    commandListWithDenoiser->finalize();

    commandListWithoutDenoiser = context->createCommandList();
    commandListWithoutDenoiser->appendLaunch(0, width, height);
    commandListWithoutDenoiser->appendPostprocessingStage(tonemapStage, width, height);
    commandListWithoutDenoiser->finalize();

    postprocessing_needs_init = false;
}

void glutInitialize(int* argc, char** argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutInitWindowPosition(100, 100);
    glutCreateWindow(APP_NAME);
    glutHideWindow();
}


void glutRun()
{
    // Initialize GL state                                                            
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    glutShowWindow();
    glutReshapeWindow(width, height);

    // register glut callbacks
    glutDisplayFunc(glutDisplay);
    glutIdleFunc(glutDisplay);
    glutReshapeFunc(glutResize);
    glutKeyboardFunc(glutKeyboardPress);
    glutMouseFunc(glutMousePress);
    glutMotionFunc(glutMouseMotion);

    registerExitHandler();

    glutMainLoop();
}


//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------

void glutDisplay()
{
    updateCamera();

    if (postprocessing_needs_init)
    {
        setupPostprocessing();
    }

    Variable(denoiserStage->queryVariable("blend"))->setFloat(denoiseBlend);

    bool isEarlyFrame = (frame_number <= numNonDenoisedFrames);
    if (isEarlyFrame)
    {
        commandListWithoutDenoiser->execute();
    }
    else
    {
        commandListWithDenoiser->execute();
    }

    
        if (isEarlyFrame)
        {
            sutil::displayBufferGL(getTonemappedBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);
        }
        else
        {
            RTsize trainingSize = 0;
            trainingDataBuffer->getSize(trainingSize);            
            sutil::displayBufferGL(denoisedBuffer, BUFFER_PIXEL_FORMAT_DEFAULT, true);
        }

    

    {
        static unsigned frame_count = 0;
        sutil::displayFps(frame_count++);
    }

    char str[64];
    sprintf(str, "#%d", frame_number);
    sutil::displayText(str, (float)width - 50, (float)height - 20);

    frame_number++;

    glutSwapBuffers();
}


void glutKeyboardPress(unsigned char k, int x, int y)
{
    float speed = 10.0;
    switch (k)
    {
    case('w'):
        camera_eye.z += speed;
        camera_lookat.z += speed;
        camera_changed = true;
        break;
    case('s'):
        camera_eye.z -= speed;
        camera_lookat.z -= speed;
        camera_changed = true;
        break;
    case('a'):
        camera_eye.x += speed;
        camera_lookat.x += speed;
        camera_changed = true;
        break;
    case('d'):
        camera_eye.x -= speed;
        camera_lookat.x -= speed;
        camera_changed = true;
        break;
    }
}


void glutMousePress(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_button = button;
        mouse_prev_pos = make_int2(x, y);
    }
}


void glutMouseMotion(int x, int y)
{
    if (mouse_button == GLUT_LEFT_BUTTON)
    {
        const float2 from = { static_cast<float>(mouse_prev_pos.x),
                              static_cast<float>(mouse_prev_pos.y) };
        const float2 to = { static_cast<float>(x),
                              static_cast<float>(y) };

        const float2 a = { from.x / width, from.y / height };
        const float2 b = { to.x / width, to.y / height };

        camera_rotate = arcball.rotate(b, a);
        camera_changed = true;
    }

    mouse_prev_pos = make_int2(x, y);
}


void glutResize(int w, int h)
{
    if (w == (int)width && h == (int)height) return;

    camera_changed = true;

    width = w;
    height = h;
    sutil::ensureMinimumSize(width, height);

    sutil::resizeBuffer(getOutputBuffer(), width, height);
    sutil::resizeBuffer(getTonemappedBuffer(), width, height);
    sutil::resizeBuffer(getAlbedoBuffer(), width, height);
    sutil::resizeBuffer(getNormalBuffer(), width, height);
    sutil::resizeBuffer(denoisedBuffer, width, height);

    glViewport(0, 0, width, height);

    postprocessing_needs_init = true;

    glutPostRedisplay();
}


//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    
    try
    {
        glutInitialize(&argc, argv);

        glewInit();

        createContext();

        loadTrainingFile(training_file);

        setupCamera();
        loadGeometry();

        context->validate();
        
        glutRun();        

        return 0;
    }
    SUTIL_CATCH(context->get())
}

