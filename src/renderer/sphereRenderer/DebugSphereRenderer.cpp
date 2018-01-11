#include "DebugSphereRenderer.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <cuda_gl_interop.h>

#include "GeometryBuilder.hpp"

#define CAMERA_SPEED 10000
#define CAMERA_ROT_SPEED 0.002
#define PI 3.14159265359
#define ANGLE_EPSILON 0.1

DebugSphereRenderer::DebugSphereRenderer(int windowWidth, int windowHeight, Particles* particles)
{
    camera.setProjectionMatrix(45.0f, windowWidth, windowHeight, 1.0f, 100000.0f);
    camera.position = glm::vec3(0, 0, 15000);
    camera.setOrientation(glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));
    this->particles = particles;
}

void DebugSphereRenderer::init() {
    // Set OpenGL settings
    glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glCullFace(GL_BACK);

    // Load shader
    shaderProgram.source(GL_VERTEX_SHADER, "shaders/sphereVertPlain.glsl");
    shaderProgram.source(GL_FRAGMENT_SHADER, "shaders/sphereFrag.glsl");
    shaderProgram.link();
    glUseProgram(shaderProgram.getId());

    // Create Model
    std::vector<glm::vec3> vertices;
    GeometryBuilder::buildSphere(12, 1, vertices);
    sphereModel.loadVertexData(vertices, shaderProgram.getId(), "aPos");

    cameraAzimuthAngle = 0;
    cameraPolarAngle = 0;
}

glm::vec4 *DebugSphereRenderer::allocateParticlesAndInit_cpu(int numParticles, glm::vec4 *particlesPos)
{
    return NULL;
}

cudaGraphicsResource_t DebugSphereRenderer::allocateParticlesAndInit_gpu(int numParticles, glm::vec4 *particlesPos)
{
    /*
    // SSBO allocation
    thi = static_cast<size_t>(numParticles);

    particleSSBOLocation = glGetProgramResourceIndex(shaderProgram.getId(), GL_SHADER_STORAGE_BLOCK, "particles_ssbo");
    glGenBuffers(1, &particleSSBOBufferObject);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, particleSSBOLocation, particleSSBOBufferObject);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * sizeof(glm::vec4), particlesPos, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);*/

/*
    cudaGraphicsGLRegisterBuffer(&vboParticlesPos_cuda,
                                 particleSSBOBufferObject,
                                 cudaGraphicsRegisterFlagsNone);*/
    cudaGraphicsResource_t vboParticlesPos_cuda;
    return vboParticlesPos_cuda;
}

void DebugSphereRenderer::updateCamera(float frameTime) {
    ShpereRendererInputDerivedData inputData = inputHandler.getDerivedData();
    glm::vec3 transformedVelocity = camera.orientation * inputData.cameraLocalVelocity;
    camera.position += transformedVelocity * (CAMERA_SPEED * frameTime);

    cameraAzimuthAngle += inputData.mouseMovement.x * CAMERA_ROT_SPEED;
    cameraPolarAngle -= inputData.mouseMovement.y * CAMERA_ROT_SPEED;
    if (cameraPolarAngle > PI / 2 - ANGLE_EPSILON)
        cameraPolarAngle = PI / 2 - ANGLE_EPSILON;
    else if (cameraPolarAngle < -PI / 2 + ANGLE_EPSILON)
        cameraPolarAngle = -PI / 2 + ANGLE_EPSILON;

    glm::vec3 cameraForwardVector(
            cos(cameraPolarAngle) * sin(cameraAzimuthAngle),
            sin(cameraPolarAngle),
            -cos(cameraPolarAngle) * cos(cameraAzimuthAngle));

    camera.setOrientation(cameraForwardVector, glm::vec3(0, 1, 0));
}

void DebugSphereRenderer::render() //const std::vector<Sphere *> &spheres, float frameTime
{
    // Update
    updateCamera(0.01f); // TODO: provide frame time...

    // Draw
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for(int i = 0; i < particles->numParticles; i++)
    {
        GLint MVPUniformLocation = glGetUniformLocation(shaderProgram.getId(), "mvp");


        glm::mat4 viewProjectionMatrix = camera.getModelViewProjectionMatrix(glm::translate(glm::mat4(1), glm::vec3(particles->pos[i])));
        std::cout << "Camera: (" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << ")" << std::endl;

        glUniformMatrix4fv(MVPUniformLocation, 1, GL_FALSE, glm::value_ptr(viewProjectionMatrix));

        // TODO: Set color and radius in shader from particle type (OGL needs access to particle pos, type and radius. In case of time pressure, the latter two can be faked by hard-coding the shader)
        // Set color

        GLint colorUniformLocation = glGetUniformLocation(shaderProgram.getId(), "inColor");
        glUniform4fv(colorUniformLocation, 1, glm::value_ptr(glm::vec4(0.8, 0.2, 0.2, 1.0)));

        // Draw solid and then set the color to be slightly darker and draw wireframe
        sphereModel.drawSolid();
        glUniform4fv(colorUniformLocation, 1, glm::value_ptr(glm::vec4(0.5, 0.0, 0.0, 1)));
        sphereModel.drawWireframe();
    }
}

void DebugSphereRenderer::destroy() {

}

Camera_I * DebugSphereRenderer::getCamera()
{
    return &camera;
}

InputHandler_I * DebugSphereRenderer::getInputHandler()
{
    return &inputHandler;
}



