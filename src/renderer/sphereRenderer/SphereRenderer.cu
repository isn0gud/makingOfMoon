#include "SphereRenderer.cuh"

#include <glm/gtc/type_ptr.hpp>
#include <cuda_gl_interop.h>

#include "GeometryBuilder.hpp"

#define CAMERA_SPEED 10000
#define CAMERA_ROT_SPEED 0.002
#define PI 3.14159265359
#define ANGLE_EPSILON 0.1

SphereRenderer::SphereRenderer(int windowWidth, int windowHeight)
{
    camera.setProjectionMatrix(45.0f, windowWidth, windowHeight, 1.0f, 100000.0f);
    camera.position = glm::vec3(0, 0, 15000);
    camera.setOrientation(glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));
}

void SphereRenderer::init() {
    // Set OpenGL settings
    glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glCullFace(GL_BACK);

    // Load shader
    shaderProgram.source(GL_VERTEX_SHADER, "shaders/sphereVertSSBO.glsl");
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

void SphereRenderer::fillAttributeSSBO(Particles *particles)
{
    glm::vec4* attributeArray = new glm::vec4[particles->numParticles];
    for(int i = 0; i < particles->numParticles; i++)
    {
        glm::vec3 color = Particles::getMaterialColor(particles->type[i]);
        attributeArray[i] = glm::vec4(color.x, color.y, color.z, particles->radius[i]);
    }

    GLuint particleAttributeSSBOLocation = 2; // Hard-coded in shader
    GLuint particleAttributeSSBOBufferObject;
    glGenBuffers(1, &particleAttributeSSBOBufferObject);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, particleAttributeSSBOLocation, particleAttributeSSBOBufferObject);
    glBufferData(GL_SHADER_STORAGE_BUFFER, particles->numParticles * sizeof(glm::vec4), attributeArray, GL_STATIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    delete [] attributeArray;
}

glm::vec4 *SphereRenderer::allocateParticlesAndInit_cpu(Particles* particles)
{
    fillAttributeSSBO(particles);

    // SSBO allocation
    GLuint particlePositionSSBOLocation = 1; // // Hard-coded in shader
    GLuint particlePositionSSBOBufferObject;
    glGenBuffers(1, &particlePositionSSBOBufferObject);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, particlePositionSSBOLocation, particlePositionSSBOBufferObject);
    glNamedBufferStorage(particlePositionSSBOBufferObject, particles->numParticles * sizeof(glm::vec4), particles->pos,
                         GL_MAP_WRITE_BIT | GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_PERSISTENT_BIT);

    //Mapping gpu memory to cpu memory for easy writes.
    glm::vec4 *particlePosPointer;
    this->numParticles = static_cast<size_t>(particles->numParticles);
    particlePosPointer = (glm::vec4 *) glMapNamedBufferRange(particlePositionSSBOBufferObject, 0, particles->numParticles * sizeof(glm::vec4),
                                                             GL_MAP_WRITE_BIT | GL_MAP_READ_BIT |
                                                             GL_MAP_PERSISTENT_BIT | GL_MAP_PERSISTENT_BIT);

    if (!particlePosPointer) {
        GLenum error = glGetError();
        fprintf(stderr, "Buffer map failed! %d (%s)\n", error, glewGetErrorString(error)); //gluErrorString(error));
        return nullptr;
    } else {
        return particlePosPointer;
    }
}

cudaGraphicsResource_t SphereRenderer::allocateParticlesAndInit_gpu(Particles* particles)
{
    fillAttributeSSBO(particles);

    this->numParticles = static_cast<size_t>(particles->numParticles);
    GLuint particlePositionSSBOLocation = 1; // Hard-coded in shader
    GLuint particlePositionSSBOBufferObject;
    glGenBuffers(1, &particlePositionSSBOBufferObject);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, particlePositionSSBOLocation, particlePositionSSBOBufferObject);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * sizeof(glm::vec4), particles->pos, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    cudaGraphicsResource_t vboParticlesPos_cuda;
    cudaGraphicsGLRegisterBuffer(&vboParticlesPos_cuda,
                                 particlePositionSSBOBufferObject,
                                 cudaGraphicsRegisterFlagsNone);
    return vboParticlesPos_cuda;
}

void SphereRenderer::updateCamera(float frameTime) {
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

void SphereRenderer::render(float frameTime)
{
    // Update
    updateCamera(frameTime);

    // Draw
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Set Model-View-Projection Matrix
    GLint MVPUniformLocation = glGetUniformLocation(shaderProgram.getId(), "mvp");
    glm::mat4 viewProjectionMatrix = camera.getViewProjectionMatrix();
    glUniformMatrix4fv(MVPUniformLocation, 1, GL_FALSE, glm::value_ptr(viewProjectionMatrix));

    // Set color
    GLint colorUniformLocation = glGetUniformLocation(shaderProgram.getId(), "inColor");
    glUniform4fv(colorUniformLocation, 1, glm::value_ptr(glm::vec4(0.0, 0.0, 0.0, 0.0)));

    // Draw solid and then set the color to be slightly darker and draw wireframe
    sphereModel.drawSolidInstanced(numParticles);
    glUniform4fv(colorUniformLocation, 1, glm::value_ptr(glm::vec4(0.25, 0.25, 0.25, 0)));
    sphereModel.drawWireframeInstanced(numParticles);
}

void SphereRenderer::destroy() {

}

Camera_I * SphereRenderer::getCamera()
{
    return &camera;
}

InputHandler_I * SphereRenderer::getInputHandler()
{
    return &inputHandler;
}




