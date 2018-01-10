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
    shaderProgram.source(GL_VERTEX_SHADER, "shaders/vertexShader.glsl");
    shaderProgram.source(GL_FRAGMENT_SHADER, "shaders/fragmentShader.glsl");
    shaderProgram.link();
    glUseProgram(shaderProgram.getId());

    // Create Model
    std::vector<glm::vec3> vertices;
    GeometryBuilder::buildSphere(12, 1, vertices);
    sphereModel.loadVertexData(vertices, shaderProgram.getId(), "aPos");

    cameraAzimuthAngle = 0;
    cameraPolarAngle = 0;
}

glm::vec4 *SphereRenderer::allocateParticlesAndInit_cpu(int numParticles, glm::vec4 *particlesPos)
{
    // SSBO allocation & data upload
    glNamedBufferStorage(vboParticlesPos, numParticles * sizeof(glm::vec4), particlesPos,
                         GL_MAP_WRITE_BIT | GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT |
                         GL_MAP_COHERENT_BIT); // Buffer storage is fixed size compared to BuferData

    //Mapping gpu memory to cpu memory for easy writes.
    glm::vec4 *particlePosPointer;
    this->numParticles = static_cast<size_t>(numParticles);
    particlePosPointer = (glm::vec4 *) glMapNamedBufferRange(vboParticlesPos, 0, numParticles * sizeof(glm::vec4),
                                                             GL_MAP_WRITE_BIT | GL_MAP_READ_BIT |
                                                             GL_MAP_PERSISTENT_BIT |
                                                             GL_MAP_COHERENT_BIT);

    if (!particlePosPointer) {
        GLenum error = glGetError();
        fprintf(stderr, "Buffer map failed! %d (%s)\n", error, glewGetErrorString(error)); //gluErrorString(error));
        return nullptr;
    } else {
        return particlePosPointer;
    }
}

cudaGraphicsResource_t SphereRenderer::allocateParticlesAndInit_gpu(int numParticles, glm::vec4 *particlesPos)
{
    // SSBO allocation & data upload
    /*glNamedBufferStorage(vboParticlesPos, numParticles * sizeof(glm::vec4), particlesPos,
                         GL_MAP_WRITE_BIT | GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT |
                         GL_MAP_COHERENT_BIT);*/ // Buffer storage is fixed size compared to BuferData
    this->numParticles = static_cast<size_t>(numParticles);

    particleSSBOLocation = glGetProgramResourceIndex(shaderProgram.getId(), GL_SHADER_STORAGE_BLOCK, "particles_ssbo");
    glGenBuffers(1, &particleSSBOBufferObject);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, particleSSBOLocation, particleSSBOBufferObject);
    glBufferData(GL_SHADER_STORAGE_BUFFER, numParticles * sizeof(glm::vec4), particlesPos, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    cudaGraphicsResource_t vboParticlesPos_cuda;
    cudaGraphicsGLRegisterBuffer(&vboParticlesPos_cuda,
                                 particleSSBOBufferObject,
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

void SphereRenderer::render() //const std::vector<Sphere *> &spheres, float frameTime
{
    // Update
    updateCamera(0.01f); // TODO: provide frame time...

    // Draw
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLint MVPUniformLocation = glGetUniformLocation(shaderProgram.getId(), "mvp");
    glm::mat4 viewProjectionMatrix = camera.getViewProjectionMatrix();
    glUniformMatrix4fv(MVPUniformLocation, 1, GL_FALSE, glm::value_ptr(viewProjectionMatrix));

    // TODO: Set color and radius in shader from particle type (OGL needs access to particle pos, type and radius. In case of time pressure, the latter two can be faked by hard-coding the shader)
    // Set color

    GLint colorUniformLocation = glGetUniformLocation(shaderProgram.getId(), "inColor");
    glUniform4fv(colorUniformLocation, 1, glm::value_ptr(glm::vec4(0.8, 0.2, 0.2, 1.0)));

    // Draw solid and then set the color to be slightly darker and draw wireframe
    sphereModel.drawSolidInstanced(numParticles);
    glUniform4fv(colorUniformLocation, 1, glm::value_ptr(glm::vec4(0.5, 0.0, 0.0, 1)));
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



