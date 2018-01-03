#include "ParticleSphereRenderer.hpp"

#include "GeometryBuilder.hpp"

#define CAMERA_SPEED 10000
#define CAMERA_ROT_SPEED 0.002
#define PI 3.14159265359
#define ANGLE_EPSILON 0.1

void ParticleSphereRenderer::init() {
    // Set OpenGL settings
    glClearColor(0.3f, 0.3f, 0.3f, 0.0f);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glCullFace(GL_BACK);

    // Load shader

    shaderProgram.source(GL_VERTEX_SHADER, "shaders/vertexShader.glsl");
    shaderProgram.source(GL_FRAGMENT_SHADER, "shaders/fragmentShader.glsl");
    shaderProgram.link();


//    // Setup camera
//    camera.setProjectionMatrix(45.0f, windowWidth, windowHeight, 1.0f, 100000.0f);
//    camera.position = glm::vec3(0, 0, 15000);
//    camera.setOrientation(glm::vec3(0, 0, -1), glm::vec3(0, 1, 0));

    // Create Model
    std::vector<glm::vec3> vertices;
    GeometryBuilder::buildSphere(12, 1, vertices);
    sphereModel.loadVertexData(vertices, shaderProgram, "aPos");

}

//void ParticleSphereRenderer::updateCamera(float frameTime) {
//    auto inputData = inputHandler.getDerivedData();
//    camera.position += camera.orientation * inputData.cameraLocalVelocity * CAMERA_SPEED * frameTime;
//
//    cameraAzimuthAngle += inputData.mouseMovement.x * CAMERA_ROT_SPEED;
//    cameraPolarAngle -= inputData.mouseMovement.y * CAMERA_ROT_SPEED;
//
//    if (cameraPolarAngle > PI / 2 - ANGLE_EPSILON)
//        cameraPolarAngle = PI / 2 - ANGLE_EPSILON;
//    else if (cameraPolarAngle < -PI / 2 + ANGLE_EPSILON)
//        cameraPolarAngle = -PI / 2 + ANGLE_EPSILON;
//
//    glm::vec3 cameraForwardVector(
//            cos(cameraPolarAngle) * sin(cameraAzimuthAngle),
//            sin(cameraPolarAngle),
//            -cos(cameraPolarAngle) * cos(cameraAzimuthAngle));
//    camera.setOrientation(cameraForwardVector, glm::vec3(0, 1, 0));
//}

void ParticleSphereRenderer::render() {
    //const std::vector<Sphere *> &spheres, float frameTime
    // Update
    updateCamera(frameTime);

    // Draw
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    for (Sphere *sphere : spheres) {
        // Set Model-View-Projection-Matrix
        GLint MVPUniformLocation = glGetUniformLocation(shaderProgramId, "mvp");
        glm::mat4 modelTransformationMatrix =
                sphere->getTransformationMatrix() * glm::scale(glm::vec3(sphere->getRadius()));
        glm::mat4 mvp = camera.getModelViewProjectionMatrix(modelTransformationMatrix);
        glUniformMatrix4fv(MVPUniformLocation, 1, GL_FALSE, glm::value_ptr(mvp));

        // Set color
        GLint colorUniformLocation = glGetUniformLocation(shaderProgramId, "inColor");
        glUniform4fv(colorUniformLocation, 1, glm::value_ptr(sphere->getColor()));

        // Draw solid and then set the color to be slightly darker and draw wireframe
        sphereModel.drawSolid();
        glUniform4fv(colorUniformLocation, 1, glm::value_ptr(sphere->getColor() - glm::vec4(0.25, 0.25, 0.25, 1)));
        sphereModel.drawWireframe();
    }

    WindowManager::getInstance()->swapBuffers();
}



glm::vec4 *ParticleSphereRenderer::allocateParticlesAndInit_cpu(int numParticles, glm::vec4 *particlesPos) {
    return nullptr;
}

cudaGraphicsResource_t ParticleSphereRenderer::allocateParticlesAndInit_gpu(int numParticles, glm::vec4 *particlesPos) {
    return nullptr;
}

void ParticleSphereRenderer::destroy() {

}

ParticleSphereRenderer::ParticleSphereRenderer(Camera_I *camera) : camera(camera) {}
