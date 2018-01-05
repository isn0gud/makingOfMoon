#include "ParticleRenderer.cuh"

#include <algorithm>
#include <glm/gtc/type_ptr.hpp>
#include <cuda_gl_interop.h>
#include "CameraRotateCenter.hpp"


const int FBO_MARGIN = 50;


void ParticleRenderer::init() {

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // OpenGL initialization
    glewExperimental = GL_TRUE;
    GLenum error = glewInit();
    if (error != GLEW_OK) {
        std::cout << "PROBLEMS! CANNOT LOAD OpenGL!" << std::endl;
        throw std::runtime_error("Can't load GL");
    }

    createFlareTexture();
    createVaosVbos();
    initShaders();
    initFbos();
    setUniforms();
}


std::vector<float> genFlareTex(int tex_size) {
    std::vector<float> pixels(static_cast<unsigned long>(tex_size * tex_size));
    float sigma2 = static_cast<float>(tex_size) / 2.0f;
    float A = 1.0;
    for (int i = 0; i < tex_size; ++i) {
        float i1 = i - tex_size / 2;
        for (int j = 0; j < tex_size; ++j) {
            float j1 = j - tex_size / 2;
            // gamma corrected gauss
            pixels[i * tex_size + j] = pow(A * exp(-((i1 * i1) / (2 * sigma2) + (j1 * j1) / (2 * sigma2))), 2.2f);
        }
    }
    return pixels;
}

void ParticleRenderer::createFlareTexture() {
    texSize = 16;
    glCreateTextures(GL_TEXTURE_2D, 1, &flareTex);
    glTextureStorage2D(flareTex, 1, GL_R32F, texSize, texSize);
    glTextureParameteri(flareTex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    {
        std::vector<float> pixels = genFlareTex(texSize);
        glTextureSubImage2D(flareTex, 0, 0, 0,
                            texSize, texSize, GL_RED, GL_FLOAT, pixels.data());
    }
}

void ParticleRenderer::createVaosVbos() {
    // Particle VAO
    glCreateVertexArrays(1, &vaoParticles);
    glCreateBuffers(1, &vboParticlesPos);
//    glCreateBuffers(1, &ssboVelocities);
    glVertexArrayVertexBuffer(vaoParticles, 0, vboParticlesPos, 0, sizeof(glm::vec4));
//    glVertexArrayVertexBuffer(vaoParticles, 1, ssboVelocities, 0, sizeof(glm::vec4));

    // Position
    glEnableVertexArrayAttrib(vaoParticles, 0);
    glVertexArrayAttribFormat(vaoParticles, 0, 4, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vaoParticles, 0, 0);

    // Velocity
    glEnableVertexArrayAttrib(vaoParticles, 1);
    glVertexArrayAttribFormat(vaoParticles, 1, 4, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vaoParticles, 1, 1);

    // Deferred VAO
    glCreateVertexArrays(1, &vaoDeferred);
    glCreateBuffers(1, &vboDeferred);
    glVertexArrayVertexBuffer(vaoDeferred, 0, vboDeferred, 0, sizeof(glm::vec2));
    // Position
    glEnableVertexArrayAttrib(vaoDeferred, 0);
    glVertexArrayAttribFormat(vaoDeferred, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(vaoDeferred, 0, 0);

    // Deferred tri
    glm::vec2 tri[3] = {
            glm::vec2(-2, -1),
            glm::vec2(+2, -1),
            glm::vec2(0, 4)};
    glNamedBufferStorage(vboDeferred, 3 * sizeof(glm::vec2), tri, 0);
}


void ParticleRenderer::initShaders() {
    programHdr.source(GL_VERTEX_SHADER, "shaders/main.vert");
    programHdr.source(GL_FRAGMENT_SHADER, "shaders/main.frag");
    programHdr.source(GL_GEOMETRY_SHADER, "shaders/main.geom");
    programHdr.link();

    programTonemap.source(GL_VERTEX_SHADER, "shaders/deferred.vert");
    programTonemap.source(GL_FRAGMENT_SHADER, "shaders/tonemap.frag");
    programTonemap.link();
    if (BLUR) {
        programBlur.source(GL_VERTEX_SHADER, "shaders/deferred.vert");
        programBlur.source(GL_FRAGMENT_SHADER, "shaders/blur.frag");
        programBlur.link();
    }
    programLum.source(GL_VERTEX_SHADER, "shaders/deferred.vert");
    programLum.source(GL_FRAGMENT_SHADER, "shaders/luminance.frag");
    programLum.link();
}

void ParticleRenderer::initFbos() {
    int blur_dsc = 2;
    blurDownscale = blur_dsc;

    glCreateFramebuffers(4, fbos);
    glCreateTextures(GL_TEXTURE_2D, 4, attachs);

    int base_width = camera->getWindowWidth() + 2 * FBO_MARGIN;
    int base_height = camera->getWindowHeight() + 2 * FBO_MARGIN;

    int widths[] = {base_width,
                    base_width / blur_dsc,
                    base_width / blur_dsc,
                    base_width / 2};

    int heights[] = {base_height,
                     base_height / blur_dsc,
                     base_height / blur_dsc,
                     base_height / 2};

    lumLod = (int) floor(log2((double)max(base_width, base_height) / 2));
    int mipmaps[] = {1, 1, 1, lumLod + 1};
    GLenum types[] = {GL_RGBA16F, GL_RGBA16F, GL_RGBA16F, GL_R16F};
    GLenum min_filters[] = {GL_LINEAR, GL_LINEAR, GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR};

    for (int i = 0; i < 4; ++i) {
        glTextureStorage2D(attachs[i], mipmaps[i], types[i], widths[i], heights[i]);
        glTextureParameteri(attachs[i], GL_TEXTURE_MIN_FILTER, min_filters[i]);
        glTextureParameteri(attachs[i], GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTextureParameteri(attachs[i], GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glNamedFramebufferTexture(fbos[i], GL_COLOR_ATTACHMENT0, attachs[i], 0);
    }
}

void ParticleRenderer::setUniforms() {
    // const Uniforms
//    glProgramUniform1f(programInteraction.getId(), 0, SIM_dt);
//    glProgramUniform1f(programInteraction.getId(), 1, G);
//    glProgramUniform1f(programInteraction.getId(), 2, DAMPING);
//    glProgramUniform1f(programIntegration.getId(), 0, SIM_dt);
    // NDC sprite size
    glProgramUniform2f(programHdr.getId(), 8,
                       texSize / float(2 * camera->getWindowWidth()),
                       texSize / float(2 * camera->getWindowHeight()));
    // Blur sample offset length
    glProgramUniform2f(programBlur.getId(), 0,
                       (float) blurDownscale / camera->getWindowWidth(),
                       (float) blurDownscale / camera->getWindowHeight());
}


void ParticleRenderer::render() {

    camera->applyInput();

    // Particle HDR rendering
    glViewport(0, 0, camera->getWindowWidth() + 2 * FBO_MARGIN, camera->getWindowHeight() + 2 * FBO_MARGIN);
    glBindVertexArray(vaoParticles);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glBindFramebuffer(GL_FRAMEBUFFER, fbos[0]);
    glUseProgram(programHdr.getId());
    glClear(GL_COLOR_BUFFER_BIT);
    glProgramUniformMatrix4fv(programHdr.getId(), 0, 1, GL_FALSE, glm::value_ptr(camera->getView()));
    glProgramUniformMatrix4fv(programHdr.getId(), 4, 1, GL_FALSE, glm::value_ptr(camera->getProj()));
    glBindTextureUnit(0, flareTex);
    glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(numParticles));

    glBindVertexArray(vaoDeferred);
    glDisable(GL_BLEND);

    glViewport(0, 0,
               (camera->getWindowWidth() + 2 * FBO_MARGIN) / blurDownscale,
               (camera->getWindowHeight() + 2 * FBO_MARGIN) / blurDownscale);
    glUseProgram(programBlur.getId());

    // Blur pingpong (N horizontal blurs then N vertical blurs)
    int loop = 0;
    for (int i = 0; i < 2; ++i) {
        if (i == 0) glProgramUniform2f(programBlur.getId(), 1, 1, 0);
        else
            glProgramUniform2f(programBlur.getId(), 1, 0, 1);
        for (int j = 0; j < 100; ++j) {
            GLuint fbo = fbos[(loop % 2) + 1];
            GLuint attach = attachs[loop ? ((loop + 1) % 2 + 1) : 0];
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glBindTextureUnit(0, attach);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            loop++;
        }
    }

    // Average luminance
    glViewport(0, 0,
               (camera->getWindowWidth() + 2 * FBO_MARGIN) / 2,
               (camera->getWindowHeight() + 2 * FBO_MARGIN) / 2);
    glBindFramebuffer(GL_FRAMEBUFFER, fbos[3]);
    glUseProgram(programLum.getId());
    glBindTextureUnit(0, attachs[0]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glGenerateTextureMipmap(attachs[3]);

    // Tonemapping step (direct to screen)
    glViewport(0, 0, camera->getWindowWidth(), camera->getWindowHeight());
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(programTonemap.getId());
    glProgramUniform1i(programTonemap.getId(), 0, lumLod);
    glBindTextureUnit(0, attachs[0]);
    glBindTextureUnit(1, attachs[2]);
    glBindTextureUnit(2, attachs[3]);
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

ParticleRenderer::ParticleRenderer(int windowWidth, int windowHeight)
{
    camera = new CameraRotateCenter(windowWidth, windowHeight);
    inputHandler = new SpriteRendererInputHandler(camera);
}

Camera_I* ParticleRenderer::getCamera()
{
    return camera;
}

InputHandler_I* ParticleRenderer::getInputHandler()
{
    return inputHandler;
}

void ParticleRenderer::destroy() {

}

glm::vec4 *ParticleRenderer::allocateParticlesAndInit_cpu(int numParticles, glm::vec4 *particlesPos) {
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

cudaGraphicsResource_t ParticleRenderer::allocateParticlesAndInit_gpu(int numParticles, glm::vec4 *particlesPos) {
    // SSBO allocation & data upload
    glNamedBufferStorage(vboParticlesPos, numParticles * sizeof(glm::vec4), particlesPos,
                         GL_MAP_WRITE_BIT | GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT |
                         GL_MAP_COHERENT_BIT); // Buffer storage is fixed size compared to BuferData
    this->numParticles = static_cast<size_t>(numParticles);

    cudaGraphicsResource_t vboParticlesPos_cuda;
    cudaGraphicsGLRegisterBuffer(&vboParticlesPos_cuda,
                                 vboParticlesPos,
                                 cudaGraphicsRegisterFlagsNone);
    return vboParticlesPos_cuda;
}
