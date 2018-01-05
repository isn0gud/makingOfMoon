#pragma once

#include "../../common.hpp"


#include <vector>
#include <driver_types.h>

#include "../../Renderer_I.hpp"
#include "../../ShaderProgram.hpp"
#include "CameraRotateCenter.hpp"
#include "SpriteRendererInputHandler.hpp"

class ParticleSpriteRenderer : public Renderer_I {

public:
    void init() override;


    explicit ParticleSpriteRenderer(int windowWidth, int windowHeight);

    glm::vec4 *allocateParticlesAndInit_cpu(int numParticles, glm::vec4 *particlesPos) override;

    cudaGraphicsResource_t allocateParticlesAndInit_gpu(int numParticles, glm::vec4 *particlesPos) override;


    void render() override;

    void destroy() override;

    Camera_I *getCamera() override;

    InputHandler_I *getInputHandler() override;

private:
    /// Generates the star flare texture
    void createFlareTexture();

    /// Creates the VAO and VBO objects
    void createVaosVbos();

    /// Loads the shaders into the gl state
    void initShaders();

    /// Initializes and supplies the framebuffers with valid data
    void initFbos();

    /// Supplies the gl state with nbody simulation parameters
    void setUniforms();

    GLuint flareTex{};           ///< Texture for the star flare
    GLuint vaoParticles{};       ///< Vertex definition for points
    GLuint vboParticlesPos{};   ///< Particle position buffer
    //    GLuint ssboVelocities{};     ///< Particle velocity buffer
    GLuint vaoDeferred{};        ///< Vertex definition for deferred
    GLuint vboDeferred{};        ///< Vertex buffer of deferred fullscreen tri

    /** Shader programs **/
    ShaderProgram programHdr{};         ///< HDR rendering step
    ShaderProgram programBlur{};        ///< Bloom blurring step
    ShaderProgram programLum{};         ///< Average luminance step
    ShaderProgram programTonemap{};     ///< Tonemapping step

    GLuint fbos[4]{};             ///< FBOs (0 for hdr, 1 & 2 for blur ping pong, 3 for luminance)
    GLuint attachs[4]{};          ///< Respective FBO attachments.

    int texSize{};               ///< Flare texture size in pixels
    int lumLod{};                ///< Luminance texture level to sample from
    int blurDownscale{};         ///< Downscale factor for the blurring step

    size_t numParticles{};
    CameraRotateCenter* camera;
    SpriteRendererInputHandler* inputHandler;
};
