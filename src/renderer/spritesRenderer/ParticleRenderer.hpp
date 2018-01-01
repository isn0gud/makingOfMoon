#pragma once

#include "../../common.hpp"


#include <vector>

#include "../../RendererCPU.hpp"
#include "../../ShaderProgram.hpp"

class ParticleRenderer : public RendererCPU {

public:
    void init() override;

    explicit ParticleRenderer(Camera *camera);

    Particles *allocateParticles(int numParticles) override;

    void render() override;

    void destroy() override;

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
    Camera *camera;

};