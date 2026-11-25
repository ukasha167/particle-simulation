#include "renderer.hpp"

#include <array>
#include <cmath>
#include <string>
#include <raymath.h>
#include <rlgl.h>

Texture2D Renderer::sprite;

namespace
{
    // ONE DRAW CALL FOR THE WHOLE FIELD.
    //
    // EVERY PARTICLE IS AN INSTANCE OF THE SAME SIX-VERTEX QUAD. THE CPU UPLOADS
    // THREE FLOATS PER PARTICLE (CENTRE AND HEAT) AND THE VERTEX SHADER PLACES THE
    // QUAD AND PICKS ITS COLOUR. THE rlgl BATCH THIS REPLACES MADE NINE FUNCTION
    // CALLS AND WROTE FOUR FULL VERTICES PER PARTICLE ON THE CPU, WHICH AT 100k
    // COST 9 ms A FRAME. THIS COSTS 0.7 ms.
    constexpr int INSTANCE_STRIDE = 3;

    alignas(64) std::array<float, TOTAL_PARTICLES_COUNT * INSTANCE_STRIDE> instanceData;

    Shader particleShader = {};
    int mvpLoc = -1;
    int radiusLoc = -1;
    int coldLog2Loc = -1;
    int paletteLoc = -1;
    int spriteLoc = -1;

    unsigned int quadVao = 0;
    unsigned int cornerVbo = 0;
    unsigned int instanceVbo = 0;

    // THE #version LINE AND THE PALETTE SIZE ARE PASTED IN FRONT OF THIS AT LOAD TIME,
    // SO THE ARRAY BELOW ALWAYS MATCHES HEAT_PALETTE.
    //
    // HEAT IS 1.0 AT HEAT_HOT_SPEED'S ENERGY AND coldLog2 IS log2 OF THE HEAT AT
    // HEAT_COLD_SPEED, SO position RUNS FROM 0 AT COLD TO 1 AT HOT, ONE EVEN STEP PER
    // HALVING OF ENERGY IN BETWEEN.
    const char *VERTEX_SHADER_BODY = R"(
layout(location = 0) in vec2 corner;
layout(location = 1) in vec3 instance;

uniform mat4 mvp;
uniform float radius;
uniform float coldLog2;
uniform vec3 palette[PALETTE_SIZE];

out vec2 uv;
out vec3 tint;

void main()
{
    float position = clamp(1.0 - log2(max(instance.z, 1e-8)) / coldLog2, 0.0, 1.0);
    float scaled = position * float(PALETTE_SIZE - 1);
    int stop = min(int(scaled), PALETTE_SIZE - 2);
    tint = mix(palette[stop], palette[stop + 1], scaled - float(stop));

    uv = corner * 0.5 + 0.5;
    gl_Position = mvp * vec4(instance.xy + corner * radius, 0.0, 1.0);
}
)";

    const char *FRAGMENT_SHADER = R"(#version 330
in vec2 uv;
in vec3 tint;

uniform sampler2D sprite;

out vec4 finalColor;

void main()
{
    finalColor = vec4(tint, 1.0) * texture(sprite, uv);
}
)";

    Vector3 rgbOf(const uint32_t hex)
    {
        return {static_cast<float>((hex >> 16) & 0xFFu) / 255.0f,
                static_cast<float>((hex >> 8) & 0xFFu) / 255.0f,
                static_cast<float>(hex & 0xFFu) / 255.0f};
    }
} // namespace

uint8_t Renderer::loadAsset(const char *imageName, const uint8_t diameter)
{
    Image img = LoadImage(imageName); // DEFAULT IMAGE WIDTH IS 1024x1024

    if (diameter <= 0 || img.data == nullptr)
    {
        return 1;
    }

    // THE IMAGE IS QUITE LARGE, WE NEED TO RESIZE IT
    ImageResize(&img, diameter, diameter);

    // LOAD IT TO TEXTURE MEMORY (PROBABLY GPU)
    sprite = LoadTextureFromImage(img);
    SetTextureFilter(sprite, TEXTURE_FILTER_BILINEAR);

    UnloadImage(img);

    const std::string vertexShader = "#version 330\n#define PALETTE_SIZE " +
                                     std::to_string(HEAT_PALETTE.size()) + "\n" + VERTEX_SHADER_BODY;

    // A SHADER THAT FAILS TO COMPILE COMES BACK AS raylib'S DEFAULT ONE
    particleShader = LoadShaderFromMemory(vertexShader.c_str(), FRAGMENT_SHADER);
    if (particleShader.id == rlGetShaderIdDefault())
    {
        return 1;
    }

    mvpLoc = GetShaderLocation(particleShader, "mvp");
    radiusLoc = GetShaderLocation(particleShader, "radius");
    coldLog2Loc = GetShaderLocation(particleShader, "coldLog2");
    paletteLoc = GetShaderLocation(particleShader, "palette");
    spriteLoc = GetShaderLocation(particleShader, "sprite");

    // THE UNIFORMS THAT NEVER CHANGE ARE SET ONCE
    const float radius = Particle::radius;
    const float coldLog2 = std::log2((HEAT_COLD_SPEED * HEAT_COLD_SPEED) / (HEAT_HOT_SPEED * HEAT_HOT_SPEED));
    const int spriteUnit = 0;

    std::array<float, HEAT_PALETTE.size() * 3> paletteRgb;
    for (size_t k = 0; k < HEAT_PALETTE.size(); k++)
    {
        const Vector3 rgb = rgbOf(HEAT_PALETTE[k]);
        paletteRgb[k * 3] = rgb.x;
        paletteRgb[k * 3 + 1] = rgb.y;
        paletteRgb[k * 3 + 2] = rgb.z;
    }

    SetShaderValue(particleShader, radiusLoc, &radius, SHADER_UNIFORM_FLOAT);
    SetShaderValue(particleShader, coldLog2Loc, &coldLog2, SHADER_UNIFORM_FLOAT);
    SetShaderValueV(particleShader, paletteLoc, paletteRgb.data(), SHADER_UNIFORM_VEC3,
                    static_cast<int>(HEAT_PALETTE.size()));
    SetShaderValue(particleShader, spriteLoc, &spriteUnit, SHADER_UNIFORM_INT);

    // A UNIT QUAD AS TWO TRIANGLES, CORNERS IN [-1, 1]. (-1, -1) IS THE TOP LEFT,
    // WHICH MAPS TO TEXTURE COORDINATE (0, 0) EXACTLY AS THE OLD BATCH DID.
    constexpr float corners[12] = {
        -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
        -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f};

    quadVao = rlLoadVertexArray();
    rlEnableVertexArray(quadVao);

    cornerVbo = rlLoadVertexBuffer(corners, sizeof(corners), false);
    rlSetVertexAttribute(0, 2, RL_FLOAT, false, 0, 0);
    rlEnableVertexAttribute(0);

    instanceVbo = rlLoadVertexBuffer(nullptr, static_cast<int>(sizeof(instanceData)), true);
    rlSetVertexAttribute(1, INSTANCE_STRIDE, RL_FLOAT, false, 0, 0);
    rlEnableVertexAttribute(1);
    rlSetVertexAttributeDivisor(1, 1);

    rlDisableVertexArray();

    return 0;
}

void Renderer::drawParticles(const Particle &particles, const uint32_t currentParticlesCount)
{
    if (sprite.id == 0 || currentParticlesCount == 0)
    {
        return;
    }

    const float *__restrict posX = particles.posX;
    const float *__restrict posY = particles.posY;
    const float *__restrict heat = particles.heat;
    float *__restrict out = instanceData.data();

    for (uint32_t i = 0; i < currentParticlesCount; i++)
    {
        out[i * 3] = posX[i];
        out[i * 3 + 1] = posY[i];
        out[i * 3 + 2] = heat[i];
    }

    // WHATEVER raylib HAS QUEUED SO FAR IS DRAWN FIRST, SO LAYERING STAYS IN CALL ORDER
    rlDrawRenderBatchActive();

    rlUpdateVertexBuffer(instanceVbo, out, static_cast<int>(currentParticlesCount * sizeof(float) * INSTANCE_STRIDE), 0);

    rlEnableShader(particleShader.id);
    rlSetUniformMatrix(mvpLoc, MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection()));

    rlActiveTextureSlot(0);
    rlEnableTexture(sprite.id);

    rlEnableVertexArray(quadVao);
    rlDrawVertexArrayInstanced(0, 6, static_cast<int>(currentParticlesCount));
    rlDisableVertexArray();

    rlDisableTexture();
    rlDisableShader();
}

void Renderer::drawStats(const uint16_t fps, const uint32_t currentParticlesCount,
                         const uint32_t threads)
{
    DrawText(TextFormat("FPS: %d\nParticles: %d\nThreads: %d", fps, currentParticlesCount, threads),
             10, 10, 24, WHITE);
}

void Renderer::unloadAsset()
{
    if (quadVao > 0)
    {
        rlUnloadVertexArray(quadVao);
        rlUnloadVertexBuffer(cornerVbo);
        rlUnloadVertexBuffer(instanceVbo);
        quadVao = 0;
    }

    if (particleShader.id > 0 && particleShader.id != rlGetShaderIdDefault())
    {
        UnloadShader(particleShader);
    }

    if (sprite.id > 0)
    {
        UnloadTexture(sprite);
    }
}
