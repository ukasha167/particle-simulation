#include "solver.hpp"

#include <cstdlib>
#include <new>

// ---------------------------   STORAGE   ---------------------------

float *Particle::posX = nullptr;
float *Particle::posY = nullptr;
float *Particle::velX = nullptr;
float *Particle::velY = nullptr;
float *Particle::heat = nullptr;

Particle Solver::particles;

Solver::ParticleBuffer Solver::bufferA;
Solver::ParticleBuffer Solver::bufferB;
bool Solver::usingBufferA = true;

alignas(64) std::array<uint32_t, CELL_RANGE_SIZE> Solver::cellRange;
alignas(64) std::array<uint32_t, TOTAL_PARTICLES_COUNT> Solver::particleCellIDs;

uint32_t *Solver::cellCursors = nullptr;
uint32_t Solver::cursorStride = 0;

std::array<uint32_t, ThreadPool::MAX_THREADS + 1> Solver::chunkBase;
std::array<uint32_t, ThreadPool::MAX_THREADS> Solver::histogramLo;
std::array<uint32_t, ThreadPool::MAX_THREADS> Solver::histogramHi;
alignas(64) std::array<uint32_t, CELL_RANGE_SIZE> Solver::runningCursor;
std::array<uint64_t, PADDED_GRID_HEIGHT> Solver::rowWork;
std::array<Solver::RowBand, ThreadPool::MAX_THREADS * 2> Solver::bands;
uint32_t Solver::bandCount = 0;

// ---------------------------   LOCAL HELPERS   ---------------------------

namespace
{
    constexpr float BOUND_MIN_X = PARTICLE_RADIUS;
    constexpr float BOUND_MAX_X = static_cast<float>(SCREEN_WIDTH) - PARTICLE_RADIUS;
    constexpr float BOUND_MIN_Y = PARTICLE_RADIUS;
    constexpr float BOUND_MAX_Y = static_cast<float>(SCREEN_HEIGHT) - PARTICLE_RADIUS;

    // UNIFORM IN [-1, 1)
    inline float randSigned()
    {
        return (static_cast<float>(rand() % 2048) * (1.0f / 1024.0f)) - 1.0f;
    }

    // HEAT IS KINETIC ENERGY MEASURED AGAINST THE ENERGY AT HEAT_HOT_SPEED, SO 1.0 IS
    // THE TOP OF THE PALETTE. IT MAY CLIMB TO 4.0, TWO HALF-LIVES ABOVE THAT, SO A
    // HARD HIT HOLDS THE HOTTEST COLOUR FOR A MOMENT BEFORE IT STARTS TO COOL.
    constexpr float HEAT_PER_SPEED_SQ = 1.0f / (HEAT_HOT_SPEED * HEAT_HOT_SPEED);
    constexpr float HEAT_CEILING = 4.0f;

    // THE SHARE OF ITS EXCESS HEAT A PARTICLE LOSES EACH FRAME
    const float HEAT_FALL = 1.0f - std::pow(0.5f, 1.0f / (HEAT_HALF_LIFE * SIMULATION_FPS));

    inline float heatOf(const float vx, const float vy)
    {
        return std::fmin((vx * vx + vy * vy) * HEAT_PER_SPEED_SQ, HEAT_CEILING);
    }

    // SPLITS THE PADDED GRID'S ROWS BETWEEN WORKERS. THE PREFIX-SUM PASSES WALK
    // CELLS, BUT THEY DO IT IN WHOLE ROWS SO THAT THE PER-ROW WORK ESTIMATE THEY
    // ALSO PRODUCE NEVER STRADDLES TWO THREADS.
    inline void rowSliceOf(const uint32_t worker, const uint32_t workers,
                           uint32_t &begin, uint32_t &end)
    {
        constexpr uint32_t ROWS = PADDED_GRID_HEIGHT;

        const uint32_t share = ROWS / workers;
        const uint32_t extra = ROWS % workers;

        begin = worker * share + (worker < extra ? worker : extra);
        end = begin + share + (worker < extra ? 1u : 0u);
    }

    // CLAMPS ONE PARTICLE BACK INSIDE THE DOMAIN AND SETTLES ITS WALL VELOCITY.
    // A HIT SLOWER THAN WALL_RESTING_SPEED STOPS DEAD RATHER THAN BOUNCING, WHICH
    // LETS A PILE COME TO REST ON THE FLOOR INSTEAD OF SIMMERING. AT ZERO THE FLOOR
    // NEVER GOES DEAD AND EVERY HIT KEEPS WALL_RESTITUTION OF ITS SPEED.
    //
    // THERE IS NO BRANCH IN HERE, SO THE LOOPS AROUND IT VECTORISE. EVERY SELECT PICKS
    // BETWEEN THE SAME VALUES THE OLD if/else DID, IN THE SAME ORDER, SO THE RESULT IS
    // BIT FOR BIT THE SAME. INTEGRATION WENT FROM 3.1 ms A FRAME TO 1.6 ms.
    inline void resolveBounds(float &x, float &y, float &vx, float &vy)
    {
        const bool below = y > BOUND_MAX_Y;
        const bool above = y < BOUND_MIN_Y;
        const bool intoY = (below & (vy > 0.0f)) | (above & (vy < 0.0f));
        const float bouncedY = (std::fabs(vy) < WALL_RESTING_SPEED) ? 0.0f : -vy * WALL_RESTITUTION;

        y = below ? BOUND_MAX_Y : (above ? BOUND_MIN_Y : y);
        vy = intoY ? bouncedY : vy;
        vx = (below | above) ? vx * WALL_FRICTION : vx;

        const bool right = x > BOUND_MAX_X;
        const bool left = x < BOUND_MIN_X;
        const bool intoX = (right & (vx > 0.0f)) | (left & (vx < 0.0f));
        const float bouncedX = (std::fabs(vx) < WALL_RESTING_SPEED) ? 0.0f : -vx * WALL_RESTITUTION;

        x = right ? BOUND_MAX_X : (left ? BOUND_MIN_X : x);
        vx = intoX ? bouncedX : vx;
        vy = (right | left) ? vy * WALL_FRICTION : vy;
    }

    // THE CONTACT SWEEP MOVES POSITIONS AND NOTHING ELSE.
    //
    // THE KICK A CONTACT GIVES IS ALWAYS ITS POSITION FIX TIMES CONTACT_KICK (SEE
    // solveContact). SO THE WHOLE KICK A PARTICLE COLLECTS DURING A SWEEP IS
    // CONTACT_KICK TIMES HOW FAR THE SWEEP MOVED IT, AND THE SWEEP NEVER HAS TO TOUCH
    // A VELOCITY. encodeKick() STORES v - CONTACT_KICK * x AS SOON AS A PARTICLE HAS
    // BEEN PLACED, THE SWEEP MOVES x AND y, AND decodeKick() ADDS CONTACT_KICK * x
    // BACK AFTERWARDS AND FINDS THE KICKED VELOCITY WAITING. THE SAME RESULT UP TO
    // ROUNDING, WITH HALF THE MEMORY PER CONTACT AND NO KICK ARITHMETIC. THE CONTACT
    // SWEEPS WENT FROM 13.5 ms A FRAME TO 10.1 ms.
    //
    // x IS MEASURED FROM THE MIDDLE OF THE SCREEN SO THAT CONTACT_KICK * x STAYS SMALL
    // ENOUGH FOR A float TO HOLD TO A TENTH OF A PIXEL PER SECOND, THE SAME GRAIN THE
    // KICK ITSELF HAS.
    //
    // FRICTION AND THE KICK CAP BOTH NEED A VELOCITY IN THE MIDDLE OF A CONTACT, SO
    // TURNING EITHER ON PUTS THE VELOCITIES BACK INTO THE SWEEP AND THE ENCODING OFF.
    constexpr bool CONTACTS_TOUCH_VELOCITY = (PARTICLE_FRICTION > 0.0f) || (MAX_CONTACT_KICK > 0.0f);

    constexpr float KICK_ORIGIN_X = static_cast<float>(SCREEN_WIDTH) * 0.5f;
    constexpr float KICK_ORIGIN_Y = static_cast<float>(SCREEN_HEIGHT) * 0.5f;

    inline void encodeKick(const float x, const float y, float &vx, float &vy)
    {
        if constexpr (!CONTACTS_TOUCH_VELOCITY)
        {
            vx -= CONTACT_KICK * (x - KICK_ORIGIN_X);
            vy -= CONTACT_KICK * (y - KICK_ORIGIN_Y);
        }
    }

    inline void decodeKick(const float x, const float y, float &vx, float &vy)
    {
        if constexpr (!CONTACTS_TOUCH_VELOCITY)
        {
            vx += CONTACT_KICK * (x - KICK_ORIGIN_X);
            vy += CONTACT_KICK * (y - KICK_ORIGIN_Y);
        }
    }

    // ONE CONTACT.
    //
    // THE POSITION FIX IS FED BACK INTO VELOCITY, WHICH IS WHAT THE ORIGINAL VERLET
    // SOLVER DID AND WHERE THE WHOLE LOOK OF THIS SIMULATION COMES FROM. IT IS ALSO
    // THE ONLY THING PUSHING A COMPACTED PILE BACK APART, BECAUSE SYMMETRIC
    // RELAXATION ALONE CANNOT CARRY THE FLOOR'S SUPPORT UP SEVENTY LAYERS.
    //
    // THAT KICK IS ALSO HOW A PILE HOLDS UP ITS OWN WEIGHT. EVERY CONTACT HAS TO PASS
    // THE MOMENTUM OF EVERYTHING ABOVE IT DOWN TOWARDS THE FLOOR, EVERY SUB-STEP, AND
    // THE ONLY WAY IT CAN IS BY OVERLAPPING DEEPLY ENOUGH TO KICK THAT HARD. SO THE
    // OVERLAP UNDER A PILE GROWS WITH ITS DEPTH AND WITH THE SQUARE OF THE SUB-STEP.
    // A DEEP PARTICLE THEN TAKES A HUGE KICK FROM BELOW AND A HUGE KICK FROM ABOVE ON
    // EVERY SUB-STEP, AND PAST A FEW PERCENT OVERLAP THOSE STOP CANCELLING AND THE
    // PILE ERUPTS FROM THE FLOOR UP. THE CURE IS A SMALLER SUB-STEP, NOT A WEAKER
    // KICK. SEE SUB_STEPS IN defines.hpp.
    //
    // Brace = false: symmetric, momentum conserving. The only mode used by default.
    // Brace = true:  the lower particle is pinned VERTICALLY so the floor's support
    //                climbs the column. Horizontal stays symmetric, and the pass is
    //                perfectly inelastic. Both of those are load bearing, see below.
    //
    // PARTICLE i IS PASSED IN REGISTERS RATHER THAN BY INDEX. IT TAKES PART IN EVERY
    // PAIR IN ITS INNER LOOP, AND SINCE THE COMPILER CANNOT PROVE i AND j NEVER
    // ALIAS, GOING THROUGH MEMORY WOULD FORCE RELOADS AND STORES ON EVERY SINGLE
    // CANDIDATE. THE CALLER LOADS IT ONCE AND WRITES IT BACK ONCE.
    template <bool Brace>
    inline void solveContact(float &posIX, float &posIY, float &velIX, float &velIY,
                             const uint32_t j,
                             float *__restrict px, float *__restrict py,
                             float *__restrict vx, float *__restrict vy)
    {
        const float dx = posIX - px[j];
        const float dy = posIY - py[j];
        const float distSq = dx * dx + dy * dy;

        // FRICTION, THE KICK CAP AND THE SLOP BAND NEED TO KNOW WHETHER THIS IS A
        // CONTACT AT ALL. THE DEFAULT PATH NEVER ASKS, SEE fix BELOW.
        if constexpr (CONTACTS_TOUCH_VELOCITY || PENETRATION_SLOP > 0.0f)
        {
            if (distSq >= MIN_COLLISION_DIST_SQ || distSq < 1e-8f)
            {
                return;
            }
        }

        // HOW THE VERTICAL SHARE IS SPLIT. HALF EACH NORMALLY; IN A BRACE PASS THE
        // LOWER PARTICLE IS PINNED AND THE UPPER ONE ABSORBS THE LOT.
        float weightI = 0.5f;
        float weightJ = 0.5f;
        if constexpr (Brace)
        {
            const bool iIsLower = (posIY > py[j]);
            weightI = iIsLower ? (0.5f - 0.5f * VERTICAL_BRACE) : (0.5f + 0.5f * VERTICAL_BRACE);
            weightJ = iIsLower ? (0.5f + 0.5f * VERTICAL_BRACE) : (0.5f - 0.5f * VERTICAL_BRACE);
        }

        // 1. NON-PENETRATION. HORIZONTAL ALWAYS SPLITS EVENLY: A PAIR SITTING SIDE BY
        //    SIDE MUST NEVER BE ABLE TO BRACE AGAINST EACH OTHER, BECAUSE THAT IS
        //    SHEAR STRENGTH AND SHEAR STRENGTH IS A SAND DUNE.
        //
        //    fix IS (D - dist) / dist, THE OVERLAP MEASURED ALONG THE UNNORMALISED
        //    dx, dy. WITH e = 1 - distSq / D^2 IT EQUALS 1 / sqrt(1 - e) - 1, WHICH IS
        //    e/2 + 3e^2/8 + 5e^3/16 + ... THREE TERMS ARE WITHIN 0.05% AT 5% OVERLAP,
        //    0.4% AT 10% AND 3% AT 20%, AND EVERY TERM LEFT OUT IS POSITIVE, SO A DEEP
        //    OVERLAP IS UNDER-CORRECTED AND NEVER OVERSHOT. NO SQUARE ROOT AND NO
        //    DIVISION ON THE HOTTEST LINE IN THE PROGRAM.
        //
        //    AND NO BRANCH. A PAIR THAT IS NOT TOUCHING HAS e <= 0, fmax MAKES THAT 0,
        //    AND 0 MAKES fix AND BOTH MOVES 0. THE CPU GUESSES "TOUCHING OR NOT" WELL IN
        //    A SETTLED LATTICE AND BADLY IN A PILE THAT IS MOVING: WITH THE BRANCH IN,
        //    CPU COUNTERS PUT 37% OF A SETTLED SWEEP AND 58% OF ONE UNDER THE MOUSE ON
        //    WORK THAT GOT THROWN AWAY. WITHOUT IT A SETTLED 100k FRAME WENT FROM 15.2 ms
        //    TO 13.1 ms, AND ONE UNDER THE MOUSE FROM 24.1 ms TO 14.7 ms.
        //
        //    AN if OR A ?: HERE READS AS BRANCHLESS AND IS NOT. CLANG TURNS BOTH BACK
        //    INTO A BRANCH, SO KEEP THE fmax.
        float fix;
        if constexpr (PENETRATION_SLOP > 0.0f)
        {
            const float dist = std::sqrt(distSq);
            fix = std::fmax(0.0f, MIN_COLLISION_DIST - dist - PENETRATION_SLOP) / dist;
        }
        else
        {
            const float e = std::fmax(0.0f, 1.0f - distSq * (1.0f / MIN_COLLISION_DIST_SQ));
            fix = e * (0.5f + e * (0.375f + e * 0.3125f));
        }

        const float scale = POSITION_CORRECTION * fix;

        const float moveX = dx * scale * 0.5f;
        const float moveIY = dy * scale * weightI;
        const float moveJY = dy * scale * weightJ;

        posIX += moveX;
        posIY += moveIY;
        px[j] -= moveX;
        py[j] -= moveJY;

        // 2. THE SHOVE BECOMES SPEED, LESS COLLISION_REACTION_LOSS.
        //
        //    NOTHING HERE COMPARES THE RESULT AGAINST THE SPEED THE PAIR ARRIVED
        //    WITH, SO A CONTACT CAN LEAVE FASTER THAN IT CAME IN. THAT IS THE POP,
        //    AND IT IS ALSO THE ONLY FORCE ACTIVELY DRIVING A COMPACTED PILE APART,
        //    SO IT HAS TO STAY. MAX_CONTACT_KICK CAN CAP IT, BUT A CAP STARVES A DEEP
        //    PILE OF SUPPORT. KEEPING DEEP PILES STABLE IS THE JOB OF SUB_STEPS.
        //
        //    BY DEFAULT THE KICK IS NOT APPLIED HERE AT ALL, SEE encodeKick. THE CAP
        //    AND FRICTION NEED IT CONTACT BY CONTACT, SO THEY BRING THIS PATH BACK.
        if constexpr (CONTACTS_TOUCH_VELOCITY)
        {
            float kickScale = CONTACT_KICK;
            if constexpr (MAX_CONTACT_KICK > 0.0f)
            {
                const float wanted = scale * std::sqrt(distSq) * CONTACT_KICK;
                if (wanted > MAX_CONTACT_KICK)
                {
                    kickScale = CONTACT_KICK * (MAX_CONTACT_KICK / wanted);
                }
            }

            velIX += moveX * kickScale;
            velIY += moveIY * kickScale;
            vx[j] -= moveX * kickScale;
            vy[j] -= moveJY * kickScale;

            // 3. TANGENTIAL FRICTION, COMPILED OUT ENTIRELY WHEN IT IS OFF. WITH IT ON A
            //    HEAP HOLDS AN ANGLE OF REPOSE; WITH IT OFF THE PARTICLES SLIDE OVER ONE
            //    ANOTHER AND THE MATERIAL FLOWS.
            if constexpr (PARTICLE_FRICTION > 0.0f)
            {
                const float invDist = 1.0f / std::sqrt(distSq);
                const float nx = dx * invDist;
                const float ny = dy * invDist;

                const float relVx = velIX - vx[j];
                const float relVy = velIY - vy[j];
                const float relVn = relVx * nx + relVy * ny;

                const float tanVx = relVx - relVn * nx;
                const float tanVy = relVy - relVn * ny;
                const float fricX = PARTICLE_FRICTION * tanVx;
                const float fricY = PARTICLE_FRICTION * tanVy;

                velIX -= fricX * 0.5f;
                velIY -= fricY * weightI;
                vx[j] += fricX * 0.5f;
                vy[j] += fricY * weightJ;
            }
        }
    }
} // namespace

// ---------------------------   PUBLIC FUNCTIONS   ---------------------------

void Solver::preComputeInitialValues()
{
    ThreadPool::start(hw::suggestedThreadCount());

    // ONE PRIVATE HISTOGRAM PER THREAD. THE STRIDE IS ROUNDED TO 16 WORDS SO EVERY
    // THREAD'S BLOCK STARTS ON ITS OWN CACHE LINE; SHARING A LINE ACROSS TWO
    // HISTOGRAMS WOULD PUT TWO CORES ON THE SAME LINE FOR THE WHOLE COUNTING PASS.
    cursorStride = (CELL_RANGE_SIZE + 15u) & ~15u;

    const size_t words = static_cast<size_t>(cursorStride) * ThreadPool::threadCount();
    cellCursors = static_cast<uint32_t *>(
        ::operator new[](words * sizeof(uint32_t), std::align_val_t(64)));
    std::fill(cellCursors, cellCursors + words, 0u);

    // EVERY HISTOGRAM IS ALL ZEROES, SO EVERY WINDOW STARTS OUT EMPTY
    histogramLo.fill(UINT32_MAX);
    histogramHi.fill(0u);

    currentParticlesCount = 0;
    usingBufferA = true;
    bindBuffers();

    bandCount = 0;

    // AN EMPTY GRID: EVERY CELL RANGE IS [0, 0), SO THE FIRST SPAWN QUERY FINDS
    // NOTHING AND THE FIRST COLLISION SWEEP TOUCHES NOTHING.
    std::fill(cellRange.begin(), cellRange.end(), 0u);
    std::fill(rowWork.begin(), rowWork.end(), 0ull);

    std::fill(bufferA.posX.begin(), bufferA.posX.end(), 0.0f);
    std::fill(bufferA.posY.begin(), bufferA.posY.end(), 0.0f);
    std::fill(bufferA.velX.begin(), bufferA.velX.end(), 0.0f);
    std::fill(bufferA.velY.begin(), bufferA.velY.end(), 0.0f);
    std::fill(bufferA.heat.begin(), bufferA.heat.end(), 0.0f);
}

void Solver::shutdown()
{
    ThreadPool::stop();

    if (cellCursors != nullptr)
    {
        ::operator delete[](cellCursors, std::align_val_t(64));
        cellCursors = nullptr;
    }
}

uint32_t Solver::generateParticles(const uint32_t count)
{
    // THE NOZZLE CHOKES AT ITS PHYSICAL THROUGHPUT. ASKING FOR MORE THAN IT CAN
    // PASS USED TO STACK PARTICLES INSIDE ONE ANOTHER AT THE SPAWN POINT.
    uint32_t wanted = std::min(count, EMITTER_CAPACITY);
    wanted = std::min(wanted, TOTAL_PARTICLES_COUNT - currentParticlesCount);

    if (wanted == 0)
    {
        return 0;
    }

    // UNIT VECTOR ACROSS THE NOZZLE MOUTH
    constexpr float perpX = -EMITTER_DIR_Y;
    constexpr float perpY = EMITTER_DIR_X;

    // RINGS ARE SPREAD EVENLY OVER THE DISTANCE THE STREAM COVERS IN ONE FRAME,
    // SO CONSECUTIVE FRAMES BUTT UP AGAINST EACH OTHER AND THE JET READS AS ONE
    // CONTINUOUS BODY OF MATERIAL RATHER THAN A BURST OF CLUMPS.
    constexpr float ringStride = EMITTER_STEP / static_cast<float>(EMITTER_RINGS);
    constexpr float laneCentre = 0.5f * static_cast<float>(EMITTER_LANES - 1);

    float *__restrict px = Particle::posX;
    float *__restrict py = Particle::posY;
    float *__restrict vx = Particle::velX;
    float *__restrict vy = Particle::velY;
    float *__restrict heat = Particle::heat;

    uint32_t emitted = 0;

    for (uint32_t k = 0; k < wanted; k++)
    {
        const uint32_t lane = k % EMITTER_LANES;
        const uint32_t ring = k / EMITTER_LANES;

        // ODD RINGS ARE OFFSET BY HALF A LANE. A SQUARE LATTICE IS VISIBLE AS A GRID
        // OF DOTS FOR THE FIRST COUPLE OF HUNDRED PIXELS OUT OF THE NOZZLE AND READS
        // AS PRINTED RATHER THAN POURED; STAGGERING IT PACKS HEXAGONALLY, WHICH IS
        // WHAT LOOSE MATERIAL ACTUALLY DOES, AND BREAKS THE PATTERN UP.
        const float stagger = (ring & 1u) ? (0.5f * SPAWN_SPACING) : 0.0f;

        const float across = (static_cast<float>(lane) - laneCentre) * SPAWN_SPACING + stagger;
        const float along = static_cast<float>(ring) * ringStride;

        const float x = EMITTER_POS_X + perpX * across + EMITTER_DIR_X * along;
        const float y = EMITTER_POS_Y + perpY * across + EMITTER_DIR_Y * along;

        // IF THE PILE HAS BACKED UP INTO THE NOZZLE, DO NOT FIRE INTO IT.
        if (!isSpawnSlotFree(x, y))
        {
            continue;
        }

        const float speed = EMITTER_SPEED * (1.0f + EMITTER_SPEED_JITTER * randSigned());
        const float spread = EMITTER_SPREAD * randSigned();

        const uint32_t idx = currentParticlesCount + emitted;

        px[idx] = x;
        py[idx] = y;
        vx[idx] = EMITTER_DIR_X * speed + perpX * spread;
        vy[idx] = EMITTER_DIR_Y * speed + perpY * spread;

        // BORN AT ITS OWN ENERGY RATHER THAN WARMING UP FROM COLD ON THE WAY OUT
        heat[idx] = heatOf(vx[idx], vy[idx]);

        emitted++;
    }

    currentParticlesCount += emitted;
    return emitted;
}

void Solver::updateSimulationState()
{
    // THE HEAD OF THE PIPELINE. FROM HERE ON, THE WALL PASS THAT CLOSES A SUB-STEP
    // AND THE INTEGRATION THAT OPENS THE NEXT ONE ARE THE SAME LOOP: BOTH ARE
    // PURELY PER-PARTICLE, SO FUSING THEM IS THE IDENTICAL ORDER OF OPERATIONS
    // WITH ONE FEWER READ-AND-WRITE SWEEP OVER HALF A MEGABYTE PER SUB-STEP.
    integrateAndBound<false>(SUB_DT);

    for (uint8_t s = 0; s < SUB_STEPS; s++)
    {
        buildGrid();

        // SYMMETRIC RELAXATION. MOMENTUM CONSERVING, BOTH PARTICLES FREE TO MOVE,
        // NOBODY PINNED. THIS IS THE PASS THAT CARRIES THE RESTITUTION, SO IT IS
        // WHERE ALL THE SPRING IN THE SIMULATION COMES FROM.
        for (uint8_t iter = 0; iter < COLLISION_ITERATIONS; iter++)
        {
            solveContacts<false>();
        }

        // THE BRACE PASS. WITHOUT IT A PILE MORE THAN A FEW DOZEN LAYERS DEEP
        // COMPRESSES INTO ITSELF. SEE VERTICAL_BRACE IN defines.hpp.
        if constexpr (VERTICAL_BRACE > 0.0f)
        {
            // PIN THE FLOOR FIRST, SO THE BOTTOM LAYER IS ALREADY AT REST WHEN THE
            // PASS STARTS PROPAGATING SUPPORT UPWARDS FROM IT.
            applyBoundaries<false>();
            solveContacts<true>();
        }

        // CONTACTS CAN SHOVE A PARTICLE THROUGH A WALL, SO THE DOMAIN IS THE LAST
        // CONSTRAINT ENFORCED AND IT IS ENFORCED ABSOLUTELY.
        if (s + 1u < SUB_STEPS)
        {
            integrateAndBound<true>(SUB_DT);
        }
        else
        {
            applyBoundaries<true>();
        }
    }

    updateHeat();
}

void Solver::mousePush(const Vector2 &pos, const float radius)
{
    const float radiusSq = radius * radius;
    const float impulse = MOUSE_FORCE * SIMULATION_DT;
    const Vector2 centre = pos;

    ThreadPool::parallelFor(
        currentParticlesCount, PARALLEL_MIN_PARTICLES,
        [radiusSq, impulse, radius, centre](uint32_t begin, uint32_t end) {
            float *__restrict px = Particle::posX;
            float *__restrict py = Particle::posY;
            float *__restrict vx = Particle::velX;
            float *__restrict vy = Particle::velY;

            for (uint32_t i = begin; i < end; i++)
            {
                const float dx = px[i] - centre.x;
                const float dy = py[i] - centre.y;
                const float distSq = dx * dx + dy * dy;

                if (distSq < radiusSq && distSq > 1e-4f)
                {
                    const float dist = std::sqrt(distSq);
                    const float falloff = 1.0f - (dist / radius);
                    const float kick = impulse * falloff / dist;

                    vx[i] += dx * kick;
                    vy[i] += dy * kick;
                }
            }
        });
}

void Solver::mousePull(const Vector2 &pos, const float radius)
{
    const float radiusSq = radius * radius;
    const float impulse = MOUSE_FORCE * SIMULATION_DT;
    const Vector2 centre = pos;

    ThreadPool::parallelFor(
        currentParticlesCount, PARALLEL_MIN_PARTICLES,
        [radiusSq, impulse, radius, centre](uint32_t begin, uint32_t end) {
            float *__restrict px = Particle::posX;
            float *__restrict py = Particle::posY;
            float *__restrict vx = Particle::velX;
            float *__restrict vy = Particle::velY;

            for (uint32_t i = begin; i < end; i++)
            {
                const float dx = centre.x - px[i];
                const float dy = centre.y - py[i];
                const float distSq = dx * dx + dy * dy;

                if (distSq < radiusSq && distSq > 1e-4f)
                {
                    const float dist = std::sqrt(distSq);
                    const float falloff = 1.0f - (dist / radius);
                    const float kick = impulse * falloff / dist;

                    vx[i] += dx * kick;
                    vy[i] += dy * kick;

                    // EVERYTHING FALLING TOWARDS ONE POINT PICKS UP ENORMOUS SPEED AT
                    // THE CENTRE. BLEED IT OFF SO THE CURSOR GATHERS A BLOB INSTEAD OF
                    // FIRING ONE OUT THE OTHER SIDE.
                    vx[i] *= MOUSE_PULL_DAMPING;
                    vy[i] *= MOUSE_PULL_DAMPING;
                }
            }
        });
}

// ---------------------------   PRIVATE FUNCTIONS   ---------------------------

void Solver::bindBuffers()
{
    ParticleBuffer &live = usingBufferA ? bufferA : bufferB;

    Particle::posX = live.posX.data();
    Particle::posY = live.posY.data();
    Particle::velX = live.velX.data();
    Particle::velY = live.velY.data();
    Particle::heat = live.heat.data();
}

template <bool SettleFirst>
void Solver::integrateAndBound(const float dt)
{
    const float gravityStep = GRAVITY * dt;

    ThreadPool::parallelFor(
        currentParticlesCount, PARALLEL_MIN_PARTICLES,
        [dt, gravityStep](uint32_t begin, uint32_t end) {
            // RE-DERIVED INSIDE THE WORKER: __restrict DOES NOT SURVIVE A LAMBDA
            // CAPTURE, AND WITHOUT IT THE COMPILER ASSUMES THE FOUR STREAMS MAY
            // ALIAS AND REFUSES TO VECTORISE ANY OF THIS.
            float *__restrict px = Particle::posX;
            float *__restrict py = Particle::posY;
            float *__restrict vx = Particle::velX;
            float *__restrict vy = Particle::velY;

            for (uint32_t i = begin; i < end; i++)
            {
                float x = px[i];
                float y = py[i];
                float velocityX = vx[i];
                float velocityY = vy[i];

                // THE TAIL OF THE PREVIOUS SUB-STEP, FOLDED IN SO THE PARTICLE IS
                // TOUCHED ONCE INSTEAD OF TWICE. THE SWEEP THAT JUST RAN ONLY MOVED THE
                // POSITION, SO THE KICK IT GAVE IS PICKED UP FIRST.
                if constexpr (SettleFirst)
                {
                    decodeKick(x, y, velocityX, velocityY);
                    resolveBounds(x, y, velocityX, velocityY);
                }

                velocityY += gravityStep;

                // SAFETY VALVE. A PARTICLE THAT CROSSES MORE THAN A CELL IN ONE SUB-STEP CAN
                // TUNNEL STRAIGHT THROUGH A NEIGHBOUR BEFORE THE GRID EVER SEES THE CONTACT.
                // A SELECT RATHER THAN AN if, SO THE LOOP STAYS VECTORISED.
                const float speedSq = velocityX * velocityX + velocityY * velocityY;
                const float speedScale = (speedSq > MAX_SPEED_SQ) ? MAX_SPEED / std::sqrt(speedSq) : 1.0f;
                velocityX *= speedScale;
                velocityY *= speedScale;

                x += velocityX * dt;
                y += velocityY * dt;

                resolveBounds(x, y, velocityX, velocityY);

                // THE CONTACT SWEEP COMES NEXT AND WILL ONLY MOVE x AND y
                encodeKick(x, y, velocityX, velocityY);

                px[i] = x;
                py[i] = y;
                vx[i] = velocityX;
                vy[i] = velocityY;
            }
        });
}

// A PARALLEL COUNTING SORT THAT PRODUCES BYTE-FOR-BYTE THE SAME ORDERING THE
// SERIAL ONE DID.
//
// THE TRICK IS THE PRIVATE CURSOR BLOCK. EACH THREAD COUNTS ITS OWN SLICE OF THE
// PARTICLES INTO ITS OWN HISTOGRAM, THE PREFIX SUM THEN HANDS THREAD t THE EXACT
// OUTPUT SLOTS ITS SLICE OWNS IN EVERY CELL, AND THE SCATTER RUNS WITH NO ATOMICS
// AND NO CONTENTION AT ALL. BECAUSE THE OFFSETS ARE LAID OUT IN THREAD ORDER AND
// THE SLICES ARE IN INDEX ORDER, THE SORT STAYS STABLE, WHICH IS WHAT KEEPS THE
// RESULT IDENTICAL TO THE SINGLE-THREADED VERSION.
//
// THE PARTICLES ARE STILL IN LAST SUB-STEP'S ORDER WHEN THIS RUNS, SO A THREAD'S
// SLICE ONLY EVER LANDS IN ITS OWN QUARTER OR SO OF THE ROWS. EACH HISTOGRAM IS
// THEREFORE CLEARED, MERGED AND REWRITTEN ONLY INSIDE THE WINDOW OF CELLS IT WAS
// ACTUALLY WRITTEN IN, NOT ACROSS ALL HUNDRED THOUSAND CELLS FOUR TIMES OVER. THE
// REBUILD WENT FROM 3.6 ms A FRAME TO 2.3 ms.
void Solver::buildGrid()
{
    const uint32_t count = currentParticlesCount;
    const uint32_t threads = ThreadPool::threadCount();

    const uint32_t stride = cursorStride;
    const uint32_t workers = (threads > 1u && count >= PARALLEL_MIN_PARTICLES) ? threads : 1u;
    uint32_t *const tally = cellCursors;

    // ---- PASS 1: WHICH CELL EACH PARTICLE IS IN, AND A PRIVATE HISTOGRAM ----
    const auto countPass = [count, stride, workers, tally](uint32_t worker) {
        uint32_t *__restrict mine = tally + static_cast<size_t>(worker) * stride;

        // ONLY WHAT THE LAST REBUILD LEFT BEHIND NEEDS WIPING
        if (histogramLo[worker] <= histogramHi[worker])
        {
            std::fill(mine + histogramLo[worker], mine + histogramHi[worker] + 1u, 0u);
        }

        uint32_t begin;
        uint32_t end;
        ThreadPool::sliceOf(count, worker, workers, begin, end);

        const float *__restrict px = Particle::posX;
        const float *__restrict py = Particle::posY;
        uint32_t *__restrict ids = particleCellIDs.data();

        uint32_t lo = UINT32_MAX;
        uint32_t hi = 0u;

        for (uint32_t i = begin; i < end; i++)
        {
            int32_t cellX = static_cast<int32_t>(px[i] * INV_CELL_SIZE);
            int32_t cellY = static_cast<int32_t>(py[i] * INV_CELL_SIZE);

            cellX = std::clamp(cellX, 0, static_cast<int32_t>(GRID_WIDTH) - 1);
            cellY = std::clamp(cellY, 0, static_cast<int32_t>(GRID_HEIGHT) - 1);

            const uint32_t id = static_cast<uint32_t>(cellY + 1) * PADDED_GRID_WIDTH +
                                static_cast<uint32_t>(cellX + 1);

            ids[i] = id;
            mine[id]++;
            lo = std::min(lo, id);
            hi = std::max(hi, id);
        }

        histogramLo[worker] = lo;
        histogramHi[worker] = hi;
    };

    // ---- PASS 2a: PER-CHUNK MERGE, LOCAL PREFIX AND ROW WORK ----
    // ONLY THE PART OF EACH THREAD'S WINDOW THAT FALLS INSIDE THIS CHUNK IS READ.
    const auto mergePass = [stride, workers, tally](uint32_t worker) {
        uint32_t rowBegin;
        uint32_t rowEnd;
        rowSliceOf(worker, workers, rowBegin, rowEnd);

        const uint32_t first = rowBegin * PADDED_GRID_WIDTH;
        const uint32_t last = rowEnd * PADDED_GRID_WIDTH;

        uint32_t *__restrict range = cellRange.data();
        std::fill(range + first, range + last, 0u);

        for (uint32_t t = 0; t < workers; t++)
        {
            const uint32_t lo = std::max(first, histogramLo[t]);
            const uint32_t hiEnd = std::min(last, histogramHi[t] + 1u);
            const uint32_t *__restrict src = tally + static_cast<size_t>(t) * stride;

            for (uint32_t c = lo; c < hiEnd; c++)
            {
                range[c] += src[c];
            }
        }

        uint32_t running = 0;
        for (uint32_t r = rowBegin; r < rowEnd; r++)
        {
            const uint32_t rowFirst = r * PADDED_GRID_WIDTH;
            uint64_t work = 0;

            for (uint32_t c = rowFirst; c < rowFirst + PADDED_GRID_WIDTH; c++)
            {
                const uint32_t n = range[c];
                range[c] = running;
                running += n;

                // PAIR TESTS OUT OF A CELL GO AS THE SQUARE OF ITS OCCUPANCY. THE
                // OCCUPANCY IS IN HAND RIGHT HERE, SO THE BAND PLANNER'S WEIGHTS ARE
                // COLLECTED FOR NOTHING.
                work += static_cast<uint64_t>(n) * n;
            }

            rowWork[r] = work;
        }

        chunkBase[worker] = running;
    };

    // ---- PASS 2c: LIFT THE LOCAL PREFIXES TO GLOBAL AND CUT THE CURSORS ----
    // THREADS TAKE THEIR SLOTS INSIDE A CELL IN THREAD ORDER, WHICH IS WHAT KEEPS THE
    // SORT STABLE. runningCursor REMEMBERS HOW FAR INTO EACH CELL THE EARLIER THREADS
    // REACH, SO EACH THREAD ONLY HAS TO WALK ITS OWN WINDOW.
    const auto cursorPass = [stride, workers, tally](uint32_t worker) {
        uint32_t rowBegin;
        uint32_t rowEnd;
        rowSliceOf(worker, workers, rowBegin, rowEnd);

        const uint32_t first = rowBegin * PADDED_GRID_WIDTH;
        const uint32_t last = rowEnd * PADDED_GRID_WIDTH;
        const uint32_t base = chunkBase[worker];

        uint32_t *__restrict range = cellRange.data();
        uint32_t *__restrict running = runningCursor.data();

        for (uint32_t c = first; c < last; c++)
        {
            const uint32_t start = base + range[c];
            range[c] = start;
            running[c] = start;
        }

        for (uint32_t t = 0; t < workers; t++)
        {
            const uint32_t lo = std::max(first, histogramLo[t]);
            const uint32_t hiEnd = std::min(last, histogramHi[t] + 1u);
            uint32_t *__restrict slot = tally + static_cast<size_t>(t) * stride;

            for (uint32_t c = lo; c < hiEnd; c++)
            {
                const uint32_t n = slot[c];
                slot[c] = running[c];
                running[c] += n;
            }
        }
    };

    // ---- PASS 3: SCATTER STRAIGHT INTO THE OTHER BUFFER SET ----
    // READS ARE SEQUENTIAL AND WRITES ARE ALMOST SEQUENTIAL, BECAUSE THE PARTICLES
    // WERE ALREADY SORTED LAST SUB-STEP AND HAVE MOVED LESS THAN A PIXEL SINCE.
    // ONE PASS, NO PERMUTATION ARRAY, NO COPY BACK.
    const ParticleBuffer &src = usingBufferA ? bufferA : bufferB;
    ParticleBuffer &dst = usingBufferA ? bufferB : bufferA;

    const auto scatterPass = [count, stride, workers, tally, &src, &dst](uint32_t worker) {
        uint32_t begin;
        uint32_t end;
        ThreadPool::sliceOf(count, worker, workers, begin, end);

        uint32_t *__restrict cursor = tally + static_cast<size_t>(worker) * stride;
        const uint32_t *__restrict ids = particleCellIDs.data();

        for (uint32_t i = begin; i < end; i++)
        {
            const uint32_t slot = cursor[ids[i]]++;

            dst.posX[slot] = src.posX[i];
            dst.posY[slot] = src.posY[i];
            dst.velX[slot] = src.velX[i];
            dst.velY[slot] = src.velY[i];
            dst.heat[slot] = src.heat[i];
        }
    };

    if (workers <= 1u)
    {
        countPass(0u);
        mergePass(0u);
        chunkBase[0] = 0u;
        cursorPass(0u);
        scatterPass(0u);
    }
    else
    {
        ThreadPool::dispatch(countPass);
        ThreadPool::dispatch(mergePass);

        // THE ONLY SERIAL STEP LEFT IS AN EXCLUSIVE SCAN OF ONE NUMBER PER THREAD.
        uint32_t running = 0;
        for (uint32_t t = 0; t < workers; t++)
        {
            const uint32_t chunk = chunkBase[t];
            chunkBase[t] = running;
            running += chunk;
        }

        ThreadPool::dispatch(cursorPass);
        ThreadPool::dispatch(scatterPass);
    }

    // THE CELL RANGE TABLE IS READ ONE PAST THE LAST REAL CELL'S BOTTOM-RIGHT
    // NEIGHBOUR. THOSE SPARE SLOTS ALL SIT AT THE END OF THE PARTICLE LIST.
    for (uint32_t c = PADDED_TOTAL_CELLS; c < CELL_RANGE_SIZE; c++)
    {
        cellRange[c] = count;
    }

    usingBufferA = !usingBufferA;
    bindBuffers();

    planBands();
}

// SPLITS THE GRID INTO TWO INTERLEAVED ROW BANDS PER THREAD AND SWEEPS THEM IN
// TWO PASSES: ALL THE ODD-INDEXED BANDS, THEN ALL THE EVEN-INDEXED ONES.
//
// WHY ROWS AND NOT COLUMNS: THE PARTICLES ARE SORTED ROW-MAJOR, SO A BAND OF ROWS
// IS ONE CONTIGUOUS RUN OF THE PARTICLE ARRAYS. A BAND OF COLUMNS WOULD BE ONE
// SHORT STRIDED RUN PER ROW, AND EVERY BAND EDGE WOULD SHARE A CACHE LINE WITH ITS
// NEIGHBOUR ON ALL 166 ROWS. WITH ROWS THERE IS EXACTLY ONE SHARED LINE PER EDGE.
//
// WHY TWO PASSES: SWEEPING ROW r READS AND WRITES ROWS r AND r + 1, SO A BAND
// REACHES ONE ROW INTO ITS LOWER NEIGHBOUR. RUNNING ONLY EVERY SECOND BAND AT A
// TIME LEAVES A WHOLE UNTOUCHED BAND BETWEEN ANY TWO RUNNING ONES, WHICH MAKES
// THEIR FOOTPRINTS DISJOINT: NO LOCKS, NO ATOMICS, AND NOTHING SHARED. AN EARLIER
// VERSION USED FAT BANDS PLUS A SECOND PASS OVER THE SINGLE SEAM ROWS BETWEEN
// THEM; THAT SECOND PASS HAD ONLY (threads - 1) ROWS OF WORK TO SPREAD OVER ALL
// THE THREADS AND COST 11% OF THE FRAME FOR 2% OF THE GRID. EQUAL BANDS COST THE
// SAME TOTAL WORK BUT KEEP EVERY CORE BUSY IN BOTH PASSES.
//
// THE ODD BANDS GO FIRST BECAUSE EACH ONE SITS BELOW ITS EVEN PREDECESSOR: BY THE
// TIME AN EVEN BAND SWEEPS ITS BOTTOM ROW, THE BAND UNDERNEATH HAS ALREADY BEEN
// SETTLED THIS PASS AND ITS SUPPORT IS THERE TO PUSH BACK.
//
// WHAT THIS COSTS: GAUSS-SEIDEL SUPPORT NO LONGER TRAVELS THE FULL HEIGHT OF THE
// BOX IN ONE SWEEP, ONLY THE HEIGHT OF A BAND. THAT IS STILL TENS OF ROWS, AND THE
// FRAME RUNS SUB_STEPS x COLLISION_ITERATIONS SWEEPS, SO A PILE STILL STANDS. EVERY
// CONTACT IS STILL FOUND AND STILL FULLY RESOLVED; ONLY THE ORDER CHANGES.
void Solver::planBands()
{
    const uint32_t threads = ThreadPool::threadCount();

    // ONE BAND PER THREAD PER PASS, AND EVERY BAND NEEDS AT LEAST ONE ROW.
    uint32_t perPass = threads;
    while (perPass > 1u && static_cast<uint32_t>(GRID_HEIGHT) < perPass * 2u)
    {
        --perPass;
    }

    if (perPass <= 1u || currentParticlesCount < PARALLEL_MIN_PARTICLES)
    {
        bandCount = 0; // 0 MEANS "SWEEP THE WHOLE GRID ON THE CALLING THREAD"
        return;
    }

    const uint32_t wanted = perPass * 2u;

    uint64_t total = 0;
    for (int32_t r = 1; r <= GRID_HEIGHT; r++)
    {
        total += rowWork[static_cast<uint32_t>(r)];
    }

    bandCount = wanted;

    int32_t row = 1;
    uint64_t placed = 0;

    for (uint32_t b = 0; b < wanted; b++)
    {
        if (b + 1u == wanted)
        {
            bands[b] = {row, GRID_HEIGHT};
            break;
        }

        const uint32_t bandsLeft = wanted - b;
        const uint64_t target = (total - placed) / bandsLeft;

        // THE FURTHEST THIS BAND MAY REACH AND STILL LEAVE ONE ROW FOR EVERY BAND
        // THAT COMES AFTER IT.
        const int32_t lastAllowedEnd = GRID_HEIGHT - static_cast<int32_t>(bandsLeft - 1u);

        int32_t end = row;
        uint64_t taken = rowWork[static_cast<uint32_t>(row)];

        while (end < lastAllowedEnd && taken < target)
        {
            ++end;
            taken += rowWork[static_cast<uint32_t>(end)];
        }

        // THE LOOP ABOVE STOPS ON THE FIRST ROW THAT CROSSES THE TARGET, WHICH
        // OVERSHOOTS BY UP TO A WHOLE ROW EVERY TIME AND PUSHES THE ERROR DOWN
        // INTO THE LAST BAND. IF STOPPING ONE ROW SHORT LANDS CLOSER, DO THAT.
        if (end > row && taken >= target)
        {
            const uint64_t shorter = taken - rowWork[static_cast<uint32_t>(end)];
            if (target - shorter < taken - target)
            {
                taken = shorter;
                --end;
            }
        }

        bands[b] = {row, end};
        placed += taken;
        row = end + 1;
    }
}

template <bool Brace>
void Solver::sweepRows(const int32_t rowFrom, const int32_t rowTo)
{
    float *__restrict px = Particle::posX;
    float *__restrict py = Particle::posY;
    float *__restrict vx = Particle::velX;
    float *__restrict vy = Particle::velY;
    const uint32_t *__restrict range = cellRange.data();

    // SWEEP ROW BY ROW FROM THE FLOOR UPWARDS.
    //
    // THIS DIRECTION IS MANDATORY FOR THE SHOCK PASS: IT PINS THE LOWER PARTICLE OF
    // EVERY CONTACT, SO IT ONLY WORKS IF EVERYTHING BELOW HAS ALREADY BEEN SETTLED
    // THIS SWEEP. IT HELPS THE SYMMETRIC PASSES FOR THE SAME UNDERLYING REASON.
    //
    // GAUSS-SEIDEL CARRIES A CORRECTION ONE CONTACT FURTHER PER PASS. IN A PILE THE
    // SUPPORT COMES FROM THE GROUND, SO A TOP-DOWN SWEEP NEEDS ROUGHLY ONE PASS PER
    // STACKED LAYER BEFORE THE FLOOR'S PUSH REACHES THE TOP, AND A HUNDRED-DEEP PILE
    // SIMPLY NEVER CONVERGES. SWEEPING WITH THE SUPPORT DIRECTION CARRIES IT THROUGH
    // A WHOLE COLUMN IN A SINGLE PASS, FOR FREE.
    for (int32_t cellY = rowFrom; cellY >= rowTo; --cellY)
    {
        const uint32_t rowBase = static_cast<uint32_t>(cellY) * PADDED_GRID_WIDTH;

        // EVERY THIRD CELL, IN THREE PASSES. A CELL READS AND WRITES ONLY ITSELF, THE
        // CELL TO ITS RIGHT AND THE THREE BELOW, SO TWO CELLS THREE APART NEVER SHARE A
        // PARTICLE AND NEIGHBOURING ITERATIONS OF THIS LOOP DO NOT DEPEND ON EACH OTHER.
        // THE CPU CAN THEN WORK ON SEVERAL AT ONCE INSTEAD OF WAITING FOR EACH CONTACT
        // TO LAND BEFORE STARTING THE NEXT. ABOUT 10% FASTER, AND THE CHANGE OF ORDER
        // MOVES MEAN OVERLAP FROM 2.00% TO 2.12%.
        for (uint32_t phase = 0; phase < 3u; ++phase)
        for (uint32_t cellX = 1u + phase; cellX <= GRID_WIDTH; cellX += 3u)
        {
            const uint32_t cell = rowBase + cellX;
            const uint32_t begin = range[cell];
            const uint32_t end = range[cell + 1];

            if (begin == end)
            {
                continue;
            }

            // EACH UNORDERED PAIR IS VISITED EXACTLY ONCE BY ONLY EVER LOOKING AT
            // NEIGHBOURS WITH A HIGHER CELL INDEX: THE CELL TO THE RIGHT AND THE
            // THREE BELOW. THE OTHER FOUR NEIGHBOURS FIND US INSTEAD. THAT HALVES
            // THE PAIR TESTS AND DROPS THE p1 < p2 BRANCH ENTIRELY.
            //
            // AND BECAUSE THE GRID IS ROW-MAJOR AND THE PARTICLES ARE SORTED BY CELL
            // INDEX, THIS CELL AND THE ONE TO ITS RIGHT ARE ONE CONTIGUOUS RUN, AND
            // SO ARE THE THREE BELOW. NINE SCATTERED LOOKUPS PER PARTICLE BECOME TWO
            // STRAIGHT-LINE SCANS PER CELL.
            const uint32_t rightEnd = range[cell + 2];
            const uint32_t belowBegin = range[cell + PADDED_GRID_WIDTH - 1];
            const uint32_t belowEnd = range[cell + PADDED_GRID_WIDTH + 2];

            for (uint32_t i = begin; i < end; ++i)
            {
                // NO j IN EITHER RUN BELOW IS EVER EQUAL TO i (THE FIRST STARTS AT
                // i + 1, THE SECOND IS A DIFFERENT CELL), SO KEEPING i IN LOCALS FOR
                // THE DURATION IS STILL EXACT GAUSS-SEIDEL, NOT AN APPROXIMATION.
                float posIX = px[i];
                float posIY = py[i];
                float velIX = 0.0f;
                float velIY = 0.0f;
                if constexpr (CONTACTS_TOUCH_VELOCITY)
                {
                    velIX = vx[i];
                    velIY = vy[i];
                }

                for (uint32_t j = i + 1; j < rightEnd; ++j)
                {
                    solveContact<Brace>(posIX, posIY, velIX, velIY, j, px, py, vx, vy);
                }

                for (uint32_t j = belowBegin; j < belowEnd; ++j)
                {
                    solveContact<Brace>(posIX, posIY, velIX, velIY, j, px, py, vx, vy);
                }

                px[i] = posIX;
                py[i] = posIY;
                if constexpr (CONTACTS_TOUCH_VELOCITY)
                {
                    vx[i] = velIX;
                    vy[i] = velIY;
                }
            }
        }
    }
}

template <bool Brace>
void Solver::solveContacts()
{
    if (bandCount == 0u)
    {
        sweepRows<Brace>(GRID_HEIGHT, 1);
        return;
    }

    // PASS ONE: THE ODD BANDS, EACH ONE BELOW ITS EVEN NEIGHBOUR.
    ThreadPool::dispatch([](uint32_t worker) {
        const uint32_t b = worker * 2u + 1u;
        if (b < bandCount)
        {
            sweepRows<Brace>(bands[b].last, bands[b].first);
        }
    });

    // PASS TWO: THE EVEN BANDS. TOGETHER THE TWO PASSES SWEEP EVERY ROW EXACTLY
    // ONCE, SO NO CONTACT IS MISSED AND NONE IS RESOLVED TWICE.
    ThreadPool::dispatch([](uint32_t worker) {
        const uint32_t b = worker * 2u;
        if (b < bandCount)
        {
            sweepRows<Brace>(bands[b].last, bands[b].first);
        }
    });
}

template <bool Finish>
void Solver::applyBoundaries()
{
    ThreadPool::parallelFor(
        currentParticlesCount, PARALLEL_MIN_PARTICLES,
        [](uint32_t begin, uint32_t end) {
            float *__restrict px = Particle::posX;
            float *__restrict py = Particle::posY;
            float *__restrict vx = Particle::velX;
            float *__restrict vy = Particle::velY;

            for (uint32_t i = begin; i < end; i++)
            {
                float x = px[i];
                float y = py[i];
                float velocityX = vx[i];
                float velocityY = vy[i];

                decodeKick(x, y, velocityX, velocityY);
                resolveBounds(x, y, velocityX, velocityY);

                // A FRAME ENDS WITH PLAIN VELOCITY FOR THE EMITTER, THE MOUSE AND THE
                // HEAT PASS. INSIDE A SUB-STEP THE NEXT SWEEP STILL EXPECTS THE ENCODING.
                if constexpr (!Finish)
                {
                    encodeKick(x, y, velocityX, velocityY);
                }

                px[i] = x;
                py[i] = y;
                vx[i] = velocityX;
                vy[i] = velocityY;
            }
        });
}

// HEAT CLIMBS HEAT_RISE OF THE WAY TO A HIGHER ENERGY EACH FRAME AND SHEDS HEAT_FALL
// OF ITS EXCESS OVER A LOWER ONE. A JOLT THAT LASTS ONE FRAME ONLY SHOWS AT HEAT_RISE
// OF ITS ENERGY. THE CORE OF A REAL PUSH MOVES WELL PAST HEAT_HOT_SPEED AND ITS HEAT
// AIMS FOR HEAT_CEILING, SO IT STILL REACHES THE TOP COLOUR WITHIN A FEW FRAMES.
// fmax AND fmin PICK THE RATE WITHOUT A BRANCH.
void Solver::updateHeat()
{
    ThreadPool::parallelFor(
        currentParticlesCount, PARALLEL_MIN_PARTICLES,
        [](uint32_t begin, uint32_t end) {
            const float *__restrict vx = Particle::velX;
            const float *__restrict vy = Particle::velY;
            float *__restrict heat = Particle::heat;

            for (uint32_t i = begin; i < end; i++)
            {
                const float gap = heatOf(vx[i], vy[i]) - heat[i];
                heat[i] += std::fmax(gap, 0.0f) * HEAT_RISE + std::fmin(gap, 0.0f) * HEAT_FALL;
            }
        });
}

bool Solver::isSpawnSlotFree(const float x, const float y)
{
    if (currentParticlesCount == 0)
    {
        return true;
    }

    int32_t cellX = std::clamp(static_cast<int32_t>(x * INV_CELL_SIZE), 0, static_cast<int32_t>(GRID_WIDTH) - 1);
    int32_t cellY = std::clamp(static_cast<int32_t>(y * INV_CELL_SIZE), 0, static_cast<int32_t>(GRID_HEIGHT) - 1);

    const float *__restrict px = Particle::posX;
    const float *__restrict py = Particle::posY;

    for (int32_t dy = -1; dy <= 1; ++dy)
    {
        const uint32_t centre = static_cast<uint32_t>(cellY + 1 + dy) * PADDED_GRID_WIDTH +
                                static_cast<uint32_t>(cellX + 1);

        const uint32_t begin = cellRange[centre - 1];
        const uint32_t end = cellRange[centre + 2];

        for (uint32_t k = begin; k < end; ++k)
        {
            const float dx = px[k] - x;
            const float dy2 = py[k] - y;

            if (dx * dx + dy2 * dy2 < MIN_COLLISION_DIST_SQ)
            {
                return false;
            }
        }
    }

    return true;
}
