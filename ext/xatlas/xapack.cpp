// xapack.cpp  — pack-only UVs using xatlas (keeps your islands)
// Build: see CMake below or g++ one-liner later.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <utility>
#include <cmath>
#include <queue>
#include <algorithm> // std::max
#include <cfloat>
#include <vector>
#include "xatlas.h"
#include "fast_obj.h"   // fast_obj.h / fast_obj.c

static inline float tri_area3D(const float* a, const float* b, const float* c) {
    float ux=b[0]-a[0], uy=b[1]-a[1], uz=b[2]-a[2];
    float vx=c[0]-a[0], vy=c[1]-a[1], vz=c[2]-a[2];
    float cx=uy*vz-uz*vy, cy=uz*vx-ux*vz, cz=ux*vy-uy*vx;
    return 0.5f*std::sqrt(cx*cx+cy*cy+cz*cz);
}
static inline float tri_area2D(const float* a, const float* b, const float* c) {
    float ux=b[0]-a[0], uy=b[1]-a[1];
    float vx=c[0]-a[0], vy=c[1]-a[1];
    return 0.5f*std::fabs(ux*vy-uy*vx);
}

struct PairKey {
    uint32_t p; // position index
    uint32_t t; // texcoord index
    bool operator==(const PairKey& o) const { return p==o.p && t==o.t; }
};
struct PairKeyHash {
    size_t operator()(const PairKey& k) const noexcept {
        return (size_t)k.p * 73856093u ^ (size_t)k.t * 19349663u;
    }
};

static void die(const char* msg) {
    std::fprintf(stderr, "%s\n", msg);
    std::exit(1);
}

// outPath: destination OBJ
// atlas  : xatlas::Atlas* after PackCharts()
// V      : your unified positions (size = 3 * unifiedVertexCount)
// pad_px : pixel guard to keep UVs away from 0/1 borders (1..8 typical)
static void writePackedOBJ(const char* outPath,
                                  const xatlas::Atlas* atlas,
                                  const std::vector<float>& V,
                                  uint32_t pad_px = 1)
{
    const xatlas::Mesh& am = atlas->meshes[0];
    if (am.vertexCount == 0 || !am.vertexArray || am.indexCount == 0 || !am.indexArray) {
        std::fprintf(stderr, "xapack: empty atlas output\n");
        return;
    }

    // --- rebuild positions in atlas order using xref ---
    std::vector<float> V_atlas(3 * am.vertexCount);
    for (uint32_t i = 0; i < am.vertexCount; ++i) {
        const uint32_t o = am.vertexArray[i].xref;     // original unified-vertex id
        V_atlas[3*i+0] = V[3*o+0];
        V_atlas[3*i+1] = V[3*o+1];
        V_atlas[3*i+2] = V[3*o+2];
    }

    // --- compute UV bounds per page (atlasIndex), since your header has no width/height ---
    int pages = 1;
    for (uint32_t i = 0; i < am.vertexCount; ++i)
        pages = std::max(pages, int(am.vertexArray[i].atlasIndex) + 1);

    if (pages > 1) {
        std::fprintf(stderr, "[warn] xatlas produced %d pages; "
                             "this writer maps every page into 0..1 (overlap). "
                             "Increase resolution or pre-scale so it fits one page.\n", pages);
    }

    std::vector<float> umin(pages,  FLT_MAX), vmin(pages,  FLT_MAX);
    std::vector<float> umax(pages, -FLT_MAX), vmax(pages, -FLT_MAX);
    for (uint32_t i = 0; i < am.vertexCount; ++i) {
        const int a = std::max(0, am.vertexArray[i].atlasIndex);
        const float u = am.vertexArray[i].uv[0];
        const float v = am.vertexArray[i].uv[1];
        umin[a] = std::min(umin[a], u); umax[a] = std::max(umax[a], u);
        vmin[a] = std::min(vmin[a], v); vmax[a] = std::max(vmax[a], v);
    }

    // guard in the same "pixel units" as vertexArray.uv
    const float guard_px = std::max(1.0f, float(pad_px));

    FILE* f = std::fopen(outPath, "w");
    if (!f) { perror("fopen"); return; }

    // v (positions in atlas vertex order)
    for (uint32_t i = 0; i < am.vertexCount; ++i)
        std::fprintf(f, "v %g %g %g\n",
            V_atlas[3*i+0], V_atlas[3*i+1], V_atlas[3*i+2]);

    // vt (normalize per page with a border; NO clamp)
    const bool flipV = false;
    for (uint32_t i = 0; i < am.vertexCount; ++i) {
        const int a = std::max(0, am.vertexArray[i].atlasIndex);
        const float W = std::max(umax[a] - umin[a], 1e-6f);
        const float H = std::max(vmax[a] - vmin[a], 1e-6f);
        float u = (am.vertexArray[i].uv[0] - umin[a] + guard_px) / (W + 2.0f*guard_px);
        float v = (am.vertexArray[i].uv[1] - vmin[a] + guard_px) / (H + 2.0f*guard_px);
        if (flipV) v = 1.0f - v;
        std::fprintf(f, "vt %.7g %.7g\n", u, v);
    }

    // f (use atlas indices for both v and vt)
    for (uint32_t i = 0; i < am.indexCount; i += 3)
        std::fprintf(f, "f %u/%u %u/%u %u/%u\n",
            am.indexArray[i+0]+1, am.indexArray[i+0]+1,
            am.indexArray[i+1]+1, am.indexArray[i+1]+1,
            am.indexArray[i+2]+1, am.indexArray[i+2]+1);

    std::fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 6 || argc > 8) {
        std::fprintf(stderr,
            "Usage:\n"
            "  %s in.obj out.obj <resolution> <padding_px> <rotate 0|1> [texelsPerUnit] [bruteForce 0|1]\n"
            "Notes:\n"
            "  texelsPerUnit=0 keeps your current scale; >0 enforces uniform density.\n",
            argv[0]);
        return 2;
    }
    const char* inPath   = argv[1];
    const char* outPath  = argv[2];
    const uint32_t res   = (uint32_t)std::strtoul(argv[3], nullptr, 10);
    const uint32_t pad   = (uint32_t)std::strtoul(argv[4], nullptr, 10);
    const bool rotate    = std::strtoul(argv[5], nullptr, 10) != 0;
    const float tpu      = (argc >= 7) ? std::strtof(argv[6], nullptr) : 0.0f; // 0 = keep scale
    const bool brute     = (argc >= 8) ? (std::strtoul(argv[7], nullptr, 10) != 0) : false;

    fastObjMesh* m = fast_obj_read(inPath);
    if (!m) die("Failed to read input OBJ");

    if (m->face_count == 0) die("No faces in OBJ");
    if (m->position_count == 0 || m->texcoord_count == 0)
        die("OBJ must contain positions and texcoords (vt).");

    // Build unified (position, texcoord) vertex stream so v/vt indices align 1:1.
    std::vector<float> V;  // xyz for unified vertices
    std::vector<float> UV; // uv  for unified vertices
    std::vector<uint32_t> I; // unified triangle indices (xatlas expects triangles)
    V.reserve(m->index_count * 3);
    UV.reserve(m->index_count * 2);
    I.reserve(m->index_count);

    std::unordered_map<PairKey, uint32_t, PairKeyHash> lut;
    lut.reserve(m->index_count);

    // fast_obj stores faces as a flat index array of triplets (p,t,n) per corner.
    size_t idx = 0;
    for (unsigned f = 0; f < m->face_count; ++f) {
        const unsigned fv = m->face_vertices[f];
        if (fv != 3) die("Non-triangle face found. Re-export triangulated OBJ.");
        for (int k = 0; k < 3; ++k) {
            const fastObjIndex io = m->indices[idx++]; // io.p, io.t, io.n (0-based)
            PairKey key{ (uint32_t)io.p, (uint32_t)io.t };
            auto it = lut.find(key);
            uint32_t vid;
            if (it == lut.end()) {
                vid = (uint32_t)(V.size() / 3);
                lut.emplace(key, vid);
                // append position
                V.push_back(m->positions[io.p * 3 + 0]);
                V.push_back(m->positions[io.p * 3 + 1]);
                V.push_back(m->positions[io.p * 3 + 2]);
                // append texcoord
                UV.push_back(m->texcoords[io.t * 2 + 0]);
                UV.push_back(m->texcoords[io.t * 2 + 1]);
            } else {
                vid = it->second;
            }
            I.push_back(vid);
        }
    }

    // Build face adjacency by shared vertex (seams are split, so this == UV islands)
    const uint32_t faceCount = (uint32_t)(I.size() / 3);
    std::vector<std::vector<uint32_t>> adj(faceCount);
    {
        const uint32_t vertCount = (uint32_t)(UV.size() / 2);
        std::vector<std::vector<uint32_t>> vf(vertCount); // vertex -> faces
        for (uint32_t f = 0; f < faceCount; ++f) {
            uint32_t i0 = I[3*f+0], i1 = I[3*f+1], i2 = I[3*f+2];
            vf[i0].push_back(f); vf[i1].push_back(f); vf[i2].push_back(f);
        }
        for (uint32_t v = 0; v < vertCount; ++v) {
            const auto &faces = vf[v];
            for (size_t a=0;a<faces.size();++a)
                for (size_t b=a+1;b<faces.size();++b) {
                    adj[faces[a]].push_back(faces[b]);
                    adj[faces[b]].push_back(faces[a]);
                }
        }
    }

    std::vector<int> compId(faceCount, -1);
    std::vector<std::vector<uint32_t>> islands;
    for (uint32_t f0 = 0; f0 < faceCount; ++f0) if (compId[f0] < 0) {
        std::vector<uint32_t> comp;
        std::queue<uint32_t> q; q.push(f0); compId[f0] = (int)islands.size();
        while(!q.empty()) {
            uint32_t f = q.front(); q.pop();
            comp.push_back(f);
            for (uint32_t g : adj[f]) if (compId[g] < 0) { compId[g] = compId[f0]; q.push(g); }
        }
        islands.push_back(std::move(comp));
    }

    // Scale each island’s UVs so texel density ∝ 3D area
    const float EPS = 1e-12f;
    std::vector<uint32_t> faceMat(faceCount, 0); // group id per face
    for (uint32_t cid = 0; cid < islands.size(); ++cid) {
        const auto &faces = islands[cid];
        // Collect unique vertices in this island
        std::vector<uint32_t> verts;
        {
            std::vector<char> mark(UV.size()/2, 0);
            verts.reserve(faces.size()*2);
            for (uint32_t f : faces) {
                uint32_t i0=I[3*f+0], i1=I[3*f+1], i2=I[3*f+2];
                if(!mark[i0]){ mark[i0]=1; verts.push_back(i0); }
                if(!mark[i1]){ mark[i1]=1; verts.push_back(i1); }
                if(!mark[i2]){ mark[i2]=1; verts.push_back(i2); }
            }
        }
        // Areas
        double A3D = 0.0, Auv = 0.0;
        for (uint32_t f : faces) {
            uint32_t i0=I[3*f+0], i1=I[3*f+1], i2=I[3*f+2];
            const float *pa=&V[3*i0], *pb=&V[3*i1], *pc=&V[3*i2];
            const float *ta=&UV[2*i0], *tb=&UV[2*i1], *tc=&UV[2*i2];
            A3D += tri_area3D(pa,pb,pc);
            Auv += tri_area2D(ta,tb,tc);
            faceMat[f] = cid; // keep faces of this island grouped
        }
        float s = 1.0f;
        if (Auv > EPS && A3D > EPS) s = (float)std::sqrt(A3D / Auv);

        // Scale island about its UV centroid (keeps relative positions)
        double cx=0, cy=0; for (uint32_t v : verts){ cx+=UV[2*v+0]; cy+=UV[2*v+1]; }
        cx /= verts.size(); cy /= verts.size();
        for (uint32_t v : verts) {
            UV[2*v+0] = (float)( (UV[2*v+0] - cx) * s + cx );
            UV[2*v+1] = (float)( (UV[2*v+1] - cy) * s + cy );
        }
    }

    xatlas::UvMeshDecl decl{};
    decl.vertexCount  = (uint32_t)(UV.size()/2);
    decl.vertexUvData = UV.data();
    decl.vertexStride = sizeof(float)*2;
    decl.indexCount   = (uint32_t)I.size();
    decl.indexData    = I.data();
    decl.indexFormat  = xatlas::IndexFormat::UInt32;
    // prevent island merging across overlaps
    decl.faceMaterialData = faceMat.data();

    xatlas::Atlas* atlas = xatlas::Create();
    auto addErr = xatlas::AddUvMesh(atlas, decl);
    if (addErr != xatlas::AddMeshError::Success) { std::fprintf(stderr, "AddUvMesh failed\n"); return 3; }

    xatlas::ChartOptions chart{};
    xatlas::ComputeCharts(atlas, chart);

    xatlas::PackOptions pack{};
    pack.resolution    = res;
    pack.padding       = pad;
    pack.rotateCharts  = rotate;
    pack.texelsPerUnit = 0.0f; // << keep the precomputed scale
    xatlas::PackCharts(atlas, pack);

    writePackedOBJ(outPath, atlas, V, pad);

    fast_obj_destroy(m);
    xatlas::Destroy(atlas);
    return 0;
}