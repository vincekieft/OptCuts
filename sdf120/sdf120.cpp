// sdf120.cpp — CGAL SDF (returns per-face values in ORIGINAL face order)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <limits>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>

// ---- Pick the correct SDF header depending on CGAL version you have ----
#if defined(__has_include) && __has_include(<CGAL/Polygon_mesh_processing/segmentation.h>)
  #include <CGAL/Polygon_mesh_processing/segmentation.h>
  namespace CGAL_SDF = CGAL::Polygon_mesh_processing;
#elif defined(__has_include) && __has_include(<CGAL/mesh_segmentation.h>)
  #include <CGAL/mesh_segmentation.h>              // your distro has this one
  namespace CGAL_SDF = CGAL;                       // sdf_values is in CGAL::
#else
  #error "CGAL segmentation header not found. Install/update libcgal-dev."
#endif

namespace py = pybind11;

using Kernel = CGAL::Simple_cartesian<double>;
using Point  = Kernel::Point_3;
using Mesh   = CGAL::Surface_mesh<Point>;
using face_descriptor = boost::graph_traits<Mesh>::face_descriptor;

py::array_t<double> compute_sdf(
    py::array_t<double, py::array::c_style | py::array::forcecast> V,  // (n,3)
    py::array_t<int,    py::array::c_style | py::array::forcecast> F,  // (m,3)
    int rays = 24,
    double cone_deg = 120.0,
    bool postprocess = true
){
    if (V.ndim() != 2 || V.shape(1) != 3) throw std::runtime_error("V must be (n,3)");
    if (F.ndim() != 2 || F.shape(1) != 3) throw std::runtime_error("F must be (m,3)");

    const size_t n = static_cast<size_t>(V.shape(0));
    const size_t m = static_cast<size_t>(F.shape(0));

    Mesh mesh;
    mesh.reserve(n, 0, m);

    // add vertices
    auto Vbuf = V.unchecked<2>();
    std::vector<Mesh::Vertex_index> vidx(n);
    for (size_t i = 0; i < n; ++i) {
        vidx[i] = mesh.add_vertex(Point(Vbuf(i,0), Vbuf(i,1), Vbuf(i,2)));
    }

    // property to remember original face index
    auto orig_map = mesh.add_property_map<face_descriptor, std::size_t>("f:orig", std::size_t(-1)).first;

    // add faces, remembering original index
    auto Fbuf = F.unchecked<2>();
    for (size_t i = 0; i < m; ++i) {
        const int ia = Fbuf(i,0), ib = Fbuf(i,1), ic = Fbuf(i,2);
        if (ia < 0 || ib < 0 || ic < 0) continue;
        if ((size_t)ia >= n || (size_t)ib >= n || (size_t)ic >= n) continue;
        if (ia == ib || ib == ic || ic == ia) continue;

        Mesh::Face_index f = mesh.add_face(Mesh::Vertex_index(ia),
                                           Mesh::Vertex_index(ib),
                                           Mesh::Vertex_index(ic));
        if (f != Mesh::null_face()) {
            put(orig_map, f, i);
        }
    }

    if (num_faces(mesh) == 0)
        throw std::runtime_error("No valid faces in mesh (after cleaning).");

    // compute SDF
    auto sdf_map = mesh.add_property_map<face_descriptor, double>("f:sdf", 0.0).first;
    const double cone_rad = cone_deg * M_PI / 180.0;
    CGAL_SDF::sdf_values(mesh, sdf_map, cone_rad, rays, postprocess);

    // output in ORIGINAL face order; NaN for faces that CGAL skipped
    py::array_t<double> out({ (py::ssize_t)m });
    auto o = out.mutable_unchecked<1>();
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    for (py::ssize_t i = 0; i < (py::ssize_t)m; ++i) o(i) = NaN;

    for (face_descriptor f : faces(mesh)) {
        std::size_t idx = get(orig_map, f);
        if (idx != std::size_t(-1) && idx < (std::size_t)m) {
            o((py::ssize_t)idx) = get(sdf_map, f);
        }
    }
    return out;
}

PYBIND11_MODULE(sdf120, m) {
    m.doc() = "Fast SDF with configurable cone angle (CGAL-backed), aligned to original face order";
    m.def("compute_sdf", &compute_sdf,
          py::arg("V"), py::arg("F"),
          py::arg("rays") = 24,
          py::arg("cone_deg") = 120.0,
          py::arg("postprocess") = true);
}
