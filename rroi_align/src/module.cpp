#include <torch/extension.h>

#include "roi_pooling.h"
#include "rroi_align_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rroi_align_forward", &rroi_align_forward_cuda, "RROI Align Forward C++",
      py::arg("pooled_height"), py::arg("pooled_width"), py::arg("spatial_scale"),
      py::arg("features"), py::arg("rois"), py::arg("output"),
      py::arg("idx_x"), py::arg("idx_y")
    );
    m.def("rroi_align_backward", &rroi_align_backward_cuda, "RROI Align Backward C++",
      py::arg("pooled_height"), py::arg("pooled_width"), py::arg("spatial_scale"),
      py::arg("top_grad"), py::arg("rois"), py::arg("bottom_grad"),
      py::arg("idx_x"), py::arg("idx_y")
    );
    m.def("roi_pooling_forward", &roi_pooling_forward, "ROI Pooling Forward C++",
      py::arg("pooled_height"), py::arg("pooled_width"), py::arg("spatial_scale"),
      py::arg("features"), py::arg("rois"), py::arg("output")
    );
}
