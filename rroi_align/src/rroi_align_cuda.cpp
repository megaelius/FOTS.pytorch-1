#include "rroi_align_cuda.h"

#include <math.h>
#include <c10/cuda/CUDAStream.h>

#include "rroi_align_kernel.h"

int rroi_align_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
            torch::Tensor features, torch::Tensor rois, torch::Tensor output,
            torch::Tensor idx_x, torch::Tensor idx_y)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Grab the input tensor
    float * data_flat = features.data_ptr<float>();
    float * rois_flat = rois.data_ptr<float>();

    float * output_flat = output.data_ptr<float>();
    float * idx_x_flat = idx_x.data_ptr<float>();           // 每个rroi bin的中心索引
    float * idx_y_flat = idx_y.data_ptr<float>();
    // int * argmax_flat = THCudaIntTensor_data(state, argmax);

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 6)
    {
    return 0;
    }

    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);

    RROIAlignForwardLaucher(
    data_flat, spatial_scale, num_rois, data_height,
    data_width, num_channels, pooled_height,
    pooled_width, rois_flat,
    output_flat, idx_x_flat, idx_y_flat, stream);

    return 1;
}



// 反向传播
int rroi_align_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
            torch::Tensor top_grad, torch::Tensor rois, torch::Tensor bottom_grad,
            torch::Tensor idx_x, torch::Tensor idx_y)
{
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Grab the input tensor
    float * top_grad_flat = top_grad.data_ptr<float>();
    float * rois_flat = rois.data_ptr<float>();

    float * bottom_grad_flat = bottom_grad.data_ptr<float>();
    float * idx_x_flat = idx_x.data_ptr<float>();
    float * idx_y_flat = idx_y.data_ptr<float>();

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 6)
    {
    return 0;
    }

    // batch size
    int batch_size = bottom_grad.size(0);

    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    RROIAlignBackwardLaucher(
    top_grad_flat, spatial_scale, batch_size, num_rois, data_height,
    data_width, num_channels, pooled_height,
    pooled_width, rois_flat, bottom_grad_flat,
    idx_x_flat, idx_y_flat, stream);

    return 1;
}
