#include "rroi_align_cpu.h"
#include <iostream>
#include <math.h>

using namespace std;

int RROIAlignForwardLaucherCpu(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const float* bottom_rois,
    float* top_data, float* con_idx_x, float* con_idx_y)
    {
    const int output_size = num_rois * pooled_height * pooled_width * channels;
    for(int ind=0; ind<output_size; ++ind){
        // +0.5 shift removed
        int imageWidth = width;
        int imageHeight = height;

        // (n, c, ph, pw) is an element in the pooled output
        int n = ind;
        int pw = n % pooled_width;
        n /= pooled_width;
        int ph = n % pooled_height;
        n /= pooled_height;
        int c = n % channels;
        n /= channels;
        //cout << n << endl;

        const float* offset_bottom_rois = bottom_rois + n * 6; // 标注信息

        int roi_batch_ind = offset_bottom_rois[0];
        float cx = offset_bottom_rois[1];
        float cy = offset_bottom_rois[2];
        float h = offset_bottom_rois[3];
        float w = offset_bottom_rois[4];
        float angle = offset_bottom_rois[5]/180.0*3.1415926535;

        //TransformPrepare
        float roi_pooled_width = pooled_height * w / h;     // 不同的高宽比
        float dx = -roi_pooled_width/2.0;
        float dy = -pooled_height/2.0;
        float Sx = w*spatial_scale/roi_pooled_width;
        float Sy = h*spatial_scale/pooled_height;
        float Alpha = cos(angle);
        float Beta = sin(angle);
        float Dx = cx*spatial_scale;
        float Dy = cy*spatial_scale;

        float M[2][3];                  // 旋转矩阵
        M[0][0] = Alpha*Sx;
        M[0][1] = Beta*Sy;
        M[0][2] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
        M[1][0] = -Beta*Sx;
        M[1][1] = Alpha*Sy;
        M[1][2] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;

        float P[8];                 // 求原roi中4个点的坐标8个值
        P[0] = M[0][0]*pw+M[0][1]*ph+M[0][2];
        P[1] = M[1][0]*pw+M[1][1]*ph+M[1][2];
        P[2] = M[0][0]*pw+M[0][1]*(ph+1)+M[0][2];
        P[3] = M[1][0]*pw+M[1][1]*(ph+1)+M[1][2];
        P[4] = M[0][0]*(pw+1)+M[0][1]*ph+M[0][2];
        P[5] = M[1][0]*(pw+1)+M[1][1]*ph+M[1][2];
        P[6] = M[0][0]*(pw+1)+M[0][1]*(ph+1)+M[0][2];
        P[7] = M[1][0]*(pw+1)+M[1][1]*(ph+1)+M[1][2];

        float zerooo = 0;
        float widthminus1 = imageWidth-1.0;
        float heightminus1 = imageHeight-1.0;

        // 求原rroi的中心，并用双线性插值求出f(x,y)
        float leftMost = (max(round(min(min(P[0],P[2]),min(P[4],P[6]))),zerooo));
        float rightMost= (min(round(max(max(P[0],P[2]),max(P[4],P[6]))),widthminus1));
        float topMost= (max(round(min(min(P[1],P[3]),min(P[5],P[7]))),zerooo));
        float bottomMost= (min(round(max(max(P[1],P[3]),max(P[5],P[7]))),heightminus1));

        const float* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

        float bin_cx = (leftMost + rightMost) / 2.0; // rroi的中心
        float bin_cy = (topMost + bottomMost) / 2.0;

        const bool in_rroi = pw <= roi_pooled_width;    // 是否在rroi之内
        if (in_rroi){

            int bin_l = (int)floor(bin_cx);
            int bin_r = (int)ceil(bin_cx);
            int bin_t = (int)floor(bin_cy);
            int bin_b = (int)ceil(bin_cy);

            float lt_value = 0.0;
            if (bin_t > 0 && bin_l > 0 && bin_t < height && bin_l < width)
            lt_value = offset_bottom_data[bin_t * width + bin_l];
            float rt_value = 0.0;
            if (bin_t > 0 && bin_r > 0 && bin_t < height && bin_r < width)
            rt_value = offset_bottom_data[bin_t * width + bin_r];
            float lb_value = 0.0;
            if (bin_b > 0 && bin_l > 0 && bin_b < height && bin_l < width)
            lb_value = offset_bottom_data[bin_b * width + bin_l];
            float rb_value = 0.0;
            if (bin_b > 0 && bin_r > 0 && bin_b < height && bin_r < width)
            rb_value = offset_bottom_data[bin_b * width + bin_r];

            float rx = bin_cx - floor(bin_cx);
            float ry = bin_cy - floor(bin_cy);

            float wlt = (1.0 - rx) * (1.0 - ry);
            float wrt = rx * (1.0 - ry);
            float wrb = rx * ry;
            float wlb = (1.0 - rx) * ry;

            float inter_val = 0.0;

            inter_val += lt_value * wlt;
            inter_val += rt_value * wrt;
            inter_val += rb_value * wrb;
            inter_val += lb_value * wlb;

            //atomicAdd(top_data + index, static_cast<float>(inter_val));
            //atomicAdd(con_idx_x + index, static_cast<float>(bin_cx));
            //atomicAdd(con_idx_y + index, static_cast<float>(bin_cy));

            top_data[ind] += inter_val;
            con_idx_x[ind] += bin_cx;
            con_idx_y[ind] += bin_cy;
        }
        else{
            // float inter_val = 0.0;
            // float bin_cx = 0.0;            // -2只是为了反向传播时做标记，其他值也是可以的
            // float bin_cy = 0.0;
            // atomicAdd(top_data + index, static_cast<float>(inter_val));     // 可能多个点加了-2
            // atomicAdd(con_idx_x + index, static_cast<float>(bin_cx));
            // atomicAdd(con_idx_y + index, static_cast<float>(bin_cy));
            continue;
        }

    }
    return 1;
}

int rroi_align_forward_cpu(int pooled_height, int pooled_width, float spatial_scale,
            torch::Tensor features, torch::Tensor rois, torch::Tensor output,
            torch::Tensor idx_x, torch::Tensor idx_y)
{

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

    RROIAlignForwardLaucherCpu(
    data_flat, spatial_scale, num_rois, data_height,
    data_width, num_channels, pooled_height,
    pooled_width, rois_flat,
    output_flat, idx_x_flat, idx_y_flat);

    return 1;
}

int RROIAlignBackwardLaucherCpu(
    const float* top_diff,
    const float spatial_scale,
    const int batch_size,
    const int num_rois,
    const int height,
    const int width,
    const int channels,
    const int pooled_height,
    const int pooled_width,
    const float* bottom_rois,
    float* bottom_diff,
    const float* con_idx_x,
    const float* con_idx_y)
{
    const int output_size = num_rois * pooled_height * pooled_width * channels;
    for(int ind=0;ind<output_size;++ind){
        int n = ind;
        int pw = n % pooled_width;
        n /= pooled_width;
        int ph = n % pooled_height;
        n /= pooled_height;
        int c = n % channels;
        n /= channels;

        const float* offset_bottom_rois = bottom_rois + n * 6;            // 第i个rroi
        int roi_batch_ind = offset_bottom_rois[0];
        float h = offset_bottom_rois[3];
        float w = offset_bottom_rois[4];
        float* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;      // 反向梯度的索引

        float bin_cx = con_idx_x[ind];        // 每个rroi中心点的坐标
        float bin_cy = con_idx_y[ind];

        // check whether in rroi
        float roi_pooled_width = pooled_height * w / h;     // 不同的高宽比

        const bool not_in_rroi = (pw > roi_pooled_width);    // 可能多个点多次加了-2, 所以不能采用这种方式判断

        if (not_in_rroi){                   // 如果不再rroi内则跳过当前循环，否则就按原来的操作
            continue;
        }
        else{

            float rx = bin_cx - floor(bin_cx);
            float ry = bin_cy - floor(bin_cy);

            float wlt = (1.0 - rx) * (1.0 - ry);
            float wrt = rx * (1.0 - ry);
            float wrb = rx * ry;
            float wlb = (1.0 - rx) * ry;

            int min_x = (int)floor(bin_cx);
            int max_x = (int)ceil(bin_cx);
            int min_y = (int)floor(bin_cy);
            int max_y = (int)ceil(bin_cy);

            float top_diff_of_bin = top_diff[ind];

            float v1 = wlt * top_diff_of_bin;
            float v2 = wrt * top_diff_of_bin;
            float v3 = wrb * top_diff_of_bin;
            float v4 = wlb * top_diff_of_bin;

            // Atomic add
            // float* + int * int + int
            if (min_y > 0 && min_x  > 0 && min_y < height - 1 && min_x < width - 1){
                //atomicAdd(offset_bottom_diff + min_y * width + min_x, static_cast<float>(v1));
                offset_bottom_diff[min_y * width + min_x] += v1;
            }
            if (min_y > 0 && max_x < width - 1 && min_y < height - 1 && max_x > 0){
                //atomicAdd(offset_bottom_diff + min_y * width + max_x, static_cast<float>(v2));
                offset_bottom_diff[min_y * width + max_x] += v2;
            }
            if (max_y < height - 1 && max_x < width - 1 && max_y > 0 && max_x > 0){
                //atomicAdd(offset_bottom_diff + max_y * width + max_x, static_cast<float>(v3));
                offset_bottom_diff[max_y * width + max_x] += v3;
            }
            if (max_y < height - 1 && min_x > 0 && max_y > 0 && min_x < width - 1){
                //atomicAdd(offset_bottom_diff + max_y * width + min_x, static_cast<float>(v4));
                offset_bottom_diff[max_y * width + min_x] += v4;
            }

        }
    }

    return 1;
}

// 反向传播
int rroi_align_backward_cpu(int pooled_height, int pooled_width, float spatial_scale,
            torch::Tensor top_grad, torch::Tensor rois, torch::Tensor bottom_grad,
            torch::Tensor idx_x, torch::Tensor idx_y)
{

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

    RROIAlignBackwardLaucherCpu(
    top_grad_flat, spatial_scale, batch_size, num_rois, data_height,
    data_width, num_channels, pooled_height,
    pooled_width, rois_flat, bottom_grad_flat,
    idx_x_flat, idx_y_flat);

    return 1;
}
