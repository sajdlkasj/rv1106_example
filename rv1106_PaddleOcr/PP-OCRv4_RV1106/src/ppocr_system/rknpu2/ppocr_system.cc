#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <cmath>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "ppocr_system.h"
#include "common.h"
#include "file_utils.h"
#include "image_utils.h"

struct ppocr_bench_metrics_t {
    double det_pre_ms;
    double det_npu_ms;
    double det_output_ms;
    double det_post_ms;
    double det_total_ms;
    double rec_pre_ms;
    double rec_npu_ms;
    double rec_output_ms;
    double rec_post_ms;
    double rec_total_ms;
    double crop_ms;
    int rec_calls;
};

static ppocr_bench_metrics_t g_bench_metrics;

static bool ppocr_bench_enabled()
{
    const char* value = getenv("PPOCR_BENCH");
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static double ppocr_now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

bool CompareBox(const std::array<int, 8>& result1, const std::array<int, 8>& result2)
{
    if (result1[1] < result2[1]) 
    {
        return true;
    } else if (result1[1] == result2[1]) 
    {
        return result1[0] < result2[0];
    } else 
    {
        return false;
    }
}

void SortBoxes(std::vector<std::array<int, 8>>* boxes)
{
    std::sort(boxes->begin(), boxes->end(), CompareBox);

    if (boxes->size() == 0)
    {
        return;
    }
    
    for (int i = 0; i < boxes->size() - 1; i++) {
        for (int j = i; j >=0 ; j--){
            if (std::abs((*boxes)[j + 1][1] - (*boxes)[j][1]) < 10 && ((*boxes)[j + 1][0] < (*boxes)[j][0])) 
            {
                std::swap((*boxes)[i], (*boxes)[i + 1]);
            }
        }
    }

}

cv::Mat GetRotateCropImage(const cv::Mat& srcimage, const std::array<int, 8>& box)
{
    cv::Mat image;
    srcimage.copyTo(image);

    std::vector<std::vector<int>> points;

    for (int i = 0; i < 4; ++i) {
        std::vector<int> tmp;
        tmp.push_back(box[2 * i]);
        tmp.push_back(box[2 * i + 1]);
        points.push_back(tmp);
    }
    int x_collect[4] = {box[0], box[2], box[4], box[6]};
    int y_collect[4] = {box[1], box[3], box[5], box[7]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (int i = 0; i < points.size(); i++) {
        points[i][0] -= left;
        points[i][1] -= top;
    }

    int img_crop_width = int(sqrt(pow(points[0][0] - points[1][0], 2) +
                                    pow(points[0][1] - points[1][1], 2)));
    int img_crop_height = int(sqrt(pow(points[0][0] - points[3][0], 2) +
                                    pow(points[0][1] - points[3][1], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
    pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
    pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
    pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M,
                        cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE);

    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return dst_img;
    }
}

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
            "zp=%d, scale=%f\n",
            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}


static int copy_to_input(rknn_app_context_t* app_ctx, const uint8_t* src, size_t src_stride)
{
    if (app_ctx->input_mem == NULL || app_ctx->input_mem->virt_addr == NULL || src == NULL) {
        printf("invalid zero-copy input memory\n");
        return -1;
    }
    const uint32_t width_stride = app_ctx->input_attrs[0].w_stride ? app_ctx->input_attrs[0].w_stride : app_ctx->model_width;
    const uint32_t height_stride = app_ctx->input_attrs[0].h_stride ? app_ctx->input_attrs[0].h_stride : app_ctx->model_height;
    const size_t channels = app_ctx->model_channel;
    const size_t dst_stride = (size_t)width_stride * channels;
    const size_t row_bytes = (size_t)app_ctx->model_width * channels;
    memset(app_ctx->input_mem->virt_addr, 0, app_ctx->input_mem->size);
    for (int y = 0; y < app_ctx->model_height && y < (int)height_stride; ++y) {
        memcpy((uint8_t*)app_ctx->input_mem->virt_addr + (size_t)y * dst_stride,
               src + (size_t)y * src_stride, row_bytes);
    }
    return 0;
}


static float fp16_to_float(uint16_t value)
{
    const uint32_t sign = (uint32_t)(value & 0x8000) << 16;
    uint32_t exponent = (value >> 10) & 0x1f;
    uint32_t mantissa = value & 0x03ff;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int exp = -14;
            while ((mantissa & 0x0400) == 0) {
                mantissa <<= 1;
                --exp;
            }
            mantissa &= 0x03ff;
            bits = sign | (uint32_t)(exp + 127) << 23 | mantissa << 13;
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000 | mantissa << 13;
    } else {
        bits = sign | (exponent + 112) << 23 | mantissa << 13;
    }
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int get_output_float(rknn_app_context_t* app_ctx, std::vector<float>& result)
{
    if (app_ctx->output_attrs == NULL || app_ctx->output_mems == NULL ||
        app_ctx->output_mems[0] == NULL || app_ctx->output_mems[0]->virt_addr == NULL) {
        printf("invalid zero-copy output memory\n");
        return -1;
    }
    const rknn_tensor_attr& attr = app_ctx->output_attrs[0];
    const size_t count = attr.n_elems;
    result.resize(count);
    if (attr.type == RKNN_TENSOR_FLOAT32) {
        memcpy(result.data(), app_ctx->output_mems[0]->virt_addr, count * sizeof(float));
    } else if (attr.type == RKNN_TENSOR_FLOAT16) {
        const uint16_t* src = (const uint16_t*)app_ctx->output_mems[0]->virt_addr;
        for (size_t i = 0; i < count; ++i) result[i] = fp16_to_float(src[i]);
    } else if (attr.type == RKNN_TENSOR_INT8) {
        const int8_t* src = (const int8_t*)app_ctx->output_mems[0]->virt_addr;
        for (size_t i = 0; i < count; ++i) {
            result[i] = attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
                ? (src[i] - attr.zp) * attr.scale
                : (float)src[i] * (1.0f / (1 << attr.fl));
        }
    } else if (attr.type == RKNN_TENSOR_UINT8) {
        const uint8_t* src = (const uint8_t*)app_ctx->output_mems[0]->virt_addr;
        for (size_t i = 0; i < count; ++i) {
            result[i] = attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC
                ? ((int)src[i] - attr.zp) * attr.scale
                : (float)src[i];
        }
    } else {
        printf("unsupported output tensor type: %s\n", get_type_string(attr.type));
        return -1;
    }
    return 0;
}

int init_ppocr_model(const char* model_path, rknn_app_context_t* app_ctx)
{
    int ret;
    int model_len = 0;
    char* model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Get Model Input Output Number
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[i]));
    }

    // Get Model Output Info
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(output_attrs[i]));
    }

    // RV1106 requires the zero-copy rknn_set_io_mem input path.
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;
    app_ctx->input_mem = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
    if (app_ctx->input_mem == NULL) {
        printf("rknn_create_mem for input failed\n");
        rknn_destroy(ctx);
        return -1;
    }
    ret = rknn_set_io_mem(ctx, app_ctx->input_mem, &input_attrs[0]);
    if (ret < 0) {
        printf("input rknn_set_io_mem fail! ret=%d\n", ret);
        rknn_destroy_mem(ctx, app_ctx->input_mem);
        app_ctx->input_mem = NULL;
        rknn_destroy(ctx);
        return -1;
    }

    // RV1106 requires zero-copy memory binding for model outputs as well.
    app_ctx->output_mems = (rknn_tensor_mem**)calloc(io_num.n_output, sizeof(rknn_tensor_mem*));
    if (app_ctx->output_mems == NULL) {
        printf("allocate output memory table failed\n");
        rknn_destroy_mem(ctx, app_ctx->input_mem);
        app_ctx->input_mem = NULL;
        rknn_destroy(ctx);
        return -1;
    }
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
        const uint32_t output_size = output_attrs[i].size_with_stride ? output_attrs[i].size_with_stride : output_attrs[i].size;
        app_ctx->output_mems[i] = rknn_create_mem(ctx, output_size);
        if (app_ctx->output_mems[i] == NULL) {
            printf("rknn_create_mem for output %u failed\n", i);
            for (uint32_t j = 0; j < i; ++j) rknn_destroy_mem(ctx, app_ctx->output_mems[j]);
            free(app_ctx->output_mems);
            app_ctx->output_mems = NULL;
            rknn_destroy_mem(ctx, app_ctx->input_mem);
            app_ctx->input_mem = NULL;
            rknn_destroy(ctx);
            return -1;
        }
        ret = rknn_set_io_mem(ctx, app_ctx->output_mems[i], &output_attrs[i]);
        if (ret < 0) {
            printf("output rknn_set_io_mem fail! ret=%d\n", ret);
            for (uint32_t j = 0; j <= i; ++j) rknn_destroy_mem(ctx, app_ctx->output_mems[j]);
            free(app_ctx->output_mems);
            app_ctx->output_mems = NULL;
            rknn_destroy_mem(ctx, app_ctx->input_mem);
            app_ctx->input_mem = NULL;
            rknn_destroy(ctx);
            return -1;
        }
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height  = input_attrs[0].dims[2];
        app_ctx->model_width   = input_attrs[0].dims[3];
    } else {
        printf("model is NHWC input fmt\n");
        app_ctx->model_height  = input_attrs[0].dims[1];
        app_ctx->model_width   = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    printf("model input height=%d, width=%d, channel=%d\n",
        app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}

int release_ppocr_model(rknn_app_context_t* app_ctx)
{
    if (app_ctx->output_mems != NULL && app_ctx->rknn_ctx != 0) {
        for (uint32_t i = 0; i < app_ctx->io_num.n_output; ++i) {
            if (app_ctx->output_mems[i] != NULL) rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->output_mems[i]);
        }
        free(app_ctx->output_mems);
        app_ctx->output_mems = NULL;
    }
    if (app_ctx->input_mem != NULL && app_ctx->rknn_ctx != 0) {
        rknn_destroy_mem(app_ctx->rknn_ctx, app_ctx->input_mem);
        app_ctx->input_mem = NULL;
    }
    if (app_ctx->input_attrs != NULL) {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL) {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0) {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}

int inference_ppocr_det_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, ppocr_det_postprocess_params* params, ppocr_det_result* out_result)
{
    int ret;
    image_buffer_t img;
    const double total_start_ms = ppocr_now_ms();

    memset(&img, 0, sizeof(image_buffer_t));

    // Pre Process
    img.width = app_ctx->model_width;
    img.height = app_ctx->model_height;
    img.format = IMAGE_FORMAT_RGB888;
    img.size = get_image_size(&img);
    img.virt_addr = (unsigned char*)malloc(img.size);
    if (img.virt_addr == NULL) {
        printf("malloc buffer size:%d fail!\n", img.size);
        return -1;
    }

    ret = convert_image(src_img, &img, NULL, NULL, 0);
    if (ret < 0) {
        printf("convert_image fail! ret=%d\n", ret);
        return -1;
    }

    ret = copy_to_input(app_ctx, img.virt_addr, (size_t)img.width * app_ctx->model_channel);
    if (ret < 0) return -1;

    const double pre_end_ms = ppocr_now_ms();

    float scale_w = (float)src_img->width / (float)img.width;
    float scale_h = (float)src_img->height / (float)img.height;

    // Run
    // printf("rknn_run\n");
    const double npu_start_ms = ppocr_now_ms();
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    const double npu_end_ms = ppocr_now_ms();
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Read the zero-copy output and dequantize to float for DB post-processing.
    std::vector<float> output_data;
    const double output_start_ms = ppocr_now_ms();
    ret = get_output_float(app_ctx, output_data);
    if (ret < 0) goto out;
    {
        const double output_end_ms = ppocr_now_ms();

        // Post Process
        ret = dbnet_postprocess(output_data.data(), app_ctx->model_width, app_ctx->model_height,
                                                params->threshold, params->box_threshold, params->use_dilate, params->db_score_mode, 
                                                params->db_unclip_ratio, params->db_box_type,
                                                scale_w, scale_h, out_result);
        const double post_end_ms = ppocr_now_ms();

        g_bench_metrics.det_pre_ms = pre_end_ms - total_start_ms;
        g_bench_metrics.det_npu_ms = npu_end_ms - npu_start_ms;
        g_bench_metrics.det_output_ms = output_end_ms - output_start_ms;
        g_bench_metrics.det_post_ms = post_end_ms - output_end_ms;
        g_bench_metrics.det_total_ms = post_end_ms - total_start_ms;
        if (ppocr_bench_enabled()) {
            printf("BENCH_DET pre_ms=%.3f npu_ms=%.3f output_ms=%.3f post_ms=%.3f total_ms=%.3f boxes=%d\n",
                   g_bench_metrics.det_pre_ms, g_bench_metrics.det_npu_ms,
                   g_bench_metrics.det_output_ms, g_bench_metrics.det_post_ms,
                   g_bench_metrics.det_total_ms, out_result->count);
        }
    }
    
out:
    if (img.virt_addr != NULL) {
        free(img.virt_addr);
    }

    return ret;
}

int inference_ppocr_rec_model(rknn_app_context_t* app_ctx, image_buffer_t* src_img, ppocr_rec_result* out_result)
{
    int ret;
    int allow_slight_change = 1;
    const double total_start_ms = ppocr_now_ms();


    // Pre Process
    float ratio = src_img->width / float(src_img->height);
    int resized_w;
    int imgW = app_ctx->model_width, imgH = app_ctx->model_height;
    if (std::ceil(imgH*ratio) > imgW) {
        resized_w = imgW;
    }
    else {
        resized_w = std::ceil(imgH*ratio);
    }

    // The RKNN model is quantized with mean/std=[127.5]/[127.5].
    // Feed uint8 pixels and let the compiled model apply the input normalization.
    cv::Mat img_M(src_img->height, src_img->width, CV_8UC3,
                  (uint8_t*)src_img->virt_addr);
    cv::resize(img_M, img_M, cv::Size(resized_w, imgH));
    if (resized_w < imgW) {
        cv::copyMakeBorder(img_M, img_M, 0, 0, 0, imgW - resized_w,
                           cv::BORDER_CONSTANT, cv::Scalar(128, 128, 128));
    }

    ret = copy_to_input(app_ctx, img_M.data, img_M.step[0]);
    if (ret < 0) return -1;

    const double pre_end_ms = ppocr_now_ms();

    // Run
    // printf("rknn_run\n");
    const double npu_start_ms = ppocr_now_ms();
    ret = rknn_run(app_ctx->rknn_ctx, nullptr);
    const double npu_end_ms = ppocr_now_ms();
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    // Read the zero-copy output and convert FP16 to float for CTC decoding.
    int out_len_seq = app_ctx->model_width / 8;
    std::vector<float> output_data;
    bool needs_normalization = false;
    bool has_negative_values = false;
    const double output_start_ms = ppocr_now_ms();
    ret = get_output_float(app_ctx, output_data);
    if (ret < 0) goto out;
    {
    const double output_end_ms = ppocr_now_ms();

    // Normalize the CTC rows if the RV1106 output buffer is not a probability distribution.
    for (int t = 0; t < out_len_seq; ++t) {
        const float* row = output_data.data() + (size_t)t * MODEL_OUT_CHANNEL;
        float row_sum = 0.f;
        float row_max = row[0];
        float row_min = row[0];
        for (int c = 0; c < MODEL_OUT_CHANNEL; ++c) {
            row_sum += row[c];
            row_max = std::max(row_max, row[c]);
            row_min = std::min(row_min, row[c]);
        }
        if (row_min < -1e-4f) has_negative_values = true;
        if (row_max > 1.0001f || row_sum < 0.99f || row_sum > 1.01f) {
            needs_normalization = true;
        }
    }
    if (needs_normalization) {
        static bool normalization_message_printed = false;
        if (!normalization_message_printed) {
            printf("normalizing RV1106 recognizer output rows\n");
            normalization_message_printed = true;
        }
        for (int t = 0; t < out_len_seq; ++t) {
            float* row = output_data.data() + (size_t)t * MODEL_OUT_CHANNEL;
            if (has_negative_values) {
                const float row_max = *std::max_element(row, row + MODEL_OUT_CHANNEL);
                double sum = 0.0;
                for (int c = 0; c < MODEL_OUT_CHANNEL; ++c) {
                    row[c] = std::exp(row[c] - row_max);
                    sum += row[c];
                }
                if (sum > 0.0) {
                    for (int c = 0; c < MODEL_OUT_CHANNEL; ++c) row[c] = (float)(row[c] / sum);
                }
            } else {
                double sum = 0.0;
                for (int c = 0; c < MODEL_OUT_CHANNEL; ++c) sum += row[c];
                if (sum > 0.0) {
                    for (int c = 0; c < MODEL_OUT_CHANNEL; ++c) row[c] = (float)(row[c] / sum);
                }
            }
        }
    }

    // Post Process
    ret = rec_postprocess(output_data.data(), MODEL_OUT_CHANNEL, out_len_seq, out_result);

    const double post_end_ms = ppocr_now_ms();
    const double pre_ms = pre_end_ms - total_start_ms;
    const double npu_ms = npu_end_ms - npu_start_ms;
    const double output_ms = output_end_ms - output_start_ms;
    const double post_ms = post_end_ms - output_end_ms;
    const double total_ms = post_end_ms - total_start_ms;
    g_bench_metrics.rec_pre_ms += pre_ms;
    g_bench_metrics.rec_npu_ms += npu_ms;
    g_bench_metrics.rec_output_ms += output_ms;
    g_bench_metrics.rec_post_ms += post_ms;
    g_bench_metrics.rec_total_ms += total_ms;
    g_bench_metrics.rec_calls += 1;
    if (ppocr_bench_enabled()) {
        printf("BENCH_REC call=%d pre_ms=%.3f npu_ms=%.3f output_ms=%.3f post_ms=%.3f total_ms=%.3f score=%.6f\n",
               g_bench_metrics.rec_calls, pre_ms, npu_ms, output_ms, post_ms,
               total_ms, out_result->score);
    }
    }
    
out:

    return ret;
}

int inference_ppocr_system_model(ppocr_system_app_context* sys_app_ctx, image_buffer_t* src_img, ppocr_det_postprocess_params* params, ppocr_text_recog_array_result_t* out_result)
{
    int ret;
    memset(&g_bench_metrics, 0, sizeof(g_bench_metrics));
    const double system_start_ms = ppocr_now_ms();
    // Detect Text
    ppocr_det_result det_results;
    ret = inference_ppocr_det_model(&sys_app_ctx->det_context, src_img, params, &det_results);
    if (ret != 0) {
        printf("inference_ppocr_det_model fail! ret=%d\n", ret);
        return -1;
    }

    // Recogize Text
    out_result->count = 0;
    if (det_results.count == 0) {           // detect nothing
        if (ppocr_bench_enabled()) {
            const double system_total_ms = ppocr_now_ms() - system_start_ms;
            printf("BENCH_SYSTEM det_boxes=0 accepted=0 rec_calls=0 det_ms=%.3f crop_ms=0.000 rec_ms=0.000 other_ms=%.3f total_ms=%.3f\n",
                   g_bench_metrics.det_total_ms,
                   system_total_ms - g_bench_metrics.det_total_ms,
                   system_total_ms);
        }
        return 0;
    }

    // boxes to boxes_result
    std::vector<std::array<int, 8>> boxes_result;
    for (int i=0; i < det_results.count; i++) {
        std::array<int, 8> new_box;
        new_box[0] = det_results.box[i].left_top.x;
        new_box[1] = det_results.box[i].left_top.y;
        new_box[2] = det_results.box[i].right_top.x;
        new_box[3] = det_results.box[i].right_top.y;
        new_box[4] = det_results.box[i].right_bottom.x;
        new_box[5] = det_results.box[i].right_bottom.y;
        new_box[6] = det_results.box[i].left_bottom.x;
        new_box[7] = det_results.box[i].left_bottom.y;
        boxes_result.emplace_back(new_box);
    }

    // Sort text boxes in order from top to bottom, left to right for speeding up
    SortBoxes(&boxes_result);

    // text recognize
    for (int i=0; i < boxes_result.size(); i++) {
        cv::Mat in_image = cv::Mat(src_img->height, src_img->width, CV_8UC3,(uint8_t*)src_img->virt_addr);
        const double crop_start_ms = ppocr_now_ms();
        cv::Mat crop_image = GetRotateCropImage(in_image, boxes_result[i]);
        image_buffer_t text_img;
        memset(&text_img, 0, sizeof(image_buffer_t));
        text_img.width = crop_image.cols;
        text_img.height = crop_image.rows;
        text_img.format = IMAGE_FORMAT_RGB888;
        text_img.size = get_image_size(&text_img);
        text_img.virt_addr = (unsigned char*)malloc(text_img.size);
        if (text_img.virt_addr == NULL) {
            printf("malloc buffer size:%d fail!\n", text_img.size);
            return -1;
        }
        memcpy((void *)text_img.virt_addr, crop_image.data, text_img.size);
        g_bench_metrics.crop_ms += ppocr_now_ms() - crop_start_ms;
        
        ppocr_rec_result text_result;
        text_result.score = 1.0;
        ret = inference_ppocr_rec_model(&sys_app_ctx->rec_context, &text_img, &text_result);
        if (ret != 0) {
            printf("inference_ppocr_rec_model fail! ret=%d\n", ret);
            return -1;
        }
        if (text_img.virt_addr != NULL) {
            free(text_img.virt_addr);
        }

        if (text_result.score < TEXT_SCORE) {
            continue;
        }
        out_result->text_result[out_result->count].box.left_top.x = boxes_result[i][0];
        out_result->text_result[out_result->count].box.left_top.y = boxes_result[i][1];
        out_result->text_result[out_result->count].box.right_top.x = boxes_result[i][2];
        out_result->text_result[out_result->count].box.right_top.y = boxes_result[i][3];
        out_result->text_result[out_result->count].box.right_bottom.x = boxes_result[i][4];
        out_result->text_result[out_result->count].box.right_bottom.y = boxes_result[i][5];
        out_result->text_result[out_result->count].box.left_bottom.x = boxes_result[i][6];
        out_result->text_result[out_result->count].box.left_bottom.y = boxes_result[i][7];
        out_result->text_result[out_result->count].text = text_result;
        out_result->count ++;
    }

    if (ppocr_bench_enabled()) {
        const double system_total_ms = ppocr_now_ms() - system_start_ms;
        const double known_ms = g_bench_metrics.det_total_ms + g_bench_metrics.crop_ms + g_bench_metrics.rec_total_ms;
        printf("BENCH_SYSTEM det_boxes=%d accepted=%d rec_calls=%d det_ms=%.3f crop_ms=%.3f rec_ms=%.3f other_ms=%.3f total_ms=%.3f\n",
               det_results.count, out_result->count, g_bench_metrics.rec_calls,
               g_bench_metrics.det_total_ms, g_bench_metrics.crop_ms,
               g_bench_metrics.rec_total_ms, system_total_ms - known_ms,
               system_total_ms);
    }

    return ret;
}
