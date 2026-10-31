// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>

#include "ppocr_system.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "file_utils.h"

#define INDENT "    "
#define THRESHOLD 0.3                                       // pixel score threshold
#define BOX_THRESHOLD 0.6                            // box score threshold
#define USE_DILATION false                               // whether to do dilation, true or false
#define DB_SCORE_MODE "slow"                        // slow or fast. slow for polygon mask; fast for rectangle mask
#define DB_BOX_TYPE "poly"                                // poly or quad. poly for returning polygon box; quad for returning rectangle box
#define DB_UNCLIP_RATIO 1.5                          // unclip ratio for poly type

static inline double bench_now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

static inline bool bench_enabled()
{
    const char* value = getenv("PPOCR_BENCH");
    return value != NULL && strcmp(value, "0") != 0;
}

/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    const double program_start_ms = bench_now_ms();
    if (argc != 4) {
        printf("%s <det_model_path> <rec_model_path> <image_path>\n", argv[0]);
        return -1;
    }

    const char* det_model_path = argv[1];
    const char* rec_model_path = argv[2];
    const char* image_path = argv[3];

    int ret;
    double draw_start_ms = 0.0;
    double draw_write_ms = 0.0;
    ppocr_system_app_context rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(ppocr_system_app_context));

    const double det_init_start_ms = bench_now_ms();
    ret = init_ppocr_model(det_model_path, &rknn_app_ctx.det_context);
    const double det_init_ms = bench_now_ms() - det_init_start_ms;
    if (ret != 0) {
        printf("init_ppocr_model fail! ret=%d det_model_path=%s\n", ret, det_model_path);
        return -1;
    }

    const double rec_init_start_ms = bench_now_ms();
    ret = init_ppocr_model(rec_model_path, &rknn_app_ctx.rec_context);
    const double rec_init_ms = bench_now_ms() - rec_init_start_ms;
    if (ret != 0) {
        printf("init_ppocr_model fail! ret=%d rec_model_path=%s\n", ret, rec_model_path);
        return -1;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    const double read_start_ms = bench_now_ms();
    ret = read_image(image_path, &src_image);
    const double read_ms = bench_now_ms() - read_start_ms;
    if (ret != 0) {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        return -1;
    }

    ppocr_text_recog_array_result_t results;
    ppocr_det_postprocess_params params;
    params.threshold = THRESHOLD;
    params.box_threshold = BOX_THRESHOLD;
    params.use_dilate = USE_DILATION;
    params.db_score_mode = DB_SCORE_MODE;
    params.db_box_type = DB_BOX_TYPE;
    params.db_unclip_ratio = DB_UNCLIP_RATIO;
    const unsigned char blue[] = {0, 0, 255};

    const double inference_start_ms = bench_now_ms();
    ret = inference_ppocr_system_model(&rknn_app_ctx, &src_image, &params, &results);
    const double inference_ms = bench_now_ms() - inference_start_ms;
    if (ret != 0) {
        printf("inference_ppocr_system_model fail! ret=%d\n", ret);
        goto out;
    }

    draw_start_ms = bench_now_ms();
    // Draw Objects
    printf("DRAWING OBJECT\n");
    for (int i = 0; i < results.count; i++)
    {
        printf("[%d] @ [(%d, %d), (%d, %d), (%d, %d), (%d, %d)]\n", i,
            results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, 
            results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y);
        //draw Quadrangle box
        draw_line(&src_image, results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, 255, 2);
        draw_line(&src_image, results.text_result[i].box.right_top.x, results.text_result[i].box.right_top.y, results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, 255, 2);
        draw_line(&src_image, results.text_result[i].box.right_bottom.x, results.text_result[i].box.right_bottom.y, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y, 255, 2);
        draw_line(&src_image, results.text_result[i].box.left_bottom.x, results.text_result[i].box.left_bottom.y, results.text_result[i].box.left_top.x, results.text_result[i].box.left_top.y, 255, 2);
        printf("regconize result: %s, score=%f\n", results.text_result[i].text.str, results.text_result[i].text.score);
    }
    printf("    SAVE TO ./out.jpg\n");
    write_image("./out.jpg", &src_image);
    draw_write_ms = bench_now_ms() - draw_start_ms;

    if (bench_enabled()) {
        printf("BENCH_MAIN det_init_ms=%.3f rec_init_ms=%.3f read_ms=%.3f "
               "inference_ms=%.3f draw_write_ms=%.3f before_release_ms=%.3f\n",
               det_init_ms, rec_init_ms, read_ms, inference_ms, draw_write_ms,
               bench_now_ms() - program_start_ms);
    }

out:
    ret = release_ppocr_model(&rknn_app_ctx.det_context);
    if (ret != 0) {
        printf("release_ppocr_model det_context fail! ret=%d\n", ret);
    }
    ret = release_ppocr_model(&rknn_app_ctx.rec_context);
    if (ret != 0) {
        printf("release_ppocr_model rec_context fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL) {
        free(src_image.virt_addr);
    }

    if (bench_enabled()) {
        printf("BENCH_PROGRAM total_ms=%.3f\n", bench_now_ms() - program_start_ms);
    }

    return 0;
}
