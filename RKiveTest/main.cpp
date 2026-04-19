#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <unistd.h>   // sleep()
#include <sys/time.h>
using namespace std;
using namespace cv;



// ========== 适配所有RKIVE SDK版本：替换封装函数为核心API ==========
#define MMZ_ALLOC_TYPE    RK_MMZ_ALLOC_TYPE_CMA
#define RK_MAX_FILE_LEN   256
#define RK_FAILURE        -1
#define RK_SUCCESS        0

// 替代RK_CHECK_ET_GOTO
#define RK_CHECK_ET_GOTO(ret, et, label) \
    do { \
        if ((ret) == (et)) { \
            goto label; \
        } \
    } while (0)

// 替代RK_FCLOSE
#define RK_FCLOSE(fp) \
    do { \
        if ((fp) != NULL) { \
            fclose(fp); \
            (fp) = NULL; \
        } \
    } while (0)

// 替代RK_SPRINTF
#define RK_SPRINTF(buf, len, fmt, ...) snprintf(buf, len, fmt, ##__VA_ARGS__)

// 获取当前时间（毫秒）
static inline double get_current_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

// 替代RK_CreateIveImage（核心API实现）
RK_S32 RK_CreateIveImage(IVE_SRC_IMAGE_S *pImg, RK_U32 u32Type, RK_U32 u32W, RK_U32 u32H) {
    if (pImg == NULL || u32W == 0 || u32H == 0) return RK_FAILURE;
    
    memset(pImg, 0, sizeof(IVE_SRC_IMAGE_S));
    RK_U32 u32Size = u32W * u32H; // U8C1格式
    
    MB_BLK mbBlk = NULL;
    RK_S32 s32Ret = RK_MPI_MMZ_Alloc(&mbBlk, u32Size, MMZ_ALLOC_TYPE);
    if (s32Ret != 0) return RK_FAILURE;
    
    pImg->au64PhyAddr[0] = RK_MPI_MB_Handle2PhysAddr(mbBlk);
    pImg->au64VirAddr[0] = (RK_U64)RK_MPI_MB_Handle2VirAddr(mbBlk);
    pImg->au32Stride[0] = u32W;
    pImg->u32Width = u32W;
    pImg->u32Height = u32H;
    pImg->enType = (IVE_IMAGE_TYPE_E)u32Type;
    pImg->s32Reserved = (RK_S32)mbBlk; // 保存MB_BLK用于释放
    
    return RK_SUCCESS;
}

// 替代RK_DestroyIveImage（核心API实现）
RK_VOID RK_DestroyIveImage(IVE_SRC_IMAGE_S *pImg) {
    if (pImg == NULL) return;
    MB_BLK mbBlk = (MB_BLK)pImg->s32Reserved;
    if (mbBlk != NULL) {
        RK_MPI_MMZ_Free(mbBlk);
    }
    memset(pImg, 0, sizeof(IVE_SRC_IMAGE_S));
}

// 替代RK_ReadFile（读取图像到IVE内存）
RK_S32 RK_ReadFile(const char *fileName, IVE_SRC_IMAGE_S *pImg, FILE **fp) {
    if (fileName == NULL || pImg == NULL || fp == NULL) return RK_FAILURE;
    
    // 用OpenCV读取图像（兼容常见格式）
    Mat mat = imread(fileName, IMREAD_GRAYSCALE);
    if (mat.empty()) return RK_FAILURE;
    if (mat.cols != pImg->u32Width || mat.rows != pImg->u32Height) return RK_FAILURE;
    
    // 拷贝数据到IVE内存
    memcpy((void*)pImg->au64VirAddr[0], mat.data, pImg->u32Width * pImg->u32Height);
    *fp = fopen(fileName, "rb"); // 模拟官方fp返回
    return RK_SUCCESS;
}

// 替代RK_WriteFile（保存IVE图像到文件）
RK_VOID RK_WriteFile(const char *fileName, void *pData, RK_U32 u32Size, FILE **fo) {
    if (pData == NULL || u32Size == 0) return;
    
    if (fileName == NULL) {
        *fo = NULL;
        return;
    }
    
    *fo = fopen(fileName, "wb");
    if (*fo != NULL) {
        fwrite(pData, 1, u32Size, *fo);
    }
}

// ========== 官方DilateSample逻辑（完整复用） ==========
RK_VOID DilateSample(RK_U8 *pu8Mask, const RK_CHAR *moduleName, RK_CHAR *fileName,
                     RK_CHAR *outDir, RK_U32 u32Width, RK_U32 u32Height, double &rkive_time) {
    IVE_SRC_IMAGE_S stSrc;
    IVE_DST_IMAGE_S stDst;
    IVE_DILATE_CTRL_S stCtrlDilate;
    IVE_THRESH_CTRL_S stCtrlThresh;

    RK_S32 s32Result;
    FILE *fp = NULL, *fo = NULL;
    IVE_HANDLE handle;
    RK_DOUBLE dTime;
    RK_CHAR outFile[RK_MAX_FILE_LEN];

    memset(&stSrc, 0, sizeof(IVE_SRC_IMAGE_S));
    memset(&stDst, 0, sizeof(IVE_DST_IMAGE_S));
    memset(&stCtrlDilate, 0, sizeof(IVE_DILATE_CTRL_S));
    memset(&stCtrlThresh, 0, sizeof(IVE_THRESH_CTRL_S));

    s32Result = RK_CreateIveImage(&stSrc, IVE_IMAGE_TYPE_U8C1, u32Width, u32Height);
    RK_CHECK_ET_GOTO(s32Result, RK_FAILURE, FAILURE);
    
    // 转换Dst为IVE_SRC_IMAGE_S（兼容Create/Destroy）
    s32Result = RK_CreateIveImage((IVE_SRC_IMAGE_S*)&stDst, IVE_IMAGE_TYPE_U8C1, u32Width, u32Height);
    RK_CHECK_ET_GOTO(s32Result, RK_FAILURE, FAILURE);

    s32Result = RK_ReadFile(fileName, &stSrc, &fp);
    RK_CHECK_ET_GOTO(s32Result, RK_FAILURE, FAILURE);

    memcpy(stCtrlDilate.au8Mask, pu8Mask, sizeof(RK_U8) * 25);

    // RKIVE膨胀计时
    dTime = get_current_time_ms();
    s32Result = RK_MPI_IVE_Dilate(&handle, &stSrc, &stDst, &stCtrlDilate, RK_TRUE);
    RK_CHECK_ET_GOTO(s32Result, RK_FAILURE, FAILURE);
    rkive_time = get_current_time_ms() - dTime;

    printf("RKIVE Dilate time = %.2f ms\n", rkive_time);

    if (outDir == NULL) {
        RK_WriteFile(NULL, (void *)stDst.au64VirAddr[0],
                     stDst.u32Width * stDst.u32Height, &fo);
    } else {
        RK_SPRINTF(outFile, RK_MAX_FILE_LEN, "%s/rve_%s_out.yuv", outDir, moduleName);
        RK_WriteFile(outFile, (void *)stDst.au64VirAddr[0],
                     stDst.u32Width * stDst.u32Height, &fo);
    }

FAILURE:
    RK_FCLOSE(fp);
    RK_FCLOSE(fo);
    RK_DestroyIveImage(&stSrc);
    RK_DestroyIveImage((IVE_SRC_IMAGE_S*)&stDst);
}

// ========== 3x3/5x5膨胀模板（复用官方） ==========
RK_VOID DilateSample3x3(const RK_CHAR *moduleName, RK_CHAR *fileName, RK_CHAR *outDir,
                        RK_U32 u32Width, RK_U32 u32Height, double &rkive_time) {
    RK_U8 mask[25] = {0,   0, 0, 0, 0,   0, 0, 255, 0, 0, 0, 255, 255,
                      255, 0, 0, 0, 255, 0, 0, 0,   0, 0, 0, 0};
    DilateSample(mask, moduleName, fileName, outDir, u32Width, u32Height, rkive_time);
}

RK_VOID DilateSample5x5(const RK_CHAR *moduleName, RK_CHAR *fileName, RK_CHAR *outDir,
                        RK_U32 u32Width, RK_U32 u32Height, double &rkive_time) {
    RK_U8 mask[25] = {0,   0,   255, 0, 0,   0, 0, 255, 0, 0,   255, 255, 255,
                      255, 255, 0,   0, 255, 0, 0, 0,   0, 255, 0,   0};
    DilateSample(mask, moduleName, fileName, outDir, u32Width, u32Height, rkive_time);
}

// ========== OpenCV CPU膨胀（对比基准） ==========
void OpenCVDilate(const char *fileName, RK_U32 u32Width, RK_U32 u32Height, 
                  int kernel_size, double &opencv_time) {
    // 读取图像
    Mat src = imread(fileName, IMREAD_GRAYSCALE);
    if (src.empty()) {
        printf("OpenCV read image failed!\n");
        opencv_time = -1;
        return;
    }
    
    // 构造膨胀核（匹配RKIVE模板）
    Mat kernel;
    if (kernel_size == 3) {
        kernel = (Mat_<uchar>(3,3) << 0,0,0,
                                      0,255,0,
                                      0,255,255,255,0,0,0,255,0,0,0,0,0,0);
        kernel = kernel(Rect(0,0,3,3)); // 3x3核
    } else if (kernel_size == 5) {
        kernel = (Mat_<uchar>(5,5) << 0,0,255,0,0,
                                      0,0,255,0,0,
                                      255,255,255,255,255,
                                      0,0,255,0,0,
                                      0,0,255,0,0); // 5x5核
    } else {
        printf("Unsupported kernel size: %d\n", kernel_size);
        opencv_time = -1;
        return;
    }
    
    // OpenCV膨胀计时
    Mat dst;
    double start = get_current_time_ms();
    dilate(src, dst, kernel);
    opencv_time = get_current_time_ms() - start;
    
    // 保存结果（对比用）
    char out_file[256];
    snprintf(out_file, sizeof(out_file), "opencv_dilate_%dx%d.jpg", kernel_size, kernel_size);
    imwrite(out_file, dst);
    printf("OpenCV Dilate %dx%d time = %.2f ms (result saved to %s)\n", 
           kernel_size, kernel_size, opencv_time, out_file);
}



int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();  // 记录起始时间 
    
   // OpencvNCCMatch *pOpencvNCCMatch = new OpencvNCCMatch();
    //pOpencvNCCMatch->Init();

    //const char *src_path = argv[1];
    //const char *dst_path = argv[2];
    
    //Mat MatSrc = imread(src_path);
    //Mat MatDst = imread(dst_path);
    //Mat MatOut;
    //pOpencvNCCMatch->matching(MatSrc,MatDst,MatOut);
    //imwrite("./out.jpg",MatOut);


    // 计时开始
   // auto start = high_resolution_clock::now();

    // RKIVE初始化
    RK_S32 s32Ret = RK_MPI_IVE_Init();
    if (s32Ret != RK_SUCCESS) {
        printf("RK_MPI_IVE_Init failed! ret=%d\n", s32Ret);
        return -1;
    }



    // 齿轮NCC匹配
   // NCCGearMatch(argv[1], argv[2], argv[3]);


    // 解析参数
    const char *img_path = argv[1];
    RK_U32 width = atoi(argv[2]);
    RK_U32 height = atoi(argv[3]);
    const char *out_dir = "./"; // 输出目录

    // ========== 1. 初始化RKIVE ==========
    RK_S32 ret = RK_MPI_IVE_Init();
    if (ret != RK_SUCCESS) {
        printf("RK_MPI_IVE_Init failed! ret=%d\n", ret);
        return -1;
    }

    // ========== 2. 3x3膨胀对比 ==========
    printf("\n=== 3x3 Dilate Comparison ===\n");
    double rkive_3x3_time = 0.0;
    double opencv_3x3_time = 0.0;
    
    // RKIVE 3x3膨胀
    DilateSample3x3("dilate_3x3", (char*)img_path, (char*)out_dir, width, height, rkive_3x3_time);
    // OpenCV 3x3膨胀
    OpenCVDilate(img_path, width, height, 3, opencv_3x3_time);
    
    // 3x3对比结果
    if (rkive_3x3_time > 0 && opencv_3x3_time > 0) {
        double speedup = opencv_3x3_time / rkive_3x3_time;
        //printf("3x3 Dilate Speedup: RKIVE is %.2f times faster than OpenCV\n", speedup);
    }

    // ========== 3. 5x5膨胀对比 ==========
    printf("\n=== 5x5 Dilate Comparison ===\n");
    double rkive_5x5_time = 0.0;
    double opencv_5x5_time = 0.0;
    
    // RKIVE 5x5膨胀
    DilateSample5x5("dilate_5x5", (char*)img_path, (char*)out_dir, width, height, rkive_5x5_time);
    // OpenCV 5x5膨胀
    OpenCVDilate(img_path, width, height, 5, opencv_5x5_time);
    
    // 5x5对比结果
    if (rkive_5x5_time > 0 && opencv_5x5_time > 0) {
        double speedup = opencv_5x5_time / rkive_5x5_time;
        //printf("5x5 Dilate Speedup: RKIVE is %.2f times faster than OpenCV\n", speedup);
    }

    // ========== 4. 反初始化RKIVE ==========
    RK_MPI_IVE_Deinit();

    // ========== 5. 汇总结果 ==========
    printf("\n=== Final Comparison Summary ===\n");
    printf("3x3 Dilate: RKIVE=%.2f ms | OpenCV=%.2f ms | Speedup=%.2fx\n",
           rkive_3x3_time, opencv_3x3_time, opencv_3x3_time/rkive_3x3_time);
    printf("5x5 Dilate: RKIVE=%.2f ms | OpenCV=%.2f ms | Speedup=%.2fx\n",
           rkive_5x5_time, opencv_5x5_time, opencv_5x5_time/rkive_5x5_time);



    // RKIVE反初始化
    RK_MPI_IVE_Deinit();
    printf("Program finished!\n");

    
    auto endss = std::chrono::high_resolution_clock::now();  // 记录结束时间
	std::chrono::duration<double> durationss = endss - start;  // 计算时间差 
//	std::cout << "wk_Gray_scale_matching poscess time: " << durationss.count() << " seconds" << std::endl;
    return 0;
}
