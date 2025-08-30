#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <unistd.h>   // sleep()
#include "version.h"
#include "OcrLite.h"
#include "OcrUtils.h"

int main()
{

    auto start = std::chrono::high_resolution_clock::now();

    std::string modelsDir, modelDetPath, modelClsPath, modelRecPath, keysPath;
    std::string imgPath, imgDir, imgName;
    int numThread = 1;//4
    int padding = 50;
    int maxSideLen = 384;//1024
    float boxScoreThresh = 0.6f;
    float boxThresh = 0.3f;
    float unClipRatio = 2.0f;
    bool doAngle = true;
    int flagDoAngle = 1;
    bool mostAngle = true;
    int flagMostAngle = 1;
    int flagGpu = -1;

    modelsDir = "./models";
    modelDetPath =  "./models/dbnet_op";
    modelClsPath =  "./models/angle_op";
    modelRecPath =  "./models/crnn_lite_op";
    keysPath =  "./models/keys.txt";
    imgPath = "./image/test.jpg";

    imgDir.assign(imgPath.substr(0, imgPath.find_last_of('/') + 1));
    imgName.assign(imgPath.substr(imgPath.find_last_of('/') + 1));


    int opt;
    int optionIndex = 0;
     if (modelDetPath.empty()) {
        modelDetPath = modelsDir + "/" + "dbnet_op";
    }
    if (modelClsPath.empty()) {
        modelClsPath = modelsDir + "/" + "angle_op";
    }
    if (modelRecPath.empty()) {
        modelRecPath = modelsDir + "/" + "crnn_lite_op";
    }
    if (keysPath.empty()) {
        keysPath = modelsDir + "/" + "keys.txt";
    }
    bool hasTargetImgFile = isFileExists(imgPath);
    if (!hasTargetImgFile) {
        fprintf(stderr, "Target image not found: %s\n", imgPath.c_str());
        return -1;
    }
    bool hasModelDetParam = isFileExists(modelDetPath + ".param");
    if (!hasModelDetParam) {
        fprintf(stderr, "Model dbnet file not found: %s.param\n", modelDetPath.c_str());
        return -1;
    }
    bool hasModelDetBin = isFileExists(modelDetPath + ".bin");
    if (!hasModelDetBin) {
        fprintf(stderr, "Model dbnet file not found: %s.bin\n", modelDetPath.c_str());
        return -1;
    }
    bool hasModelClsParam = isFileExists(modelClsPath + ".param");
    if (!hasModelClsParam) {
        fprintf(stderr, "Model angle file not found: %s.param\n", modelClsPath.c_str());
        return -1;
    }
    bool hasModelClsBin = isFileExists(modelClsPath + ".bin");
    if (!hasModelClsBin) {
        fprintf(stderr, "Model angle file not found: %s.bin\n", modelClsPath.c_str());
        return -1;
    }
    bool hasModelRecParam = isFileExists(modelRecPath + ".param");
    if (!hasModelRecParam) {
        fprintf(stderr, "Model crnn file not found: %s.param\n", modelRecPath.c_str());
        return -1;
    }
    bool hasModelRecBin = isFileExists(modelRecPath + ".bin");
    if (!hasModelRecBin) {
        fprintf(stderr, "Model crnn file not found: %s.bin\n", modelRecPath.c_str());
        return -1;
    }
    bool hasKeysFile = isFileExists(keysPath);
    if (!hasKeysFile) {
        fprintf(stderr, "keys file not found: %s\n", keysPath.c_str());
        return -1;
    }
    OcrLite ocrLite;
    ocrLite.setNumThread(numThread);
    ocrLite.initLogger(
            true,//isOutputConsole
            false,//isOutputPartImg
            true);//isOutputResultImg

    ocrLite.enableResultTxt(imgDir.c_str(), imgName.c_str());
    ocrLite.setGpuIndex(flagGpu);
    ocrLite.Logger("=====Input Params=====\n");
    ocrLite.Logger(
            "numThread(%d),padding(%d),maxSideLen(%d),boxScoreThresh(%f),boxThresh(%f),unClipRatio(%f),doAngle(%d),mostAngle(%d),GPU(%d)\n",
            numThread, padding, maxSideLen, boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle,
            flagGpu);

    bool initModelsRet = ocrLite.initModels(modelDetPath, modelClsPath, modelRecPath, keysPath);
    if (!initModelsRet) return -1;

    OcrResult result = ocrLite.detect(imgDir.c_str(), imgName.c_str(), padding, maxSideLen,
                                      boxScoreThresh, boxThresh, unClipRatio, doAngle, mostAngle);
    ocrLite.Logger("%s\n", result.strRes.c_str());
       // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();

    // 计算耗时，单位毫秒
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "耗时: " << duration << " ms" << std::endl;

/*

    cv::VideoCapture cap;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 320);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
    cap.open(0);

    const int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    const int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    fprintf(stderr, "%d x %d\n", w, h);

    cv::Mat bgr[9];
    for (int i = 0; i < 9; i++)
    {
        cap >> bgr[i];

        sleep(1);
    }

    cap.release();

    // combine into big image
    {
        cv::Mat out(h * 3, w * 3, CV_8UC3);
        bgr[0].copyTo(out(cv::Rect(0, 0, w, h)));
        bgr[1].copyTo(out(cv::Rect(w, 0, w, h)));
        bgr[2].copyTo(out(cv::Rect(w * 2, 0, w, h)));
        bgr[3].copyTo(out(cv::Rect(0, h, w, h)));
        bgr[4].copyTo(out(cv::Rect(w, h, w, h)));
        bgr[5].copyTo(out(cv::Rect(w * 2, h, w, h)));
        bgr[6].copyTo(out(cv::Rect(0, h * 2, w, h)));
        bgr[7].copyTo(out(cv::Rect(w, h * 2, w, h)));
        bgr[8].copyTo(out(cv::Rect(w * 2, h * 2, w, h)));

        cv::imwrite("out.jpg", out);
    }
*/
    return 0;
}
