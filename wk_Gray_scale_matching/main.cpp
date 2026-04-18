#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <unistd.h>   // sleep()
#include "OpencvNCCMatch.h"

using namespace std;
using namespace cv;


int main(int argc, char **argv)
{
    auto start = std::chrono::high_resolution_clock::now();  // 记录起始时间 
    
    OpencvNCCMatch *pOpencvNCCMatch = new OpencvNCCMatch();
    const char *src_path = argv[1];
    const char *dst_path = argv[2];
    
    Mat MatSrc = imread(src_path);
    Mat MatDst = imread(dst_path);
    Mat MatOut;
    pOpencvNCCMatch->matching(MatSrc,MatDst,MatOut);
    imwrite("./out.jpg",MatOut);
    
    auto endss = std::chrono::high_resolution_clock::now();  // 记录结束时间
	std::chrono::duration<double> durationss = endss - start;  // 计算时间差 
	std::cout << "wk_Gray_scale_matching poscess time: " << durationss.count() << " seconds" << std::endl;
    return 0;
}
