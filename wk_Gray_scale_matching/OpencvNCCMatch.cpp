#include "OpencvNCCMatch.h"


#define VISION_TOLERANCE 0.0000001
#define D2R (CV_PI / 180.0)
#define R2D (180.0 / CV_PI)
#define MATCH_CANDIDATE_NUM 5


bool compareScoreBig2Small (const s_MatchParameter& lhs, const s_MatchParameter& rhs) { return  lhs.dMatchScore > rhs.dMatchScore; }
bool comparePtWithAngle (const pair<Point2f, double> lhs, const pair<Point2f, double> rhs) { return lhs.second < rhs.second; }
/*
//From ImageShop
// 4個有符號的32位的數據相加的和。
inline int _mm_hsum_epi32 (__m128i V)      // V3 V2 V1 V0
{
	// 實測這個速度要快些，_mm_extract_epi32最慢。
	__m128i T = _mm_add_epi32 (V, _mm_srli_si128 (V, 8));  // V3+V1   V2+V0  V1  V0  
	T = _mm_add_epi32 (T, _mm_srli_si128 (T, 4));    // V3+V1+V2+V0  V2+V0+V1 V1+V0 V0 
	return _mm_cvtsi128_si32 (T);       // 提取低位 
}

// 基於SSE的字節數據的乘法。
// <param name="Kernel">需要卷積的核矩陣。 </param>
// <param name="Conv">卷積矩陣。 </param>
// <param name="Length">矩陣所有元素的長度。 </param>
inline int IM_Conv_SIMD (unsigned char* pCharKernel, unsigned char *pCharConv, int iLength)
{
	const int iBlockSize = 16, Block = iLength / iBlockSize;
	__m128i SumV = _mm_setzero_si128 ();
	__m128i Zero = _mm_setzero_si128 ();
	for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize)
	{
		__m128i SrcK = _mm_loadu_si128 ((__m128i*)(pCharKernel + Y));
		__m128i SrcC = _mm_loadu_si128 ((__m128i*)(pCharConv + Y));
		__m128i SrcK_L = _mm_unpacklo_epi8 (SrcK, Zero);
		__m128i SrcK_H = _mm_unpackhi_epi8 (SrcK, Zero);
		__m128i SrcC_L = _mm_unpacklo_epi8 (SrcC, Zero);
		__m128i SrcC_H = _mm_unpackhi_epi8 (SrcC, Zero);
		__m128i SumT = _mm_add_epi32 (_mm_madd_epi16 (SrcK_L, SrcC_L), _mm_madd_epi16 (SrcK_H, SrcC_H));
		SumV = _mm_add_epi32 (SumV, SumT);
	}
	int Sum = _mm_hsum_epi32 (SumV);
	for (int Y = Block * iBlockSize; Y < iLength; Y++)
	{
		Sum += pCharKernel[Y] * pCharConv[Y];
	}
	return Sum;
}

*/
/**/
inline uint32_t hsum_u32(uint32x4_t v)
{
    uint32x2_t s = vadd_u32(vget_low_u32(v), vget_high_u32(v));
    s = vpadd_u32(s, s);
    return vget_lane_u32(s, 0);
}
/*
inline int32_t neon_haddw_s32(int16x8_t V)
{
	int32x4_t SumV = vpaddlq_s16(V);
	SumV = vaddq_s32(SumV, vextq_s32(SumV, SumV, 1)); // Optional: Enable for
	
	return neon_hsum_epi32(SumV);
}

inline int IM_Conv_SIMD(unsigned char* pCharKernel, unsigned char* pCharConv,
int iLength)
{
	
	const int iBlockSize = 16, Block = iLength / iBlockSize;
	int32x4_t SumV = vdupq_n_s32(0);
	uint8x16_t Zero = vdupq_n_u8(0);

	for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize)
	{
		uint8x16_t SrcK = vld1q_u8(pCharKernel + Y);
		uint8x16_t SrcC = vld1q_u8(pCharConv + Y);
		int16x8_t SrcK_L = (int16x8_t)vmovl_u8(vget_low_u8(SrcK));
		int16x8_t SrcK_H = (int16x8_t)vmovl_u8(vget_high_u8(SrcK));
		int16x8_t SrcC_L = (int16x8_t)vmovl_u8(vget_low_u8(SrcC));
		int16x8_t SrcC_H = (int16x8_t)vmovl_u8(vget_high_u8(SrcC));

		int32x4_t MulLow = vmull_s16(vget_low_s16(SrcK_L), vget_low_s16(SrcC_L));
		int32x4_t MulHigh = vmull_s16(vget_high_s16(SrcK_L), vget_high_s16(SrcC_L));
		int32x4_t SumT = vaddq_s32(MulLow, MulHigh);

		MulLow = vmull_s16(vget_low_s16(SrcK_H), vget_low_s16(SrcC_H));
		MulHigh = vmull_s16(vget_high_s16(SrcK_H), vget_high_s16(SrcC_H));
		SumT = vaddq_s32(SumT, vaddq_s32(MulLow, MulHigh));

		SumV = vaddq_s32(SumV, SumT);
	}

	int32_t Sum = neon_hsum_epi32(SumV);

	for (int Y = Block * iBlockSize; Y < iLength; Y++)
	{
		Sum += pCharKernel[Y] * pCharConv[Y];
	}
	return Sum;	

    
}
*/

inline uint32_t ConvRow_NEON_Correct(
    const uint8_t* __restrict k,
    const uint8_t* __restrict s,
    int len)
{
    uint32x4_t acc = vdupq_n_u32(0);

    int i = 0;
    for (; i <= len - 16; i += 16)
    {
        uint8x16_t vk = vld1q_u8(k + i);
        uint8x16_t vs = vld1q_u8(s + i);

        uint16x8_t m0 = vmull_u8(vget_low_u8(vk),  vget_low_u8(vs));
        uint16x8_t m1 = vmull_u8(vget_high_u8(vk), vget_high_u8(vs));

        uint32x4_t t0 = vaddl_u16(vget_low_u16(m0), vget_high_u16(m0));
        uint32x4_t t1 = vaddl_u16(vget_low_u16(m1), vget_high_u16(m1));

        acc = vaddq_u32(acc, vaddq_u32(t0, t1));
    }

    uint32_t sum = vgetq_lane_u32(acc,0)+vgetq_lane_u32(acc,1)+
        		   vgetq_lane_u32(acc,2)+vgetq_lane_u32(acc,3);

    for (; i < len; ++i)
        sum += k[i] * s[i];

    return sum;
}

inline uint32_t ConvRow_NEON_Fast(
    const uint8_t* __restrict k,
    const uint8_t* __restrict s,
    int len)
{
    uint32x4_t acc = vdupq_n_u32(0);
    int i = 0;

    for (; i <= len - 16; i += 16)
    {
        uint8x16_t vk = vld1q_u8(k + i);
        uint8x16_t vs = vld1q_u8(s + i);

        uint16x8_t m0 = vmull_u8(vget_low_u8(vk),  vget_low_u8(vs));
        uint16x8_t m1 = vmull_u8(vget_high_u8(vk), vget_high_u8(vs));

        // ⭐ 关键：pairwise add + widen（单指令完成）
        acc = vpadalq_u16(acc, m0);
        acc = vpadalq_u16(acc, m1);
    }

    uint32_t sum =
        vgetq_lane_u32(acc,0) +
        vgetq_lane_u32(acc,1) +
        vgetq_lane_u32(acc,2) +
        vgetq_lane_u32(acc,3);

    for (; i < len; ++i)
        sum += k[i] * s[i];

    return sum;
}


inline uint32_t ConvRow_NEON_A55_Max(
    const uint8_t* __restrict k,
    const uint8_t* __restrict s,
    int len)
{
    uint32x4_t acc0 = vdupq_n_u32(0);
    uint32x4_t acc1 = vdupq_n_u32(0);

    int i = 0;

    // 32 字节展开，双累加器隐藏延迟
    for (; i <= len - 32; i += 32)
    {
        // -------- 第 1 组 16B --------
        uint8x16_t vk0 = vld1q_u8(k + i);
        uint8x16_t vs0 = vld1q_u8(s + i);

        uint16x8_t m00 = vmull_u8(vget_low_u8(vk0),  vget_low_u8(vs0));
        uint16x8_t m01 = vmull_u8(vget_high_u8(vk0), vget_high_u8(vs0));

        uint32x4_t t00 = vaddl_u16(vget_low_u16(m00), vget_high_u16(m00));
        uint32x4_t t01 = vaddl_u16(vget_low_u16(m01), vget_high_u16(m01));

        acc0 = vaddq_u32(acc0, vaddq_u32(t00, t01));

        // -------- 第 2 组 16B --------
        uint8x16_t vk1 = vld1q_u8(k + i + 16);
        uint8x16_t vs1 = vld1q_u8(s + i + 16);

        uint16x8_t m10 = vmull_u8(vget_low_u8(vk1),  vget_low_u8(vs1));
        uint16x8_t m11 = vmull_u8(vget_high_u8(vk1), vget_high_u8(vs1));

        uint32x4_t t10 = vaddl_u16(vget_low_u16(m10), vget_high_u16(m10));
        uint32x4_t t11 = vaddl_u16(vget_low_u16(m11), vget_high_u16(m11));

        acc1 = vaddq_u32(acc1, vaddq_u32(t10, t11));
    }

    // 合并双累加器
    acc0 = vaddq_u32(acc0, acc1);

    uint32_t sum =
        vgetq_lane_u32(acc0, 0) +
        vgetq_lane_u32(acc0, 1) +
        vgetq_lane_u32(acc0, 2) +
        vgetq_lane_u32(acc0, 3);

    // 处理剩余
    for (; i < len; ++i)
        sum += k[i] * s[i];

    return sum;
}
inline uint32_t ConvRow_NEON_A55_Ultimate(
    const uint8_t* __restrict k,
    const uint8_t* __restrict s,
    int len)
{
    uint32x4_t acc0 = vdupq_n_u32(0);
    uint32x4_t acc1 = vdupq_n_u32(0);

    int i = 0;

    for (; i <= len - 32; i += 32)
    {
        __builtin_prefetch(k + i + 64, 0, 1);
        __builtin_prefetch(s + i + 64, 0, 1);

        // 先把数据全读出来（关键）
        uint8x16_t vk0 = vld1q_u8(k + i);
        uint8x16_t vs0 = vld1q_u8(s + i);
        uint8x16_t vk1 = vld1q_u8(k + i + 16);
        uint8x16_t vs1 = vld1q_u8(s + i + 16);

        // 再开始算
        uint16x8_t m00 = vmull_u8(vget_low_u8(vk0),  vget_low_u8(vs0));
        uint16x8_t m01 = vmull_u8(vget_high_u8(vk0), vget_high_u8(vs0));
        uint16x8_t m10 = vmull_u8(vget_low_u8(vk1),  vget_low_u8(vs1));
        uint16x8_t m11 = vmull_u8(vget_high_u8(vk1), vget_high_u8(vs1));

        uint32x4_t t00 = vaddl_u16(vget_low_u16(m00), vget_high_u16(m00));
        uint32x4_t t01 = vaddl_u16(vget_low_u16(m01), vget_high_u16(m01));
        uint32x4_t t10 = vaddl_u16(vget_low_u16(m10), vget_high_u16(m10));
        uint32x4_t t11 = vaddl_u16(vget_low_u16(m11), vget_high_u16(m11));

        acc0 = vaddq_u32(acc0, vaddq_u32(t00, t01));
        acc1 = vaddq_u32(acc1, vaddq_u32(t10, t11));
    }

    acc0 = vaddq_u32(acc0, acc1);

    uint32_t sum =
        vgetq_lane_u32(acc0, 0) +
        vgetq_lane_u32(acc0, 1) +
        vgetq_lane_u32(acc0, 2) +
        vgetq_lane_u32(acc0, 3);

    for (; i < len; ++i)
        sum += k[i] * s[i];

    return sum;
}


/*
inline uint32_t ConvRow_NEON_Fast(
    const uint8_t* __restrict k,
    const uint8_t* __restrict s,
    int len)
{
    uint32x4_t acc = vdupq_n_u32(0);
    int i = 0;

    for (; i <= len - 16; i += 16)
    {
        uint8x16_t vk = vld1q_u8(k + i);
        uint8x16_t vs = vld1q_u8(s + i);

        uint16x8_t m0 = vmull_u8(vget_low_u8(vk),  vget_low_u8(vs));
        uint16x8_t m1 = vmull_u8(vget_high_u8(vk), vget_high_u8(vs));

        acc = vpadalq_u16(acc, m0);
        acc = vpadalq_u16(acc, m1);
    }

    uint32x2_t t = vadd_u32(vget_low_u32(acc), vget_high_u32(acc));
    uint32_t sum = vget_lane_u32(vpadd_u32(t, t), 0);

    for (; i < len; ++i)
        sum += k[i] * s[i];

    return sum;
}
*/





inline uint32_t IM_Conv_SIMD(const uint8_t* pCharKernel, const uint8_t* pCharConv, int iLength)
{
	const int iBlockSize = 16;
	const int Block = iLength / iBlockSize;

	uint32x4_t SumV = vdupq_n_u32(0);

	for (int Y = 0; Y < Block * iBlockSize; Y += iBlockSize)
	{
		uint8x16_t SrcK = vld1q_u8(pCharKernel + Y);
		uint8x16_t SrcC = vld1q_u8(pCharConv + Y);

		// 使用无符号扩展
		uint16x8_t SrcK_L = vmovl_u8(vget_low_u8(SrcK));
		uint16x8_t SrcK_H = vmovl_u8(vget_high_u8(SrcK));
		uint16x8_t SrcC_L = vmovl_u8(vget_low_u8(SrcC));
		uint16x8_t SrcC_H = vmovl_u8(vget_high_u8(SrcC));

		uint32x4_t MulLow = vmull_u16(vget_low_u16(SrcK_L), vget_low_u16(SrcC_L));
		uint32x4_t MulHigh = vmull_u16(vget_high_u16(SrcK_L), vget_high_u16(SrcC_L));
		uint32x4_t SumT = vaddq_u32(MulLow, MulHigh);

		MulLow = vmull_u16(vget_low_u16(SrcK_H), vget_low_u16(SrcC_H));
		MulHigh = vmull_u16(vget_high_u16(SrcK_H), vget_high_u16(SrcC_H));
		SumT = vaddq_u32(SumT, vaddq_u32(MulLow, MulHigh));

		SumV = vaddq_u32(SumV, SumT);
	}

	uint32_t Sum = hsum_u32(SumV);

	// 尾部标量补充
	for (int Y = Block * iBlockSize; Y < iLength; Y++)
		Sum += pCharKernel[Y] * pCharConv[Y];

	return Sum;
}




OpencvNCCMatch::OpencvNCCMatch(/* args */)
{
    m_bToleranceRange = false;
	
	m_dToleranceAngle = 180;
	m_iMinReduceArea = 256;
	
	m_dMaxOverlap = 0.8;
	m_iMaxPos = 10;
	m_dScore = 0.7;
    m_dTolerance1 = 0;
    m_dTolerance2 = 0;
    m_dTolerance3 = 360;
    m_dTolerance4 = 0;

	cv::setNumThreads(5);
}

int OpencvNCCMatch::creatTemplateDst(cv::Mat dst)
{
	/*
	int iTopLayer = GetTopLayer (&dst, (int)sqrt ((double)m_iMinReduceArea));
	buildPyramid (dst, m_TemplData.vecPyramid, iTopLayer);
	s_TemplData* templData = &m_TemplData;
	templData->iBorderColor = mean (dst).val[0] < 128 ? 255 : 0;
	int iSize = templData->vecPyramid.size ();
	templData->resize (iSize);

	for (int i = 0; i < iSize; i++)
	{
		double invArea = 1. / ((double)templData->vecPyramid[i].rows * templData->vecPyramid[i].cols);
		Scalar templMean, templSdv;
		double templNorm = 0, templSum2 = 0;

		meanStdDev (templData->vecPyramid[i], templMean, templSdv);
		templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

		if (templNorm < DBL_EPSILON)
		{
			templData->vecResultEqual1[i] = true;
		}
		templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];


		templSum2 /= invArea;
		templNorm = std::sqrt (templNorm);
		templNorm /= std::sqrt (invArea); // care of accuracy here


		templData->vecInvArea[i] = invArea;
		templData->vecTemplMean[i] = templMean;
		templData->vecTemplNorm[i] = templNorm;
	}
	templData->bIsPatternLearned = true;
*/

	int iTopLayer = GetTopLayer(&dst, static_cast<int>(sqrt((double)m_iMinReduceArea)));
	buildPyramid(dst, m_TemplData.vecPyramid, iTopLayer);

	s_TemplData* templData = &m_TemplData;
	templData->iBorderColor = cv::mean(dst)[0] < 128 ? 255 : 0;

	int iSize = templData->vecPyramid.size();
	templData->resize(iSize);  // 确保所有 vector 成员预留空间

	for (int i = 0; i < iSize; ++i) {
		const cv::Mat& templ = templData->vecPyramid[i];
		int area = templ.rows * templ.cols;
		if (area == 0) continue;

		double invArea = 1.0 / area;
		cv::Scalar mean, stddev;
		cv::meanStdDev(templ, mean, stddev);

		double normSdv2 = sumSquare(stddev);
		double normMean2 = sumSquare(mean);

		if (normSdv2 < DBL_EPSILON)
			templData->vecResultEqual1[i] = true;

		double templSum2 = (normSdv2 + normMean2) / invArea;//模板图像的所有像素值的平方和
		double templNorm = std::sqrt(normSdv2 / invArea);//计算模板图像的总标准差（L2范数）并归一化，为后续 NCC 等匹配算法提供标准化因子。

		templData->vecTemplMean[i] = mean;
		templData->vecTemplNorm[i] = templNorm;
		templData->vecInvArea[i] = invArea;
	}

	templData->bIsPatternLearned = true;


    return 0;
}

OpencvNCCMatch::~OpencvNCCMatch()
{
}

int OpencvNCCMatch::matching(cv::Mat src,cv::Mat dst,cv::Mat& out)
{ 
	Mat mat = src.clone();
	cv::cvtColor(src, src, cv::COLOR_BGR2GRAY);
	cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);
    auto start = std::chrono::high_resolution_clock::now();  // 记录起始时间 
	
    if (src.empty () || dst.empty ()){
        
        std::cerr << "arc and dst possible null " << std::endl;
        return -1;
    }
    if ((dst.cols < src.cols && dst.rows > src.rows) || (dst.cols > src.cols && dst.rows < src.rows))
        return -1;
	if (dst.size ().area () > src.size ().area ())
        return -1;

	creatTemplateDst(dst);	
	if (!m_TemplData.bIsPatternLearned)
	{

		return -1;
	}
		
    //決定金字塔層數 總共為1 + iLayer層
	int iTopLayer = GetTopLayer (&dst, (int)sqrt ((double)m_iMinReduceArea));
	//std::cerr << "iTopLayer:" << iTopLayer<<std::endl;

	//建立金字塔
	std::vector<cv::Mat> vecMatSrcPyr;
	//if (m_ckBitwiseNot.GetCheck ())
	//{
	//	cv::Mat matNewSrc = 255 - src;
	//	buildPyramid (matNewSrc, vecMatSrcPyr, iTopLayer);
		
	//}
	//else
		buildPyramid (src, vecMatSrcPyr, iTopLayer);

	s_TemplData* pTemplData = &m_TemplData;

	//第一階段以最頂層找出大致角度與ROI
	double dAngleStep = atan (2.0 / max (pTemplData->vecPyramid[iTopLayer].cols, pTemplData->vecPyramid[iTopLayer].rows)) * R2D;

	//std::cerr << "dAngleStepo:" << dAngleStep<<std::endl;

	vector<double> vecAngles;
    if (m_bToleranceRange)
	{
		if (m_dTolerance1 >= m_dTolerance2 || m_dTolerance3 >= m_dTolerance4)
		{
			//AfxMessageBox (L"角度範圍設定異常，左值須小於右值");
            std::cerr << "角度範圍設定異常，左值須小於右值" << std::endl;
			return -1;
		}
		for (double dAngle = m_dTolerance1; dAngle < m_dTolerance2 + dAngleStep; dAngle += dAngleStep)
			vecAngles.push_back (dAngle);
		for (double dAngle = m_dTolerance3; dAngle < m_dTolerance4 + dAngleStep; dAngle += dAngleStep)
			vecAngles.push_back (dAngle);
	}
	else
	{
		if (m_dToleranceAngle < VISION_TOLERANCE)
			vecAngles.push_back (0.0);
		else
		{
			for (double dAngle = 0; dAngle < m_dToleranceAngle + dAngleStep; dAngle += dAngleStep)
				vecAngles.push_back (dAngle);
			for (double dAngle = -dAngleStep; dAngle > -m_dToleranceAngle - dAngleStep; dAngle -= dAngleStep)
				vecAngles.push_back (dAngle);
		}
	}

	int iTopSrcW = vecMatSrcPyr[iTopLayer].cols, iTopSrcH = vecMatSrcPyr[iTopLayer].rows;


	Point2f ptCenter ((iTopSrcW - 1) / 2.0f, (iTopSrcH - 1) / 2.0f);

    int iSize = (int)vecAngles.size ();

	//std::cerr << "iSize:" << iSize<<std::endl;
	//vector<s_MatchParameter> vecMatchParameter (iSize * (m_iMaxPos + MATCH_CANDIDATE_NUM));
	vector<s_MatchParameter> vecMatchParameter;
	//Caculate lowest score at every layer
	vector<double> vecLayerScore (iTopLayer + 1, m_dScore);
	for (int iLayer = 1; iLayer <= iTopLayer; iLayer++)
		vecLayerScore[iLayer] = vecLayerScore[iLayer - 1] * 0.9;

	Size sizePat = pTemplData->vecPyramid[iTopLayer].size ();

	//std::cerr << "sizePat:" << sizePat<<std::endl;
	bool bCalMaxByBlock = (vecMatSrcPyr[iTopLayer].size ().area () / sizePat.area () > 500) && m_iMaxPos > 10;

//	std::cerr << "bCalMaxByBlock:" << bCalMaxByBlock<<std::endl;
	for (int i = 0; i < iSize; i++)
	{
		Mat matRotatedSrc, matR = getRotationMatrix2D (ptCenter, vecAngles[i], 1);
		Mat matResult;
		Point ptMaxLoc;
		double dValue, dMaxVal;
		double dRotate = clock ();
		Size sizeBest = GetBestRotationSize (vecMatSrcPyr[iTopLayer].size (), pTemplData->vecPyramid[iTopLayer].size (), vecAngles[i]);

		float fTranslationX = (sizeBest.width - 1) / 2.0f - ptCenter.x;
		float fTranslationY = (sizeBest.height - 1) / 2.0f - ptCenter.y;
		matR.at<double> (0, 2) += fTranslationX;
		matR.at<double> (1, 2) += fTranslationY;
		warpAffine (vecMatSrcPyr[iTopLayer], matRotatedSrc, matR, sizeBest, INTER_LINEAR, BORDER_CONSTANT, Scalar (pTemplData->iBorderColor));

		MatchTemplate (matRotatedSrc, pTemplData, matResult, iTopLayer, false);

		if (bCalMaxByBlock)
		{
			s_BlockMax blockMax (matResult, pTemplData->vecPyramid[iTopLayer].size ());
			blockMax.GetMaxValueLoc (dMaxVal, ptMaxLoc);
			if (dMaxVal < vecLayerScore[iTopLayer])
				continue;
			vecMatchParameter.push_back (s_MatchParameter (Point2f (ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
			for (int j = 0; j < m_iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
			{
				ptMaxLoc = GetNextMaxLoc (matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size (), dValue, m_dMaxOverlap, blockMax);
				if (dValue < vecLayerScore[iTopLayer])
					break;
				vecMatchParameter.push_back (s_MatchParameter (Point2f (ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
			}
		}
		else
		{

			minMaxLoc (matResult, 0, &dMaxVal, 0, &ptMaxLoc);

			if (dMaxVal < vecLayerScore[iTopLayer])
				continue;
			vecMatchParameter.push_back (s_MatchParameter (Point2f (ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dMaxVal, vecAngles[i]));
			for (int j = 0; j < m_iMaxPos + MATCH_CANDIDATE_NUM - 1; j++)
			{
				ptMaxLoc = GetNextMaxLoc (matResult, ptMaxLoc, pTemplData->vecPyramid[iTopLayer].size (), dValue, m_dMaxOverlap);
				if (dValue < vecLayerScore[iTopLayer])
					break;
				vecMatchParameter.push_back (s_MatchParameter (Point2f (ptMaxLoc.x - fTranslationX, ptMaxLoc.y - fTranslationY), dValue, vecAngles[i]));
			}
		}
	}
	sort (vecMatchParameter.begin (), vecMatchParameter.end (), compareScoreBig2Small);

	
	int iMatchSize = (int)vecMatchParameter.size ();
	//std::cerr << "iMatchSizes:" << iMatchSize<<std::endl;

    //the following code is quite time-consuming;
	int iDstW = pTemplData->vecPyramid[iTopLayer].cols, iDstH = pTemplData->vecPyramid[iTopLayer].rows;

	//顯示第一層結果
	bool m_bStopLayer1 = 1;
	//第一階段結束
	bool bSubPixelEstimation = true;
	int iStopLayer = m_bStopLayer1 ? 1 : 0; //设置为1时：粗匹配，牺牲精度提升速度。
	//int iSearchSize = min (m_iMaxPos + MATCH_CANDIDATE_NUM, (int)vecMatchParameter.size ());//可能不需要搜尋到全部 太浪費時間
	vector<s_MatchParameter> vecAllResult;



	for (int i = 0; i < (int)vecMatchParameter.size (); i++)
	//for (int i = 0; i < iSearchSize; i++)
	{
		double dRAngle = -vecMatchParameter[i].dMatchAngle * D2R;
		Point2f ptLT = ptRotatePt2f (vecMatchParameter[i].pt, ptCenter, dRAngle);

		double dAngleStep = atan (2.0 / max (iDstW, iDstH)) * R2D;//min改為max
		vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep;
		vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep;

		vector<s_MatchParameter> localResults; // 每线程自己的结果容器
		if (iTopLayer <= iStopLayer)
		{
			vecMatchParameter[i].pt = Point2d (ptLT * ((iTopLayer == 0) ? 1 : 2));
			localResults.push_back (vecMatchParameter[i]);
		}
		else
		{
			for (int iLayer = iTopLayer - 1; iLayer >= iStopLayer; iLayer--)
			{
				//搜尋角度
				dAngleStep = atan (2.0 / max (pTemplData->vecPyramid[iLayer].cols, pTemplData->vecPyramid[iLayer].rows)) * R2D;//min改為max
				vector<double> vecAngles;
				//double dAngleS = vecMatchParameter[i].dAngleStart, dAngleE = vecMatchParameter[i].dAngleEnd;
				double dMatchedAngle = vecMatchParameter[i].dMatchAngle;
				if (m_bToleranceRange)
				{
					for (int i = -1; i <= 1; i++)
						vecAngles.push_back (dMatchedAngle + dAngleStep * i);
				}
				else
				{
					if (m_dToleranceAngle < VISION_TOLERANCE)
						vecAngles.push_back (0.0);
					else
						for (int i = -1; i <= 1; i++)
							vecAngles.push_back (dMatchedAngle + dAngleStep * i);
				}
				Point2f ptSrcCenter ((vecMatSrcPyr[iLayer].cols - 1) / 2.0f, (vecMatSrcPyr[iLayer].rows - 1) / 2.0f);
				iSize = (int)vecAngles.size ();
				vector<s_MatchParameter> vecNewMatchParameter (iSize);
				int iMaxScoreIndex = 0;
				double dBigValue = -1;


				for (int j = 0; j < iSize; j++)
				{
					Mat matResult, matRotatedSrc;
					double dMaxValue = 0;
					Point ptMaxLoc;
					GetRotatedROI (vecMatSrcPyr[iLayer], pTemplData->vecPyramid[iLayer].size (), ptLT * 2, vecAngles[j], matRotatedSrc);

					MatchTemplate (matRotatedSrc, pTemplData, matResult, iLayer, true);
					//matchTemplate (matRotatedSrc, pTemplData->vecPyramid[iLayer], matResult, CV_TM_CCOEFF_NORMED);
					minMaxLoc (matResult, 0, &dMaxValue, 0, &ptMaxLoc);
					vecNewMatchParameter[j] = s_MatchParameter (ptMaxLoc, dMaxValue, vecAngles[j]);
					
					if (vecNewMatchParameter[j].dMatchScore > dBigValue)
					{
						iMaxScoreIndex = j;
						dBigValue = vecNewMatchParameter[j].dMatchScore;
					}
					//次像素估計
					if (ptMaxLoc.x == 0 || ptMaxLoc.y == 0 || ptMaxLoc.x == matResult.cols - 1 || ptMaxLoc.y == matResult.rows - 1)
						vecNewMatchParameter[j].bPosOnBorder = true;
					if (!vecNewMatchParameter[j].bPosOnBorder)
					{
						for (int y = -1; y <= 1; y++)
							for (int x = -1; x <= 1; x++)
								vecNewMatchParameter[j].vecResult[x + 1][y + 1] = matResult.at<float> (ptMaxLoc + Point (x, y));
					}
					//次像素估計
				}
				
				if (vecNewMatchParameter[iMaxScoreIndex].dMatchScore < vecLayerScore[iLayer])
					break;
				//次像素估計
				if (bSubPixelEstimation 
					&& iLayer == 0 
					&& (!vecNewMatchParameter[iMaxScoreIndex].bPosOnBorder) 
					&& iMaxScoreIndex != 0 
					&& iMaxScoreIndex != 2)
				{
					double dNewX = 0, dNewY = 0, dNewAngle = 0;
					SubPixEsimation ( &vecNewMatchParameter, &dNewX, &dNewY, &dNewAngle, dAngleStep, iMaxScoreIndex);
					vecNewMatchParameter[iMaxScoreIndex].pt = Point2d (dNewX, dNewY);
					vecNewMatchParameter[iMaxScoreIndex].dMatchAngle = dNewAngle;
				}
				//次像素估計

				double dNewMatchAngle = vecNewMatchParameter[iMaxScoreIndex].dMatchAngle;

				//讓坐標系回到旋轉時(GetRotatedROI)的(0, 0)
				Point2f ptPaddingLT = ptRotatePt2f (ptLT * 2, ptSrcCenter, dNewMatchAngle * D2R) - Point2f (3, 3);
				Point2f pt (vecNewMatchParameter[iMaxScoreIndex].pt.x + ptPaddingLT.x, vecNewMatchParameter[iMaxScoreIndex].pt.y + ptPaddingLT.y);
				//再旋轉
				pt = ptRotatePt2f (pt, ptSrcCenter, -dNewMatchAngle * D2R);

				if (iLayer == iStopLayer)
				{
					vecNewMatchParameter[iMaxScoreIndex].pt = pt * (iStopLayer == 0 ? 1 : 2);
					localResults.push_back (vecNewMatchParameter[iMaxScoreIndex]);
				}
				else
				{
					//更新MatchAngle ptLT
					vecMatchParameter[i].dMatchAngle = dNewMatchAngle;
					vecMatchParameter[i].dAngleStart = vecMatchParameter[i].dMatchAngle - dAngleStep / 2;
					vecMatchParameter[i].dAngleEnd = vecMatchParameter[i].dMatchAngle + dAngleStep / 2;
					ptLT = pt;
				}
			}

		}
		

		{
			vecAllResult.insert(vecAllResult.end(), localResults.begin(), localResults.end());
		}
		
	}
	//cv::ocl::setUseOpenCL(true);

	auto end = std::chrono::high_resolution_clock::now();  // 记录结束时间
	std::chrono::duration<double> duration = end - start;  // 计算时间差 
	//std::cout << "OpencvNCCMatch Time: " << duration.count() << " seconds" << std::endl;
	FilterWithScore (&vecAllResult, m_dScore);

	//最後濾掉重疊
	iDstW = pTemplData->vecPyramid[iStopLayer].cols * (iStopLayer == 0 ? 1 : 2);
	iDstH = pTemplData->vecPyramid[iStopLayer].rows * (iStopLayer == 0 ? 1 : 2);

	for (int i = 0; i < (int)vecAllResult.size (); i++)
	{
		Point2f ptLT, ptRT, ptRB, ptLB;
		double dRAngle = -vecAllResult[i].dMatchAngle * D2R;
		ptLT = vecAllResult[i].pt;
		ptRT = Point2f (ptLT.x + iDstW * (float)cos (dRAngle), ptLT.y - iDstW * (float)sin (dRAngle));
		ptLB = Point2f (ptLT.x + iDstH * (float)sin (dRAngle), ptLT.y + iDstH * (float)cos (dRAngle));
		ptRB = Point2f (ptRT.x + iDstH * (float)sin (dRAngle), ptRT.y + iDstH * (float)cos (dRAngle));
		//紀錄旋轉矩形
		vecAllResult[i].rectR = RotatedRect(ptLT, ptRT, ptRB);
	}
	FilterWithRotatedRect (&vecAllResult, 5, m_dMaxOverlap);
	//最後濾掉重疊

	//根據分數排序
	sort (vecAllResult.begin (), vecAllResult.end (), compareScoreBig2Small);
	//m_strExecureTime.Format (L"%s : %d ms", m_strLanExecutionTime, int (clock () - d1));
	//m_statusBar.SetPaneText (0, m_strExecureTime);
	
	//m_vecSingleTargetData.clear ();
	//m_listMsg.DeleteAllItems ();
	iMatchSize = (int)vecAllResult.size ();
	if (vecAllResult.size () == 0){
		
		out = mat.clone();
		return false;
	}
		
	int iW = pTemplData->vecPyramid[0].cols, iH = pTemplData->vecPyramid[0].rows;

	for (int i = 0; i < iMatchSize; i++)
	{
		s_SingleTargetMatch sstm;
		double dRAngle = -vecAllResult[i].dMatchAngle * D2R;

		sstm.ptLT = vecAllResult[i].pt;

		sstm.ptRT = Point2d (sstm.ptLT.x + iW * cos (dRAngle), sstm.ptLT.y - iW * sin (dRAngle));
		sstm.ptLB = Point2d (sstm.ptLT.x + iH * sin (dRAngle), sstm.ptLT.y + iH * cos (dRAngle));
		sstm.ptRB = Point2d (sstm.ptRT.x + iH * sin (dRAngle), sstm.ptRT.y + iH * cos (dRAngle));
		sstm.ptCenter = Point2d ((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
		sstm.dMatchedAngle = -vecAllResult[i].dMatchAngle;
		sstm.dMatchScore = vecAllResult[i].dMatchScore;

		if (sstm.dMatchedAngle < -180)
			sstm.dMatchedAngle += 360;
		if (sstm.dMatchedAngle > 180)
			sstm.dMatchedAngle -= 360;
		m_vecSingleTargetData.push_back (sstm);

		

		//Test Subpixel
		/*Point2d ptLT = vecAllResult[i].ptSubPixel;
		Point2d ptRT = Point2d (sstm.ptLT.x + iW * cos (dRAngle), sstm.ptLT.y - iW * sin (dRAngle));
		Point2d ptLB = Point2d (sstm.ptLT.x + iH * sin (dRAngle), sstm.ptLT.y + iH * cos (dRAngle));
		Point2d ptRB = Point2d (sstm.ptRT.x + iH * sin (dRAngle), sstm.ptRT.y + iH * cos (dRAngle));
		Point2d ptCenter = Point2d ((sstm.ptLT.x + sstm.ptRT.x + sstm.ptRB.x + sstm.ptLB.x) / 4, (sstm.ptLT.y + sstm.ptRT.y + sstm.ptRB.y + sstm.ptLB.y) / 4);
		CString strDiff;strDiff.Format (L"Diff(x, y):%.3f, %.3f", ptCenter.x - sstm.ptCenter.x, ptCenter.y - sstm.ptCenter.y);
		AfxMessageBox (strDiff);*/
      
        // Test Subpixel
        // 存出MATCH ROI
        // OutputRoi (sstm);
        if (i + 1 == m_iMaxPos)
			break;
	}
	//std::cerr << "iMatchSize:" << iMatchSize<<std::endl;
	//sort (m_vecSingleTargetData.begin (), m_vecSingleTargetData.end (), compareMatchResultByPosX);
	
	//m_listMsg.DeleteAllItems ();

	for (int i = 0 ; i < (int)m_vecSingleTargetData.size (); i++)
	{
		    s_SingleTargetMatch sstm = m_vecSingleTargetData[i];
		    Point ptLT (m_vecSingleTargetData[i].ptLT * 1);
			Point ptLB (m_vecSingleTargetData[i].ptLB * 1);
			Point ptRB (m_vecSingleTargetData[i].ptRB * 1);
			Point ptRT (m_vecSingleTargetData[i].ptRT * 1);
			Point ptC (m_vecSingleTargetData[i].ptCenter * 1);
			DrawDashLine (mat, ptLT, ptLB,cv::Scalar(0, 255, 0),cv::Scalar(0, 0, 255));
			DrawDashLine (mat, ptLB, ptRB,cv::Scalar(0, 255, 0),cv::Scalar(0, 0, 255));
			DrawDashLine (mat, ptRB, ptRT,cv::Scalar(0, 255, 0),cv::Scalar(0, 0, 255));
			DrawDashLine (mat, ptRT, ptLT,cv::Scalar(0, 255, 0),cv::Scalar(0, 0, 255));

			//左上及角落邊框
			Point ptDis1, ptDis2;
			if (dst.cols > dst.rows)
			{
				ptDis1 = (ptLB - ptLT) / 3;
				ptDis2 = (ptRT - ptLT) / 3 * (dst.rows / (float)dst.cols);
			}
			else
			{
				ptDis1 = (ptLB - ptLT) / 3 * (dst.cols / (float)dst.rows);
				ptDis2 = (ptRT - ptLT) / 3;
			}
			line (mat, ptLT, ptLT + ptDis1 / 2, cv::Scalar(0, 255, 0), 2, LINE_AA);
			line (mat, ptLT, ptLT + ptDis2 / 2, cv::Scalar(0, 255, 0), 2, LINE_AA);
			line (mat, ptRT, ptRT + ptDis1 / 2, cv::Scalar(0, 255, 0), 2, LINE_AA);
			line (mat, ptRT, ptRT - ptDis2 / 2, cv::Scalar(0, 255, 0), 2, LINE_AA);
			line (mat, ptRB, ptRB - ptDis1 / 2, cv::Scalar(0, 255, 0), 2, LINE_AA);
			line (mat, ptRB, ptRB - ptDis2 / 2, cv::Scalar(0, 255, 0), 2, LINE_AA);
			line (mat, ptLB, ptLB - ptDis1 / 2, cv::Scalar(0, 255, 0), 2, LINE_AA);
			line (mat, ptLB, ptLB + ptDis2 / 2, cv::Scalar(0, 255, 0), 2, LINE_AA);
			//

			DrawDashLine (mat, ptLT + ptDis1, ptLT + ptDis2,cv::Scalar(0, 255, 0),cv::Scalar(0, 0, 255));
			DrawMarkCross (mat, ptC.x, ptC.y, 2, cv::Scalar(0, 255, 0), 5);
			string str = "Angle:" + std::to_string(sstm.dMatchedAngle) +" "+ "Point:" + "(" + std::to_string((int)sstm.ptCenter.x)+","+std::to_string((int)sstm.ptCenter.y)+")";
			putText (mat, str, Point((int)sstm.ptCenter.x,(int)sstm.ptCenter.y), FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));
			std::cerr << "m_listMsg:" << static_cast<float>(sstm.dMatchScore)<<std::endl;
		
		//Msg
	}


	//m_strTotalNum.Format (L"%d", (int)m_vecSingleTargetData.size ());
	//UpdateData (FALSE);
	//m_bShowResult = true;

	out = mat.clone();
    return 0;
}
int OpencvNCCMatch::GetTopLayer (cv::Mat* matTempl, int iMinDstLength)
{
	int iTopLayer = 0;
	int iMinReduceArea = iMinDstLength * iMinDstLength;
	int iArea = matTempl->cols * matTempl->rows;
	while (iArea > iMinReduceArea)
	{
		iArea /= 4;
		iTopLayer++;
	}
	return iTopLayer;
}
Size OpencvNCCMatch::GetBestRotationSize (Size sizeSrc, Size sizeDst, double dRAngle)
{
	double dRAngle_radian = dRAngle * D2R;
	Point ptLT (0, 0), ptLB (0, sizeSrc.height - 1), ptRB (sizeSrc.width - 1, sizeSrc.height - 1), ptRT (sizeSrc.width - 1, 0);
	Point2f ptCenter ((sizeSrc.width - 1) / 2.0f, (sizeSrc.height - 1) / 2.0f);
	Point2f ptLT_R = ptRotatePt2f (Point2f (ptLT), ptCenter, dRAngle_radian);
	Point2f ptLB_R = ptRotatePt2f (Point2f (ptLB), ptCenter, dRAngle_radian);
	Point2f ptRB_R = ptRotatePt2f (Point2f (ptRB), ptCenter, dRAngle_radian);
	Point2f ptRT_R = ptRotatePt2f (Point2f (ptRT), ptCenter, dRAngle_radian);

	float fTopY = max (max (ptLT_R.y, ptLB_R.y), max (ptRB_R.y, ptRT_R.y));
	float fBottomY = min (min (ptLT_R.y, ptLB_R.y), min (ptRB_R.y, ptRT_R.y));
	float fRightX = max (max (ptLT_R.x, ptLB_R.x), max (ptRB_R.x, ptRT_R.x));
	float fLeftX = min (min (ptLT_R.x, ptLB_R.x), min (ptRB_R.x, ptRT_R.x));

	if (dRAngle > 360)
		dRAngle -= 360;
	else if (dRAngle < 0)
		dRAngle += 360;

	if (fabs (fabs (dRAngle) - 90) < VISION_TOLERANCE || fabs (fabs (dRAngle) - 270) < VISION_TOLERANCE)
	{
		return Size (sizeSrc.height, sizeSrc.width);
	}
	else if (fabs (dRAngle) < VISION_TOLERANCE || fabs (fabs (dRAngle) - 180) < VISION_TOLERANCE)
	{
		return sizeSrc;
	}
	
	double dAngle = dRAngle;

	if (dAngle > 0 && dAngle < 90)
	{
		;
	}
	else if (dAngle > 90 && dAngle < 180)
	{
		dAngle -= 90;
	}
	else if (dAngle > 180 && dAngle < 270)
	{
		dAngle -= 180;
	}
	else if (dAngle > 270 && dAngle < 360)
	{
		dAngle -= 270;
	}
	else//Debug
	{
		//AfxMessageBox (L"Unkown");
	}

	float fH1 = sizeDst.width * sin (dAngle * D2R) * cos (dAngle * D2R);
	float fH2 = sizeDst.height * sin (dAngle * D2R) * cos (dAngle * D2R);

	int iHalfHeight = (int)ceil (fTopY - ptCenter.y - fH1);
	int iHalfWidth = (int)ceil (fRightX - ptCenter.x - fH2);
	
	Size sizeRet (iHalfWidth * 2, iHalfHeight * 2);

	int bWrongSize = (sizeDst.width < sizeRet.width && sizeDst.height > sizeRet.height)
		|| (sizeDst.width > sizeRet.width && sizeDst.height < sizeRet.height
			|| sizeDst.area () > sizeRet.area ());
	if (bWrongSize)
		sizeRet = Size (int (fRightX - fLeftX + 0.5), int (fTopY - fBottomY + 0.5));

	return sizeRet;
}
Point2f OpencvNCCMatch::ptRotatePt2f (Point2f ptInput, Point2f ptOrg, double dAngle)
{
	double dWidth = ptOrg.x * 2;
	double dHeight = ptOrg.y * 2;
	double dY1 = dHeight - ptInput.y, dY2 = dHeight - ptOrg.y;

	double dX = (ptInput.x - ptOrg.x) * cos (dAngle) - (dY1 - ptOrg.y) * sin (dAngle) + ptOrg.x;
	double dY = (ptInput.x - ptOrg.x) * sin (dAngle) + (dY1 - ptOrg.y) * cos (dAngle) + dY2;

	dY = -dY + dHeight;
	return Point2f ((float)dX, (float)dY);
}

void OpencvNCCMatch::MatchTemplate (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer, bool bUseSIMD)
{
	if (false)
	{
/*
		// 先准备模板缓存到连续内存
		std::vector<uint8_t> templBuffer;
		int templRows = pTemplData->vecPyramid[iLayer].rows;
		int templCols = pTemplData->vecPyramid[iLayer].cols;
		templBuffer.reserve(templRows * templCols);
		for (int i = 0; i < templRows; ++i) {
			const uint8_t* ptr = pTemplData->vecPyramid[iLayer].ptr<uint8_t>(i);
			templBuffer.insert(templBuffer.end(), ptr, ptr + templCols);
		}

		// 结果矩阵初始化
		matResult.create(matSrc.rows - templRows + 1, matSrc.cols - templCols + 1, CV_32FC1);
		matResult.setTo(0);

		int srcStep = matSrc.cols;
		int resultRows = matResult.rows;
		int resultCols = matResult.cols;
		for (int r = 0; r < resultRows; ++r) {
			for (int c = 0; c < resultCols; ++c) {
				float sumVal = 0.f;
				const uint8_t* srcRowPtr = matSrc.ptr<uint8_t>(r);
				// 遍历模板每一行，调用IM_Conv_SIMD做每行卷积累加
				for (int t_r = 0; t_r < templRows; ++t_r) {
					const uint8_t* templRowPtr = templBuffer.data() + t_r * templCols;
					const uint8_t* srcPatchPtr = srcRowPtr + c + t_r * srcStep;
					sumVal += IM_Conv_SIMD(templRowPtr, srcPatchPtr, templCols);
				}
				matResult.at<float>(r, c) = sumVal;
			}
		}
*/		
	
	int templRows = pTemplData->vecPyramid[iLayer].rows;
	int templCols = pTemplData->vecPyramid[iLayer].cols;

	std::vector<uint8_t> templBuf(templRows * templCols);

	for (int r = 0; r < templRows; ++r)
	{
		memcpy(templBuf.data() + r * templCols,
			pTemplData->vecPyramid[iLayer].ptr<uint8_t>(r),
			templCols);
	}

	matResult.create(matSrc.rows - templRows + 1,
                 matSrc.cols - templCols + 1,
                 CV_32FC1);

	int srcStep = matSrc.cols;
	int resRows = matResult.rows;
	int resCols = matResult.cols;


for (int r = 0; r < resRows; ++r)
{
    float* resPtr = matResult.ptr<float>(r);
    const uint8_t* srcRowPtr = matSrc.ptr<uint8_t>(r);

    for (int c = 0; c < resCols; ++c)
    {
        uint32_t sumVal = 0;
        const uint8_t* srcBase = srcRowPtr + c;

        for (int tr = 0; tr < templRows; ++tr)
        {
            const uint8_t* k = templBuf.data() + tr * templCols;
            const uint8_t* s = srcBase + tr * srcStep;

            sumVal += ConvRow_NEON_A55_Max(k, s, templCols);
        }

        resPtr[c] = (float)sumVal;
    }
}




	/*
		//From ImageShop
		matResult.create (matSrc.rows - pTemplData->vecPyramid[iLayer].rows + 1,
			matSrc.cols - pTemplData->vecPyramid[iLayer].cols + 1, CV_32FC1);
		matResult.setTo (0);
		cv::Mat& matTemplate = pTemplData->vecPyramid[iLayer];

		int  t_r_end = matTemplate.rows, t_r = 0;
		#pragma omp parallel for
		for (int r = 0; r < matResult.rows; r++)
		{
			float* r_matResult = matResult.ptr<float> (r);
			uchar* r_source = matSrc.ptr<uchar> (r);
			uchar* r_template, *r_sub_source;
			for (int c = 0; c < matResult.cols; ++c, ++r_matResult, ++r_source)
			{
				r_template = matTemplate.ptr<uchar> ();
				r_sub_source = r_source;
				for (t_r = 0; t_r < t_r_end; ++t_r, r_sub_source += matSrc.cols, r_template += matTemplate.cols)
				{
					*r_matResult = *r_matResult + IM_Conv_SIMD (r_template, r_sub_source, matTemplate.cols);
				}
			}
			
		}
*/
		//From ImageShop
		//pOpenclAccelerate.matchTemplateAccelerateTMCCORR(matSrc, pTemplData->vecPyramid[iLayer], matResult);
		//MatchMethodGPU method = GPU_TM_CCORR; // 或 GPU_NCC
		//pOpenclAccelerate.MatchTemplateGPU_Switch(matSrc,  pTemplData->vecPyramid[iLayer], matResult, method);
		//cv::Point bestLoc;
		//double minVal, maxVal;
		//cv::minMaxLoc(matResult, &minVal, &maxVal, nullptr, &bestLoc);
		//std::cout << "Best match at: " << bestLoc << ", score = " << maxVal << std::endl;
		
	}
	else{
		
		//matchTemplate (matSrc, pTemplData->vecPyramid[iLayer], matResult, TM_CCORR);//
		matchTemplate (matSrc, pTemplData->vecPyramid[iLayer], matResult, TM_CCORR);
	}

/*Mat diff;
absdiff(matResult, matResult, diff);
double dMaxValue;
minMaxLoc(diff, 0, &dMaxValue, 0,0);*/
	CCOEFF_Denominator (matSrc, pTemplData, matResult, iLayer);
}
void OpencvNCCMatch::CCOEFF_Denominator (cv::Mat& matSrc, s_TemplData* pTemplData, cv::Mat& matResult, int iLayer)
{
	/* yuanshi
	if (pTemplData->vecResultEqual1[iLayer])
	{
		matResult = Scalar::all (1);
		return;
	}
	double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;

	Mat sum, sqsum;
	integral (matSrc, sum, sqsum, CV_64F);

	q0 = (double*)sqsum.data;
	q1 = q0 + pTemplData->vecPyramid[iLayer].cols;
	q2 = (double*)(sqsum.data + pTemplData->vecPyramid[iLayer].rows * sqsum.step);
	q3 = q2 + pTemplData->vecPyramid[iLayer].cols;

	double* p0 = (double*)sum.data;
	double* p1 = p0 + pTemplData->vecPyramid[iLayer].cols;
	double* p2 = (double*)(sum.data + pTemplData->vecPyramid[iLayer].rows*sum.step);
	double* p3 = p2 + pTemplData->vecPyramid[iLayer].cols;

	int sumstep = sum.data ? (int)(sum.step / sizeof (double)) : 0;
	int sqstep = sqsum.data ? (int)(sqsum.step / sizeof (double)) : 0;

	//
	double dTemplMean0 = pTemplData->vecTemplMean[iLayer][0];
	double dTemplNorm = pTemplData->vecTemplNorm[iLayer];
	double dInvArea = pTemplData->vecInvArea[iLayer];
	//

	int i, j;
	for (i = 0; i < matResult.rows; i++)
	{
		float* rrow = matResult.ptr<float> (i);
		int idx = i * sumstep;
		int idx2 = i * sqstep;

		for (j = 0; j < matResult.cols; j += 1, idx += 1, idx2 += 1)
		{
			double num = rrow[j], t;
			double wndMean2 = 0, wndSum2 = 0;

			t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
			wndMean2 += t * t;
			num -= t * dTemplMean0;
			wndMean2 *= dInvArea;


			t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
			wndSum2 += t;


			//t = std::sqrt (MAX (wndSum2 - wndMean2, 0)) * dTemplNorm;

			double diff2 = MAX (wndSum2 - wndMean2, 0);
			if (diff2 <= std::min (0.5, 10 * FLT_EPSILON * wndSum2))
				t = 0; // avoid rounding errors
			else
				t = std::sqrt (diff2)*dTemplNorm;

			if (fabs (num) < t)
				num /= t;
			else if (fabs (num) < t * 1.125)
				num = num > 0 ? 1 : -1;
			else
				num = 0;

			rrow[j] = (float)num;
		}
	}
	*/
	if (pTemplData->vecResultEqual1[iLayer]) {
        matResult = cv::Scalar::all(1);
        return;
    }

    const cv::Mat& templ = pTemplData->vecPyramid[iLayer];
    int tw = templ.cols, th = templ.rows;
    double templMean = pTemplData->vecTemplMean[iLayer][0];
    double templNorm = pTemplData->vecTemplNorm[iLayer];
    double invArea = pTemplData->vecInvArea[iLayer];

    // 计算积分图
    cv::Mat sum, sqsum;
    cv::integral(matSrc, sum, sqsum, CV_64F);

    int sumStep = static_cast<int>(sum.step / sizeof(double));
    int sqStep  = static_cast<int>(sqsum.step / sizeof(double));

    for (int y = 0; y < matResult.rows; ++y) {
        float* rrow = matResult.ptr<float>(y);
        for (int x = 0; x < matResult.cols; ++x) {
            // 滑动窗口统计量
            int idx0 = y * sumStep + x;
            int idx1 = y * sumStep + x + tw;
            int idx2 = (y + th) * sumStep + x;
            int idx3 = (y + th) * sumStep + x + tw;

            double wndSum  = sum.at<double>(y + th, x + tw) + sum.at<double>(y, x) 
                           - sum.at<double>(y + th, x) - sum.at<double>(y, x + tw);
            double wndSqSum = sqsum.at<double>(y + th, x + tw) + sqsum.at<double>(y, x) 
                            - sqsum.at<double>(y + th, x) - sqsum.at<double>(y, x + tw);

            double wndMean2 = wndSum * wndSum * invArea;
            double num = rrow[x] - wndSum * templMean;

            double denom = wndSqSum - wndMean2;
            if (denom <= std::min(0.5, 10 * FLT_EPSILON * wndSqSum)) {
                // 用于调试：打印一些异常分数
                if (y % 20 == 0 && x % 20 == 0) {
                    std::cout << "[WARNING] denom≈0 at (" << x << "," << y << "), score = 0" << std::endl;
                }
                rrow[x] = 0.0f;
            } else {
                double scale = std::sqrt(denom) * templNorm;
                if (fabs(num) < scale)
                    rrow[x] = static_cast<float>(num / scale);
                else if (fabs(num) < scale * 1.125)
                    rrow[x] = (num > 0 ? 1.f : -1.f);
                else
                    rrow[x] = 0.0f;
            }
        }
    }
}
Point OpencvNCCMatch::GetNextMaxLoc (Mat & matResult, Point ptMaxLoc, Size sizeTemplate, double & dMaxValue, double dMaxOverlap, s_BlockMax & blockMax)
{
	//比對到的區域需考慮重疊比例
	int iStartX = int (ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap));
	int iStartY = int (ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap));
	Rect rectIgnore (iStartX, iStartY, int (2 * sizeTemplate.width * (1 - dMaxOverlap))
		, int (2 * sizeTemplate.height * (1 - dMaxOverlap)));
	//塗黑
	rectangle (matResult, rectIgnore , Scalar (-1), -1);
	blockMax.UpdateMax (rectIgnore);
	Point ptReturn;
	blockMax.GetMaxValueLoc (dMaxValue, ptReturn);
	return ptReturn;
}
Point OpencvNCCMatch::GetNextMaxLoc (Mat & matResult, Point ptMaxLoc, Size sizeTemplate, double& dMaxValue, double dMaxOverlap)
{
	//比對到的區域完全不重疊 : +-一個樣板寬高
	//int iStartX = ptMaxLoc.x - iTemplateW;
	//int iStartY = ptMaxLoc.y - iTemplateH;
	//int iEndX = ptMaxLoc.x + iTemplateW;

	//int iEndY = ptMaxLoc.y + iTemplateH;
	////塗黑
	//rectangle (matResult, Rect (iStartX, iStartY, 2 * iTemplateW * (1-dMaxOverlap * 2), 2 * iTemplateH * (1-dMaxOverlap * 2)), Scalar (dMinValue), CV_FILLED);
	////得到下一個最大值
	//Point ptNewMaxLoc;
	//minMaxLoc (matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
	//return ptNewMaxLoc;

	//比對到的區域需考慮重疊比例
	int iStartX = ptMaxLoc.x - sizeTemplate.width * (1 - dMaxOverlap);
	int iStartY = ptMaxLoc.y - sizeTemplate.height * (1 - dMaxOverlap);
	//塗黑
	rectangle (matResult, Rect (iStartX, iStartY, 2 * sizeTemplate.width * (1- dMaxOverlap), 2 * sizeTemplate.height * (1- dMaxOverlap)), Scalar (-1), -1);
	//得到下一個最大值
	Point ptNewMaxLoc;
	minMaxLoc (matResult, 0, &dMaxValue, 0, &ptNewMaxLoc);
	return ptNewMaxLoc;
}
void OpencvNCCMatch::GetRotatedROI (Mat& matSrc, Size size, Point2f ptLT, double dAngle, Mat& matROI)
{
	double dAngle_radian = dAngle * D2R;
	Point2f ptC ((matSrc.cols - 1) / 2.0f, (matSrc.rows - 1) / 2.0f);
	Point2f ptLT_rotate = ptRotatePt2f (ptLT, ptC, dAngle_radian);
	Size sizePadding (size.width + 6, size.height + 6);


	Mat rMat = getRotationMatrix2D (ptC, dAngle, 1);
	rMat.at<double> (0, 2) -= ptLT_rotate.x - 3;
	rMat.at<double> (1, 2) -= ptLT_rotate.y - 3;
	//平移旋轉矩陣(0, 2) (1, 2)的減，為旋轉後的圖形偏移，-= ptLT_rotate.x - 3 代表旋轉後的圖形往-X方向移動ptLT_rotate.x - 3
	//Debug
	
	//Debug
	warpAffine (matSrc, matROI, rMat, sizePadding);
}
bool OpencvNCCMatch::SubPixEsimation (vector<s_MatchParameter>* vec, double* dNewX, double* dNewY, double* dNewAngle, double dAngleStep, int iMaxScoreIndex)
{
	//Az=S, (A.T)Az=(A.T)s, z = ((A.T)A).inv (A.T)s

	Mat matA (27, 10, CV_64F);
	Mat matZ (10, 1, CV_64F);
	Mat matS (27, 1, CV_64F);

	double dX_maxScore = (*vec)[iMaxScoreIndex].pt.x;
	double dY_maxScore = (*vec)[iMaxScoreIndex].pt.y;
	double dTheata_maxScore = (*vec)[iMaxScoreIndex].dMatchAngle;
	int iRow = 0;
	/*for (int x = -1; x <= 1; x++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int theta = 0; theta <= 2; theta++)
			{*/
	for (int theta = 0; theta <= 2; theta++)
	{
		for (int y = -1; y <= 1; y++)
		{
			for (int x = -1; x <= 1; x++)
			{
				//xx yy tt xy xt yt x y t 1
				//0  1  2  3  4  5  6 7 8 9
				double dX = dX_maxScore + x;
				double dY = dY_maxScore + y;
				//double dT = (*vec)[theta].dMatchAngle + (theta - 1) * dAngleStep;
				double dT = (dTheata_maxScore + (theta - 1) * dAngleStep) * D2R;
				matA.at<double> (iRow, 0) = dX * dX;
				matA.at<double> (iRow, 1) = dY * dY;
				matA.at<double> (iRow, 2) = dT * dT;
				matA.at<double> (iRow, 3) = dX * dY;
				matA.at<double> (iRow, 4) = dX * dT;
				matA.at<double> (iRow, 5) = dY * dT;
				matA.at<double> (iRow, 6) = dX;
				matA.at<double> (iRow, 7) = dY;
				matA.at<double> (iRow, 8) = dT;
				matA.at<double> (iRow, 9) = 1.0;
				matS.at<double> (iRow, 0) = (*vec)[iMaxScoreIndex + (theta - 1)].vecResult[x + 1][y + 1];
				iRow++;
#ifdef _DEBUG
				/*string str = format ("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f", dValueA[0], dValueA[1], dValueA[2], dValueA[3], dValueA[4], dValueA[5], dValueA[6], dValueA[7], dValueA[8], dValueA[9]);
				fileA <<  str << endl;
				str = format ("%.6f", dValueS[iRow]);
				fileS << str << endl;*/
#endif
			}
		}
	}
	//求解Z矩陣，得到k0~k9
	//[ x* ] = [ 2k0 k3 k4 ]-1 [ -k6 ]
	//| y* | = | k3 2k1 k5 |   | -k7 |
	//[ t* ] = [ k4 k5 2k2 ]   [ -k8 ]
	
	//solve (matA, matS, matZ, DECOMP_SVD);
	matZ = (matA.t () * matA).inv () * matA.t ()* matS;
	Mat matZ_t;
	transpose (matZ, matZ_t);
	double* dZ = matZ_t.ptr<double> (0);
	Mat matK1 = (Mat_<double> (3, 3) << 
		(2 * dZ[0]), dZ[3], dZ[4], 
		dZ[3], (2 * dZ[1]), dZ[5], 
		dZ[4], dZ[5], (2 * dZ[2]));
	Mat matK2 = (Mat_<double> (3, 1) << -dZ[6], -dZ[7], -dZ[8]);
	Mat matDelta = matK1.inv () * matK2;

	*dNewX = matDelta.at<double> (0, 0);
	*dNewY = matDelta.at<double> (1, 0);
	*dNewAngle = matDelta.at<double> (2, 0) * R2D;
	return true;
}
void OpencvNCCMatch::FilterWithScore (vector<s_MatchParameter>* vec, double dScore)
{
	sort (vec->begin (), vec->end (), compareScoreBig2Small);
	int iSize = vec->size (), iIndexDelete = iSize + 1;
	for (int i = 0; i < iSize; i++)
	{
		if ((*vec)[i].dMatchScore < dScore)
		{
			iIndexDelete = i;
			break;
		}
	}
	if (iIndexDelete == iSize + 1)//沒有任何元素小於dScore
		return;
	vec->erase (vec->begin () + iIndexDelete, vec->end ());
}
void OpencvNCCMatch::FilterWithRotatedRect (vector<s_MatchParameter>* vec, int iMethod, double dMaxOverLap)
{
	int iMatchSize = (int)vec->size ();
	RotatedRect rect1, rect2;
	for (int i = 0; i < iMatchSize - 1; i++)
	{
		if (vec->at (i).bDelete)
			continue;
		for (int j = i + 1; j < iMatchSize; j++)
		{
			if (vec->at (j).bDelete)
				continue;
			rect1 = vec->at (i).rectR;
			rect2 = vec->at (j).rectR;
			vector<Point2f> vecInterSec;
			int iInterSecType = rotatedRectangleIntersection (rect1, rect2, vecInterSec);
			if (iInterSecType == INTERSECT_NONE)//無交集
				continue;
			else if (iInterSecType == INTERSECT_FULL) //一個矩形包覆另一個
			{
				int iDeleteIndex;
				if (iMethod == 0)
					iDeleteIndex = (vec->at (i).dMatchScore <= vec->at (j).dMatchScore) ? j : i;
				else
					iDeleteIndex = (vec->at (i).dMatchScore >= vec->at (j).dMatchScore) ? j : i;
				vec->at (iDeleteIndex).bDelete = true;
			}
			else//交點 > 0
			{
				if (vecInterSec.size () < 3)//一個或兩個交點
					continue;
				else
				{
					int iDeleteIndex;
					//求面積與交疊比例
					SortPtWithCenter (vecInterSec);
					double dArea = contourArea (vecInterSec);
					double dRatio = dArea / rect1.size.area ();
					//若大於最大交疊比例，選分數高的
					if (dRatio > dMaxOverLap)
					{
						if (iMethod == 0)
							iDeleteIndex = (vec->at (i).dMatchScore <= vec->at (j).dMatchScore) ? j : i;
						else
							iDeleteIndex = (vec->at (i).dMatchScore >= vec->at (j).dMatchScore) ? j : i;
						vec->at (iDeleteIndex).bDelete = true;
					}
				}
			}
		}
	}
	vector<s_MatchParameter>::iterator it;
	for (it = vec->begin (); it != vec->end ();)
	{
		if ((*it).bDelete)
			it = vec->erase (it);
		else
			++it;
	}
}
void OpencvNCCMatch::SortPtWithCenter (vector<Point2f>& vecSort)
{
	int iSize = (int)vecSort.size ();
	Point2f ptCenter;
	for (int i = 0; i < iSize; i++)
		ptCenter += vecSort[i];
	ptCenter /= iSize;

	Point2f vecX (1, 0);

	vector<pair<Point2f, double>> vecPtAngle (iSize);
	for (int i = 0; i < iSize; i++)
	{
		vecPtAngle[i].first = vecSort[i];//pt
		Point2f vec1 (vecSort[i].x - ptCenter.x, vecSort[i].y - ptCenter.y);
		float fNormVec1 = vec1.x * vec1.x + vec1.y * vec1.y;
		float fDot = vec1.x;

		if (vec1.y < 0)//若點在中心的上方
		{
			vecPtAngle[i].second = acos (fDot / fNormVec1) * R2D;
		}
		else if (vec1.y > 0)//下方
		{
			vecPtAngle[i].second = 360 - acos (fDot / fNormVec1) * R2D;
		}
		else//點與中心在相同Y
		{
			if (vec1.x - ptCenter.x > 0)
				vecPtAngle[i].second = 0;
			else
				vecPtAngle[i].second = 180;
		}

	}
	sort (vecPtAngle.begin (), vecPtAngle.end (), comparePtWithAngle);
	for (int i = 0; i < iSize; i++)
		vecSort[i] = vecPtAngle[i].first;
}
void OpencvNCCMatch::DrawDashLine (Mat& matDraw, Point ptStart, Point ptEnd, Scalar color1, Scalar color2)
{
	LineIterator itLine (matDraw, ptStart, ptEnd, 8, 0);
	int iCount = itLine.count;
	bool bOdd = false;
	for (int i = 0; i < iCount; i+=1, itLine++)
	{
		if (i % 3 == 0)
		{
			//白色BGR
			(*itLine)[0] = (uchar)color2.val[0];
			(*itLine)[1] = (uchar)color2.val[1];
			(*itLine)[2] = (uchar)color2.val[2];
		}
		else
		{
			//紅色BGR
			(*itLine)[0] = (uchar)color1.val[0];
			(*itLine)[1] = (uchar)color1.val[1];
			(*itLine)[2] = (uchar)color1.val[2];
		}

	}
}
void OpencvNCCMatch::DrawMarkCross (Mat& matDraw, int iX, int iY, int iLength, Scalar color, int iThickness)
{
	if (matDraw.empty ())
		return;
	Point ptC (iX, iY);
	line (matDraw, ptC - Point (iLength, 0), ptC + Point (iLength, 0), color, iThickness);
	line (matDraw, ptC - Point (0, iLength), ptC + Point (0, iLength), color, iThickness);
}
double OpencvNCCMatch::sumSquare(const cv::Scalar& s)
{
	return s[0]*s[0] + s[1]*s[1] + s[2]*s[2] + s[3]*s[3];
}