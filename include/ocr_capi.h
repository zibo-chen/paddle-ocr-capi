/**
 * @file ocr_capi.h
 * @brief OCR C API - 基于 ocr-rs 的 C 语言接口
 *
 * 提供三层 API:
 * - 底层 API: DetModel 和 RecModel 的单独推理
 * - 普通 API: 接收 RGB/RGBA 数据，返回完整 OCR 结果
 * - 便捷 API: 接收图片路径，返回完整 OCR 结果
 */

#ifndef OCR_CAPI_H
#define OCR_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /* ============================================================================
     * 错误处理
     * ============================================================================ */

    /**
     * 错误码
     */
    typedef enum
    {
        OCR_SUCCESS = 0,           /**< 成功 */
        OCR_ERR_NULL_POINTER = 1,  /**< 空指针错误 */
        OCR_ERR_INVALID_PARAM = 2, /**< 无效参数 */
        OCR_ERR_MODEL_LOAD = 3,    /**< 模型加载失败 */
        OCR_ERR_IMAGE_LOAD = 4,    /**< 图片加载失败 */
        OCR_ERR_INFERENCE = 5,     /**< 推理失败 */
        OCR_ERR_ALLOC = 6,         /**< 内存分配失败 */
        OCR_ERR_UTF8 = 7           /**< UTF-8 编码错误 */
    } OcrErrorCode;

    /**
     * 获取最后一次错误信息
     * @return 错误信息字符串，需要调用 ocr_free_string 释放，无错误时返回 NULL
     */
    char *ocr_get_last_error(void);

    /* ============================================================================
     * 配置结构
     * ============================================================================ */

    /**
     * 推理后端类型
     */
    typedef enum
    {
        OCR_BACKEND_CPU = 0,    /**< CPU 后端 */
        OCR_BACKEND_METAL = 1,  /**< Metal 后端 (macOS/iOS) */
        OCR_BACKEND_OPENCL = 2, /**< OpenCL 后端 */
        OCR_BACKEND_VULKAN = 3  /**< Vulkan 后端 */
    } OcrBackend;

    /**
     * 精度模式
     */
    typedef enum
    {
        OCR_PRECISION_NORMAL = 0, /**< 正常精度 */
        OCR_PRECISION_LOW = 1,    /**< 低精度 (更快) */
        OCR_PRECISION_HIGH = 2    /**< 高精度 */
    } OcrPrecision;

    /**
     * 引擎配置
     */
    typedef struct
    {
        OcrBackend backend;            /**< 推理后端 */
        int thread_count;              /**< 线程数 */
        OcrPrecision precision;        /**< 精度模式 */
        unsigned int det_max_side_len; /**< 检测时图像最大边长 */
        float det_box_threshold;       /**< 检测框阈值 */
        float det_score_threshold;     /**< 检测分数阈值 */
        float rec_min_score;           /**< 识别最小置信度 */
        float min_result_confidence;   /**< 最终结果最小置信度 */
        int enable_parallel;           /**< 是否启用并行处理 (0=否, 1=是) */
    } OcrConfig;

    /**
     * 创建默认配置
     * @return 默认配置结构
     */
    OcrConfig ocr_config_default(void);

    /**
     * 创建快速模式配置
     * @return 快速模式配置
     */
    OcrConfig ocr_config_fast(void);

    /**
     * 创建 GPU 模式配置
     * @return GPU 模式配置 (macOS 使用 Metal，其他平台使用 OpenCL)
     */
    OcrConfig ocr_config_gpu(void);

    /* ============================================================================
     * 结果结构
     * ============================================================================ */

    /**
     * 文本框位置
     */
    typedef struct
    {
        int x;               /**< 左上角 x 坐标 */
        int y;               /**< 左上角 y 坐标 */
        unsigned int width;  /**< 宽度 */
        unsigned int height; /**< 高度 */
    } OcrBox;

    /**
     * 单个 OCR 结果项
     */
    typedef struct
    {
        char *text;       /**< 识别的文本 (UTF-8 编码) */
        float confidence; /**< 置信度 (0.0 - 1.0) */
        OcrBox bbox;      /**< 文本框位置 */
    } OcrResultItem;

    /**
     * OCR 结果列表
     */
    typedef struct
    {
        OcrResultItem *items; /**< 结果数组 */
        size_t count;         /**< 结果数量 */
    } OcrResultList;

    /**
     * 单个识别结果
     */
    typedef struct
    {
        char *text;       /**< 识别的文本 */
        float confidence; /**< 置信度 */
    } RecResult;

    /**
     * 检测结果列表
     */
    typedef struct
    {
        OcrBox *boxes; /**< 检测框数组 */
        float *scores; /**< 置信度数组 */
        size_t count;  /**< 数量 */
    } DetResultList;

    /* ============================================================================
     * 底层 API - 检测模型
     * ============================================================================ */

    /** 检测模型句柄 (不透明类型) */
    typedef struct DetModelHandle DetModelHandle;

    /**
     * 创建检测模型
     *
     * @param model_path 模型文件路径 (.mnn 格式)
     * @param config 配置指针，传 NULL 使用默认配置
     * @return 成功返回模型句柄，失败返回 NULL
     *
     * @note 模型句柄使用完毕后需调用 ocr_det_model_destroy 销毁
     *
     * @code
     * DetModelHandle* det = ocr_det_model_create("det.mnn", NULL);
     * if (det == NULL) {
     *     char* err = ocr_get_last_error();
     *     printf("Error: %s\n", err);
     *     ocr_free_string(err);
     * }
     * @endcode
     */
    DetModelHandle *ocr_det_model_create(const char *model_path, const OcrConfig *config);

    /**
     * 销毁检测模型
     * @param handle 模型句柄
     */
    void ocr_det_model_destroy(DetModelHandle *handle);

    /**
     * 使用检测模型检测文本区域
     *
     * @param handle 模型句柄
     * @param rgb_data RGB 图像数据 (格式: RGBRGBRGB...)
     * @param width 图像宽度
     * @param height 图像高度
     * @return 检测结果列表，count=0 表示无结果或失败
     *
     * @note 结果使用完毕后需调用 ocr_det_result_free 释放
     */
    DetResultList ocr_det_model_detect(DetModelHandle *handle,
                                       const unsigned char *rgb_data,
                                       unsigned int width,
                                       unsigned int height);

    /**
     * 释放检测结果
     * @param result 检测结果指针
     */
    void ocr_det_result_free(DetResultList *result);

    /* ============================================================================
     * 底层 API - 识别模型
     * ============================================================================ */

    /** 识别模型句柄 (不透明类型) */
    typedef struct RecModelHandle RecModelHandle;

    /**
     * 创建识别模型
     *
     * @param model_path 模型文件路径 (.mnn 格式)
     * @param charset_path 字符集文件路径
     * @param config 配置指针，传 NULL 使用默认配置
     * @return 成功返回模型句柄，失败返回 NULL
     *
     * @note 模型句柄使用完毕后需调用 ocr_rec_model_destroy 销毁
     */
    RecModelHandle *ocr_rec_model_create(const char *model_path,
                                         const char *charset_path,
                                         const OcrConfig *config);

    /**
     * 销毁识别模型
     * @param handle 模型句柄
     */
    void ocr_rec_model_destroy(RecModelHandle *handle);

    /**
     * 使用识别模型识别文本
     *
     * @param handle 模型句柄
     * @param rgb_data RGB 图像数据 (裁剪后的文本行图像)
     * @param width 图像宽度
     * @param height 图像高度
     * @return 识别结果，text=NULL 表示失败
     *
     * @note 结果使用完毕后需调用 ocr_rec_result_free 释放
     */
    RecResult ocr_rec_model_recognize(RecModelHandle *handle,
                                      const unsigned char *rgb_data,
                                      unsigned int width,
                                      unsigned int height);

    /**
     * 释放识别结果
     * @param result 识别结果指针
     */
    void ocr_rec_result_free(RecResult *result);

    /* ============================================================================
     * 普通 API - OCR 引擎 (推荐使用)
     * ============================================================================ */

    /** OCR 引擎句柄 (不透明类型) */
    typedef struct OcrEngineHandle OcrEngineHandle;

    /**
     * 创建 OCR 引擎
     *
     * @param det_model_path 检测模型文件路径
     * @param rec_model_path 识别模型文件路径
     * @param charset_path 字符集文件路径
     * @param config 配置指针，传 NULL 使用默认配置
     * @return 成功返回引擎句柄，失败返回 NULL
     *
     * @note 引擎句柄使用完毕后需调用 ocr_engine_destroy 销毁
     *
     * @code
     * // 创建引擎
     * OcrEngineHandle* engine = ocr_engine_create(
     *     "det.mnn", "rec.mnn", "keys.txt", NULL);
     *
     * // 多次识别复用引擎
     * OcrResultList result1 = ocr_engine_recognize_file(engine, "img1.jpg");
     * OcrResultList result2 = ocr_engine_recognize_file(engine, "img2.jpg");
     *
     * // 处理结果...
     *
     * // 释放资源
     * ocr_result_list_free(&result1);
     * ocr_result_list_free(&result2);
     * ocr_engine_destroy(engine);
     * @endcode
     */
    OcrEngineHandle *ocr_engine_create(const char *det_model_path,
                                       const char *rec_model_path,
                                       const char *charset_path,
                                       const OcrConfig *config);

    /**
     * 销毁 OCR 引擎
     * @param handle 引擎句柄
     */
    void ocr_engine_destroy(OcrEngineHandle *handle);

    /**
     * 使用 RGB 数据进行 OCR 识别
     *
     * @param handle 引擎句柄
     * @param rgb_data RGB 图像数据 (格式: RGBRGBRGB...)
     * @param width 图像宽度
     * @param height 图像高度
     * @return OCR 结果列表
     *
     * @note 结果使用完毕后需调用 ocr_result_list_free 释放
     */
    OcrResultList ocr_engine_recognize_rgb(OcrEngineHandle *handle,
                                           const unsigned char *rgb_data,
                                           unsigned int width,
                                           unsigned int height);

    /**
     * 使用 RGBA 数据进行 OCR 识别
     *
     * @param handle 引擎句柄
     * @param rgba_data RGBA 图像数据 (格式: RGBARGBA...)
     * @param width 图像宽度
     * @param height 图像高度
     * @return OCR 结果列表
     */
    OcrResultList ocr_engine_recognize_rgba(OcrEngineHandle *handle,
                                            const unsigned char *rgba_data,
                                            unsigned int width,
                                            unsigned int height);

    /* ============================================================================
     * 便捷 API - 直接使用图片路径
     * ============================================================================ */

    /**
     * 使用图片路径进行 OCR 识别
     *
     * @param handle 引擎句柄
     * @param image_path 图片文件路径 (支持 jpg, png, bmp 等格式)
     * @return OCR 结果列表
     *
     * @note 结果使用完毕后需调用 ocr_result_list_free 释放
     *
     * @code
     * OcrResultList result = ocr_engine_recognize_file(engine, "test.jpg");
     * for (size_t i = 0; i < result.count; i++) {
     *     printf("Text: %s (%.2f%%)\n",
     *            result.items[i].text,
     *            result.items[i].confidence * 100);
     *     printf("  Position: (%d, %d, %u, %u)\n",
     *            result.items[i].bbox.x,
     *            result.items[i].bbox.y,
     *            result.items[i].bbox.width,
     *            result.items[i].bbox.height);
     * }
     * ocr_result_list_free(&result);
     * @endcode
     */
    OcrResultList ocr_engine_recognize_file(OcrEngineHandle *handle,
                                            const char *image_path);

    /**
     * 一次性 OCR 识别 (创建临时引擎)
     *
     * @param det_model_path 检测模型路径
     * @param rec_model_path 识别模型路径
     * @param charset_path 字符集路径
     * @param image_path 图片路径
     * @param config 配置指针，传 NULL 使用默认配置
     * @return OCR 结果列表
     *
     * @warning 此函数每次调用都会创建和销毁引擎，性能较低。
     *          如需多次识别，请使用 ocr_engine_create 创建持久引擎。
     */
    OcrResultList ocr_recognize_file_once(const char *det_model_path,
                                          const char *rec_model_path,
                                          const char *charset_path,
                                          const char *image_path,
                                          const OcrConfig *config);

    /* ============================================================================
     * 内存管理
     * ============================================================================ */

    /**
     * 释放 OCR 结果列表
     * @param result 结果列表指针
     */
    void ocr_result_list_free(OcrResultList *result);

    /**
     * 释放字符串
     * @param s 字符串指针
     */
    void ocr_free_string(char *s);

    /* ============================================================================
     * 版本信息
     * ============================================================================ */

    /**
     * 获取库版本号
     * @return 版本字符串 (不需要释放)
     */
    const char *ocr_version(void);

#ifdef __cplusplus
}
#endif

#endif /* OCR_CAPI_H */
