/**
 * OCR C API 示例
 *
 * 编译方法:
 *   gcc -o example example.c -L ../target/release -locr_capi -I../include
 *
 * 运行前确保已编译库:
 *   cargo build --release
 */

#include <stdio.h>
#include <stdlib.h>
#include "ocr_capi.h"

void print_error(void)
{
    char *err = ocr_get_last_error();
    if (err)
    {
        fprintf(stderr, "Error: %s\n", err);
        ocr_free_string(err);
    }
}

/**
 * 示例 1: 使用便捷 API (推荐)
 */
void example_simple_api(const char *det_path, const char *rec_path,
                        const char *charset_path, const char *image_path)
{
    printf("\n=== 示例 1: 便捷 API ===\n");

    // 创建引擎 (只需创建一次，可重复使用)
    OcrEngineHandle *engine = ocr_engine_create(
        det_path, rec_path, charset_path, NULL);

    if (!engine)
    {
        print_error();
        return;
    }

    // 识别图片
    OcrResultList result = ocr_engine_recognize_file(engine, image_path);

    printf("识别结果数量: %zu\n", result.count);
    for (size_t i = 0; i < result.count; i++)
    {
        printf("[%zu] 文本: %s\n", i + 1, result.items[i].text);
        printf("    置信度: %.2f%%\n", result.items[i].confidence * 100);
        printf("    位置: (%d, %d, %u, %u)\n",
               result.items[i].bbox.x,
               result.items[i].bbox.y,
               result.items[i].bbox.width,
               result.items[i].bbox.height);
    }

    // 释放结果
    ocr_result_list_free(&result);

    // 销毁引擎
    ocr_engine_destroy(engine);
}

/**
 * 示例 2: 使用配置
 */
void example_with_config(const char *det_path, const char *rec_path,
                         const char *charset_path, const char *image_path)
{
    printf("\n=== 示例 2: 带配置的 API ===\n");

    // 创建 GPU 配置
    OcrConfig config = ocr_config_gpu();
    config.thread_count = 8;
    config.min_result_confidence = 0.6;

    OcrEngineHandle *engine = ocr_engine_create(
        det_path, rec_path, charset_path, &config);

    if (!engine)
    {
        printf("GPU 不可用，回退到 CPU\n");
        config = ocr_config_default();
        engine = ocr_engine_create(det_path, rec_path, charset_path, &config);
    }

    if (!engine)
    {
        print_error();
        return;
    }

    OcrResultList result = ocr_engine_recognize_file(engine, image_path);

    printf("识别结果数量: %zu\n", result.count);
    for (size_t i = 0; i < result.count; i++)
    {
        printf("  %s (%.2f%%)\n",
               result.items[i].text,
               result.items[i].confidence * 100);
    }

    ocr_result_list_free(&result);
    ocr_engine_destroy(engine);
}

/**
 * 示例 3: 底层 API - 单独使用检测模型
 */
void example_detection_only(const char *det_path, const char *image_path)
{
    printf("\n=== 示例 3: 检测 API ===\n");

    DetModelHandle *det = ocr_det_model_create(det_path, NULL);
    if (!det)
    {
        print_error();
        return;
    }

    // 这里需要自己读取图片为 RGB 数据
    // 假设我们有 rgb_data, width, height
    printf("检测模型已加载，请使用实际的 RGB 数据调用 ocr_det_model_detect\n");

    ocr_det_model_destroy(det);
}

/**
 * 示例 4: 底层 API - 单独使用识别模型
 */
void example_recognition_only(const char *rec_path, const char *charset_path)
{
    printf("\n=== 示例 4: 识别 API ===\n");

    RecModelHandle *rec = ocr_rec_model_create(rec_path, charset_path, NULL);
    if (!rec)
    {
        print_error();
        return;
    }

    // 这里需要自己提供裁剪后的文本行 RGB 数据
    printf("识别模型已加载，请使用实际的裁剪文本行 RGB 数据调用 ocr_rec_model_recognize\n");

    ocr_rec_model_destroy(rec);
}

/**
 * 示例 5: 批量处理多张图片
 */
void example_batch_processing(const char *det_path, const char *rec_path,
                              const char *charset_path,
                              const char **image_paths, int count)
{
    printf("\n=== 示例 5: 批量处理 ===\n");

    // 创建一次引擎
    OcrEngineHandle *engine = ocr_engine_create(
        det_path, rec_path, charset_path, NULL);

    if (!engine)
    {
        print_error();
        return;
    }

    // 复用引擎处理多张图片
    for (int i = 0; i < count; i++)
    {
        printf("\n处理图片 %d: %s\n", i + 1, image_paths[i]);

        OcrResultList result = ocr_engine_recognize_file(engine, image_paths[i]);

        for (size_t j = 0; j < result.count; j++)
        {
            printf("  %s\n", result.items[j].text);
        }

        ocr_result_list_free(&result);
    }

    // 最后销毁引擎
    ocr_engine_destroy(engine);
}

/**
 * 示例 6: 方向分类 (单独使用方向模型)
 */
void example_orientation_only(const char *ori_path, const char *image_path)
{
    printf("\n=== 示例 6: 方向分类 ===\n");

    OriModelHandle *ori = ocr_ori_model_create(ori_path, NULL);
    if (!ori)
    {
        print_error();
        return;
    }

    OriResult result = ocr_ori_model_classify_file(ori, image_path);

    if (result.angle >= 0)
    {
        printf("图片方向: %d°\n", result.angle);
        printf("置信度: %.2f%%\n", result.confidence * 100);
        printf("类别索引: %zu\n", result.class_idx);
    }
    else
    {
        printf("方向分类失败\n");
        print_error();
    }

    ocr_ori_model_destroy(ori);
}

/**
 * 示例 7: 带旋转矫正的 OCR (推荐用于可能旋转的图片)
 */
void example_with_orientation(const char *det_path, const char *rec_path,
                              const char *charset_path, const char *ori_path,
                              const char *image_path)
{
    printf("\n=== 示例 7: 带旋转矫正的 OCR ===\n");

    // 创建带方向矫正的引擎
    OcrEngineHandle *engine = ocr_engine_create_with_ori(
        det_path, rec_path, charset_path, ori_path, NULL);

    if (!engine)
    {
        print_error();
        return;
    }

    // 即使图片是旋转的，也能正确识别
    OcrResultList result = ocr_engine_recognize_file(engine, image_path);

    printf("识别结果数量: %zu\n", result.count);
    for (size_t i = 0; i < result.count; i++)
    {
        printf("[%zu] 文本: %s\n", i + 1, result.items[i].text);
        printf("    置信度: %.2f%%\n", result.items[i].confidence * 100);
        printf("    位置: (%d, %d, %u, %u)\n",
               result.items[i].bbox.x,
               result.items[i].bbox.y,
               result.items[i].bbox.width,
               result.items[i].bbox.height);
    }

    ocr_result_list_free(&result);
    ocr_engine_destroy(engine);
}

int main(int argc, char *argv[])
{
    printf("OCR C API 示例\n");
    printf("版本: %s\n", ocr_version());

    if (argc < 5)
    {
        printf("\n用法: %s <det_model> <rec_model> <charset> <image> [ori_model]\n", argv[0]);
        printf("\n示例:\n");
        printf("  %s models/det.mnn models/rec.mnn models/keys.txt test.jpg\n", argv[0]);
        printf("  %s models/det.mnn models/rec.mnn models/keys.txt test.jpg models/doc_ori.mnn\n", argv[0]);
        return 1;
    }

    const char *det_path = argv[1];
    const char *rec_path = argv[2];
    const char *charset_path = argv[3];
    const char *image_path = argv[4];

    // 运行基础示例
    example_simple_api(det_path, rec_path, charset_path, image_path);
    example_with_config(det_path, rec_path, charset_path, image_path);
    example_detection_only(det_path, image_path);
    example_recognition_only(rec_path, charset_path);

    // 方向矫正示例 (需要第 5 个参数: 方向模型路径)
    if (argc >= 6)
    {
        const char *ori_path = argv[5];
        example_orientation_only(ori_path, image_path);
        example_with_orientation(det_path, rec_path, charset_path, ori_path, image_path);
    }

    return 0;
}
