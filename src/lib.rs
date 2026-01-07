//! OCR C API
//!
//! 提供 OCR 功能的 C 语言接口，支持三层 API：
//! - 底层 API: DetModel 和 RecModel 的单独推理
//! - 普通 API: 接收 RGB 数据，返回完整 OCR 结果
//! - 便捷 API: 接收图片路径，返回完整 OCR 结果

use image::{DynamicImage, RgbImage};
use libc::{c_char, c_float, c_int, c_uchar, c_uint, size_t};
use ocr_rs::{
    Backend, DetModel, DetOptions, InferenceConfig, OcrEngine, OcrEngineConfig, OcrResult_,
    PrecisionMode, RecModel, RecOptions,
};
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

// ============================================================================
// 错误处理
// ============================================================================

/// 错误码
#[repr(C)]
pub enum OcrErrorCode {
    /// 成功
    Success = 0,
    /// 空指针错误
    NullPointer = 1,
    /// 无效参数
    InvalidParameter = 2,
    /// 模型加载失败
    ModelLoadFailed = 3,
    /// 图片加载失败
    ImageLoadFailed = 4,
    /// 推理失败
    InferenceFailed = 5,
    /// 内存分配失败
    AllocationFailed = 6,
    /// UTF-8 编码错误
    Utf8Error = 7,
}

thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
}

fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(msg));
}

/// 获取最后一次错误信息
/// 返回的字符串需要调用 ocr_free_string 释放
#[no_mangle]
pub extern "C" fn ocr_get_last_error() -> *mut c_char {
    LAST_ERROR.with(|e| {
        if let Some(msg) = e.borrow().as_ref() {
            CString::new(msg.as_str())
                .map(|s| s.into_raw())
                .unwrap_or(ptr::null_mut())
        } else {
            ptr::null_mut()
        }
    })
}

// ============================================================================
// 配置结构
// ============================================================================

/// 推理后端类型
#[repr(C)]
#[derive(Clone, Copy)]
pub enum OcrBackend {
    CPU = 0,
    Metal = 1,
    OpenCL = 2,
    Vulkan = 3,
}

impl From<OcrBackend> for Backend {
    fn from(backend: OcrBackend) -> Self {
        match backend {
            OcrBackend::CPU => Backend::CPU,
            OcrBackend::Metal => Backend::Metal,
            OcrBackend::OpenCL => Backend::OpenCL,
            OcrBackend::Vulkan => Backend::Vulkan,
        }
    }
}

/// 精度模式
#[repr(C)]
#[derive(Clone, Copy)]
pub enum OcrPrecision {
    Normal = 0,
    Low = 1,
    High = 2,
}

impl From<OcrPrecision> for PrecisionMode {
    fn from(precision: OcrPrecision) -> Self {
        match precision {
            OcrPrecision::Normal => PrecisionMode::Normal,
            OcrPrecision::Low => PrecisionMode::Low,
            OcrPrecision::High => PrecisionMode::High,
        }
    }
}

/// 引擎配置
#[repr(C)]
pub struct OcrConfig {
    /// 推理后端
    pub backend: OcrBackend,
    /// 线程数
    pub thread_count: c_int,
    /// 精度模式
    pub precision: OcrPrecision,
    /// 检测时图像最大边长
    pub det_max_side_len: c_uint,
    /// 检测框阈值
    pub det_box_threshold: c_float,
    /// 检测分数阈值
    pub det_score_threshold: c_float,
    /// 识别最小置信度
    pub rec_min_score: c_float,
    /// 最终结果最小置信度
    pub min_result_confidence: c_float,
    /// 是否启用并行处理
    pub enable_parallel: c_int,
}

impl Default for OcrConfig {
    fn default() -> Self {
        Self {
            backend: OcrBackend::CPU,
            thread_count: 4,
            precision: OcrPrecision::Normal,
            det_max_side_len: 960,
            det_box_threshold: 0.5,
            det_score_threshold: 0.3,
            rec_min_score: 0.3,
            min_result_confidence: 0.5,
            enable_parallel: 1,
        }
    }
}

/// 创建默认配置
#[no_mangle]
pub extern "C" fn ocr_config_default() -> OcrConfig {
    OcrConfig::default()
}

/// 创建快速模式配置
#[no_mangle]
pub extern "C" fn ocr_config_fast() -> OcrConfig {
    OcrConfig {
        precision: OcrPrecision::Low,
        det_max_side_len: 640,
        ..Default::default()
    }
}

/// 创建 GPU 模式配置
#[no_mangle]
pub extern "C" fn ocr_config_gpu() -> OcrConfig {
    OcrConfig {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        backend: OcrBackend::Metal,
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        backend: OcrBackend::OpenCL,
        ..Default::default()
    }
}

// ============================================================================
// 结果结构
// ============================================================================

/// 文本框位置
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OcrBox {
    /// 左上角 x 坐标
    pub x: c_int,
    /// 左上角 y 坐标
    pub y: c_int,
    /// 宽度
    pub width: c_uint,
    /// 高度
    pub height: c_uint,
}

/// 单个 OCR 结果
#[repr(C)]
pub struct OcrResultItem {
    /// 识别的文本 (需要调用 ocr_free_string 释放)
    pub text: *mut c_char,
    /// 置信度 (0.0 - 1.0)
    pub confidence: c_float,
    /// 文本框位置
    pub bbox: OcrBox,
}

/// OCR 结果列表
#[repr(C)]
pub struct OcrResultList {
    /// 结果数组
    pub items: *mut OcrResultItem,
    /// 结果数量
    pub count: size_t,
}

/// 识别结果 (单个文本)
#[repr(C)]
pub struct RecResult {
    /// 识别的文本
    pub text: *mut c_char,
    /// 置信度
    pub confidence: c_float,
}

/// 检测结果列表
#[repr(C)]
pub struct DetResultList {
    /// 检测框数组
    pub boxes: *mut OcrBox,
    /// 置信度数组
    pub scores: *mut c_float,
    /// 数量
    pub count: size_t,
}

// ============================================================================
// 底层 API - 检测模型
// ============================================================================

/// 检测模型句柄
pub struct DetModelHandle {
    model: DetModel,
}

/// 创建检测模型
///
/// # 参数
/// - `model_path`: 模型文件路径
/// - `config`: 可选配置，传 NULL 使用默认配置
///
/// # 返回
/// 成功返回模型句柄，失败返回 NULL
#[no_mangle]
pub extern "C" fn ocr_det_model_create(
    model_path: *const c_char,
    config: *const OcrConfig,
) -> *mut DetModelHandle {
    if model_path.is_null() {
        set_last_error("model_path is null".to_string());
        return ptr::null_mut();
    }

    let path = match unsafe { CStr::from_ptr(model_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in model_path: {}", e));
            return ptr::null_mut();
        }
    };

    let (inference_config, det_options) = if config.is_null() {
        (None, DetOptions::default())
    } else {
        let cfg = unsafe { &*config };
        let inf_cfg = InferenceConfig {
            thread_count: cfg.thread_count,
            precision_mode: cfg.precision.into(),
            backend: cfg.backend.into(),
            ..Default::default()
        };
        let det_opts = DetOptions {
            max_side_len: cfg.det_max_side_len,
            box_threshold: cfg.det_box_threshold,
            score_threshold: cfg.det_score_threshold,
            ..Default::default()
        };
        (Some(inf_cfg), det_opts)
    };

    match DetModel::from_file(path, inference_config) {
        Ok(model) => {
            let model = model.with_options(det_options);
            Box::into_raw(Box::new(DetModelHandle { model }))
        }
        Err(e) => {
            set_last_error(format!("Failed to load det model: {}", e));
            ptr::null_mut()
        }
    }
}

/// 销毁检测模型
#[no_mangle]
pub extern "C" fn ocr_det_model_destroy(handle: *mut DetModelHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// 使用检测模型进行检测
///
/// # 参数
/// - `handle`: 模型句柄
/// - `rgb_data`: RGB 图像数据 (格式: RGBRGBRGB...)
/// - `width`: 图像宽度
/// - `height`: 图像高度
///
/// # 返回
/// 成功返回检测结果，失败返回 count=0 的结果
#[no_mangle]
pub extern "C" fn ocr_det_model_detect(
    handle: *mut DetModelHandle,
    rgb_data: *const c_uchar,
    width: c_uint,
    height: c_uint,
) -> DetResultList {
    let empty_result = DetResultList {
        boxes: ptr::null_mut(),
        scores: ptr::null_mut(),
        count: 0,
    };

    if handle.is_null() || rgb_data.is_null() {
        set_last_error("Null pointer".to_string());
        return empty_result;
    }

    let model = unsafe { &(*handle).model };
    let data_len = (width * height * 3) as usize;
    let data = unsafe { slice::from_raw_parts(rgb_data, data_len) };

    let image = match RgbImage::from_raw(width, height, data.to_vec()) {
        Some(img) => DynamicImage::ImageRgb8(img),
        None => {
            set_last_error("Failed to create image from RGB data".to_string());
            return empty_result;
        }
    };

    match model.detect(&image) {
        Ok(boxes) => {
            if boxes.is_empty() {
                return empty_result;
            }

            let count = boxes.len();
            let mut result_boxes = Vec::with_capacity(count);
            let mut result_scores = Vec::with_capacity(count);

            for text_box in boxes {
                result_boxes.push(OcrBox {
                    x: text_box.rect.left(),
                    y: text_box.rect.top(),
                    width: text_box.rect.width(),
                    height: text_box.rect.height(),
                });
                result_scores.push(text_box.score);
            }

            let boxes_ptr = result_boxes.as_mut_ptr();
            let scores_ptr = result_scores.as_mut_ptr();
            std::mem::forget(result_boxes);
            std::mem::forget(result_scores);

            DetResultList {
                boxes: boxes_ptr,
                scores: scores_ptr,
                count,
            }
        }
        Err(e) => {
            set_last_error(format!("Detection failed: {}", e));
            empty_result
        }
    }
}

/// 释放检测结果
#[no_mangle]
pub extern "C" fn ocr_det_result_free(result: *mut DetResultList) {
    if result.is_null() {
        return;
    }
    let result = unsafe { &mut *result };
    if !result.boxes.is_null() && result.count > 0 {
        unsafe {
            Vec::from_raw_parts(result.boxes, result.count, result.count);
            Vec::from_raw_parts(result.scores, result.count, result.count);
        }
    }
    result.boxes = ptr::null_mut();
    result.scores = ptr::null_mut();
    result.count = 0;
}

// ============================================================================
// 底层 API - 识别模型
// ============================================================================

/// 识别模型句柄
pub struct RecModelHandle {
    model: RecModel,
}

/// 创建识别模型
///
/// # 参数
/// - `model_path`: 模型文件路径
/// - `charset_path`: 字符集文件路径
/// - `config`: 可选配置，传 NULL 使用默认配置
///
/// # 返回
/// 成功返回模型句柄，失败返回 NULL
#[no_mangle]
pub extern "C" fn ocr_rec_model_create(
    model_path: *const c_char,
    charset_path: *const c_char,
    config: *const OcrConfig,
) -> *mut RecModelHandle {
    if model_path.is_null() || charset_path.is_null() {
        set_last_error("model_path or charset_path is null".to_string());
        return ptr::null_mut();
    }

    let model_path_str = match unsafe { CStr::from_ptr(model_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in model_path: {}", e));
            return ptr::null_mut();
        }
    };

    let charset_path_str = match unsafe { CStr::from_ptr(charset_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in charset_path: {}", e));
            return ptr::null_mut();
        }
    };

    let (inference_config, rec_options) = if config.is_null() {
        (None, RecOptions::default())
    } else {
        let cfg = unsafe { &*config };
        let inf_cfg = InferenceConfig {
            thread_count: cfg.thread_count,
            precision_mode: cfg.precision.into(),
            backend: cfg.backend.into(),
            ..Default::default()
        };
        let rec_opts = RecOptions {
            min_score: cfg.rec_min_score,
            ..Default::default()
        };
        (Some(inf_cfg), rec_opts)
    };

    match RecModel::from_file(model_path_str, charset_path_str, inference_config) {
        Ok(model) => {
            let model = model.with_options(rec_options);
            Box::into_raw(Box::new(RecModelHandle { model }))
        }
        Err(e) => {
            set_last_error(format!("Failed to load rec model: {}", e));
            ptr::null_mut()
        }
    }
}

/// 销毁识别模型
#[no_mangle]
pub extern "C" fn ocr_rec_model_destroy(handle: *mut RecModelHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// 使用识别模型进行识别
///
/// # 参数
/// - `handle`: 模型句柄
/// - `rgb_data`: RGB 图像数据 (格式: RGBRGBRGB...)
/// - `width`: 图像宽度
/// - `height`: 图像高度
///
/// # 返回
/// 识别结果，text 为 NULL 表示失败
#[no_mangle]
pub extern "C" fn ocr_rec_model_recognize(
    handle: *mut RecModelHandle,
    rgb_data: *const c_uchar,
    width: c_uint,
    height: c_uint,
) -> RecResult {
    let empty_result = RecResult {
        text: ptr::null_mut(),
        confidence: 0.0,
    };

    if handle.is_null() || rgb_data.is_null() {
        set_last_error("Null pointer".to_string());
        return empty_result;
    }

    let model = unsafe { &(*handle).model };
    let data_len = (width * height * 3) as usize;
    let data = unsafe { slice::from_raw_parts(rgb_data, data_len) };

    let image = match RgbImage::from_raw(width, height, data.to_vec()) {
        Some(img) => DynamicImage::ImageRgb8(img),
        None => {
            set_last_error("Failed to create image from RGB data".to_string());
            return empty_result;
        }
    };

    match model.recognize(&image) {
        Ok(result) => {
            let text = match CString::new(result.text) {
                Ok(s) => s.into_raw(),
                Err(e) => {
                    set_last_error(format!("Failed to convert text: {}", e));
                    return empty_result;
                }
            };
            RecResult {
                text,
                confidence: result.confidence,
            }
        }
        Err(e) => {
            set_last_error(format!("Recognition failed: {}", e));
            empty_result
        }
    }
}

/// 释放识别结果
#[no_mangle]
pub extern "C" fn ocr_rec_result_free(result: *mut RecResult) {
    if result.is_null() {
        return;
    }
    let result = unsafe { &mut *result };
    if !result.text.is_null() {
        unsafe {
            drop(CString::from_raw(result.text));
        }
        result.text = ptr::null_mut();
    }
}

// ============================================================================
// 普通 API - OCR 引擎 (接收 RGB 数据)
// ============================================================================

/// OCR 引擎句柄
pub struct OcrEngineHandle {
    engine: OcrEngine,
}

/// 创建 OCR 引擎
///
/// # 参数
/// - `det_model_path`: 检测模型文件路径
/// - `rec_model_path`: 识别模型文件路径
/// - `charset_path`: 字符集文件路径
/// - `config`: 可选配置，传 NULL 使用默认配置
///
/// # 返回
/// 成功返回引擎句柄，失败返回 NULL
#[no_mangle]
pub extern "C" fn ocr_engine_create(
    det_model_path: *const c_char,
    rec_model_path: *const c_char,
    charset_path: *const c_char,
    config: *const OcrConfig,
) -> *mut OcrEngineHandle {
    if det_model_path.is_null() || rec_model_path.is_null() || charset_path.is_null() {
        set_last_error("One or more paths are null".to_string());
        return ptr::null_mut();
    }

    let det_path = match unsafe { CStr::from_ptr(det_model_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in det_model_path: {}", e));
            return ptr::null_mut();
        }
    };

    let rec_path = match unsafe { CStr::from_ptr(rec_model_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in rec_model_path: {}", e));
            return ptr::null_mut();
        }
    };

    let charset = match unsafe { CStr::from_ptr(charset_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in charset_path: {}", e));
            return ptr::null_mut();
        }
    };

    let engine_config = if config.is_null() {
        None
    } else {
        let cfg = unsafe { &*config };
        Some(
            OcrEngineConfig::new()
                .with_backend(cfg.backend.into())
                .with_threads(cfg.thread_count)
                .with_precision(cfg.precision.into())
                .with_det_options(DetOptions {
                    max_side_len: cfg.det_max_side_len,
                    box_threshold: cfg.det_box_threshold,
                    score_threshold: cfg.det_score_threshold,
                    ..Default::default()
                })
                .with_rec_options(RecOptions {
                    min_score: cfg.rec_min_score,
                    ..Default::default()
                })
                .with_parallel(cfg.enable_parallel != 0)
                .with_min_result_confidence(cfg.min_result_confidence),
        )
    };

    match OcrEngine::new(det_path, rec_path, charset, engine_config) {
        Ok(engine) => Box::into_raw(Box::new(OcrEngineHandle { engine })),
        Err(e) => {
            set_last_error(format!("Failed to create OCR engine: {}", e));
            ptr::null_mut()
        }
    }
}

/// 销毁 OCR 引擎
#[no_mangle]
pub extern "C" fn ocr_engine_destroy(handle: *mut OcrEngineHandle) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

/// 使用 RGB 数据进行 OCR 识别
///
/// # 参数
/// - `handle`: 引擎句柄
/// - `rgb_data`: RGB 图像数据 (格式: RGBRGBRGB...)
/// - `width`: 图像宽度
/// - `height`: 图像高度
///
/// # 返回
/// OCR 结果列表，使用完毕后需要调用 ocr_result_list_free 释放
#[no_mangle]
pub extern "C" fn ocr_engine_recognize_rgb(
    handle: *mut OcrEngineHandle,
    rgb_data: *const c_uchar,
    width: c_uint,
    height: c_uint,
) -> OcrResultList {
    let empty_result = OcrResultList {
        items: ptr::null_mut(),
        count: 0,
    };

    if handle.is_null() || rgb_data.is_null() {
        set_last_error("Null pointer".to_string());
        return empty_result;
    }

    let engine = unsafe { &(*handle).engine };
    let data_len = (width * height * 3) as usize;
    let data = unsafe { slice::from_raw_parts(rgb_data, data_len) };

    let image = match RgbImage::from_raw(width, height, data.to_vec()) {
        Some(img) => DynamicImage::ImageRgb8(img),
        None => {
            set_last_error("Failed to create image from RGB data".to_string());
            return empty_result;
        }
    };

    convert_ocr_results(engine.recognize(&image))
}

/// 使用 RGBA 数据进行 OCR 识别
///
/// # 参数
/// - `handle`: 引擎句柄
/// - `rgba_data`: RGBA 图像数据 (格式: RGBARGBA...)
/// - `width`: 图像宽度
/// - `height`: 图像高度
///
/// # 返回
/// OCR 结果列表
#[no_mangle]
pub extern "C" fn ocr_engine_recognize_rgba(
    handle: *mut OcrEngineHandle,
    rgba_data: *const c_uchar,
    width: c_uint,
    height: c_uint,
) -> OcrResultList {
    let empty_result = OcrResultList {
        items: ptr::null_mut(),
        count: 0,
    };

    if handle.is_null() || rgba_data.is_null() {
        set_last_error("Null pointer".to_string());
        return empty_result;
    }

    let engine = unsafe { &(*handle).engine };
    let data_len = (width * height * 4) as usize;
    let data = unsafe { slice::from_raw_parts(rgba_data, data_len) };

    let image = match image::RgbaImage::from_raw(width, height, data.to_vec()) {
        Some(img) => DynamicImage::ImageRgba8(img),
        None => {
            set_last_error("Failed to create image from RGBA data".to_string());
            return empty_result;
        }
    };

    convert_ocr_results(engine.recognize(&image))
}

// ============================================================================
// 便捷 API - 直接使用图片路径
// ============================================================================

/// 使用图片路径进行 OCR 识别
///
/// # 参数
/// - `handle`: 引擎句柄
/// - `image_path`: 图片文件路径
///
/// # 返回
/// OCR 结果列表
#[no_mangle]
pub extern "C" fn ocr_engine_recognize_file(
    handle: *mut OcrEngineHandle,
    image_path: *const c_char,
) -> OcrResultList {
    let empty_result = OcrResultList {
        items: ptr::null_mut(),
        count: 0,
    };

    if handle.is_null() || image_path.is_null() {
        set_last_error("Null pointer".to_string());
        return empty_result;
    }

    let path = match unsafe { CStr::from_ptr(image_path) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("Invalid UTF-8 in image_path: {}", e));
            return empty_result;
        }
    };

    let image = match image::open(path) {
        Ok(img) => img,
        Err(e) => {
            set_last_error(format!("Failed to load image: {}", e));
            return empty_result;
        }
    };

    let engine = unsafe { &(*handle).engine };
    convert_ocr_results(engine.recognize(&image))
}

/// 一次性 OCR 识别 (创建临时引擎)
///
/// 注意：此函数会创建临时引擎，适合单次调用。
/// 如需多次调用，建议使用 ocr_engine_create 创建持久引擎。
///
/// # 参数
/// - `det_model_path`: 检测模型路径
/// - `rec_model_path`: 识别模型路径
/// - `charset_path`: 字符集路径
/// - `image_path`: 图片路径
/// - `config`: 可选配置
///
/// # 返回
/// OCR 结果列表
#[no_mangle]
pub extern "C" fn ocr_recognize_file_once(
    det_model_path: *const c_char,
    rec_model_path: *const c_char,
    charset_path: *const c_char,
    image_path: *const c_char,
    config: *const OcrConfig,
) -> OcrResultList {
    let empty_result = OcrResultList {
        items: ptr::null_mut(),
        count: 0,
    };

    // 创建临时引擎
    let engine_handle = ocr_engine_create(det_model_path, rec_model_path, charset_path, config);
    if engine_handle.is_null() {
        return empty_result;
    }

    // 执行识别
    let result = ocr_engine_recognize_file(engine_handle, image_path);

    // 销毁临时引擎
    ocr_engine_destroy(engine_handle);

    result
}

// ============================================================================
// 内存管理
// ============================================================================

/// 释放 OCR 结果列表
#[no_mangle]
pub extern "C" fn ocr_result_list_free(result: *mut OcrResultList) {
    if result.is_null() {
        return;
    }

    let result = unsafe { &mut *result };
    if !result.items.is_null() && result.count > 0 {
        // 重建 Vec 以便正确释放内存
        let mut items = unsafe { Vec::from_raw_parts(result.items, result.count, result.count) };

        // 遍历并释放每个 text 字符串
        for item in items.iter_mut() {
            if !item.text.is_null() {
                unsafe {
                    // 释放 CString
                    let _ = CString::from_raw(item.text);
                }
                // 将指针设为 null，避免悬空指针
                item.text = ptr::null_mut();
            }
        }

        // items Vec 会在这里自动 drop，释放数组内存
    }
    result.items = ptr::null_mut();
    result.count = 0;
}

/// 释放字符串
#[no_mangle]
pub extern "C" fn ocr_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}

// ============================================================================
// 内部辅助函数
// ============================================================================

fn convert_ocr_results(results: Result<Vec<OcrResult_>, ocr_rs::OcrError>) -> OcrResultList {
    let empty_result = OcrResultList {
        items: ptr::null_mut(),
        count: 0,
    };

    match results {
        Ok(results) => {
            if results.is_empty() {
                return empty_result;
            }

            let count = results.len();
            let mut items = Vec::with_capacity(count);

            for result in results {
                let text = match CString::new(result.text) {
                    Ok(s) => s.into_raw(),
                    Err(_) => continue,
                };

                items.push(OcrResultItem {
                    text,
                    confidence: result.confidence,
                    bbox: OcrBox {
                        x: result.bbox.rect.left(),
                        y: result.bbox.rect.top(),
                        width: result.bbox.rect.width(),
                        height: result.bbox.rect.height(),
                    },
                });
            }

            if items.is_empty() {
                return empty_result;
            }

            let count = items.len();
            let items_ptr = items.as_mut_ptr();
            std::mem::forget(items);

            OcrResultList {
                items: items_ptr,
                count,
            }
        }
        Err(e) => {
            set_last_error(format!("OCR recognition failed: {}", e));
            empty_result
        }
    }
}

// ============================================================================
// 版本信息
// ============================================================================

/// 获取库版本号
#[no_mangle]
pub extern "C" fn ocr_version() -> *const c_char {
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}
