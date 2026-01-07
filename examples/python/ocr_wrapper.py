"""
OCR C API Python Wrapper

使用 ctypes 封装 OCR C API，提供 Pythonic 的接口
"""

import ctypes
import os
import platform
from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BBox:
    """文本框位置"""
    x: int
    y: int
    width: int
    height: int


@dataclass
class OcrResult:
    """OCR 识别结果"""
    text: str
    confidence: float
    bbox: BBox


class OcrBackend:
    """推理后端"""
    CPU = 0
    METAL = 1
    OPENCL = 2
    VULKAN = 3


class OcrPrecision:
    """精度模式"""
    NORMAL = 0
    LOW = 1
    HIGH = 2


class _OcrConfig(ctypes.Structure):
    """引擎配置 (C 结构体)"""
    _fields_ = [
        ("backend", ctypes.c_int),
        ("thread_count", ctypes.c_int),
        ("precision", ctypes.c_int),
        ("det_max_side_len", ctypes.c_uint),
        ("det_box_threshold", ctypes.c_float),
        ("det_score_threshold", ctypes.c_float),
        ("rec_min_score", ctypes.c_float),
        ("min_result_confidence", ctypes.c_float),
        ("enable_parallel", ctypes.c_int),
    ]


class _OcrBox(ctypes.Structure):
    """文本框 (C 结构体)"""
    _fields_ = [
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("width", ctypes.c_uint),
        ("height", ctypes.c_uint),
    ]


class _OcrResultItem(ctypes.Structure):
    """OCR 结果项 (C 结构体)"""
    _fields_ = [
        ("text", ctypes.c_void_p),  # 使用 c_void_p 避免 ctypes 自动管理
        ("confidence", ctypes.c_float),
        ("bbox", _OcrBox),
    ]


class _OcrResultList(ctypes.Structure):
    """OCR 结果列表 (C 结构体)"""
    _fields_ = [
        ("items", ctypes.POINTER(_OcrResultItem)),
        ("count", ctypes.c_size_t),
    ]


class OcrConfig:
    """OCR 引擎配置"""
    
    def __init__(
        self,
        backend: int = OcrBackend.CPU,
        thread_count: int = 4,
        precision: int = OcrPrecision.NORMAL,
        det_max_side_len: int = 960,
        det_box_threshold: float = 0.5,
        det_score_threshold: float = 0.3,
        rec_min_score: float = 0.3,
        min_result_confidence: float = 0.5,
        enable_parallel: bool = True,
    ):
        self.backend = backend
        self.thread_count = thread_count
        self.precision = precision
        self.det_max_side_len = det_max_side_len
        self.det_box_threshold = det_box_threshold
        self.det_score_threshold = det_score_threshold
        self.rec_min_score = rec_min_score
        self.min_result_confidence = min_result_confidence
        self.enable_parallel = enable_parallel
    
    @classmethod
    def default(cls):
        """默认配置"""
        return cls()
    
    @classmethod
    def fast(cls):
        """快速模式"""
        return cls(
            precision=OcrPrecision.LOW,
            det_max_side_len=640,
        )
    
    @classmethod
    def gpu(cls):
        """GPU 模式"""
        system = platform.system()
        backend = OcrBackend.METAL if system == "Darwin" else OcrBackend.OPENCL
        return cls(backend=backend)
    
    def _to_c_struct(self) -> _OcrConfig:
        """转换为 C 结构体"""
        return _OcrConfig(
            backend=self.backend,
            thread_count=self.thread_count,
            precision=self.precision,
            det_max_side_len=self.det_max_side_len,
            det_box_threshold=self.det_box_threshold,
            det_score_threshold=self.det_score_threshold,
            rec_min_score=self.rec_min_score,
            min_result_confidence=self.min_result_confidence,
            enable_parallel=1 if self.enable_parallel else 0,
        )


class OcrEngine:
    """OCR 引擎"""
    
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        charset_path: str,
        config: Optional[OcrConfig] = None,
        lib_path: Optional[str] = None,
    ):
        """
        初始化 OCR 引擎
        
        Args:
            det_model_path: 检测模型路径
            rec_model_path: 识别模型路径
            charset_path: 字符集文件路径
            config: 配置选项，None 使用默认配置
            lib_path: 动态库路径，None 自动查找
        """
        # 加载动态库
        self._lib = self._load_library(lib_path)
        self._setup_functions()
        
        # 创建引擎
        config_ptr = None
        if config:
            c_config = config._to_c_struct()
            config_ptr = ctypes.byref(c_config)
        
        self._handle = self._lib.ocr_engine_create(
            det_model_path.encode('utf-8'),
            rec_model_path.encode('utf-8'),
            charset_path.encode('utf-8'),
            config_ptr,
        )
        
        if not self._handle:
            error = self._get_last_error()
            raise RuntimeError(f"Failed to create OCR engine: {error}")
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, '_handle') and hasattr(self, '_lib') and self._handle:
            self._lib.ocr_engine_destroy(self._handle)
            self._handle = None  # 防止 double free
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        if self._handle:
            self._lib.ocr_engine_destroy(self._handle)
            self._handle = None
    
    def recognize_file(self, image_path: str) -> List[OcrResult]:
        """
        识别图片文件
        
        Args:
            image_path: 图片文件路径
        
        Returns:
            OCR 结果列表
        """
        result_list = self._lib.ocr_engine_recognize_file(
            self._handle,
            image_path.encode('utf-8')
        )
        
        return self._convert_result_list(result_list)
    
    def recognize_rgb(
        self,
        rgb_data: bytes,
        width: int,
        height: int
    ) -> List[OcrResult]:
        """
        识别 RGB 图像数据
        
        Args:
            rgb_data: RGB 图像数据 (格式: RGBRGBRGB...)
            width: 图像宽度
            height: 图像高度
        
        Returns:
            OCR 结果列表
        """
        rgb_array = (ctypes.c_ubyte * len(rgb_data)).from_buffer_copy(rgb_data)
        
        result_list = self._lib.ocr_engine_recognize_rgb(
            self._handle,
            rgb_array,
            width,
            height,
        )
        
        return self._convert_result_list(result_list)
    
    def recognize_rgba(
        self,
        rgba_data: bytes,
        width: int,
        height: int
    ) -> List[OcrResult]:
        """
        识别 RGBA 图像数据
        
        Args:
            rgba_data: RGBA 图像数据 (格式: RGBARGBA...)
            width: 图像宽度
            height: 图像高度
        
        Returns:
            OCR 结果列表
        """
        rgba_array = (ctypes.c_ubyte * len(rgba_data)).from_buffer_copy(rgba_data)
        
        result_list = self._lib.ocr_engine_recognize_rgba(
            self._handle,
            rgba_array,
            width,
            height,
        )
        
        return self._convert_result_list(result_list)
    
    @staticmethod
    def get_version() -> str:
        """获取库版本"""
        lib = OcrEngine._load_library()
        lib.ocr_version.restype = ctypes.c_char_p
        version = lib.ocr_version()
        return version.decode('utf-8')
    
    def _get_last_error(self) -> str:
        """获取最后的错误信息"""
        error_ptr = self._lib.ocr_get_last_error()
        if error_ptr:
            error = ctypes.string_at(error_ptr).decode('utf-8')
            self._lib.ocr_free_string(error_ptr)
            return error
        return "Unknown error"
    
    def _convert_result_list(self, result_list: _OcrResultList) -> List[OcrResult]:
        """转换 C 结果列表为 Python 对象"""
        results = []
        
        # 检查是否有有效结果
        if result_list.count == 0 or not result_list.items:
            return results
        
        try:
            for i in range(result_list.count):
                item = result_list.items[i]
                # text 是 c_void_p，需要手动转换为字符串
                if item.text:
                    text = ctypes.string_at(item.text).decode('utf-8')
                else:
                    text = ""
                
                results.append(OcrResult(
                    text=text,
                    confidence=item.confidence,
                    bbox=BBox(
                        x=item.bbox.x,
                        y=item.bbox.y,
                        width=item.bbox.width,
                        height=item.bbox.height,
                    )
                ))
        finally:
            # 释放 C 结果
            self._lib.ocr_result_list_free(ctypes.byref(result_list))
        
        return results
    
    @staticmethod
    def _load_library(lib_path: Optional[str] = None) -> ctypes.CDLL:
        """加载动态库"""
        if lib_path:
            return ctypes.CDLL(lib_path)
        
        # 自动查找库文件
        system = platform.system()
        if system == "Darwin":
            lib_name = "libocr_capi.dylib"
        elif system == "Linux":
            lib_name = "libocr_capi.so"
        elif system == "Windows":
            lib_name = "ocr_capi.dll"
        else:
            raise RuntimeError(f"Unsupported platform: {system}")
        
        # 查找路径
        search_paths = [
            # 相对于当前文件的路径
            Path(__file__).parent.parent.parent / "target" / "release" / lib_name,
            Path(__file__).parent.parent.parent / "target" / "debug" / lib_name,
            # 系统路径
            Path("/usr/local/lib") / lib_name,
            Path("/usr/lib") / lib_name,
        ]
        
        for path in search_paths:
            if path.exists():
                return ctypes.CDLL(str(path))
        
        raise RuntimeError(
            f"Could not find {lib_name}. "
            "Please build the library first: cargo build --release"
        )
    
    def _setup_functions(self):
        """设置函数签名"""
        # ocr_engine_create
        self._lib.ocr_engine_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(_OcrConfig),
        ]
        self._lib.ocr_engine_create.restype = ctypes.c_void_p
        
        # ocr_engine_destroy
        self._lib.ocr_engine_destroy.argtypes = [ctypes.c_void_p]
        self._lib.ocr_engine_destroy.restype = None
        
        # ocr_engine_recognize_file
        self._lib.ocr_engine_recognize_file.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        self._lib.ocr_engine_recognize_file.restype = _OcrResultList
        
        # ocr_engine_recognize_rgb
        self._lib.ocr_engine_recognize_rgb.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self._lib.ocr_engine_recognize_rgb.restype = _OcrResultList
        
        # ocr_engine_recognize_rgba
        self._lib.ocr_engine_recognize_rgba.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self._lib.ocr_engine_recognize_rgba.restype = _OcrResultList
        
        # ocr_result_list_free
        self._lib.ocr_result_list_free.argtypes = [ctypes.POINTER(_OcrResultList)]
        self._lib.ocr_result_list_free.restype = None
        
        # ocr_get_last_error
        self._lib.ocr_get_last_error.argtypes = []
        self._lib.ocr_get_last_error.restype = ctypes.c_char_p
        
        # ocr_free_string
        self._lib.ocr_free_string.argtypes = [ctypes.c_char_p]
        self._lib.ocr_free_string.restype = None
        
        # ocr_version
        self._lib.ocr_version.argtypes = []
        self._lib.ocr_version.restype = ctypes.c_char_p
