//
//  OcrEngine.swift
//  ocr-rs-capi-example
//
//  Swift wrapper for ocr_capi C API
//

import Foundation
import UIKit

/// OCR 识别结果项
struct OcrTextItem: Identifiable {
    let id = UUID()
    let text: String
    let confidence: Float
    let bbox: CGRect
}

/// OCR 引擎封装
final class OcrEngine {
    private var engine: OpaquePointer?

    /// 使用 Bundle 中的模型文件初始化引擎
    init(detModel: String, recModel: String, charset: String, useGPU: Bool = true) throws {
        guard let detPath = Bundle.main.path(forResource: detModel, ofType: "mnn"),
              let recPath = Bundle.main.path(forResource: recModel, ofType: "mnn"),
              let charsetPath = Bundle.main.path(forResource: charset, ofType: "txt")
        else {
            throw OcrError.modelNotFound
        }

        var config = useGPU ? ocr_config_gpu() : ocr_config_default()
        engine = ocr_engine_create(detPath, recPath, charsetPath, &config)

        guard engine != nil else {
            let errMsg = Self.getLastError() ?? "Unknown error"
            throw OcrError.engineCreateFailed(errMsg)
        }
    }

    deinit {
        if let engine = engine {
            ocr_engine_destroy(engine)
        }
    }

    /// 识别图片文件（通过路径）
    func recognizeFile(at path: String) -> [OcrTextItem] {
        guard let engine = engine else { return [] }
        var result = ocr_engine_recognize_file(engine, path)
        defer { ocr_result_list_free(&result) }
        return Self.convertResults(result)
    }

    /// 识别 UIImage
    func recognize(image: UIImage) -> [OcrTextItem] {
        guard let engine = engine,
              let cgImage = image.cgImage else { return [] }

        let width = cgImage.width
        let height = cgImage.height

        // 将 UIImage 转换为 RGBA 数据
        guard let rgbaData = Self.imageToRGBA(cgImage: cgImage, width: width, height: height) else {
            return []
        }

        var result = ocr_engine_recognize_rgba(
            engine,
            rgbaData,
            UInt32(width),
            UInt32(height)
        )
        defer { ocr_result_list_free(&result) }
        return Self.convertResults(result)
    }

    // MARK: - Private Helpers

    private static func convertResults(_ result: OcrResultList) -> [OcrTextItem] {
        var items: [OcrTextItem] = []
        for i in 0 ..< result.count {
            let item = result.items[i]
            guard let cText = item.text else { continue }
            let text = String(cString: cText)
            let bbox = CGRect(
                x: CGFloat(item.bbox.x),
                y: CGFloat(item.bbox.y),
                width: CGFloat(item.bbox.width),
                height: CGFloat(item.bbox.height)
            )
            items.append(OcrTextItem(text: text, confidence: item.confidence, bbox: bbox))
        }
        return items
    }

    private static func imageToRGBA(cgImage: CGImage, width: Int, height: Int) -> UnsafeMutablePointer<UInt8>? {
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let totalBytes = bytesPerRow * height

        let rawData = UnsafeMutablePointer<UInt8>.allocate(capacity: totalBytes)

        guard let context = CGContext(
            data: rawData,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            rawData.deallocate()
            return nil
        }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        return rawData
    }

    private static func getLastError() -> String? {
        guard let errPtr = ocr_get_last_error() else { return nil }
        let msg = String(cString: errPtr)
        ocr_free_string(errPtr)
        return msg
    }
}

/// OCR 错误类型
enum OcrError: LocalizedError {
    case modelNotFound
    case engineCreateFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "模型文件未找到，请确保 .mnn 和 .txt 文件已添加到 Bundle"
        case .engineCreateFailed(let msg):
            return "OCR 引擎创建失败: \(msg)"
        }
    }
}
