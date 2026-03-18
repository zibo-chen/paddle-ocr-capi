//
//  ContentView.swift
//  ocr-rs-capi-example
//
//  Created by chenzibo on 2026/3/13.
//

import PhotosUI
import SwiftUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var ocrResults: [OcrTextItem] = []
    @State private var isProcessing = false
    @State private var errorMessage: String?
    @State private var engineReady = false
    @State private var engine: OcrEngine?
    @State private var processingTime: TimeInterval = 0
    @State private var enableOriCorrection = false
    @State private var orientationResult: OrientationResult?
    @State private var oriModelAvailable = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // 图片显示区域
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 300)
                        .cornerRadius(12)
                        .shadow(radius: 4)
                } else {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(.systemGray6))
                        .frame(height: 200)
                        .overlay {
                            VStack(spacing: 8) {
                                Image(systemName: "doc.text.viewfinder")
                                    .font(.system(size: 48))
                                    .foregroundStyle(.secondary)
                                Text("选择一张图片开始 OCR 识别")
                                    .foregroundStyle(.secondary)
                            }
                        }
                }

                // 选择图片按钮
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    Label("选择图片", systemImage: "photo.on.rectangle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isProcessing || !engineReady)

                // 旋转矫正开关
                if oriModelAvailable {
                    Toggle(isOn: $enableOriCorrection) {
                        Label("旋转矫正", systemImage: "rotate.right")
                    }
                    .onChange(of: enableOriCorrection) { _, _ in
                        Task { await reloadEngine() }
                    }
                }

                // 方向检测结果徽章
                if let ori = orientationResult, ori.isValid {
                    HStack(spacing: 6) {
                        Image(systemName: ori.angle == 0 ? "checkmark.circle.fill" : "arrow.uturn.right.circle.fill")
                            .foregroundStyle(ori.angle == 0 ? .green : .orange)
                        Text("检测到旋转: \(ori.angleDescription)")
                            .font(.caption)
                        Text(String(format: "(%.0f%%)", ori.confidence * 100))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Spacer()
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(Color(.systemGray6))
                    .cornerRadius(8)
                }

                // 状态信息
                if !engineReady {
                    HStack {
                        ProgressView()
                        Text("正在加载 OCR 引擎...")
                            .foregroundStyle(.secondary)
                    }
                }

                if isProcessing {
                    HStack {
                        ProgressView()
                        Text("正在识别...")
                            .foregroundStyle(.secondary)
                    }
                }

                if let error = errorMessage {
                    Text(error)
                        .foregroundStyle(.red)
                        .font(.caption)
                        .padding(.horizontal)
                }

                // OCR 结果列表
                if !ocrResults.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("识别结果 (\(ocrResults.count) 项)")
                                .font(.headline)
                            Spacer()
                            Text(String(format: "%.0fms", processingTime * 1000))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        List(ocrResults) { item in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(item.text)
                                    .font(.body)
                                HStack {
                                    Text(String(format: "置信度: %.1f%%", item.confidence * 100))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                    Text("(\(Int(item.bbox.origin.x)), \(Int(item.bbox.origin.y)))")
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                        }
                        .listStyle(.plain)
                    }
                }

                Spacer()

                // 版本信息
                if let version = ocrVersion() {
                    Text("ocr-capi v\(version)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding()
            .navigationTitle("OCR Demo")
            .onChange(of: selectedItem) { _, newItem in
                Task {
                    await loadAndRecognize(item: newItem)
                }
            }
            .task {
                await initEngine()
            }
        }
    }

    // MARK: - Private Methods

    private func initEngine() async {
        // 检查方向模型是否存在于 Bundle
        let hasOriModel = Bundle.main.path(forResource: "PP-LCNet_x1_0_doc_ori", ofType: "mnn") != nil
        oriModelAvailable = hasOriModel

        await reloadEngine()
    }

    private func reloadEngine() async {
        engineReady = false
        engine = nil
        errorMessage = nil

        do {
            let useOri = enableOriCorrection && oriModelAvailable
            let eng = try await Task.detached {
                if useOri {
                    return try OcrEngine(
                        detModel: "ch_PP-OCRv4_det_infer",
                        recModel: "ch_PP-OCRv4_rec_infer",
                        charset: "ppocr_keys_v4",
                        oriModel: "PP-LCNet_x1_0_doc_ori",
                        useGPU: true
                    )
                } else {
                    return try OcrEngine(
                        detModel: "ch_PP-OCRv4_det_infer",
                        recModel: "ch_PP-OCRv4_rec_infer",
                        charset: "ppocr_keys_v4",
                        useGPU: true
                    )
                }
            }.value
            engine = eng
            engineReady = true
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func loadAndRecognize(item: PhotosPickerItem?) async {
        guard let item = item else { return }
        errorMessage = nil
        ocrResults = []
        orientationResult = nil

        // 加载图片
        guard let data = try? await item.loadTransferable(type: Data.self),
              let image = UIImage(data: data)
        else {
            errorMessage = "无法加载图片"
            return
        }

        selectedImage = image
        isProcessing = true

        // 先进行方向检测（仅在未启用自动矫正时展示参考信息）
        if !enableOriCorrection, oriModelAvailable,
           let oriPath = Bundle.main.path(forResource: "PP-LCNet_x1_0_doc_ori", ofType: "mnn")
        {
            let result = await Task.detached {
                var config = ocr_config_default()
                let oriHandle = ocr_ori_model_create(oriPath, &config)
                defer { ocr_ori_model_destroy(oriHandle) }
                guard let oriHandle, let cgImage = image.cgImage else {
                    return OrientationResult(classIndex: 0, angle: -1, confidence: 0)
                }
                let w = cgImage.width
                let h = cgImage.height
                let bytesPerRow = 4 * w
                let rawData = UnsafeMutablePointer<UInt8>.allocate(capacity: bytesPerRow * h)
                defer { rawData.deallocate() }
                guard let ctx = CGContext(
                    data: rawData,
                    width: w, height: h,
                    bitsPerComponent: 8,
                    bytesPerRow: bytesPerRow,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                ) else {
                    return OrientationResult(classIndex: 0, angle: -1, confidence: 0)
                }
                ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: w, height: h))

                let rgbSize = w * h * 3
                let rgbData = UnsafeMutablePointer<UInt8>.allocate(capacity: rgbSize)
                defer { rgbData.deallocate() }
                for i in 0 ..< w * h {
                    rgbData[i * 3 + 0] = rawData[i * 4 + 0]
                    rgbData[i * 3 + 1] = rawData[i * 4 + 1]
                    rgbData[i * 3 + 2] = rawData[i * 4 + 2]
                }
                let r = ocr_ori_model_classify(oriHandle, rgbData, UInt32(w), UInt32(h))
                return OrientationResult(classIndex: r.class_idx, angle: Int(r.angle), confidence: r.confidence)
            }.value
            orientationResult = result
        }

        // 在后台线程执行 OCR
        let eng = engine
        let start = CFAbsoluteTimeGetCurrent()
        let results = await Task.detached {
            eng?.recognize(image: image) ?? []
        }.value
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        ocrResults = results
        processingTime = elapsed
        isProcessing = false

        if results.isEmpty {
            errorMessage = "未识别到文本"
        }
    }

    private func ocrVersion() -> String? {
        guard let ptr = ocr_version() else { return nil }
        return String(cString: ptr)
    }
}

#Preview {
    ContentView()
}


import PhotosUI
import SwiftUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var ocrResults: [OcrTextItem] = []
    @State private var isProcessing = false
    @State private var errorMessage: String?
    @State private var engineReady = false
    @State private var engine: OcrEngine?
    @State private var processingTime: TimeInterval = 0

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // 图片显示区域
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxHeight: 300)
                        .cornerRadius(12)
                        .shadow(radius: 4)
                } else {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color(.systemGray6))
                        .frame(height: 200)
                        .overlay {
                            VStack(spacing: 8) {
                                Image(systemName: "doc.text.viewfinder")
                                    .font(.system(size: 48))
                                    .foregroundStyle(.secondary)
                                Text("选择一张图片开始 OCR 识别")
                                    .foregroundStyle(.secondary)
                            }
                        }
                }

                // 选择图片按钮
                PhotosPicker(selection: $selectedItem, matching: .images) {
                    Label("选择图片", systemImage: "photo.on.rectangle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isProcessing || !engineReady)

                // 状态信息
                if !engineReady {
                    HStack {
                        ProgressView()
                        Text("正在加载 OCR 引擎...")
                            .foregroundStyle(.secondary)
                    }
                }

                if isProcessing {
                    HStack {
                        ProgressView()
                        Text("正在识别...")
                            .foregroundStyle(.secondary)
                    }
                }

                if let error = errorMessage {
                    Text(error)
                        .foregroundStyle(.red)
                        .font(.caption)
                        .padding(.horizontal)
                }

                // OCR 结果列表
                if !ocrResults.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Text("识别结果 (\(ocrResults.count) 项)")
                                .font(.headline)
                            Spacer()
                            Text(String(format: "%.0fms", processingTime * 1000))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        List(ocrResults) { item in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(item.text)
                                    .font(.body)
                                HStack {
                                    Text(String(format: "置信度: %.1f%%", item.confidence * 100))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    Spacer()
                                    Text("(\(Int(item.bbox.origin.x)), \(Int(item.bbox.origin.y)))")
                                        .font(.caption2)
                                        .foregroundStyle(.tertiary)
                                }
                            }
                        }
                        .listStyle(.plain)
                    }
                }

                Spacer()

                // 版本信息
                if let version = ocrVersion() {
                    Text("ocr-capi v\(version)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding()
            .navigationTitle("OCR Demo")
            .onChange(of: selectedItem) { _, newItem in
                Task {
                    await loadAndRecognize(item: newItem)
                }
            }
            .task {
                await initEngine()
            }
        }
    }

    // MARK: - Private Methods

    private func initEngine() async {
        do {
            // 在后台线程初始化引擎
            let eng = try await Task.detached {
                // 使用中文 v4 模型（需要将模型文件添加到 Bundle）
                // 也可以替换为 v5 模型
                try OcrEngine(
                    detModel: "ch_PP-OCRv4_det_infer",
                    recModel: "ch_PP-OCRv4_rec_infer",
                    charset: "ppocr_keys_v4",
                    useGPU: true
                )
            }.value
            engine = eng
            engineReady = true
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func loadAndRecognize(item: PhotosPickerItem?) async {
        guard let item = item else { return }
        errorMessage = nil
        ocrResults = []

        // 加载图片
        guard let data = try? await item.loadTransferable(type: Data.self),
              let image = UIImage(data: data)
        else {
            errorMessage = "无法加载图片"
            return
        }

        selectedImage = image
        isProcessing = true

        // 在后台线程执行 OCR
        let eng = engine
        let start = CFAbsoluteTimeGetCurrent()
        let results = await Task.detached {
            eng?.recognize(image: image) ?? []
        }.value
        let elapsed = CFAbsoluteTimeGetCurrent() - start

        ocrResults = results
        processingTime = elapsed
        isProcessing = false

        if results.isEmpty {
            errorMessage = "未识别到文本"
        }
    }

    private func ocrVersion() -> String? {
        guard let ptr = ocr_version() else { return nil }
        return String(cString: ptr)
    }
}

#Preview {
    ContentView()
}
