package io.github.qingshu.yns.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession
import io.github.qingshu.yns.dto.DetectStatus

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
abstract class AbstractDetect: AutoCloseable {
    var status: DetectStatus = DetectStatus.PREPROCESS
    var tensors: MutableMap<String, OnnxTensor>? = null
    var result: OrtSession.Result? = null

    override fun close() {
        // mat.release()
        result?.close()
        tensors?.forEach { it.value.close() }
    }
}