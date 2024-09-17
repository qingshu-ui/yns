package io.github.qingshu.yns.test

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtSession.Result
import org.opencv.core.Mat

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the GPL-3.0 License.
 * See the LICENSE file for details.
 */
interface OnnxModel<T> {

    fun preprocess(input: T): T

    fun runInference(tensors: T): T

    fun postprocess(result: T): T
}