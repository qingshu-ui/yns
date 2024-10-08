package io.github.qingshu.yns.onnx

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
interface OnnxModel<T> {

    fun preprocess(input: T): T

    fun runInference(tensors: T): T

    fun postprocess(result: T): T
}