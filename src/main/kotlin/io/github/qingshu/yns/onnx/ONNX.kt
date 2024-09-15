package io.github.qingshu.yns.onnx

import org.opencv.core.Mat

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the example-ayaka project.
 *
 * This project is licensed under the GPL-3.0 License.
 * See the LICENSE file for details.
 */
interface ONNX : AutoCloseable {

    fun load(vararg paths: String)

    fun detect(vararg input: String): Any

    fun detect(vararg input: Mat): Any

    companion object {
        fun createSiamese(modelPath: String): ONNX {
            return Siamese().apply {
                load(modelPath)
            }
        }
    }
}