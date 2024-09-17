package io.github.qingshu.yns.test

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import org.opencv.core.Size

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the GPL-3.0 License.
 * See the LICENSE file for details.
 */
@Suppress("unused", "MemberVisibilityCanBePrivate")
abstract class AbstractOnnxModel<T>(
    private val modelPath: String,
    private val option: OrtSession.SessionOptions = OrtSession.SessionOptions(),
): OnnxModel<T> {
    val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    val session: OrtSession = env.createSession(modelPath, option)
    val inputNames: Set<String> = session.inputNames
    var inputShape = listOf<LongArray>()
    var inputSize = mapOf<String, Size>()


    init {
        val (shape, size) = session.inputInfo.run {
            val shapes = mutableListOf<LongArray>()
            val inputSize = mutableMapOf<String, Size>()
            inputNames.forEach {
                val shape = (this[it]?.info as TensorInfo).shape
                shapes.add(shape)
                inputSize[it] = Size(shape[2].toDouble(), shape[3].toDouble())
            }
            listOf(shapes, inputSize)
        }

        @Suppress("UNCHECKED_CAST")
        inputShape = shape as List<LongArray>
        @Suppress("UNCHECKED_CAST")
        inputSize = size as Map<String, Size>
    }
}