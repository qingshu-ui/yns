package io.github.qingshu.yns.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import io.github.qingshu.yns.dto.DetectStatus
import org.opencv.core.Size

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the GPL-3.0 License.
 * See the LICENSE file for details.
 */
@Suppress("unused", "MemberVisibilityCanBePrivate")
abstract class AbstractOnnxModel<T: AbstractDetect>(
    modelPath: String,
    options: OrtSession.SessionOptions = OrtSession.SessionOptions(),
): OnnxModel<T>, AutoCloseable {
    val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    val session: OrtSession = env.createSession(modelPath, options)
    val inputNames: List<String> = session.inputNames.toList()
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

    override fun close() {
        session.close()
        env.close()
    }

    override fun runInference(tensors: T): T {
        checkStatus(tensors.status, DetectStatus.INFERENCE)
        val mTensors = tensors.tensors
        var result: OrtSession.Result
        synchronized(session) {
            result = session.run(mTensors)
        }
        tensors.result = result
        tensors.status = DetectStatus.POSTPROCESS
        return tensors
    }

    protected fun checkStatus(status: DetectStatus, required: DetectStatus){
        require(status == required) {
            "Invalid state for required: $required"
        }
    }
}