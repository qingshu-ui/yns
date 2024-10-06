package io.github.qingshu.yns.config

import ai.onnxruntime.OrtSession
import io.github.qingshu.yns.onnx.impl.SiameseOnnxModel
import io.github.qingshu.yns.onnx.impl.YoloOnnxModel
import io.github.qingshu.yns.service.TextSelectCaptcha
import io.github.qingshu.yns.service.TextSelectCaptchaImpl
import nu.pattern.OpenCV
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
@Configuration
class TextSelectCaptchaConfig {

    @Bean
    @ConditionalOnProperty(
        prefix = "yns.text-select.captcha",
        name = ["enable"],
        havingValue = "true",
        matchIfMissing = false
    )
    fun textSelectCaptcha(config: TextSelectCaptchaProperties): TextSelectCaptcha {
        val yoloModelPath = config.yoloModelPath
        val siameseModelPath = config.siameseModelPath
        val labelPath = config.labelPath
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val yoloModel = YoloOnnxModel(yoloModelPath, options)
        val siameseModel = SiameseOnnxModel(siameseModelPath, options)
        return TextSelectCaptchaImpl(yoloModel, siameseModel, labelPath)
    }

    init {
        OpenCV.loadLocally()
    }
}