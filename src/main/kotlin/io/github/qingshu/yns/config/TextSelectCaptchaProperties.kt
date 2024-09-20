package io.github.qingshu.yns.config

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.stereotype.Component

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
@Component
@ConfigurationProperties(prefix = "yns.text-select.captcha")
data class TextSelectCaptchaProperties(
    /**
     * YoloV8 模型文件路径
     */
    var yoloModelPath: String = "",

    /**
     * Siamese 模型文件路径
     */
    var siameseModelPath: String = "",

    /**
     * 对应 YOLO 模型的标签文件
     */
    var labelPath: String = "",

    /**
     * 是否启用此功能
     */
    var enable: Boolean = false,
)