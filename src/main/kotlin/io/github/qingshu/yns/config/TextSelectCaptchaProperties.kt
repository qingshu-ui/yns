package io.github.qingshu.yns.config

import org.springframework.boot.context.properties.ConfigurationProperties
import org.springframework.stereotype.Component
import kotlin.io.path.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.notExists

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

    /**
     * 在推理过程中产出图片的缓存目录
     */
    var imageCachePath: String = "cache"
) {
    init {
        initCacheDir()
    }

    private fun initCacheDir() {
        val cachePath = Path(imageCachePath)
        if (cachePath.notExists()) {
            cachePath.createDirectories()
        }
    }
}