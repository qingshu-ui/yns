package io.github.qingshu.yns.service

import io.github.qingshu.yns.dto.Detection
import org.opencv.core.Mat

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
interface TextSelectCaptcha {

    fun run(imagePath: String): List<Detection>

    fun run(image: Mat): List<Detection>
}