package io.github.qingshu.yns.controller

import io.github.qingshu.yns.service.TextSelectCaptcha
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.PostMapping
import org.springframework.web.bind.annotation.RequestMapping
import org.springframework.web.bind.annotation.RequestParam
import org.springframework.web.bind.annotation.RestController
import org.springframework.web.multipart.MultipartFile

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
@RestController
@ConditionalOnProperty(
    prefix = "yns.text-select.captcha",
    name = ["enable"],
    havingValue = "true",
    matchIfMissing = false,
)
@RequestMapping("/text-select.captcha")
class TextSelectCaptchaController(
    val captcha: TextSelectCaptcha
) {

    @PostMapping("/reason")
    fun reason(@RequestParam("image") image: MultipartFile): ResponseEntity<Map<String, Any>> {
        TODO("Not yet implemented")
    }
}