package io.github.qingshu.yns.controller

import io.github.qingshu.yns.config.TextSelectCaptchaProperties
import io.github.qingshu.yns.service.TextSelectCaptcha
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile
import java.nio.file.Files
import kotlin.io.path.Path

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
    val service: TextSelectCaptcha,
    val cfg: TextSelectCaptchaProperties,
) {

    @PostMapping("/reason")
    fun reason(@RequestParam("image") image: MultipartFile): ResponseEntity<Any> {
        val response =
            service.run(image) ?: return ResponseEntity.badRequest().body(mapOf("error" to "Could not load image"))
        return ResponseEntity.ok(response)
    }

    @GetMapping("/cache")
    fun cache(@RequestParam("file") file: String): ResponseEntity<Any> {
        val imagePath = Path(cfg.imageCachePath, file)

        return if (Files.exists(imagePath)) {
            val imageBytes = Files.readAllBytes(imagePath)

            ResponseEntity.ok()
                .contentType(MediaType.IMAGE_PNG)
                .body(imageBytes)
        } else {
            ResponseEntity.notFound().build()
        }
    }

}