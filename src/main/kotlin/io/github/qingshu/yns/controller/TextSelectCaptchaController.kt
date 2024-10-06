package io.github.qingshu.yns.controller

import io.github.qingshu.yns.config.TextSelectCaptchaProperties
import io.github.qingshu.yns.dto.ReasonResponseDto
import io.github.qingshu.yns.entity.ImageCacheEntity
import io.github.qingshu.yns.service.ImageCacheService
import io.github.qingshu.yns.service.TextSelectCaptcha
import org.opencv.core.MatOfByte
import org.opencv.imgcodecs.Imgcodecs
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty
import org.springframework.http.MediaType
import org.springframework.http.ResponseEntity
import org.springframework.web.bind.annotation.*
import org.springframework.web.multipart.MultipartFile
import org.springframework.web.servlet.support.ServletUriComponentsBuilder
import java.nio.file.Files
import java.util.*
import kotlin.io.path.Path
import kotlin.io.path.pathString

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
    val captcha: TextSelectCaptcha,
    val cfg: TextSelectCaptchaProperties,
    val service: ImageCacheService,
) {

    @PostMapping("/reason")
    fun reason(@RequestParam("image") image: MultipartFile): ResponseEntity<Any> {
        val mat = Imgcodecs.imdecode(MatOfByte(*image.bytes), Imgcodecs.IMREAD_COLOR).apply {
            if (empty()) return ResponseEntity.badRequest().body(mapOf("error" to "Could not load image"))
        }
        val detections = captcha.run(mat)
        detections.forEachIndexed { index, detection ->
            detection.drawWithIndex(mat, index)
        }
        val fileName = "${UUID.randomUUID()}.png"
        val savePath = Path(cfg.imageCachePath, fileName).pathString
        Imgcodecs.imwrite(savePath, mat)
        service.save(ImageCacheEntity(fileName = fileName))
        val cacheUrl = ServletUriComponentsBuilder.fromCurrentContextPath()
            .pathSegment("text-select.captcha/cache")
            .queryParam("file", fileName)
            .build().toUriString()
        return ResponseEntity.ok(ReasonResponseDto(cacheUrl, detections))
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