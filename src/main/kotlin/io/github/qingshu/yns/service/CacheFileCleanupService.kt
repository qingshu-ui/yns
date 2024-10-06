package io.github.qingshu.yns.service

import io.github.qingshu.yns.config.TextSelectCaptchaProperties
import io.github.qingshu.yns.repository.ImageCacheRepository
import org.springframework.scheduling.annotation.Scheduled
import org.springframework.stereotype.Service
import java.nio.file.Files
import java.nio.file.Path
import java.time.LocalDateTime

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
@Service
class CacheFileCleanupService(
    val repository: ImageCacheRepository,
    val cfg: TextSelectCaptchaProperties,
) {

    @Scheduled(cron = "0 0 * * * ?")
    fun cleanExpiredFiles() {
        val now = LocalDateTime.now()
        val expiredFiles = repository.findAllByExpiresAtBefore(now)

        expiredFiles.forEach { entity ->
            val filePath = Path.of(cfg.imageCachePath, entity.fileName)

            if (Files.exists(filePath)) {
                Files.delete(filePath)
            }
            repository.deleteById(entity.id)
        }
    }
}