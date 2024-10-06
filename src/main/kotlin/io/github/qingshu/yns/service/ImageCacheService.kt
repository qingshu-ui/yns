package io.github.qingshu.yns.service

import io.github.qingshu.yns.entity.ImageCacheEntity
import java.time.LocalDateTime

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
interface ImageCacheService {

    fun save(entity: ImageCacheEntity): ImageCacheEntity

    fun findExpiredFiles(now: LocalDateTime): List<ImageCacheEntity>
}