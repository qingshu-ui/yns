package io.github.qingshu.yns.service.impl

import io.github.qingshu.yns.entity.ImageCacheEntity
import io.github.qingshu.yns.repository.ImageCacheRepository
import io.github.qingshu.yns.service.ImageCacheService
import org.springframework.stereotype.Service
import java.time.LocalDateTime

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
@Service
class ImageCacheServiceImpl(
    val repository: ImageCacheRepository,
) : ImageCacheService {
    override fun save(entity: ImageCacheEntity): ImageCacheEntity {
        return repository.save(entity)
    }

    override fun findExpiredFiles(now: LocalDateTime): List<ImageCacheEntity> {
        return repository.findAllByExpiresAtBefore(now)
    }
}