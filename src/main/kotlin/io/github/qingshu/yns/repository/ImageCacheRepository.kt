package io.github.qingshu.yns.repository

import io.github.qingshu.yns.entity.ImageCacheEntity
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.transaction.annotation.Transactional
import java.time.LocalDateTime

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
interface ImageCacheRepository : JpaRepository<ImageCacheEntity, Int> {

    @Transactional
    override fun <S : ImageCacheEntity> save(entity: S): S

    fun findAllByExpiresAtBefore(now: LocalDateTime): List<ImageCacheEntity>
}