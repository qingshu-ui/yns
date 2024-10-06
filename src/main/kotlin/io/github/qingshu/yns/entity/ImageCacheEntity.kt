package io.github.qingshu.yns.entity

import jakarta.persistence.*
import java.time.LocalDateTime

@Entity
@Table(name = "image_cache")
data class ImageCacheEntity(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    val id: Int = 0,

    @Column(nullable = false)
    val fileName: String,

    @Column(nullable = false)
    val expiresAt: LocalDateTime = LocalDateTime.now().plusDays(3),
)
