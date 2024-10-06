package io.github.qingshu.yns.config

import io.github.qingshu.yns.annotation.Slf4j
import io.github.qingshu.yns.annotation.Slf4j.Companion.log
import org.springframework.boot.jdbc.DataSourceBuilder
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.Configuration
import javax.sql.DataSource

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
@Slf4j
@Configuration
class DataSourceConfig {

    @Bean
    fun getDataSource(): DataSource {
        val builder = DataSourceBuilder.create()

        builder.driverClassName("org.sqlite.JDBC")
        builder.url("jdbc:sqlite:yns.sqlite.db")
        log.info("使用 SQLite 数据库")
        return builder.build()
    }
}