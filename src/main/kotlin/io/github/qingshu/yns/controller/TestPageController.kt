package io.github.qingshu.yns.controller

import org.springframework.stereotype.Controller
import org.springframework.web.bind.annotation.GetMapping

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the yns project.
 *
 * This project is licensed under the MIT License.
 * See the LICENSE file for details.
 */
@Controller
class TestPageController {

    @GetMapping("reason")
    fun reason() = "reason"
}