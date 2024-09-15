package io.github.qingshu.yns.exception

/**
 * Copyright (c) 2024 qingshu.
 * This file is part of the example-ayaka project.
 *
 * This project is licensed under the GPL-3.0 License.
 * See the LICENSE file for details.
 */
class UnloadedModelException(message: String = "No model loaded") : Exception(message)