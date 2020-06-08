# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

"""This module contains algorithms implementing preprocessing steps that might help
filter information that correlates with sensitive attributes.
"""

from ._information_filter import InformationFilter

__all__ = ["InformationFilter"]
