from __future__ import annotations

from step2point.core.results import ValidationResult
from step2point.metrics.shower_shapes import shower_moments
from step2point.validation.base import Validator


class ShowerMomentsValidator(Validator):
    name = "shower_moments"

    def run(self, before, after) -> ValidationResult:
        m_pre = shower_moments(before)
        m_post = shower_moments(after)
        out = {}
        for key, v_pre in m_pre.items():
            v_post = m_post[key]
            out[f"{key}_difference"] = float(v_post - v_pre)
        return ValidationResult(self.name, out)
