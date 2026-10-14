"""Canned world-building replies shared by tests that build a world through the mock LLM."""

from __future__ import annotations

import json

# AbilityDesigner's reply. Three entries serve any cast of up to three: an index past the cast
# is dropped like any out-of-range IndexedRef, so the two-figure test worlds use it too.
MOCK_ABILITY_DESIGN_RESPONSE = json.dumps({
    "reason": "一人能征惯战，一人长于谋算，一人年老体衰。",
    "cast": [
        {
            "index": 1, "basis": "久经沙场，胆气过人",
            "attributes": {"体魄": 4, "身手": 4, "心智": 3, "口才": 2, "胆识": 5},
            "skills": [{"name": "骑射", "level": 3}, {"name": "兵法", "level": 2}],
            "cash": 500,
        },
        {
            "index": 2, "basis": "善于结纳，不长武艺",
            "attributes": {"体魄": 3, "身手": 2, "心智": 4, "口才": 4, "胆识": 3},
            "skills": [{"name": "权谋", "level": 2}],
            "cash": 800,
        },
        {
            "index": 3, "basis": "年老体衰，阅历深厚",
            "attributes": {"体魄": 1, "身手": 2, "心智": 4, "口才": 3, "胆识": 3},
            "skills": [],
            "cash": 0,
        },
    ],
}, ensure_ascii=False)
