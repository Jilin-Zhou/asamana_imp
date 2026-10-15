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
            "carries": [1],
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

# EventDeckDesigner's reply: the minimum deck, one card that needs rain and one that needs night.
MOCK_EVENT_DECK_RESPONSE = json.dumps({
    "reason": "宫城内外，消息与天时都会自己生出事来。",
    "cards": [
        {"title": f"坊间传言{i}", "synopsis": f"坊间又起一则传言，第{i}则。", "scope": "local",
         "severity": "low", "weight": 1, "sky": "any", "daypart": "any"}
        for i in range(1, 11)
    ] + [
        {"title": "大雨冲垮坊墙", "synopsis": "连日大雨，一段坊墙塌了，路被堵住。", "scope": "local",
         "severity": "medium", "weight": 2, "sky": "rain", "daypart": "any"},
        {"title": "夜半宵禁", "synopsis": "金吾卫夜里加紧巡街，犯夜者被拿。", "scope": "global",
         "severity": "medium", "weight": 1, "sky": "any", "daypart": "night"},
    ],
}, ensure_ascii=False)
