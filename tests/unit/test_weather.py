"""Weather: the transition table, seeded reproducibility, persistence and the words it reaches prompts as."""

from __future__ import annotations

import re

import pytest

from core.interfaces.phenomenon import Phenomenon
from core.interfaces.severity import Severity
from engine.clock import WorldTime
from engine.weather import (
    TICK_SECONDS, Climate, Season, Sky, Weather, WeatherSystem, change_notice, transition_row,
)


def _at(hours: float, *, month: int = 7) -> WorldTime:
    return WorldTime(step=0, elapsed_seconds=int(hours * 3600), start_month=month)


@pytest.mark.parametrize("climate", list(Climate))
@pytest.mark.parametrize("season", list(Season))
@pytest.mark.parametrize("sky", list(Sky))
def test_every_transition_row_is_a_distribution(climate: Climate, season: Season, sky: Sky) -> None:
    row = transition_row(sky, climate, season)
    assert set(row) == set(Sky)
    assert all(p >= 0 for p in row.values())
    assert sum(row.values()) == pytest.approx(1.0)


def test_no_snow_in_a_tropical_year() -> None:
    for season in Season:
        assert transition_row(Sky.CLEAR, Climate.TROPICAL, season)[Sky.SNOW] == 0


@pytest.mark.parametrize(("month", "season"), [(1, Season.WINTER), (4, Season.SPRING), (7, Season.SUMMER), (10, Season.AUTUMN), (12, Season.WINTER)])
def test_season_follows_the_month(month: int, season: Season) -> None:
    assert Season.of_month(month) is season


def _history(seed: str, ticks: int = 40) -> list[Weather]:
    system = WeatherSystem.opening(climate=Climate.HUMID_SUBTROPICAL, seed=seed, month=6)
    seen = [system.weather]
    for i in range(1, ticks + 1):
        system.advance(_at(i * TICK_SECONDS / 3600))
        seen.append(system.weather)
    return seen


def test_the_same_world_sees_the_same_weather() -> None:
    assert _history("w1") == _history("w1")


def test_two_worlds_see_different_weather() -> None:
    assert _history("w1") != _history("w2")


def test_a_long_step_runs_every_tick_it_crossed() -> None:
    """One 12-hour step must land where four 3-hour steps do: the sky doesn't care about steps."""
    stepped = WeatherSystem(climate=Climate.COLD, seed="s")
    for hours in (3, 6, 9, 12):
        stepped.advance(_at(hours))
    leapt = WeatherSystem(climate=Climate.COLD, seed="s")
    leapt.advance(_at(12))
    assert stepped.weather == leapt.weather
    assert stepped.snapshot() == leapt.snapshot()


def test_advance_reports_the_sky_before_it() -> None:
    system = WeatherSystem(climate=Climate.TROPICAL, seed="s", weather=Weather(Sky.RAIN, Severity.HIGH))
    before = system.advance(_at(30))
    assert before == Weather(Sky.RAIN, Severity.HIGH)


def test_a_restored_world_carries_on_exactly() -> None:
    original = WeatherSystem(climate=Climate.ARID, seed="s")
    original.advance(_at(9))
    restored = WeatherSystem.restore(original.snapshot(), seed="s", climate=Climate.TEMPERATE_MONSOON)
    assert restored.climate is Climate.ARID
    for system in (original, restored):
        system.advance(_at(60))
    assert restored.weather == original.weather


@pytest.mark.parametrize("payload", [None, {}, {"sky": "hail", "tick": "x"}])
def test_a_world_saved_without_weather_opens_clear_under_its_climate(payload: object) -> None:
    system = WeatherSystem.restore(payload, seed="s", climate=Climate.COLD)
    assert system.weather == Weather()
    assert system.climate is Climate.COLD


def test_a_sky_without_strength_has_none() -> None:
    assert Weather(Sky.CLEAR, Severity.HIGH).severity is Severity.LOW


def test_weather_reaches_prompts_as_words_only() -> None:
    for sky in Sky:
        for severity in Severity:
            text = Weather(sky, severity).describe()
            assert text and not re.search(r"[0-9a-z]", text)


def test_only_precipitation_and_wind_are_drawn() -> None:
    assert Weather(Sky.RAIN).phenomenon is Phenomenon.RAIN
    assert Weather(Sky.OVERCAST).phenomenon is Phenomenon.NONE


def test_heavy_rain_and_snow_slow_travel() -> None:
    assert Weather(Sky.RAIN, Severity.HIGH).travel_factor == 1.25
    assert Weather(Sky.SNOW, Severity.HIGH).travel_factor == 1.5
    assert Weather(Sky.RAIN, Severity.MEDIUM).travel_factor == 1.0


def test_change_notices() -> None:
    assert change_notice(Weather(), Weather()) == ""
    assert change_notice(Weather(), Weather(Sky.RAIN, Severity.LOW)) == "下起雨来了。"
    assert change_notice(Weather(Sky.RAIN), Weather()) == "雨停了。"
    assert change_notice(Weather(Sky.RAIN, Severity.LOW), Weather(Sky.RAIN, Severity.HIGH)) == "雨势大了起来。"
    assert change_notice(Weather(Sky.SNOW), Weather(Sky.OVERCAST)) == "雪停了。天仍阴着。"


def test_the_situation_header_carries_the_sky() -> None:
    from core.interfaces.perception import LocationView, Situation
    from core.prompts import SituationVoice, render_situation_header

    here = Situation(location_view=LocationView(name="大殿"), time_label="六月初四午时", weather="外头下着大雨")
    assert render_situation_header(here, voice=SituationVoice.FIRST) == "我此刻在大殿，当前时间为六月初四午时，外头下着大雨。"
    assert render_situation_header(here, voice=SituationVoice.THIRD).endswith("；天气：外头下着大雨。")
    assert "天气" not in render_situation_header(Situation(time_label="午时"), voice=SituationVoice.THIRD)


def test_a_turn_of_the_sky_is_noticed_everywhere_and_heavy_weather_slows_travel() -> None:
    from core.interfaces.place import Place
    from engine.environment import EnvironmentSystem

    env = EnvironmentSystem()
    env.space.register_place(Place(place_id="hall", name="大殿", connections={"yard": 3600}))
    env.space.register_place(Place(place_id="yard", name="庭中", connections={"hall": 3600}))
    env.place_agent(agent_id="a", location_id="hall")
    env.begin_step(step=1, world_time=_at(3), weather=Weather(Sky.SNOW, Severity.HIGH), weather_notice="下起大雪来了。")
    spatial = env.spatial_for(agent_id="a")
    assert spatial.weather_label == "外头大雪纷飞"
    assert [e.content for e in spatial.ambient_events] == ["下起大雪来了。"]
    assert [r.travel_seconds for r in spatial.reachable_locations] == [5400]


async def test_a_world_runs_under_its_sky(mock_build_container, test_config) -> None:
    from engine.application import NarrativeApplication
    from interaction.models import StepEvent

    app = NarrativeApplication(mock_build_container, test_config)
    world = await app.build_world("玄武门之变", template="changan_iso")
    await app.run_world(world.world_id, steps=1)
    snapshots = mock_build_container.snapshot
    opening = (await snapshots.load(world.world_id, 0)).metadata["weather"]
    after = (await snapshots.load(world.world_id, 1)).metadata["weather"]
    assert opening["climate"] == after["climate"] == "temperate_monsoon"
    assert StepEvent.from_snapshot(await snapshots.load(world.world_id, 1)).weather is not None
    assert world.environment.current_weather_label
