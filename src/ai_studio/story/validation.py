"""Cross-file checks for the story layer.

validate_story() returns EVERY problem as a list of sentences instead of
stopping at the first one, so that whoever fills the files (a person, or
a script acting as director) can fix them in one pass. check_story() raises
StoryValidationError when that list is not empty.

What is checked:
  durations    every shot is a whole number of frames at the plan's fps; each
               scene's duration equals the sum of its shots; the plan's total
               equals the sum of its scenes
  characters   every character in a shot, in a keyframe requirement and in
               the story bible exists in the character bible
  keyframes    every keyframe requirement either points, by SHA-256, to an
               asset recorded as available (a character's reference asset or
               one of the plan's shared assets), or is marked missing
  identity     the plan belongs to the story; one hash is one file

verify_assets() is the only function that touches the disk: it compares the
recorded hashes with the files that are actually there. Media files are
git-ignored, so an absent file is reported, not treated as a broken plan.
"""

from __future__ import annotations

from pathlib import Path

from ai_studio.paths import resolve_path
from ai_studio.quality.store import sha256_file
from ai_studio.story.schema import CharacterBible, ReferenceAsset, ShotPlan, StoryBible

_FRAME_EPSILON = 1e-6


class StoryValidationError(ValueError):
    """The story files contradict each other; .problems lists every problem."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def asset_index(characters: CharacterBible, plan: ShotPlan | None = None) -> dict[str, ReferenceAsset]:
    """Every AVAILABLE asset by SHA-256: the characters' reference assets plus
    the plan's shared assets."""
    index: dict[str, ReferenceAsset] = {}
    for asset in _all_assets(characters, plan):
        if asset.sha256 is not None:
            index.setdefault(asset.sha256, asset)
    return index


def _all_assets(characters: CharacterBible, plan: ShotPlan | None) -> list[ReferenceAsset]:
    assets = [asset for character in characters.characters for asset in character.reference_assets]
    if plan is not None:
        assets.extend(plan.assets)
    return assets


def _whole_frames(seconds: float, fps: int) -> bool:
    frames = seconds * fps
    return abs(frames - round(frames)) <= _FRAME_EPSILON


def _duration_problems(plan: ShotPlan) -> list[str]:
    problems = []
    for scene in plan.scenes:
        for shot in scene.shots:
            if not _whole_frames(shot.duration_seconds, plan.fps):
                problems.append(
                    f"shot {shot.id}: {shot.duration_seconds:g} s is not a whole number of "
                    f"frames at {plan.fps} fps"
                )
        shots_total = sum(shot.duration_seconds for shot in scene.shots)
        if abs(shots_total - scene.duration_seconds) > _FRAME_EPSILON:
            problems.append(
                f"scene {scene.id}: its shots add up to {shots_total:g} s, "
                f"but duration_seconds is {scene.duration_seconds:g}"
            )
    scenes_total = sum(scene.duration_seconds for scene in plan.scenes)
    if abs(scenes_total - plan.total_duration_seconds) > _FRAME_EPSILON:
        problems.append(
            f"shot plan {plan.plan_id}: its scenes add up to {scenes_total:g} s, "
            f"but total_duration_seconds is {plan.total_duration_seconds:g}"
        )
    return problems


def _asset_problems(characters: CharacterBible, plan: ShotPlan) -> list[str]:
    """One hash is one file: the same SHA-256 under two different paths means
    one of the two entries is wrong."""
    problems = []
    path_by_hash: dict[str, str] = {}
    for asset in _all_assets(characters, plan):
        if asset.sha256 is None:
            continue
        known = path_by_hash.setdefault(asset.sha256, asset.path)
        if known != asset.path:
            problems.append(
                f"asset {asset.id}: sha256 {asset.sha256[:12]}… is recorded for both "
                f"{known} and {asset.path}"
            )
    return problems


def _shot_problems(plan: ShotPlan, characters: CharacterBible) -> list[str]:
    problems = []
    known = set(characters.ids)
    assets = asset_index(characters, plan)
    for shot in plan.shots:
        for character_id in shot.characters:
            if character_id not in known:
                problems.append(f"shot {shot.id}: character {character_id!r} is not in the character bible")
        start_frames = [keyframe for keyframe in shot.keyframes if keyframe.role == "start_frame"]
        if len(start_frames) > 1:
            problems.append(
                f"shot {shot.id}: {len(start_frames)} start_frame keyframes; a shot has one frame 0"
            )
        used: dict[str, str] = {}
        for keyframe in shot.keyframes:
            where = f"shot {shot.id}, keyframe {keyframe.id}"
            if keyframe.character is not None and keyframe.character not in shot.characters:
                problems.append(
                    f"{where}: character {keyframe.character!r} is not listed in the shot's characters"
                )
            if keyframe.asset_sha256 is None:
                continue  # openly marked missing
            asset = assets.get(keyframe.asset_sha256)
            if asset is None:
                problems.append(
                    f"{where}: asset_sha256 {keyframe.asset_sha256[:12]}… matches no available asset "
                    "in the character bible or the plan's assets (record the asset, or mark the "
                    "keyframe missing)"
                )
                continue
            if keyframe.asset_sha256 in used:
                # generation.request: start_frame must not also be a reference image
                problems.append(
                    f"{where}: uses the same asset as keyframe {used[keyframe.asset_sha256]}"
                )
            used[keyframe.asset_sha256] = keyframe.id
            if keyframe.character is not None and keyframe.character in known:
                own = {a.sha256 for a in characters.get(keyframe.character).reference_assets}
                if keyframe.asset_sha256 not in own:
                    problems.append(
                        f"{where}: the asset is not one of {keyframe.character}'s reference assets"
                    )
    return problems


def validate_story(
    plan: ShotPlan, characters: CharacterBible, story: StoryBible | None = None
) -> list[str]:
    """Every problem found, as sentences; an empty list means consistent."""
    problems = _duration_problems(plan)
    problems += _asset_problems(characters, plan)
    problems += _shot_problems(plan, characters)
    if story is not None:
        if plan.story_id != story.story_id:
            problems.append(
                f"shot plan {plan.plan_id} is for story {plan.story_id!r}, "
                f"but the story bible is {story.story_id!r}"
            )
        known = set(characters.ids)
        for character_id in story.characters:
            if character_id not in known:
                problems.append(f"story bible: character {character_id!r} is not in the character bible")
        if story.characters:
            for shot in plan.shots:
                for character_id in shot.characters:
                    if character_id in known and character_id not in story.characters:
                        problems.append(
                            f"shot {shot.id}: character {character_id!r} is not listed in the story bible"
                        )
    return problems


def check_story(plan: ShotPlan, characters: CharacterBible, story: StoryBible | None = None) -> None:
    problems = validate_story(plan, characters, story)
    if problems:
        raise StoryValidationError(problems)


def missing_keyframes(plan: ShotPlan) -> list[tuple[str, str, str]]:
    """The production to-do list: (shot id, keyframe id, description) for every
    keyframe requirement that is marked missing, in playing order."""
    return [
        (shot.id, keyframe.id, keyframe.description)
        for shot in plan.shots
        for keyframe in shot.keyframes
        if keyframe.status == "missing"
    ]


def verify_assets(
    characters: CharacterBible, plan: ShotPlan | None = None, root: str | Path | None = None
) -> list[str]:
    """Compare recorded hashes with the files on disk. Reads files; changes
    nothing. Returns one sentence per asset that is absent or different."""
    problems = []
    for asset in _all_assets(characters, plan):
        path = resolve_path(asset.path, root)
        if asset.status == "missing":
            if path.is_file():
                problems.append(
                    f"asset {asset.id}: marked missing, but {asset.path} exists "
                    f"(sha256 {sha256_file(path)}); record it as available"
                )
            continue
        if not path.is_file():
            problems.append(f"asset {asset.id}: {asset.path} is not on this machine")
        elif sha256_file(path) != asset.sha256:
            problems.append(
                f"asset {asset.id}: {asset.path} does not match the recorded sha256 "
                f"{asset.sha256[:12]}… (a used asset must never be edited or replaced)"
            )
    return problems
