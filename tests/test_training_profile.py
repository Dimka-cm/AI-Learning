from __future__ import annotations

import importlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_training_profile_captures_owner_choices():
    profile = yaml.safe_load((ROOT / "config" / "training_profile.yaml").read_text(encoding="utf-8"))

    assert profile["owner"]["no_insults"] is True
    assert profile["sources"]["drive_allowed"] is True
    assert profile["internet"]["mode"] == "отдельный инструмент поиска, с источниками"
    assert profile["first_github_actions_run"]["data_source"] == "hf"
    assert profile["first_github_actions_run"]["max_steps"] == 300
    assert profile["future_ui"]["exact_clone_of_claude"] is False

    priorities = profile["code_priorities"]
    for key in ("cpp", "unreal_engine_5", "java", "mixins", "gradle", "python", "blueprint"):
        assert key in priorities


def test_eval_prompts_cover_required_domains():
    data = yaml.safe_load((ROOT / "eval" / "model_quality_prompts.yaml").read_text(encoding="utf-8"))
    categories = {p["category"] for p in data["prompts"]}

    required = {
        "chat", "behavior", "pc_agent_safety", "internet", "math", "python",
        "cpp", "c", "csharp", "java", "javascript_typescript", "kotlin",
        "minecraft", "gradle", "gradle_kotlin_dsl", "unreal_engine_5",
    }
    assert required <= categories
    assert len(data["prompts"]) >= 30


def test_demo_data_contains_new_code_domains():
    make_data = importlib.import_module("scripts.make_data")
    corpus = "\n".join(make_data.CODE_SNIPPETS + make_data.CODE_QA)

    needles = [
        "Python", "Java", "TypeScript", "Kotlin", "Gradle", "Mixin",
        "NeoForge", "Fabric", "Blueprint", "UE5",
    ]
    for needle in needles:
        assert needle in corpus
