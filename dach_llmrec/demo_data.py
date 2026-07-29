from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def create_demo_database(output: str | Path) -> Path:
    """Create a tiny SQLite database with the tables required by the prototype."""

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    conn = sqlite3.connect(str(output))
    try:
        _create_schema(conn)
        _insert_demo_rows(conn)
        conn.commit()
    finally:
        conn.close()
    return output


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE norm_recipe_v1(
          recipe_id INT,
          name TEXT,
          description TEXT,
          cuisine_id INT,
          cuisine_name TEXT,
          source_id INT,
          cookstep_summary TEXT,
          ingredient_count INT,
          seasoning_count INT,
          step_count INT,
          cooking_methods TEXT,
          taste_tags TEXT,
          content_status TEXT,
          recommendable INT,
          restriction_reasons TEXT,
          source_status TEXT,
          pipeline_version TEXT
        );

        CREATE TABLE norm_ingredient_v1(
          ingredient_id INT,
          name TEXT,
          alias TEXT,
          english_name TEXT,
          foodtype_id INT,
          ingredient_type_id INT,
          nutrients TEXT,
          nutrition_status TEXT,
          source_status TEXT,
          pipeline_version TEXT,
          entity_scope TEXT,
          entity_class TEXT
        );

        CREATE TABLE norm_recipe_ingredient_v1(
          row_id INT,
          recipe_id INT,
          raw_name TEXT,
          raw_description TEXT,
          raw_ingredient_id INT,
          resolved_ingredient_id INT,
          term_ingredient_id INT,
          resolution_status TEXT,
          quantity REAL,
          normalized_unit TEXT,
          is_main INT,
          confidence REAL,
          review_status TEXT,
          is_food_input INT,
          exclusion_reason TEXT,
          pipeline_version TEXT
        );

        CREATE TABLE ingredient2taste(food INTEGER, taste INTEGER, PRIMARY KEY(food,taste));
        CREATE TABLE hci(id INTEGER, name TEXT, description TEXT, parent INTEGER, PRIMARY KEY(id));
        CREATE TABLE hcirecommendrecipe(id INTEGER, name TEXT, description TEXT, hci INTEGER, recipe INTEGER, quantity REAL, cmiunit INTEGER, intensity INTEGER, PRIMARY KEY(id));
        CREATE TABLE hcirecommendingredient(id INTEGER, name TEXT, description TEXT, hci INTEGER, ingredient INTEGER, quantity REAL, cmiunit INTEGER, intensity INTEGER, mealtype INTEGER, PRIMARY KEY(id));

        CREATE TABLE norm_recipe_nutrition_feature_eligibility_v1(recipe_id TEXT, recipe_name TEXT, nutrition_feature_tier TEXT, data_status TEXT, confidence TEXT, reason_codes TEXT);

        CREATE TABLE norm_synthetic_user_v1(user_id INT, age_years INT, sex TEXT, activity_level TEXT, diet_goal TEXT, data_origin TEXT, generator_version TEXT, random_seed INT);
        CREATE TABLE norm_synthetic_user_taste_v1(user_id INT, taste_id INT, preference INT, data_origin TEXT);
        CREATE TABLE norm_synthetic_user_health_goal_v1(user_id INT, hci_id INT, priority INT, is_clinical_diagnosis INT, data_origin TEXT);
        CREATE TABLE norm_synthetic_user_sport_v1(user_id INT, sport_id INT, sessions_per_week INT, minutes_per_session INT, intensity TEXT, data_origin TEXT);
        CREATE TABLE norm_synthetic_feedback_event_v1(event_id INT, user_id INT, recipe_id INT, event_type TEXT, event_time TEXT, rank_position INT, data_origin TEXT, generator_version TEXT);

        CREATE TABLE userfondnessingredient(id INTEGER, name TEXT, description TEXT, user INTEGER, ingredient INTEGER, intensity INTEGER, PRIMARY KEY(id));
        CREATE TABLE useravoidingredient(id INTEGER, name TEXT, description TEXT, user INTEGER, ingredient INTEGER, intensity INTEGER, PRIMARY KEY(id));
        CREATE TABLE useravoidrecipe(id INTEGER, name TEXT, description TEXT, user INTEGER, recipe INTEGER, intensity INTEGER, PRIMARY KEY(id));

        CREATE TABLE diseaseavoidrecipe(id INTEGER, name TEXT, description TEXT, disease INTEGER, recipe INTEGER, intensity INTEGER, PRIMARY KEY(id));
        CREATE TABLE diseaseavoidingredient(id INTEGER, name TEXT, description TEXT, disease INTEGER, ingredient INTEGER, intensity INTEGER, PRIMARY KEY(id));
        CREATE TABLE diseaserecommendrecipe(id INTEGER, name TEXT, description TEXT, disease INTEGER, recipe INTEGER, intensity INTEGER, PRIMARY KEY(id));
        CREATE TABLE diseaserecommendingredient(id INTEGER, name TEXT, description TEXT, disease INTEGER, ingredient INTEGER, intensity INTEGER, PRIMARY KEY(id));
        """
    )


def _insert_demo_rows(conn: sqlite3.Connection) -> None:
    recipes = [
        (1, "番茄鸡蛋汤", "清淡家常汤", 1, "家常菜", 1, "", 2, 1, 3, ["煮"], ["酸", "淡"], "complete", 1, "", "demo", "demo"),
        (2, "鸡胸肉蔬菜沙拉", "高蛋白轻食", 1, "轻食", 1, "", 3, 1, 4, ["拌"], ["淡"], "complete", 1, "", "demo", "demo"),
        (3, "红烧肥肉", "高脂菜", 1, "家常菜", 1, "", 1, 2, 5, ["烧"], ["咸"], "complete", 1, "", "demo", "demo"),
        (4, "燕麦牛奶粥", "早餐粥", 1, "家常菜", 1, "", 2, 0, 3, ["煮"], ["甜", "淡"], "complete", 1, "", "demo", "demo"),
        (5, "禁推示例菜", "不可推荐", 1, "测试菜", 1, "", 1, 0, 1, ["炸"], ["咸"], "complete", 0, "demo_not_recommendable", "demo", "demo"),
    ]
    conn.executemany(
        """
        INSERT INTO norm_recipe_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                recipe_id,
                name,
                description,
                cuisine_id,
                cuisine_name,
                source_id,
                summary,
                ingredient_count,
                seasoning_count,
                step_count,
                json.dumps(methods, ensure_ascii=False),
                json.dumps(tastes, ensure_ascii=False),
                content_status,
                recommendable,
                restriction_reasons,
                source_status,
                pipeline_version,
            )
            for (
                recipe_id,
                name,
                description,
                cuisine_id,
                cuisine_name,
                source_id,
                summary,
                ingredient_count,
                seasoning_count,
                step_count,
                methods,
                tastes,
                content_status,
                recommendable,
                restriction_reasons,
                source_status,
                pipeline_version,
            ) in recipes
        ],
    )

    ingredients = [
        (1, "番茄", 10, "observed"),
        (2, "鸡蛋", 20, "observed"),
        (3, "鸡胸肉", 20, "observed"),
        (4, "生菜", 30, "observed"),
        (5, "肥肉", 20, "observed"),
        (6, "燕麦", 40, "observed"),
        (7, "牛奶", 50, "observed"),
    ]
    conn.executemany(
        """
        INSERT INTO norm_ingredient_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                ingredient_id,
                name,
                "",
                "",
                foodtype_id,
                None,
                "{}",
                nutrition_status,
                "demo",
                "demo",
                "source_ingredient",
                "food",
            )
            for ingredient_id, name, foodtype_id, nutrition_status in ingredients
        ],
    )

    recipe_ingredients = [
        (1, 1, 1, 1.0, 1),
        (2, 1, 2, 1.0, 1),
        (3, 2, 3, 1.0, 1),
        (4, 2, 4, 0.5, 0),
        (5, 3, 5, 1.0, 1),
        (6, 4, 6, 1.0, 1),
        (7, 4, 7, 0.5, 0),
        (8, 5, 5, 1.0, 1),
    ]
    conn.executemany(
        """
        INSERT INTO norm_recipe_ingredient_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                row_id,
                recipe_id,
                "",
                "",
                ingredient_id,
                ingredient_id,
                ingredient_id,
                "demo",
                weight,
                "g",
                is_main,
                1.0,
                "demo",
                1,
                "",
                "demo",
            )
            for row_id, recipe_id, ingredient_id, weight, is_main in recipe_ingredients
        ],
    )

    conn.executemany("INSERT INTO ingredient2taste VALUES (?,?)", [(1, 3), (2, 7), (3, 7), (4, 7), (5, 5), (6, 1), (7, 1)])
    conn.executemany("INSERT INTO hci VALUES (?,?,?,?)", [(5, "免疫调节", "", None), (19, "增强免疫", "", 5), (9, "代谢调节", "", None)])
    conn.executemany("INSERT INTO hcirecommendrecipe VALUES (?,?,?,?,?,?,?,?)", [(1, "", "", 19, 2, None, None, 5), (2, "", "", 9, 4, None, None, 4)])
    conn.executemany("INSERT INTO hcirecommendingredient VALUES (?,?,?,?,?,?,?,?,?)", [(1, "", "", 19, 3, None, None, 5, None), (2, "", "", 9, 6, None, None, 4, None)])
    conn.executemany("INSERT INTO norm_recipe_nutrition_feature_eligibility_v1 VALUES (?,?,?,?,?,?)", [(1, "番茄鸡蛋汤", "standard", "demo", "1.0", ""), (2, "鸡胸肉蔬菜沙拉", "standard", "demo", "1.0", ""), (3, "红烧肥肉", "sensitivity_only", "demo", "0.6", ""), (4, "燕麦牛奶粥", "standard", "demo", "1.0", ""), (5, "禁推示例菜", "exclude_from_nutrition_model", "demo", "0.2", "")])

    conn.executemany("INSERT INTO norm_synthetic_user_v1 VALUES (?,?,?,?,?,?,?,?)", [(1, 30, "female", "moderate", "balanced_diet", "synthetic", "demo", 1), (2, 45, "male", "light", "fat_loss", "synthetic", "demo", 1), (3, 60, "female", "moderate", "maintain", "synthetic", "demo", 1)])
    conn.executemany("INSERT INTO norm_synthetic_user_taste_v1 VALUES (?,?,?,?)", [(1, 7, 2, "synthetic"), (1, 1, 1, "synthetic"), (1, 5, -2, "synthetic"), (2, 7, 2, "synthetic"), (3, 1, 2, "synthetic")])
    conn.executemany("INSERT INTO norm_synthetic_user_health_goal_v1 VALUES (?,?,?,?,?)", [(1, 5, 1, 0, "synthetic"), (2, 9, 1, 0, "synthetic"), (3, 5, 2, 0, "synthetic")])
    conn.executemany("INSERT INTO norm_synthetic_user_sport_v1 VALUES (?,?,?,?,?,?)", [(1, 1, 3, 40, "moderate", "synthetic"), (2, 1, 2, 30, "light", "synthetic")])
    conn.executemany("INSERT INTO userfondnessingredient VALUES (?,?,?,?,?,?)", [(1, "", "", 1, 2, 5), (2, "", "", 2, 3, 5), (3, "", "", 1000001, 1, 3)])
    conn.executemany("INSERT INTO useravoidingredient VALUES (?,?,?,?,?,?)", [(1, "", "", 1, 5, 5)])
    conn.executemany("INSERT INTO useravoidrecipe VALUES (?,?,?,?,?,?)", [(1, "", "", 2, 3, 5)])
    conn.executemany(
        "INSERT INTO norm_synthetic_feedback_event_v1 VALUES (?,?,?,?,?,?,?,?)",
        [
            (1, 1, 1, "impression", "2026-01-01 08:00:00", 1, "synthetic", "demo"),
            (2, 1, 1, "click", "2026-01-01 08:01:00", 1, "synthetic", "demo"),
            (3, 1, 2, "save", "2026-01-02 08:00:00", 2, "synthetic", "demo"),
            (4, 1, 3, "dislike", "2026-01-03 08:00:00", 3, "synthetic", "demo"),
            (5, 2, 2, "cook", "2026-01-04 08:00:00", 1, "synthetic", "demo"),
            (6, 2, 4, "click", "2026-06-15 08:00:00", 1, "synthetic", "demo"),
            (7, 3, 4, "save", "2026-06-16 08:00:00", 1, "synthetic", "demo"),
        ],
    )
    conn.executemany("INSERT INTO diseaseavoidrecipe VALUES (?,?,?,?,?,?)", [(1, "", "", 100, 3, 5)])
    conn.executemany("INSERT INTO diseaseavoidingredient VALUES (?,?,?,?,?,?)", [(1, "", "", 100, 5, 5)])
    conn.executemany("INSERT INTO diseaserecommendrecipe VALUES (?,?,?,?,?,?)", [(1, "", "", 100, 4, 4)])
    conn.executemany("INSERT INTO diseaserecommendingredient VALUES (?,?,?,?,?,?)", [(1, "", "", 100, 6, 4)])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a tiny DACH-LLMRec demo SQLite database.")
    parser.add_argument("--output", default="data/demo.sqlite")
    args = parser.parse_args(argv)
    path = create_demo_database(args.output)
    print(json.dumps({"output": str(path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
