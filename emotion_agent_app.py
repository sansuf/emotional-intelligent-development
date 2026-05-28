from __future__ import annotations

import json
import os
import re
import hashlib
import hmac
import secrets
import sqlite3
import uuid
from collections import Counter
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "emotion_agent.db"
HOST = "127.0.0.1"
PORT = 8501
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

NEGATIVE_EMOTIONS = {"sadness", "anxiety", "anger", "frustration"}
TRIGGER_INTENSITY = 6
STRATEGY_WINDOW = 3


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with db_connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                intervention_style TEXT NOT NULL DEFAULT 'empathetic',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                dominant_emotion TEXT,
                dominant_social_state TEXT,
                mean_intensity REAL,
                summary TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT,
                confidence REAL,
                intensity INTEGER,
                valence TEXT,
                social_state TEXT,
                strategy TEXT,
                is_intervention INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS dialogue_slots (
                session_id TEXT PRIMARY KEY,
                recent_status TEXT,
                pressure_source TEXT,
                sleep_energy TEXT,
                key_people TEXT,
                support_level TEXT,
                conflict_signal TEXT,
                missing_info TEXT,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );
            """
        )
        user_cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "password_hash" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return f"pbkdf2_sha256${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str | None) -> bool:
    if not stored_hash:
        return False
    try:
        algorithm, salt, expected = stored_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000)
    return hmac.compare_digest(digest.hex(), expected)


def register_user(username: str, password: str, style: str) -> dict:
    username = (username or "").strip()[:40]
    style = style if style in {"empathetic", "rational", "light", "positive"} else "empathetic"
    if len(username) < 3:
        raise ValueError("Username must contain at least 3 characters.")
    if len(password or "") < 6:
        raise ValueError("Password must contain at least 6 characters.")
    with db_connect() as conn:
        existing = conn.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            if existing["password_hash"]:
                raise ValueError("Username already exists.")
            conn.execute(
                "UPDATE users SET password_hash = ?, intervention_style = ? WHERE id = ?",
                (hash_password(password), style, existing["id"]),
            )
            get_active_session(existing["id"])
            return {"user_id": existing["id"], "username": username, "style": style}
        user_id = f"user_{uuid.uuid4().hex[:10]}"
        conn.execute(
            """
            INSERT INTO users (id, username, password_hash, intervention_style, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, username, hash_password(password), style, now_iso()),
        )
    get_active_session(user_id)
    return {"user_id": user_id, "username": username, "style": style}


def login_user(username: str, password: str, style: str) -> dict:
    username = (username or "").strip()[:40]
    style = style if style in {"empathetic", "rational", "light", "positive"} else "empathetic"
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row or not verify_password(password or "", row["password_hash"]):
            raise ValueError("Invalid username or password.")
        conn.execute("UPDATE users SET intervention_style = ? WHERE id = ?", (style, row["id"]))
    get_active_session(row["id"])
    return {"user_id": row["id"], "username": username, "style": style}


def require_authenticated_user(payload: dict) -> str:
    username = (payload.get("username") or "").strip()[:40]
    password = payload.get("password") or ""
    if not username or not password:
        raise ValueError("Please login first.")
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            raise ValueError("Invalid username or password.")
        return row["id"]


def get_or_create_user(username: str, style: str) -> str:
    username = (username or "demo_user").strip()[:40] or "demo_user"
    style = style if style in {"empathetic", "rational", "light", "positive"} else "empathetic"
    with db_connect() as conn:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if row:
            conn.execute("UPDATE users SET intervention_style = ? WHERE id = ?", (style, row["id"]))
            return row["id"]
        user_id = f"user_{uuid.uuid4().hex[:10]}"
        conn.execute(
            "INSERT INTO users (id, username, password_hash, intervention_style, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, username, None, style, now_iso()),
        )
        return user_id


def get_active_session(user_id: str) -> str:
    with db_connect() as conn:
        row = conn.execute(
            "SELECT id FROM sessions WHERE user_id = ? AND ended_at IS NULL ORDER BY started_at DESC LIMIT 1",
            (user_id,),
        ).fetchone()
        if row:
            return row["id"]
        session_id = f"sess_{uuid.uuid4().hex[:10]}"
        conn.execute(
            "INSERT INTO sessions (id, user_id, started_at) VALUES (?, ?, ?)",
            (session_id, user_id, now_iso()),
        )
        conn.execute(
            "INSERT OR REPLACE INTO dialogue_slots (session_id, missing_info, updated_at) VALUES (?, ?, ?)",
            (session_id, json.dumps(["recent status", "available support", "sleep or energy"]), now_iso()),
        )
        return session_id


def reset_session(username: str, style: str, password: str | None = None) -> str:
    if password:
        user_id = require_authenticated_user({"username": username, "password": password})
    else:
        user_id = get_or_create_user(username, style)
    with db_connect() as conn:
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE user_id = ? AND ended_at IS NULL",
            (now_iso(), user_id),
        )
    return get_active_session(user_id)


def contains_any(text: str, words: list[str]) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in words)


def is_greeting_only(text: str) -> bool:
    lowered = text.strip().lower()
    greetings = {
        "hi",
        "hello",
        "hey",
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "hello!",
        "hi!",
        "你好。",
        "你好！",
    }
    return lowered in greetings


def means_no_major_issue(text: str) -> bool:
    return contains_any(
        text,
        [
            "nothing",
            "nothing much",
            "not much",
            "no problem",
            "fine",
            "okay",
            "ok",
            "没什么",
            "没啥",
            "还好",
            "还行",
            "正常",
            "没有压力",
            "没压力",
            "没什么事情",
            "没什么事",
        ],
    )


def means_no_available_support(text: str) -> bool:
    lowered = text.strip().lower()
    exact_no = {"没有", "没有。", "没有！", "no", "no.", "no!"}
    if lowered in exact_no:
        return True
    return contains_any(
        text,
        [
            "no one",
            "nobody",
            "do not know who",
            "don't know who",
            "no such person",
            "no support",
            "alone",
            "没有这样的人",
            "没有这种人",
            "没人",
            "没有人",
            "不知道问谁",
            "不知道可以问谁",
            "没有可以问的人",
            "没有可以讨论的人",
            "没人可以讨论",
            "没人能帮",
            "只能自己",
            "一个人",
        ],
    )


def detect_emotion(text: str) -> tuple[str, float, int, str]:
    lowered = text.lower()
    if is_greeting_only(text):
        return "neutral", 0.8, 1, "neutral"
    if means_no_available_support(text):
        return "sadness", 0.76, 5, "negative"
    if means_no_major_issue(text):
        return "neutral", 0.78, 2, "neutral"
    rules = [
        ("joy", ["happy", "excited", "proud", "great", "helpful", "finished", "完成", "开心", "高兴", "兴奋", "顺利", "有帮助"], 0.92, 8, "positive"),
        ("anxiety", ["worry", "worried", "anxious", "nervous", "presentation", "can't sleep", "cannot sleep", "焦虑", "担心", "紧张", "睡不着", "汇报"], 0.90, 7, "negative"),
        ("frustration", ["stuck", "bug", "nothing works", "failed", "fix", "frustrated", "卡住", "bug", "失败", "改不好", "没用"], 0.88, 7, "negative"),
        ("anger", ["angry", "mad", "unfair", "annoyed", "生气", "愤怒", "不公平"], 0.86, 7, "negative"),
        ("sadness", ["sad", "lonely", "hopeless", "disappointed", "cry", "难过", "失落", "孤独", "没人", "沮丧"], 0.87, 7, "negative"),
        ("surprise", ["surprised", "unexpected", "suddenly", "没想到", "突然", "惊讶"], 0.80, 5, "neutral"),
    ]
    for emotion, words, confidence, intensity, valence in rules:
        if contains_any(lowered, words):
            if re.search(r"\bvery\b|really|so |太|很|特别|一直|always", lowered):
                intensity = min(10, intensity + 1)
            return emotion, confidence, intensity, valence
    return "neutral", 0.72, 3, "neutral"


def deepseek_chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 500) -> str | None:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return None
    body = json.dumps(
        {
            "model": DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = Request(
        DEEPSEEK_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=20) as res:
            payload = json.loads(res.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"DeepSeek call failed; using rule fallback. Reason: {exc}")
        return None


def extract_json_object(text: str) -> dict | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def update_dialogue_slots(session_id: str, text: str) -> dict:
    lowered = text.lower()
    with db_connect() as conn:
        row = conn.execute("SELECT * FROM dialogue_slots WHERE session_id = ?", (session_id,)).fetchone()
        slots = dict(row) if row else {"session_id": session_id}

        if means_no_major_issue(text):
            slots["recent_status"] = text[:240]
            slots["pressure_source"] = "none reported"
        if means_no_available_support(text):
            slots["support_level"] = "weak"
            slots["key_people"] = "none reported"
        if contains_any(lowered, ["project", "homework", "presentation", "exam", "work", "任务", "项目", "作业", "考试", "汇报", "学习"]):
            slots["recent_status"] = text[:240]
        if contains_any(lowered, ["pressure", "deadline", "worry", "stuck", "bug", "stress", "压力", "截止", "焦虑", "卡住", "困难"]):
            slots["pressure_source"] = text[:240]
        if contains_any(lowered, ["sleep", "tired", "energy", "睡", "累", "精力", "疲惫"]):
            slots["sleep_energy"] = text[:240]
        if contains_any(lowered, ["teammate", "friend", "teacher", "family", "classmate", "队友", "朋友", "老师", "家人", "同学"]):
            slots["key_people"] = text[:240]
        if contains_any(lowered, ["help", "support", "ask", "said it was helpful", "帮助", "支持", "可以问", "有帮助"]):
            slots["support_level"] = "weak" if means_no_available_support(text) else "available"
        if contains_any(lowered, ["conflict", "argue", "unfair", "ignored", "冲突", "吵", "不公平", "忽视"]):
            slots["conflict_signal"] = text[:240]

        missing = []
        if not slots.get("recent_status"):
            missing.append("recent status")
        if not slots.get("pressure_source"):
            missing.append("pressure source")
        if not slots.get("sleep_energy"):
            missing.append("recent sleep or energy")
        if not slots.get("support_level") and not slots.get("key_people"):
            missing.append("available support")

        slots["missing_info"] = json.dumps(missing)
        slots["updated_at"] = now_iso()
        conn.execute(
            """
            INSERT OR REPLACE INTO dialogue_slots
            (session_id, recent_status, pressure_source, sleep_energy, key_people, support_level, conflict_signal, missing_info, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                slots.get("recent_status"),
                slots.get("pressure_source"),
                slots.get("sleep_energy"),
                slots.get("key_people"),
                slots.get("support_level"),
                slots.get("conflict_signal"),
                slots["missing_info"],
                slots["updated_at"],
            ),
        )
        return slots


def infer_social_state(slots: dict, text: str) -> str:
    lowered = text.lower()
    support_level = slots.get("support_level")
    if slots.get("conflict_signal"):
        return "conflictual"
    if means_no_available_support(text):
        return "isolated"
    if support_level == "weak":
        return "isolated"
    if support_level == "available" or slots.get("key_people"):
        return "supportive"
    return "unknown"


def rule_analyse_state(session_id: str, text: str) -> dict:
    slots = update_dialogue_slots(session_id, text)
    emotion, confidence, intensity, valence = detect_emotion(text)
    social_state = infer_social_state(slots, text)
    missing_info = json.loads(slots.get("missing_info") or "[]")
    return {
        "emotion": emotion,
        "confidence": confidence,
        "intensity": intensity,
        "valence": valence,
        "social_state": social_state,
        "missing_info": missing_info,
    }


def analyse_state(session_id: str, text: str) -> dict:
    slots = update_dialogue_slots(session_id, text)
    dialogue_context = {
        "recent_status": slots.get("recent_status"),
        "pressure_source": slots.get("pressure_source"),
        "sleep_energy": slots.get("sleep_energy"),
        "key_people": slots.get("key_people"),
        "support_level": slots.get("support_level"),
        "conflict_signal": slots.get("conflict_signal"),
        "missing_info": json.loads(slots.get("missing_info") or "[]"),
    }
    prompt = f"""
你是一个情绪与社交状态识别模块。请根据用户最新输入和半结构式对话上下文进行分析。
只返回一个合法 JSON 对象，不要输出解释文字。

JSON schema:
{{
  "emotion": "joy|sadness|anxiety|anger|frustration|neutral|surprise",
  "confidence": 0.0,
  "intensity": 0,
  "valence": "positive|neutral|negative",
  "social_state": "supportive|isolated|conflictual|unstable|unknown",
  "missing_info": ["recent status|pressure source|recent sleep or energy|available support"]
}}

规则：
- intensity 范围是 0 到 10。
- 如果信息不足，不要强行判断 social_state，使用 unknown，并在 missing_info 中写缺少的信息。
- 如果用户表达没人可问、缺少支持、孤立感，social_state 倾向 isolated。
- 如果用户提到冲突、争吵、不公平，social_state 倾向 conflictual。
- 如果用户提到队友、朋友、老师、家人提供帮助，social_state 倾向 supportive。

半结构式上下文:
{json.dumps(dialogue_context, ensure_ascii=False)}

用户最新输入:
{text}
"""
    raw = deepseek_chat(
        [
            {"role": "system", "content": "You return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=300,
    )
    parsed = extract_json_object(raw) if raw else None
    if not parsed:
        return rule_analyse_state(session_id, text)

    fallback = rule_analyse_state(session_id, text)
    emotion = str(parsed.get("emotion", fallback["emotion"]))
    if emotion not in {"joy", "sadness", "anxiety", "anger", "frustration", "neutral", "surprise"}:
        emotion = fallback["emotion"]
    valence = str(parsed.get("valence", fallback["valence"]))
    if valence not in {"positive", "neutral", "negative"}:
        valence = fallback["valence"]
    social_state = str(parsed.get("social_state", fallback["social_state"]))
    if social_state not in {"supportive", "isolated", "conflictual", "unstable", "unknown"}:
        social_state = fallback["social_state"]
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", fallback["confidence"]))))
    except (TypeError, ValueError):
        confidence = fallback["confidence"]
    try:
        intensity = max(0, min(10, int(parsed.get("intensity", fallback["intensity"]))))
    except (TypeError, ValueError):
        intensity = fallback["intensity"]
    missing_info = parsed.get("missing_info", fallback["missing_info"])
    if not isinstance(missing_info, list):
        missing_info = fallback["missing_info"]
    return {
        "emotion": emotion,
        "confidence": confidence,
        "intensity": intensity,
        "valence": valence,
        "social_state": social_state,
        "missing_info": [str(item) for item in missing_info],
    }


def recent_user_turns(session_id: str) -> list[dict]:
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT emotion, intensity, valence, social_state, strategy
            FROM messages
            WHERE session_id = ? AND role = 'user'
            ORDER BY created_at DESC
            LIMIT 8
            """,
            (session_id,),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def select_feedback_strategy(turns: list[dict], latest: dict) -> tuple[str, bool]:
    if latest["missing_info"]:
        return "continue_semi_structured_dialogue", False
    if latest["valence"] == "positive":
        return "positive_reinforcement", False
    if latest["emotion"] not in NEGATIVE_EMOTIONS:
        return "normal_reflection", False
    if latest["intensity"] < TRIGGER_INTENSITY:
        return "light_support", False

    window = (turns + [latest])[-STRATEGY_WINDOW:]
    consecutive_negative = sum(1 for turn in window if turn.get("emotion") in NEGATIVE_EMOTIONS)
    if consecutive_negative >= STRATEGY_WINDOW and latest["social_state"] in {"isolated", "conflictual"}:
        return "escalated_intervention", True
    return "standard_intervention", True


def next_question(missing_info: list[str]) -> str:
    if "recent status" in missing_info:
        return "最近你主要在忙什么？有没有一件让你压力比较大的事情？"
    if "pressure source" in missing_info:
        return "你觉得现在最大的压力来源是什么？是任务本身、时间、还是和别人协作有关？"
    if "recent sleep or energy" in missing_info:
        return "这几天你的睡眠和精力怎么样？这种状态持续多久了？"
    if "available support" in missing_info:
        return "你身边有没有可以讨论这件事的人，比如同学、队友、朋友或老师？"
    return "你愿意再具体说说最近发生了什么吗？"


def build_reply(state: dict, strategy: str, style: str) -> str:
    style_names = {
        "empathetic": "共情型",
        "rational": "理性型",
        "light": "轻松型",
        "positive": "积极型",
    }
    deepseek_prompt = f"""
你是一个校园/学习场景中的情绪支持智能体。请根据识别结果生成一段中文回复。

要求：
- 回复要简短，2 到 4 句。
- 不要进行医学诊断。
- 如果策略是 escalated_intervention，可以温和建议联系可信任的人、老师、辅导员或专业支持。
- 如果策略是 continue_semi_structured_dialogue，优先继续追问缺失信息。
- 语气风格：{style_names.get(style, "共情型")}

识别结果:
{json.dumps(state, ensure_ascii=False)}

策略:
{strategy}
"""
    generated = deepseek_chat(
        [
            {"role": "system", "content": "你是安全、克制、支持性的情绪反馈助手。"},
            {"role": "user", "content": deepseek_prompt},
        ],
        temperature=0.5,
        max_tokens=260,
    )
    if generated:
        return generated.strip()

    if strategy == "continue_semi_structured_dialogue":
        return f"好的，我先继续了解一点背景。{next_question(state['missing_info'])}"
    if strategy == "positive_reinforcement":
        return "听起来你最近有一个不错的进展，而且身边也有正向反馈。可以把这次做得好的地方记下来，后面遇到压力时它会成为一个很有用的参考。"
    if strategy == "normal_reflection":
        return "我理解了。你现在的状态看起来比较平稳。我们可以继续聊聊最近的任务、人际支持，或者你想改善的一件小事。"
    if strategy == "light_support":
        return "这听起来有些消耗你。可以先把问题拆成一个很小的下一步，比如只整理目前卡住的点，或者先找一个人确认思路。"
    if strategy == "standard_intervention":
        if style == "rational":
            return "我们先把问题拆开：现在最影响你的是任务难度、时间压力，还是缺少支持？先选一个最可控的小步骤，会比一次解决全部问题更容易。"
        if style == "light":
            return "这个状态确实不轻松。先别急着和整个问题硬碰硬，可以先挑一个最小的点处理，让大脑有一点重新启动的空间。"
        return "听起来你现在的压力已经比较明显了，而且这件事让你有些孤立或卡住。你的感受是合理的。我们可以先找一个能立刻减轻负担的小动作，比如联系队友确认一个问题，或者把困难写成三条。"
    return "你已经连续表达了比较强的负面感受，而且社交支持似乎偏弱。这个时候不建议一个人硬撑。可以先联系一个可信任的人，例如队友、朋友、老师或辅导员。如果这种状态持续或影响睡眠，也可以考虑寻求专业支持。"


def save_message(session_id: str, role: str, content: str, state: dict | None, strategy: str, is_intervention: bool) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO messages
            (id, session_id, role, content, emotion, confidence, intensity, valence, social_state, strategy, is_intervention, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"msg_{uuid.uuid4().hex[:12]}",
                session_id,
                role,
                content,
                state.get("emotion") if state else None,
                state.get("confidence") if state else None,
                state.get("intensity") if state else None,
                state.get("valence") if state else None,
                state.get("social_state") if state else None,
                strategy,
                1 if is_intervention else 0,
                now_iso(),
            ),
        )


def aggregate_session(session_id: str) -> dict:
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
    messages = [dict(row) for row in rows]
    user_turns = [m for m in messages if m["role"] == "user" and m["emotion"]]
    if not user_turns:
        return {
            "dominant_emotion": "neutral",
            "dominant_social_state": "unknown",
            "mean_intensity": 0,
            "turns": len(messages),
            "interventions": 0,
            "summary": "还没有足够的对话记录。",
        }
    dominant_emotion = Counter(m["emotion"] for m in user_turns).most_common(1)[0][0]
    dominant_social_state = Counter(m["social_state"] or "unknown" for m in user_turns).most_common(1)[0][0]
    mean_intensity = round(sum(m["intensity"] or 0 for m in user_turns) / len(user_turns), 2)
    interventions = sum(1 for m in messages if m["is_intervention"])
    summary = (
        f"本轮会话中，主要情绪为 {dominant_emotion}，主要社交状态为 {dominant_social_state}，"
        f"平均情绪强度为 {mean_intensity}/10，共触发 {interventions} 次干预。"
    )
    with db_connect() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET dominant_emotion = ?, dominant_social_state = ?, mean_intensity = ?, summary = ?
            WHERE id = ?
            """,
            (dominant_emotion, dominant_social_state, mean_intensity, summary, session_id),
        )
    return {
        "dominant_emotion": dominant_emotion,
        "dominant_social_state": dominant_social_state,
        "mean_intensity": mean_intensity,
        "turns": len(messages),
        "interventions": interventions,
        "summary": summary,
    }


def handle_user_message(payload: dict) -> dict:
    username = payload.get("username", "demo_user")
    style = payload.get("style", "empathetic")
    content = (payload.get("message") or "").strip()
    if not content:
        raise ValueError("Message cannot be empty.")

    user_id = require_authenticated_user(payload)
    session_id = get_active_session(user_id)
    if is_greeting_only(content):
        state = {
            "emotion": "neutral",
            "confidence": 0.8,
            "intensity": 1,
            "valence": "neutral",
            "social_state": "unknown",
            "missing_info": ["recent status", "pressure source", "recent sleep or energy", "available support"],
        }
        strategy = "continue_semi_structured_dialogue"
        reply = "你好，我会先通过几个问题了解你的近况和社交支持情况。最近你主要在忙什么？有没有让你感觉有压力的事情？"
        save_message(session_id, "user", content, state, strategy, False)
        save_message(session_id, "assistant", reply, state, strategy, False)
        summary = aggregate_session(session_id)
        return {
            "session_id": session_id,
            "state": state,
            "strategy": strategy,
            "is_intervention": False,
            "reply": reply,
            "summary": summary,
        }
    state = analyse_state(session_id, content)
    turns = recent_user_turns(session_id)
    strategy, is_intervention = select_feedback_strategy(turns, state)
    save_message(session_id, "user", content, state, strategy, is_intervention)
    reply = build_reply(state, strategy, style)
    save_message(session_id, "assistant", reply, state, strategy, is_intervention)
    summary = aggregate_session(session_id)
    return {
        "session_id": session_id,
        "state": state,
        "strategy": strategy,
        "is_intervention": is_intervention,
        "reply": reply,
        "summary": summary,
    }


def dashboard(username: str, style: str, password: str | None = None) -> dict:
    if password:
        user_id = require_authenticated_user({"username": username, "password": password})
    else:
        user_id = get_or_create_user(username, style)
    session_id = get_active_session(user_id)
    with db_connect() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        ).fetchall()
        history_rows = conn.execute(
            """
            SELECT m.*, s.started_at AS session_started_at
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE s.user_id = ? AND m.role = 'user'
            ORDER BY m.created_at ASC
            """,
            (user_id,),
        ).fetchall()
        session_rows = conn.execute(
            """
            SELECT id, started_at, ended_at, dominant_emotion, dominant_social_state, mean_intensity, summary
            FROM sessions
            WHERE user_id = ?
            ORDER BY started_at DESC
            LIMIT 8
            """,
            (user_id,),
        ).fetchall()
        slots = conn.execute(
            "SELECT * FROM dialogue_slots WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    history = [dict(row) for row in history_rows]
    emotion_counts = Counter(row["emotion"] or "unknown" for row in history)
    social_counts = Counter(row["social_state"] or "unknown" for row in history)
    trend = [
        {
            "created_at": row["created_at"],
            "emotion": row["emotion"],
            "intensity": row["intensity"] or 0,
            "social_state": row["social_state"],
            "strategy": row["strategy"],
        }
        for row in history
    ]
    return {
        "session_id": session_id,
        "messages": [dict(row) for row in rows],
        "history": trend[-50:],
        "emotion_counts": dict(emotion_counts),
        "social_counts": dict(social_counts),
        "sessions": [dict(row) for row in session_rows],
        "slots": dict(slots) if slots else {},
        "summary": aggregate_session(session_id),
    }


INDEX_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Emotion Intervention Agent</title>
  <style>
    :root {
      --bg: #f6f7f2;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #66706a;
      --line: #d9ddd3;
      --green: #2f6f5e;
      --blue: #315f8c;
      --red: #a74444;
      --gold: #a06a1b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #fffdf8;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 { font-size: 20px; margin: 0; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
    main {
      display: grid;
      grid-template-columns: minmax(360px, 1.2fr) minmax(320px, 0.8fr);
      gap: 18px;
      padding: 18px;
      max-width: 1280px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
    }
    .controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    input, select, textarea, button {
      font: inherit;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: white;
      color: var(--ink);
    }
    input, select { height: 34px; padding: 0 10px; }
    button {
      height: 34px;
      padding: 0 12px;
      background: var(--green);
      color: white;
      border-color: var(--green);
      cursor: pointer;
    }
    button.secondary { background: white; color: var(--green); }
    .auth-status {
      min-width: 120px;
      color: var(--muted);
      font-size: 12px;
    }
    .chat {
      display: flex;
      flex-direction: column;
      min-height: calc(100vh - 110px);
    }
    .chat-log {
      padding: 16px;
      overflow-y: auto;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      max-width: 82%;
      line-height: 1.45;
    }
    .user { align-self: flex-end; background: #eef4ff; border-color: #c9d8ef; }
    .assistant { align-self: flex-start; background: #fffdf8; }
    .intervention { border-color: #e3b45d; background: #fff7e8; }
    .meta { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .composer {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      padding: 14px;
      border-top: 1px solid var(--line);
    }
    textarea {
      min-height: 52px;
      resize: vertical;
      padding: 10px;
    }
    .side { padding: 14px; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      min-height: 70px;
    }
    .metric b { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; }
    .metric span { font-size: 20px; }
    .json, .slots {
      background: #f7f8f4;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      overflow-x: auto;
      white-space: pre-wrap;
      font-family: Consolas, monospace;
      font-size: 12px;
    }
    .chart {
      border: 1px solid var(--line);
      border-radius: 8px;
      height: 170px;
      margin: 12px 0;
      padding: 10px;
    }
    .bars { display: grid; gap: 8px; margin: 12px 0; }
    .bar-row {
      display: grid;
      grid-template-columns: 92px 1fr 32px;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }
    .bar-track {
      height: 10px;
      border-radius: 999px;
      background: #eef0ea;
      overflow: hidden;
    }
    .bar-fill { height: 100%; background: var(--green); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-top: 8px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 6px 4px;
      text-align: left;
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 600; }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; padding: 10px; }
      .chat { min-height: 70vh; }
      header { align-items: flex-start; flex-direction: column; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>情绪识别与情绪干预智能体</h1>
      <div class="sub">半结构式对话 · 情绪/社交状态识别 · 策略选择 · 情绪历史看板</div>
    </div>
    <div class="controls">
      <input id="username" value="demo_user" aria-label="username" />
      <input id="password" type="password" value="demo123" aria-label="password" />
      <select id="style" aria-label="style">
        <option value="empathetic">共情型</option>
        <option value="rational">理性型</option>
        <option value="light">轻松型</option>
        <option value="positive">积极型</option>
      </select>
      <button onclick="register()">注册</button>
      <button onclick="login()">登录</button>
      <button class="secondary" onclick="logout()">退出</button>
      <button class="secondary" onclick="resetSession()">新会话</button>
      <span id="authStatus" class="auth-status">未登录</span>
    </div>
  </header>
  <main>
    <section class="chat">
      <div id="chatLog" class="chat-log"></div>
      <div class="composer">
        <textarea id="message" placeholder="输入一句近况，例如：I have a presentation tomorrow and I keep worrying..."></textarea>
        <button onclick="sendMessage()">发送</button>
      </div>
    </section>
    <section class="side">
      <h2>Dashboard</h2>
      <div class="metric-grid">
        <div class="metric"><b>主要情绪</b><span id="dominantEmotion">neutral</span></div>
        <div class="metric"><b>社交状态</b><span id="socialState">unknown</span></div>
        <div class="metric"><b>平均强度</b><span id="meanIntensity">0</span></div>
        <div class="metric"><b>干预次数</b><span id="interventions">0</span></div>
      </div>
      <div class="chart"><svg id="chart" width="100%" height="150"></svg></div>
      <h3>情绪历史统计</h3>
      <div id="emotionBars" class="bars"></div>
      <h3>社交状态统计</h3>
      <div id="socialBars" class="bars"></div>
      <h3>历史记录</h3>
      <table>
        <thead><tr><th>时间</th><th>情绪</th><th>强度</th><th>社交状态</th><th>策略</th></tr></thead>
        <tbody id="historyRows"></tbody>
      </table>
      <h3>最新识别结果</h3>
      <pre id="latestState" class="json">{}</pre>
      <h3>半结构式信息槽</h3>
      <pre id="slots" class="slots">{}</pre>
      <h3>会话摘要</h3>
      <p id="summary" class="sub">还没有足够的对话记录。</p>
    </section>
  </main>
  <script>
    const chatLog = document.getElementById("chatLog");
    const messageBox = document.getElementById("message");
    let latestState = {};
    let loggedIn = false;

    function settings() {
      return {
        username: document.getElementById("username").value || "demo_user",
        password: document.getElementById("password").value || "",
        style: document.getElementById("style").value
      };
    }

    async function api(path, body) {
      const res = await fetch(path, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({...settings(), ...body})
      });
      if (!res.ok) {
        const text = await res.text();
        try {
          const payload = JSON.parse(text);
          throw new Error(payload.error || text);
        } catch {
          throw new Error(text);
        }
      }
      return await res.json();
    }

    function setAuthStatus(text) {
      document.getElementById("authStatus").textContent = text;
    }

    async function register() {
      try {
        await api("/api/register", {});
        loggedIn = true;
        setAuthStatus("已注册并登录");
        chatLog.innerHTML = "";
        addMessage("assistant", "注册成功。我会先通过几个问题了解你的近况和社交支持情况。最近你主要在忙什么？", "start", false);
        await loadDashboard();
      } catch (error) {
        setAuthStatus(error.message);
      }
    }

    async function login() {
      try {
        await api("/api/login", {});
        loggedIn = true;
        setAuthStatus("已登录");
        chatLog.innerHTML = "";
        await loadDashboard();
        addMessage("assistant", "欢迎回来。你可以继续描述最近的状态，右侧会显示你的情绪历史和统计。", "login", false);
      } catch (error) {
        setAuthStatus(error.message);
      }
    }

    function logout() {
      loggedIn = false;
      latestState = {};
      chatLog.innerHTML = "";
      setAuthStatus("未登录");
      document.getElementById("latestState").textContent = "{}";
    }

    function addMessage(role, content, strategy, isIntervention) {
      const node = document.createElement("div");
      node.className = `msg ${role} ${isIntervention ? "intervention" : ""}`;
      const label = role === "user" ? "User" : "Agent";
      node.innerHTML = `<div class="meta">${label}${strategy ? " · " + strategy : ""}</div><div></div>`;
      node.lastChild.textContent = content;
      chatLog.appendChild(node);
      chatLog.scrollTop = chatLog.scrollHeight;
    }

    async function sendMessage() {
      if (!loggedIn) {
        setAuthStatus("请先注册或登录");
        return;
      }
      const text = messageBox.value.trim();
      if (!text) return;
      messageBox.value = "";
      addMessage("user", text, "", false);
      try {
        const result = await api("/api/message", {message: text});
        latestState = result.state;
        addMessage("assistant", result.reply, result.strategy, result.is_intervention);
        await loadDashboard();
      } catch (error) {
        setAuthStatus(error.message);
      }
    }

    async function resetSession() {
      if (!loggedIn) {
        setAuthStatus("请先注册或登录");
        return;
      }
      chatLog.innerHTML = "";
      latestState = {};
      await api("/api/reset", {});
      await loadDashboard();
      addMessage("assistant", "你好，我会通过几个问题了解你的近况和社交支持情况。最近你主要在忙什么？", "start", false);
    }

    async function loadDashboard() {
      const result = await api("/api/dashboard", {});
      document.getElementById("dominantEmotion").textContent = result.summary.dominant_emotion;
      document.getElementById("socialState").textContent = result.summary.dominant_social_state;
      document.getElementById("meanIntensity").textContent = result.summary.mean_intensity;
      document.getElementById("interventions").textContent = result.summary.interventions;
      document.getElementById("summary").textContent = result.summary.summary;
      document.getElementById("latestState").textContent = JSON.stringify(latestState, null, 2);
      document.getElementById("slots").textContent = JSON.stringify(result.slots, null, 2);
      drawChart(result.history || result.messages.filter(m => m.role === "user"));
      renderBars("emotionBars", result.emotion_counts || {});
      renderBars("socialBars", result.social_counts || {});
      renderHistory(result.history || []);
    }

    function renderBars(targetId, counts) {
      const node = document.getElementById(targetId);
      node.innerHTML = "";
      const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
      const total = entries.reduce((sum, item) => sum + item[1], 0) || 1;
      if (!entries.length) {
        node.innerHTML = `<div class="sub">暂无历史数据</div>`;
        return;
      }
      entries.forEach(([label, count]) => {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `<span>${label}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.round(count * 100 / total)}%"></div></div><span>${count}</span>`;
        node.appendChild(row);
      });
    }

    function renderHistory(history) {
      const tbody = document.getElementById("historyRows");
      tbody.innerHTML = "";
      history.slice(-12).reverse().forEach(item => {
        const row = document.createElement("tr");
        row.innerHTML = `<td>${(item.created_at || "").replace("T", " ")}</td><td>${item.emotion || ""}</td><td>${item.intensity ?? ""}</td><td>${item.social_state || ""}</td><td>${item.strategy || ""}</td>`;
        tbody.appendChild(row);
      });
      if (!history.length) {
        tbody.innerHTML = `<tr><td colspan="5" class="sub">暂无历史记录</td></tr>`;
      }
    }

    function drawChart(points) {
      const svg = document.getElementById("chart");
      svg.innerHTML = "";
      const width = svg.clientWidth || 400;
      const height = 150;
      const pad = 18;
      const values = points.map(p => Number(p.intensity || 0));
      const coords = values.map((v, i) => {
        const x = pad + (values.length <= 1 ? 0 : i * (width - pad * 2) / (values.length - 1));
        const y = height - pad - v * (height - pad * 2) / 10;
        return [x, y];
      });
      const axis = document.createElementNS("http://www.w3.org/2000/svg", "path");
      axis.setAttribute("d", `M${pad},${pad} V${height-pad} H${width-pad}`);
      axis.setAttribute("fill", "none");
      axis.setAttribute("stroke", "#aeb5aa");
      svg.appendChild(axis);
      if (coords.length) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", coords.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" "));
        path.setAttribute("fill", "none");
        path.setAttribute("stroke", "#315f8c");
        path.setAttribute("stroke-width", "3");
        svg.appendChild(path);
        coords.forEach(([x, y], i) => {
          const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          circle.setAttribute("cx", x);
          circle.setAttribute("cy", y);
          circle.setAttribute("r", "4");
          circle.setAttribute("fill", "#2f6f5e");
          svg.appendChild(circle);
        });
      }
    }

    messageBox.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
      }
    });
    setAuthStatus("请注册或登录");
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.respond_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            path = urlparse(self.path).path
            if path == "/api/register":
                self.respond_json(register_user(payload.get("username", ""), payload.get("password", ""), payload.get("style", "empathetic")))
            elif path == "/api/login":
                self.respond_json(login_user(payload.get("username", ""), payload.get("password", ""), payload.get("style", "empathetic")))
            elif path == "/api/message":
                self.respond_json(handle_user_message(payload))
            elif path == "/api/dashboard":
                self.respond_json(dashboard(payload.get("username", "demo_user"), payload.get("style", "empathetic"), payload.get("password")))
            elif path == "/api/reset":
                session_id = reset_session(payload.get("username", "demo_user"), payload.get("style", "empathetic"), payload.get("password"))
                self.respond_json({"session_id": session_id})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.respond_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def respond_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def respond_text(self, text: str, content_type: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{now_iso()}] {fmt % args}")


def main() -> None:
    init_db()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Emotion Agent prototype running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
