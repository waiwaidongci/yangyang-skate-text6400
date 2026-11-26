"""Deterministic rental desk records with whole historical and evaluation shifts."""

from random import Random

from .settings import COLUMNS

HISTORY = {
    "blade": (
        "右脚冰刀刃口有缺口", "左脚刀刃卷边，推冰发飘", "冰刀刃不平要重磨",
        "滑过补冰线后刀刃崩角", "鞋底冰刀刮冰有毛刺", "外刃钝，起步打滑",
        "刀刃生锈而且挂冰屑", "后跟刀刃磨平了，蹬不住", "冰刀前尖磕出缺口",
        "转向时冰刀内刃打滑", "冰刀外侧毛刺，横滑刮冰", "刀刃磨损，刹停时打飘",
    ),
    "boot": (
        "鞋内衬开胶，磨脚后跟", "左脚鞋壳开裂，脚踝被磨红", "鞋垫翻卷，脚掌顶疼",
        "鞋舌边缘缝线开了", "鞋头内衬破洞露海绵", "鞋面脱胶已经张口",
        "鞋壳接缝漏水，内衬湿透", "鞋筒内衬磨穿把袜子刮破",
        "旧鞋垫脱胶粘袜子", "鞋舌泡棉挤成一团，顶脚背",
    ),
    "fastener": (
        "鞋扣卡舌扣不住，反复跳开", "绑带卡扣断齿，拉紧就弹开",
        "冰刀底座螺丝松了，拿起来晃", "鞋扣带少了一颗铆钉",
        "上排卡扣扳手折断", "刀架铆钉脱落，底座左右晃",
        "右鞋棘轮扣勾不住绑带", "脚背卡扣缺齿，锁不紧",
        "鞋跟刀架螺丝一拧就松", "绑带固定螺丝掉在换鞋区",
    ),
}
HOLDOUT = (
    (
        ("冰刀刃口磕掉一块，侧滑时挂冰", "blade"), ("刀刃有毛刺还打飘", "blade"),
        ("外刃太钝，蹬冰总滑", "blade"), ("鞋筒内衬翻起来磨袜子", "boot"),
        ("鞋舌顶脚背，泡棉跑出来了", "boot"), ("鞋扣卡舌裂了，一扳就开", "fastener"),
        ("右刀架螺丝连着底座晃动", "fastener"), ("换了袜子右脚仍不舒服", "boot"),
    ),
    (
        ("冰刀前尖崩角，停刹吃不住冰", "blade"), ("推冰两次都飘，发现刀刃有缺口", "blade"),
        ("内刃生锈，刮下来是冰屑", "blade"), ("鞋壳接缝开裂，鞋内进水", "boot"),
        ("鞋垫翻卷还磨出小洞", "boot"), ("右侧绑带一拉就弹，棘轮扣滑掉", "fastener"),
        ("鞋跟刀架铆钉没了", "fastener"), ("冰刀外侧毛刺，卡扣好好的", "blade"),
    ),
    (
        ("左脚刀刃卷边，磨完再给柜台", "blade"), ("外刃刹停打滑，刀身见锈点", "blade"),
        ("鞋壳裂了一道，脚踝有压痕", "boot"), ("鞋面张口露出内衬", "boot"),
        ("卡扣缺齿，踩进去锁不紧了", "fastener"), ("鞋扣带的固定螺丝丢了一颗", "fastener"),
        ("底座螺丝松，但冰刀刃也有毛刺", "fastener"), ("可能磨脚，问顾客说停不下来", "blade"),
    ),
)
PENDING = (
    (
        "左脚冰刀刃口缺口，推冰打飘", "鞋垫翻卷，内衬磨脚后跟", "绑带卡扣缺齿，扣上就松",
        "鞋扣附近磨脚，有点松", "冰刀外刃打滑，刀架螺丝也松", "甲乙丙丁",
    ),
    (
        "刀刃卷边还有毛刺，麻烦重磨", "鞋壳开裂，内衬湿透了", "刀架铆钉掉了，底座左右晃",
        "这双有点不对，回头问原柜员", "鞋扣有动静，鞋舌也不舒服", "鞋底冰刀有缺口，刮冰跑偏",
    ),
)


def make_shift(prefix: str, index: int, split: str, notes: list[tuple[str, str | None]]) -> list[dict]:
    shift_id = f"SR-{prefix}{index:02d}"
    return [
        dict(zip(COLUMNS, (f"{shift_id}-{position:02d}", shift_id, split, message, label)))
        for position, (message, label) in enumerate(notes, start=1)
    ]


def demo_rows() -> list[dict]:
    rng = Random(251125)
    rows = []
    for index in range(1, 13):
        notes = []
        for label, count in (("blade", 4), ("boot", 2), ("fastener", 2)):
            for message in rng.sample(HISTORY[label], count):
                prefix = rng.choice(("", "顾客退回：", "租鞋时发现：", "交班记录："))
                notes.append((f"{prefix}{message}", label))
        rng.shuffle(notes)
        rows.extend(make_shift("H", index, "train", notes))
    for index, notes in enumerate(HOLDOUT, start=1):
        rows.extend(make_shift("V", index, "holdout", list(notes)))
    for index, notes in enumerate(PENDING, start=1):
        rows.extend(make_shift("P", index, "pending", [(note, None) for note in notes]))
    return rows
