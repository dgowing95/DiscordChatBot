"""classes/card_kit.py: the bot's cards, pure. How a long card is split over
messages, the markdown safety of text the bot did not write, and how a card
reads back in a prompt."""
import pytest

from classes import card_kit as K


def _texts(blocks):
    return [b.content for b in blocks if isinstance(b, K.Text)]


def _chars(blocks):
    return sum(len(t) for t in _texts(blocks))


def test_a_short_card_is_one_message_label_title_body_footer():
    spec = K.CardSpec(label="🔎 Searching the web", title="Keyboards", body=[K.Text("body")],
                      footer="a hint")
    [blocks] = K.split(spec)
    assert _texts(blocks) == ["-# 🔎 Searching the web", "### Keyboards", "body", "-# a hint"]


def test_a_panel_has_a_bigger_title():
    [blocks] = K.split(K.CardSpec(title="What I can do", heading=2))
    assert _texts(blocks) == ["## What I can do"]


def test_a_long_card_splits_with_the_header_once_and_the_footer_last_and_nothing_cut():
    paragraphs = [f"para {i} " + "word " * 200 for i in range(10)]
    spec = K.CardSpec(label="📘 Skills", title="All of them", body=[K.Text("\n\n".join(paragraphs))],
                      footer="the end")
    cards = K.split(spec)
    assert len(cards) > 1
    assert all(_chars(c) <= K.MAX_TEXT for c in cards)
    assert _texts(cards[0])[:2] == ["-# 📘 Skills", "### All of them"]
    assert all("-# 📘 Skills" not in _texts(c) for c in cards[1:])
    assert _texts(cards[-1])[-1] == "-# the end"
    assert all("-# the end" not in _texts(c) for c in cards[:-1])
    joined = "\n".join(t for c in cards for t in _texts(c))
    assert all(p.strip() in joined for p in paragraphs)


def test_emoji_count_twice_as_discord_counts_them_and_a_card_keeps_room_to_spare():
    # Discord counts UTF-16 units: an emoji is two. A card filled to 4000 by
    # Python's count was refused ("exceeds maximum size of 4000"), checked live.
    spec = K.CardSpec(label="🧠 Working", body=K.listing("✅ 🔎 🌐 " * 40 for _ in range(30)))
    cards = K.split(spec)
    for c in cards:
        assert sum(K.units(t) for t in _texts(c)) <= K.TEXT_BUDGET < K.MAX_TEXT
    assert K.units("🧠") == 2 and K.units("a") == 1


def test_a_cut_never_leaves_a_code_block_open():
    code = "```sh\n" + "\n".join(f"echo line {i}" for i in range(600)) + "\n```"
    cards = K.split(K.CardSpec(body=[K.Text(code)]))
    assert len(cards) > 1
    for c in cards:
        for text in _texts(c):
            assert text.count("```") % 2 == 0
    assert _texts(cards[1])[0].startswith("```sh\n")
    joined = "\n".join(t for c in cards for t in _texts(c))
    assert "echo line 0\n" in joined and "echo line 599" in joined


def test_many_items_split_on_the_component_limit_and_gaps_never_start_a_card():
    spec = K.CardSpec(label="📘 Skills", body=K.listing(f"item {i}" for i in range(60)))
    cards = K.split(spec)
    assert len(cards) > 1
    for c in cards:
        assert 1 + sum(K._components(b) for b in c) <= K.MAX_COMPONENTS
        assert not isinstance(c[0], (K.Gap, K.Divider)) and not isinstance(c[-1], (K.Gap, K.Divider))
    assert sum(len(_texts(c)) for c in cards) == 61  # 60 items and the label


def test_buttons_stay_with_the_end_of_a_long_card():
    spec = K.CardSpec(body=[K.Text("word " * 1500), K.Buttons((K.Link("Open", "https://x.example"),))])
    cards = K.split(spec)
    assert any(isinstance(b, K.Buttons) for b in cards[-1])
    assert not any(isinstance(b, K.Buttons) for c in cards[:-1] for b in c)


def test_an_empty_card_still_has_something_to_send():
    assert K.split(K.CardSpec()) == [[K.Text("​")]]


@pytest.mark.parametrize("text, safe", [
    ("# Big", "\\# Big"),
    ("-# small", "\\-# small"),
    ("> quoted", "\\> quoted"),
    ("plain\n## two", "plain\n\\## two"),
    ("no change", "no change"),
])
def test_text_the_bot_did_not_write_never_turns_into_a_heading_or_a_quote(text, safe):
    assert K.safe(text) == safe


def test_quote_quotes_every_line():
    assert K.quote("one\n# two\n\nthree") == "> one\n> \\# two\n>\n> three"


def test_bold_and_small():
    assert K.bold("a **b**") == "**a \\*\\*b\\*\\***"
    assert K.small("one\ntwo") == "-# one\n-# two"


def test_finished_takes_the_buttons_off_and_greys_it():
    spec = K.CardSpec(label="🧵 Plan", body=[K.Text("x"), K.Divider(),
                                             K.Buttons((K.Link("a", "https://a.example"),))],
                      colour=K.ASK, footer="answer soon")
    done = K.finished(spec, "closed")
    assert done.colour == K.CLOSED and done.footer == "closed" and done.label == "🧵 Plan"
    assert done.body == [K.Text("x")]


def test_the_kind_marker_round_trips():
    for kind, cid in K.KIND_IDS.items():
        assert K.kind_of_id(cid) == kind
    assert K.kind_of_id(3) is None and K.kind_of_id(None) is None


def test_a_notice_reads_as_a_status_card_without_its_decoration():
    text = K.notice_for_history("Bot", ["-# 🔎 Searching the web", "**keyboards**", "> a\n>\n> b"])
    assert text.startswith("[Status card posted by 'Bot', not a chat message]\n")
    assert text.endswith("🔎 Searching the web\n**keyboards**\na\nb")


def test_details_skip_empty_values():
    assert K.details([("When", "daily"), ("Next", ""), ("Where", "#general")]) == \
        "**When** daily\n**Where** #general"


def test_a_whole_card_moves_a_text_to_the_next_message_instead_of_cutting_it():
    texts = [K.Text(f"block {i}\n" + "x" * 1500) for i in range(5)]
    cut = K.split(K.CardSpec(body=list(texts)))
    whole = K.split(K.CardSpec(body=list(texts), whole=True))
    assert any(not b.content.startswith("block") for card in cut for b in card if isinstance(b, K.Text))
    assert [b.content for card in whole for b in card] == [t.content for t in texts]
    # One that cannot fit any message is still cut, never dropped.
    huge = K.split(K.CardSpec(body=[K.Text("y" * 9000)], whole=True))
    assert "".join(b.content for card in huge for b in card).count("y") == 9000


def test_a_long_text_of_a_notice_is_cut_in_the_history_not_on_the_card():
    request = "word " * 600  # 3000 characters, pasted by someone
    text = K.notice_for_history("bot", ["-# 🧠 Working on it", K.quote(request), "short note"])
    assert len(text) < K.HISTORY_TEXT_CHARS + 200 and "…" in text and "short note" in text
