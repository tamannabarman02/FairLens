import pytest

from fairlens.detection.rule_based import RuleBasedDetector
from fairlens.evaluation import facts
from fairlens.mitigation.rule_based import RuleBasedRewriter

det = RuleBasedDetector()
rw = RuleBasedRewriter()


@pytest.mark.parametrize("text,biased", [
    ("Looking for a young, aggressive salesman with 5 years of experience.", True),
    ("Women are naturally better suited to nursing.", True),
    ("Muslims are terrorists.", True),
    ("The committee approved the budget on Tuesday.", False),
    ("The nurse checked the patient's chart.", False),
])
def test_rule_detector(text, biased):
    r = det.predict(text)
    assert (r.label == "biased") == biased
    if biased:
        assert r.spans and all(s.explanation for s in r.spans)
        assert all(text[s.start:s.end] == s.text for s in r.spans)


def test_negated_generalisation_is_weaker():
    assert det.predict("Not all women are bad at math.").score < det.predict("All women are bad at math.").score


def test_rule_rewrite_keeps_requirements():
    src = "Looking for a young, aggressive salesman with 5 years of experience."
    out = rw.rewrite(src).text
    assert "salesman" not in out and "young" not in out and "5 years" in out
    assert det.predict(out).score < det.predict(src).score
    assert facts.check(src, out)["fact_retention"] == 1.0


def test_rule_rewrite_article_and_pronouns():
    assert rw.rewrite("He or she must bring his or her ID.").text == "They must bring their ID."
    assert rw.rewrite("We need a fireman.").text == "We need a firefighter."


# --- regression: tricky inputs for the fact checker --------------------------------------------
def test_negation_flip_detected():
    c = facts.check("The candidate does not need a degree.", "The candidate needs a degree.")
    assert not c["checks"]["negation"]["pass"] and c["fact_retention"] < 1


def test_number_change_detected():
    c = facts.check("Requires 5 years of experience.", "Requires 3 years of experience.")
    assert c["checks"]["numbers"]["missing"] == ["5"]


def test_number_words_equivalent():
    assert facts.check("Requires five years of experience.", "Requires 5 years of experience.")["checks"]["numbers"]["pass"]


def test_entity_and_date_retained_or_lost():
    o = "Priya Shah joined Acme Corp on 3 March 2021."
    assert facts.check(o, "Priya Shah joined Acme Corp on 3 March 2021 as staff.")["fact_retention"] == 1.0
    c = facts.check(o, "A new person joined the company.")
    assert not c["checks"]["entities"]["pass"] and not c["checks"]["dates"]["pass"]


def test_quoted_speech_must_be_verbatim():
    o = 'The manager said "girls cannot code" during the meeting.'
    c = facts.check(o, 'The manager said "some people cannot code" during the meeting.')
    assert not c["checks"]["quotes"]["pass"]


def test_gendered_role_normalised_but_role_change_caught():
    assert facts.check("The salesman called.", "The salesperson called.")["checks"]["roles"]["pass"]
    assert not facts.check("The nurse called.", "The doctor called.")["checks"]["roles"]["pass"]
