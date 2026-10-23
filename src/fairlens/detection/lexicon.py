"""Transparent bias lexicon shared by the rule-based detector and the rule-based rewriter.

Every entry carries a category, a weight (how strongly the match signals bias on its own),
a plain-language explanation and, where one exists, a neutral replacement. Weights are hand-set
priors documented in docs/METHODOLOGY.md; they are not learned and not probabilities.
Sources for word lists: gender-coded job-ad wording (Gaucher, Friesen & Kay, 2011), age-coded
job-ad wording (Burn et al., 2019 / EEOC guidance), and media "loaded language" examples used in
BABE annotation guidelines (Spinde et al., 2021).
"""
from __future__ import annotations

GENDERED_TERMS: dict[str, str] = {
    "salesman": "salesperson", "salesmen": "salespeople", "saleswoman": "salesperson",
    "chairman": "chairperson", "chairmen": "chairpersons", "chairwoman": "chairperson",
    "businessman": "businessperson", "businessmen": "businesspeople", "businesswoman": "businessperson",
    "fireman": "firefighter", "firemen": "firefighters",
    "policeman": "police officer", "policemen": "police officers", "policewoman": "police officer",
    "mailman": "mail carrier", "mailmen": "mail carriers", "postman": "postal worker",
    "spokesman": "spokesperson", "spokeswoman": "spokesperson", "spokesmen": "spokespeople",
    "cameraman": "camera operator", "foreman": "supervisor", "workman": "worker", "workmen": "workers",
    "manpower": "workforce", "man-hours": "work hours", "mankind": "humankind", "man-made": "artificial",
    "stewardess": "flight attendant", "stewardesses": "flight attendants", "waitress": "server",
    "waitresses": "servers", "headmaster": "head teacher", "headmistress": "head teacher",
    "congressman": "member of Congress", "congresswoman": "member of Congress",
    "cleaning lady": "cleaner", "cleaning ladies": "cleaners", "lady doctor": "doctor",
    "male nurse": "nurse", "female engineer": "engineer", "female doctor": "doctor", "female pilot": "pilot",
    "housewife": "homemaker", "housewives": "homemakers", "middleman": "intermediary",
}

# Age-coded wording (job ads especially). Replacement "" means drop the word.
AGE_CODED: dict[str, str] = {
    "young": "", "youthful": "", "digital native": "digitally skilled person",
    "digital natives": "digitally skilled people", "recent graduate": "candidate",
    "recent graduates": "candidates", "fresh graduate": "candidate", "fresh graduates": "candidates",
    "too old": "unsuitable", "over the hill": "experienced", "elderly": "older",
    "senile": "forgetful", "old-timers": "long-serving staff",
}

# Masculine-coded personality words in job adverts (mild signal on their own).
GENDER_CODED_TRAITS: dict[str, str] = {
    "aggressive": "proactive", "dominant": "leading", "ninja": "expert", "rockstar": "expert",
    "manly": "strong", "bossy": "assertive", "hysterical": "upset", "shrill": "loud",
}

LOADED_LANGUAGE: dict[str, str] = {
    "so-called": "", "regime": "government", "thugs": "people", "slammed": "criticized",
    "blasted": "criticized", "radical": "", "extremist": "", "propaganda": "messaging",
    "outrageous": "controversial", "absurd": "disputed", "disastrous": "harmful",
    "shocking": "notable", "ridiculous": "questionable", "illegals": "undocumented immigrants",
    "illegal aliens": "undocumented immigrants", "crippled": "disabled", "the handicapped": "disabled people",
    "wheelchair-bound": "wheelchair user", "suffers from": "has", "confined to a wheelchair": "uses a wheelchair",
}

GROUPS: dict[str, list[str]] = {
    "gender": ["women", "men", "girls", "boys", "ladies", "females", "males", "mothers", "fathers",
               "wives", "husbands", "a woman", "a man", "woman", "man", "she", "he"],
    "age": ["old people", "older people", "the elderly", "elderly people", "seniors", "millennials",
            "boomers", "teenagers", "young people", "kids these days", "older workers", "older employees"],
    "race_ethnicity": ["asians", "africans", "black people", "blacks", "white people", "whites", "hispanics",
                       "latinos", "mexicans", "arabs", "indians", "chinese people", "the chinese",
                       "immigrants", "foreigners", "gypsies", "native americans"],
    "religion": ["muslims", "christians", "jews", "hindus", "sikhs", "buddhists", "atheists", "catholics"],
    "sexual_orientation": ["gays", "gay people", "lesbians", "gay men", "homosexuals", "trans people",
                           "transgender people", "bisexuals"],
    "disability": ["disabled people", "the disabled", "autistic people", "blind people", "deaf people",
                   "people in wheelchairs", "the mentally ill"],
    "socioeconomic": ["poor people", "the poor", "rich people", "the rich", "homeless people",
                      "welfare recipients", "working-class people"],
    "nationality": ["americans", "russians", "germans", "french people", "the french", "italians",
                    "irish people", "the irish", "nigerians", "pakistanis", "brits", "europeans"],
}

GENERALIZING_PREDICATES = [
    "are", "is", "always", "never", "can't", "cannot", "can not", "tend to", "should", "shouldn't",
    "don't", "do not", "aren't", "belong", "lack", "are naturally", "are inherently", "are born",
    "are just", "are all", "all are",
]

NEGATIVE_TRAITS = [
    "lazy", "violent", "stupid", "dirty", "criminal", "criminals", "terrorist", "terrorists", "greedy",
    "cheap", "weak", "emotional", "hysterical", "irrational", "frail", "slow", "senile", "dangerous",
    "savage", "primitive", "inferior", "untrustworthy", "dishonest", "aggressive", "submissive",
    "bad at", "can't drive", "cannot drive", "not suited", "unfit", "incompetent", "less intelligent",
    "uneducated", "thieves", "illegal", "nagging", "bossy", "too old", "can't handle", "cannot handle",
    "better suited", "naturally better", "naturally worse", "belong in the kitchen", "stay at home",
]

PHRASE_PATTERNS: list[tuple[str, str, float, str]] = [
    # (regex, category, weight, explanation)
    (r"\blike a girl\b", "gender", 0.6, "Uses gender as an insult ('like a girl')."),
    (r"\bman up\b", "gender", 0.5, "Ties strength or courage to masculinity."),
    (r"\bfor a (?:woman|girl|lady)\b", "gender", 0.55, "Implies the achievement is surprising because of gender."),
    (r"\b(?:women|girls) (?:belong|should stay) (?:in|at) (?:the )?(?:kitchen|home)\b", "gender", 0.9,
     "Prescribes a domestic role based on gender."),
    (r"\b(?:boys|men) don'?t cry\b", "gender", 0.6, "Prescribes emotional behaviour based on gender."),
    (r"\bthrows? like a girl\b", "gender", 0.6, "Uses gender as an insult."),
    (r"\b(?:under|below|no older than) (?:the age of )?\d{2}\b", "age", 0.45, "States an age limit; usually unnecessary and age-discriminatory."),
    (r"\b(?:native|mother[- ]tongue) english speakers?\b", "nationality", 0.4, "Native-speaker requirements can exclude qualified non-native speakers; consider 'fluent'."),
    (r"\bculture fit\b", "stereotypical_language", 0.2, "Vague 'culture fit' language can mask bias."),
    (r"\b(?:he or she|his or her)\b", "gender", 0.1, "Binary phrasing; 'they/their' is more inclusive."),
]

QUANTIFIERS = ["all", "every", "most", "typical", "those", "these", "such"]

CATEGORY_LABELS = {
    "gender": "Gender", "age": "Age", "race_ethnicity": "Race / ethnicity", "religion": "Religion",
    "sexual_orientation": "Sexual orientation", "disability": "Disability", "socioeconomic": "Socioeconomic",
    "nationality": "Nationality", "stereotypical_language": "Stereotypical language",
    "loaded_language": "Loaded language",
}
