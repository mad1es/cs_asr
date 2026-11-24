"""CS-Audit prototype: code-switch detection and E1-E6 error classification
for Kazakh / Kazakh-Russian ASR output.

Parts
  (a) detector   - character n-gram naive Bayes word language ID (kk vs ru)
  (b) aligner    - word-level Levenshtein alignment of reference and hypothesis
  (c) classifier - assigns each error to E1..E6
  (d) simulator  - injects errors of known type into reference sentences

Run `python3 csaudit.py` for the 20-seed simulated experiment (writes
results.json and fig_results.png). Pure Python + numpy/matplotlib for the figure.
"""
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict

CLASSES = ["E1", "E2", "E3", "E4", "E5", "E6"]
CLASS_NAMES = {
    "E1": "substitution", "E2": "deletion", "E3": "insertion",
    "E4": "wrong language/script", "E5": "loanword/number", "E6": "hallucination",
}

# ---------------------------------------------------------------- data
# Kazakh sentences. Established loanwords (телефон, компьютер, автобус,
# университет) are Kazakh vocabulary and must NOT be flagged as Russian.
KZ_SENTENCES = """
Мен таңертең жұмысқа ерте келдім
Ол кеше ауылдан қайтып оралды
Балалар аулада доп ойнап жүр
Әкем көлікті жөндеуге апарды
Бүгін кешке қонаққа барамыз
Сіз қай қалада тұрасыз
Менің інім биыл мектепті бітіреді
Қыста бұл жерде қар көп жауады
Мұғалім бізге жаңа тақырыпты түсіндірді
Біз демалыста тауға шығамыз
Апам дәмді бауырсақ пісірді
Ертең пойызбен Алматыға жүремін
Сабақтан кейін кітапханада отырдық
Ол әлі хатқа жауап берген жоқ
Бұл жолы бәрі жақсы өтті
Көшеде адамдар өте көп екен
Біздің үйдің жанында саябақ бар
Мен бұл кітапты екі күнде оқып шықтым
Ол әнді әдемі айтады
Досым маған көмектесуге уәде берді
Түскі асты асханада ішеміз
Жаңбыр жауып тұр сондықтан үйде қаламыз
Шешем базардан көкөніс әкелді
Бала ұйықтап жатыр ақырын сөйле
Бұл сұраққа жауап беру қиын
Келесі аптада үш емтихан басталады
Ағам ауруханада дәрігер болып істейді
Біз жиналысты ертеңге қалдырдық
Мен сені аялдамада күтемін
Ол телефонмен сөйлесіп отыр
Компьютерім тағы да істемей қалды
Бүгін теледидардан жаңалықтарды көрдім
Мектепке автобуспен барамын
Кеше күні бойы жұмыс істедім
Сен неге кешігіп келдің
Біз бұл жобаны бір айда аяқтаймыз
Қарындасым университетке түсті
Таңертең шай ішіп нан жедім
Пойыз сағат жетіде келеді
Олар жаңа пәтерге көшіп келді
Әжем бізге бес ертегі айтып берді
Мен саған кейінірек хабарласамын
""".strip().splitlines()

# Kazakh-Russian code-switched sentences; Russian(-origin) words in [brackets].
CS_SENTENCES = """
Мен [вообще] бұл жерге келмеймін
[Короче] ертең кездесейік
Бұл [задание] өте [интересный] екен
Маған [срочно] [справка] керек
Біз [встречаны] кейінге қалдырдық
[Начальник] бүгін [нормально] сөйлесті
Ол [зачёт] тапсыра алмады
[Честно] айтсам маған ұнамады
Менің [зарплатам] әлі түскен жоқ
[Скидка] бар ма екен сұрап көрші
Бұл [магазинде] [цены] қымбат
[Ладно] онда ертең хабарласамын
Ол [опять] кешігіп келді
Мен [квартираны] [снимать] еткім келеді
[Короче] бәрі [нормально] болады
[Слушай] сен қашан келесің
Біз [экзаменге] [готовиться] етіп жатырмыз
[Документтерді] [завтра] әкелемін
[Кстати] кеше саған хабарласқанмын
Бұл [вопрос] өте маңызды
Ол [работада] әлі отыр
[Просто] маған уақыт жетпеді
[Сдачаны] қайтарасыз ба
Сен [звонить] етуді ұмытпа
Бізде [проблема] шықты
[Удобно] болса ертең кел
[Конечно] көмектесемін
Мен екі сағат [пробкада] тұрып қалдым
[Наверное] ол үйде шығар
[Общежитиеде] жарық жоқ
""".strip().splitlines()

# Extra Russian words for training the "ru" class (none of them occur in the
# 20 real test phrases, see soniox_eval.py).
RU_WORDS = """
здравствуйте спасибо пожалуйста хорошо плохо сейчас потом всегда никогда
сегодня вчера неделя месяц время деньги работа город улица дорога машина
человек ребёнок семья друг подруга учитель студент урок книга тетрадь ручка
вода хлеб молоко мясо сахар чай кофе обед ужин завтрак кухня комната дверь
окно стол стул кровать шкаф магазин рынок больница аптека врач лекарство
понимаю думаю знаю хочу могу нужно надо можно нельзя давай давайте буду будет
было была были есть нет да конечно наверное может быть очень много мало
быстро медленно вместе отдельно правильно неправильно точно примерно
главное короче вообще кстати типа значит ладно ничего нормально прикольно
сложно легко интересно скучно вопрос ответ задание экзамен оценка группа
занятие практика лекция семинар деканат общежитие стипендия справка заявление
документ паспорт очередь пробка остановка скидка цена сдача карта чек
перевод зарплата премия отпуск больничный выходной праздник подарок встреча
опоздал перезвоню позвоню напиши скинь отправь проверь посмотри подожди
""".split()

# Established loanwords counted as Kazakh vocabulary.
LOANWORDS = {"телефон", "компьютер", "автобус", "университет", "интернет", "такси"}
LOAN_VARIANTS = {
    "телефон": ["тилифон", "телефн"], "компьютер": ["компютер", "кампьютер"],
    "автобус": ["афтобус", "автобс"], "университет": ["универститет", "унивеситет"],
}
NUMBERS = {"бір": "1", "екі": "2", "үш": "3", "төрт": "4", "бес": "5", "алты": "6",
           "жеті": "7", "сегіз": "8", "тоғыз": "9"}
FILLERS = {"ээ", "эээ", "э", "мм", "ммм", "эм", "ым"}

KK_LETTERS = set("әғқңөұүһі")
RU_ONLY_LETTERS = set("вфцчщэюя")
LATIN = re.compile(r"[a-z]")
CYR = re.compile(r"[а-яәғқңөұүһі]")


# ---------------------------------------------------------------- text utils
def normalise(text):
    """Lowercase, ё->е, strip punctuation (keep inner hyphens), drop fillers."""
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[^\w\s-]", " ", t)
    words = [w.strip("-") for w in t.split()]
    return [w for w in words if w and w not in FILLERS]


def parse_marked(sentence):
    """'Мен [вообще] ...' -> (['мен','вообще',...], ['kk','ru',...])"""
    words, langs = [], []
    for tok in sentence.split():
        lang = "ru" if tok.startswith("[") else "kk"
        w = normalise(tok.strip("[]"))
        if w:
            words.append(w[0])
            langs.append(lang)
    return words, langs


# ---------------------------------------------------------------- (a) detector
class NgramNB:
    """Character 1-4-gram multinomial naive Bayes, two classes kk / ru."""

    def __init__(self, n_max=4, alpha=0.5):
        self.n_max, self.alpha = n_max, alpha

    def _grams(self, word):
        w = f"<{word}>"
        return [w[i:i + n] for n in range(1, self.n_max + 1) for i in range(len(w) - n + 1)]

    def fit(self, words, labels):
        self.counts = {"kk": Counter(), "ru": Counter()}
        n_words = Counter(labels)
        for w, y in zip(words, labels):
            self.counts[y].update(self._grams(w))
        self.vocab = set(self.counts["kk"]) | set(self.counts["ru"])
        self.totals = {c: sum(v.values()) for c, v in self.counts.items()}
        self.prior = {c: math.log(n_words[c] / len(labels)) for c in self.counts}
        return self

    def log_odds_ru(self, word):
        V = len(self.vocab) + 1
        s = {}
        for c in ("kk", "ru"):
            s[c] = self.prior[c] + sum(
                math.log((self.counts[c][g] + self.alpha) / (self.totals[c] + self.alpha * V))
                for g in self._grams(word))
        return s["ru"] - s["kk"]

    def lang(self, word):
        if LATIN.search(word) and not CYR.search(word):
            return "latin"
        if any(ch.isdigit() for ch in word):
            return "num"
        return "ru" if self.log_odds_ru(word) > 0 else "kk"

    def is_code_switched(self, words):
        return any(self.lang(w) == "ru" for w in words)


def letter_rule_lang(word):
    """Baseline: Russian if no Kazakh-specific letter and a Russian-only letter."""
    s = set(word)
    return "ru" if not (s & KK_LETTERS) and (s & RU_ONLY_LETTERS) else "kk"


def train_detector(kz_sents, cs_sents):
    words, labels = [], []
    for s in kz_sents + cs_sents:
        w, l = parse_marked(s)
        words += w
        labels += l
    for w in RU_WORDS:
        words.append(normalise(w)[0])
        labels.append("ru")
    return NgramNB().fit(words, labels)


# ---------------------------------------------------------------- (b) aligner
def align(ref, hyp):
    """Levenshtein word alignment -> list of (op, ref_idx, hyp_idx).
    op in {'ok','sub','del','ins'}; idx is None where not applicable."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
                          d[i - 1][j] + 1, d[i][j - 1] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("ok" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def wer_counts(ref, hyp):
    ops = align(ref, hyp)
    errs = sum(op != "ok" for op, _, _ in ops)
    return errs, len(ref)


# ---------------------------------------------------------------- (c) classifier
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_number_word(w):
    return any(w.startswith(stem) for stem in NUMBERS)


def classify(ref, hyp, det):
    """Return list of (anchor, class, ref_word, hyp_word).
    anchor = ('r', i) for errors on reference word i, ('h', j) for inserted word j."""
    ops = align(ref, hyp)
    out = []
    ins_runs = []  # consecutive insertion runs (list of hyp indices)
    run = []
    for op, i, j in ops:
        if op == "ins":
            run.append(j)
        else:
            if run:
                ins_runs.append(run)
            run = []
    if run:
        ins_runs.append(run)
    halluc = set()
    last_ref_hyp = max([j for op, i, j in ops if op in ("ok", "sub")], default=-1)
    for r in ins_runs:
        if len(r) >= 2 and r[0] > last_ref_hyp:   # invented tail after the last ref word
            halluc.update(r)
            continue
        words = [hyp[j] for j in r]
        for k, j in enumerate(r):
            prev_w = hyp[j - 1] if j > 0 else None
            if hyp[j] == prev_w or len(r) >= 3:
                halluc.add(j)
            elif k + 1 < len(r) and words[k + 1] == hyp[j]:
                halluc.add(j)

    for op, i, j in ops:
        if op == "ok":
            continue
        if op == "del":
            out.append((("r", i), "E2", ref[i], None))
        elif op == "ins":
            out.append((("h", j), "E6" if j in halluc else "E3", None, hyp[j]))
        else:
            r, h = ref[i], hyp[j]
            lr, lh = det.lang(r), det.lang(h)
            if lh == "latin" and lr != "latin":
                c = "E4"
            elif lh == "num" or lr == "num" or (is_number_word(r) and lh == "num"):
                c = "E5"
            elif r in LOANWORDS and edit_distance(r, h) <= 3:
                c = "E5"
            elif lr == "kk" and lh == "ru" and not (set(h) & KK_LETTERS):
                c = "E4"  # Kazakh word written in Russian spelling / as Russian
            else:
                c = "E1"
            out.append((("r", i), c, r, h))
    return out


def classify_sdi(ref, hyp):
    """Baseline: standard substitution/deletion/insertion breakdown."""
    m = {"sub": "E1", "del": "E2", "ins": "E3"}
    return [((("r", i) if op != "ins" else ("h", j)), m[op], None, None)
            for op, i, j in align(ref, hyp) if op != "ok"]


# ---------------------------------------------------------------- (d) simulator
LATIN_MAP = dict(zip("абвгдезийклмнопрстуфхыэ",
                     "abvgdezijklmnoprstufhye"))
LATIN_MAP.update({"ж": "zh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ю": "yu",
                  "я": "ya", "ь": "", "ъ": "", "е": "e"})
KK_TO_RU = dict(zip("әғқңөұүһі", "агкноуухи"))
SUFFIXES = ["ды", "ді", "мын", "мін", "да", "де", "ға", "ге", "ны", "ні", "лар", "лер",
            "ған", "ген", "ып", "іп", "ады", "еді", "сың", "міз"]
INSERT_WORDS = ["және", "да", "ал", "енді", "ғой", "сол", "бір"]
HALLUC_TAILS = [["субтитрлерді", "дайындаған"], ["рахмет", "көргеніңізге", "рахмет"],
                ["келесі", "видеода", "кездескенше"]]


def translit_latin(w):
    return "".join(LATIN_MAP.get(ch, KK_TO_RU.get(ch, ch)) for ch in w)


def kk_to_ru_spelling(w):
    return "".join(KK_TO_RU.get(ch, ch) for ch in w)


def corrupt_substitution(w, rng, vocab):
    if len(w) > 4 and rng.random() < 0.7:
        cut = rng.randint(1, min(3, len(w) - 3))
        return w[:-cut] + rng.choice(SUFFIXES)
    cand = rng.choice(vocab)
    return cand if cand != w else w + "ды"


def simulate(words, langs, rng, vocab, p_err=0.18):
    """Inject errors into a reference. Returns hyp words and gold labels
    {anchor: class} with anchors as in classify()."""
    n = len(words)
    hyp_tokens = []        # (word, source) where source = ('r', i) or 'ins:<class>'
    gold_ref = {}
    for i, (w, l) in enumerate(zip(words, langs)):
        if rng.random() >= p_err:
            hyp_tokens.append((w, ("r", i)))
            continue
        options = [("E1", 0.40), ("E2", 0.17), ("E3", 0.13)]
        if l == "ru":
            options.append(("E4", 0.35))
        elif not (set(w) & KK_LETTERS) or rng.random() < 0.5:
            options.append(("E4", 0.08))
        else:
            options.append(("E4", 0.12))
        if w in LOANWORDS or is_number_word(w):
            options.append(("E5", 0.9))
        cls = rng.choices([c for c, _ in options], weights=[p for _, p in options])[0]
        if cls == "E1":
            nw = corrupt_substitution(w, rng, vocab)
            if nw == w:
                hyp_tokens.append((w, ("r", i)))
                continue
            hyp_tokens.append((nw, ("r", i)))
            gold_ref[("r", i)] = "E1"
        elif cls == "E2":
            gold_ref[("r", i)] = "E2"
        elif cls == "E3":
            hyp_tokens.append((w, ("r", i)))
            ins = rng.choice(INSERT_WORDS)
            if ins == w:
                continue
            hyp_tokens.append((ins, "E3"))
        elif cls == "E4":
            nw = translit_latin(w) if l == "ru" else (
                kk_to_ru_spelling(w) if set(w) & KK_LETTERS else translit_latin(w))
            hyp_tokens.append((nw, ("r", i)))
            gold_ref[("r", i)] = "E4"
        elif cls == "E5":
            stem = next((s for s in NUMBERS if w.startswith(s)), None)
            if stem:
                rest = w[len(stem):]
                nw = NUMBERS[stem] + (("-" + rest) if rest else "")
            else:
                base = next(lw for lw in LOANWORDS if w.startswith(lw))
                nw = rng.choice(LOAN_VARIANTS.get(base, [base[:-1]])) + w[len(base):]
            hyp_tokens.append((nw, ("r", i)))
            gold_ref[("r", i)] = "E5"
    # utterance-level hallucination (E6): repetition or invented tail
    if rng.random() < 0.12 and hyp_tokens:
        if rng.random() < 0.6:
            k = rng.randint(1, len(hyp_tokens) - 1) if len(hyp_tokens) > 1 else 1
            w = hyp_tokens[k - 1][0]
            hyp_tokens[k:k] = [(w, "E6")] * rng.randint(2, 4)
        else:
            hyp_tokens += [(t, "E6") for t in rng.choice(HALLUC_TAILS)]
    hyp = [w for w, _ in hyp_tokens]
    gold = dict(gold_ref)
    for j, (_, src) in enumerate(hyp_tokens):
        if isinstance(src, str):
            gold[("h", j)] = src
    return hyp, gold


# ---------------------------------------------------------------- evaluation
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score_errors(pairs):
    """pairs: list of (gold_dict, pred_list). Returns per-class P/R/F and macro-F1."""
    tp, fp, fn = Counter(), Counter(), Counter()
    for gold, pred in pairs:
        pd = {a: c for a, c, _, _ in pred}
        for a, c in pd.items():
            if gold.get(a) == c:
                tp[c] += 1
            else:
                fp[c] += 1
        for a, c in gold.items():
            if pd.get(a) != c:
                fn[c] += 1
    per = {c: prf(tp[c], fp[c], fn[c]) for c in CLASSES}
    return per, sum(per[c][2] for c in CLASSES) / len(CLASSES)


def run_seed(seed, n_test=600):
    rng = random.Random(seed)
    kz, cs = KZ_SENTENCES[:], CS_SENTENCES[:]
    rng.shuffle(kz)
    rng.shuffle(cs)
    kz_tr, kz_te = kz[:int(.6 * len(kz))], kz[int(.6 * len(kz)):]
    cs_tr, cs_te = cs[:int(.6 * len(cs))], cs[int(.6 * len(cs)):]
    det = train_detector(kz_tr, cs_tr)
    vocab = sorted({w for s in KZ_SENTENCES for w in parse_marked(s)[0]})

    # test utterances: 1-2 held-out sentences joined; half contain code-switching
    utts = []
    for _ in range(n_test):
        is_cs = rng.random() < 0.5
        parts = [rng.choice(cs_te if is_cs else kz_te)]
        if rng.random() < 0.5:
            parts.append(rng.choice(kz_te))
        rng.shuffle(parts)
        words, langs = [], []
        for p in parts:
            w, l = parse_marked(p)
            words += w
            langs += l
        utts.append((words, langs))

    # detection
    det_c = {"nb": Counter(), "rule": Counter()}
    wrd_c = {"nb": Counter(), "rule": Counter()}
    for words, langs in utts:
        gold_cs = "ru" in langs
        for name, f in (("nb", lambda w: det.lang(w) == "ru"),
                        ("rule", lambda w: letter_rule_lang(w) == "ru")):
            preds = [f(w) for w in words]
            pred_cs = any(preds)
            det_c[name][(gold_cs, pred_cs)] += 1
            for g, p in zip(langs, preds):
                wrd_c[name][(g == "ru", p)] += 1

    def f_from(c):
        return prf(c[(True, True)], c[(False, True)], c[(True, False)])

    # classification
    pairs_cs, pairs_sdi = [], []
    wer_e, wer_n = {"kz": 0, "cs": 0}, {"kz": 0, "cs": 0}
    true_share, pred_share = Counter(), Counter()
    for words, langs in utts:
        hyp, gold = simulate(words, langs, rng, vocab)
        pred = classify(words, hyp, det)
        pairs_cs.append((gold, pred))
        pairs_sdi.append((gold, classify_sdi(words, hyp)))
        e, n = wer_counts(words, hyp)
        g = "cs" if "ru" in langs else "kz"
        wer_e[g] += e
        wer_n[g] += n
        true_share.update(gold.values())
        pred_share.update(c for _, c, _, _ in pred)

    per, macro = score_errors(pairs_cs)
    per_b, macro_b = score_errors(pairs_sdi)
    return {
        "seed": seed,
        "det_utt": {"rule": f_from(det_c["rule"]), "nb": f_from(det_c["nb"])},
        "det_word_f1": {"rule": f_from(wrd_c["rule"])[2], "nb": f_from(wrd_c["nb"])[2]},
        "cls_macro_f1": {"csaudit": macro, "sdi": macro_b},
        "cls_per_class": {"csaudit": {c: per[c] for c in CLASSES},
                          "sdi": {c: per_b[c] for c in CLASSES}},
        "sim_wer": {g: wer_e[g] / wer_n[g] for g in wer_e},
        "share_true_E4": true_share["E4"] / sum(true_share.values()),
        "share_pred_E4": pred_share["E4"] / sum(pred_share.values()),
    }


T_975_19 = 2.093  # t critical value, 19 degrees of freedom


def mean_ci(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, T_975_19 * sd / math.sqrt(len(xs))


def summarise(runs):
    S = {}

    def put(key, xs):
        m, h = mean_ci(xs)
        S[key] = {"mean": round(m, 4), "ci95": round(h, 4)}

    for name in ("rule", "nb"):
        for k, idx in (("P", 0), ("R", 1), ("F1", 2)):
            put(f"det_utt_{name}_{k}", [r["det_utt"][name][idx] for r in runs])
        put(f"det_word_{name}_F1", [r["det_word_f1"][name] for r in runs])
    for name in ("csaudit", "sdi"):
        put(f"cls_{name}_macroF1", [r["cls_macro_f1"][name] for r in runs])
        for c in CLASSES:
            for k, idx in (("P", 0), ("R", 1), ("F1", 2)):
                put(f"cls_{name}_{c}_{k}", [r["cls_per_class"][name][c][idx] for r in runs])
    put("cls_gap", [r["cls_macro_f1"]["csaudit"] - r["cls_macro_f1"]["sdi"] for r in runs])
    put("sim_wer_kz", [r["sim_wer"]["kz"] for r in runs])
    put("sim_wer_cs", [r["sim_wer"]["cs"] for r in runs])
    put("share_true_E4", [r["share_true_E4"] for r in runs])
    put("share_pred_E4", [r["share_pred_E4"] for r in runs])
    S["criteria"] = {
        "detector_F1>=0.85": S["det_utt_nb_F1"]["mean"] >= 0.85,
        "classifier_macroF1>=0.80": S["cls_csaudit_macroF1"]["mean"] >= 0.80,
        "classifier_gap>=0.15": S["cls_gap"]["mean"] >= 0.15,
    }
    return S


def plot(S, path="fig_results.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    x = [0, 1]
    for k, (name, lab, col) in enumerate((("rule", "Letter rule", "#9aa5b1"),
                                          ("nb", "Char n-gram NB", "#1f4e79"))):
        vals = [S[f"det_utt_{name}_F1"], S[f"det_word_{name}_F1"]]
        ax[0].bar([xi + (k - .5) * .38 for xi in x], [v["mean"] for v in vals], .36,
                  yerr=[v["ci95"] for v in vals], capsize=3, color=col, label=lab)
    ax[0].axhline(0.85, ls="--", c="#c0392b", lw=1)
    ax[0].text(1.45, 0.86, "target 0.85", ha="right", fontsize=8, color="#c0392b")
    ax[0].set_xticks(x, ["Utterance", "Word"])
    ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("F1 (mean, 95% CI)")
    ax[0].legend(fontsize=8, loc="lower left")
    xs = range(len(CLASSES))
    for k, (name, lab, col) in enumerate((("sdi", "S/D/I baseline", "#9aa5b1"),
                                          ("csaudit", "CS-Audit (E1-E6)", "#1f4e79"))):
        v = [S[f"cls_{name}_{c}_F1"] for c in CLASSES]
        ax[1].bar([xi + (k - .5) * .38 for xi in xs], [a["mean"] for a in v], .36,
                  yerr=[a["ci95"] for a in v], capsize=3, color=col, label=lab)
    ax[1].set_xticks(list(xs), CLASSES)
    ax[1].set_ylim(0, 1.18)
    ax[1].set_ylabel("F1 (mean, 95% CI)")
    ax[1].legend(fontsize=8, loc="upper left", ncol=2)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    seeds = range(20)
    runs = [run_seed(s) for s in seeds]
    S = summarise(runs)
    json.dump({"summary": S, "runs": runs}, open("results.json", "w"), indent=1,
              ensure_ascii=False, default=list)
    f = lambda k: f'{S[k]["mean"]:.3f} ± {S[k]["ci95"]:.3f}'
    print("Code-switch detection (utterance level, 20 seeds x 600 utterances)")
    print(f"  letter rule  F1 {f('det_utt_rule_F1')}  (P {f('det_utt_rule_P')}, R {f('det_utt_rule_R')})")
    print(f"  char-NB      F1 {f('det_utt_nb_F1')}  (P {f('det_utt_nb_P')}, R {f('det_utt_nb_R')})")
    print(f"  word level F1: rule {f('det_word_rule_F1')} / NB {f('det_word_nb_F1')}")
    print("Error classification")
    print(f"  S/D/I baseline macro-F1 {f('cls_sdi_macroF1')}")
    print(f"  CS-Audit       macro-F1 {f('cls_csaudit_macroF1')}  (gap {f('cls_gap')})")
    print("  per class F1: " + ", ".join(f"{c} {S[f'cls_csaudit_{c}_F1']['mean']:.3f}" for c in CLASSES))
    print("  per class R : " + ", ".join(f"{c} {S[f'cls_csaudit_{c}_R']['mean']:.3f}" for c in CLASSES))
    print(f"Simulated WER: Kazakh {f('sim_wer_kz')}, code-switched {f('sim_wer_cs')}")
    print(f"E4 share: true {f('share_true_E4')}, estimated {f('share_pred_E4')}")
    print("Success criteria:", S["criteria"])
    if "--no-plot" not in sys.argv:
        plot(S)


if __name__ == "__main__":
    main()
