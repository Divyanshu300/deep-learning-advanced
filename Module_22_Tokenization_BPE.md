# 📘 MODULE 22: Tokenization — How LLMs Cut Up Text ✂️🔤

**Difficulty:** 🟡 Medium
**Time:** 240 minutes
**Prerequisite:** Modules 1-21
**Tools:** Python (no PyTorch needed!), optionally `tiktoken`
**Phase:** 6 — Inside a Modern LLM (the first module!)

---

## 📖 How to Read These Notes

- Everything explained **like you're 10 years old** and seeing it for the first time 🧒
- **Every line of every method** is explained, with a small example
- 🐞 **Debug traces**: we run each method step by step and print what happens — **all outputs are real**, copied from actually running the code
- ⭐ A whole part (Part H) just on **gluing**, because it's the heart of BPE

---

## 📕 THE NAME LIST

| Name | What it is | Example |
|------|-----------|---------|
| `text` | the words we start with | `"the cat"` |
| `ids` | a list of numbers, one per token | `[116, 104, 101, 32, 99, 97, 116]` |
| `pair` | two tokens sitting side by side | `(97, 116)` = `'a'` + `'t'` |
| `count_pairs` | the method that counts every pair | |
| `merge` | the method that GLUES one pair everywhere | |
| `new_id` | the number given to a freshly glued token | `256` |
| `vocab` | a dictionary: id → the text it stands for | `vocab[256] = b'at'` |
| `merges` | the rule book: pair → new id | `{(97, 116): 256, ...}` |
| `encode` | text → ids (using the rule book) | `"the cat"` → `[259, 99, 256]` |
| `decode` | ids → text | `[259, 99, 256]` → `"the cat"` |

---

# PART A: WHY CUT TEXT AT ALL?

## A.1 Networks Only Understand Numbers

```
   a neural network can't read letters — only numbers 🔢

   "the cat sat"  →  cut it into pieces (TOKENS)  →  give each piece a number (ID)
```

## A.2 You've Already Cut Text Two Ways

```
   PROJECT #3 (sentiment):   cut into WORDS       →  10,000-word vocabulary  📖
   PROJECT #4 (tiny GPT):    cut into CHARACTERS  →  65-character vocabulary 🔤
```

**Both had problems.** Real LLMs use a clever middle way.

---

# PART B: THE PROBLEMS WITH WHOLE WORDS 📖

## B.1 Unknown Words Become `<unk>`

```
   a 10,000-word vocabulary meets...

      a typo:            "beautful"        →  <unk>  ❓
      a new word:        "ChatGPT"         →  <unk>  ❓
      a name:            "Tanvi"           →  <unk>  ❓
      another language:  "नमस्ते"           →  <unk>  ❓
      a rare word:       "unbelievably"    →  <unk>  ❓
```

> 🧒 **A dictionary with missing pages** 📕 — every missing word becomes "???" and its meaning is lost forever.

## B.2 Related Words Are Strangers

```
   "play"    →  id 812        "played"   →  id 2,377
   "plays"   →  id 4,019      "playing"  →  id 6,550

   Four separate words — no hint they're all about playing! 🤷
```

## B.3 The Vocabulary Explodes

```
   hundreds of thousands of English words + every language + every name
   → millions of rows in the embedding table 😰
```

---

# PART C: THE PROBLEMS WITH CHARACTERS 🔤

## C.1 Text Gets VERY Long

```
   "unbelievable"        →  12 tokens
   a 1,000-word essay    →  roughly 5,000+ tokens

   attention compares every token with every token (Module 15):
      100 tokens   →        10,000 comparisons
      5,000 tokens →    25,000,000 comparisons  💥
```

## C.2 The Window Holds Very Little

```
   Project #4 saw 128 characters at once ≈ 25 words ≈ ONE sentence 🔭
```

## C.3 Each Token Means Very Little

```
   "t" alone means nothing 🤷 — the model wastes effort learning to SPELL
```

> 🧒 **Reading a book one letter at a time** 🐌 — you'd forget the start of the chapter before reaching the end!

---

# PART D: ⭐ THE MIDDLE WAY — SUBWORDS 🧩

## D.1 The Idea

```
   ╔═══════════════════════════════════════════════════════════╗
   ║   Common words stay WHOLE.                                  ║
   ║   Rare words are built from smaller, reusable PIECES.       ║
   ╚═══════════════════════════════════════════════════════════╝
```

## D.2 Three Ways to Cut "unbelievable"

```
   CHARACTERS (12 tokens, each means little):
   ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
   │u│n│b│e│l│i│e│v│a│b│l│e│
   └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

   WHOLE WORDS (1 token — but a new word breaks it):
   ┌────────────┐                         ┌─────┐
   │unbelievable│     "unbelievably"  →   │ UNK │  ❌
   └────────────┘                         └─────┘

   SUBWORDS (3 meaningful, reusable pieces):
   ┌──┬──────┬────┐
   │un│believ│able│      "unbelievably" still works: un + believ + ably ✅
   └──┴──────┴────┘
```

## D.3 More Examples (the idea — each real tokenizer cuts slightly differently)

```
   "playing"     →  play + ing
   "replayed"    →  re + play + ed
   "unplayable"  →  un + play + able        ← "play" REUSED every time ♻️
   "the"         →  the                     ← common → stays whole
```

> 🧒 **LEGO bricks** 🧱 — a set of common bricks can build ANY castle, even ones nobody has built before!

## D.4 The Best of Both Worlds

```
  ┌────────────────┬────────────┬───────────────┬────────────────────┐
  │                │ CHARACTERS │ WHOLE WORDS   │ SUBWORDS ⭐        │
  ├────────────────┼────────────┼───────────────┼────────────────────┤
  │ unknown words? │ ✅ never   │ ❌ <unk>      │ ✅ never*          │
  │ short?         │ ❌ long    │ ✅ short      │ ✅ fairly short    │
  │ vocab size     │ ✅ tiny    │ ❌ huge       │ ✅ medium (~50k)   │
  │ pieces mean    │ ❌ little  │ ✅ a lot      │ ✅ quite a lot     │
  │ something?     │            │               │                    │
  └────────────────┴────────────┴───────────────┴────────────────────┘
        * with the byte trick in Part G
```

---

# PART E: ⭐ BPE BY HAND — HOW THE PIECES ARE CHOSEN

## E.1 The Four Rules

```
   1. START:   every single letter is its own token
   2. COUNT:   find which two side-by-side tokens appear together MOST often
   3. GLUE:    join that pair into ONE new token, everywhere it appears
   4. REPEAT:  steps 2 and 3, until you have as many tokens as you want
```

**BPE = Byte Pair Encoding** — "pair" because we glue PAIRS, "byte" because real tokenizers start from bytes (Part G).

> 🧒 **Two friends who are ALWAYS seen together** 👫 get one shared nickname. Then that pair is always seen with a third friend, so the group gets a bigger nickname... 👨‍👩‍👧

## E.2 Our Tiny "Book"

```
   "hug" × 10     "pug" × 5     "pun" × 12     "bun" × 4     "hugs" × 5

   (instead of writing "hug" ten times, we just remember it appears 10 times)
```

## E.3 ROUND 1 — Count Every Side-by-Side Pair

**First, list each word's pairs:**

```
   "hug"   h u g      →  pairs:  h+u,  u+g
   "pug"   p u g      →  pairs:  p+u,  u+g
   "pun"   p u n      →  pairs:  p+u,  u+n
   "bun"   b u n      →  pairs:  b+u,  u+n
   "hugs"  h u g s    →  pairs:  h+u,  u+g,  g+s
```

**Then add up, weighting each pair by how often its word appears:**

```
   u + g   →  hug (10) + pug (5) + hugs (5)   =  20   ⭐ WINNER
   p + u   →  pug (5)  + pun (12)             =  17
   u + n   →  pun (12) + bun (4)              =  16
   h + u   →  hug (10) + hugs (5)             =  15
   g + s   →  hugs (5)                        =   5
   b + u   →  bun (4)                         =   4

   → GLUE "u" + "g" into "ug"
```

```
   BEFORE:   h u g  |  p u g  |  p u n  |  b u n  |  h u g s
   AFTER:    h ug   |  p ug   |  p u n  |  b u n  |  h ug s
                ↑         ↑                              ↑
             glued     glued                          glued
```

## E.4 ROUND 2 — Count Again (things have changed!)

```
   "hug"   h ug       →  h+ug
   "pug"   p ug       →  p+ug
   "pun"   p u n      →  p+u,  u+n
   "bun"   b u n      →  b+u,  u+n
   "hugs"  h ug s     →  h+ug, ug+s

   u + n   →  pun (12) + bun (4)    =  16   ⭐ WINNER
   h + ug  →  hug (10) + hugs (5)   =  15
   p + u   →  pun (12)              =  12   ← was 17! pug's "p u" is gone (it's "p ug" now)
   p + ug  →  pug (5)               =   5
   ug + s  →  hugs (5)              =   5
   b + u   →  bun (4)               =   4

   → GLUE "u" + "n" into "un"
```

> 🧒 **Notice p+u dropped from 17 to 12!** After round 1, the "u" in "pug" got glued to its "g" — so it can't pair with "p" any more. **Every glue changes the next round's counts.** That's why we must re-count every round. 🔁

## E.5 ROUND 3

```
   "hug" h ug · "pug" p ug · "pun" p un · "bun" b un · "hugs" h ug s

   h + ug  →  15  ⭐      p + un → 12      p + ug → 5      ug + s → 5      b + un → 4

   → GLUE "h" + "ug" into "hug"      ← a glued token glued AGAIN!
```

## E.6 ROUND 4

```
   "hug" hug · "pug" p ug · "pun" p un · "bun" b un · "hugs" hug s

   p + un  →  12  ⭐      p + ug → 5       hug + s → 5     b + un → 4

   → GLUE "p" + "un" into "pun"
```

## E.7 The Whole Journey

```
   start:     h u g  |  p u g  |  p u n  |  b u n  |  h u g s
   round 1:   h ug   |  p ug   |  p u n  |  b u n  |  h ug s     (u+g → ug)
   round 2:   h ug   |  p ug   |  p un   |  b un   |  h ug s     (u+n → un)
   round 3:   hug    |  p ug   |  p un   |  b un   |  hug s      (h+ug → hug)
   round 4:   hug    |  p ug   |  pun    |  b un   |  hug s      (p+un → pun)
```

## E.8 What We Learned

```
   starting vocabulary (7):     b  g  h  n  p  s  u
   after 4 glues (11):          b  g  h  n  p  s  u  ug  un  hug  pun

   ⭐ THE RULE BOOK (the merges, in order):
      rule 1:  u + g   →  ug
      rule 2:  u + n   →  un
      rule 3:  h + ug  →  hug
      rule 4:  p + un  →  pun
```

> 🧒 **The rule book IS the tokenizer.** 📜 Real tokenizers just have about 50,000 rules instead of 4!

---

# PART F: CUTTING NEW WORDS BY HAND ✂️

## F.1 The Rule

```
   To cut a new word: apply the rules IN THE SAME ORDER they were learned.
```

## F.2 🔍 "unhug" — Never in Our Book!

```
   start:                   u  n  h  u  g
   rule 1 (u + g → ug):     u  n  h  ug         ← found "u g" at the end, glued
   rule 2 (u + n → un):     un  h  ug           ← found "u n" at the start, glued
   rule 3 (h + ug → hug):   un  hug             ← found "h ug", glued
   rule 4 (p + un → pun):   un  hug             ← no "p" here → nothing happens

   RESULT:  [un] [hug]   ✅
```

## F.3 🔍 More Words (all checked by running code)

```
   "bug"      →  b ug               (rule 1)
   "hugs"     →  hug s              (rules 1, 3)
   "pun"      →  pun                (rules 2, 4)
   "bunhug"   →  b un hug           (rules 1, 2, 3)  ← a made-up word, still sensible!
   "hugpun"   →  hug pun            (rules 1, 2, 3, 4)
```

## F.4 ⚠️ "mug" — Uh-Oh

```
   "mug"  →  m  ug
             ↑
      "m" was NEVER in our book → not in the vocabulary → unknown ❓

   The unknown problem is back — smaller (an unknown LETTER instead of
   an unknown WORD), but still there. Part G fixes it for good. 🔧
```

---

# PART G: ⭐ BYTES — WHY NOTHING IS EVER UNKNOWN 🔢

## G.1 Every Piece of Text Is Made of Bytes

A computer stores all text as **bytes** — numbers from 0 to 255 — using a code called **UTF-8**.

```
   character     bytes                      how many
   ─────────     ─────                      ────────
   "a"           [97]                           1
   "A"           [65]                           1
   " " (space)   [32]                           1
   "é"           [195, 169]                     2
   "क" (Hindi)   [224, 164, 149]                3
   "😀"           [240, 159, 152, 128]           4
```

**Whole words (all from running the code):**

```
   "hi"        →  [104, 105]
   "cat"       →  [99, 97, 116]
   "the cat"   →  [116, 104, 101, 32, 99, 97, 116]
                    t    h    e   ␣   c   a   t
```

## G.2 The Starter Box of 256 Bricks 🧱

```
   A byte-level tokenizer STARTS with all 256 possible bytes as tokens 0 to 255.

   Since EVERY text is made of bytes, EVERY text can always be written
   with these 256 basic tokens:

      English ✅   accents ✅   Hindi ✅   Chinese ✅   emojis ✅   typos ✅

   → <unk> can NEVER happen. ❌❓ is gone forever!
   → new glued tokens get ids 256, 257, 258, ... (0-255 are already taken)
```

> 🧒 **Like owning every basic LEGO shape ever made** 🧱. BPE glues common combinations into bigger bricks — but you can *always* fall back to the basic ones.

---

# PART H: ⭐⭐ GLUING — EVERYTHING ABOUT IT 🧴

"Gluing" (also called **merging**) is the ONE action BPE keeps repeating. Let's understand it completely.

## H.1 Gluing Seen Three Ways

Take the text `"cat sat"` and glue the pair `'a'` + `'t'`:

```
  ── VIEW 1: as TEXT ──
     before:   c | a | t |   | s | a | t          7 pieces
     after:    c | at  |   | s | at              5 pieces
                   ↑               ↑
                glued           glued

  ── VIEW 2: as IDS (what the computer actually holds) ──
     before:   [99, 97, 116, 32, 115, 97, 116]
     after:    [99, 256, 32, 115, 256]
                    ↑              ↑
           (97, 116) became ONE new number: 256

  ── VIEW 3: as LENGTH ──
     before:   7 tokens
     after:    5 tokens        (2 glues happened → 2 fewer tokens)
```

> 🧒 **Like a pearl necklace** 📿. Gluing doesn't add or remove any pearls — it just ties two neighbouring pearls into one charm. Same pearls, fewer separate pieces!

## H.2 ⭐ The Golden Rule: Every Glue Removes EXACTLY One Token

```
   Two tokens go in, ONE comes out.
   2 − 1 = 1 token saved per glue.
```

**Proof from the real run** — the length always shrinks by exactly the number of glues:

```
   round   pair glued         times seen     length before → after    shrank by
   ─────   ──────────         ──────────     ─────────────────────    ─────────
     1     'a' + 't'              12              92 → 80                12  ✅
     2     'e' + ' '               9              80 → 71                 9  ✅
     3     't' + 'h'               8              71 → 63                 8  ✅
     4     'th' + 'e '             8              63 → 55                 8  ✅
     5     'at' + ' '              7              55 → 48                 7  ✅
     6     ' ' + 'the '            6              48 → 42                 6  ✅
     7     'at' + '.'              4              42 → 38                 4  ✅
     8     'at ' + 's'             3              38 → 35                 3  ✅
     9     'at s' + 'at '          3              35 → 32                 3  ✅
    10     'at sat ' + 'o'         3              32 → 29                 3  ✅
```

> 🧒 **That's why BPE picks the MOST common pair** — the more times a pair appears, the more tokens one rule removes. Round 1 alone removed 12! 🏆

## H.3 Gluing Goes Left to Right — and Each Token Glues Only Once

What happens with `"aaa"` when we glue `'a'` + `'a'`?

```
   ids:  [97, 97, 97]          a  a  a

   look at positions 0,1:  'a','a' → MATCH → glue → [256]       then skip BOTH
   look at position 2:     'a' alone (nothing after it) → keep → [256, 97]

   RESULT:  [256, 97]     →     "aa" + "a"
```

```
   ❓ Why not glue positions 1,2 as well?
      Because the 'a' at position 1 is ALREADY used in the first glue!
      A token can't be in two glues at once.
```

**More examples (all real output):**

```
   [97, 97, 97]        glue (97, 97)    →  [256, 97]        a a a     → aa  a
   [97, 97, 97, 97]    glue (97, 97)    →  [256, 256]       a a a a   → aa  aa
   [97, 116]           glue (97, 116)   →  [256]            a t       → at
   [1, 2, 3]           glue (5, 6)      →  [1, 2, 3]        no match  → unchanged
```

> 🧒 **Like partners at a dance** 💃🕺 — once two people pair up, neither can pair with anyone else. Pairing goes left to right along the line, so in "a a a" the first two pair up, and the third waits alone.

## H.4 What Gets REMEMBERED When We Glue

Every glue adds one line to two notebooks:

```
   📜 the RULE BOOK (merges):    which pair → which new id
         (97, 116)  →  256

   📖 the DICTIONARY (vocab):    which id → which text
         256  →  b'a' + b't'  =  b'at'
```

### 🔍 How the dictionary entry is made — gluing bytes together

```python
vocab[256] = vocab[97] + vocab[116]
```

```
   vocab[97]   = b'a'
   vocab[116]  = b't'
   b'a' + b't' = b'at'      ← "+" on bytes just sticks them end to end 📎
```

**More byte-gluing examples:**

```
   b'th'  + b'e '   =  b'the '
   b' '   + b'the ' =  b' the '
   b'at'  + b'.'    =  b'at.'
```

> 🧒 **What is the little `b` in `b'at'`?** It means "these are BYTES, not normal text". Bytes are the raw building blocks — they can be glued even when they're only *half* of a letter (like one of the two bytes in "é"). 🧱

## H.5 ⭐ Gluing Glued Tokens — Tokens Have Family Trees 🌳

In round 4, BPE glued two tokens that were **themselves** glued:

```
   round 3 made:  258 = 't' + 'h'   = 'th'
   round 2 made:  257 = 'e' + ' '   = 'e '
   round 4 glued: 258 + 257         = 'the '   → id 259
```

**The family tree of token 259 (real output):**

```
   259 = 'the '
      258 = 'th'
         116 = 't'
         104 = 'h'
      257 = 'e '
         101 = 'e'
          32 = ' '
```

**And the family tree of the strangest token, 265 (real output):**

```
   265 = 'at sat o'
      264 = 'at sat '
         263 = 'at s'
            260 = 'at '
               256 = 'at'
                   97 = 'a'
                  116 = 't'
                32 = ' '
            115 = 's'
         260 = 'at '
            256 = 'at'
                97 = 'a'
               116 = 't'
             32 = ' '
      111 = 'o'
```

> 🧒 **Every big token is built from two smaller ones, which are built from two smaller ones... all the way down to single bytes.** 🌳 Like Russian nesting dolls 🪆 — open any token and you find two tokens inside!

## H.6 Gluing NEVER Loses Information ✅

```
   To un-glue (decode), just look each id up in the dictionary:

      259  →  b'the '
      99   →  b'c'
      256  →  b'at'
                      glued:  b'the cat'   →   "the cat"  ✅

   The dictionary stores the FULL text of every token, so any list of
   ids can always be turned back into the exact original text.
```

## H.7 ⚠️ Ties — When Two Pairs Are Equally Common

In round 3, two pairs were tied:

```
   ('t', 'h')    seen 8 times
   ('h', 'e ')   seen 8 times     ← a tie! 🤝
```

```
   Python's max() picks the FIRST one it met while counting.
   The text starts "th...", so ('t', 'h') was met first → it wins.

   🧒 Like two runners finishing at exactly the same time — the judge
      picks whoever crossed the camera's line first. 📸

   On big real datasets ties are rare and don't matter much — but they
   mean the result can depend on the ORDER of the text.
```

---

# PART I: ⭐⭐⭐ THE CODE — EVERY METHOD, EVERY LINE, WITH DEBUG RUNS 🐞

We'll build a complete byte-level BPE tokenizer from scratch — **just 5 small methods**:

```
   count_pairs  →  "which neighbours appear together, and how often?"   🧮
   merge        →  "glue one pair everywhere it appears"                🧴
   train        →  "repeat: count, pick the winner, glue, write it down" 🔁
   encode       →  "cut NEW text using the rule book"                  ✂️
   decode       →  "turn ids back into text"                           📖
```

---

## I.0 Seven Little Python Tools You'll Meet (with examples)

### 🔧 1. `text.encode("utf-8")` — text → bytes

```python
list("hi".encode("utf-8"))        # → [104, 105]
list("é".encode("utf-8"))         # → [195, 169]
```

### 🔧 2. Slicing `ids[1:]` — "everything except the first"

```python
[10, 20, 30][1:]                  # → [20, 30]
[10, 20, 30][:2]                  # → [10, 20]   (the first two)
```

### 🔧 3. `zip(a, b)` — walk two lists side by side, making pairs

```python
list(zip([1, 2, 3], [20, 30]))    # → [(1, 20), (2, 30)]     stops at the shorter list
```

### 🔧 4. `Counter(...)` — count how often each thing appears

```python
from collections import Counter
Counter(["a", "b", "a"])          # → Counter({'a': 2, 'b': 1})
```

### 🔧 5. `max(..., key=...)` and `lambda` — "the biggest, judged by THIS"

```python
max([("x", 3), ("y", 7)], key=lambda item: item[1])   # → ('y', 7)
```

```
   lambda item: item[1]   is a tiny nameless method: "give me the item's 2nd part"
      ("x", 3)  →  3
      ("y", 7)  →  7       ← biggest → ("y", 7) wins

   🧒 Like judging a race by TIME, not by the runner's name 🏃
```

### 🔧 6. `dict.get(key, default)` and `float("inf")`

```python
rules = {"a": 1, "b": 2}
rules.get("a", 99)                # → 1      (found)
rules.get("z", 99)                # → 99     (not found → use the default)

min(5, float("inf"), 3)           # → 3      (infinity is bigger than everything)
```

### 🔧 7. Bytes: `bytes([97])`, `b"".join(...)`, `.decode()`

```python
bytes([97])                        # → b'a'
b"th" + b"e "                      # → b'the '
b"".join([b"the ", b"c", b"at"])   # → b'the cat'
b"the cat".decode("utf-8")         # → 'the cat'
```

---

## I.1 METHOD 1 — `count_pairs` 🧮

```python
def count_pairs(ids):                       # line 1
    return Counter(zip(ids, ids[1:]))       # line 2
```

### 🔍 Line by line

```
   line 1   def count_pairs(ids):
            → a method that takes a list of token ids

   line 2   return Counter(zip(ids, ids[1:]))
            → read it from the INSIDE out:
               ① ids[1:]               the list shifted by one
               ② zip(ids, ids[1:])     line them up → every side-by-side pair
               ③ Counter(...)          count each pair
               ④ return                hand the counts back
```

### 🐞 DEBUG RUN — "banana" (real output)

```
   ids          : [98, 97, 110, 97, 110, 97]          b a n a n a
   ids[1:]      : [97, 110, 97, 110, 97]                a n a n a

   zip lines them up:

      ids     :  98   97  110   97  110   97
      ids[1:] :  97  110   97  110   97
                 ──  ───  ───  ───  ───
      pairs   : (98,97) (97,110) (110,97) (97,110) (110,97)
      letters : (b,a)   (a,n)    (n,a)    (a,n)    (n,a)

   Counter      : {(98, 97): 1, (97, 110): 2, (110, 97): 2}
                    b+a: 1       a+n: 2        n+a: 2
```

> 🧒 **Why does `zip` make neighbours?** Because `ids[1:]` is the same list slid one step left ⬅️. Line the two up, and each number sits right above *its right-hand neighbour*. Like two rows of children where everyone in the back row shuffled one seat over! 🪑

### 🐞 DEBUG RUN 2 — "the cat"

```
   ids    : [116, 104, 101, 32, 99, 97, 116]      t h e ␣ c a t

   pairs  : (116,104) (104,101) (101,32) (32,99) (99,97) (97,116)
              t+h       h+e       e+␣      ␣+c     c+a     a+t

   Counter: every pair seen once → each count is 1
```

---

## I.2 METHOD 2 — `merge` (the glue!) 🧴

```python
def merge(ids, pair, new_id):                                    # line 1
    out, i = [], 0                                               # line 2
    while i < len(ids):                                          # line 3
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:    # line 4
            out.append(new_id)                                   # line 5
            i += 2                                               # line 6
        else:                                                    # line 7
            out.append(ids[i])                                   # line 8
            i += 1                                               # line 9
    return out                                                   # line 10
```

### 🔍 Line by line

```
   line 1   def merge(ids, pair, new_id):
            → takes: the ids, the pair to glue, and the new number to use

   line 2   out, i = [], 0
            → out = an empty list where the answer will be built 📦
            → i   = our finger 👆, starting at position 0

   line 3   while i < len(ids):
            → keep going until the finger walks off the end

   line 4   if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            → two checks:
               ① i < len(ids) - 1     is there a NEXT token? (not at the last one)
               ② (ids[i], ids[i+1]) == pair     does THIS + NEXT match the pair?

   line 5   out.append(new_id)
            → MATCH! write the ONE new glued id instead of the two old ones

   line 6   i += 2
            → jump the finger over BOTH old tokens (they're used up!)

   line 7-9 else: out.append(ids[i]);  i += 1
            → no match: copy this token as it is, move the finger ONE step

   line 10  return out
            → hand back the new, shorter list
```

### 🐞 DEBUG RUN — gluing `'a'+'t'` in "cat sat" (real output)

```
   merge([99, 97, 116, 32, 115, 97, 116], pair=(97, 116), new_id=256)
           c   a   t   ␣   s    a   t

   i=0: look at (99, 97)   c+a  → no     → append 99,  step 1 → out = [99]
   i=1: look at (97, 116)  a+t  → MATCH  → append 256, jump 2 → out = [99, 256]
   i=3: look at (32, 115)  ␣+s  → no     → append 32,  step 1 → out = [99, 256, 32]
   i=4: look at (115, 97)  s+a  → no     → append 115, step 1 → out = [99, 256, 32, 115]
   i=5: look at (97, 116)  a+t  → MATCH  → append 256, jump 2 → out = [99, 256, 32, 115, 256]
   i=7: finger is off the end → STOP

   RESULT: [99, 256, 32, 115, 256]        c  at  ␣  s  at      7 → 5 tokens ✅
```

> 🧒 **Notice the finger jumps from i=1 straight to i=3** — position 2 (the 't') was already glued into "at", so we skip it. 👆⏭️

### 🐞 Why the check `i < len(ids) - 1` matters

```
   At the LAST token there is no ids[i + 1] — asking for it would crash 💥
   (an "index out of range" error).

   ids = [99, 97, 116]     len = 3,  so len − 1 = 2
      i = 0:  0 < 2 ✅  → safe to look at ids[1]
      i = 1:  1 < 2 ✅  → safe to look at ids[2]
      i = 2:  2 < 2 ❌  → the last token: don't look ahead, just copy it

   🧒 "Is there anyone to my right?" — check before turning to talk to them! 👀
```

### 🐞 Four Edge Cases (real output)

```
   merge([97, 97, 97],     (97, 97),  256)  →  [256, 97]     a a a   →  aa a
   merge([97, 97, 97, 97], (97, 97),  256)  →  [256, 256]    a a a a →  aa aa
   merge([1, 2, 3],        (5, 6),    256)  →  [1, 2, 3]     no match → unchanged
   merge([97, 116],        (97, 116), 256)  →  [256]         the whole list glues into one
```

**Trace of the first one, step by step:**

```
   [97, 97, 97]
   i=0: (97, 97) MATCH → out = [256],      jump to i=2
   i=2: last token, nothing after it → copy → out = [256, 97],  i=3 → STOP
```

---

## I.3 METHOD 3 — `train` (count → pick → glue → write down) 🔁

```python
def train(text, num_merges):                                            # line 1
    ids = list(text.encode("utf-8"))                                    # line 2
    vocab = {i: bytes([i]) for i in range(256)}                         # line 3
    merges = {}                                                         # line 4
    for k in range(num_merges):                                         # line 5
        pair, freq = max(count_pairs(ids).items(), key=lambda x: x[1])  # line 6
        new_id = 256 + k                                                # line 7
        ids = merge(ids, pair, new_id)                                  # line 8
        merges[pair] = new_id                                           # line 9
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]                 # line 10
        print(f"merge {k+1:>2}: {vocab[new_id].decode()!r:11} seen {freq:>2}x   tokens now {len(ids)}")
    return merges, vocab                                                # line 12
```

### 🔍 Line by line

```
   line 2   ids = list(text.encode("utf-8"))
            → turn the text into byte ids — every character becomes 1+ numbers
              "the cat ..." → [116, 104, 101, 32, 99, ...]

   line 3   vocab = {i: bytes([i]) for i in range(256)}
            → the STARTER BOX 🧱 — the dictionary begins with all 256 bytes:
                 {0: b'\x00', ..., 97: b'a', 98: b'b', ..., 255: b'\xff'}

   line 4   merges = {}
            → the RULE BOOK 📜 starts empty

   line 5   for k in range(num_merges):
            → do this num_merges times: k = 0, 1, 2, ...

   line 6   pair, freq = max(count_pairs(ids).items(), key=lambda x: x[1])
            → count all pairs, then pick the one with the BIGGEST count
                 .items() gives  ((97, 116), 12), ((101, 32), 9), ...
                 key=lambda x: x[1]  → judge each by its count
                 → pair = (97, 116),  freq = 12

   line 7   new_id = 256 + k
            → the next free number: 256, then 257, then 258, ...

   line 8   ids = merge(ids, pair, new_id)
            → GLUE that pair everywhere 🧴 (Method 2)

   line 9   merges[pair] = new_id
            → write the rule: (97, 116) → 256  📜

   line 10  vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            → write the dictionary entry: 256 → b'a' + b't' = b'at'  📖

   line 12  return merges, vocab
            → hand back the rule book and the dictionary — that IS the tokenizer!
```

### 🐞 DEBUG RUN — Round by Round on Our Text (real output)

```
   text = "the cat sat on the mat. the cat ate the rat. the rat sat on the hat.
           the hat sat on the cat."

   START:  92 tokens
   first 16 : [116, 104, 101, 32, 99, 97, 116, 32, 115, 97, 116, 32, 111, 110, 32, 116]
   as text  :   t    h    e   ␣   c   a    t   ␣   s    a    t   ␣   o    n   ␣   t
```

**ROUND 1** (k = 0)

```
   line 6 — the top pairs:     'a'+'t' 12×   'e'+' ' 9×   't'+'h' 8×   'h'+'e' 8×
            winner:            (97, 116)  =  'a' + 't'   seen 12×
   line 7 — new_id:            256 + 0 = 256
   line 8 — glue:              92 → 80 tokens   (12 glues)
   line 9 — rule book:         (97, 116) → 256
   line 10 — dictionary:       vocab[256] = b'a' + b't' = b'at'

   first 16 now : [116, 104, 101, 32, 99, 256, 32, 115, 256, 32, 111, 110, 32, 116, 104, 101]
   as text      :   t    h    e   ␣   c   at   ␣   s    at   ␣   o    n   ␣   t    h    e
```

**ROUND 2** (k = 1)

```
   line 6 — the top pairs:     'e'+' ' 9×   't'+'h' 8×   'h'+'e' 8×   'at'+' ' 7×
            winner:            (101, 32)  =  'e' + ' '   seen 9×
   line 7 — new_id:            257
   line 8 — glue:              80 → 71 tokens   (9 glues)
   line 10 — dictionary:       vocab[257] = b'e' + b' ' = b'e '

   first 16 now : [116, 104, 257, 99, 256, 32, 115, 256, 32, 111, 110, 32, 116, 104, 257, 109]
   as text      :   t    h   e␣    c   at   ␣   s    at   ␣   o    n   ␣   t    h   e␣    m
```

**ROUND 3** (k = 2) — a tie! 🤝

```
   line 6 — the top pairs:     't'+'h' 8×   'h'+'e ' 8×   'at'+' ' 7×   ' '+'t' 6×
            TIE at 8 → max() keeps the FIRST one it met → ('t', 'h')
            winner:            (116, 104)  =  't' + 'h'   seen 8×
   line 7 — new_id:            258
   line 8 — glue:              71 → 63 tokens   (8 glues)
   line 10 — dictionary:       vocab[258] = b't' + b'h' = b'th'

   first 16 now : [258, 257, 99, 256, 32, 115, 256, 32, 111, 110, 32, 258, 257, 109, 256, 46]
   as text      :  th   e␣    c   at   ␣   s    at   ␣   o    n   ␣   th   e␣    m   at    .
```

**ROUND 4** (k = 3) — gluing two glued tokens! 🌳

```
   line 6 — the top pairs:     'th'+'e ' 8×   'at'+' ' 7×   ' '+'th' 6×   'at'+'.' 4×
            winner:            (258, 257)  =  'th' + 'e '   seen 8×
   line 7 — new_id:            259
   line 8 — glue:              63 → 55 tokens   (8 glues)
   line 10 — dictionary:       vocab[259] = b'th' + b'e ' = b'the '   ⭐ the word "the"!

   first 16 now : [259, 99, 256, 32, 115, 256, 32, 111, 110, 32, 259, 109, 256, 46, 32, 259]
   as text      : the␣   c   at   ␣   s    at   ␣   o    n   ␣  the␣   m   at    .   ␣  the␣
```

**ROUNDS 5 to 10 — the summary (real output)**

```
   round   new id   glued                     seen   tokens
   ─────   ──────   ─────                     ────   ──────
     5      260     'at'      + ' '       →   'at '          7×     48
     6      261     ' '       + 'the '    →   ' the '        6×     42
     7      262     'at'      + '.'       →   'at.'          4×     38
     8      263     'at '     + 's'       →   'at s'         3×     35
     9      264     'at s'    + 'at '     →   'at sat '      3×     32
    10      265     'at sat ' + 'o'       →   'at sat o'     3×     29
```

### 🐞 The final rule book (real output)

```
   merges = {
      (97, 116): 256,      'a' + 't'           → 'at'
      (101, 32): 257,      'e' + ' '           → 'e '
      (116, 104): 258,     't' + 'h'           → 'th'
      (258, 257): 259,     'th' + 'e '         → 'the '
      (256, 32): 260,      'at' + ' '          → 'at '
      (32, 259): 261,      ' ' + 'the '        → ' the '
      (256, 46): 262,      'at' + '.'          → 'at.'
      (260, 115): 263,     'at ' + 's'         → 'at s'
      (263, 260): 264,     'at s' + 'at '      → 'at sat '
      (264, 111): 265,     'at sat ' + 'o'     → 'at sat o'
   }
```

---

## I.4 METHOD 4 — `decode` (ids → text) 📖

```python
def decode(ids, vocab):                                                  # line 1
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")   # line 2
```

### 🔍 Line 2, read from the inside out

```
   ① vocab[i] for i in ids     look up every id's bytes in the dictionary
   ② b"".join(...)             glue all those bytes end to end 📎
   ③ .decode("utf-8")          turn the bytes back into readable text
   ④ errors="replace"          if some bytes are broken, show � instead of crashing
```

### 🐞 DEBUG RUN — `decode([259, 99, 256])` (real output)

```
   ids             : [259, 99, 256]

   ① look up each  : vocab[259] = b'the '
                     vocab[99]  = b'c'
                     vocab[256] = b'at'

   ② glue          : b'the ' + b'c' + b'at'  =  b'the cat'

   ③ as text       : 'the cat'   ✅
```

### 🔍 Why `errors="replace"`?

```
   "é" is TWO bytes: [195, 169]. Each byte ALONE is not a valid letter.

   decode([195])       → '�'      (half a letter — shown as a replacement mark)
   decode([195, 169])  → 'é'      (both halves together → a real letter ✅)

   🧒 Like half a jigsaw piece — it only makes a picture with its partner 🧩
```

---

## I.5 METHOD 5 — `encode` (cut NEW text with the rule book) ✂️

```python
def encode(text, merges):                                                # line 1
    ids = list(text.encode("utf-8"))                                     # line 2
    while len(ids) >= 2:                                                 # line 3
        pairs = count_pairs(ids)                                         # line 4
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))     # line 5
        if pair not in merges:                                           # line 6
            break                                                        # line 7
        ids = merge(ids, pair, merges[pair])                             # line 8
    return ids                                                           # line 9
```

### 🔍 Line by line

```
   line 2   ids = list(text.encode("utf-8"))
            → start from raw bytes, like training did

   line 3   while len(ids) >= 2:
            → keep going while there are at least 2 tokens (a pair needs two!)

   line 4   pairs = count_pairs(ids)
            → list every side-by-side pair in the text right now

   line 5   pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
            → ⭐ the clever line! For each pair, ask the rule book for its id:
                 learned pair   → its id (256, 257, ...)
                 unknown pair   → infinity (∞)
              then pick the SMALLEST → the rule that was learned EARLIEST

   line 6-7 if pair not in merges: break
            → if even the smallest is ∞, no pair here has a rule → STOP

   line 8   ids = merge(ids, pair, merges[pair])
            → glue that pair using its rule's id 🧴

   line 9   return ids
```

### ⭐ Why pick the SMALLEST id (the oldest rule)?

```
   Rules were learned in ORDER, and later rules are built from earlier ones:

      rule 256: 'a' + 't'    → 'at'
      rule 260: 'at' + ' '   → 'at '      ← NEEDS 'at' to exist first!

   If we tried rule 260 first, there'd be no 'at' to glue yet.
   Smallest id first = replay the rules in the SAME order as training. 🔁

   🧒 Like following a recipe 🍰 — you must mix the batter (step 1)
      before you can bake it (step 5)!
```

### 🐞 DEBUG RUN 1 — `encode("the cat")` (real output)

```
   START:  [116, 104, 101, 32, 99, 97, 116]     t h e ␣ c a t

   LOOP 1
     pairs and their rule ids:
        't'+'h' → 258     'h'+'e' → ∞     'e'+' ' → 257
        ' '+'c' → ∞       'c'+'a' → ∞     'a'+'t' → 256   ⭐ smallest
     → glue 'a'+'t' with id 256
     now:  [116, 104, 101, 32, 99, 256]      t h e ␣ c at

   LOOP 2
     pairs:  't'+'h' → 258   'h'+'e' → ∞   'e'+' ' → 257 ⭐   ' '+'c' → ∞   'c'+'at' → ∞
     → glue 'e'+' ' with id 257
     now:  [116, 104, 257, 99, 256]          t h e␣ c at

   LOOP 3
     pairs:  't'+'h' → 258 ⭐   'h'+'e ' → ∞   'e '+'c' → ∞   'c'+'at' → ∞
     → glue 't'+'h' with id 258
     now:  [258, 257, 99, 256]               th e␣ c at

   LOOP 4
     pairs:  'th'+'e ' → 259 ⭐   'e '+'c' → ∞   'c'+'at' → ∞
     → glue 'th'+'e ' with id 259
     now:  [259, 99, 256]                    the␣ c at

   LOOP 5
     pairs:  'the '+'c' → ∞   'c'+'at' → ∞
     → every pair is ∞ → no rule applies → STOP 🛑

   RESULT:  [259, 99, 256]     7 bytes → 3 tokens ✅
```

### 🐞 DEBUG RUN 2 — `encode("that cat")` (real output)

```
   START:  [116, 104, 97, 116, 32, 99, 97, 116]     t h a t ␣ c a t

   LOOP 1   smallest: 'a'+'t' → 256   (it appears TWICE — both get glued!)
            now: [116, 104, 256, 32, 99, 256]          t h at ␣ c at

   LOOP 2   pairs: 't'+'h' → 258 ⭐   'h'+'at' → ∞   'at'+' ' → 260   ' '+'c' → ∞   'c'+'at' → ∞
            now: [258, 256, 32, 99, 256]               th at ␣ c at

   LOOP 3   pairs: 'th'+'at' → ∞   'at'+' ' → 260 ⭐   ' '+'c' → ∞   'c'+'at' → ∞
            now: [258, 260, 99, 256]                   th at␣ c at

   LOOP 4   pairs: 'th'+'at ' → ∞   'at '+'c' → ∞   'c'+'at' → ∞   → STOP 🛑

   RESULT:  [258, 260, 99, 256]     "th" + "at " + "c" + "at"
```

> 🧒 **"that" was never in the training text** — but encode still cut it into known pieces ("th" + "at "). That's the power of subwords! 🧱

---

## I.6 A Second Version — BPE on Word Counts (the "hug" example in code)

This is how Part E was done, using **strings** instead of byte ids:

```python
from collections import Counter

corpus = {"hug": 10, "pug": 5, "pun": 12, "bun": 4, "hugs": 5}   # word → how often
words  = {w: list(w) for w in corpus}                           # each word as pieces

def count_pairs_weighted():
    counts = Counter()
    for w, freq in corpus.items():                  # every word, and how often it appears
        pieces = words[w]
        for a, b in zip(pieces, pieces[1:]):        # every side-by-side pair in it
            counts[(a, b)] += freq                  # ⭐ add the WORD's count, not just 1
    return counts

def glue(pieces, pair):
    out, i = [], 0
    while i < len(pieces):
        if i < len(pieces) - 1 and (pieces[i], pieces[i + 1]) == pair:
            out.append(pieces[i] + pieces[i + 1])   # ⭐ glue two STRINGS: "u" + "g" = "ug"
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return out

rules = []
for r in range(4):
    counts = count_pairs_weighted()
    pair, freq = max(counts.items(), key=lambda x: x[1])
    rules.append(pair)
    for w in corpus:
        words[w] = glue(words[w], pair)
    print(f"round {r+1}: {pair[0]!r} + {pair[1]!r} seen {freq}x  ->  {[words[w] for w in corpus]}")

def cut(word):
    pieces = list(word)
    for pair in rules:                              # apply the rules IN ORDER
        pieces = glue(pieces, pair)
    return pieces

for w in ["unhug", "bug", "bunhug", "mug"]:
    print(f"cut({w!r}) -> {cut(w)}")
```

### 🐞 OUTPUT (real)

```
round 1: 'u' + 'g' seen 20x  ->  [['h', 'ug'], ['p', 'ug'], ['p', 'u', 'n'], ['b', 'u', 'n'], ['h', 'ug', 's']]
round 2: 'u' + 'n' seen 16x  ->  [['h', 'ug'], ['p', 'ug'], ['p', 'un'], ['b', 'un'], ['h', 'ug', 's']]
round 3: 'h' + 'ug' seen 15x  ->  [['hug'], ['p', 'ug'], ['p', 'un'], ['b', 'un'], ['hug', 's']]
round 4: 'p' + 'un' seen 12x  ->  [['hug'], ['p', 'ug'], ['pun'], ['b', 'un'], ['hug', 's']]
cut('unhug') -> ['un', 'hug']
cut('bug') -> ['b', 'ug']
cut('bunhug') -> ['b', 'un', 'hug']
cut('mug') -> ['m', 'ug']
```
✅ Exactly matches every round we did by hand in Part E!

### 🔍 The two key differences from the byte version

```
   ① counts[(a, b)] += freq        adds the WORD's count (10 for "hug"),
                                    because "hug" stands for 10 copies of the word

   ② pieces[i] + pieces[i + 1]     glues STRINGS directly ("u" + "g" = "ug"),
                                    instead of inventing a new id number
```

### 🐞 Tracing `count_pairs_weighted` for round 1, word by word

```
   w = "hug",  freq = 10,  pieces = ['h','u','g']    → (h,u) +10   (u,g) +10
   w = "pug",  freq = 5,   pieces = ['p','u','g']    → (p,u) +5    (u,g) +5
   w = "pun",  freq = 12,  pieces = ['p','u','n']    → (p,u) +12   (u,n) +12
   w = "bun",  freq = 4,   pieces = ['b','u','n']    → (b,u) +4    (u,n) +4
   w = "hugs", freq = 5,   pieces = ['h','u','g','s']→ (h,u) +5    (u,g) +5   (g,s) +5

   totals:  (u,g) = 10 + 5 + 5 = 20 ⭐   (p,u) = 5 + 12 = 17   (u,n) = 12 + 4 = 16
            (h,u) = 10 + 5 = 15          (g,s) = 5             (b,u) = 4
```

---

## I.7 ⭐ The Whole Program, Top to Bottom

```python
from collections import Counter

def count_pairs(ids):
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train(text, num_merges):
    ids = list(text.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    for k in range(num_merges):
        pair, freq = max(count_pairs(ids).items(), key=lambda x: x[1])
        new_id = 256 + k
        ids = merge(ids, pair, new_id)
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        print(f"merge {k+1:>2}: {vocab[new_id].decode()!r:11} seen {freq:>2}x   tokens now {len(ids)}")
    return merges, vocab

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        pairs = count_pairs(ids)
        pair = min(pairs, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge(ids, pair, merges[pair])
    return ids

def decode(ids, vocab):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")

text = "the cat sat on the mat. the cat ate the rat. the rat sat on the hat. the hat sat on the cat."
merges, vocab = train(text, 10)

for sentence in ["the cat", "that cat", "the hat sat", "théâtre"]:
    ids = encode(sentence, merges)
    back = decode(ids, vocab)
    print(f"{sentence!r:14} -> {ids}   decode -> {back!r}   same? {back == sentence}")
```

### 🐞 THE FULL SEQUENTIAL RUN (real output)

```
merge  1: 'at'        seen 12x   tokens now 80
merge  2: 'e '        seen  9x   tokens now 71
merge  3: 'th'        seen  8x   tokens now 63
merge  4: 'the '      seen  8x   tokens now 55
merge  5: 'at '       seen  7x   tokens now 48
merge  6: ' the '     seen  6x   tokens now 42
merge  7: 'at.'       seen  4x   tokens now 38
merge  8: 'at s'      seen  3x   tokens now 35
merge  9: 'at sat '   seen  3x   tokens now 32
merge 10: 'at sat o'  seen  3x   tokens now 29
'the cat'      -> [259, 99, 256]   decode -> 'the cat'   same? True
'that cat'     -> [258, 260, 99, 256]   decode -> 'that cat'   same? True
'the hat sat'  -> [259, 104, 263, 256]   decode -> 'the hat sat'   same? True
'théâtre'      -> [258, 195, 169, 195, 162, 116, 114, 101]   decode -> 'théâtre'   same? True
```

### 🔍 What happens, in order, when you press RUN ▶️

```
   1. Python reads the 5 method definitions (nothing runs yet — just learning the recipes 📖)
   2. train(text, 10) runs:
         text → 92 byte ids → 10 rounds of count → pick → glue → write down
         prints one line per round, returns the rule book + dictionary
   3. For each of the 4 test sentences:
         encode  → bytes → replay rules smallest-id-first → token ids
         decode  → look up → glue bytes → text
         compare → same as the original? ✅ True every time
```

### 🔍 Reading the "théâtre" line

```
   [258, 195, 169, 195, 162, 116, 114, 101]
    th   └─ é ──┘  └─ â ──┘   t    r    e

   "th" was learned → glued into 258 ✅
   é and â never appeared in training → they stay as raw bytes
   ...but NOTHING is unknown, and it decodes back perfectly ✅
```

---

# PART J: WHAT THE REAL RUN TEACHES US 🔬

## J.1 It Discovered Real Pieces All By Itself 🎉

```
   ✅ round 1: 'at'     — the most common pair (cat, sat, mat, rat, hat, ate!)
   ✅ round 4: 'the '   — it found the most common WORD with no help
   📉 92 tokens → 29 tokens after just 10 glues
```

```
   tokens
   92 ┤█
   80 ┤█ █
   71 ┤█ █ █
   63 ┤█ █ █ █
   55 ┤█ █ █ █ █
   48 ┤█ █ █ █ █ █
   42 ┤█ █ █ █ █ █ █
   38 ┤█ █ █ █ █ █ █ █
   35 ┤█ █ █ █ █ █ █ █ █
   32 ┤█ █ █ █ █ █ █ █ █ █
   29 ┤█ █ █ █ █ █ █ █ █ █ █
      └┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─ glues done
       0 1 2 3 4 5 6 7 8 9 10

   Big drops early (12, 9, 8...), small drops later (3, 3, 3) —
   the most useful glues come first! 📉
```

## J.2 ⚠️ The Strange Token "at sat o"

```
   'at sat o' glues the END of one word to the START of others.
   It only happened because our tiny text repeats "...at sat on..." so much.
```

**How REAL tokenizers stop this:** they first chop the text into word-sized chunks, and only allow glues INSIDE a chunk:

```
   "the cat sat on"  →  ["the", " cat", " sat", " on"]
                          ↑ glues allowed inside each chunk only ✂️

   so "at" + " " + "s" can NEVER be glued — they're in different chunks!
```

> 🧒 **"You can glue LEGO bricks together, but never glue two different models together"** 🧱🚫🧱

## J.3 Where the Space Goes

```
   our run made:          'the '   (space at the END)
   GPT-style tokenizers:  ' the'   (space at the START)

   So in a real tokenizer:
      "the cat"   →  "the" + " cat"      ← " cat" with a space
      "cat food"  →  "cat" + " food"     ← "cat" with no space

   " cat" and "cat" are DIFFERENT tokens with DIFFERENT ids!
```

---

# PART K: REAL TOKENIZERS — THE NUMBERS 📊

## K.1 Vocabulary Sizes

```
   model              vocabulary size
   ─────              ───────────────
   GPT-2                  50,257
   Llama 2                32,000
   GPT-4 (cl100k)        about 100,000
   Llama 3               128,256

   🧒 Ours had 256 + 10 = 266. Real ones do tens of thousands of glues!
```

## K.2 A Handy Rule of Thumb (English)

```
   1 token  ≈  about 4 characters  ≈  about ¾ of a word

      100 tokens        ≈  about 75 words
      a 300-word email  ≈  about 400 tokens
      a 1,000-word essay≈  about 1,300 tokens
```

## K.3 Why Not a HUGE Vocabulary?

```
   Every token needs a row in the embedding table (Module 12)
   and a score in the final output layer.

   GPT-2:   50,257 tokens × 768 numbers  =  38,597,376 settings
            just for the embedding table! 😮

  ┌──────────────────────┬─────────────────────────┬──────────────────────────┐
  │                      │ SMALL vocab (e.g. 8k)    │ LARGE vocab (e.g. 200k)  │
  ├──────────────────────┼─────────────────────────┼──────────────────────────┤
  │ text length          │ ❌ more tokens per text  │ ✅ fewer tokens          │
  │ embedding table      │ ✅ small                 │ ❌ big                   │
  │ practice per token   │ ✅ each seen very often  │ ❌ rare ones seen rarely │
  └──────────────────────┴─────────────────────────┴──────────────────────────┘

   Goldilocks again 🐻 — most models choose between about 32,000 and 200,000.
```

---

# PART L: ⭐ TOKENIZATION EXPLAINS STRANGE LLM BEHAVIOUR 🤔

**The model never sees letters — only token ids.**

## L.1 Counting Letters

```
   "How many r's are in strawberry?"

   You see:         s t r a w b e r r y       → easy to count 🔍
   The model sees:  a few chunks, for example  [str] [aw] [berry]

   🧒 The letters are hidden INSIDE the chunks, like sweets inside
      wrapped boxes 🎁 — the model must remember what's in each box.
```

## L.2 Spelling Backwards

```
   "Write 'lollipop' backwards."      → hard (letters hidden in chunks) 🔄
   "Reverse: l o l l i p o p"         → easier (each letter is its own token)
```

## L.3 Maths With Big Numbers

```
   "1234567" might be cut like  [123] [456] [7]
   "1234568" might be cut differently!

   🧒 Like doing sums where the digits come in random-sized bundles 🔢
```

## L.4 Other Languages Cost More

```
   English: 1 byte per letter, very common in training → many glues → FEW tokens
   Hindi:   3 bytes per letter (Part G), less common    → fewer glues → MORE tokens

   → less fits in the context window, and paid APIs (charged per token)
     can cost more for the same meaning 💰
```

## L.5 Spaces Matter

```
   "hello" and " hello" are DIFFERENT tokens.
   A stray space at the end of a prompt can slightly confuse what comes next. 🤏
```

## L.6 Why This Matters to YOU 🎯

```
   📏 context windows are measured in TOKENS, not words
   💰 API prices are charged per TOKEN
   ⚡ generation speed depends on how many TOKENS are written
```

---

# PART M: SPECIAL TOKENS 🎫

```
   <|endoftext|>   "this document has ended"          🏁
   <pad>           filler to make lengths match       (Project #3!)
   chat markers    "a USER message starts here"
                   "the ASSISTANT's reply starts here"
```

```
   A chat like:                     might reach the model like:

   User: Hi!                        <start-user> Hi! <end>
   Assistant: Hello!                <start-assistant> Hello! <end>

   (each model family uses its own exact marker names)
```

> 🧒 **Like stage directions in a play** 🎭 — "[Enter NARRATOR]" isn't spoken aloud; it tells the actors what's happening. In **Module 32**, special tokens are how an LLM says *"I want to call a tool"* 🔧.

---

# PART N: WHERE THE TOKENIZER SITS IN AN LLM

```
   "the cat"  →  [tokenizer: encode]  →  [259, 99, 256]  →  embedding table (M12)
                                                                   ↓
                                                             transformer (M15)
                                                                   ↓
   " sat"     ←  [tokenizer: decode]  ←  pick one id    ←  scores, one per token
```

```
   ⭐ The tokenizer is NOT part of the neural network:
      → it's trained FIRST, with counting and gluing only (no gradients!)
      → then it's FROZEN — it never changes again 🧊
      → only then is the model trained on its token ids

   🧒 Like choosing the alphabet BEFORE writing the books 📚
```

---

# PART O: TRY A REAL TOKENIZER 🔬

```python
# pip install tiktoken
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")        # a GPT-4-era tokenizer
ids = enc.encode("How many r's are in strawberry?")
print(len(ids), "tokens")
print([enc.decode([i]) for i in ids])             # see each piece
```

```
   Things to try 🔍:
      "strawberry"        → how is it cut?
      "1234567"           → how are the digits bundled?
      "hello" vs " hello" → different ids?
      a sentence in Hindi → how many more tokens?

   (Exact outputs depend on the tokenizer version — run it and look!)
```

---

# 📋 MODULE 22 MASTER RECAP

```
   1. LLMs cut text into TOKENS and turn each into an ID before thinking 🔢

   2. WHOLE WORDS: short ✅ — but <unk>, strangers, and a huge vocab ❌
      CHARACTERS: never unknown ✅ — but very long, little meaning ❌
      SUBWORDS ⭐: common words whole, rare words in reusable pieces 🧱

   3. BPE = count side-by-side pairs → GLUE the most common → repeat
         our book: ug (20×) → un (16×) → hug (15×) → pun (12×)
         every glue changes the next round's counts — so RE-COUNT each round 🔁

   4. THE RULE BOOK (merges) IS the tokenizer 📜
      new words: replay the rules in order → "unhug" → un + hug

   5. BYTE-LEVEL: start from all 256 bytes → NOTHING is ever unknown
         é = 2 bytes · क = 3 bytes · 😀 = 4 bytes

   6. ⭐ GLUING:
         two tokens in → ONE new id out → exactly one token fewer
         left to right; a token can only be in one glue ([97,97,97] → [256, 97])
         vocab[new] = vocab[left] + vocab[right]  (bytes stuck end to end 📎)
         glued tokens can be glued again → family trees 🌳🪆
         decoding just looks up and re-joins → nothing is ever lost ✅
         ties → max() keeps the first pair it met 🤝

   7. FIVE METHODS:
         count_pairs → Counter(zip(ids, ids[1:]))
         merge       → walk with a finger 👆: match → new id, jump 2; else copy, step 1
         train       → count → max → new_id = 256 + k → merge → write rule + dictionary
         encode      → replay rules, SMALLEST id first (oldest rule first) 🍰
         decode      → look up, join bytes, decode

   8. REAL RUN: 92 → 29 tokens; found "the " by itself 🎉
      odd tokens like "at sat o" → real tokenizers pre-split into words ✂️

   9. Real vocab sizes ~32k to 128k+ · 1 token ≈ 4 English characters

  10. The model sees CHUNKS, not letters → counting, reversing,
      big-number maths and some languages are harder 🤔

  11. The tokenizer is trained FIRST (counting only), then frozen 🧊
```

---

# 🤔 COMMON DOUBTS

**Q1: Is the tokenizer a neural network?**
> 🧒 No! BPE is just **counting and gluing** — no gradients, no backprop. It's trained first, frozen, and only then is the neural network trained on its ids 🧊.

**Q2: Why must we re-count the pairs after every glue?**
> 🧒 Because gluing changes the neighbours! In Part E, "p + u" dropped from 17 to 12 after round 1, because the "u" in "pug" got glued to its "g" 🔁.

**Q3: Why does every glue remove exactly one token?**
> 🧒 Two tokens go in, one comes out: 2 − 1 = 1. So a pair seen 12 times removes 12 tokens (Part H.2) ✅.

**Q4: Why does `merge([97, 97, 97], (97, 97), 256)` give `[256, 97]` and not two glues?**
> 🧒 Gluing walks left to right, and each token can only be in ONE glue. The middle 'a' is used up by the first glue, so the last 'a' is left alone 💃🕺.

**Q5: Why does `encode` pick the SMALLEST rule id?**
> 🧒 Smaller id = learned earlier. Later rules are built from earlier ones (rule 260 'at ' needs rule 256 'at' first), so we must replay them in the original order 🍰.

**Q6: What does `float("inf")` do in `encode`?**
> 🧒 It gives unlearned pairs a score of infinity, so `min()` never picks them unless *nothing* is learned — and then we stop ♾️.

**Q7: Why do we need the check `i < len(ids) - 1` in `merge`?**
> 🧒 At the last token there's no right-hand neighbour — looking for one would crash 💥. Check before you look! 👀

**Q8: What is the `b` in `b'at'`?**
> 🧒 It marks BYTES, not normal text. Bytes can be glued even when they're only half a letter, like one byte of "é" 🧱.

**Q9: Can two different texts give the same ids?**
> 🧒 No — every id maps to exact bytes, so decoding always gives back the one original text. Encode and decode are perfect opposites ✅.

**Q10: Why do LLMs struggle to count the letters in "strawberry"?**
> 🧒 They never see the letters — just a few chunks, with the letters hidden inside 🎁.

**Q11: Why does the same sentence cost more tokens in some languages?**
> 🧒 Some alphabets need more bytes per letter, and rarer languages get fewer glues learned — so they're cut into more, smaller pieces 💰.

**Q12: Do all LLMs use the same tokenizer?**
> 🧒 No — each model family trains its own, with its own size and rules. The same sentence can be a different number of tokens in different models.

---

# ✅ QUICK PRACTICE

**Q1:** What does `count_pairs([1, 2, 1, 2])` return?
<details><summary>Answer</summary>
Pairs: (1,2), (2,1), (1,2) → Counter({(1, 2): 2, (2, 1): 1}).
</details>

**Q2:** What does `merge([1, 2, 1, 2], (1, 2), 300)` return? Trace it.
<details><summary>Answer</summary>
i=0: (1,2) MATCH → [300], jump to i=2. i=2: (1,2) MATCH → [300, 300], jump to i=4 → stop. Result: [300, 300].
</details>

**Q3:** What does `merge([5, 5, 5, 5, 5], (5, 5), 300)` return?
<details><summary>Answer</summary>
i=0 match → [300], i=2 match → [300, 300], i=4 is the last token → copy → [300, 300, 5].
</details>

**Q4:** A pair is seen 7 times. How many tokens does gluing it remove?
<details><summary>Answer</summary>
Exactly 7 — each glue removes one token.
</details>

**Q5:** `vocab[260] = vocab[256] + vocab[32]`, where `vocab[256] = b'at'` and `vocab[32] = b' '`. What is `vocab[260]`?
<details><summary>Answer</summary>
b'at ' — the bytes stuck end to end.
</details>

**Q6:** Using the hug rule book (u+g, u+n, h+ug, p+un), cut "hugpun".
<details><summary>Answer</summary>
h u g p u n → h ug p u n → h ug p un → hug p un → hug pun. Result: [hug] [pun].
</details>

**Q7:** In `encode`, the pairs present have rule ids {258, ∞, 257, ∞, 256}. Which is glued first, and why?
<details><summary>Answer</summary>
256 — the smallest id, meaning the oldest rule. Rules must be replayed in the order they were learned.
</details>

**Q8:** Draw the family tree of token 259 ('the ').
<details><summary>Answer</summary>

```
259 = 'the '
   258 = 'th'   →  116 't'  +  104 'h'
   257 = 'e '   →  101 'e'  +   32 ' '
```
</details>

**Q9:** Round 1 counts were u+g 20, p+u 17, u+n 16. After gluing u+g, why does p+u fall to 12?
<details><summary>Answer</summary>
"pug" (5×) is now "p ug", so its "p u" pair no longer exists. Only "pun" (12×) still has "p u".
</details>

**Q10:** Why can a byte-level tokenizer never produce `<unk>`?
<details><summary>Answer</summary>
Its first 256 tokens are every possible byte, and all text is made of bytes — so any text can always be written with them.
</details>

**Q11:** About how many tokens is a 600-word English essay?
<details><summary>Answer</summary>
About 800 tokens (600 ÷ 0.75).
</details>

**Q12:** `decode([195])` alone shows `'�'`, but `decode([195, 169])` shows `'é'`. Why?
<details><summary>Answer</summary>
"é" is two bytes. One byte alone is half a letter, so it's shown as a replacement mark; both together make the real letter 🧩.
</details>

---

# 🎬 WHAT'S NEXT: MODULE 23 — MODERN LLM ARCHITECTURE + PROJECT #7 🏗️

```
   Your Project #4 GPT used a 2019-style design. Today's LLMs upgraded
   almost every part:

      learned positions  →  RoPE                (spinning positions 🌀)
      LayerNorm          →  RMSNorm             (a simpler tidy-up 🧹)
      ReLU               →  SwiGLU              (a smarter feed-forward 🚪)
      plain attention    →  GQA                 (sharing keys to save memory 💾)
      one feed-forward   →  Mixture of Experts  (many specialists 👩‍🔬👨‍🔧)

   And PROJECT #7: upgrade your own Tiny GPT to a modern design! 🚀
```

---

*Module 22 Complete! You can build a tokenizer from scratch, trace every glue, and explain why LLMs see the world in chunks! ✂️🔤*
