# 📘 MODULE 20: Vision Transformers + CLIP — Teaching Machines to Connect Pictures and Words 👁️📝

**Difficulty:** 🟠 Medium-Hard
**Time:** 240 minutes
**Prerequisite:** Modules 1-19 + Projects #1-#5
**Tools:** Google Colab, PyTorch, Hugging Face `transformers`
**Phase:** 5 — Generative Models for Images

---

## 📖 How to Read These Notes

- Everything explained **like you're 10 years old** and seeing it for the first time 🧒
- **Lots of small examples at every step** — every number checked with Python 🧮
- **The same names from start to finish** (see the list below)
- Pictures everywhere (ASCII diagrams) 🖼️
- 🔍 Code with the exact output it prints

---

## 📕 THE NAME LIST

### 👁️ Vision Transformer words

| Name | What it is | Example |
|------|-----------|---------|
| `picture` | the image going in | 28×28 pixels |
| `patch` | one small square cut from the picture 🧩 | 7×7 pixels = 49 numbers |
| `patch_vector` | a patch after the embedding layer | 64 numbers |
| `position` | a stamp saying where a patch came from 📍 | `[0.1, 0.0, 0.0]` |
| `cls_token` | the extra "class representative" token 🙋 | `[0.2, 0.2, 0.2]` |
| `tokens` | CLS + all the patch vectors, in a row | 17 tokens for MNIST |
| `look_percent` | attention percentages (Module 15) | add to 100% |

### 🗺️ CLIP words

| Name | What it is | Example |
|------|-----------|---------|
| `image_vector` | a picture's spot on the shared map | `[0.96, 0.28]` |
| `text_vector` | a caption's spot on the shared map | `[1.00, 0.00]` |
| `scores` | the grid comparing every picture with every caption | a 3×3 grid |
| `scale` | the stretch before softmax (temperature 🌡️) | 10 |
| `targets` | which caption is the right one for each picture | `[0, 1, 2]` |

---

# PART A: WHY THIS MODULE EXISTS

## A.1 The Big Goal

Next module you'll learn how **"a cat wearing a hat" → a brand-new picture** works. For that, a model must understand **two languages at once**:

```
   📝 WORDS:     "a cat wearing a hat"
   🖼️ PICTURES:  thousands of pixel numbers

   The model must know these two MEAN THE SAME THING. 🤝
```

## A.2 The Two Tools You'll Learn

```
   TOOL 1 — VISION TRANSFORMER (ViT) 👁️
      the Module 15 transformer, taught to READ PICTURES

   TOOL 2 — CLIP 🗺️
      puts pictures AND captions on ONE shared map, so
      "a photo of a dog" lands right next to photos of dogs
```

## A.3 🧒 The Bilingual Dictionary 📖

```
   An English–French dictionary puts matching words side by side:

      dog   ↔  chien
      cat   ↔  chat
      car   ↔  voiture

   CLIP is a dictionary between PICTURES and WORDS:

      🐶  ↔  "a dog"
      🐱  ↔  "a cat"
      🚗  ↔  "a car"
```

---

# PART B: THE PROBLEM — TRANSFORMERS READ LISTS 📜

## B.1 A Transformer Wants a Line of Tokens

```
   Module 15:   "the cat sat"  →  [the] [cat] [sat]    ✅ 3 tokens in a row
```

## B.2 But a Picture Is a GRID

```
   ┌──┬──┬──┬──┬──┬──┐
   │  │  │  │  │  │  │    28 rows × 28 columns of pixels
   ├──┼──┼──┼──┼──┼──┤    ← this is not a sentence!
   │  │  │  │  │  │  │
   └──┴──┴──┴──┴──┴──┘
```

## B.3 🔍 Why Not Make Every Pixel a Token?

Attention compares **every token with every token**. Count the comparisons:

```
   MNIST (28 × 28):
      784 pixel-tokens  →  784 × 784  =  614,656 comparisons per head 😰

   A normal photo (224 × 224):
      50,176 pixel-tokens  →  50,176 × 50,176  ≈  2.5 BILLION per head 😱💥
```

> 🧒 **Like asking every grain of sand on a beach to say hello to every other grain** 🏖️. Far too slow! And a single pixel on its own means almost nothing — one dot can't tell you it's part of a cat.

---

# PART C: ⭐ THE BIG IDEA — CUT THE PICTURE INTO PATCHES 🧩

## C.1 The Idea

```
   ╔═══════════════════════════════════════════════════════════╗
   ║   Cut the picture into small SQUARES called PATCHES.        ║
   ║   Treat each patch like ONE WORD.                           ║
   ╚═══════════════════════════════════════════════════════════╝
```

## C.2 The Picture — a "7" Cut Into 16 Patches

```
        a 28 × 28 picture of a 7                    a "sentence" of 17 tokens

   ┌──────┬──────┬──────┬──────┐
   │  1 ██│  2 ██│  3 ██│  4 ██│              ┌─────┬───┬───┬───┬───┬─────┬────┐
   ├──────┼──────┼──────┼──────┤              │ CLS │ 1 │ 2 │ 3 │ 4 │ ... │ 16 │
   │  5   │  6   │  7 ██│  8   │    ────▶     └─────┴───┴───┴───┴───┴─────┴────┘
   ├──────┼──────┼──────┼──────┤                ↑
   │  9   │ 10 ██│ 11   │ 12   │                a special summary token
   ├──────┼──────┼──────┼──────┤
   │ 13   │ 14 ██│ 15   │ 16   │              each patch = one "word" of 49 pixels
   └──────┴──────┴──────┴──────┘

       ██ = a patch with ink      blank = empty background
```

## C.3 🔍 Counting Patches — Six Examples

```
   picture    patch size   across × down   patches   numbers per patch    tokens (+CLS)
   ───────    ──────────   ─────────────   ───────   ─────────────────    ─────────────
   28 × 28     7 × 7        4 × 4            16      7×7×1  =   49            17
   32 × 32     8 × 8        4 × 4            16      8×8×1  =   64            17
   64 × 64    16 × 16       4 × 4            16     16×16×1 =  256            17
   224 × 224  32 × 32       7 × 7            49     32×32×3 = 3072            50
   224 × 224  16 × 16      14 × 14          196     16×16×3 =  768           197
   224 × 224  14 × 14      16 × 16          256     14×14×3 =  588           257
```

```
   🔍 How to read a row — the 224, patch 16 one:
      224 ÷ 16 = 14 patches across, 14 down
      14 × 14 = 196 patches
      each patch: 16 × 16 pixels × 3 colours (red, green, blue) = 768 numbers
      196 patches + 1 CLS = 197 tokens
```

## C.4 🔍 Patches Make Attention Affordable

```
   tokens     comparisons per head (tokens × tokens)
   ──────     ──────────────────────────────────────
       17     17 × 17     =        289      ✅ MNIST with patches
       50     50 × 50     =      2,500      ✅
      197     197 × 197   =     38,809      ✅ a normal photo with patches
      784     784 × 784   =    614,656      😰 MNIST with pixels
   50,176                 ≈ 2,517,630,976   😱 a photo with pixels
```

## C.5 🧒 Everyday Patches

```
   📚 a comic book        → read it panel by panel, not dot by dot
   📮 a sheet of stamps   → each stamp is a patch
   🧩 a jigsaw puzzle     → each piece is a patch
   🍫 a chocolate bar     → each square is a patch
```

> 🧒 **A patch holds a meaningful little piece** — an edge, a curve, a corner — the way a comic panel holds a little piece of the story!

---

# PART D: ⭐ A TINY WALKTHROUGH — A 4×4 PICTURE OF AN "L"

## D.1 Our Picture

```
   (1 = ink, 0 = background)

        col 0  col 1  col 2  col 3
       ┌──────┬──────┬──────┬──────┐
   r0  │  1   │  0   │  0   │  0   │        ⬜⬛⬛⬛
       ├──────┼──────┼──────┼──────┤        ⬜⬛⬛⬛
   r1  │  1   │  0   │  0   │  0   │        ⬜⬛⬛⬛
       ├──────┼──────┼──────┼──────┤        ⬜⬜⬜⬜   ← an "L"!
   r2  │  1   │  0   │  0   │  0   │
       ├──────┼──────┼──────┼──────┤
   r3  │  1   │  1   │  1   │  1   │
       └──────┴──────┴──────┴──────┘

   4 ÷ 2 = 2 patches across, 2 down → 4 patches of 2 × 2 = 4 pixels each
```

## D.2 Cut Into 4 Patches, Then Flatten Each

```
   PATCH 0 (top-left)            PATCH 1 (top-right)
   ┌───┬───┐                     ┌───┬───┐
   │ 1 │ 0 │                     │ 0 │ 0 │
   ├───┼───┤  →  [1, 0, 1, 0]    ├───┼───┤  →  [0, 0, 0, 0]   ← empty!
   │ 1 │ 0 │                     │ 0 │ 0 │
   └───┴───┘                     └───┴───┘

   PATCH 2 (bottom-left)         PATCH 3 (bottom-right)
   ┌───┬───┐                     ┌───┬───┐
   │ 1 │ 0 │                     │ 0 │ 0 │
   ├───┼───┤  →  [1, 0, 1, 1]    ├───┼───┤  →  [0, 0, 1, 1]
   │ 1 │ 1 │                     │ 1 │ 1 │
   └───┴───┘                     └───┴───┘
```

### 🔍 "Flatten" — read left to right, top to bottom

```
   ┌───┬───┐
   │ a │ b │   →   [a, b, c, d]
   ├───┼───┤
   │ c │ d │       🧒 Like reading a book: first line, then the next 📖
   └───┴───┘          (exactly Project #5's .view!)
```

**Three more flattening examples:**

```
   [0, 1]          [1, 1]          [0, 0]
   [1, 1]  → [0,1,1,1]   [0, 0]  → [1,1,0,0]   [0, 1]  → [0,0,0,1]
```

---

# PART E: PATCH EMBEDDING — EACH PATCH BECOMES A "WORD VECTOR"

## E.1 Why Not a Lookup Table?

```
   Module 12:  each WORD → look it up in a table
               ("cat" is row 5 → grab row 5)

   A patch can't be looked up — there are endless possible patches!
   So we pass it through a LINEAR LAYER instead.
```

## E.2 The Grid (4 pixel numbers → 3 embedding numbers)

```
                      col 0    col 1    col 2
   row 0 (pixel 0)     0.5      0.1      0.2
   row 1 (pixel 1)     0.3      0.4      0.1
   row 2 (pixel 2)     0.2      0.3      0.5
   row 3 (pixel 3)     0.1      0.6      0.3
```

## E.3 🔍 PATCH 0 = [1, 0, 1, 0] — Every Multiplication

```
   col 0:  1×0.5 + 0×0.3 + 1×0.2 + 0×0.1  =  0.5 + 0 + 0.2 + 0  =  0.7
   col 1:  1×0.1 + 0×0.4 + 1×0.3 + 0×0.6  =  0.1 + 0 + 0.3 + 0  =  0.4
   col 2:  1×0.2 + 0×0.1 + 1×0.5 + 0×0.3  =  0.2 + 0 + 0.5 + 0  =  0.7

   PATCH 0  →  [0.7, 0.4, 0.7]
```

> 🧒 **A shortcut you can see:** pixels that are 0 contribute nothing! So only rows 0 and 2 (the "1" pixels) matter — just add those two rows: `[0.5,0.1,0.2] + [0.2,0.3,0.5] = [0.7,0.4,0.7]` ✅

## E.4 🔍 PATCH 2 = [1, 0, 1, 1] — Using the Shortcut

```
   the "1" pixels are 0, 2 and 3 → add rows 0, 2 and 3:

      row 0:  [0.5, 0.1, 0.2]
      row 2:  [0.2, 0.3, 0.5]
      row 3:  [0.1, 0.6, 0.3]
              ───────────────
              [0.8, 1.0, 1.0]
```

## E.5 All Four Patches

```
   PATCH 0  [1, 0, 1, 0]  →  [0.7, 0.4, 0.7]
   PATCH 1  [0, 0, 0, 0]  →  [0.0, 0.0, 0.0]   ← empty patch → all zeros!
   PATCH 2  [1, 0, 1, 1]  →  [0.8, 1.0, 1.0]   ← the busy corner
   PATCH 3  [0, 0, 1, 1]  →  [0.3, 0.9, 0.8]
```

> 🧒 **Exactly like the Q/K/V grids in Module 15** — every pixel feeds every output number, nothing is thrown away. And the grid's numbers are **learned**, so the model slowly discovers the best way to describe a patch 🎓.

---

# PART F: POSITIONS — "WHICH PATCH WAS WHERE?" 📍

## F.1 🔍 Why We Need Them — Twin Patches

```
   a vertical bar:

      ⬛⬜⬜⬛                PATCH 0 (top-left)      →  [0, 1, 0, 1]
      ⬛⬜⬜⬛                PATCH 2 (bottom-left)   →  [0, 1, 0, 1]
      ⬛⬜⬜⬛                                             ↑
      ⬛⬜⬜⬛                                      IDENTICAL! 😱
```

**After the embedding layer they're STILL identical:**

```
   PATCH 0  →  [0.4, 1.0, 0.4]
   PATCH 2  →  [0.4, 1.0, 0.4]       ← the transformer can't tell top from bottom!
```

> 🧒 **The Scrabble-tiles problem from Module 15** 🎲. Tip the patches into a bag, and the picture is lost. We must stamp each patch with *where it came from*.

## F.2 🔍 Adding Position Stamps Fixes It

```
   PATCH 0:  [0.4, 1.0, 0.4]  +  position 0 [0.1, 0.0, 0.0]  =  [0.5, 1.0, 0.4]
   PATCH 2:  [0.4, 1.0, 0.4]  +  position 2 [0.0, 0.0, 0.1]  =  [0.4, 1.0, 0.5]
                                                                   ↑
                                                       now DIFFERENT! ✅
```

## F.3 🔍 Adding Positions to Our "L"

```
                 patch_vector        + position            = with position
   PATCH 0:   [0.7, 0.4, 0.7]   +  [0.1, 0.0, 0.0]   =   [0.8, 0.4, 0.7]
   PATCH 1:   [0.0, 0.0, 0.0]   +  [0.0, 0.1, 0.0]   =   [0.0, 0.1, 0.0]
   PATCH 2:   [0.8, 1.0, 1.0]   +  [0.0, 0.0, 0.1]   =   [0.8, 1.0, 1.1]
   PATCH 3:   [0.3, 0.9, 0.8]   +  [0.1, 0.1, 0.0]   =   [0.4, 1.0, 0.8]
```

> 🧒 Even the **empty patch** now says something: `[0.0, 0.1, 0.0]` means *"empty, AND I'm in the top-right corner."* Even "nothing here" is useful! 📍

## F.4 🔍 ViT's Positions Are LEARNED

```
   Module 15 showed two kinds of position stamps:
      sin/cos waves (a fixed formula 🌊)
      a learned table (Project #4's position_table)

   ViT uses a LEARNED table — one row of numbers per patch position,
   trained along with everything else. 🎓
```

---

# PART G: THE CLS TOKEN — THE CLASS REPRESENTATIVE 🙋

## G.1 What It Is

```
   We add ONE extra token at the FRONT that isn't a patch at all:

      cls_token = [0.2, 0.2, 0.2]       ← learned, the SAME for every picture

   tokens:  [CLS]  [PATCH 0]  [PATCH 1]  [PATCH 2]  [PATCH 3]
            5 tokens × 3 numbers  →  shape (5, 3)
```

## G.2 🧒 Why Add a Token That Isn't Part of the Picture?

```
   A class of 16 students each knows about ONE patch.
   The teacher asks: "So... what's in the WHOLE picture?" 🤔

   Rather than asking everyone, the class elects a REPRESENTATIVE 🙋
   who LISTENS to all 16 students, then reports the summary.

   CLS is that representative. It owns no patch — so it's free
   to gather information from ALL of them!
```

## G.3 🔍 More Everyday Representatives

```
   👨‍💼 a team captain who collects everyone's ideas before a match
   📰 a newspaper editor who reads every reporter's story and writes the headline
   🧑‍⚖️ a jury foreman who listens to every juror, then announces the verdict
```

---

# PART H: ATTENTION BETWEEN PATCHES 👀

*(For tiny numbers we skip the Q/K/V grids here — a real ViT uses them exactly as in Module 15.)*

## H.1 🔍 STEP 1 — Scores: CLS · Each Token

```
   cls_token = [0.2, 0.2, 0.2]

   vs CLS      [0.2, 0.2, 0.2]:   0.2×0.2 + 0.2×0.2 + 0.2×0.2  =  0.04 + 0.04 + 0.04  =  0.12
   vs PATCH 0  [0.8, 0.4, 0.7]:   0.16 + 0.08 + 0.14                                  =  0.38
   vs PATCH 1  [0.0, 0.1, 0.0]:   0.00 + 0.02 + 0.00                                  =  0.02  ← empty
   vs PATCH 2  [0.8, 1.0, 1.1]:   0.16 + 0.20 + 0.22                                  =  0.58  ← busy
   vs PATCH 3  [0.4, 1.0, 0.8]:   0.08 + 0.20 + 0.16                                  =  0.44
```

## H.2 🔍 STEP 2 — Scale by √3 = 1.732

```
   0.12 ÷ 1.732 = 0.069
   0.38 ÷ 1.732 = 0.219
   0.02 ÷ 1.732 = 0.012
   0.58 ÷ 1.732 = 0.335
   0.44 ÷ 1.732 = 0.254
```

## H.3 🔍 STEP 3 — Softmax → look_percent

```
   e^0.069 = 1.072
   e^0.219 = 1.245
   e^0.012 = 1.012
   e^0.335 = 1.398
   e^0.254 = 1.289
             ─────
     SUM   = 6.016

      CLS        1.072 ÷ 6.016 = 17.8%   ███████
      PATCH 0    1.245 ÷ 6.016 = 20.7%   ████████
      PATCH 1    1.012 ÷ 6.016 = 16.8%   ███████    ← the empty patch: LEAST
      PATCH 2    1.398 ÷ 6.016 = 23.2%   █████████  ← the busy corner: MOST 🎯
      PATCH 3    1.289 ÷ 6.016 = 21.4%   ████████
                                ──────
                                100.0% ✓
```

## H.4 🔍 STEP 4 — Blend

```
   new CLS = 0.178×CLS + 0.207×PATCH0 + 0.168×PATCH1 + 0.232×PATCH2 + 0.214×PATCH3
           = [0.473, 0.582, 0.608]

   old CLS:  [0.200, 0.200, 0.200]    — knew NOTHING about the picture
   new CLS:  [0.473, 0.582, 0.608]    — has listened to every patch! 🙋
```

> 🧒 **The representative listened to everyone** — a bit more to the interesting patches, a bit less to the empty one. After 4 blocks of this, CLS holds a rich summary of the WHOLE picture. 🎯

## H.5 🔍 Patches Talk to EACH OTHER Too

```
   In the "7":  the TOP-BAR patch looks at the STEM patch
                → "a bar on top + a slanted stem = probably a 7!" 🔗

   In the "L":  the CORNER patch looks at the patch ABOVE it
                → "a line going up + a line going right = an L!" 🔗

   In a photo:  a patch with a WHISKER looks at a patch with a POINTY EAR
                → "whiskers + pointy ears = probably a cat!" 🐱
```

## H.6 Why "No Mask"?

```
   GPT (Module 15) = a WRITER ✍️  → masked, can't peek at future words
   ViT             = a READER 📖  → NO mask: every patch sees every patch

   🧒 A picture has no "future" — the whole thing is there at once.
      ViT reads everything together, like BERT. 👀
```

---

# PART I: THE WHOLE VISION TRANSFORMER

## I.1 Top to Bottom

```
          ┌─────────────────────────────┐
          │  picture        28 × 28      │
          └──────────────┬──────────────┘
                         ▼
          ┌─────────────────────────────┐
          │  cut into 16 patches         │   reshaping
          │  49 pixels each              │
          └──────────────┬──────────────┘
                         ▼
          ┌─────────────────────────────┐
          │  patch embedding             │   LEARNED 🎓
          │  49 → 64 numbers each        │
          └──────────────┬──────────────┘
                         ▼
          ┌─────────────────────────────┐
          │  add CLS + positions         │   reshaping
          │  17 tokens of 64             │
          └──────────────┬──────────────┘
                         ▼
          ┌─────────────────────────────┐
          │  transformer blocks × 4      │   LEARNED 🎓
          │  Module 15 — no mask         │
          └──────────────┬──────────────┘
                         ▼
          ┌─────────────────────────────┐
          │  keep only the CLS token     │   reshaping
          │  the summary — 64 numbers    │
          └──────────────┬──────────────┘
                         ▼
          ┌─────────────────────────────┐
          │  head: 64 → 10 scores        │   LEARNED 🎓
          │  "it's a 7"                  │
          └─────────────────────────────┘
```

## I.2 🔍 The Shape Journey (one batch of 128)

```
   picture          (128, 1, 28, 28)
       ↓ cut + embed
   patch_vectors    (128, 16, 64)
       ↓ put CLS in front
   tokens           (128, 17, 64)
       ↓ + positions
   tokens           (128, 17, 64)      same shape — we ADD
       ↓ 4 transformer blocks
   tokens           (128, 17, 64)      same shape — blocks stack 🧱
       ↓ keep token 0
   cls_output       (128, 64)
       ↓ head
   scores           (128, 10)          one score per digit
```

---

# PART J: ViT vs CNN — TWO WAYS TO LOOK 🔍

## J.1 Side by Side

```
  ┌──────────────────────────────────┬──────────────────────────────────┐
  │ CNN (Module 10) 🔦                │ VISION TRANSFORMER 👀             │
  ├──────────────────────────────────┼──────────────────────────────────┤
  │ looks through a small window      │ every patch sees every patch      │
  │ that slides around                │ from the very first block         │
  ├──────────────────────────────────┼──────────────────────────────────┤
  │ BUILT-IN rule: "nearby pixels     │ NO built-in rule — must LEARN     │
  │ belong together"                  │ which patches relate              │
  ├──────────────────────────────────┼──────────────────────────────────┤
  │ learns well from SMALL datasets   │ needs LOTS of data, or a          │
  │ ✅ (MNIST ~99%)                   │ pretrained start 📚               │
  ├──────────────────────────────────┼──────────────────────────────────┤
  │ links far-apart things only in    │ links far-apart patches in        │
  │ deep layers                       │ ONE step ⚡                        │
  └──────────────────────────────────┴──────────────────────────────────┘
```

## J.2 🧒 Two Ways to Explore a New City 🏙️

```
   CNN = WALKING 🚶
      You learn your own street first, then the neighbourhood,
      then the whole city. Great if your visit is short!

   ViT = A HELICOPTER 🚁
      You see the whole city at once — but it takes many flights
      before you understand how all the streets connect.
```

> 🧒 **With huge amounts of data, the helicopter wins** — that's why big modern image models, including CLIP, mostly use ViTs. With a small dataset like MNIST, walking often wins.

---

# PART K: A NEW PROBLEM — A FIXED LIST OF ANSWERS 📋

## K.1 The Trouble With Normal Classifiers

```
   Our ViT ends with:  head: 64 → 10 scores
   It can ONLY ever answer one of those 10 digits.

   A normal photo classifier is trained on a FIXED list:
      ["cat", "dog", "car", ... 1000 labels]
```

## K.2 🔍 What Goes Wrong — Three Examples

```
   show it a fox wearing sunglasses 🦊😎
      → it MUST pick one of its 1000 labels, even if none fit!

   you want to add "fox"
      → collect thousands of fox photos and RETRAIN 😩

   you want "a SLEEPING dog" vs "a RUNNING dog"
      → not in the list → impossible 🚫
```

**What if a model could understand ANY description — even ones it never trained on?** That's CLIP. 🗺️

---

# PART L: ⭐ CLIP — TWO ENCODERS, ONE SHARED MAP 🗺️

## L.1 The Name

```
   C  =  Contrastive    → learns by CONTRASTING right and wrong pairs 🧲👋
   L  =  Language       → understands words 📝
   I  =  Image          → understands pictures 🖼️
   P  =  Pre-training   → trained once, on a huge pile of data, then reused
```

## L.2 The Picture

```
          PICTURE SIDE                              WORDS SIDE

   ┌────────────────────────┐              ┌────────────────────────┐
   │  a photo of a dog 🐶    │              │  "a photo of a dog" 📝  │
   │  224 × 224 pixels      │              │  a caption              │
   └───────────┬────────────┘              └───────────┬────────────┘
               ▼                                       ▼
   ┌────────────────────────┐              ┌────────────────────────┐
   │  IMAGE ENCODER         │              │  TEXT ENCODER          │
   │  a Vision Transformer  │              │  a text Transformer    │
   └───────────┬────────────┘              └───────────┬────────────┘
               ▼                                       ▼
   ┌────────────────────────┐              ┌────────────────────────┐
   │  image_vector          │              │  text_vector           │
   │  512 numbers           │              │  512 numbers           │
   └───────────┬────────────┘              └───────────┬────────────┘
               └───────────────────┬───────────────────┘
                                   ▼
                    ┌──────────────────────────────┐
                    │     ONE SHARED MAP 🗺️          │
                    │  matching pairs land CLOSE    │
                    └──────────────────────────────┘
```

> 🧒 **Two different machines** — one reads pictures 👁️, one reads words 📝 — **but they both write their answer on the SAME map.** That's the whole trick!

## L.3 🔍 Where Does CLIP Learn From?

```
   400 MILLION (picture, caption) pairs collected from the internet:

      🖼️ a photo of a golden retriever    📝 "my dog Max at the beach"
      🖼️ a photo of a red sports car      📝 "a red Ferrari on the road"
      🖼️ a photo of a birthday cake       📝 "happy 7th birthday Lily!"
      🖼️ a photo of a mountain lake       📝 "sunrise over the lake, so peaceful"
      ... × 400,000,000
```

> 🧒 **No human labelled any of these!** The caption that already sat next to each picture on a web page IS the label. **Self-supervised** again — like your Project #5 denoiser, where the data supervised itself. 🎓

---

# PART M: MEASURING "CLOSENESS" — COSINE SIMILARITY 📐

## M.1 Step 1 — Make Every Arrow Length 1 ("Normalize")

```
   length of [a, b] = √(a² + b²)          (Pythagoras! 📐)
   normalized = divide each number by the length
```

**Four examples:**

```
   [3, 4]   length √(9 + 16)   = √25  = 5     →  [3÷5, 4÷5]   = [0.6, 0.8]
   [6, 8]   length √(36 + 64)  = √100 = 10    →  [6÷10, 8÷10] = [0.6, 0.8]   ← the SAME arrow!
   [1, 1]   length √(1 + 1)    = √2   = 1.414 →  [0.707, 0.707]
   [5, 0]   length √(25 + 0)   = 5            →  [1, 0]
```

> 🧒 **We only care which way an arrow POINTS, not how long it is** ➡️. Like a compass: "north-east" is north-east whether you walk 1 step or 100. `[3, 4]` and `[6, 8]` point the same way, so after normalizing they're identical.

## M.2 Step 2 — Dot Product of Two Length-1 Arrows

```
   same direction       [1, 0] · [ 1, 0]       =   1      → identical meaning 🟢
   in between           [1, 0] · [0.6, 0.8]    =   0.6    → fairly similar
   45° apart            [1, 0] · [0.707, 0.707]=   0.707  → quite similar
   right angle          [1, 0] · [ 0, 1]       =   0      → unrelated 😐
   opposite             [1, 0] · [−1, 0]       =  −1      → opposite 🔴
```

```
   ←──────────────────────────────────────────────────→
   −1                      0                         +1
   opposite             unrelated                  identical
   🔴                      😐                         🟢
```

> 🧒 **The "do we agree?" meter from Module 14** 📏 — but fairer, because a long arrow can't win just by being long!

---

# PART N: ⭐⭐ THE MATCHING GAME — CLIP'S TRAINING, WITH NUMBERS 🎯

## N.1 The Game

```
   Each training step shows CLIP a BATCH of pictures and their captions.
   Its job: find each picture's partner. 🧑‍🤝‍🧑

   🧒 Like a party game where everyone wrote their name on a card,
      the cards got shuffled, and you must match each person to their card!
```

Let's play with a batch of 3 (real CLIP uses about 32,000 at once!):

```
   pictures:   🐶 dog photo     🐱 cat photo     🚗 car photo
   captions:   "a dog"          "a cat"          "a car"
```

## N.2 Step 1 — The Encoders' Vectors (2 numbers each, length 1)

```
   image_vector                         text_vector
   🐶 dog photo   [ 0.96, 0.28]         "a dog"   [ 1.00, 0.00]
   🐱 cat photo   [ 0.80, 0.60]         "a cat"   [ 0.60, 0.80]
   🚗 car photo   [−0.60, 0.80]         "a car"   [−0.80, 0.60]
```

**Check one is really length 1:**
```
   dog photo:  √(0.96² + 0.28²) = √(0.9216 + 0.0784) = √1.0000 = 1 ✅
```

## N.3 Step 2 — Compare EVERY Picture With EVERY Caption (9 dot products)

```
   dog photo · "a dog" =  0.96×1.00 + 0.28×0.00          =  0.96
   dog photo · "a cat" =  0.96×0.60 + 0.28×0.80 = 0.576 + 0.224 =  0.80
   dog photo · "a car" =  0.96×(−0.80) + 0.28×0.60 = −0.768 + 0.168 = −0.60

   cat photo · "a dog" =  0.80×1.00 + 0.60×0.00          =  0.80
   cat photo · "a cat" =  0.80×0.60 + 0.60×0.80 = 0.48 + 0.48 =  0.96
   cat photo · "a car" =  0.80×(−0.80) + 0.60×0.60 = −0.64 + 0.36 = −0.28

   car photo · "a dog" = −0.60×1.00 + 0.80×0.00          = −0.60
   car photo · "a cat" = −0.60×0.60 + 0.80×0.80 = −0.36 + 0.64 =  0.28
   car photo · "a car" = −0.60×(−0.80) + 0.80×0.60 = 0.48 + 0.48 =  0.96
```

## N.4 The Grid

```
                      "a dog"     "a cat"     "a car"
                    ┌──────────┬──────────┬──────────┐
     dog photo      │ ⭐ 0.96  │   0.80   │  −0.60   │
                    ├──────────┼──────────┼──────────┤
     cat photo      │   0.80   │ ⭐ 0.96  │  −0.28   │
                    ├──────────┼──────────┼──────────┤
     car photo      │  −0.60   │   0.28   │ ⭐ 0.96  │
                    └──────────┴──────────┴──────────┘

     ⭐ = the TRUE pairs, on the diagonal ↘ — they must win every row
```

**Reading it:**

```
   ✅ every diagonal cell (0.96) is the biggest in its row
   🐶🐱 dog and cat score 0.80 with each other — both furry pets!
   🚗  the car is far from both animals (−0.60 and −0.28)
```

## N.5 Step 3 — Each Row Is a Multiple-Choice Question 📝

**"Which caption belongs to the dog photo?"** — softmax across its row:

```
   scores:   [0.96, 0.80, −0.60]

   e^0.96  = 2.612
   e^0.80  = 2.226
   e^−0.60 = 0.549
             ─────
     SUM   = 5.386

   "a dog":  2.612 ÷ 5.386 = 48.5%   ← right, but barely ahead 😬
   "a cat":  2.226 ÷ 5.386 = 41.3%
   "a car":  0.549 ÷ 5.386 = 10.2%

   loss = −ln(0.485) = 0.724
```

> 🧒 **Cosine scores are squashed between −1 and 1**, so softmax can't make a confident choice — 48% vs 41% is nearly a coin flip! We need to stretch them out. 📏

## N.6 Step 4 — The Stretch (TEMPERATURE again! 🌡️)

CLIP multiplies every score by a **scale** before softmax:

```
   ── scale 10 ──
   scores × 10:  [9.6, 8.0, −6.0]

   e^9.6  = 14,764.8
   e^8.0  =  2,981.0
   e^−6.0 =      0.0025
              ─────────
     SUM  = 17,745.8

   "a dog":  14,764.8 ÷ 17,745.8 = 83.2%   ← clearly ahead now ✅
   "a cat":   2,981.0 ÷ 17,745.8 = 16.8%
   "a car":                      ≈  0.0%

   loss = −ln(0.832) = 0.184
```

```
   ── scale 100 ── (where real CLIP ends up)
   scores × 100:  [96, 80, −60]

   "a dog":  99.99999%  →  loss ≈ 0.0000001   ← super confident 💪
```

**The three side by side:**

```
   scale     share for the right answer    loss
   ─────     ──────────────────────────    ──────────
     1               48.5%                 0.724      😬 mushy
    10               83.2%                 0.184      ✅ decisive
   100               99.99999%             0.0000001  💪 very sure
```

> 🧒 **Exactly Module 9's temperature and Module 15's √d scaling** 🌡️ — multiplying scores before softmax controls how confident it gets. CLIP doesn't hand-pick this number; it **learns** it during training, and it's capped at 100 so it can't run away.

## N.7 Step 5 — All Three Rows

```
   SCALE 1:
   row          right answer    its share    loss
   ───          ────────────    ─────────    ─────
   dog photo     "a dog"          48.5%      0.724
   cat photo     "a cat"          46.7%      0.762
   car photo     "a car"          58.2%      0.540
                                             ─────
                                   average = 0.675

   SCALE 10:
   row          right answer    its share    loss
   ───          ────────────    ─────────    ─────
   dog photo     "a dog"          83.2%      0.184
   cat photo     "a cat"          83.2%      0.184
   car photo     "a car"          99.9%      0.001   ← easy! nothing else is close
                                             ─────
                                   average = 0.123
```

> 🧒 **Look WHERE the loss is!** The dog and cat rows are hard — those captions are similar (0.80). The car row is almost free. So training will mostly work on **pulling dogs and cats apart** 🐶↔️🐱.

## N.8 Step 6 — Both Directions! 🔁

```
   ROWS:     "for each PICTURE, which caption?"     → image-to-text loss
   COLUMNS:  "for each CAPTION, which picture?"     → text-to-image loss

   CLIP loss = (rows loss + columns loss) ÷ 2
```

```
   Our grid happens to be symmetric (it's the same flipped across the
   diagonal), so both directions give 0.123:

   CLIP loss = (0.123 + 0.123) ÷ 2 = 0.123
```

> 🧒 **Why both?** So the map works both ways — searching for a picture with words, AND picking a caption for a picture. A dictionary has to work English→French AND French→English! 📖🔁

## N.9 Step 7 — Which Way Does It Push? (the Module 14 formula!)

```
   push on the dog photo's row = predicted − truth   (at scale 10)

      "a dog":  0.832 − 1  =  −0.168    → pull dog photo TOWARD "a dog"    🧲
      "a cat":  0.168 − 0  =  +0.168    → push dog photo AWAY from "a cat" 👋
      "a car":  0.000 − 0  =   0.000    → already far — leave it alone 😴
```

```
   ╔═══════════════════════════════════════════════════════════╗
   ║   CONTRASTIVE LEARNING:                                     ║
   ║      every RIGHT pair is pulled TOGETHER   🧲               ║
   ║      every WRONG pair is pushed APART      👋               ║
   ╚═══════════════════════════════════════════════════════════╝
```

> 🧒 **Like magnets** 🧲 — matching pairs attract, non-matching pairs repel. After 400 million pictures of attraction and repulsion, everything finds its place on the map!

---

# PART O: THE SHARED MAP 🗺️

## O.1 Our Batch, Drawn

Because every vector has length 1, they all sit on a circle:

```
                          ■ "a cat"
                    ● car photo        ● cat photo
              ■ "a car"
                                              ● dog photo
                                              ■ "a dog"
                             ·
                         (centre)

          ● = a photo        ■ = a caption

   🐶 dog photo and "a dog"  → point almost the SAME way  (0.96) ✅
   🐱 cat photo and "a cat"  → point almost the same way  (0.96) ✅
   dogs and cats             → nearby — both are pets      (0.80)
   🚗 the car pair           → round the other side        (−0.60)
```

> 🧒 **Photos and words now live in ONE neighbourhood** 🏘️. A picture of a dog and the *letters* "a dog" are next-door neighbours, even though one is pixels and the other is text!

## O.2 With 512 Numbers, There's Room for Everything

```
   Real CLIP uses 512 numbers, not 2 — so millions of ideas get their own spot:

      "a dog"   "a puppy"   "a dog on a beach"   "a watercolour painting of a dog"
      "a sleepy dog"   "two dogs playing"   "a dog wearing a party hat" 🎉

   similar ideas → nearby spots · different ideas → far apart
```

---

# PART P: ⭐ SUPERPOWER 1 — ZERO-SHOT CLASSIFICATION 🎯

## P.1 The Recipe

```
   STEP 1:  write a caption for every label you care about
               "a photo of a dog"   "a photo of a cat"   "a photo of a car"

   STEP 2:  run them through the TEXT encoder   →  3 text_vectors

   STEP 3:  run the new picture through the IMAGE encoder  →  1 image_vector

   STEP 4:  the most similar caption wins! 🏆
```

## P.2 🔍 Example 1 — A Photo It Has Never Seen

```
   new image_vector:  [0.98, 0.20]

   vs "a dog" [ 1.00, 0.00]:   0.98×1.00 + 0.20×0.00              =  0.980  🏆
   vs "a cat" [ 0.60, 0.80]:   0.98×0.60 + 0.20×0.80 = 0.588 + 0.160 =  0.748
   vs "a car" [−0.80, 0.60]:   0.98×(−0.80) + 0.20×0.60 = −0.784 + 0.120 = −0.664

   → "it's a dog!" 🐶
```

## P.3 🔍 Example 2 — Another New Photo

```
   new image_vector:  [−0.70, 0.71]

   vs "a dog":   −0.70×1.00 + 0.71×0.00                =  −0.700
   vs "a cat":   −0.70×0.60 + 0.71×0.80 = −0.420 + 0.568 =  0.148
   vs "a car":   −0.70×(−0.80) + 0.71×0.60 = 0.560 + 0.426 =  0.986  🏆

   → "it's a car!" 🚗
```

## P.4 ⭐ Add a Label It Was NEVER Trained On

```
   Want to know if it's a fox?  Just write  "a photo of a fox"  🦊
      → embed it → compare → done!

   NO fox photos needed.  NO retraining.  🎉
```

```
   "ZERO-SHOT" = you showed it ZERO examples of that label — and it still works!
```

> 🧒 **Like a friend who has never seen a zebra** 🦓. You say *"it's a horse with black and white stripes"* — and they spot one instantly at the zoo! Words alone are enough.

## P.5 🔍 Why "a photo of a ___" and Not Just "dog"?

```
   The internet captions CLIP learned from are sentences, like
   "a photo of my dog" — rarely a single word.

   So a caption that SOUNDS like its training data tends to work better:

      "dog"                  → okay
      "a photo of a dog"     → usually better ✅

   🧒 Speak to it in the kind of sentences it grew up reading! 📖
```

---

# PART Q: SUPERPOWER 2 — SEARCH PICTURES WITH WORDS 🔍

```
   type:   "a red car parked by the sea"
              ↓ text encoder
   a text_vector
              ↓ find the NEAREST image_vectors (Project #5's Cell 7d!)
   the photos that best match your words 🖼️🖼️🖼️
```

```
   PROJECT #5:  search with an example PICTURE  →  similar pictures
   CLIP:        search with a SENTENCE          →  matching pictures 🎉
```

> 🧒 The photos live on the map; your words drop a pin on the same map 📍; you grab the nearest photos. **That's exactly what a vector database does** — Module 30! 🗄️

---

# PART R: ⭐ WHY CLIP MATTERS FOR WHAT'S NEXT

```
   TEXT-TO-IMAGE (Module 21) 🎨
      "a cat wearing a hat"  →  CLIP's TEXT encoder  →  meaning vectors
                                                            ↓
                        they GUIDE the diffusion denoiser as it paints,
                        through CROSS-ATTENTION (Module 14!) 🔦

   PICTURE-READING CHATBOTS 💬
      models that answer questions about photos use a CLIP-style
      VISION encoder as their "eyes" 👀
```

> 🧒 **CLIP is the translator between pictures and words** 🤝 — almost every modern system that mixes images and text has a CLIP-like part inside.

---

# PART S: ⭐⭐ THE CODE — DRY RUN, LINE BY LINE

## S.1 Cutting Patches by Hand — a 4×4 Picture With Numbered Pixels

```python
import torch

picture = torch.arange(16).reshape(1, 1, 4, 4)     # pixels numbered 0 to 15
print(picture[0, 0])
```

**🔍 OUTPUT:**
```
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15]])
```

```python
patches = picture.unfold(2, 2, 2).unfold(3, 2, 2)  # cut rows, then columns, into 2s
patches = patches.reshape(1, 4, 4)                 # 4 patches, 4 pixels each
print(patches[0])
```

**🔍 OUTPUT:**
```
tensor([[ 0,  1,  4,  5],      ← PATCH 0 (top-left)
        [ 2,  3,  6,  7],      ← PATCH 1 (top-right)
        [ 8,  9, 12, 13],      ← PATCH 2 (bottom-left)
        [10, 11, 14, 15]])     ← PATCH 3 (bottom-right)
```

### 🔍 Check PATCH 0 against the picture

```
   ┌────┬────┬────┬────┐
   │ 0  │ 1  │ 2  │ 3  │      PATCH 0 is the top-left 2×2 square:
   ├────┼────┼────┼────┤         0  1
   │ 4  │ 5  │ 6  │ 7  │         4  5
   ├────┼────┼────┼────┤      flattened → [0, 1, 4, 5]  ✅
   │ 8  │ 9  │ 10 │ 11 │
   ├────┼────┼────┼────┤
   │ 12 │ 13 │ 14 │ 15 │
   └────┴────┴────┴────┘
```

### 🔍 What `.unfold(dim, size, step)` does — a tiny example

```
   a row:  [0, 1, 2, 3]

   .unfold(0, 2, 2)  →  windows of size 2, jumping 2 each time
                    →  [[0, 1], [2, 3]]

   🧒 Like cutting a strip of paper into pieces of 2 ✂️
      — once along the rows, once along the columns → squares!
```

---

## S.2 ⭐ The Clever Way — a Conv2d Does Cutting AND Embedding 🧩

```python
import torch.nn as nn

patch_embed = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=7)

pictures = torch.rand(128, 1, 28, 28)
x = patch_embed(pictures)
print("after conv:", x.shape)

x = x.flatten(2).transpose(1, 2)
print("as tokens: ", x.shape)
```

**🔍 OUTPUT:**
```
after conv: torch.Size([128, 64, 4, 4])
as tokens:  torch.Size([128, 16, 64])
```

### 🔍 Why a Conv2d works — the Module 10 link

```
   kernel_size=7   →  the window is 7 × 7  (one patch!)
   stride=7        →  it JUMPS 7 pixels at a time → windows never overlap

   28 ÷ 7 = 4 positions across, 4 down → exactly our 16 patches 🧩

   Each of the 64 filters looks at all 49 pixels of a patch
   and makes ONE number → 64 numbers per patch

   🧒 That IS "Linear(49 → 64) applied to every patch"!
```

### 🔍 Proof — the settings are identical

```
   Conv2d(1, 64, kernel 7):   64 × 1 × 7 × 7 + 64  =  3,136 + 64  =  3,200
   Linear(49, 64):            49 × 64 + 64         =  3,136 + 64  =  3,200   ✅
```

### 🔍 The shape dance, one step at a time

```
   (128, 64, 4, 4)       64 numbers at each of the 4 × 4 patch spots
        ↓ .flatten(2)    squash the 4 × 4 grid into one line of 16
   (128, 64, 16)
        ↓ .transpose(1, 2)   swap so each PATCH is a row
   (128, 16, 64)         ← 16 tokens of 64 numbers ✅
```

**A tiny version you can picture:**

```
   flatten(2) on a 2 × 2 grid of spots:     [[A, B],
                                             [C, D]]   →   [A, B, C, D]

   transpose(1, 2):    (64 numbers, 16 patches)  →  (16 patches, 64 numbers)
   🧒 Like turning a table on its side so each patch becomes a row 🔄
```

---

## S.3 A Tiny Vision Transformer

```python
class TinyViT(nn.Module):
    def __init__(self, patch=7, d=64, heads=4, blocks=4, classes=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(1, d, kernel_size=patch, stride=patch)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, d) * 0.02)    # the class rep 🙋
        self.pos_embed   = nn.Parameter(torch.randn(1, 17, d) * 0.02)   # 17 positions 📍
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, batch_first=True)
        self.blocks = nn.TransformerEncoder(layer, blocks)
        self.head   = nn.Linear(d, classes)

    def forward(self, pictures):                       # (B, 1, 28, 28)
        x = self.patch_embed(pictures)                 # (B, 64, 4, 4)
        x = x.flatten(2).transpose(1, 2)               # (B, 16, 64)
        cls = self.cls_token.expand(x.size(0), -1, -1) # (B, 1, 64)
        x = torch.cat([cls, x], dim=1)                 # (B, 17, 64)
        x = x + self.pos_embed                         # add positions 📍
        x = self.blocks(x)                             # (B, 17, 64)
        return self.head(x[:, 0])                      # CLS only → (B, 10)
```

### 🔍 Line by line

| Line | What it does 🧒 |
|------|----------------|
| `nn.Parameter(...)` | a learnable setting that isn't a layer — the CLS vector and the position table get trained too 🎓 |
| `* 0.02` | start them as tiny random numbers (Module 9's careful start) |
| `.expand(x.size(0), -1, -1)` | one CLS per picture in the batch — without copying memory |
| `torch.cat([cls, x], dim=1)` | put CLS at the FRONT of the line 🙋 |
| `x + self.pos_embed` | stamp every token with its position 📍 |
| `nn.TransformerEncoder` | the Module 15 blocks — attention + add & norm + feed-forward |
| *(no mask passed!)* | every patch sees every patch — a reader 📖 |
| `x[:, 0]` | keep only token 0 — the class rep's summary |

### 🔍 `.expand` — an example

```
   cls_token shape:  (1, 1, 64)       ONE class rep
   .expand(128, -1, -1)
   result shape:     (128, 1, 64)     the same rep "seated" in all 128 classrooms

   -1 means "keep this size as it is" 📏
   🧒 Like a teacher appearing on a video call in 128 classrooms at once —
      one teacher, no copies! 📺
```

### 🔍 `torch.cat([cls, x], dim=1)` — an example

```
   cls:  (128,  1, 64)
   x:    (128, 16, 64)
                ↓ glue along dim 1 (the "which token" dimension)
   out:  (128, 17, 64)       1 + 16 = 17 tokens ✅
```

### 🔍 Shapes for one batch

```python
model = TinyViT()
scores = model(torch.rand(128, 1, 28, 28))
print(scores.shape)
```
**🔍 OUTPUT:** `torch.Size([128, 10])` — one score per digit, for every picture ✅

### 🔍 Counting the settings — exactly

```
   patch_embed                64 × 1 × 7 × 7 + 64              =      3,200
   cls_token                  1 × 1 × 64                       =         64
   pos_embed                  1 × 17 × 64                      =      1,088
   ONE transformer block:
      attention  Q,K,V grids  3 × 64 × 64 + 3 × 64 = 12,288 + 192 = 12,480
                 output grid  64 × 64 + 64                      =  4,160
      feed-forward            64 × 128 + 128 + 128 × 64 + 64    = 16,576
      2 LayerNorms            2 × (64 + 64)                     =    256
                                                       ONE block = 33,472
   4 blocks                   4 × 33,472                       =    133,888
   head                       64 × 10 + 10                     =        650
                                                                  ─────────
                                                          TOTAL =  138,890
```

```python
print(f"{sum(p.numel() for p in model.parameters()):,}")
```
**🔍 OUTPUT:** `138,890` ✅

> 🧒 **Where's the weight?** About 96% of the settings live in the 4 transformer blocks (133,888 of 138,890). The patch cutting is cheap — the *thinking* is expensive! 🧠

### 🔍 How well would it do?

```
   Trained on MNIST, a tiny ViT like this usually lands a little BELOW
   your Project #2 CNN's ~99%.

   🧒 That's Part J in action: the CNN's built-in "nearby pixels belong
      together" rule is a big head start on a small dataset. 🚶 vs 🚁
```

---

## S.4 ⭐ The CLIP Loss — With Our Exact Numbers

```python
import torch.nn.functional as F

image_vectors = torch.tensor([[0.96, 0.28], [0.80, 0.60], [-0.60, 0.80]])
text_vectors  = torch.tensor([[1.00, 0.00], [0.60, 0.80], [-0.80, 0.60]])

image_vectors = F.normalize(image_vectors, dim=-1)    # every arrow → length 1
text_vectors  = F.normalize(text_vectors,  dim=-1)

scale   = 10
scores  = image_vectors @ text_vectors.T * scale      # the 3×3 grid, stretched
targets = torch.arange(3)                             # [0, 1, 2] — the DIAGONAL!

loss_images = F.cross_entropy(scores,   targets)      # rows:    picture → caption
loss_texts  = F.cross_entropy(scores.T, targets)      # columns: caption → picture
loss = (loss_images + loss_texts) / 2

print(scores)
print("targets:", targets)
print("loss:", round(loss.item(), 4))
```

**🔍 OUTPUT:**
```
tensor([[ 9.6000,  8.0000, -6.0000],
        [ 8.0000,  9.6000, -2.8000],
        [-6.0000,  2.8000,  9.6000]])
targets: tensor([0, 1, 2])
loss: 0.123
```
✅ Exactly the 0.123 we worked out by hand in Part N!

### 🔍 Line by line

| Line | What it does 🧒 |
|------|----------------|
| `F.normalize(..., dim=-1)` | divide every vector by its own length → length 1 (Part M) |
| `image_vectors @ text_vectors.T` | all 9 dot products in ONE line — the grid! |
| `* scale` | the stretch 🌡️ |
| `torch.arange(3)` | `[0, 1, 2]` — "row 0's answer is column 0, row 1's is column 1…" |
| `F.cross_entropy(scores, targets)` | softmax each row, then −ln of the right answer, averaged |
| `scores.T` | flip the grid → rows become columns → the other direction 🔁 |
| `/ 2` | average the two directions |

### 🔍 `image_vectors @ text_vectors.T` — the shapes

```
   image_vectors     (3, 2)       3 pictures × 2 numbers
   text_vectors.T    (2, 3)       flipped: 2 numbers × 3 captions
                ↓  @
   scores            (3, 3)       every picture vs every caption ✅

   🧒 The inner 2s match and disappear — the multiply-and-add happening!
      (The same shape rule as Q @ Kᵀ in Module 15.)
```

### ⭐ The clever line: `targets = torch.arange(3)`

```
   torch.arange(3)  →  [0, 1, 2]
   torch.arange(5)  →  [0, 1, 2, 3, 4]

   row 0 (dog photo):  right answer = column 0 ("a dog")
   row 1 (cat photo):  right answer = column 1 ("a cat")
   row 2 (car photo):  right answer = column 2 ("a car")

   🧒 Each batch is built from matching pairs IN ORDER, so the right answer
      for row i is ALWAYS column i — the diagonal! 🎯 No human labels needed.
```

---

## S.5 Using the REAL CLIP — Free From Hugging Face 🤗

### 🔍 Zero-shot classification

```python
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import requests

model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url    = "http://images.cocodataset.org/val2017/000000039769.jpg"   # two cats on a sofa
image  = Image.open(requests.get(url, stream=True).raw)
labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]

inputs  = processor(text=labels, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs   = outputs.logits_per_image.softmax(dim=1)

for label, p in zip(labels, probs[0]):
    print(f"{label:20s} {p.item():.1%}")
```

**🔍 Expected output (roughly):**
```
a photo of a cat     99.x%   🏆
a photo of a dog      0.x%
a photo of a car      0.x%
```

| Line | What it does 🧒 |
|------|----------------|
| `from_pretrained(...)` | download CLIP's already-trained settings — no training needed! 📥 |
| `CLIPProcessor` | resizes the photo to 224×224 AND turns the captions into tokens, in one step |
| `padding=True` | makes all captions the same length so they fit in one batch 📏 |
| `logits_per_image` | the stretched score grid (Part N.6), one row per image |
| `.softmax(dim=1)` | turn each row into percentages that add to 100% |

### 🔍 Decode the model's name: `clip-vit-base-patch32`

```
   clip     → the CLIP model 🗺️
   vit      → its image encoder is a Vision Transformer 👁️
   base     → the medium size (there are bigger ones)
   patch32  → it cuts photos into 32 × 32 patches 🧩

   224 ÷ 32 = 7 across, 7 down  →  7 × 7 = 49 patches + CLS = 50 tokens

   🧒 Every idea in this module is hiding inside that one name!
```

### 🔍 Try your own labels

```
   ["a photo of two cats", "a photo of one cat"]
   ["a photo of a sleeping animal", "a photo of a running animal"]
   ["a pink sofa", "a blue sofa"]

   🧒 Anything you can DESCRIBE, you can test — that's zero-shot 🎯
```

### 🔍 Search pictures with words

```python
import torch.nn.functional as F

text_inputs  = processor(text=["a red car"], return_tensors="pt", padding=True)
image_inputs = processor(images=my_photos, return_tensors="pt")   # a list of PIL images

with torch.no_grad():
    text_vector   = F.normalize(model.get_text_features(**text_inputs),  dim=-1)  # (1, 512)
    image_vectors = F.normalize(model.get_image_features(**image_inputs), dim=-1) # (N, 512)

scores = (image_vectors @ text_vector.T).squeeze(1)    # (N,)  one score per photo
best   = scores.argsort(descending=True)               # best match first
```

```
   🧒 Project #5's image search, upgraded:
      your words become a 512-number pin on the map 📍,
      and argsort ranks every photo by how close it is. 🔍
```

---

# 📋 MODULE 20 MASTER RECAP

```
   1. GOAL: connect PICTURES 🖼️ and WORDS 📝 — needed for text-to-image

   2. VISION TRANSFORMER (ViT) 👁️
         pixels as tokens = far too many (614,656 comparisons for MNIST 😰)
         → cut the picture into PATCHES 🧩 — each patch = one "word"
           MNIST: 16 patches of 49 · a photo: 196 patches of 768
         → patch embedding: Linear (or a Conv2d!) → a vector per patch
           (an empty patch → all zeros)
         → + POSITIONS 📍 (twin patches must be told apart!)
         → + a CLS token 🙋 — the class rep that summarizes everything
         → the SAME Module 15 transformer blocks, with NO mask 📖

   3. CNN vs ViT: walking 🚶 (built-in "nearby" rule, good with little data)
                  vs helicopter 🚁 (sees everything, needs lots of data)

   4. THE PROBLEM: normal classifiers know only a FIXED list of labels 📋

   5. CLIP 🗺️ = an image encoder + a text encoder → ONE shared map
         trained on 400 million (picture, caption) pairs — no human labels

   6. COSINE SIMILARITY 📐: normalize to length 1, then dot product
         1 = same direction · 0 = unrelated · −1 = opposite

   7. CONTRASTIVE TRAINING 🧲👋 — the matching game:
         a batch → an N×N grid → the DIAGONAL must win every row AND column
         loss 0.675 at scale 1  →  0.123 at scale 10
         the scale is TEMPERATURE again 🌡️ (learned, capped at 100)
         push = predicted − truth (Module 14!)

   8. SUPERPOWERS:
         zero-shot classification 🎯 — invent labels, no retraining 🦊
         text-to-image search 🔍 — the seed of vector databases 🗄️

   9. CLIP's text encoder GUIDES diffusion (Module 21) 🎨, and CLIP-style
      vision encoders are the "eyes" of picture-reading chatbots 👀
```

---

# 🤔 COMMON DOUBTS

**Q1: Why not treat every pixel as a token?**
> 🧒 Far too many! 784 tokens → 614,656 comparisons per head for MNIST, and about 2.5 billion for a 224×224 photo 😱. Patches cut that down hugely — and one pixel alone means almost nothing.

**Q2: How big should a patch be?**
> 🧒 Smaller patches → more tokens → more detail but slower 🐢. Bigger patches → fewer tokens → faster but blurrier understanding ⚡. For 224×224 photos, 16 or 32 are common (196 or 49 patches).

**Q3: What does the CLS token contain at the start?**
> 🧒 Just a learned vector — the same for every picture. It knows nothing about this picture until attention lets it listen to the patches 🙋. Its *final* value is the summary.

**Q4: Why do patches need positions?**
> 🧒 Attention sees all tokens at once, with no sense of order. Two identical patches (top and bottom of a vertical bar) would be indistinguishable without a stamp 📍 — Part F showed both came out as `[0.4, 1.0, 0.4]`.

**Q5: Why can a Conv2d do the patch embedding?**
> 🧒 A 7×7 window that jumps 7 pixels lands on each patch exactly once, and its 64 filters each mix all 49 pixels — exactly Linear(49 → 64) per patch. Same 3,200 settings! 🧩

**Q6: Is a ViT better than a CNN?**
> 🧒 With huge datasets, usually yes 🚁. With small ones like MNIST, a CNN's built-in "nearby pixels belong together" rule often wins 🚶.

**Q7: Why does CLIP normalize vectors to length 1?**
> 🧒 So only the DIRECTION matters ➡️. Otherwise a vector could score high just by being long, not by meaning the same thing.

**Q8: Why multiply the scores by a scale?**
> 🧒 Cosine scores are squashed between −1 and 1, so softmax would be mushy (48% vs 41%). Stretching lets it be decisive — temperature again 🌡️.

**Q9: Where do CLIP's labels come from?**
> 🧒 Each batch is built from matching pairs, so the right answer for row *i* is always column *i* — the diagonal IS the label! That's why `targets = torch.arange(N)` 🎯.

**Q10: Why train in both directions?**
> 🧒 So the map works both ways — picture → caption AND caption → picture. A dictionary must work English→French AND French→English 📖🔁.

**Q11: What does "zero-shot" mean?**
> 🧒 It works on a label you showed it ZERO examples of — you just describe it in words 🦓.

**Q12: Why write "a photo of a dog" instead of just "dog"?**
> 🧒 CLIP learned from internet captions, which are sentences. A prompt that sounds like its training data usually works better 📖.

**Q13: How is CLIP used in text-to-image?**
> 🧒 Its text encoder turns your prompt into meaning vectors, and the image-making model looks at them through cross-attention (Module 14) while it paints 🎨.

---

# ✅ QUICK PRACTICE

**Q1:** A 32×32 picture is cut into 8×8 patches. How many patches, how many pixels each, and how many tokens with CLS?
<details><summary>Answer</summary>
32 ÷ 8 = 4 across, 4 down → 16 patches of 64 pixels → 17 tokens.
</details>

**Q2:** A 224×224 colour photo with 32×32 patches: how many numbers in each patch?
<details><summary>Answer</summary>
32 × 32 × 3 colours = 3,072.
</details>

**Q3:** How many attention comparisons per head with 50 tokens?
<details><summary>Answer</summary>
50 × 50 = 2,500.
</details>

**Q4:** Flatten this 2×2 patch: top row [0, 1], bottom row [1, 1].
<details><summary>Answer</summary>
[0, 1, 1, 1] — left to right, top to bottom.
</details>

**Q5:** Using the Part E grid, embed the patch [0, 1, 0, 0].
<details><summary>Answer</summary>
Only pixel 1 is "on", so it's just row 1: [0.3, 0.4, 0.1].
</details>

**Q6:** Normalize [6, 8]. Then normalize [0, 5].
<details><summary>Answer</summary>
[6, 8] → length 10 → [0.6, 0.8]. [0, 5] → length 5 → [0, 1].
</details>

**Q7:** Cosine similarity of [0.6, 0.8] and [0.8, 0.6]?
<details><summary>Answer</summary>
0.6×0.8 + 0.8×0.6 = 0.48 + 0.48 = 0.96 — very similar.
</details>

**Q8:** A row of scores is [0.9, 0.1] and the first is right. What's the loss at scale 1? And at scale 10?
<details><summary>Answer</summary>
Scale 1: e^0.9 = 2.460, e^0.1 = 1.105 → share 2.460 ÷ 3.565 = 69.0% → loss 0.371. Scale 10: [9, 1] → share ≈ 99.97% → loss ≈ 0.0003 🌡️.
</details>

**Q9:** In a CLIP batch of 4 pairs, what is `targets`?
<details><summary>Answer</summary>
`torch.arange(4)` = [0, 1, 2, 3] — the diagonal of the 4×4 grid.
</details>

**Q10:** The dog photo gives "a cat" a 16.8% share. Which way does training push them, and by how much?
<details><summary>Answer</summary>
predicted − truth = 0.168 − 0 = +0.168 → push them APART 👋.
</details>

**Q11:** A new photo's vector is [0.6, 0.8]. With our three captions, which label wins?
<details><summary>Answer</summary>
"a dog": 0.60. "a cat": 0.36 + 0.64 = 1.00 🏆. "a car": −0.48 + 0.48 = 0.00. → a cat!
</details>

**Q12:** Decode `clip-vit-base-patch16`. How many tokens does a 224×224 photo become?
<details><summary>Answer</summary>
A ViT image encoder with 16×16 patches: 224 ÷ 16 = 14 → 14 × 14 = 196 patches + CLS = 197 tokens.
</details>

---

# 🎬 WHAT'S NEXT: MODULE 21 — DIFFUSION MODELS 🌫️➡️🖼️ + PROJECT #6

```
   Everything comes together:

      🗜️ the VAE (Module 18)         → squeezes pictures into a small latent space
      📝 CLIP's text encoder (M20)    → turns your words into meaning vectors
      🌫️ a denoiser (Project #5!)     → starts from pure static and removes noise,
                                        step by step, guided by your words

      "a cat wearing a hat"  →  📺 → 🌫️ → 🌁 → 🖼️ 🐱🎩

   This is how Stable Diffusion, DALL·E and Midjourney work.
```

---

*Module 20 Complete! You understand how transformers read pictures, and how CLIP puts pictures and words on one shared map! 👁️🗺️*
