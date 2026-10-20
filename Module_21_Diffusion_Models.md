# 📘 MODULE 21: Diffusion Models — From Static to Pictures 🌫️➡️🖼️

**Difficulty:** 🔴 Hard (but built entirely from things you know!)
**Time:** 300 minutes
**Prerequisite:** Modules 1-20 + Projects #1-#5
**Tools:** Google Colab, PyTorch
**Phase:** 5 — Generative Models for Images (the finale!)

---

## 📖 How to Read These Notes

- Everything explained **like you're 10 years old** and seeing it for the first time 🧒
- **Lots of small examples at every step** — every number checked with Python 🧮
- **The same names from start to finish** (see the list below)
- Pictures everywhere (ASCII diagrams) 🖼️
- 🔍 Code with what goes IN and what comes OUT

---

## 📕 THE NAME LIST

| Name | What it is | Example |
|------|-----------|---------|
| `picture` | the clean original (pixels from −1 to 1) | a pixel of `0.8` |
| `noise` | random dice rolls from the bell curve 🎲 | `−1.2` |
| `step` | how far along the noising we are | `0` (clean) to `1000` (static) |
| `KEEP` | how much of the picture survives at a step | `0.671` at our step 3 |
| `NOISE_AMOUNT` | how much static is mixed in at a step | `0.742` at our step 3 |
| `noisy` | the mixed result | `−0.353` |
| `guessed_noise` | the network's guess of the noise 🔍 | `−1.1` |
| `guess` | the network's guess of the final clean picture | `0.689` |
| `guidance_scale` | how strongly to follow the prompt 🔊 | `7.5` |

---

# PART A: WHERE WE ARE — THREE CREATORS, THREE PROBLEMS

## A.1 What You've Met So Far

```
  ┌──────────────┬──────────────────────────────────────────────────┐
  │ VAE (M18)    │ a smooth map, easy to train... but BLURRY 🌫️      │
  ├──────────────┼──────────────────────────────────────────────────┤
  │ GAN (M19)    │ SHARP pictures... but the two players FIGHT 🥊,    │
  │              │ training wobbles 🎢, and it can collapse to one    │
  │              │ kind of picture 💥                                 │
  ├──────────────┼──────────────────────────────────────────────────┤
  │ DIFFUSION 🆕 │ sharp ✨  AND  stable 🧘  AND  lots of variety 🌈   │
  │              │ (its only weakness: it's SLOW — we'll see why)    │
  └──────────────┴──────────────────────────────────────────────────┘
```

## A.2 🧒 Three Cooks

```
   🍳 The VAE cook:       always makes edible food, but it's all mushy
   🍳 The GAN cook:       makes amazing food, but argues with the food
                          critic all day — and sometimes only ever makes pizza 🍕
   🍳 The diffusion cook: makes amazing food of every kind, calmly...
                          just very, very slowly 🐢
```

---

# PART B: THE BIG IDEA — BREAK IT SLOWLY, FIX IT SLOWLY 🏖️

## B.1 🧒 The Sandcastle Story

```
   1. You build a beautiful sandcastle 🏰
   2. The tide comes in. Each wave washes away a LITTLE bit 🌊
   3. After 1,000 waves, it's just a flat pile of sand 🏖️

   Now imagine a robot that has watched MILLIONS of sandcastles
   get washed away, wave by wave. It has learned ONE skill:

      "Looking at this sand, what did the LAST wave wash away?"

   Give the robot a flat pile of sand, and it runs the movie BACKWARDS —
   undoing one wave at a time — until a castle appears! 🏰✨
```

## B.2 The Two Directions

```
   FORWARD  (easy, no learning):
      picture  →  add a little noise  × 1000  →  pure static 📺

   BACKWARD (the network learns this):
      static   →  remove a little noise × 1000 →  a NEW picture 🖼️
```

## B.3 The Picture

```
     clean       step 1       step 2       step 3     pure static
   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
   │ ▀▀▀▀█ │   │ ▀▀▀▀█·│   │·▀▀▀▀█·│   │·:·▀·:·│   │:·:·:·:│
   │    █  │──▶│  · █  │──▶│ · ·█ ·│──▶│·:·:·:·│──▶│·:·:·:·│
   │   █   │◀──│   █ · │◀──│ ·█ · ·│◀──│:·:·:·:│◀──│:·:·:·:│
   │  █    │   │ ·█    │   │ █ · · │   │·:·:·:·│   │·:·:·:·│
   └───────┘   └───────┘   └───────┘   └───────┘   └───────┘

   ──▶  add a little noise:    a fixed recipe — anyone can do it 🎲
   ◀──  remove a little noise: THE NEURAL NETWORK learns this 🎓
```

> 🧒 **The forward arrows are free** — anyone can splash paint on a picture. **The backward arrows are the clever part** — that's what the network spends all its training learning. 🎓

## B.4 🧒 Why Is It Called "Diffusion"?

```
   Drop ink into a glass of water 💧🥛
   It slowly SPREADS OUT until the whole glass is evenly grey.
   That spreading is called DIFFUSION.

   Our pictures "spread out" into noise the same way —
   and the model learns to UN-spread them. 🔄
```

## B.5 🔍 More Everyday "Slowly Fall Apart, Slowly Rebuild" Examples

```
   📷 a photo coming into focus as you turn the lens
   🗿 a sculptor chipping a statue out of a rough block, bit by bit
   🧩 a jigsaw that gets scattered piece by piece — and put back piece by piece
   🌫️ a morning fog that slowly clears to reveal the town
```

---

# PART C: THE FORWARD PROCESS — ADDING NOISE, WITH NUMBERS 🌊

## C.1 The Recipe

```
   ╔═══════════════════════════════════════════════════════════╗
   ║   noisy  =  KEEP × picture  +  NOISE_AMOUNT × noise         ║
   ╚═══════════════════════════════════════════════════════════╝

   KEEP          → how much of the picture SURVIVES   (starts at 1, shrinks)
   NOISE_AMOUNT  → how much static is MIXED IN        (starts at 0, grows)
   noise         → a bell-curve dice roll 🎲 (Module 18!)
```

> 🧒 **Like mixing paint** 🎨 — a big scoop of your picture plus a small scoop of grey static. Every step, the picture scoop shrinks and the static scoop grows.

## C.2 🔍 A 5-Step Schedule (small, so we can see it)

```
   step   KEEP    NOISE_AMOUNT    check: KEEP² + NOISE_AMOUNT²
   ────   ─────   ────────────    ────────────────────────────
    0     1.000      0.000         1.000 + 0.000 = 1.000
    1     0.949      0.316         0.900 + 0.100 = 1.000
    2     0.837      0.548         0.700 + 0.300 = 1.000
    3     0.671      0.742         0.450 + 0.550 = 1.000
    4     0.447      0.894         0.200 + 0.800 = 1.000
    5     0.141      0.990         0.020 + 0.980 = 1.000
```

> 🧒 **The two squares ALWAYS add to 1** ⚖️ — like a see-saw. As the picture side goes down, the noise side goes up by exactly the right amount, so the numbers never explode or shrink away to nothing.

## C.3 🔍 Example 1 — A Bright Pixel Dissolving

```
   picture pixel:  0.8  (fairly bright — pixels run from −1 to 1, like the GAN!)
   dice roll:     −1.2  (one random noise number, the same all the way)

   step 1:  0.949 × 0.8  +  0.316 × (−1.2)  =  0.759 − 0.379  =   0.379
   step 2:  0.837 × 0.8  +  0.548 × (−1.2)  =  0.670 − 0.657  =   0.012
   step 3:  0.671 × 0.8  +  0.742 × (−1.2)  =  0.537 − 0.890  =  −0.353
   step 4:  0.447 × 0.8  +  0.894 × (−1.2)  =  0.358 − 1.073  =  −0.716
   step 5:  0.141 × 0.8  +  0.990 × (−1.2)  =  0.113 − 1.188  =  −1.075
```

```
   pixel value
   0.8 ┤●
       │  ╲
   0.4 ┤    ●
       │      ╲
   0.0 ┤        ●
       │          ╲
  −0.4 ┤            ●
       │              ╲
  −0.8 ┤                ●
       │                  ╲
  −1.2 ┤                    ●   ← almost entirely the dice roll (−1.2) now
       └──┬────┬────┬────┬────┬── step
          0    1    2    3    4    5
```

## C.4 🔍 Example 2 — A Dark Pixel Dissolving

```
   picture pixel: −0.6  (dark)
   dice roll:      0.5

   step 1:  0.949 × (−0.6)  +  0.316 × 0.5  =  −0.569 + 0.158  =  −0.411
   step 2:  0.837 × (−0.6)  +  0.548 × 0.5  =  −0.502 + 0.274  =  −0.228
   step 3:  0.671 × (−0.6)  +  0.742 × 0.5  =  −0.403 + 0.371  =  −0.032
   step 4:  0.447 × (−0.6)  +  0.894 × 0.5  =  −0.268 + 0.447  =   0.179
   step 5:  0.141 × (−0.6)  +  0.990 × 0.5  =  −0.085 + 0.495  =   0.410

   It started dark (−0.6) and ended near the dice roll (0.5) — the picture is gone!
```

## C.5 🔍 Example 3 — Same Pixel, Different Dice 🎲

```
   pixel 0.8 with dice −1.2  →  step 5:  −1.075
   pixel 0.8 with dice  0.7  →  step 5:   0.806

   At step 5 the value is almost JUST the dice:
      0.990 × (−1.2) = −1.188   (plus a tiny 0.113 of picture)
      0.990 ×   0.7  =  0.693   (plus a tiny 0.113 of picture)

   🧒 By the end, the picture has almost completely disappeared —
      what's left is mostly random dice. 📺
```

## C.6 ⭐ The Shortcut: Jump Straight to ANY Step ⏩

```
   You do NOT need to add noise 500 times to reach step 500!
   The recipe jumps there in ONE line:

      noisy at step 500 = KEEP(500) × picture + NOISE_AMOUNT(500) × noise
```

> 🧒 **Like skipping to minute 25 of a video** ⏩ instead of watching the first 24 minutes. This shortcut makes training fast — we can make a noisy picture at *any* step instantly.

## C.7 🔍 The Real Schedule — 1,000 Steps

```
   step     KEEP     NOISE_AMOUNT     what you'd see
   ────     ─────    ────────────     ──────────────
      1     1.000       0.010         the picture, untouched (1% static)
    100     0.947       0.321         a little speckled
    250     0.724       0.690         the digit is still clear
    500     0.280       0.960         a faint ghost of the digit 👻
    750     0.058       0.998         almost nothing left
   1000     0.006       1.000         pure static 📺
```

```
   step    KEEP (the picture) 🖼️              NOISE_AMOUNT (the static) 📺
   ────    ─────────────────────────────    ─────────────────────────────
      1    ██████████████████████ 1.000                            0.010
    100    █████████████████████  0.947     ▒▒▒▒▒▒▒                0.321
    200    ██████████████████     0.812     ▒▒▒▒▒▒▒▒▒▒▒▒▒          0.584
    300    ██████████████         0.630     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒      0.777
    400    ██████████             0.442     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒   0.897
    500    ██████                 0.280     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  0.960
    600    ████                   0.161     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 0.987
    700    ██                     0.083     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 0.997
    800    █                      0.039     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 0.999
    900                           0.017     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 1.000
   1000                           0.006     ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ 1.000

   🧒 The picture bar shrinks while the static bar grows — they cross
      just after step 250, and by step 1000 the picture is gone.
```

---

# PART D: ⭐ THE DENOISER'S JOB — SPOT THE SPECKLES 🔍

## D.1 The Clever Twist

The network is **NOT** asked to redraw the clean picture. It's asked:

```
   ╔═══════════════════════════════════════════════════════════╗
   ║   "Look at this noisy picture. WHICH NOISE was added?"      ║
   ╚═══════════════════════════════════════════════════════════╝
```

## D.2 🧒 Why Guess the Noise Instead of the Picture?

```
   A photo with dust on it 📷✨

   JOB A:  "Redraw the whole photo perfectly from memory."   😰 hard!
   JOB B:  "Point at every speck of dust."                   😊 much easier!

   Once you know exactly where the dust is, you just WIPE IT OFF —
   and the clean photo is what's left. 🧹
```

**More examples of "spot the problem, don't redo everything":**

```
   📝 Proofreading: circle the typos — don't rewrite the whole essay
   🎵 Tuning a guitar: find which string is off — don't build a new guitar
   🧹 Cleaning a window: find the smudges — don't replace the glass
```

## D.3 🔍 One Training Step — Every Number

```
   1. take a real picture              (4 pixels, to keep it small)
   2. pick a RANDOM step               → step 3
   3. roll RANDOM noise                → true noise = [−1.2, 0.5, 0.3, −0.7]
   4. make the noisy picture           → using the recipe (Part C)
   5. the network guesses the noise    → guessed    = [−0.9, 0.4, 0.6, −0.7]
   6. score it with MSE (Module 2!):

      differences:  [−0.9 − (−1.2),   0.4 − 0.5,   0.6 − 0.3,   −0.7 − (−0.7)]
                  = [ 0.3,           −0.1,         0.3,           0.0 ]
      squared:      [0.09,  0.01,  0.09,  0.00]
      SUM:          0.19
      average:      0.19 ÷ 4 = 0.0475

      loss = 0.0475
```

```
   ⭐ It's JUST MSE! No detective, no tug-of-war, no fighting.
      That's why diffusion training is CALM and STABLE 🧘 — unlike GANs 🥊.
```

## D.4 The Training Step as a Picture

```
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │ clean picture│   │ random step  │   │ random noise │
   │  from MNIST  │   │  say step 3  │   │ (the answer!)│──────────┐
   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘          │
          └──────────────────┼──────────────────┘                  │
                             ▼                                     │
              ┌──────────────────────────────┐                     │
              │ mix: KEEP × picture +        │                     │
              │      NOISE_AMOUNT × noise    │                     │
              └──────────────┬───────────────┘                     │
                             ▼                                     │
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐          │
   │ noisy picture│──▶│   DENOISER   │──▶│ guessed noise│          │
   │  (the input) │   │ (also told   │   │ (the output) │          │
   └──────────────┘   │  the step)   │   └──────┬───────┘          │
                      └──────────────┘          ▼                  │
                                   ┌──────────────────────────┐    │
                                   │ compare: MSE loss         │◀───┘
                                   │ guessed vs TRUE noise     │
                                   └──────────────────────────┘
```

> 🧒 **You've built this before!** It's your **Project #5 denoiser** with two upgrades: it's told *how noisy* the picture is (the step number), and it points at the noise instead of redrawing the picture. 🧹

## D.5 🔍 A Good Noise Guess = A Recovered Picture

If we know the noise, we can **undo the recipe**:

```
   recipe:     noisy   = KEEP × picture + NOISE_AMOUNT × noise
   undo it:    picture = (noisy − NOISE_AMOUNT × noise) ÷ KEEP
```

### Example A — our pixel at step 3

```
   noisy = −0.353,   KEEP = 0.671,   NOISE_AMOUNT = 0.742,   real pixel = 0.800

   guess −1.2 (PERFECT):  (−0.353 − 0.742 × −1.2) ÷ 0.671 = 0.537 ÷ 0.671 = 0.800  ✅ exact!
   guess −1.1 (close):    (−0.353 − 0.742 × −1.1) ÷ 0.671 = 0.463 ÷ 0.671 = 0.689  🟡
   guess −0.9 (off):      (−0.353 − 0.742 × −0.9) ÷ 0.671 = 0.315 ÷ 0.671 = 0.468  ❌
```

### Example B — the same pixel at step 2

```
   noisy = 0.012,   KEEP = 0.837,   NOISE_AMOUNT = 0.548

   guess −1.2 (PERFECT):  (0.012 − 0.548 × −1.2) ÷ 0.837 = 0.800  ✅
   guess −1.0 (off):      (0.012 − 0.548 × −1.0) ÷ 0.837 = 0.669  🟡
```

> 🧒 **A perfect noise guess gives back the perfect picture.** The better the network spots the speckles, the cleaner the picture it recovers. 🎯

---

# PART E: WHY THE NETWORK NEEDS THE STEP NUMBER ⏱️

## E.1 The Same Speckle Means Different Things

```
   step 5 (of 1000):    "barely noisy — just polish tiny specks" ✨
   step 500:            "half static — the shape is a faint ghost" 👻
   step 995:            "almost pure static — make big, bold guesses" 📺
```

> 🧒 **Like a doctor who needs to know how long you've been ill** 🩺. Same cough, different treatment on day 1 and in week 3! The step number tells the denoiser how much noise to expect.

## E.2 How the Step Number Gets In

```
   the step number  →  a short vector of numbers  →  taped onto the input

   two ways to make that vector (both from Module 15!):
      🌊 sin/cos waves — like positional encoding
      📋 a learned table — like Project #4's position_table
```

---

# PART F: ⭐ GENERATING A BRAND-NEW PICTURE

## F.1 The Recipe — Run the Movie Backwards ⏪

```
   STEP A:  start with PURE STATIC 📺  (just random dice rolls)

   STEP B:  at each step:
               1. the network guesses the NOISE
               2. from that, make a GUESS of the final picture
               3. move a LITTLE toward that guess (not all the way!)

   STEP C:  repeat until step 0  →  a brand-new picture 🖼️
```

## F.2 🔍 Example 1 — Watch ONE Pixel Come to Life

Using our 5-step schedule, starting from random static of **0.50**:

```
   step   pixel     network's guess     noise the network     pixel after
          now       of the FINAL pixel  thinks is mixed in    this step
   ────   ─────     ──────────────────  ──────────────────    ───────────
    5     0.500          0.40 🌫️              0.448               0.580
    4     0.580          0.62                0.338               0.667
    3     0.667          0.74                0.229               0.745
    2     0.745          0.79                0.153               0.798
    1     0.798          0.80 ✨             0.123               0.800  🎉
```

### 🔍 How ONE row is calculated (step 5)

```
   1. the network looks at 0.500 and guesses the final pixel ≈ 0.40

   2. so the noise mixed in must be:
         (pixel now − KEEP₅ × guess) ÷ NOISE₅
      = (0.500 − 0.141 × 0.40) ÷ 0.990
      = (0.500 − 0.056) ÷ 0.990
      = 0.444 ÷ 0.990
      = 0.448

   3. step back to step 4 using STEP 4's amounts:
         KEEP₄ × guess + NOISE₄ × noise
      = 0.447 × 0.40 + 0.894 × 0.448
      = 0.179 + 0.401
      = 0.580
```

### 🔍 And step 4, the same way

```
   1. the network looks at 0.580 and now guesses ≈ 0.62  (clearer!)
   2. noise = (0.580 − 0.447 × 0.62) ÷ 0.894 = (0.580 − 0.277) ÷ 0.894 = 0.338
   3. step 3: 0.671 × 0.62 + 0.742 × 0.338 = 0.416 + 0.251 = 0.667
```

### 🔍 Read the columns!

```
   the GUESS gets clearer:    0.40 → 0.62 → 0.74 → 0.79 → 0.80    🌫️ → ✨
   the NOISE shrinks:         0.448 → 0.338 → 0.229 → 0.153 → 0.123
   the PIXEL settles:         0.500 → 0.580 → 0.667 → 0.745 → 0.800
```

> 🧒 **Like a photo slowly coming into focus** 📷. At first everything is fuzzy, so the guess is rough. Each step there's less static in the way, so the network sees more clearly and fixes its guess. 🔍

## F.3 🔍 Example 2 — A Different Start Becomes a DARK Pixel

Start from random static of **−0.70** instead:

```
   step   pixel now   network's guess    noise mixed in    pixel after
   ────   ─────────   ───────────────    ──────────────    ───────────
    5      −0.700        −0.30 🌫️           −0.664           −0.728
    4      −0.728        −0.55              −0.539           −0.769
    3      −0.769        −0.68              −0.422           −0.800
    2      −0.800        −0.74              −0.330           −0.806
    1      −0.806        −0.75 ✨           −0.300           −0.750  🎉
```

```
   start  0.50  →  a BRIGHT pixel (0.80)  →  maybe part of a "7"
   start −0.70  →  a DARK pixel  (−0.75)  →  maybe the background of a "3"
```

> 🧒 Notice the pixel even **wobbles a little** at the end (−0.806 then −0.750) — the *guess* keeps improving, and the pixel follows it home. Real generation wobbles like this too!

## F.4 ⭐ Why Not Jump Straight to the First Guess?

```
   At step 5, the network's first guess was 0.40 — fuzzy and unsure 🌫️
   (remember: MSE "sits on the fence" when unsure — Project #5!)

   ONE BIG JUMP:    use 0.40 as the answer           →  a BLURRY picture ❌
   SMALL STEPS:     move a little, look again, fix   →  ends at 0.80, SHARP ✨
```

> 🧒 **Like walking down a dark staircase** 🪜. You don't jump to the bottom — you take one step, feel for the next, take another. Each small step lets the network check and correct itself. **That's the secret of diffusion's sharpness!**

## F.5 🔍 A Different Start → A Different Picture 🎲

```
   start from a random patch of static A   →  a "7"  ✏️
   start from a random patch of static B   →  a "3"  ✏️
   start from a random patch of static C   →  a "0"  ✏️

   🧒 Every random start leads to a DIFFERENT picture —
      so diffusion naturally produces VARIETY 🌈. No mode collapse!
```

---

# PART G: WHY DIFFUSION WINS (and Its One Weakness)

## G.1 The Scorecard

```
  ┌─────────────┬───────────┬────────────────┬───────────────┬──────────────┐
  │             │ SHARP? ✨ │ VARIETY? 🌈    │ STABLE? 🧘    │ FAST? ⚡      │
  ├─────────────┼───────────┼────────────────┼───────────────┼──────────────┤
  │ VAE (M18)   │ ❌ blurry │ ✅             │ ✅            │ ✅ 1 step     │
  │ GAN (M19)   │ ✅        │ ❌ can collapse │ ❌ wobbly     │ ✅ 1 step     │
  │ DIFFUSION   │ ✅        │ ✅             │ ✅ plain MSE  │ ❌ many steps │
  └─────────────┴───────────┴────────────────┴───────────────┴──────────────┘
```

## G.2 Why Each Tick

```
   SHARP    → small steps let it commit to details one at a time 🪜
   VARIETY  → every training picture was noised, so it learned the way back
              to ALL kinds of pictures — and every start is different 🎲
   STABLE   → one network, one simple MSE loss — no fighting 🧘
   SLOW     → it runs the network once PER STEP 🐢
```

## G.3 🔍 How Slow? Count the Network Runs

```
   VAE or GAN:                       1 network run per picture        ⚡
   diffusion, all training steps:    1,000 network runs per picture   🐢
   diffusion, a smart sampler:       about 20 to 50 runs              🏃
   with guidance (Part J):           ×2  →  about 40 to 100 runs
```

## G.4 The Speed Fix 🏃

```
   TRAINING:     learn all 1,000 tiny steps
   GENERATING:   smarter "samplers" skip ahead → about 20 to 50 steps

   🧒 Like a staircase where you learn every single step, but later
      you can safely take them two or three at a time! 🪜🏃

   1,000 ÷ 50 = 20 times fewer network runs ⚡
```

---

# PART H: WHAT DOES THE DENOISER NETWORK LOOK LIKE?

## H.1 The U-Net — Shaped Like a "U"

```
        noisy picture                              guessed noise
             │                                          ▲
             ▼                                          │
        ┌─────────┐ ─────────── skip ──────────▶ ┌─────────┐
        │  wide   │                               │  wide   │
        └────┬────┘                               └────▲────┘
             ▼                                         │
          ┌──────┐ ─────────── skip ─────────▶ ┌──────┐
          │narrow│                              │narrow│
          └──┬───┘                              └──▲───┘
             ▼                                     │
              └───────────▶ ┌────────┐ ───────────┘
                            │ middle │
                            └────────┘

        squeezes DOWN ⬇️                       grows back UP ⬆️
```

## H.2 🧒 You Know Both Halves!

```
   squeezes down, grows back up   →  your AUTOENCODER (Module 17) 🗜️🎨
   skip connections across        →  ResNet's highway (Module 10) 🛣️
                                     they carry fine details straight across,
                                     so they're not lost in the squeeze
```

> 🧒 **Why the skip connections?** Squeezing loses detail (the suitcase from Module 17 🧳). The skips are like **passing a note around the bottleneck** 📝 — "don't forget, there was a sharp edge right here!"

## H.3 The Newer Way: Transformers

```
   Newer models swap the U-Net for a TRANSFORMER working on PATCHES —
   exactly Module 20's ViT idea! 👁️🧩
```

---

# PART I: ⭐ FROM WORDS TO PICTURES — TEXT CONDITIONING 📝➡️🖼️

So far the model makes *some* picture. How do we make it draw **"a cat wearing a hat"**? Three ingredients.

## I.1 Ingredient 1 — CLIP Turns Words Into Vectors (Module 20!)

```
   "a cat wearing a hat"  →  CLIP's text encoder  →  one vector per word-piece

      "a"        →  [ ... 768 numbers ... ]
      "cat"      →  [ ... ]
      "wearing"  →  [ ... ]
      "a"        →  [ ... ]
      "hat"      →  [ ... ]
```

## I.2 Ingredient 2 — Cross-Attention: Each Patch Reads the Words (Module 14!)

Inside the denoiser, every patch of the picture becomes a **query** 🙋 and looks at the **words**.

### 🔍 Example 1 — a patch at the TOP OF THE HEAD

```
   scores:         "a"    "cat"   "wearing"   "a"    "hat"
                   0.5     1.9      1.2       0.5     2.9

   e^:             1.649   6.686    3.320     1.649   18.174      SUM = 31.478

   look_percent:   5.2%    21.2%    10.5%     5.2%    57.7%  🎩
                                                        ↑
                                  this patch mostly listens to "hat"!
```

### 🔍 Example 2 — a patch on the FACE

```
   scores:         "a"    "cat"   "wearing"   "a"    "hat"
                   0.5     2.8      1.0       0.5     1.2

   look_percent:   6.4%    63.8%    10.5%     6.4%    12.9%  🐱
                             ↑
                this patch mostly listens to "cat"!
```

```
   ┌──────────────────────┐
   │      🎩  ← listens    │    "hat"
   │    ╱    ╲   to "hat"  │
   │   │ 🐱   │ ← listens  │    "cat"
   │   │ face │   to "cat" │
   │    ╲____╱             │
   └──────────────────────┘
```

> 🧒 **Module 14's cross-attention — the Hindi decoder looking at English words!** 🔦 Here, each picture patch checks the sentence to find out what *it* should become.

## I.3 Ingredient 3 — Classifier-Free Guidance: Turn Up the Prompt 🔊

### 🔍 How the model learns it

```
   During training, the caption is randomly HIDDEN about 10% of the time.

   So the model learns BOTH:
      denoising WITH a prompt      ("a cat wearing a hat")
      denoising WITHOUT a prompt   ("" — just make SOMETHING)
```

### 🔍 How it's used — run TWICE, exaggerate the difference

```
   noise guess WITHOUT the prompt:   0.40
   noise guess WITH the prompt:      0.46
   difference the prompt makes:      0.46 − 0.40 = 0.06

   final guess = without + guidance_scale × difference
```

```
   guidance_scale   final guess                     what you get
   ──────────────   ───────────                     ────────────
        0           0.40 + 0   × 0.06 = 0.40        ignores your prompt 🤷
        1           0.40 + 1   × 0.06 = 0.46        follows it, but weakly
        3           0.40 + 3   × 0.06 = 0.58        clearer
        7.5         0.40 + 7.5 × 0.06 = 0.85        a popular default ✅
       15           0.40 + 15  × 0.06 = 1.30        overdone — harsh colours 🔥
```

> 🧒 **Like a music equalizer** 🎚️ — find "the part the prompt adds", then turn its volume up. A bit louder = a picture that really matches your words. Too loud = distorted! (A cousin of temperature 🌡️ — another "how strongly" knob.)

### 🔍 Another way to picture it

```
   without the prompt ●───────▶● with the prompt
                      0.40     0.46
                               └────────────────────────▶● guidance 7.5
                                                         0.85

   🧒 "You were heading THAT way because of the prompt? Then go
      MUCH further that way!" 🏃
```

---

# PART J: ⭐ STABLE DIFFUSION — EVERYTHING COMES TOGETHER 🎉

## J.1 The Speed Trick: Work in the VAE's Small Space (Module 18!)

```
   a 512 × 512 colour picture  =  512 × 512 × 3  =  786,432 numbers
                                          ↓   VAE encoder 🗜️
   a 64 × 64 × 4 latent        =   64 ×  64 × 4  =   16,384 numbers

   786,432 ÷ 16,384 = 48 times smaller!
```

**The same 48× squeeze at other sizes:**

```
   picture             numbers       latent          numbers    smaller by
   ───────             ───────       ──────          ───────    ──────────
   256 × 256 × 3       196,608       32 × 32 × 4       4,096       48×
   512 × 512 × 3       786,432       64 × 64 × 4      16,384       48×
   1024 × 1024 × 3   3,145,728      128 × 128 × 4     65,536       48×
```

```
   Diffusion runs on the SMALL latent → about 48× less work per step ⚡
   This is called LATENT DIFFUSION.
```

> 🧒 **Like planning a house on a small sketch** 📐 instead of building full-size walls every time you change your mind. Do all the thinking small, then build the real thing once at the end! 🏠

## J.2 The Whole Machine

```
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │   your prompt     │──▶│ CLIP text encoder │──▶│   word vectors    │
   │"a cat wearing a   │   │    (Module 20)    │   │   (the meaning)   │
   │     hat"          │   └──────────────────┘   └────────┬─────────┘
   └──────────────────┘                                    │ cross-attention
                                                            ▼ (Module 14)
   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
   │   random noise    │──▶│  U-Net denoiser   │──▶│   clean latent    │
   │ 64 × 64 × 4 latent│   │ removes a little  │   │   64 × 64 × 4     │
   └──────────────────┘   │      noise        │   └────────┬─────────┘
                          └────────▲─────────┘            │
                                   │                       ▼
                                   └─ repeat ~50 ─┘  ┌──────────────────┐
                                                     │   VAE decoder     │
                                                     │   (Module 18)     │
                                                     └────────┬─────────┘
                                                              ▼
                                                     ┌──────────────────┐
                                                     │   your picture    │
                                                     │   512 × 512 🐱🎩   │
                                                     └──────────────────┘
```

## J.3 🔍 Every Part, and Where You Learned It

```
   📝 CLIP text encoder     → Module 20   (words → meaning vectors)
   🔦 cross-attention       → Module 14   (picture patches read the words)
   🌫️ the denoiser          → Project #5 + this module (spot the speckles)
   🗜️🎨 the VAE              → Module 18   (squeeze to latent, decode to picture)
   🛣️ skip connections      → Module 10   (ResNet's highway)
   🌊 the step number        → Module 15   (sin/cos position waves)
   🎲 the bell-curve dice    → Module 18   (randn)
   📏 the MSE loss           → Module 2    (the very first loss you learned!)
```

> 🧒 **Every single part of Stable Diffusion is something you've already learned.** 🎉 Phase 5 wasn't a list of separate topics — it was building the pieces of this ONE machine!

## J.4 🔍 One Full Generation, Counted

```
   1. CLIP reads your prompt                         →  1 run
   2. the U-Net, 50 steps × 2 (with + without prompt) →  100 runs
   3. the VAE decoder turns the latent into a picture →  1 run

   ≈ 102 network runs for one picture 🖼️
   (and that's why a GPU helps so much! ⚡)
```

---

# PART K: ⭐⭐ THE CODE — DRY RUN, LINE BY LINE

## K.1 The Noise Schedule

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

T = 1000
betas        = torch.linspace(1e-4, 0.02, T)       # a tiny bit of noise per step
alphas       = 1 - betas                           # how much survives ONE step
alpha_bar    = torch.cumprod(alphas, dim=0)        # how much survives from the START
keep         = alpha_bar.sqrt()                    # KEEP
noise_amount = (1 - alpha_bar).sqrt()              # NOISE_AMOUNT

for step in [0, 99, 249, 499, 749, 999]:
    print(f"step {step+1:>4}:  keep {keep[step]:.3f}   noise {noise_amount[step]:.3f}")
```

**🔍 OUTPUT:**
```
step    1:  keep 1.000   noise 0.010
step  100:  keep 0.947   noise 0.321
step  250:  keep 0.724   noise 0.690
step  500:  keep 0.280   noise 0.960
step  750:  keep 0.058   noise 0.998
step 1000:  keep 0.006   noise 1.000
```
✅ Exactly the table in Part C.7.

### 🔍 Line by line

| Line | What it does 🧒 |
|------|----------------|
| `torch.linspace(1e-4, 0.02, T)` | 1,000 tiny noise amounts, growing evenly from 0.0001 to 0.02 |
| `alphas = 1 - betas` | how much SURVIVES each single step (0.9999 down to 0.98) |
| `torch.cumprod(alphas, dim=0)` | multiply them all up → how much survives from the start to step t |
| `.sqrt()` | turns those into KEEP and NOISE_AMOUNT (Part C) ✅ |
| `step+1` in the print | Python counts from 0, but we call the first step "step 1" 📏 |

### 🔍 `torch.linspace` — an example

```
   torch.linspace(0, 1, 5)   →   [0.00, 0.25, 0.50, 0.75, 1.00]

   🧒 "Give me 5 evenly spaced numbers from 0 to 1" — like marks on a ruler 📏
```

### 🔍 `torch.cumprod` — two examples

```
   survive each step:   [0.9,  0.8,  0.5]
   cumprod:             [0.9,  0.9 × 0.8,  0.9 × 0.8 × 0.5]
                      = [0.9,  0.72,       0.36]

   survive each step:   [0.5,  0.5,  0.5,  0.5]
   cumprod:             [0.5,  0.25, 0.125, 0.0625]

   🧒 Like a cake losing a slice each day 🍰 — after 3 days you have
      90% × 80% × 50% = 36% left. The losses MULTIPLY up!
```

---

## K.2 Making a Noisy Picture (the shortcut ⏩)

```python
def make_noisy(picture, step):
    noise = torch.randn_like(picture)                 # the dice rolls 🎲
    k = keep[step].view(-1, 1)                        # (B,) → (B, 1)
    n = noise_amount[step].view(-1, 1)
    return k * picture + n * noise, noise             # the noisy picture AND the answer
```

### 🔍 A tiny dry run — 2 pictures, 3 pixels each

```
   picture = [[ 0.8, −0.6,  0.0],          step = [2, 4]
              [ 0.5,  0.5, −1.0]]          (a different step for each picture!)

   (using the 5-step toy schedule from Part C, so the numbers stay small)

   keep[step]          = [0.837, 0.447]     → .view(-1, 1) →  [[0.837],
                                                               [0.447]]

   k × picture:
      row 1:  0.837 × [ 0.8, −0.6,  0.0]  =  [ 0.670, −0.502,  0.000]
      row 2:  0.447 × [ 0.5,  0.5, −1.0]  =  [ 0.224,  0.224, −0.447]

   (then n × noise is added to each row the same way)
```

### ⭐ Why `.view(-1, 1)`?

```
   picture:       (128, 784)      128 pictures, 784 pixels each
   keep[step]:    (128,)          one KEEP number per picture
   .view(-1, 1):  (128, 1)        → now it lines up with each ROW

   (128, 1) × (128, 784)  →  each picture's KEEP multiplies ALL 784 of its pixels ✅

   🧒 One stamp per picture, pressed onto every pixel of that picture! 🔖
```

### 🔍 Why return the noise too?

```
   return k * picture + n * noise, noise
                                    ↑
                            THE ANSWER for training!

   🧒 We made the noise ourselves, so we know exactly what it was.
      That's the answer sheet the network is marked against. 📝
```

---

## K.3 A Tiny Denoiser (picture + step → guessed noise)

```python
class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.step_table = nn.Embedding(T, 64)          # step number → 64 numbers ⏱️
        self.net = nn.Sequential(
            nn.Linear(784 + 64, 512), nn.ReLU(),
            nn.Linear(512, 512),      nn.ReLU(),
            nn.Linear(512, 784),                       # ⭐ NO squashing at the end!
        )

    def forward(self, noisy, step):
        step_vector = self.step_table(step)                       # (B, 64)
        return self.net(torch.cat([noisy, step_vector], dim=1))   # (B, 784)
```

### ⭐ Why NO Sigmoid or Tanh at the end?

```
   our earlier DECODERS output PIXELS  → squashed to 0..1 (Sigmoid) or −1..1 (Tanh)
   this network outputs NOISE          → which can be ANY size

      noise values you might need:   −2.7,  −1.2,  0.4,  1.9,  3.1

      Tanh would squash 1.9 → 0.956 and 3.1 → 0.996 ❌
      The network could NEVER guess big noise correctly! 🚫
```

### 🔍 `nn.Embedding(T, 64)` — a table with one row per step

```
   step 0    →  row 0    →  64 numbers
   step 1    →  row 1    →  64 numbers
   ...
   step 999  →  row 999  →  64 numbers

   🧒 Exactly like Module 12's word table — but the "words" are step numbers! 📋
      (Module 15's sin/cos waves would work too.)
```

### 🔍 `torch.cat([noisy, step_vector], dim=1)`

```
   noisy:        (B, 784)     the picture
   step_vector:  (B, 64)      the "how noisy?" tag
                    ↓ tape them side by side
   input:        (B, 848)     784 + 64 = 848 ✅

   🧒 We staple a label onto the picture saying how noisy it is! 🏷️
```

### 🔍 Counting the settings — exactly

```
   step_table:   Embedding(1000, 64)    1000 × 64            =     64,000
   Linear(848, 512):                     848 × 512 + 512     =    434,688
   Linear(512, 512):                     512 × 512 + 512     =    262,656
   Linear(512, 784):                     512 × 784 + 784     =    402,192
                                                               ─────────
                                                       TOTAL = 1,163,536
```

---

## K.4 One Training Step

```python
picture = clean_batch.view(clean_batch.size(0), -1) * 2 - 1   # flatten, pixels −1..1
step    = torch.randint(0, T, (picture.size(0),))             # a random step per picture 🎲

noisy, true_noise = make_noisy(picture, step)
guessed_noise     = model(noisy, step)
loss = F.mse_loss(guessed_noise, true_noise)                  # just MSE! 🧘

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 🔍 Line by line

| Line | What it does 🧒 |
|------|----------------|
| `.view(B, -1) * 2 - 1` | flatten to 784, and rescale 0..1 → −1..1 (like the GAN) |
| `torch.randint(0, T, (B,))` | a random step for EACH picture, e.g. `[512, 3, 877, 241, ...]` |
| `make_noisy(picture, step)` | the shortcut ⏩ — jump each picture to its own step |
| `model(noisy, step)` | guess the noise, knowing how noisy it is |
| `F.mse_loss(guessed, true)` | how wrong was the guess? (Part D.3) |
| `zero_grad / backward / step` | the same three training lines since Module 7 🔁 |

### 🔍 Why a different random step for every picture?

```
   step = [512, 3, 877, 241, ...]

      picture 1 practises a HALF-noisy clean-up   👻
      picture 2 practises a TINY polish           ✨
      picture 3 practises a HUGE clean-up         📺
      picture 4 practises a medium clean-up

   🧒 One batch practises easy, medium AND hard jobs all at once! 🎯
      Over many batches, every one of the 1,000 steps gets practised.
```

---

## K.5 Generating: Walk Back From Static ⏪

```python
model.eval()
with torch.no_grad():
    x = torch.randn(16, 784)                                    # start: pure static 📺
    for t in reversed(range(T)):                                # 999, 998, ..., 0
        step  = torch.full((16,), t)
        noise = model(x, step)                                  # 1. guess the noise
        guess = ((x - noise_amount[t] * noise) / keep[t]).clamp(-1, 1)   # 2. guess the picture
        if t > 0:
            x = keep[t-1] * guess + noise_amount[t-1] * noise   # 3. step back a little
        else:
            x = guess                                           # the finished picture 🎉

new_digits = (x + 1) / 2                                        # back to 0..1 for drawing
```

### 🔍 Line by line

| Line | What it does 🧒 |
|------|----------------|
| `torch.randn(16, 784)` | 16 pictures of pure static — 16 different starts → 16 different digits 🎲 |
| `reversed(range(T))` | count DOWN from 999 to 0 — the movie backwards ⏪ |
| `torch.full((16,), t)` | the same step number for all 16 pictures |
| `model(x, step)` | 1. guess the noise |
| `(x − noise_amount × noise) / keep` | 2. undo the recipe → a guess of the picture (Part D.5) |
| `.clamp(-1, 1)` | keep that guess inside the valid pixel range 🚧 |
| `keep[t-1] × guess + noise_amount[t-1] × noise` | 3. step back one — **exactly Part F's table!** |
| `x = guess` at `t == 0` | the last step: the guess IS the finished picture |
| `(x + 1) / 2` | undo the −1..1 rescale, so we can draw it |

### 🔍 `reversed(range(5))` and `torch.full` — examples

```
   range(5)             →  0, 1, 2, 3, 4
   reversed(range(5))   →  4, 3, 2, 1, 0       ⏪ counting down

   torch.full((4,), 7)  →  [7, 7, 7, 7]        the same number, 4 times
```

### 🔍 `.clamp(-1, 1)` — examples

```
   guess 0.83  →  0.83   (fine ✅)
   guess 1.40  →  1.00   (brighter than white? no! 🚧)
   guess −2.1  → −1.00   (darker than black? no! 🚧)

   🧒 Early guesses (from almost pure static) can be wild —
      clamping keeps them sensible. Same fence as Project #5! 🚧
```

> 🧒 **This loop is Part F's table, done for 784 pixels and 1,000 steps.** This simple version (no fresh randomness added along the way) is called **DDIM** — one of the "smart samplers" that can also skip steps to go faster. 🪜🏃

---

# 📋 MODULE 21 MASTER RECAP

```
   1. VAE = blurry 🌫️ · GAN = sharp but fights and collapses 🥊
      DIFFUSION = sharp ✨ + varied 🌈 + stable 🧘 (but slow 🐢)

   2. THE IDEA 🏖️: break a picture slowly into static (easy, no learning),
      learn to undo ONE small step (the network), then walk back from static

   3. FORWARD: noisy = KEEP × picture + NOISE_AMOUNT × noise
         KEEP² + NOISE_AMOUNT² = 1 always ⚖️
         by the last step, the pixel is almost just the dice 🎲
         jump straight to ANY step in one line ⏩

   4. THE DENOISER'S JOB: guess the NOISE (spot the speckles 🔍),
      told which step it's at ⏱️ — trained with plain MSE

   5. A perfect noise guess → undo the recipe → the exact picture ✅
         picture = (noisy − NOISE_AMOUNT × noise) ÷ KEEP

   6. GENERATE: start from static, step back a little at a time 🪜
         the guess sharpens 0.40 → 0.80 · small steps = sharp pictures
         a different start = a different picture (0.80 vs −0.75) 🎲

   7. FASTER: train on 1,000 steps, sample in ~20-50 with smart samplers

   8. THE NETWORK: a U-Net (autoencoder shape + skip connections),
      or a transformer on patches

   9. TEXT-TO-IMAGE:
         CLIP's text encoder → word vectors 📝
         CROSS-ATTENTION → each patch reads the words it needs 🔦
            (the hat patch → 57.7% on "hat" · the face patch → 63.8% on "cat")
         GUIDANCE SCALE → exaggerate what the prompt adds 🔊 (7.5 popular)

  10. STABLE DIFFUSION = latent diffusion: work in the VAE's latent,
      48× smaller ⚡ — and every part is something you already know 🎉
```

---

# 🤔 COMMON DOUBTS

**Q1: If the forward process just adds noise, why do we need it at all?**
> 🧒 It makes the **training examples**! For every real picture we can create noisy versions at any step — and we know exactly what noise we added, so we have the answer to mark against 📝.

**Q2: Why guess the noise instead of the clean picture?**
> 🧒 Pointing at specks of dust is easier than redrawing a whole photo 🔍. And once you know the noise, you can undo the recipe to get the picture anyway (Part D.5).

**Q3: Why do KEEP² and NOISE_AMOUNT² always add to 1?**
> 🧒 It keeps the numbers the same overall size at every step ⚖️ — the picture fades exactly as fast as the static grows, so nothing explodes or vanishes.

**Q4: Why is diffusion training so stable compared to GANs?**
> 🧒 There's only one network and one plain MSE loss — no detective, no tug-of-war 🧘. Just "guess the noise, see how wrong you were."

**Q5: Why many small steps instead of one big one?**
> 🧒 From pure static the first guess is fuzzy (MSE sits on the fence 🌫️). Small steps let the network look again, with less static in the way each time, and sharpen its guess 🪜.

**Q6: Why doesn't diffusion suffer from mode collapse?**
> 🧒 It learned the way back to EVERY training picture, and every generation starts from different random static 🎲 — so it naturally spreads across all kinds of pictures 🌈.

**Q7: Why does the denoiser NOT end with Sigmoid or Tanh?**
> 🧒 It outputs NOISE, which can be any size (like 3.1). Squashing would stop it from ever guessing big noise 🚫.

**Q8: What does the guidance scale do?**
> 🧒 It runs the denoiser with and without your prompt, then turns up the difference 🔊. Higher = follows your words more strongly; too high = harsh, overdone pictures 🔥.

**Q9: Why hide the caption 10% of the time during training?**
> 🧒 So the same model learns to denoise both WITH and WITHOUT a prompt — which is exactly what guidance needs to compare 🔀.

**Q10: Why work in the VAE's latent space?**
> 🧒 It's 48× smaller, so each of the ~50 steps is much cheaper ⚡. The VAE decoder turns the final latent into a full picture once, at the end 🏠.

**Q11: Is this how DALL·E and Midjourney work too?**
> 🧒 The same family of ideas — diffusion guided by text — though each company's exact recipe differs and isn't always public. Stable Diffusion's recipe is fully open, which is why we study it 📖.

---

# ✅ QUICK PRACTICE

**Q1:** KEEP = 0.6 and NOISE_AMOUNT = 0.8. A pixel of 0.5 gets noise 1.0. What's the noisy value? Does 0.6² + 0.8² = 1?
<details><summary>Answer</summary>
0.6 × 0.5 + 0.8 × 1.0 = 0.30 + 0.80 = 1.10. And 0.36 + 0.64 = 1.00 ✅.
</details>

**Q2:** With those amounts, the noisy value is 1.10 and the network guesses the noise perfectly (1.0). Recover the pixel.
<details><summary>Answer</summary>
(1.10 − 0.8 × 1.0) ÷ 0.6 = 0.30 ÷ 0.6 = 0.5 ✅ — exactly the original.
</details>

**Q3:** True noise = [0.5, −1.0], guessed noise = [0.3, −1.0]. What's the MSE?
<details><summary>Answer</summary>
Differences [−0.2, 0.0] → squared [0.04, 0.00] → average 0.02.
</details>

**Q4:** Each step keeps 90%, 90% and 50%. How much survives after all three? (That's `cumprod`.)
<details><summary>Answer</summary>
0.9 × 0.9 × 0.5 = 0.405 → 40.5%.
</details>

**Q5:** What does `torch.linspace(0, 10, 3)` give?
<details><summary>Answer</summary>
[0, 5, 10] — three evenly spaced numbers from 0 to 10.
</details>

**Q6:** Noise guess without the prompt = 0.20, with = 0.30, guidance scale 5. Final guess?
<details><summary>Answer</summary>
0.20 + 5 × (0.30 − 0.20) = 0.20 + 0.50 = 0.70.
</details>

**Q7:** Why does the denoiser NOT end with Sigmoid or Tanh?
<details><summary>Answer</summary>
It outputs NOISE, which can be any size. Squashing would stop it from guessing big values 🚫.
</details>

**Q8:** Two generations start from different static. Same picture?
<details><summary>Answer</summary>
No — different starts lead to different pictures 🎲. That's where diffusion's variety comes from 🌈.
</details>

**Q9:** A latent is 32 × 32 × 4. How many numbers, and how much smaller than a 256 × 256 × 3 picture?
<details><summary>Answer</summary>
4,096 numbers. 196,608 ÷ 4,096 = 48× smaller.
</details>

**Q10:** With guidance and 50 steps, about how many U-Net runs make one picture?
<details><summary>Answer</summary>
50 steps × 2 (with and without the prompt) = 100 runs.
</details>

**Q11:** In `make_noisy`, why do we write `keep[step].view(-1, 1)`?
<details><summary>Answer</summary>
To turn shape (B,) into (B, 1), so each picture's KEEP number multiplies all 784 of its pixels 🔖.
</details>

---

# 🎬 WHAT'S NEXT: PROJECT #6 — TEXT-TO-IMAGE, END TO END 🎨

```
   PART 1 — Build your OWN diffusion model 🌫️➡️🔢
      train the tiny denoiser on MNIST and watch static turn into digits,
      step by step — Part F's table, live!

   PART 2 — Open up Stable Diffusion 🔍🎨
      run a real text-to-image model on free Colab, then take it apart:
      see the CLIP text encoder, the U-Net and the VAE as separate pieces,
      and try different prompts, step counts and guidance scales.
```

After that: **Phase 6 — Inside a Modern LLM** 🤖

---

*Module 21 Complete! You understand the machine behind modern image generators — and every piece of it is something you already learned! 🌫️➡️🖼️*
