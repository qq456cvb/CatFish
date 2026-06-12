# CatFish

A solution to the "小猫钓鱼" (cat fishing) puzzle from the **Geeker's Party 2022 spring monthly contest**: how long does the children's card game last? The script answers it twice — by Monte-Carlo simulation and by an analytic random-walk argument — and plots the two distributions side by side.

## The Game

Two players split a deck evenly. Each turn a player plays a card onto a shared pile; if it matches a card already in the pile, they scoop up everything from the match onward. A player loses when their hand is empty. (`playgame` implements these full rules with random play.)

## The Analysis (`montecarlo.py`)

- **Simplified model** (`game_simp`) — each turn the hand-size difference behaves like a **lazy ±1 random walk**: with probability ¼ player A wins a card, ¼ player B does, ½ nothing changes. The game ends when one hand is empty, i.e. when the walk hits ±c (c = half the deck).
- **Analytic distribution** (`calc_P`) — the probability that the game survives n turns is computed exactly: the n-step distribution comes from convolution powers of `[¼, ½, ¼]`, and absorption at the two barriers is handled by a **reflection-principle inclusion–exclusion** over images at odd multiples of c.
- **Validation** — the main block runs 100 000 simulated games with an 8-card deck, histograms the game lengths, and overlays the analytic curve, which matches the Monte-Carlo distribution.

## Running

Requires NumPy, SciPy, Matplotlib, and tqdm.

```bash
python montecarlo.py
```

Edit `cards` in the main block to change the deck (e.g. a full 54-card deck), and swap `game_simp` for `playgame` to simulate the real rules rather than the random-walk approximation.
