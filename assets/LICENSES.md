# Assets: manifest, credits and licences

Every asset in this folder is free to use in this project. **One sound needs attribution** (CC BY 4.0), so show the credits below in the README and on the demo's title or credits screen.

| Asset | Source | Author | Licence |
|---|---|---|---|
| `sprites/cars/*`, `sprites/props/*` | [Racing Pack](https://kenney.nl/assets/racing-pack) | Kenney (kenney.nl) | CC0 (`licenses/kenney_racing_pack_CC0.txt`) |
| `textures/ground/asphalt.jpg` | ambientCG **Asphalt025C** (1K, Color map) | ambientCG.com | CC0 |
| `textures/ground/grass.jpg` | ambientCG **Grass005** | ambientCG.com | CC0 |
| `textures/ground/gravel.jpg` | ambientCG **Gravel035** | ambientCG.com | CC0 |
| `textures/ground/sand.jpg` | ambientCG **Sand093C** | ambientCG.com | CC0 |
| `fonts/rajdhani_*.ttf` | Google Fonts: Rajdhani | Indian Type Foundry | OFL 1.1 (`licenses/OFL_rajdhani.txt`) |
| `fonts/share_tech_mono.ttf` | Google Fonts: Share Tech Mono | Carrois Apostrophe | OFL 1.1 (`licenses/OFL_share_tech_mono.txt`) |
| `fonts/orbitron_*.ttf` | Google Fonts: Orbitron | Matt McInerney | OFL 1.1 (`licenses/OFL_orbitron.txt`) |
| `sounds/engine_*.ogg` | [freesound #623436](https://freesound.org/s/623436/) "Delorean DMC-12 V6 PRV engine" | **skyernaklea** | **CC BY 4.0: attribution required.** Modified (cut, looped, pitch-flattened, re-encoded). |
| `sounds/tyre_screech_*.ogg` | [freesound #614627](https://freesound.org/s/614627/) "screeching tyres / tires" | johnnydekk | CC0 (credit appreciated) |

**Credits line for the README and the game:**
> Engine sound: "Delorean DMC-12 V6 PRV engine" by skyernaklea (freesound.org/s/623436), CC BY 4.0, edited. Tyre sound by johnnydekk (CC0). Sprites by Kenney (CC0). Textures by ambientCG (CC0). Fonts: Rajdhani, Share Tech Mono, Orbitron (SIL OFL 1.1).

---

## Manifest and usage notes

### Sprites
- **`cars/car_{1..4}_{black,blue,green,orange,yellow}.png`:** 4 car shapes × 5 colours, each about 70 × 116–131 px.
  - Shapes 1–3 are Kenney's car 1–3. Shape 4 is Kenney's car 5. Kenney's car 4 is a pickup truck and was dropped.
  - Kenney's "red" colour is visibly orange, so it's named `orange`.
  - The sprites face **up** in the image, so rotate them by the heading.
- **`cars/car_N_tint.png` + `car_N_tint_mask.png`:** tintable versions, for giving 10–20 cars unique colours.
  - The body is light grey with its shading kept; windows and highlights are unchanged.
  - **How to tint:** multiply `car_N_tint.png` by the colour, then paste the result back through the mask's alpha. Only the body changes colour, and the windows stay white.
- **Size note:** the sprites' aspect ratio is about 1.85, while the physics car is 4.5 × 1.9 m (2.37). Scale the sprite to the car's **length**; it will look slightly wider than the physics box, which is fine.
- **`props/`:** trackside scenery.
  - barriers: `barrier_*`, `barrier_*_race`
  - tyre stacks: `tyres_*`
  - cones: `cone`, `cone_down`
  - trees: `tree_large`, `tree_small`
  - grandstands: `grandstand_full`, `grandstand_empty`, `grandstand_roof_*`
  - tents: `tent_*`
  - start gantry: `start_lights`

### Ground textures (1024 × 1024, seamless)
- **`asphalt.jpg`:** the original had large repeating stains that showed a visible pattern when tiled. They were removed with a periodic low-pass filter, which keeps the tile seamless and the fine grain intact. The original is in `_source/`.
- **`grass.jpg`, `gravel.jpg`, `sand.jpg`:** renamed only.
- **Physics:** gravel and sand are **visual run-off variants only**. They use the grass `Surface` physics.
- **Suggested world size of one tile (set in config):** asphalt 6 m, grass 8 m, gravel 4 m, sand 10 m.

### Fonts
- **HUD:** `rajdhani_semibold` / `rajdhani_bold`
- **Body text:** `rajdhani_regular` / `rajdhani_medium`
- **Numbers (lap times, telemetry):** `share_tech_mono`
- **Title and menu:** `orbitron_bold` / `orbitron_black`

### Sounds
All sounds are mono, 44.1 kHz, OGG Vorbis (q6). The engine clips share one gain (peak −3 dBFS across the set), so their **relative loudness is natural**: idle is about 20 dB quieter than a full rev. Set per-clip volumes in config.

**Engine** (`engine_map.json` gives the pitch/RPM curve of each clip):

| File | Length | Cut from source | RPM ratio (idle = 1) | Use |
|---|---|---|---|---|
| `engine_rev_up.ogg` | 3.06 s | 32.70–35.76 s | 1.19 → 2.88 (rises steadily) | Throttle on. Seek to the current RPM. |
| `engine_high_loop.ogg` | 0.53 s loop | 35.10–35.79 s, pitch- and loudness-flattened | 2.88 constant | Crossfade into this when rev_up ends (holding redline). |
| `engine_rev_down.ogg` | 8.05 s | 58.75–66.80 s | 2.34 → 1.0 (falls steadily) | Throttle off. Seek to the current RPM. |
| `engine_idle_loop.ogg` | 2.00 s loop | 64.70–66.95 s, 0.25 s crossfade | 1.0 | Idle. Cut from the end of the rev_down take, so the transition is seamless. |

**Tyres:**

| File | Length | Cut from source | Use |
|---|---|---|---|
| `tyre_screech_loop.ogg` | 2.03 s loop | 3.70–5.98 s, 0.25 s crossfade | Sustained slide. Volume follows slip. |
| `tyre_screech_short_1.ogg` | 1.47 s | 14.03–15.50 s | One-shot: brief slide or hard braking |
| `tyre_screech_short_2.ogg` | 1.24 s | 82.36–83.60 s | One-shot: alternate |

### `_source/` (not tracked by git)
The original downloads, kept so clips can be re-cut later:
- `sounds/623436__…engine.wav` (80.8 s, 48 kHz)
- `sounds/614627__…tires.wav` (86.5 s, 44.1 kHz)
- `Asphalt025C_1K-JPG_Color.jpg`

It's safe to delete this folder if you don't need to re-cut anything.
