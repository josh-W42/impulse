# Tab-target PvP MMO on impulse — Design

**Date:** 2026-10-19
**Status:** Draft (for review)

## Purpose

A third-person online RPG with **tab-target combat** (World of Warcraft style)
and **PvP**, running in the browser on impulse. Like solstice, the game lives in
its own repository; this doc also lists the engine milestones it needs from
impulse.

Success criteria:

- Several players log in from a browser, walk around one zone, target each
  other, and fight with abilities that have cast times, cooldowns and a global
  cooldown.
- The server is authoritative: a modified client cannot teleport, ignore
  cooldowns, or hit out of range or through walls.
- Measured capacity: how many players one zone server holds (bot load test).

Not goals for the first playable: quests, items and an economy, PvE AI beyond
training dummies, multiple zones.

## Setting and art

High fantasy in the spirit of WoW (two factions at war, swords and spells,
bright readable colours) but our own world, names and lore: no borrowed races,
places, factions or ability names. Faction names come with the game's name.

Art direction: **stylized low-poly**. It reads well from a third-person camera,
stays inside the browser's draw and memory budget, and matches the CC0 packs we
can use now: KayKit (animated adventurer characters, dungeon and forest kits),
Quaternius (characters, nature) and Kenney (the props already in
`assets/models/`). Check each pack's licence before adding it, as
`assets/models/LICENSES.md` does today.

## Starting classes

Two contrasting classes, enough to test melee against ranged, interrupts and
crowd control. Numbers are placeholders for G0 tuning.

**Warrior** (melee, plate). Resource: **rage**, 0–100, gained by dealing and
taking damage, drains out of combat.

| Ability | Cast | Range | Cooldown | Effect |
|---|---|---|---|---|
| Heavy Strike | instant | 5 m | GCD | Weapon damage; 15 rage |
| Charge | instant | 8–25 m | 15 s | Rush to the target, 1 s stun, +20 rage; out of combat only |
| Hamstring | instant | 5 m | GCD | 50% slow for 10 s; 10 rage |
| Shield Bash | instant | 5 m | 12 s | Interrupts a cast, locks that spell school for 4 s; 10 rage |
| Shield Wall | instant | self | 120 s | 40% less damage taken for 8 s |

**Mage** (ranged, cloth). Resource: **mana**, regenerates slowly.

| Ability | Cast | Range | Cooldown | Effect |
|---|---|---|---|---|
| Firebolt | 2.0 s | 35 m | GCD | Fire damage (projectile) |
| Frost Lance | 1.5 s | 35 m | GCD | Frost damage, 40% slow for 6 s |
| Frost Ring | instant | 10 m around self | 25 s | Roots enemies for 6 s; damage may break it |
| Blink | instant | 15 m forward | 15 s | Teleport; a capsule sweep stops it at walls; breaks roots and stuns |
| Counterspell | instant | 35 m | 24 s | Interrupts a cast, locks that spell school for 6 s |

Charge and Blink are the two abilities that move a character through the world,
so they exercise the M13 kinematic bodies and shape casts directly.

## What tab-target changes

Combat is resolved from **who is targeted**, not from where a swing or projectile
lands. That moves most of the hard netcode problems out of the way:

- **No lag compensation.** The server never rewinds hitboxes. An ability hits if,
  when the server resolves it, the target is in range, in line of sight, and (for
  some abilities) in front of the caster.
- **Latency is hidden by game design, not prediction:** cast bars start on the
  client immediately; a short **spell-queue window** (~400 ms) lets the next
  ability be pressed before the global cooldown ends; range checks get a small
  leeway (~1 m) for the difference between what the client sees and the server's
  state.
- **WebSockets are good enough for the long term.** TCP's head-of-line blocking
  hurts twitch shooters; tab-target tolerates it. WebTransport stays optional.
- **Physics plays a supporting role:** character movement against terrain and
  buildings, line-of-sight raycasts, knockbacks and charges, and props. Players
  do not collide with each other (as in WoW), which removes the worst
  prediction mismatches.

## Architecture

Game repository, one-directional dependencies:

```
client (Game impl, native + wasm) ──┐
                                    ├──►  shared (sim + protocol) ──►  impulse-physics
server (native bin, tokio) ─────────┘
client ──► impulse-engine (render, input, egui)
```

- **`shared`**: entity state, the character controller, ability and aura rules,
  and the network protocol. Pure Rust, no I/O, so the server and the client run
  exactly the same movement and rules code. Depends on `impulse-physics`, not
  on the engine.
- **`server`**: one process per zone. Fixed 60 Hz physics, gameplay tick at
  30 Hz, snapshots to each client at 20 Hz. Async I/O with tokio; the simulation
  itself runs on one thread and never awaits.
- **`client`**: an `impulse_engine::Game`. Renders the world, sends inputs and
  ability requests, draws the UI with egui.

Entities are plain structs in a `Vec` indexed by generational ids, like
`World`'s bodies. No ECS (same rule as the engine).

### Movement

- Each player is a **kinematic capsule** moved by a character controller
  (walk, run, jump, gravity, slopes up to ~45°, step-up ~0.4 m, sliding along
  walls), built on capsule sweeps against static geometry.
- **Input-based authority:** the client sends its movement input every tick
  with a sequence number and predicts its own character immediately. The server
  runs the same controller on the same inputs and sends back the authoritative
  position with the last processed sequence number. The client replays any
  unacknowledged inputs on top of it; small differences blend out over a few
  frames, large ones snap.
- Other players and NPCs are drawn **interpolated** ~100 ms behind the newest
  snapshot.
- Alternative considered: client-authoritative movement with server sanity
  checks (WoW's actual model). Smoother on bad connections, but every check
  (speed, flying, wall clipping) has to be written by hand. Rejected while the
  controller is shared code that the server can simply run.

### Targeting and combat

- **Targeting:** Tab cycles the nearest hostile units in a cone in front of the
  camera, then by distance; click selects with `cursor_ray` and a physics
  raycast against a "selectable" collision layer. Esc clears. The target is
  client state, sent with each ability request.
- **Ability request:** `Cast { ability, target, seq }`. The server checks the
  global cooldown, the ability's cooldown, resources (mana, energy), range, line
  of sight (raycast against static geometry only), facing, and crowd control
  (stunned, silenced). It rejects with a reason the client shows ("Out of
  range", "Not ready yet").
- **Cast times:** moving or being interrupted cancels a cast on the server; the
  client's cast bar follows the server's start, finish and cancel events.
- **Instant and channelled abilities**, **projectiles** that are just a delay
  proportional to distance (visual only, they always hit once launched), and
  **ground-targeted areas** (one overlap query when the ability resolves).
- **Auras:** buffs and debuffs with duration, stacks, periodic ticks (damage and
  healing over time) and stat modifiers. Crowd control is an aura: stun, root,
  silence, slow, with **diminishing returns** per category in PvP.
- **Data-driven:** classes, abilities and auras are RON files, as scenes are in
  the engine. The server loads them; the client loads the same files for
  tooltips and cast bars.

### PvP rules

- Two factions. Zones are marked sanctuary, contested (open-world PvP) or
  arena.
- Duels by invitation anywhere outside sanctuaries.
- Death: corpse, then respawn at the zone's graveyard after a short timer.
- Later: an instanced 2v2/3v3 arena. Each match is its own small zone process,
  so it reuses everything.

### Networking

- Transport: WebSocket (`tokio-tungstenite` on the server, `web-sys` WebSocket
  in the browser, `tungstenite` on native clients).
- Encoding: binary, `serde` + `postcard`. Positions quantized (1 cm),
  orientations as a yaw angle (characters only turn around the vertical axis).
- **Snapshots** send a delta against the last snapshot the client acknowledged:
  only entities that changed, only fields that changed.
- **Events** (cast started, damage taken, aura applied, death) go reliably, in
  order, alongside snapshots. TCP gives this for free.
- **Interest management:** a uniform grid (32 m cells); each client hears about
  entities within ~100 m. Combat events about the player always arrive,
  whatever the distance.

### Hosting

- **Zone server: [Render](https://render.com)** as a Docker web service.
  Render supports WebSockets on web services. The free tier sleeps after 15
  minutes without HTTP requests or WebSocket messages and takes about a minute
  to wake: fine for development demos, not for players. Move to a paid,
  always-on instance once people play.
- Deploys replace the instance and drop every connection (graceful shutdown
  window 30 s, extendable to 300 s). The server saves characters on SIGTERM and
  the client reconnects automatically (G3).
- Shared CPUs make tick times jittery. The server reports tick time (mean, p95,
  max) from G1 on; if p95 stops fitting the 33 ms tick, move to a dedicated-CPU
  instance or another host.
- Alternatives if Render stops fitting: Fly.io (cheap per-second machines in
  many regions, so zones can sit near players) or a plain VPS (best price per
  CPU, but we run the box ourselves).
- **Client:** static files, like the engine demo on Firebase Hosting today, or
  a Render static site next to the server.

### Persistence

- SQLite first (one file next to the server), Postgres when there is more than
  one server process.
- Stored: accounts (argon2 password hashes), characters (class, level, position,
  zone), and later inventory. Saved on logout and every 60 s.
- The server is the only writer; clients never talk to the database.

### Client UI (egui)

Player and target frames (health, resource, auras), action bar with cooldown
sweeps, cast bar, nameplates and floating combat text projected from world space,
a combat log, and chat.

## Engine milestones (impulse)

| # | Milestone | Contents |
|---|---|---|
| M13 | Gameplay physics | `BodyKind::Kinematic` (moved by setting a target pose; pushes dynamic bodies, is not pushed); setting position and velocity; collision layers and masks on colliders; raycasts with a layer filter; shape casts (sphere, capsule) and overlap queries |
| M14 | Character and camera | Kinematic character controller (in `impulse-physics`, so the server can use it headless); `ThirdPersonCamera` (orbit on right-drag, zoom, pulls in when terrain is in the way), alongside `RtsCamera` |
| M15 | Skeletal animation | glTF skins and clips (currently ignored, rest pose only); GPU skinning in the scene shader; clip playback and crossfades (idle, run, cast, hit, death) |

M13 and M14 come first: the game cannot start without them. M15 can follow the
first networked milestone; capsules with a facing marker are enough to test
combat.

## Game milestones

| # | Milestone | Done when |
|---|---|---|
| G0 | Single-player combat loop | One character walks a small zone (M14), tab-targets training dummies, and casts 4–5 abilities with cast bar, global cooldown, cooldowns and auras. No networking, so the rules can be tuned fast |
| G1 | Networked zone | Server and client split along `shared`; movement prediction and reconciliation; interpolated remote players; abilities resolved on the server. Two browser tabs fight each other |
| G2 | PvP | Factions, zone PvP rules, duels, death and respawn, crowd control with diminishing returns, two classes |
| G3 | Accounts and persistence | Login, character creation, saved characters, reconnect after a dropped connection |
| G4 | Scale | Delta snapshots, interest management, headless bot clients; a load test that reports players per zone and bandwidth per player |

Every milestone ships a browser demo, as in the engine.

## Risks

- **Character art.** Tab-target combat reads badly without animation (who is
  casting, who got hit). Plan on CC0 animated characters (e.g. Quaternius) and
  budget M15 seriously.
- **Wasm is single-threaded.** The client simulates only its own character
  plus rendering, so this should be fine; measure it in G1.
- **Server cost per player.** Unknown until G4. Physics is cheap here (static
  world, few dynamic bodies); snapshot encoding and the interest grid are the
  likely hot spots.

## Open questions

1. Name and repository for the game (and the two faction names).

Decided 2026-10-19: high-fantasy setting with stylized low-poly art; Warrior
and Mage as the starting classes; Render for the zone server.
