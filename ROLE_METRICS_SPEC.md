# Overwatch Coach Metric Spec

## Scoring Model

- Total score: `100`
- Common metrics: `60`
- Role-specific metrics: `40`
- Roles: `dps`, `tank`, `support`
- Inputs must be observable from recorded gameplay video only.
- Excluded signal: `Tab key usage`
  - Reason: keyboard input cannot be reliably inferred from recorded video.

## Design Principles

- Every metric must map to visible evidence in the recording:
  - crosshair movement
  - target location
  - camera motion
  - wall or cover occupancy
  - health bar changes
  - kill feed
  - ultimate gauge
  - ally or enemy outline visibility
- Metrics should be treated as coaching proxies, not perfect esports telemetry.
- Feedback output should separate:
  - `evaluation_basis`: why the model judged it that way
  - `feedback_direction`: what habit should change
  - `action_item`: what to practice next

## Common Metrics: 60 Points

### Aim: 20

#### 1. Tracking Stability: 12
- Feasibility: `High`
- Evidence:
  - average pixel distance between crosshair and target bounding-box center
  - distance variance during continuous fights
- Interpretation:
  - lower distance and smoother variance imply stronger tracking
- Coaching direction:
  - reduce prediction-heavy flicking
  - sync mouse movement more smoothly to target motion

#### 2. Aim Mobility Range: 8
- Feasibility: `Medium`
- Evidence:
  - time spent locked near screen center
  - horizontal camera sweep width
  - screen-edge scanning frequency
- Interpretation:
  - overly narrow visual range suggests tunnel vision
- Coaching direction:
  - widen visual sweep to detect side threats earlier

### Move: 15

#### 3. Combat Cover Rate: 10
- Feasibility: `Medium`
- Evidence:
  - during firing or damage intake, wall or cover occupancy inside screen-side 20% bands
  - time spent exposing full body in open space
- Interpretation:
  - higher side-cover occupancy generally implies better corner discipline
- Coaching direction:
  - reduce open-field exposure
  - hug corners before and during fights

#### 4. Meaningless Jump Frequency: 5
- Feasibility: `High`
- Evidence:
  - repeated vertical camera shake pattern without clear movement utility
  - jump frequency during neutral or straight duels
- Interpretation:
  - habitual jumps create predictable airborne trajectory
- Coaching direction:
  - correct panic jumping
  - replace with strafe and cover usage

### Judge: 15

#### 5. Information Scan Frequency: 8
- Feasibility: `High`
- Evidence:
  - left-right camera sweeps from optical-flow spikes
  - time spent staring forward without situational checks
- Interpretation:
  - moderate intentional scanning is healthy
  - constant forward fixation indicates weak awareness
- Coaching direction:
  - use reload, downtime, and cover moments to check both teams

#### 6. Regroup Discipline: 7
- Feasibility: `Medium`
- Evidence:
  - kill feed indicating ally disadvantage
  - forward movement or firing during disadvantage windows
- Interpretation:
  - continued aggression during lost fights lowers regroup quality
- Coaching direction:
  - stop isolated re-entry
  - wait for team regroup before recommitting

### Op: 10

#### 7. Fight Tempo Discipline: 10
- Feasibility: `Medium`
- Evidence:
  - timing of re-engage after losing or winning momentum
  - continuation or disengage behavior after clear fight-state shifts
- Interpretation:
  - good tempo means joining favorable fights and leaving lost ones
- Coaching direction:
  - improve patience in bad fights and decisiveness in winning fights

## DPS Metrics: 40 Points

### Op: 12

#### 1. Target Focus Priority: 12
- Feasibility: `Medium`
- Evidence:
  - ratio of crosshair dwell time on enemy tank vs backline targets
  - target-switch behavior after enemy support or DPS becomes visible
- Interpretation:
  - excessive tank fixation reduces kill conversion pressure
- Coaching direction:
  - shift attention toward high-value enemy targets

### Judge: 16

#### 2. Side Angle Occupancy: 10
- Feasibility: `Medium`
- Evidence:
  - visual angle between player lane and friendly core lane
  - time spent on high ground or side lane pixels
- Interpretation:
  - sustained side-angle presence implies stronger pressure geometry
- Coaching direction:
  - avoid only front-facing duels
  - pressure from side or high-ground routes

#### 3. Effective Range Tempo: 6
- Feasibility: `Low-Medium`
- Evidence:
  - approximate target size as distance proxy
  - timing of first shot relative to effective range
- Interpretation:
  - firing too early can expose position before lethal range
- Coaching direction:
  - tighten trigger discipline
  - shoot when kill pressure is real

### Op: 12

#### 4. Ultimate Investment Efficiency: 12
- Feasibility: `Medium`
- Evidence:
  - idle time after ultimate reaches 100%
  - kill feed creation after ultimate use
  - use in already won or lost fights
- Interpretation:
  - strong value can be one clean kill, not only multi-kill
- Coaching direction:
  - improve ult cycle speed and tempo usage

## Tank Metrics: 40 Points

### Op: 22

#### 1. Choke Control: 14
- Feasibility: `Medium`
- Evidence:
  - time spent anchoring corners or narrow approach zones
  - enemy forward progress halt duration while player holds line
- Interpretation:
  - stronger choke control means less free enemy space gain
- Coaching direction:
  - pre-claim corners and avoid giving up line for free

#### 2. Pre-Fight Resource Preservation: 8
- Feasibility: `High`
- Evidence:
  - health-bar loss during poke or neutral before full engagement
  - unnecessary open-space exposure before team commit
- Interpretation:
  - losing too much HP early drains support resources
- Coaching direction:
  - advance with walls and corners
  - preserve support bandwidth before engagement

### Judge: 8

#### 3. Skill-Counted Entry: 8
- Feasibility: `Low`
- Evidence:
  - visible enemy CC effect usage before player engage tool commit
- Interpretation:
  - useful in theory but risky in MVP due to weak reliability
- MVP recommendation:
  - keep as experimental or disabled by default

### Physical: 10

#### 4. Aggro Ping-Pong Survival: 10
- Feasibility: `Medium`
- Evidence:
  - reaction time after HP falls below 30%
  - defensive ability timing
  - fast retreat turn or disengage behavior
- Interpretation:
  - strong tanks absorb pressure and still exit before dying
- Coaching direction:
  - leave earlier at critical HP
  - convert pressure tanking into survivable space-making

## Support Metrics: 40 Points

### Judge: 18

#### 1. Heal-Damage Tempo Shift: 8
- Feasibility: `Medium`
- Evidence:
  - when allies are stable above 80% HP, crosshair time on enemy outlines
- Interpretation:
  - support should add damage during safe windows
- Coaching direction:
  - avoid heal-only autopilot
  - use tempo windows to pressure enemies

#### 2. Critical Ally Reaction Time: 10
- Feasibility: `Medium`
- Evidence:
  - delay between visible ally critical marker and crosshair movement toward that ally
- Interpretation:
  - shorter reaction suggests stronger triage awareness
- Coaching direction:
  - improve panic-response speed to sudden ally collapse

### Op: 10

#### 3. Self-Survival Cooldown Preservation: 10
- Feasibility: `Low-Medium`
- Evidence:
  - self-heal or self-save ability use while HP remains safe
  - visible cooldown state if UI can be read reliably
- Interpretation:
  - wasteful self-preservation reduces team save capacity
- Coaching direction:
  - stop spending major defensive cooldowns too early

### Judge: 12

#### 4. Survival Line Maintenance: 12
- Feasibility: `Medium`
- Evidence:
  - approximate distance from friendly tank
  - duration of damage indicator while exposed
- Interpretation:
  - overcommitting to save a doomed ally often causes double loss
- Coaching direction:
  - prioritize safe heal angles behind cover

## MVP Metric Set Recommendation

These should be enabled first.

### Common MVP
- Tracking Stability
- Aim Mobility Range
- Combat Cover Rate
- Meaningless Jump Frequency
- Information Scan Frequency
- Regroup Discipline

### DPS MVP
- Target Focus Priority
- Side Angle Occupancy
- Ultimate Investment Efficiency

### Tank MVP
- Choke Control
- Pre-Fight Resource Preservation
- Aggro Ping-Pong Survival

### Support MVP
- Heal-Damage Tempo Shift
- Critical Ally Reaction Time
- Survival Line Maintenance

## Experimental or Phase-2 Metrics

- DPS Effective Range Tempo
- Tank Skill-Counted Entry
- Support Self-Survival Cooldown Preservation

These require stronger OCR, better effect detection, or cleaner hero-state parsing.

## Prompt/Schema Mapping

Each feedback item should contain:

- `lane`: `strength` or `weakness`
- `category`: `aim`, `move`, `judge`, `op`
- `metric_name`: exact metric label
- `summary`: what happened in that clip
- `evaluation_basis`: visible evidence from the recording
- `feedback_direction`: coaching purpose
- `action_item`: concrete next-step routine
- `confidence`: model confidence

## UI Recommendation

- Show `Common 60` and `Role 40` as separate score groups.
- In detailed view, label each clip with:
  - `Common`
  - `Role-Specific`
- Group detailed feedback into:
  - `잘한 장면`
  - `보완 장면`
  - `데스 원인`

## Implementation Notes

- Do not infer keyboard input such as `Tab`.
- Favor stable observable proxies over fragile exact interpretations.
- If a metric cannot be computed confidently for a given hero or clip, lower confidence rather than fabricating precision.
