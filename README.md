You are building a Windows desktop application for symbolic MIDI analysis and generation. The user wants something functionally closer to InstaComposer than to a passive MIDI generator, but with a deeper analytic model of musical functions. The program must analyse large MIDI corpora, infer musical structures, train task-specific generation modules, and generate MIDI parts in response to existing material.
Assume CUDA-enabled PyTorch. The program must have a GUI. It must be usable by a musician who wants to import MIDI, analyse it, train or fine-tune models, generate parts, and export MIDI.
The core requirement is not simply “generate melodies”. The system must identify and model musical functions such as motif, motif mutation, countermelody, ornamentation, pattern variation, counterpoint, transposition, inversion, chord change, modulation, reharmonisation, bassline response, drum-bass interaction, and phrase-level development.
Do not build this as one undifferentiated music model. Build it as a hybrid symbolic-analysis and machine-learning system.
The program must preserve expressive timing. Do not force all MIDI into four-bar quantised loops. Store both a metrical representation and a performance representation. For each event, keep absolute time, beat-relative time, estimated bar/beat position, microtiming deviation, velocity, duration, pitch, pitch class, track, channel, instrument, and local harmonic context.
The system must not assume one genre. It should pretrain on broad MIDI corpora, then allow smaller style collections to steer generation. The model must not collapse into a brittle genre classifier. Treat style as a control signal, not as the sole training objective.
The program must distinguish between musical objects and musical operations. A motif is an object. Transposition, inversion, augmentation, diminution, ornamentation, fragmentation, reharmonisation and rhythmic displacement are operations. A useful generator must learn both.
The first deliverable is not a finished AI composer. The first deliverable is a robust analysis engine and dataset builder. Only after that should generation models be trained.
Build target
Build a local Windows desktop application called MotifLab.
It should have five major components:
MIDI ingestion and analysis engine
Symbolic music-function labelling system
Training pipeline
Generation and critic/ranking pipeline
GUI for importing, analysing, training, generating and exporting MIDI
Use Python 3.11 or 3.12. Use PyTorch with CUDA. Use MidiTok for tokenisation. Use MusPy for dataset handling and evaluation where appropriate. Use music21 for harmonic, intervallic and score-level analysis where useful. Use pretty_midi or symusic where needed for low-level MIDI event handling. Use PySide6 for the first GUI unless there is a strong reason to use Tauri.
The program must be modular. Each component should be callable from both CLI and GUI.
Overall architecture
Use this repository structure:
motiflab/
  README.md
  pyproject.toml
  environment.yml
  requirements.txt
  motiflab/
    __init__.py
    core/
      midi_io.py
      event_model.py
      timing.py
      quantisation.py
      segmentation.py
      track_classification.py
      harmonic_analysis.py
      phrase_analysis.py
    analysis/
      motif.py
      transformations.py
      ornamentation.py
      counterpoint.py
      countermelody.py
      bassline.py
      drums.py
      harmony.py
      modulation.py
      pattern_mutation.py
      labels.py
      scoring.py
    data/
      corpus_index.py
      dataset_builder.py
      cleaning.py
      augmentation.py
      metadata.py
      splits.py
    tokenisation/
      tokenizer_config.py
      encode.py
      decode.py
      task_tokens.py
    models/
      backbone.py
      task_heads.py
      critic.py
      ranker.py
      losses.py
      checkpoints.py
    train/
      train_backbone.py
      train_task.py
      train_critic.py
      evaluate.py
      config.py
    generate/
      context.py
      generate_melody.py
      generate_bassline.py
      generate_countermelody.py
      generate_motif_mutation.py
      generate_ornamentation.py
      generate_harmony.py
      generate_drums.py
      generate_arrangement.py
      candidate_ranker.py
      export.py
    gui/
      app.py
      main_window.py
      analysis_panel.py
      training_panel.py
      generation_panel.py
      piano_roll_view.py
      settings_panel.py
    server/
      api.py
    tests/
      test_event_model.py
      test_motif_detection.py
      test_transformations.py
      test_counterpoint.py
      test_tokenisation_roundtrip.py
      test_generation_constraints.py
  scripts/
    scan_corpus.py
    build_dataset.py
    train_backbone.py
    train_task_model.py
    train_critic.py
    launch_gui.py
  configs/
    default.yaml
    tokenizer.yaml
    training_backbone.yaml
    training_tasks.yaml
    critic.yaml
    gui.yaml
  examples/
    input_midis/
    output_midis/
The GUI must call the same backend functions as the CLI. Do not duplicate logic inside GUI code.
Stage 1: MIDI ingestion
Build a robust MIDI parser.
For each imported MIDI file, create an internal representation:
@dataclass
class NoteEvent:
    file_id: str
    track_id: int
    channel: int | None
    instrument_program: int | None
    is_drum: bool
    pitch: int
    pitch_class: int
    onset_seconds: float
    offset_seconds: float
    duration_seconds: float
    onset_tick: int
    offset_tick: int
    duration_tick: int
    velocity: int
    estimated_tempo: float | None
    bar_index: int | None
    beat_index: float | None
    onset_beat: float | None
    duration_beats: float | None
    microtiming_ms: float | None
    metrical_strength: float | None
Also define:
@dataclass
class TrackSummary:
    track_id: int
    instrument_program: int | None
    is_drum: bool
    note_count: int
    pitch_min: int
    pitch_max: int
    median_pitch: float
    median_duration_beats: float
    density_notes_per_bar: float
    polyphony_rate: float
    syncopation_score: float
    likely_role: Literal[
        "melody", "bass", "drums", "chords", "pad", "arp", "countermelody",
        "lead", "unknown"
    ]
The parser must tolerate corrupt MIDI files. If a file fails, log it and continue. Never let one bad file stop the corpus scan.
Implement track-role classification. Use heuristics first:
Bass tracks: low median pitch, monophonic or near-monophonic, strong relation to chord roots or low rhythmic ostinato.
Drums: MIDI channel 10 or percussion program, repeated pitch classes in drum range, high density, percussive durations.
Melody: higher register, locally salient velocity, relatively monophonic, phrase-like contour, often top voice.
Chords/pads: polyphonic, longer duration, vertical pitch stacks.
Arps: repeating broken-chord patterns, medium-to-high density, pitch-class relation to harmonic field.
Countermelody: mid/high register, independent from main melody, not simply chordal accompaniment, rhythmically and registrally distinct.
Do not assume these classifications will be perfect. Store confidence scores.
Stage 2: Timing and segmentation
Build a timing module that estimates metre, beat grid, bar positions and phrase boundaries without destroying expressive timing.
The program must support three timing layers:
raw time: seconds/ticks exactly as in the file
metrical time: estimated beat/bar position
expressive residual: deviation from estimated grid
Quantisation must be optional and reversible. The model can use quantised tokens for some tasks, but the original event timing must be retained.
Implement:
estimate_tempo_map(midi) -> TempoMap
estimate_beat_grid(events, tempo_map) -> BeatGrid
assign_metrical_positions(events, beat_grid) -> list[NoteEvent]
estimate_phrase_boundaries(events, tracks, harmony) -> list[Phrase]
estimate_sections(events, harmony, density, instrumentation) -> list[Section]
Phrase boundaries should use multiple cues: rests, melodic cadences, changes in density, registral reset, harmonic cadence, drum fill, and repetition boundaries.
Stage 3: Mathematical definitions of musical functions
This is the most important part of the project. Implement explicit definitions and scores for the following.
3.1 Motif
A motif is a short sequence of structurally salient note events that recurs under one or more transformations.
Represent a candidate motif as:
@dataclass
class Motif:
    id: str
    track_id: int
    events: list[NoteEvent]
    start_beat: float
    end_beat: float
    interval_vector: list[int]
    contour_vector: list[int]
    duration_ratio_vector: list[float]
    metrical_profile: list[float]
    accent_profile: list[float]
    harmonic_context: list[str] | None
    salience_score: float
A motif candidate should normally contain 3–16 notes, but this should be configurable.
Define motif similarity as a weighted combination of:
interval similarity
pitch-contour similarity
rhythmic-ratio similarity
metrical-position similarity
accent/velocity similarity
harmonic-function similarity
edit distance allowing insertion/deletion
Implement:
motif_similarity(a: Motif, b: Motif) -> float
find_motif_candidates(track_events) -> list[Motif]
cluster_motifs(motifs) -> list[MotifFamily]
The system must recognise transformed repetitions. The following transformations must be supported:
exact repetition
transposition
inversion
retrograde
rhythmic augmentation
rhythmic diminution
fragmentation
extension
ornamented repetition
metrical displacement
register displacement
mode shift
diatonic alteration
chromatic alteration
3.2 Transformation operations
Represent operations explicitly:
class TransformationType(Enum):
    EXACT = "exact"
    TRANSPOSE = "transpose"
    INVERT = "invert"
    RETROGRADE = "retrograde"
    AUGMENT = "augment"
    DIMINISH = "diminish"
    FRAGMENT = "fragment"
    EXTEND = "extend"
    ORNAMENT = "ornament"
    DISPLACE_RHYTHM = "displace_rhythm"
    DISPLACE_METRE = "displace_metre"
    CHANGE_REGISTER = "change_register"
    DIATONIC_SHIFT = "diatonic_shift"
    CHROMATIC_SHIFT = "chromatic_shift"
    REHARMONISE = "reharmonise"
Implement:
detect_transformation(source: Motif, target: Motif) -> TransformationSequence
apply_transformation(source: Motif, transformation: TransformationSequence) -> Motif
The detection function should return both the most likely transformation and a confidence score.
3.3 Ornamentation
An ornament is a locally dependent note or group of notes that decorates a structurally stronger note.
Implement ornament detection using:
short duration relative to local median
weak metrical position
stepwise approach/escape
neighbour-note relation
passing-note relation
grace-like timing
low structural salience
Ornament types:
passing tone
neighbour tone
escape tone
appoggiatura-like event
anticipation
turn
mordent-like event
pickup flourish
slide/glide approximation
Implement:
detect_ornaments(events, harmonic_context, beat_grid) -> list[Ornament]
strip_ornaments(events) -> StructuralLine
generate_ornaments(structural_line, style_controls) -> list[NoteEvent]
The generator must be able to add ornaments while preserving the original structural line.
3.4 Countermelody
A countermelody is an independent melodic line that fits the harmonic and rhythmic context while remaining distinct from the primary melody.
Define countermelody quality using:
harmonic compatibility with chord/key context
registral separation from main melody
rhythmic independence
avoidance of persistent parallel motion
avoidance of masking main melody
phrase complementarity
call-and-response relation
motivic relation without direct copying
Implement:
score_countermelody(main_melody, candidate, harmony, metre) -> CountermelodyScore
generate_countermelody(context, controls) -> list[NoteEvent]
The score object should include separate sub-scores:
@dataclass
class CountermelodyScore:
    harmonic_fit: float
    rhythmic_independence: float
    melodic_coherence: float
    registral_fit: float
    anti_masking: float
    motivic_relation: float
    total: float
3.5 Counterpoint
Do not try to implement only species counterpoint. The user wants electronic music and IDM, not a strict Renaissance exercise. Implement counterpoint as a flexible independence-and-compatibility metric.
Counterpoint analysis should include:
consonance/dissonance profile against another line
parallel fifth/octave detection, optional penalty
contrary/oblique/similar motion
voice crossing
voice overlap
registral independence
rhythmic independence
dissonance preparation/resolution where applicable
tension-release shape
Implement strict and relaxed modes:
strict_counterpoint_mode = useful for classical-style testing
relaxed_counterpoint_mode = useful for IDM, ambient, electronic, pop, jazz-informed material
Implement:
analyse_counterpoint(line_a, line_b, harmony, mode="relaxed") -> CounterpointAnalysis
score_counterpoint(line_a, line_b, harmony, mode="relaxed") -> float
3.6 Harmony and chord changes
Implement chord estimation from vertical pitch collections, bass notes, metrical strength and sustained notes.
Represent chords as:
@dataclass
class ChordEvent:
    start_beat: float
    end_beat: float
    root: int | None
    bass: int | None
    quality: str | None
    pitch_classes: set[int]
    confidence: float
    local_key: str | None
    roman: str | None
Implement:
estimate_chords(events, beat_grid) -> list[ChordEvent]
estimate_local_key(chords, melody, window_beats) -> list[KeyRegion]
detect_chord_changes(chords) -> list[ChordChange]
detect_modulations(chords, melody, sections) -> list[Modulation]
A modulation/key change should not be defined as a single accidental chord. Require a sustained change of tonal centre, harmonic support, and ideally a phrase or cadence boundary.
3.7 Pattern mutation
A pattern mutation is a controlled transformation of a repeated rhythmic/melodic/harmonic pattern.
Represent it as:
@dataclass
class PatternMutation:
    source_pattern_id: str
    target_pattern_id: str
    operations: list[TransformationOperation]
    mutation_strength: float
    recognisability: float
    novelty: float
    functional_context: str
Mutation strength should increase with larger pitch, rhythm, register, density and harmonic changes. Recognisability should remain high when contour, rhythm, metrical accent or harmonic role remains similar.
Implement:
detect_pattern_mutations(patterns) -> list[PatternMutation]
generate_pattern_mutation(pattern, controls) -> Pattern
Controls:
mutation_strength
preserve_rhythm
preserve_contour
preserve_harmony
increase_density
decrease_density
make_softer
make_more_idm
make_less_regular
make_more_syncopated
Stage 4: Dataset building
Build a dataset builder that converts analysed MIDI into training examples.
Each training example should include:
@dataclass
class TrainingExample:
    task: str
    input_context: MusicalContext
    target: list[NoteEvent]
    labels: dict
    source_file: str
    confidence: float
Supported tasks:
continue_melody
melody_from_chords
melody_from_bass_and_drums
bass_from_drums
bass_from_chords
drums_from_bass
countermelody_from_melody_and_chords
ornament_structural_line
strip_ornamentation
mutate_motif
transpose_motif
invert_motif
reharmonise_melody
chords_from_melody
modulate_section
fill_gap
vary_loop
generate_bridge
generate_b_section
The dataset builder must produce both positive and negative examples for critic training.
Positive examples: real musical continuations or real related lines from the corpus.
Negative examples: corrupted or mismatched examples, including:
wrong chord context
random transposition outside key
excessive parallel motion
unrelated rhythm
bad register
overcrowding against main melody
too much repetition
too little repetition
bad bass-drum alignment
destroyed motif recognisability
unresolved dissonance in strict mode
Do not label all generated negatives as equally bad. Assign diagnostic labels explaining what is wrong.
Stage 5: Tokenisation
Use a tokenisation scheme that can encode:
pitch
duration
velocity
onset position
bar/beat position
instrument/track role
chord context
key context
section label
phrase label
task token
style controls
motif id or motif-family id
transformation operation
microtiming residual bucket
density control
syncopation control
Do not rely on plain piano-roll representation. The model needs symbolic event tokens.
Create task prefix tokens:
<TASK_MELODY_FROM_CHORDS>
<TASK_BASS_FROM_DRUMS>
<TASK_COUNTERMELODY>
<TASK_ORNAMENT>
<TASK_STRIP_ORNAMENTS>
<TASK_MUTATE_MOTIF>
<TASK_REHARMONISE>
<TASK_MODULATE>
<TASK_FILL_GAP>
<TASK_VARY_PATTERN>
Create control tokens:
<STYLE_IDM_SOFT>
<STYLE_IDM_GLITCH>
<STYLE_AMBIENT>
<STYLE_JAZZISH>
<STYLE_CLASSICAL>
<DENSITY_LOW>
<DENSITY_MEDIUM>
<DENSITY_HIGH>
<SYNCOPATION_LOW>
<SYNCOPATION_MEDIUM>
<SYNCOPATION_HIGH>
<TENSION_LOW>
<TENSION_MEDIUM>
<TENSION_HIGH>
<MUTATION_SUBTLE>
<MUTATION_MEDIUM>
<MUTATION_STRONG>
<ORNAMENTATION_NONE>
<ORNAMENTATION_LIGHT>
<ORNAMENTATION_HEAVY>
The user must be able to generate with style controls without needing a large genre-specific training set.
Stage 6: Model design
Do not train one separate full model for every concept unless the code is explicitly configured for it. Use a shared symbolic transformer backbone with separate task adapters or heads.
Implement:
shared backbone model
task-specific adapters
task-specific output heads where needed
critic/ranker model
optional lightweight classifiers for analysis labels
The generator should work by conditional generation:
input context + task token + controls -> candidate output
Examples:
chord progression + <TASK_MELODY_FROM_CHORDS> -> melody
drum pattern + <TASK_BASS_FROM_DRUMS> -> bassline
melody + chords + <TASK_COUNTERMELODY> -> countermelody
motif + <TASK_MUTATE_MOTIF> + <MUTATION_MEDIUM> -> motif variation
structural melody + <TASK_ORNAMENT> -> ornamented melody
melody + <TASK_REHARMONISE> -> chord progression
Use candidate generation plus ranking. For each request, generate multiple candidates, score them with the critic and explicit musical metrics, then return the best candidates to the GUI.
Do not start with a classic GAN. For symbolic MIDI, implement a critic/ranker first. The critic should judge whether a candidate is musically plausible and functionally appropriate in context.
The critic should output:
@dataclass
class CriticScore:
    harmonic_fit: float
    rhythmic_fit: float
    motivic_coherence: float
    novelty: float
    repetition_balance: float
    register_fit: float
    density_fit: float
    style_fit: float
    counterpoint_fit: float
    total: float
    failure_reasons: list[str]
Later, an adversarial or RL-style training loop can be added, but do not make it the foundation.
Stage 7: Generation behaviour
The generation module must support these operations from the GUI and CLI:
Melody from chords
Input:
chord progression
tempo/metre
optional bassline
optional drums
style controls
density controls
range controls
Output:
one or more melody candidates
MIDI export
analysis report explaining motif use, phrase shape, chord-tone/non-chord-tone treatment
Bassline from drums
Input:
drum part
optional chords
tempo/metre
style controls
Output:
bassline candidates
scores for drum-lock, syncopation, harmonic fit, repetition/mutation balance
Countermelody from melody and chords
Input:
main melody
chords
optional bass/drums
desired register
density
style controls
Output:
countermelody candidates
counterpoint/countermelody score
anti-masking score
Motif mutation
Input:
selected motif
mutation strength
preserve rhythm yes/no
preserve contour yes/no
preserve harmony yes/no
Output:
motif variations
operation labels
recognisability score
novelty score
Ornamentation
Input:
structural line
style
ornamentation amount
Output:
ornamented line
ornament labels
structural notes preserved
Reharmonisation
Input:
melody
desired harmonic tension
style
optional original chords
Output:
chord progression candidates
local key analysis
modulation labels if relevant
Pattern variation
Input:
loop or phrase
desired mutation strength
number of bars
style controls
Output:
variation candidates
pattern-mutation report
Stage 8: GUI requirements
Build the GUI only after the backend functions work from the command line.
The GUI must have these panels:
Library panel
The user can:
choose corpus folders
scan MIDI files
see valid/invalid file counts
see track-role summaries
filter by instrument, role, style tag, metre, tempo, density
Analysis panel
The user can import a MIDI file and see:
track roles
chord estimate
local key regions
motif families
detected transformations
phrase boundaries
section boundaries
counterpoint analysis
ornament analysis
pattern mutation analysis
The GUI should provide a piano-roll view. It does not need to be DAW-grade in v1, but it must show notes, tracks, selected motifs and generated output.
Training panel
The user can:
build dataset
choose tasks
start training
pause/resume where possible
view GPU memory usage
view loss curves
view validation metrics
load checkpoints
Generation panel
The user can:
import a MIDI context
select task
set controls
generate candidates
audition candidates through a simple MIDI player if possible
view critic scores
export selected candidate as MIDI
export all candidates as MIDI
Controls must include:
density
syncopation
mutation strength
ornamentation amount
register
harmonic tension
style
temperature
number of candidates
length in bars/beats
preserve original rhythm
preserve original contour
preserve original harmony
Settings panel
The user can set:
CUDA device
model checkpoint folder
dataset folder
MIDI output folder
default PPQ
default soundfont or MIDI playback device
maximum sequence length
batch size
mixed precision on/off
The program must run on Windows.
Provide setup instructions for:
NVIDIA driver
CUDA-compatible PyTorch install
Python environment
package installation
launching GUI
running dataset scan
running training
running generation
Implement GPU detection:
import torch
def get_gpu_info():
    if not torch.cuda.is_available():
        return {"cuda": False}
    return {
        "cuda": True,
        "device_name": torch.cuda.get_device_name(0),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
    }
Use mixed precision where safe.
The program must degrade gracefully to CPU for analysis tasks, but training and generation should warn the user if CUDA is unavailable.
Stage 10: CLI commands
Implement these commands:
motiflab scan-corpus --input "D:/MIDI" --output "D:/MotifLab/index"
motiflab analyse --input "song.mid" --output "analysis.json"
motiflab build-dataset --index "D:/MotifLab/index" --tasks all --output "D:/MotifLab/dataset"
motiflab train-backbone --config configs/training_backbone.yaml
motiflab train-task --task melody_from_chords --config configs/training_tasks.yaml
motiflab train-critic --config configs/critic.yaml
motiflab generate --task melody_from_chords --input context.mid --output out.mid
motiflab gui
Each command must print useful progress and write logs.
Stage 11: Evaluation and tests
Implement unit tests before large-scale training.
Required tests:
MIDI import does not crash on empty/corrupt files
tokenisation round-trip preserves notes within tolerance
motif similarity recognises exact transposition
motif similarity recognises inversion
ornament stripper preserves structural notes
counterpoint scorer penalises voice crossing
countermelody scorer penalises masking main melody
chord estimator identifies simple triads
modulation detector does not call one chromatic chord a key change
generation output contains valid MIDI notes
critic returns diagnostic failure reasons
Create synthetic MIDI examples for tests. Do not rely only on external datasets.
Evaluation metrics:
valid MIDI rate
pitch range validity
note density match
harmonic fit
rhythmic fit
motif recognisability
novelty
self-similarity
repetition balance
counterpoint score
style-control adherence
human audition notes
The GUI should show both model score and diagnostic scores. Do not hide all judgement behind one opaque “AI score”.
Stage 12: Development order
Build in this order:
Project skeleton and environment.
MIDI import/export.
Internal event representation.
Track-role classification.
Timing, beat grid and expressive residuals.
Chord/key estimation.
Motif detection and transformation detection.
Ornament detection.
Countermelody and counterpoint scoring.
Dataset builder.
Tokenisation.
Simple baseline models.
Critic/ranker.
Candidate generation.
GUI.
Training dashboard.
Packaging for Windows.
Do not start with the GUI. Do not start with the neural model. Do not start with DAW integration. The system’s value depends on the analysis layer.
Stage 13: Minimum viable prototype
The first usable prototype must do the following:
import a MIDI file
show tracks and likely roles
estimate chords
find repeated/transformed motifs
detect transposition and inversion
detect simple ornaments
generate a melody from a chord progression using a small trained or pretrained model
generate several candidates
rank candidates with explicit musical scores
export selected MIDI
run through a basic Windows GUI
Do not implement all advanced features before this works.
Stage 14: Second milestone
The second milestone must add:
bassline from drum part
countermelody from melody and chords
motif mutation controls
ornamentation generator
critic/ranker training from real/corrupted examples
style-control tokens
training panel in GUI
checkpoint loading/saving
Stage 15: Third milestone
The third milestone must add:
larger corpus training
IDM-oriented fine-tuning
section-level generation
reharmonisation
modulation suggestions
pattern variation over multi-bar spans
export as separate MIDI stems
batch generation
better piano-roll UI
Stage 16: Dataset strategy
Use a broad corpus first. The system must learn general symbolic music structure before being steered towards IDM.
Recommended dataset approach:
general multitrack MIDI corpus for broad pretraining
expressive performance MIDI for timing/velocity modelling
small curated user corpus for style steering
synthetic transformations for motif/transformation tasks
corrupted examples for critic training
The user may not have enough IDM MIDI. Therefore, the model must not depend on IDM labels alone. Use control features that approximate IDM-relevant musical properties:
syncopation
microtiming
repetition with mutation
density
glitch-like interruption
irregular phrase lengths
soft velocity profile
modal harmony
ambiguous tonality
bass-drum interlock
non-square pattern variation
Implement these as measurable musical controls rather than just genre tags.
Stage 17: What not to do
Do not make a simple four-bar loop generator.
Do not quantise away expressive timing.
Do not rely only on genre labels.
Do not build a chatbot interface as the main program.
Do not generate audio. This is a MIDI/symbolic system.
Do not assume all MIDI tracks are correctly labelled.
Do not assume channel 10 is the only drum indicator.
Do not train only on piano if the target includes bass, drums, chords and melodic counterlines.
Do not make a Max for Live device first. Build the engine first, then later expose it to Max/DAW workflows through MIDI export, local API, OSC, or a plugin bridge.
Do not implement a classic GAN as the first critic system. Use candidate generation plus ranking.
Stage 18: Local API for future DAW or Max integration
After the GUI and engine work, expose a local FastAPI server:
POST /analyse_midi
POST /generate/melody_from_chords
POST /generate/bass_from_drums
POST /generate/countermelody
POST /generate/motif_mutation
POST /generate/ornamentation
POST /generate/reharmonise
GET /models
GET /gpu
Each endpoint should accept a MIDI file or symbolic JSON context and return generated MIDI plus an analysis report.
This will later allow a Max for Live device, Ableton script, Reaper script, or external DAW tool to call the engine.
Stage 19: Output reports
Every generation should produce not only MIDI but also a report:
{
  "task": "countermelody_from_melody_and_chords",
  "model_checkpoint": "...",
  "controls": {
    "style": "idm_soft",
    "density": "medium",
    "syncopation": "high"
  },
  "critic_score": {
    "total": 0.82,
    "harmonic_fit": 0.91,
    "rhythmic_independence": 0.77,
    "motivic_coherence": 0.74,
    "register_fit": 0.89
  },
  "detected_operations": [
    "motif_fragmentation",
    "metrical_displacement",
    "ornamented_repetition"
  ],
  "warnings": [
    "candidate 3 has possible register masking against main melody"
  ]
}
The user needs to know why a candidate was selected.
Stage 20: Coding standards
Write clear, typed Python.
Use dataclasses or Pydantic models for internal representations.
Use YAML configs.
Use logging, not print-only debugging.
Use tqdm for CLI progress.
Use pytest.
Use small synthetic MIDI fixtures.
Every major module must have docstrings explaining the musical assumptions.
Where the system uses a heuristic, make it explicit and configurable.
Where confidence is low, return a confidence score rather than pretending certainty.

Begin by creating the project skeleton and implementing these files:
motiflab/core/event_model.py
motiflab/core/midi_io.py
motiflab/core/timing.py
motiflab/analysis/transformations.py
motiflab/analysis/motif.py
motiflab/tests/test_event_model.py
motiflab/tests/test_motif_detection.py
scripts/analyse_midi.py

The first working demo should:
load a MIDI file
extract notes
estimate rough beat positions
find simple repeated motifs
detect exact repetition and transposition
export analysis.json

Do not proceed to neural training until this works.

After each implementation stage, do the following:

Run the tests.
Explain what works.
Explain what is still heuristic or weak.
List the next files to implement.
Do not skip ahead.
Do not replace the symbolic analysis layer with a generic transformer.
Do not remove expressive timing fields.
