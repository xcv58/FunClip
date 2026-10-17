# Local glossary audio benchmark

This compares no glossary against the selected glossary through the same
`VideoClipper.recog` path used by the service. It retains original ASR output,
aligned sentence data, active/unsupported hotwords, display formatting status,
model hashes and final-reference hashes. It makes no LLM calls.

The model IDs, VAD, punctuation model, cached weights and MPS device are unchanged.
The runner uses their cached absolute paths with updates/remote code disabled;
missing weights fail rather than downloading a replacement. It alternates the
baseline/glossary order across cases. It is a small repeated-measures comparison,
not an independently transcribed corpus or an overall CER measurement.

```sh
funclip-env/bin/python scripts/eval_glossary_audio.py \
  --manifest eval/glossary_audio_cases/manifest.json \
  --audio-root eval/model_migration_runs/2026-10-17/three_episode_review \
  --references-root '/Volumes/Crucial X10/srt' \
  --output eval/model_migration_runs/2026-10-17/glossary_audio/new_run
```

Provide the separately held, SHA-pinned 16 kHz mono audio derivatives under
`--audio-root`. The twenty windows total 347 seconds from 澎湖海鲜, 花火大会 and
花穹云雾林. Neither source media nor final SRTs are output destinations. All 24
final references and cached model files are hashed before/after recognition.
Use a new output directory for each run; audio and generated transcripts stay
in ignored local artifacts.

The initial run used 15 preselected term-clip targets: baseline 9/15, glossary
15/15, with no target losses. These count accepted name presence in a clip,
including already-correct names; they do not measure every occurrence or every
word. The profile vocabulary came from the same episodes' approved references
and previous local menu/sign evidence, so this is not held-out generalization.

Reference review exposed two annotation omissions: gardens-70-82 also contains
云雾林, and gardens-1153-1162 contains 花穹 and 云雾林 and is not a negative control.
The versioned manifest includes these corrections and has 18 targets and eight
negative controls. Rescoring the saved outputs gives 9/18 to 18/18; that is a
post-hoc annotation audit, not a separate ASR run. The initial 15-target manifest
and raw results remain unchanged in local artifacts.

There is a likely extra 冰卷 insertion around seafood 796 seconds, outside the
scored 澎湖 target in that clip. The reference says 谢谢谢谢, baseline ASR added
姐姐, and hotwords changed it to 冰卷. This prevents calling the result globally
error-free. Eight audited negative controls had no newly inserted glossary
names, but that small sample does not rule out errors elsewhere. Crab spellings
红蟳/红𫊻 are excluded from the score because the current model cannot encode them;
its 胡鲟洲 error remains a review candidate. Unclear negation, noisy service
dialogue and music passages remain unresolved outside this name-presence score.

Local evidence lives in
`eval/model_migration_runs/2026-10-17/glossary_audio/`: `paired/results.json`,
`paired/metadata.json`, `AUDIT.json` and `REPORT.html`. LLM integration is checked
with controlled responses in unit tests; this delivery makes no new empirical
claim about live English translation quality.
