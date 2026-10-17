# Review queue audit cases

Eight pinned cases are copied from the unchanged three-video audit. They cover
restaurant names, the 时间/时辣 hint side effect, two unresolved service passages,
浮罗山背, unsupported Malay spellings, Network/net worth and 10500/10100.

The original audit JSON hashes and copied fixture hashes establish provenance.
`profiles.json` preserves the audit hotwords and adds only review annotations:
普罗山贝 and Network as possible confusions and 榴莲蜜 as a distinct spoken alias.
These annotations never replace recognition. The profiles are local evaluation
inputs, not new production catalog entries or independent vocabulary accuracy data.
Final SRTs and source videos are not copied into or modified by this fixture set.

Run the offline cases with:

```sh
funclip-env/bin/python scripts/eval_review_queue.py --output /private/tmp/review-proof
```

Add the existing local `--audit-root` and `--live-audio` for audio-backed exports,
75 cached payloads and fresh reference/hinted recognition. The runner checks
pinned archive/model hashes before and after. No remote media or LLM call is used.
