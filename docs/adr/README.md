# Architecture decision records

One file per decision, written when the decision is made. Copy `template.md`, take the next number, and never rewrite an accepted ADR. Supersede it with a new one instead.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-rustfs-object-storage.md) | RustFS as the S3 object store behind one set of endpoint env vars | Accepted |
| [0002](0002-compose-core-full-profiles.md) | Two Compose profiles: `core` and `full` | Accepted |

Still to write (see the implementation plan, Phase 6): async queue, ONNX + CAM constraint, human promotion gate, label-driven retraining.
