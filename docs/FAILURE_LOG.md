# Failure Log

Persistent English log of failure causes and lessons encountered across
development and runtime of the warranty-analysis pipeline. New entries are
appended by `src/warranty/failure_log.py` via `record(step, context, exc,
lesson, status)`.

## Entry format

```
## YYYY-MM-DD HH:MM:SS — <short title>
- **Step**: <Cell N or module.function>
- **Input context**: <row index, VIN, filename, etc.>
- **Error**: <exception type + message>
- **Root cause**: <why it failed>
- **Lesson / Fix**: <what to change or guard against>
- **Status**: Open | Resolved
```

---

<!-- New entries are appended below this line by failure_log.record() -->
