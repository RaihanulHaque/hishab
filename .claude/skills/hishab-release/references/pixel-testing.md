# Testing on the Pixel 8

The Pixel (adb serial `38041FDJH00B57`) is the user's own phone with their real Hishab ledger.
On 2026-10-20 a script tapped "the composer" while the app was on the History screen: the taps
opened a real entry's edit sheet and typed over its description (nothing was saved, by luck).
Blind coordinate taps are therefore not allowed.

## Rules

- Use `scripts/app_messages.sh` for in-app messages. It reads the screen through
  `uiautomator dump` before every tap: it only taps when the Home screen is showing (balance card
  and "Recent", no edit or review sheet), checks the composer is empty after clearing and holds
  exactly the typed text before sending, prints the review sheet's items, and closes it with Back
  (never "Save all"). Any unexpected screen aborts with a screenshot and touches nothing.
- If the phone is not on Home, ask the user to put it there rather than navigating yourself.
- If the user may be using the phone (unexpected text in the composer, a different screen), stop.
- Model files for benchmarks go in `/data/local/tmp/kiln/`; delete what you pushed when done.

## Speed

- `kiln-cli` on the phone: `cd /data/local/tmp/kiln && LD_LIBRARY_PATH=. ./kiln-cli -m <gguf> --raw -p '<prompt>' -n 128`
  prints `generate: N tok ... (X tok/s)`.
- Compare two files only as alternating pairs with ~15 s rest; report the median of the paired
  ratios. Check `adb shell dumpsys thermalservice` (status ≥ 1 means throttling) and
  `dumpsys battery` (charging slows it); say so in the write-up.
- llama.cpp tools built for Android need OpenMP (`-DGGML_OPENMP=ON` with the NDK's `libomp.a`)
  and pinning to the big cores (`-t 5 --cpu-mask 1F0`), or numbers are meaningless.

## In-app timing

The `hishab:` logcat line has `prompt N tok (M cached) X ms, gen K tok ... Y tok/s`. M should be
~143 (the system-prompt checkpoint); 0 means the first request after load.
