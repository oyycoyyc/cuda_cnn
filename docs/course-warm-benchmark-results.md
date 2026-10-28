# Warm full-request benchmark — actual H20 results

The user clarified that the primary inference timing is warmup followed by
more than100 repetitions, reporting mean and median, including each request's
input transfer, preprocessing and result readback. Planned actual protocol:
5warmups+101measured complete10000-image requests, batch1024, same already
course-trained20epoch model. This protocol has now actually run. Every round
recomputed all predictions and read them back; all106vectors matched internally.

The measured interval includes H2D, GPU FP32 preprocessing, forward+argmax,
D2H, synchronization and host result checks. Model/image disk loading, CUDA
initialization, one-time allocation/weight upload and final stdout formatting
are excluded and will be disclosed. This is not GPU-only event timing.

## Measured result

| Metric | Actual value |
|---|---:|
|Images per complete request|10000|
|Untallied warmup requests|5|
|Measured complete requests|101|
|Arithmetic mean|67.82692348514851ms|
|Median|57.752155ms|
|Minimum|7.550677ms|
|Maximum|638.715843ms|
|Correct predictions|9911/10000 (99.11%)|

**Warm mean<100ms: PASS; warm median<100ms: PASS.** This does not mean that
every individual round was<100ms. In particular,638.715843ms is a retained
measured outlier and is included in the arithmetic mean. No measured sample
was discarded, reordered into a smaller subset or replaced. The5warmup times
are93.895522,113.087978,57.724755,132.007085,108.231758ms; none enters the
101-round mean/median. The whole tool process took9004.474383ms, including
one-time initialization and allwarmups/measurements/output; that process wall
is retained separately and is not the primary warm request metric.

The main controller independently reparsed all107stderr records (5warmups,
101measurements,1completion), recomputed arithmetic mean and sorted middle
value, and compared raw prediction stdout against both the original C1 test
labels and converted external-checker labels. Predictions also exactly match
the preserved reference. The GPU tool receives no labels. Hashes for tool,
officialinfer, model, metadata, images and external labels match bothbefore
andafter execution and were rechecked independently.

| Identity | SHA256 |
|---|---|
|Actual measured warm CUDA tool|`465b53d46f12a6f8cc6dde34e5da26e1a4a6e1ac9ec12bf3bdc6047ba2b6d782`|
|Official infer (separate identity, not the timed toolprocess)|`c7c929225d6abfd5b23bae4aead58ac622100a6ede0362b25a1daaadd3ffae0e`|
|Actual20epoch-trained best checkpoint|`89e1d79758ab50960ef27d24f51d3aca7d722452c84602f41e3f402d2ae92eb9`|

The measured development tool invokes the same reviewed GPU input/forward/
argmax components as officialinfer, with reusable preallocated storage and
the identical batch layout. It is not a cold six-argument infer subprocess
or a GPU-only event measurement. All raw process output,5warmup/101measured
records, sixbefore/afterhashes and statistics are retained in
`course-warm-benchmark-v1` experiment JSON. The collector's97helper tests
pass on Linux/Python3.12.12 with no skips (70prior+27warm).

The previous1657.41ms is a historical cold-process median under a different
timing boundary, not this newly requested metric and not a paired speedup
claim. The actual course training completed20epochs, bestepoch15, CRCf473014f;
no retraining occurred for this timing amendment. Unit input remains the
user-approved project assumption, not confirmed official preprocessing;
official leaderboard timing remains unknown. Collector review and updated
warm report/PNG/archive acceptance remain pending; currentrootartifacts still
show the historicalcoldscope until version-preserving regeneration.
