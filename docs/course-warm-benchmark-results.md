# Warm full-request benchmark — pending measurement

The user clarified that the primary inference timing is warmup followed by
more than100 repetitions, reporting mean and median, including each request's
input transfer, preprocessing and result readback. Planned actual protocol:
5warmups+101measured complete10000-image requests, batch1024, same already
course-trained20epoch model. Every round recomputes all predictions.

The measured interval includes H2D, GPU FP32 preprocessing, forward+argmax,
D2H, synchronization and host result checks. Model/image disk loading, CUDA
initialization, one-time allocation/weight upload and final stdout formatting
are excluded and will be disclosed. This is not GPU-only event timing.

No warm mean/median or success claim exists yet. The previous1657.41ms is a
historical cold-process median, not this newly requested metric. Full model
training and full10000 accuracy99.11% are already validated; no retraining is
needed. Plans/spec and the new dedicated ledger define the remaining work.
