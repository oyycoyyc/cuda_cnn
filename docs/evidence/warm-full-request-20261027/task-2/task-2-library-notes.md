# CPython subprocess timeout retention

Context7 `/python/cpython`, official `Doc/library/subprocess.rst` consulted
2026-10-28 after actualPython3.12.12/Linux failure. `Popen.communicate(timeout)`
mayraiseTimeoutExpired; it doesnotkillthechild. Officialcleanup pattern:

```python
proc = subprocess.Popen(..., stdout=subprocess.PIPE, stderr=subprocess.PIPE)
try:
    out, err = proc.communicate(timeout=timeout)
except subprocess.TimeoutExpired:
    proc.kill()
    out, err = proc.communicate()
```

Documentation says retrying communication doesnotloseoutput; aftertimeout use
communicateagain ratherthan waitalone tofinishpipesandpopulate returncode.
Actualmainreproducer independently confirmed that run() exposedNonepartialdata
whenEOFprecededexit-timeout, while thePopenpattern retainedactualstderr. No
dependency/platformchange. Exactlyone child isstarted; no benchmarkretry occurs.
