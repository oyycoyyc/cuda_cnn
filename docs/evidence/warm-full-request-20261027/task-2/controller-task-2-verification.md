# Warm Task2 main H20 verification

- MainLinux -S namedmissingmoduleRED beforeimplementation,task-2-red.txt.
- Firstmain97helpersrun96pass/1timeoutretentionFAIL preservedtask-2-green.txt.
  Independentsmallstdlibreproducer confirmsPOSIXrun()TimeoutExpired-afterpipeEOF
  hidesalreadycapturedstderr; docs+mainreprosupportonePopenchild/kill/communicate
  cleanup, notweakeningtest. task-2-timeout-root-cause.txt/library-notes.md.
- FixedmainLinux97helpersPASS/no skips, same27warm+70old, unmodifiedtimeout
  close_streams=True case. task-2-fixed-green.txt. WindowsLFfixturefixwas
  objectiveformatcorrection, no assertion/countrelaxation.

## Actual full10000 W5R101 (single formal GPU process)

- OriginalreviewedTask1toolSHA465b53d46f12a6f8cc6dde34e5da26e1a4a6e1ac9ec12bf3bdc6047ba2b6d782.
  OfficialinferseparateSHA c7c929225d6abfd5b23bae4aead58ac622100a6ede0362b25a1daaadd3ffae0e.
  Actual20epochcourse_train modelretained,best15/CRCf473014f, no retraining.
- ExternalcollectoractuallylaunchedoneGPUtoolwithN10000/B1024/W5/R101,
  nolabelsargument; toolrecomputed/readback/validatedall106vectorsandmatched
  firstrealwarmupvector. Collectorstatuspass/errors[]/exit0, all6before+after
  hashesequal/current. Rawstdout/stderr/5warmup+101measuredrecordsretained.
- Mainindependently reparsesRAW107stderrlineswithseparateregex, checksphase/
  index/fullN/positivefinitewall/invariance/completion, recomputesmean=sum/101
  andmedian=sort[50], min/max, matchesstoredarrays/statistics. Sumall106inner
  durations<=wholeGPUtoolprocesswall, andwarmuptimesnotinmeasuredstatistics.
- MainparsesRAWNlinepredictionstdout, independentlycounts9911againstORIGINAL
  C1testlabels andconvertedint32labels (matching originals). Also matches
  preservedreferencebytes exactly. Nohelperfieldsaloneacceptedforaccuracyproof.
- Actualmean67.82692348514851ms,median57.752155ms,min7.550677ms,
  max638.715843ms. Mean<100andmedian<100PASS; DOESNOTmeanallrounds<100.
  All101samplesretained,including638.715843msoutlierinmean. Warmupdurations
  93.895522,113.087978,57.724755,132.007085,108.231758ms areseparate;
  processwall9004.474383ms separate fromprimarywarmrequestmetric.
- task-2-course-default20-unit-w5-r101.json isactualrawseries;
  task-2-independent-audit.json andtask-2-actual-cli.json aremainchecks.
  docs/course-warm-benchmark-results.md containsactualnumbersandscope/
  identity/outlier/unknownofficialpreprocessing+leaderboard caveats.

## Scope / pending

1886line3filediff,newcollector/newtests/newdocupdate; noTask1/CMake/GPU/
officialruntime/coldhelperchange. Currentrootreport/PNG/ZIP are OLDcoldscope,
notnewwarmartifacts; Task3strictwarmpackagependingafterTask2review/checkpoint.
No claimthatoldcoldstartupwall<100oroldhistoricalcoldartifactalreadyupdated.

Reviewagainstspec/publicprotocol, errorretention/exclusiveoutputs, sourcehash/
rawroundstats/labelsisolation, actualmeaningofmean+medianvsall-roundgate.
Sourcepolicycorrectness/timingloopwasTask1review; thisreviewownscollector.
