# Main-agent Task5 verification

- MainLinuxRED: two named missinghelper imports, nofeaturetestsran. task-5-red.txt (systemPython3.10). HelpersimplementedonlyafterRED.
- AuthoritativeGREEN: gemini_env Python3.12.12, `-S` disables sitepackages, TMPDIR is build-owned. All27tests (13fixture/14benchmark) pass without skips. task-5-green-linux.txt.
- Profile ruling: exactpinnedschema retains everyexecutedprocess. `--profile` labels all requestedRruns diagnostic in a separateartifact; mainfastR5 and profileR1 are distinctcalls/files. No hidden extra subprocess or discardedstageoutput. Source/updatedtest/doc carry thispolicy.
- Real conversion: all3fulltestmodes plus60000trainunit generated into freshownedfixtures, exactbytecounts/sourcehasheschecked. Source C1 body allimages thenalllabels, notinterleaved. task-5-fixture-audit.txt and savedmanifests.
- Preservedold810000bytesnapshotSHA verified; last10000uint8extracted intoownreference, notrerun. Source/reference manifests and actuallegacybundles pathsinprogressledger. Oldsourceweights/dataunchanged.

## Actual newGPU full10000 results (LEGACY weights only)

Unit B128/1024/4096 each5freshunprofiledprocesses:9911correct,0reference differences, identicalpredictions. Medians1519.771/1536.751/1518.187ms, everyfullwall100msgateFALSE. Separateprofile1024R1 wall1513.449ms; GPUforward+argmax78.942ms cannotbe substitutedforfullwall. Hostsetup1211.914msincludesallocation/upload/events, noisolatedplatform/nativecausalattribution.

Byte/normalized explicitmodes each1fullcorrectnessprocess also9911/0differences (notR5timingacceptance). Noadditionalbatchtuningabove4096because setupdominates. Keepdefault1024, notcherrypickminorloadnoise.

Main independently parses SAVEDRAWSTDOUT fromall18actualprocessrecords (180000predictions), comparesbyteswithpreservedreference andcountsagainstbothconvertedint32labelsandoriginalC1trailinglabels: all9911/0mismatches. This doesnotmerelytrusthelpercorrect/reference fields. task-5-independent-prediction-audit.json.

Detailedexactwalls/hashes/profile available task-5-measurement-summary.json and task-5-real-benchmarks.txt; rawrunJSONlocalcopies task-5-legacy-*.json. ActualbinarySHA c7c929225d6abfd5b23bae4aead58ac622100a6ede0362b25a1daaadd3ffae0e. No.CUDA/CMake/Task1-4sourcechangedbyTask5. Updatedresultsdoccontainsrealnumbersandclearlegacy/profiling/fulltime/sourceassumptionlimits, notfinalcourse-modelsuccess.

Remaininggates: independentspec+qualityreview, fix/revalidationifneeded, source/protectedhashproof andcheckpoint. Full20epochtraining/newmodel96%acceptance/finalartifacts remainTask6. No image/report/archivegenerated asfinalsuccessyet.
