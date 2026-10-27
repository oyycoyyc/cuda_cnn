#ifndef COURSE_MODEL_IO_H_
#define COURSE_MODEL_IO_H_

#include "checkpoint.h"
#include "interface.h"

#include <string>

// Course model bundle contract: a directory holding the unchanged checkpoint
// format in lenet.bin and a small exact version/configuration file in
// course.meta. The metadata is bound to the checkpoint bytes by an IEEE
// CRC32, which detects accidental corruption and mismatched publication; it
// is not a security signature.

namespace course {

// Owns a loaded course bundle: the canonical checkpoint plus the persisted
// preprocessing mode that inference must use.
struct CourseModel {
  Checkpoint checkpoint;
  InputMode input_mode;
};

// Loads directory/lenet.bin and directory/course.meta and returns independent
// storage. course.meta must be at most 256 bytes and consist of exactly these
// four LF-terminated lines, in this order, with no extra, unknown, duplicate,
// or reordered fields:
//   course_lenet_v1
//   input_mode=unit|byte|normalized
//   shape=1,28,28,10
//   checkpoint_crc32=XXXXXXXX
// where XXXXXXXX is exactly eight lowercase hexadecimal digits holding the
// IEEE CRC32 (reflected polynomial 0xedb88320, initial/final XOR 0xffffffff,
// identical to Python zlib.crc32) of the lenet.bin bytes. The checkpoint is
// validated by the existing checkpoint loader and C/H/W/K must be the
// supported 1/28/28/10. Missing or unreadable files, malformed or oversized
// metadata, an invalid CRC field, a CRC that does not match the checkpoint
// bytes, and any checkpoint invariant violation throw a path-qualified
// std::runtime_error.
CourseModel LoadCourseModel(const std::string& directory);

// Publishes checkpoint and input_mode as directory/lenet.bin followed by
// directory/course.meta inside the caller-owned work directory. Unrelated
// files and subdirectories are preserved untouched, because course work
// directories are not guaranteed to be empty. When either reserved name
// already exists, the existing pair is recognized first: both must be present
// as regular single-link files, course.meta must parse exactly, its CRC must
// match the lenet.bin bytes, and lenet.bin must load as an original
// checkpoint. Replacement is allowed only for such a valid pair. A lone,
// junk, mixed, symlinked, hard-linked, or otherwise malformed reserved
// artifact is refused before any temporary or destination write. Unsupported
// geometry and invalid checkpoints are also rejected without touching the
// directory.
//
// Both files are written through same-directory temporaries and renamed into
// place, with the metadata published last after the checkpoint. A validation
// or I/O failure before that publication leaves any previous entries intact.
// If a failure happens after lenet.bin is replaced but before course.meta is
// replaced, the directory holds the new checkpoint with the previous
// metadata; LoadCourseModel detects a differing-byte mix through the CRC and
// rejects it. No rollback and no transactional isolation from concurrent
// readers or writers is claimed. Errors throw std::runtime_error; allocation
// exceptions may also propagate.
void SaveCourseModel(const std::string& directory, const Checkpoint& checkpoint,
                     InputMode input_mode);

}  // namespace course

#endif  // COURSE_MODEL_IO_H_
