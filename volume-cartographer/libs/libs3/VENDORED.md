Vendored copy of github.com/SuperOptimizer/libs3 (single-TU C23 S3 client over
libcurl). Source of truth is that repo; sync by copying libs3.{c,h},
libs3_internal.h, LICENSE. Current sync: commit 01577e0 (cache resolved
credentials — stop the per-request `aws` CLI spawn that burned ~56% of CPU in
long S3 streaming sessions; PR #1).

Used by core's remote .vca streaming backend (VcaHttpChunkFetcher) for byte-range
S3 reads with IMDS/SSO/env credential resolution, connection pooling, NOSIGNAL
thread-safe transfers, and a stall watchdog.
