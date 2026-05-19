/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace gluten {

// Singleton sink for per-query / per-stage shuffle profiling rows.
//
// Activated only when env var GLUTEN_PROFILE_SHUFFLE is set to a filesystem
// path. When unset, every public method is a cheap no-op (one atomic load).
//
// CSV layout (single file, 36 columns):
//
//   record_type,query,jar,iter,writer_id,batch_idx,
//   n_rows,n_cols,
//   n_flat,n_dict,n_const,n_lazy,n_other_enc,
//   n_bigint,n_int32,n_int16,n_int8,n_real,n_double,n_decimal,
//   n_varchar,n_varbinary,n_timestamp,n_date,n_bool,n_complex,n_other_type,
//   stage_name,wall_ns,cpu_ns,n_calls,
//   bytes_raw,bytes_compressed,compress_ns,write_ns,evict_ns,n_partitions,
//   iso_timestamp,pid
//
// Each record_type uses only the relevant columns; the rest are left empty.
//
// Three companion env vars label rows (no-op if unset):
//   GLUTEN_PROFILE_QUERY   — e.g. "tpcds-q67"
//   GLUTEN_PROFILE_JAR     — e.g. "jar-profile.jar"
//   GLUTEN_PROFILE_ITER    — e.g. "1"
class ShuffleProfileSink {
 public:
  static ShuffleProfileSink& instance();

  // Cheap (one atomic load) when disabled. Inline-friendly callers should
  // gate with enabled() first to avoid even constructing argument strings.
  bool enabled() const {
    return enabled_.load(std::memory_order_relaxed);
  }

  // Assign / fetch a sequential writer id for this object. Threadsafe.
  int64_t newWriterId();

  // Per-batch encoding survey row. n_per_encoding[5] = {flat,dict,const,lazy,other};
  // n_per_type[11] = {bigint,int32,int16,int8,real,double,decimal,
  //                   varchar,varbinary,timestamp,date_bool_complex_other_bundled_into_extras}.
  // We pass them as separate counters for clarity.
  struct BatchSurvey {
    int64_t n_rows{0};
    int32_t n_cols{0};
    int32_t n_flat{0};
    int32_t n_dict{0};
    int32_t n_const{0};
    int32_t n_lazy{0};
    int32_t n_other_enc{0};
    int32_t n_bigint{0};
    int32_t n_int32{0};
    int32_t n_int16{0};
    int32_t n_int8{0};
    int32_t n_real{0};
    int32_t n_double{0};
    int32_t n_decimal{0};
    int32_t n_varchar{0};
    int32_t n_varbinary{0};
    int32_t n_timestamp{0};
    int32_t n_date{0};
    int32_t n_bool{0};
    int32_t n_complex{0};
    int32_t n_other_type{0};
  };
  void writeBatchSurvey(int64_t writer_id, int64_t batch_idx, const BatchSurvey& s);

  // Per-stage timer row (one per CpuWallTimingType per writer-stop).
  void writeStageTimer(
      int64_t writer_id,
      const std::string& stage_name,
      int64_t wall_ns,
      int64_t cpu_ns,
      int64_t n_calls);

  // Per-writer byte / compress / write breakdown row.
  struct ByteBreakdown {
    int64_t bytes_raw{0};
    int64_t bytes_compressed{0};
    int64_t compress_ns{0};
    int64_t write_ns{0};
    int64_t evict_ns{0};
    int32_t n_partitions{0};
  };
  void writeByteBreakdown(int64_t writer_id, const ByteBreakdown& b);

  // One-shot "this process is now profiling X" marker row.
  void writeRunMeta();

 private:
  ShuffleProfileSink();
  ~ShuffleProfileSink();
  ShuffleProfileSink(const ShuffleProfileSink&) = delete;
  ShuffleProfileSink& operator=(const ShuffleProfileSink&) = delete;

  void writeHeaderLocked();

  std::atomic<bool> enabled_{false};
  std::atomic<int64_t> nextWriterId_{0};
  std::atomic<bool> runMetaWritten_{false};

  // Process-wide labels resolved once from env at construction.
  std::string query_;
  std::string jar_;

  // Per-iteration label. Refreshed from a sidecar file
  // ($GLUTEN_PROFILE_SHUFFLE_ITER_FILE if set, else $GLUTEN_PROFILE_SHUFFLE
  // + ".iter") at most once per kIterRefreshIntervalMs milliseconds.
  std::string iterPath_;
  std::string iter_;
  int64_t iterLastRefreshMs_{0};
  static constexpr int64_t kIterRefreshIntervalMs = 50;

  std::mutex mu_;
  FILE* fp_{nullptr};

  // Helper: refresh iter_ from the sidecar file under lock if interval elapsed.
  void maybeRefreshIterLocked();
};

} // namespace gluten
