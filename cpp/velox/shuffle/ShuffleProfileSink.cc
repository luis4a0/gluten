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

#include "shuffle/ShuffleProfileSink.h"

#include <chrono>
#include <cerrno>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sstream>
#include <unistd.h>

namespace gluten {

namespace {

std::string envOrEmpty(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

// All rows are emitted with exactly this many columns. The mapping is:
//
//   0 record_type        | 13 n_bigint        | 27 stage_name
//   1 query              | 14 n_int32         | 28 wall_ns
//   2 jar                | 15 n_int16         | 29 cpu_ns
//   3 iter               | 16 n_int8          | 30 n_calls
//   4 writer_id          | 17 n_real          | 31 bytes_raw
//   5 batch_idx          | 18 n_double        | 32 bytes_compressed
//   6 n_rows             | 19 n_decimal       | 33 compress_ns
//   7 n_cols             | 20 n_varchar       | 34 write_ns
//   8 n_flat             | 21 n_varbinary     | 35 evict_ns
//   9 n_dict             | 22 n_timestamp     | 36 n_partitions
//  10 n_const            | 23 n_date          | 37 iso_timestamp
//  11 n_lazy             | 24 n_bool          | 38 pid
//  12 n_other_enc        | 25 n_complex
//                        | 26 n_other_type
//
constexpr int kNumColumns = 39;

void writeRow(FILE* fp, std::initializer_list<std::string> cells) {
  bool first = true;
  for (const auto& c : cells) {
    if (!first) {
      std::fputc(',', fp);
    }
    std::fputs(c.c_str(), fp);
    first = false;
  }
  std::fputc('\n', fp);
}

template <typename T>
std::string toStr(T v) {
  std::ostringstream os;
  os << v;
  return os.str();
}

} // namespace

ShuffleProfileSink& ShuffleProfileSink::instance() {
  static ShuffleProfileSink kInstance;
  return kInstance;
}

ShuffleProfileSink::ShuffleProfileSink() {
  std::string path = envOrEmpty("GLUTEN_PROFILE_SHUFFLE");
  if (path.empty()) {
    return;
  }
  bool isNewFile = (access(path.c_str(), F_OK) != 0);
  fp_ = std::fopen(path.c_str(), "a");
  if (fp_ == nullptr) {
    std::fprintf(stderr, "[ShuffleProfileSink] failed to open %s: %s\n", path.c_str(), std::strerror(errno));
    return;
  }
  query_ = envOrEmpty("GLUTEN_PROFILE_QUERY");
  jar_ = envOrEmpty("GLUTEN_PROFILE_JAR");
  // Iter is read fresh from a sidecar file on every batch (one write per
  // batch of rows), to support multi-iteration runs in a single process.
  iterPath_ = envOrEmpty("GLUTEN_PROFILE_SHUFFLE_ITER_FILE");
  if (iterPath_.empty()) {
    iterPath_ = path + ".iter";
  }
  iter_ = envOrEmpty("GLUTEN_PROFILE_ITER"); // initial fallback
  if (isNewFile) {
    std::lock_guard<std::mutex> lock(mu_);
    writeHeaderLocked();
  }
  enabled_.store(true, std::memory_order_release);
}

ShuffleProfileSink::~ShuffleProfileSink() {
  if (fp_ != nullptr) {
    std::fflush(fp_);
    std::fclose(fp_);
    fp_ = nullptr;
  }
}

void ShuffleProfileSink::writeHeaderLocked() {
  std::fputs(
      "record_type,query,jar,iter,writer_id,batch_idx,"
      "n_rows,n_cols,"
      "n_flat,n_dict,n_const,n_lazy,n_other_enc,"
      "n_bigint,n_int32,n_int16,n_int8,n_real,n_double,n_decimal,"
      "n_varchar,n_varbinary,n_timestamp,n_date,n_bool,n_complex,n_other_type,"
      "stage_name,wall_ns,cpu_ns,n_calls,"
      "bytes_raw,bytes_compressed,compress_ns,write_ns,evict_ns,n_partitions,"
      "iso_timestamp,pid\n",
      fp_);
  std::fflush(fp_);
}

int64_t ShuffleProfileSink::newWriterId() {
  return nextWriterId_.fetch_add(1, std::memory_order_relaxed);
}

void ShuffleProfileSink::maybeRefreshIterLocked() {
  if (iterPath_.empty()) {
    return;
  }
  auto now = std::chrono::steady_clock::now().time_since_epoch();
  int64_t nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
  if (nowMs - iterLastRefreshMs_ < kIterRefreshIntervalMs) {
    return;
  }
  iterLastRefreshMs_ = nowMs;
  FILE* f = std::fopen(iterPath_.c_str(), "r");
  if (f == nullptr) {
    return; // sidecar absent; keep last value
  }
  char buf[64] = {0};
  size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
  std::fclose(f);
  if (n == 0) {
    return;
  }
  std::string s(buf, n);
  // Trim trailing whitespace/newlines
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
    s.pop_back();
  }
  if (!s.empty()) {
    iter_ = s;
  }
}

void ShuffleProfileSink::writeBatchSurvey(int64_t writer_id, int64_t batch_idx, const BatchSurvey& s) {
  if (!enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  std::vector<std::string> cells(kNumColumns, "");
  cells[0] = "batch_survey";
  cells[1] = query_;
  cells[2] = jar_;
  cells[4] = toStr(writer_id);
  cells[5] = toStr(batch_idx);
  cells[6] = toStr(s.n_rows);
  cells[7] = toStr(s.n_cols);
  cells[8] = toStr(s.n_flat);
  cells[9] = toStr(s.n_dict);
  cells[10] = toStr(s.n_const);
  cells[11] = toStr(s.n_lazy);
  cells[12] = toStr(s.n_other_enc);
  cells[13] = toStr(s.n_bigint);
  cells[14] = toStr(s.n_int32);
  cells[15] = toStr(s.n_int16);
  cells[16] = toStr(s.n_int8);
  cells[17] = toStr(s.n_real);
  cells[18] = toStr(s.n_double);
  cells[19] = toStr(s.n_decimal);
  cells[20] = toStr(s.n_varchar);
  cells[21] = toStr(s.n_varbinary);
  cells[22] = toStr(s.n_timestamp);
  cells[23] = toStr(s.n_date);
  cells[24] = toStr(s.n_bool);
  cells[25] = toStr(s.n_complex);
  cells[26] = toStr(s.n_other_type);
  std::lock_guard<std::mutex> lock(mu_);
  maybeRefreshIterLocked();
  cells[3] = iter_;
  bool first = true;
  for (const auto& c : cells) {
    if (!first) {
      std::fputc(',', fp_);
    }
    std::fputs(c.c_str(), fp_);
    first = false;
  }
  std::fputc('\n', fp_);
}

void ShuffleProfileSink::writeStageTimer(
    int64_t writer_id,
    const std::string& stage_name,
    int64_t wall_ns,
    int64_t cpu_ns,
    int64_t n_calls) {
  if (!enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  std::vector<std::string> cells(kNumColumns, "");
  cells[0] = "stage_timer";
  cells[1] = query_;
  cells[2] = jar_;
  cells[4] = toStr(writer_id);
  cells[5] = "-1";
  cells[27] = stage_name;
  cells[28] = toStr(wall_ns);
  cells[29] = toStr(cpu_ns);
  cells[30] = toStr(n_calls);
  std::lock_guard<std::mutex> lock(mu_);
  maybeRefreshIterLocked();
  cells[3] = iter_;
  bool first = true;
  for (const auto& c : cells) {
    if (!first) {
      std::fputc(',', fp_);
    }
    std::fputs(c.c_str(), fp_);
    first = false;
  }
  std::fputc('\n', fp_);
}

void ShuffleProfileSink::writeByteBreakdown(int64_t writer_id, const ByteBreakdown& b) {
  if (!enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  std::vector<std::string> cells(kNumColumns, "");
  cells[0] = "byte_breakdown";
  cells[1] = query_;
  cells[2] = jar_;
  cells[4] = toStr(writer_id);
  cells[5] = "-1";
  cells[31] = toStr(b.bytes_raw);
  cells[32] = toStr(b.bytes_compressed);
  cells[33] = toStr(b.compress_ns);
  cells[34] = toStr(b.write_ns);
  cells[35] = toStr(b.evict_ns);
  cells[36] = toStr(b.n_partitions);
  std::lock_guard<std::mutex> lock(mu_);
  maybeRefreshIterLocked();
  cells[3] = iter_;
  bool first = true;
  for (const auto& c : cells) {
    if (!first) {
      std::fputc(',', fp_);
    }
    std::fputs(c.c_str(), fp_);
    first = false;
  }
  std::fputc('\n', fp_);
}

void ShuffleProfileSink::writeRunMeta() {
  if (!enabled_.load(std::memory_order_relaxed)) {
    return;
  }
  bool expected = false;
  if (!runMetaWritten_.compare_exchange_strong(expected, true)) {
    return;
  }
  std::time_t now = std::time(nullptr);
  std::tm tm_utc;
  gmtime_r(&now, &tm_utc);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

  std::vector<std::string> cells(kNumColumns, "");
  cells[0] = "run_meta";
  cells[1] = query_;
  cells[2] = jar_;
  cells[4] = "-1";
  cells[5] = "-1";
  cells[37] = buf;
  cells[38] = toStr(getpid());

  std::lock_guard<std::mutex> lock(mu_);
  maybeRefreshIterLocked();
  cells[3] = iter_;
  bool first = true;
  for (const auto& c : cells) {
    if (!first) {
      std::fputc(',', fp_);
    }
    std::fputs(c.c_str(), fp_);
    first = false;
  }
  std::fputc('\n', fp_);
  std::fflush(fp_);
}

} // namespace gluten
