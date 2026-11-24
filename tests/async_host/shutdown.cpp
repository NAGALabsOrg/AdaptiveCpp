/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

// Leaves 100 running and 100 pending host operations at the end of main and
// checks that the process still ends with status 0 and that every body ran.
//
//   async_host_shutdown_test                 runs the three variants below as child processes
//   async_host_shutdown_test child <file> <variant>
//
// variants: plain (bodies return), throwing (bodies throw after recording),
//           dropped (all event handles are dropped before main ends)

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

int run_child(const std::string &record_file, const std::string &variant) {
  using namespace std::chrono_literals;

  const int fd = ::open(record_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_APPEND, 0644);
  if(fd < 0)
    return 2;

  const bool throwing = variant == "throwing";
  const bool dropped = variant == "dropped";

  auto record_lock = std::make_shared<std::mutex>();
  auto record = [fd, record_lock, throwing](int index) {
    {
      std::lock_guard<std::mutex> guard{*record_lock};
      const std::string line = "done " + std::to_string(index) + "\n";
      if(::write(fd, line.data(), line.size()) < 0)
        std::abort();
    }
    if(throwing)
      throw std::runtime_error{"thrown after recording"};
  };

  sycl::queue q{[](sycl::exception_list) {}};

  std::vector<sycl::event> running;
  for(int i = 0; i < 100; ++i) {
    running.push_back(q.async_host([record, i]() {
      std::this_thread::sleep_for(300ms);
      record(i);
    }));
  }

  std::vector<sycl::event> pending;
  for(int i = 0; i < 100; ++i) {
    pending.push_back(q.async_host([record, i]() { record(100 + i); },
                                   running[static_cast<std::size_t>(i)]));
  }

  if(dropped) {
    running.clear();
    pending.clear();
  }

  // main ends here, with operations running and pending
  return 0;
}

int count_lines(const std::string &file) {
  std::ifstream in{file};
  std::string line;
  int lines = 0;
  while(std::getline(in, line))
    ++lines;
  return lines;
}

bool file_contains(const std::string &file, const std::string &text) {
  std::ifstream in{file};
  std::string line;
  while(std::getline(in, line)) {
    if(line.find(text) != std::string::npos)
      return true;
  }
  return false;
}

// Runs this program as a child; returns false if it did not end with status 0
// within 60 s.
bool run_variant(const std::string &variant, int &lines, bool &report_found) {
  const std::string record_file = "async_host_shutdown_" + variant + ".txt";
  const std::string error_file = "async_host_shutdown_" + variant + ".err";

  const pid_t pid = ::fork();
  if(pid == 0) {
    const int err = ::open(error_file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    ::dup2(err, STDERR_FILENO);
    ::execl("/proc/self/exe", "async_host_shutdown_test", "child", record_file.c_str(),
            variant.c_str(), static_cast<char *>(nullptr));
    ::_exit(127);
  }

  int status = 0;
  bool ended = false;
  for(int waited_ms = 0; waited_ms < 60000; waited_ms += 10) {
    if(::waitpid(pid, &status, WNOHANG) == pid) {
      ended = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if(!ended) {
    ::kill(pid, SIGKILL);
    ::waitpid(pid, &status, 0);
    std::cout << variant << ": did not end within 60 s" << std::endl;
    return false;
  }

  lines = count_lines(record_file);
  report_found = file_contains(error_file, "AddressSanitizer") ||
                 file_contains(error_file, "ThreadSanitizer") ||
                 file_contains(error_file, "terminate") ||
                 file_contains(error_file, "Aborted");
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

}

int main(int argc, char **argv) {
  if(argc == 4 && std::string{argv[1]} == "child")
    return run_child(argv[2], argv[3]);

  bool all_passed = true;
  for(const std::string variant : {"plain", "throwing", "dropped"}) {
    int lines = 0;
    bool report_found = false;
    const bool exited_cleanly = run_variant(variant, lines, report_found);
    const bool passed = exited_cleanly && lines == 200 && !report_found;
    std::cout << variant << ": exit status 0: " << exited_cleanly
              << ", completion records: " << lines << " (expected 200)"
              << ", abort or sanitizer report: " << report_found << " -> "
              << (passed ? "pass" : "FAIL") << std::endl;
    all_passed = all_passed && passed;
  }
  return all_passed ? 0 : 1;
}
