// Small C ABI around AsterMind FlowSim. Python never reimplements the allocator.
#include "flowsim/flowsim.hh"
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {
struct Handle {
  flowsim::Topology topology;
  std::unique_ptr<flowsim::EventSimulator> simulator;
  std::string output;
  std::string error;
};

template <typename Operation> int guarded(Handle *h, Operation operation) {
  try {
    h->error.clear();
    operation();
    return 0;
  } catch (const std::exception &e) {
    h->error = e.what();
    return -1;
  }
}

std::string quote(const std::string &text) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : text) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}

void path_json(std::ostream &out, const std::vector<std::string> &nodes) {
  out << '[';
  bool first = true;
  for (const auto &node : nodes) {
    if (!first) out << ',';
    first = false;
    out << quote(node);
  }
  out << ']';
}
} // namespace

EXPORT void *net_create() { return new Handle(); }
EXPORT void net_destroy(Handle *h) { delete h; }
EXPORT const char *net_error(Handle *h) { return h->error.c_str(); }
EXPORT const char *net_output(Handle *h) { return h->output.c_str(); }

EXPORT int net_add_node(Handle *h, const char *node) {
  return guarded(h, [&]() { h->topology.add_node(node); });
}
EXPORT int net_add_link(Handle *h, const char *a, const char *b, double bw, std::uint64_t latency) {
  return guarded(h, [&]() { h->topology.add_link(a, b, bw, latency, false); });
}
EXPORT int net_initialize(Handle *h) {
  return guarded(h, [&]() { h->simulator = std::make_unique<flowsim::EventSimulator>(h->topology); });
}
EXPORT int net_set_link_enabled(Handle *h, const char *a, const char *b, int enabled) {
  return guarded(h, [&]() {
    h->topology.set_link_enabled(a, b, enabled != 0);
    if (h->simulator) h->simulator->set_link_enabled(a, b, enabled != 0);
  });
}
EXPORT int net_candidate_route(Handle *h, const char *a, const char *b) {
  return guarded(h, [&]() {
    auto topology = h->topology;
    for (const auto &link : h->topology.links())
      topology.set_link_enabled(link.source, link.target, true);
    const auto route = topology.shortest_latency_route(a, b);
    std::ostringstream out;
    path_json(out, route.nodes);
    h->output = out.str();
  });
}
EXPORT int net_route(Handle *h, const char *a, const char *b) {
  return guarded(h, [&]() {
    const auto route = h->topology.shortest_latency_route(a, b);
    long double capacity = std::numeric_limits<long double>::infinity();
    for (const auto index : route.link_indices)
      capacity = std::min(capacity, h->topology.links()[index].capacity_bytes_per_ns);
    std::ostringstream out;
    out << std::setprecision(18) << "{\"latency_ns\":" << route.propagation_latency_ns
        << ",\"bandwidth_gbps\":" << static_cast<double>(capacity * 8)
        << ",\"path\":";
    path_json(out, route.nodes);
    out << '}';
    h->output = out.str();
  });
}
EXPORT int net_submit(Handle *h, const char *id, const char *a, const char *b,
                      std::uint64_t bytes, std::uint64_t ready) {
  return guarded(h, [&]() {
    if (!h->simulator) throw std::logic_error("runtime is not initialized");
    const auto delay = static_cast<long double>(ready) - h->simulator->now_ns();
    if (delay < 0) throw std::invalid_argument("ready time is in the past");
    const std::string flow_id(id), source(a), target(b);
    // Validate before enqueueing, so missing routes fail at submission time.
    h->topology.shortest_latency_route(source, target);
    h->simulator->schedule(delay, [h, flow_id, source, target, bytes]() {
      h->simulator->start_flow(flow_id, source, target, bytes, []() {});
    });
  });
}
EXPORT int net_advance(Handle *h, std::uint64_t absolute_time) {
  return guarded(h, [&]() { h->simulator->advance_until(absolute_time); });
}
EXPORT int net_next_event(Handle *h) {
  return guarded(h, [&]() {
    const auto time = h->simulator->next_event_time_ns();
    if (!std::isfinite(time)) h->output = "null";
    else {
      const auto nearest = std::round(time);
      const auto normalized = std::fabs(time - nearest) <= 1e-9L ? nearest : std::ceil(time);
      if (normalized > std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("next event time overflow");
      h->output = std::to_string(static_cast<std::uint64_t>(normalized));
    }
  });
}
EXPORT int net_snapshot(Handle *h) {
  return guarded(h, [&]() {
    const auto progress = h->simulator->progress();
    const auto report = h->simulator->completed_report();
    std::ostringstream out;
    out << std::setprecision(18) << "{\"end_time_ns\":" << report.end_time_ns << ",\"progress\":[";
    bool first = true;
    for (const auto &p : progress) {
      if (!first) out << ',';
      first = false;
      out << "{\"id\":" << quote(p.id) << ",\"remaining_bytes\":" << static_cast<double>(p.remaining_bytes)
          << ",\"rate_gbps\":" << static_cast<double>(p.rate_bytes_per_ns * 8)
          << ",\"transmitting\":" << (p.transmitting ? "true" : "false")
          << ",\"completed\":" << (p.completed ? "true" : "false")
          << ",\"failed\":" << (p.failed ? "true" : "false") << '}';
    }
    out << "],\"flows\":[";
    first = true;
    for (const auto &f : report.flows) {
      if (!first) out << ',';
      first = false;
      out << "{\"id\":" << quote(f.id) << ",\"size_bytes\":" << f.size_bytes
          << ",\"start_time_ns\":" << f.start_time_ns
          << ",\"transmission_end_ns\":" << f.transmission_end_time_ns
          << ",\"completion_time_ns\":" << f.completion_time_ns << ",\"path\":";
      path_json(out, f.path);
      out << '}';
    }
    out << "],\"failures\":[";
    first = true;
    for (const auto &p : progress) {
      if (!p.failed) continue;
      if (!first) out << ',';
      first = false;
      out << "{\"id\":" << quote(p.id) << ",\"remaining_bytes\":" << static_cast<double>(p.remaining_bytes)
          << ",\"failure_time_ns\":" << static_cast<std::uint64_t>(std::ceil(p.failure_time_ns))
          << ",\"reason\":" << quote(p.failure_reason) << ",\"path\":";
      path_json(out, p.path);
      out << '}';
    }
    out << "],\"resources\":[";
    first = true;
    for (const auto &l : report.links) {
      if (!first) out << ',';
      first = false;
      out << "{\"source\":" << quote(l.source) << ",\"target\":" << quote(l.target)
          << ",\"capacity_gbps\":" << l.bandwidth_gbps
          << ",\"bytes_transferred\":" << l.bytes_transferred
          << ",\"busy_time_ns\":" << l.busy_time_ns
          << ",\"peak_active_flows\":" << l.peak_active_flows
          << ",\"utilization\":" << l.utilization << '}';
    }
    out << "]}";
    h->output = out.str();
  });
}
