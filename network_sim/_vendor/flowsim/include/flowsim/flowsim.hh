#ifndef FLOWSIM_FLOWSIM_HH
#define FLOWSIM_FLOWSIM_HH

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace flowsim {

struct DirectedLink {
  std::size_t index;
  std::string source;
  std::string target;
  double bandwidth_gbps;
  long double capacity_bytes_per_ns;
  std::uint64_t latency_ns;
  bool enabled = true;
};

struct Route {
  std::vector<std::string> nodes;
  std::vector<std::size_t> link_indices;
  std::uint64_t propagation_latency_ns = 0;
};

class Topology {
public:
  void add_node(const std::string &node_id);
  void configure_ecmp_switch(const std::string &node_id,
                             std::uint32_t ecmp_seed);
  void add_link(const std::string &source, const std::string &target,
                double bandwidth_gbps, std::uint64_t latency_ns,
                bool bidirectional = true);

  bool has_node(const std::string &node_id) const;
  void set_link_enabled(const std::string &source, const std::string &target, bool enabled);
  Route shortest_latency_route(const std::string &source,
                               const std::string &target) const;
  Route ecmp_latency_route(const std::string &source,
                           const std::string &target,
                           const std::string &flow_id) const;

  const std::set<std::string> &nodes() const;
  const std::vector<DirectedLink> &links() const;

private:
  void add_directed_link(const std::string &source, const std::string &target,
                         double bandwidth_gbps, std::uint64_t latency_ns);

  std::set<std::string> nodes_;
  std::vector<DirectedLink> links_;
  std::map<std::string, std::vector<std::size_t>> adjacency_;
  std::map<std::string, std::vector<std::size_t>> reverse_adjacency_;
  std::map<std::string, std::uint32_t> ecmp_switch_seeds_;
  std::set<std::pair<std::string, std::string>> directed_pairs_;
};

struct FlowRequest {
  std::string id;
  std::string source;
  std::string target;
  std::uint64_t size_bytes;
  std::uint64_t start_time_ns = 0;
  bool is_control = false;
};

struct FlowResult {
  std::string id;
  std::string source;
  std::string target;
  std::uint64_t size_bytes;
  std::uint64_t start_time_ns;
  std::uint64_t transmission_end_time_ns;
  std::uint64_t completion_time_ns;
  std::uint64_t flow_completion_time_ns;
  double average_throughput_gbps;
  std::vector<std::string> path;
  bool is_control = false;
};

struct LinkResult {
  std::string source;
  std::string target;
  double bandwidth_gbps;
  std::uint64_t latency_ns;
  std::uint64_t bytes_transferred;
  std::uint64_t busy_time_ns;
  std::size_t peak_active_flows;
  double utilization;
};

struct SimulationReport {
  std::uint64_t end_time_ns;
  std::vector<FlowResult> flows;
  std::vector<LinkResult> links;
};

// Adapter extension: inspect a running flow without forcing batch completion.
struct FlowProgress {
  std::string id;
  long double remaining_bytes;
  long double rate_bytes_per_ns;
  bool transmitting;
  bool completed;
  long double transmission_end_time_ns;
  long double completion_time_ns;
  bool failed;
  long double failure_time_ns;
  std::string failure_reason;
  std::vector<std::string> path;
};

class Simulator {
public:
  explicit Simulator(Topology topology);
  SimulationReport run(const std::vector<FlowRequest> &requests) const;

private:
  Topology topology_;
};

class EventSimulator {
public:
  using Callback = std::function<void()>;

  explicit EventSimulator(Topology topology);
  ~EventSimulator();

  EventSimulator(const EventSimulator &) = delete;
  EventSimulator &operator=(const EventSimulator &) = delete;
  EventSimulator(EventSimulator &&) noexcept;
  EventSimulator &operator=(EventSimulator &&) noexcept;

  void schedule(long double delay_ns, Callback callback);
  void start_flow(const std::string &id, const std::string &source,
                  const std::string &target, std::uint64_t size_bytes,
                  Callback completion_callback, bool is_control = false);
  void run();
  void stop();

  long double now_ns() const;
  bool has_pending_work() const;
  SimulationReport report() const;
  void advance_until(long double absolute_time_ns);
  long double next_event_time_ns();
  std::vector<FlowProgress> progress();
  SimulationReport completed_report() const;
  void set_link_enabled(const std::string &source, const std::string &target, bool enabled);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace flowsim

#endif // FLOWSIM_FLOWSIM_HH
