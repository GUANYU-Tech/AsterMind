#include "flowsim/flowsim.hh"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(const bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

flowsim::Topology line_topology() {
  flowsim::Topology topology;
  topology.add_node("A");
  topology.add_node("B");
  topology.add_node("C");
  topology.add_link("A", "B", 8.0, 10);
  topology.add_link("B", "C", 8.0, 20);
  return topology;
}

void test_single_flow() {
  const flowsim::Simulator simulator(line_topology());
  const auto report = simulator.run({{"one", "A", "C", 1000, 0}});
  require(report.end_time_ns == 1030, "single flow completion time");
  require(report.flows[0].transmission_end_time_ns == 1000,
          "single flow transmission time");
  require(report.flows[0].path == std::vector<std::string>({"A", "B", "C"}),
          "single flow route");
}

void test_shared_bottleneck() {
  flowsim::Topology topology;
  topology.add_node("A");
  topology.add_node("B");
  topology.add_link("A", "B", 8.0, 0);
  const flowsim::Simulator simulator(std::move(topology));
  const auto report =
      simulator.run({{"one", "A", "B", 1000, 0}, {"two", "A", "B", 1000, 0}});
  require(report.flows[0].completion_time_ns == 2000,
          "first shared flow completion");
  require(report.flows[1].completion_time_ns == 2000,
          "second shared flow completion");
  require(report.links[0].peak_active_flows == 2,
          "shared link peak flow count");
}

void test_bandwidth_reallocation() {
  flowsim::Topology topology;
  topology.add_node("A");
  topology.add_node("B");
  topology.add_link("A", "B", 8.0, 0);
  const flowsim::Simulator simulator(std::move(topology));
  const auto report = simulator.run(
      {{"first", "A", "B", 1000, 0}, {"second", "A", "B", 1000, 500}});
  require(report.flows[0].completion_time_ns == 1500,
          "first staggered flow completion");
  require(report.flows[1].completion_time_ns == 2000,
          "second staggered flow completion");
}

void test_bidirectional_capacity_is_independent() {
  flowsim::Topology topology;
  topology.add_node("A");
  topology.add_node("B");
  topology.add_link("A", "B", 8.0, 0);
  const flowsim::Simulator simulator(std::move(topology));
  const auto report = simulator.run(
      {{"forward", "A", "B", 1000, 0}, {"reverse", "B", "A", 1000, 0}});
  require(report.flows[0].completion_time_ns == 1000,
          "forward direction capacity");
  require(report.flows[1].completion_time_ns == 1000,
          "reverse direction capacity");
}

void test_stable_shortest_route() {
  flowsim::Topology topology;
  for (const auto *node : {"A", "B", "C", "D"}) {
    topology.add_node(node);
  }
  topology.add_link("A", "C", 8.0, 10);
  topology.add_link("C", "D", 8.0, 10);
  topology.add_link("A", "B", 8.0, 10);
  topology.add_link("B", "D", 8.0, 10);
  const auto route = topology.shortest_latency_route("A", "D");
  require(route.nodes == std::vector<std::string>({"A", "B", "D"}),
          "lexicographically stable shortest route");
}

void test_ecmp_routes_spread_equal_cost_flows() {
  flowsim::Topology topology;
  for (const auto *node : {"H0", "L0", "S0", "S1", "L1", "H1"}) {
    topology.add_node(node);
  }
  topology.configure_ecmp_switch("L0", 17);
  topology.configure_ecmp_switch("S0", 31);
  topology.configure_ecmp_switch("S1", 47);
  topology.configure_ecmp_switch("L1", 61);
  topology.add_link("H0", "L0", 8.0, 10);
  topology.add_link("L0", "S0", 8.0, 10);
  topology.add_link("L0", "S1", 8.0, 10);
  topology.add_link("S0", "L1", 8.0, 10);
  topology.add_link("S1", "L1", 8.0, 10);
  topology.add_link("L1", "H1", 8.0, 10);

  std::string via_s0;
  std::string via_s1;
  for (int index = 0; index < 256 && (via_s0.empty() || via_s1.empty());
       ++index) {
    const auto id = "flow-" + std::to_string(index);
    const auto route = topology.ecmp_latency_route("H0", "H1", id);
    require(route.propagation_latency_ns == 40,
            "ECMP route must remain shortest");
    require(route.nodes.size() == 5, "ECMP route hop count");
    if (route.nodes[2] == "S0") {
      via_s0 = id;
    } else if (route.nodes[2] == "S1") {
      via_s1 = id;
    } else {
      throw std::runtime_error("ECMP chose an invalid next hop");
    }
  }
  require(!via_s0.empty() && !via_s1.empty(),
          "ECMP must distribute flows across equal-cost paths");
  require(topology.ecmp_latency_route("H0", "H1", via_s0).nodes ==
              topology.ecmp_latency_route("H0", "H1", via_s0).nodes,
          "ECMP route must be stable for a flow id");

  const flowsim::Simulator simulator(std::move(topology));
  const auto report = simulator.run(
      {{via_s0, "H0", "H1", 1000, 0}, {via_s1, "H0", "H1", 1000, 0}});
  require(report.flows[0].completion_time_ns == 2040,
          "ECMP flow through first path completion");
  require(report.flows[1].completion_time_ns == 2040,
          "ECMP flow through second path completion");
}

void test_disconnected_topology_is_rejected() {
  flowsim::Topology topology;
  topology.add_node("A");
  topology.add_node("B");
  const flowsim::Simulator simulator(std::move(topology));
  bool rejected = false;
  try {
    simulator.run({{"missing", "A", "B", 1, 0}});
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, "disconnected route should be rejected");
}

void test_event_simulator_dynamic_arrival() {
  flowsim::Topology topology;
  topology.add_node("A");
  topology.add_node("B");
  topology.add_link("A", "B", 8.0, 10);

  flowsim::EventSimulator simulator(std::move(topology));
  std::vector<std::string> completions;
  simulator.start_flow("first", "A", "B", 1000,
                       [&]() { completions.push_back("first"); });
  simulator.schedule(500, [&]() {
    simulator.start_flow("second", "A", "B", 500,
                         [&]() { completions.push_back("second"); });
  });
  simulator.run();

  const auto report = simulator.report();
  require(report.end_time_ns == 1510, "event simulation end time");
  require(report.flows.size() == 2, "event simulation flow count");
  require(report.flows[0].completion_time_ns == 1510,
          "first dynamic flow completion");
  require(report.flows[1].completion_time_ns == 1510,
          "second dynamic flow completion");
  require(completions.size() == 2, "dynamic completion callback count");
}

void test_event_simulator_reports_control_flow() {
  flowsim::Topology topology;
  topology.add_node("A");
  topology.add_node("B");
  topology.add_link("A", "B", 8.0, 10);

  flowsim::EventSimulator simulator(std::move(topology));
  simulator.start_flow("ack", "B", "A", 64, []() {}, true);
  simulator.run();
  const auto report = simulator.report();
  require(report.flows.size() == 1, "control flow report count");
  require(report.flows[0].is_control, "control flow must be marked");
}

void test_event_simulator_accepts_many_small_progress_events() {
  flowsim::Topology topology;
  topology.add_node("A");
  flowsim::EventSimulator simulator(std::move(topology));

  constexpr int event_count = 1500;
  int processed = 0;
  std::function<void()> schedule_next;
  schedule_next = [&]() {
    ++processed;
    if (processed == event_count) {
      simulator.stop();
      return;
    }
    simulator.schedule(0.0000000005L, schedule_next);
  };
  simulator.schedule(0.0L, schedule_next);
  simulator.run();

  require(processed == event_count,
          "small progress events must not be mistaken for a deadlock");
}

void test_event_simulator_completes_fractional_residuals() {
  flowsim::Topology topology;
  for (int node = 0; node < 8; ++node) {
    topology.add_node(std::to_string(node));
  }
  for (const auto &[left, right] :
       std::vector<std::pair<int, int>>{{0, 1}, {0, 2}, {0, 6}, {1, 3},
                                        {1, 7}, {2, 3}, {2, 4}, {3, 5},
                                        {4, 5}, {4, 6}, {5, 7}, {6, 7}}) {
    topology.add_link(std::to_string(left), std::to_string(right), 200.0,
                      5000);
  }

  flowsim::EventSimulator simulator(std::move(topology));
  constexpr int round_count = 14;
  int round = 0;
  int completed_in_round = 0;
  std::function<void()> launch_round;
  launch_round = [&]() {
    completed_in_round = 0;
    for (int source = 0; source < 8; ++source) {
      const int target = (source + 1) % 8;
      const auto id = "round-" + std::to_string(round) + "-flow-" +
                      std::to_string(source);
      simulator.start_flow(
          id, std::to_string(source), std::to_string(target), 131072,
          [&]() {
            ++completed_in_round;
            if (completed_in_round != 8) {
              return;
            }
            ++round;
            if (round == round_count) {
              simulator.stop();
            } else {
              launch_round();
            }
          });
    }
  };
  launch_round();
  simulator.run();

  require(round == round_count,
          "fractional residuals must not stall collective rounds");
  require(simulator.report().flows.size() == round_count * 8,
          "all collective flows must complete");
}

void test_event_simulator_completes_sub_precision_duration() {
  flowsim::Topology topology;
  topology.add_node("A");
  topology.add_node("B");
  topology.add_link("A", "B", 400.0, 0);

  flowsim::EventSimulator simulator(std::move(topology));
  bool completed = false;
  simulator.schedule(1'100'000'000.0L, [&]() {
    simulator.start_flow("tiny", "A", "B", 1,
                         [&]() { completed = true; });
  });
  simulator.run();

  require(completed,
          "sub-precision flow duration at a large timestamp must complete");
  require(simulator.report().flows.size() == 1,
          "sub-precision flow must be reported");
}

} // namespace

int main() {
  const std::vector<std::pair<std::string, std::function<void()>>> tests = {
      {"single_flow", test_single_flow},
      {"shared_bottleneck", test_shared_bottleneck},
      {"bandwidth_reallocation", test_bandwidth_reallocation},
      {"bidirectional_capacity", test_bidirectional_capacity_is_independent},
      {"stable_shortest_route", test_stable_shortest_route},
      {"ecmp_routes_spread_equal_cost_flows",
       test_ecmp_routes_spread_equal_cost_flows},
      {"disconnected_topology", test_disconnected_topology_is_rejected},
      {"event_simulator_dynamic_arrival",
       test_event_simulator_dynamic_arrival},
      {"event_simulator_reports_control_flow",
       test_event_simulator_reports_control_flow},
      {"event_simulator_many_small_progress_events",
       test_event_simulator_accepts_many_small_progress_events},
      {"event_simulator_fractional_residuals",
       test_event_simulator_completes_fractional_residuals},
      {"event_simulator_sub_precision_duration",
       test_event_simulator_completes_sub_precision_duration},
  };

  int failures = 0;
  for (const auto &[name, test] : tests) {
    try {
      test();
      std::cout << "[PASS] " << name << '\n';
    } catch (const std::exception &error) {
      ++failures;
      std::cerr << "[FAIL] " << name << ": " << error.what() << '\n';
    }
  }
  if (failures != 0) {
    std::cerr << failures << " FlowSim test(s) failed\n";
    return 1;
  }
  std::cout << tests.size() << " FlowSim tests passed\n";
  return 0;
}
