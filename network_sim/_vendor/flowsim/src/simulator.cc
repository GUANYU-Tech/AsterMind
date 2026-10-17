#include "flowsim/flowsim.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace flowsim {

namespace {

constexpr long double kEpsilon = 1e-12L;

struct FlowRuntime {
  FlowRequest request;
  Route route;
  long double remaining_bytes;
  long double rate_bytes_per_ns = 0.0L;
  long double transmission_end_time = 0.0L;
  long double completion_time = 0.0L;
  bool transmitting = false;
  bool completed = false;
};

struct DeliveryEvent {
  long double time;
  std::size_t flow_index;
};

struct DeliveryGreater {
  bool operator()(const DeliveryEvent &left, const DeliveryEvent &right) const {
    if (left.time != right.time) {
      return left.time > right.time;
    }
    return left.flow_index > right.flow_index;
  }
};

std::uint64_t ceil_time_ns(const long double value) {
  if (value <= 0.0L) {
    return 0;
  }
  const auto rounded = std::round(value);
  const auto normalized =
      std::fabs(value - rounded) <= kEpsilon ? rounded : std::ceil(value);
  if (normalized >
      static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    throw std::overflow_error("simulation time overflow");
  }
  return static_cast<std::uint64_t>(normalized);
}

void checked_add(std::uint64_t &value, const std::uint64_t increment) {
  if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
    throw std::overflow_error("link byte counter overflow");
  }
  value += increment;
}

} // namespace

Simulator::Simulator(Topology topology) : topology_(std::move(topology)) {}

SimulationReport
Simulator::run(const std::vector<FlowRequest> &requests) const {
  std::set<std::string> flow_ids;
  std::vector<FlowRuntime> runtimes;
  runtimes.reserve(requests.size());
  for (const auto &request : requests) {
    if (request.id.empty()) {
      throw std::invalid_argument("flow id must not be empty");
    }
    if (!flow_ids.insert(request.id).second) {
      throw std::invalid_argument("duplicate flow id: " + request.id);
    }
    auto route = topology_.ecmp_latency_route(request.source, request.target,
                                               request.id);
    runtimes.push_back({request, std::move(route),
                        static_cast<long double>(request.size_bytes)});
  }

  std::vector<std::size_t> arrivals(requests.size());
  for (std::size_t index = 0; index < arrivals.size(); ++index) {
    arrivals[index] = index;
  }
  std::stable_sort(
      arrivals.begin(), arrivals.end(), [&](const auto left, const auto right) {
        return requests[left].start_time_ns < requests[right].start_time_ns;
      });

  const auto &topology_links = topology_.links();
  std::vector<std::uint64_t> link_bytes(topology_links.size(), 0);
  std::vector<long double> link_busy_time(topology_links.size(), 0.0L);
  std::vector<std::size_t> link_peak_flows(topology_links.size(), 0);
  std::vector<std::size_t> link_active_flows(topology_links.size(), 0);
  std::vector<std::size_t> active;
  std::priority_queue<DeliveryEvent, std::vector<DeliveryEvent>,
                      DeliveryGreater>
      deliveries;

  auto recompute_rates = [&]() {
    std::fill(link_active_flows.begin(), link_active_flows.end(), 0);
    for (const auto flow_index : active) {
      runtimes[flow_index].rate_bytes_per_ns = 0.0L;
      for (const auto link_index : runtimes[flow_index].route.link_indices) {
        ++link_active_flows[link_index];
      }
    }
    for (std::size_t link_index = 0; link_index < link_active_flows.size();
         ++link_index) {
      link_peak_flows[link_index] =
          std::max(link_peak_flows[link_index], link_active_flows[link_index]);
    }

    std::vector<long double> residual_capacity;
    residual_capacity.reserve(topology_links.size());
    for (const auto &link : topology_links) {
      residual_capacity.push_back(link.capacity_bytes_per_ns);
    }
    std::vector<std::size_t> unresolved = active;

    while (!unresolved.empty()) {
      std::vector<std::size_t> unresolved_per_link(topology_links.size(), 0);
      for (const auto flow_index : unresolved) {
        for (const auto link_index : runtimes[flow_index].route.link_indices) {
          ++unresolved_per_link[link_index];
        }
      }

      long double increment = std::numeric_limits<long double>::infinity();
      std::size_t limiting_link = topology_links.size();
      for (std::size_t link_index = 0; link_index < unresolved_per_link.size();
           ++link_index) {
        if (unresolved_per_link[link_index] == 0) {
          continue;
        }
        const auto candidate =
            residual_capacity[link_index] /
            static_cast<long double>(unresolved_per_link[link_index]);
        if (candidate < increment) {
          increment = candidate;
          limiting_link = link_index;
        }
      }
      if (!std::isfinite(increment) || increment <= kEpsilon ||
          limiting_link == topology_links.size()) {
        throw std::runtime_error(
            "unable to allocate positive bandwidth to active flows");
      }

      for (const auto flow_index : unresolved) {
        runtimes[flow_index].rate_bytes_per_ns += increment;
      }
      for (std::size_t link_index = 0; link_index < unresolved_per_link.size();
           ++link_index) {
        residual_capacity[link_index] -=
            increment *
            static_cast<long double>(unresolved_per_link[link_index]);
        if (residual_capacity[link_index] < kEpsilon) {
          residual_capacity[link_index] = 0.0L;
        }
      }

      std::vector<bool> saturated(topology_links.size(), false);
      for (std::size_t link_index = 0; link_index < topology_links.size();
           ++link_index) {
        const auto tolerance = std::max(
            kEpsilon,
            topology_links[link_index].capacity_bytes_per_ns * kEpsilon);
        saturated[link_index] = unresolved_per_link[link_index] > 0 &&
                                residual_capacity[link_index] <= tolerance;
      }
      saturated[limiting_link] = true;

      std::vector<std::size_t> next_unresolved;
      for (const auto flow_index : unresolved) {
        bool is_limited = false;
        for (const auto link_index : runtimes[flow_index].route.link_indices) {
          if (saturated[link_index]) {
            is_limited = true;
            break;
          }
        }
        if (!is_limited) {
          next_unresolved.push_back(flow_index);
        }
      }
      unresolved = std::move(next_unresolved);
    }
  };

  std::size_t arrival_cursor = 0;
  std::size_t completed_count = 0;
  long double current_time = 0.0L;

  while (completed_count < runtimes.size()) {
    long double next_time = std::numeric_limits<long double>::infinity();
    if (arrival_cursor < arrivals.size()) {
      next_time = std::min(
          next_time, static_cast<long double>(
                         requests[arrivals[arrival_cursor]].start_time_ns));
    }
    if (!deliveries.empty()) {
      next_time = std::min(next_time, deliveries.top().time);
    }
    for (const auto flow_index : active) {
      const auto rate = runtimes[flow_index].rate_bytes_per_ns;
      if (rate <= kEpsilon) {
        throw std::runtime_error("active flow has zero bandwidth");
      }
      next_time =
          std::min(next_time,
                   current_time + runtimes[flow_index].remaining_bytes / rate);
    }
    if (!std::isfinite(next_time)) {
      throw std::runtime_error(
          "simulation deadlock: no event can make progress");
    }
    if (next_time + kEpsilon < current_time) {
      throw std::logic_error("simulation time moved backwards");
    }

    const long double elapsed = std::max(0.0L, next_time - current_time);
    for (const auto flow_index : active) {
      auto &runtime = runtimes[flow_index];
      runtime.remaining_bytes -= runtime.rate_bytes_per_ns * elapsed;
      if (runtime.remaining_bytes < kEpsilon) {
        runtime.remaining_bytes = 0.0L;
      }
    }
    for (std::size_t link_index = 0; link_index < link_active_flows.size();
         ++link_index) {
      if (link_active_flows[link_index] > 0) {
        link_busy_time[link_index] += elapsed;
      }
    }
    current_time = next_time;

    for (const auto flow_index : active) {
      auto &runtime = runtimes[flow_index];
      if (runtime.remaining_bytes > kEpsilon) {
        continue;
      }
      runtime.transmitting = false;
      runtime.transmission_end_time = current_time;
      for (const auto link_index : runtime.route.link_indices) {
        checked_add(link_bytes[link_index], runtime.request.size_bytes);
      }
      deliveries.push({current_time + static_cast<long double>(
                                          runtime.route.propagation_latency_ns),
                       flow_index});
    }
    active.erase(std::remove_if(active.begin(), active.end(),
                                [&](const auto index) {
                                  return !runtimes[index].transmitting;
                                }),
                 active.end());

    while (!deliveries.empty() &&
           deliveries.top().time <= current_time + kEpsilon) {
      const auto event = deliveries.top();
      deliveries.pop();
      auto &runtime = runtimes[event.flow_index];
      if (!runtime.completed) {
        runtime.completion_time = event.time;
        runtime.completed = true;
        ++completed_count;
      }
    }

    while (arrival_cursor < arrivals.size() &&
           static_cast<long double>(
               requests[arrivals[arrival_cursor]].start_time_ns) <=
               current_time + kEpsilon) {
      const auto flow_index = arrivals[arrival_cursor++];
      auto &runtime = runtimes[flow_index];
      if (runtime.request.size_bytes == 0 ||
          runtime.route.link_indices.empty()) {
        runtime.transmission_end_time = current_time;
        runtime.completion_time = current_time;
        runtime.completed = true;
        ++completed_count;
        continue;
      }
      runtime.transmitting = true;
      active.push_back(flow_index);
    }

    recompute_rates();
  }

  SimulationReport report{};
  report.end_time_ns = 0;
  report.flows.reserve(runtimes.size());
  for (const auto &runtime : runtimes) {
    const auto start_time = runtime.request.start_time_ns;
    const auto transmission_end = ceil_time_ns(runtime.transmission_end_time);
    const auto completion_time = ceil_time_ns(runtime.completion_time);
    const auto fct = completion_time - start_time;
    const double throughput =
        fct == 0 ? 0.0
                 : static_cast<double>(runtime.request.size_bytes) * 8.0 /
                       static_cast<double>(fct);
    report.flows.push_back({runtime.request.id, runtime.request.source,
                            runtime.request.target, runtime.request.size_bytes,
                            start_time, transmission_end, completion_time, fct,
                            throughput, runtime.route.nodes,
                            runtime.request.is_control});
    report.end_time_ns = std::max(report.end_time_ns, completion_time);
  }

  report.links.reserve(topology_links.size());
  for (std::size_t link_index = 0; link_index < topology_links.size();
       ++link_index) {
    const auto &link = topology_links[link_index];
    double utilization = 0.0;
    if (report.end_time_ns > 0) {
      utilization =
          static_cast<double>(static_cast<long double>(link_bytes[link_index]) /
                              (link.capacity_bytes_per_ns *
                               static_cast<long double>(report.end_time_ns)));
      utilization = std::min(1.0, std::max(0.0, utilization));
    }
    report.links.push_back({link.source, link.target, link.bandwidth_gbps,
                            link.latency_ns, link_bytes[link_index],
                            ceil_time_ns(link_busy_time[link_index]),
                            link_peak_flows[link_index], utilization});
  }
  return report;
}

} // namespace flowsim
