#include "flowsim/flowsim.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>

namespace flowsim {

namespace {

struct RouteState {
  std::uint64_t latency;
  std::vector<std::string> path;
  std::string node;
};

struct RouteStateGreater {
  bool operator()(const RouteState &left, const RouteState &right) const {
    if (left.latency != right.latency) {
      return left.latency > right.latency;
    }
    if (left.path != right.path) {
      return left.path > right.path;
    }
    return left.node > right.node;
  }
};

struct DistanceState {
  std::uint64_t latency;
  std::string node;
};

struct DistanceStateGreater {
  bool operator()(const DistanceState &left,
                  const DistanceState &right) const {
    if (left.latency != right.latency) {
      return left.latency > right.latency;
    }
    return left.node > right.node;
  }
};

std::uint32_t fnv1a_append(std::uint32_t value, const std::string &text) {
  constexpr std::uint32_t kPrime = 16777619U;
  for (const auto character : text) {
    value ^= static_cast<unsigned char>(character);
    value *= kPrime;
  }
  return value;
}

std::uint32_t mix_ecmp_hash(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x85ebca6bU;
  value ^= value >> 13U;
  value *= 0xc2b2ae35U;
  return value ^ (value >> 16U);
}

} // namespace

void Topology::add_node(const std::string &node_id) {
  if (node_id.empty()) {
    throw std::invalid_argument("node id must not be empty");
  }
  if (!nodes_.insert(node_id).second) {
    throw std::invalid_argument("duplicate node id: " + node_id);
  }
  adjacency_[node_id] = {};
  reverse_adjacency_[node_id] = {};
}

void Topology::configure_ecmp_switch(const std::string &node_id,
                                     const std::uint32_t ecmp_seed) {
  if (!has_node(node_id)) {
    throw std::invalid_argument("ECMP switch does not exist: " + node_id);
  }
  if (!ecmp_switch_seeds_.insert({node_id, ecmp_seed}).second) {
    throw std::invalid_argument("duplicate ECMP switch: " + node_id);
  }
}

void Topology::add_link(const std::string &source, const std::string &target,
                        const double bandwidth_gbps,
                        const std::uint64_t latency_ns,
                        const bool bidirectional) {
  add_directed_link(source, target, bandwidth_gbps, latency_ns);
  if (bidirectional) {
    add_directed_link(target, source, bandwidth_gbps, latency_ns);
  }
}

void Topology::add_directed_link(const std::string &source,
                                 const std::string &target,
                                 const double bandwidth_gbps,
                                 const std::uint64_t latency_ns) {
  if (!has_node(source)) {
    throw std::invalid_argument("link source does not exist: " + source);
  }
  if (!has_node(target)) {
    throw std::invalid_argument("link target does not exist: " + target);
  }
  if (source == target) {
    throw std::invalid_argument("self links are not supported: " + source);
  }
  if (!std::isfinite(bandwidth_gbps) || !(bandwidth_gbps > 0.0)) {
    throw std::invalid_argument("link bandwidth must be positive");
  }

  const auto pair = std::make_pair(source, target);
  if (!directed_pairs_.insert(pair).second) {
    throw std::invalid_argument("duplicate directed link: " + source + " -> " +
                                target);
  }

  const std::size_t index = links_.size();
  links_.push_back({index, source, target, bandwidth_gbps,
                    static_cast<long double>(bandwidth_gbps) / 8.0L,
                    latency_ns});
  adjacency_[source].push_back(index);
  reverse_adjacency_[target].push_back(index);
  shortest_routes_.clear();
  auto &outgoing = adjacency_[source];
  std::sort(outgoing.begin(), outgoing.end(),
            [&](const auto left, const auto right) {
              return std::tie(links_[left].target, links_[left].index) <
                     std::tie(links_[right].target, links_[right].index);
            });
  auto &incoming = reverse_adjacency_[target];
  std::sort(incoming.begin(), incoming.end(),
            [&](const auto left, const auto right) {
              return std::tie(links_[left].source, links_[left].index) <
                     std::tie(links_[right].source, links_[right].index);
            });
}

bool Topology::has_node(const std::string &node_id) const {
  return nodes_.find(node_id) != nodes_.end();
}

void Topology::set_link_enabled(const std::string &source, const std::string &target, bool enabled) {
  for (auto &link : links_) {
    if (link.source == source && link.target == target) {
      if (link.enabled != enabled) shortest_routes_.clear();
      link.enabled = enabled;
      return;
    }
  }
  throw std::invalid_argument("unknown directed link");
}

Route Topology::shortest_latency_route(const std::string &source,
                                       const std::string &target) const {
  const auto key = std::make_pair(source, target);
  if (const auto found = shortest_routes_.find(key); found != shortest_routes_.end()) {
    return found->second;
  }
  if (!has_node(source)) {
    throw std::invalid_argument("flow source does not exist: " + source);
  }
  if (!has_node(target)) {
    throw std::invalid_argument("flow target does not exist: " + target);
  }
  if (source == target) {
    return {{source}, {}, 0};
  }

  const auto infinity = std::numeric_limits<std::uint64_t>::max();
  std::map<std::string, std::uint64_t> distances;
  std::map<std::string, std::vector<std::string>> best_paths;
  std::map<std::string, std::size_t> previous_links;
  for (const auto &node : nodes_) {
    distances[node] = infinity;
  }

  std::priority_queue<RouteState, std::vector<RouteState>, RouteStateGreater>
      queue;
  distances[source] = 0;
  best_paths[source] = {source};
  queue.push({0, {source}, source});

  while (!queue.empty()) {
    auto state = queue.top();
    queue.pop();
    if (state.latency != distances[state.node] ||
        state.path != best_paths[state.node]) {
      continue;
    }
    if (state.node == target) {
      break;
    }

    for (const auto link_index : adjacency_.at(state.node)) {
      const auto &link = links_[link_index];
      if (!link.enabled) continue;
      // Adapter gate edges have zero latency. Reject cycles in tie paths.
      if (std::find(state.path.begin(), state.path.end(), link.target) !=
          state.path.end()) {
        continue;
      }
      if (link.latency_ns > infinity - state.latency) {
        throw std::overflow_error("route latency overflow");
      }
      const auto candidate_latency = state.latency + link.latency_ns;
      auto candidate_path = state.path;
      candidate_path.push_back(link.target);

      const bool is_shorter = candidate_latency < distances[link.target];
      const bool is_stable_tie =
          candidate_latency == distances[link.target] &&
          (best_paths.find(link.target) == best_paths.end() ||
           candidate_path < best_paths[link.target]);
      if (is_shorter || is_stable_tie) {
        distances[link.target] = candidate_latency;
        best_paths[link.target] = candidate_path;
        previous_links[link.target] = link_index;
        queue.push({candidate_latency, candidate_path, link.target});
      }
    }
  }

  if (distances[target] == infinity) {
    throw std::runtime_error("no route from " + source + " to " + target);
  }

  std::vector<std::size_t> reversed_links;
  std::string current = target;
  while (current != source) {
    const auto found = previous_links.find(current);
    if (found == previous_links.end()) {
      throw std::logic_error("failed to reconstruct route");
    }
    reversed_links.push_back(found->second);
    current = links_[found->second].source;
  }
  std::reverse(reversed_links.begin(), reversed_links.end());
  if (shortest_routes_.size() >= 65536) shortest_routes_.clear();
  Route route{best_paths[target], reversed_links, distances[target]};
  shortest_routes_.emplace(key, route);
  return route;
}

Route Topology::ecmp_latency_route(const std::string &source,
                                   const std::string &target,
                                   const std::string &flow_id) const {
  if (flow_id.empty()) {
    throw std::invalid_argument("ECMP flow id must not be empty");
  }
  if (!has_node(source)) {
    throw std::invalid_argument("flow source does not exist: " + source);
  }
  if (!has_node(target)) {
    throw std::invalid_argument("flow target does not exist: " + target);
  }
  if (source == target) {
    return {{source}, {}, 0};
  }

  const auto infinity = std::numeric_limits<std::uint64_t>::max();
  std::map<std::string, std::uint64_t> distances;
  for (const auto &node : nodes_) {
    distances[node] = infinity;
  }
  distances[target] = 0;

  std::priority_queue<DistanceState, std::vector<DistanceState>,
                      DistanceStateGreater>
      queue;
  queue.push({0, target});
  while (!queue.empty()) {
    const auto state = queue.top();
    queue.pop();
    if (state.latency != distances[state.node]) {
      continue;
    }
    for (const auto link_index : reverse_adjacency_.at(state.node)) {
      const auto &link = links_[link_index];
      if (!link.enabled) continue;
      if (link.latency_ns > infinity - state.latency) {
        throw std::overflow_error("route latency overflow");
      }
      const auto candidate = state.latency + link.latency_ns;
      if (candidate >= distances[link.source]) {
        continue;
      }
      distances[link.source] = candidate;
      queue.push({candidate, link.source});
    }
  }

  if (distances[source] == infinity) {
    throw std::runtime_error("no route from " + source + " to " + target);
  }

  std::vector<std::string> nodes{source};
  std::vector<std::size_t> route_links;
  std::set<std::string> visited{source};
  std::string current = source;
  std::uint64_t route_latency = 0;
  std::size_t hop = 0;
  while (current != target) {
    std::vector<std::size_t> candidates;
    for (const auto link_index : adjacency_.at(current)) {
      const auto &link = links_[link_index];
      if (!link.enabled) continue;
      const auto next_distance = distances[link.target];
      if (next_distance == infinity || link.latency_ns > infinity - next_distance ||
          link.latency_ns + next_distance != distances[current] ||
          visited.find(link.target) != visited.end()) {
        continue;
      }
      candidates.push_back(link_index);
    }
    if (candidates.empty()) {
      throw std::logic_error("failed to construct an ECMP route from " + source +
                             " to " + target);
    }

    const auto switch_it = ecmp_switch_seeds_.find(current);
    std::size_t candidate_index = 0;
    if (switch_it != ecmp_switch_seeds_.end()) {
      // This follows ns-3's switch-level ECMP structure: a stable connection
      // key (source, target, flow ID as a source-port surrogate) is mixed with
      // a per-switch seed. Host nodes normally have one uplink and retain the
      // canonical first route if they do not.
      std::uint32_t hash = switch_it->second;
      hash = fnv1a_append(hash, source);
      hash = fnv1a_append(hash, target);
      hash = fnv1a_append(hash, flow_id);
      hash ^= static_cast<std::uint32_t>(hop);
      candidate_index = static_cast<std::size_t>(mix_ecmp_hash(hash) %
                                                 candidates.size());
    }
    const auto selected = candidates[candidate_index];
    const auto &link = links_[selected];
    route_links.push_back(selected);
    nodes.push_back(link.target);
    visited.insert(link.target);
    if (link.latency_ns > infinity - route_latency) {
      throw std::overflow_error("route latency overflow");
    }
    route_latency += link.latency_ns;
    current = link.target;
    ++hop;
  }

  return {std::move(nodes), std::move(route_links), route_latency};
}

const std::set<std::string> &Topology::nodes() const { return nodes_; }

const std::vector<DirectedLink> &Topology::links() const { return links_; }

} // namespace flowsim
