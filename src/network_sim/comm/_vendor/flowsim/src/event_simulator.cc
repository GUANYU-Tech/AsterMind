#include "flowsim/flowsim.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace flowsim {

namespace {

constexpr long double kEventEpsilon = 1e-9L;
constexpr long double kByteEpsilon = 1e-6L;

std::uint64_t event_ceil_time_ns(const long double value) {
  if (value <= 0.0L) {
    return 0;
  }
  const auto rounded = std::round(value);
  const auto normalized = std::fabs(value - rounded) <= kEventEpsilon
                              ? rounded
                              : std::ceil(value);
  if (normalized >
      static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
    throw std::overflow_error("simulation time overflow");
  }
  return static_cast<std::uint64_t>(normalized);
}

void event_checked_add(std::uint64_t &value,
                       const std::uint64_t increment) {
  if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
    throw std::overflow_error("link byte counter overflow");
  }
  value += increment;
}

} // namespace

class EventSimulator::Impl {
public:
  explicit Impl(Topology topology)
      : topology_(std::move(topology)),
        link_bytes_(topology_.links().size(), 0),
        link_busy_time_(topology_.links().size(), 0.0L),
        link_peak_flows_(topology_.links().size(), 0),
        link_active_flows_(topology_.links().size(), 0) {}

  void schedule(const long double delay_ns, Callback callback) {
    if (!std::isfinite(delay_ns) || delay_ns < 0.0L) {
      throw std::invalid_argument(
          "event delay must be finite and non-negative");
    }
    if (!callback) {
      throw std::invalid_argument("event callback must not be empty");
    }
    const auto event_time = current_time_ + delay_ns;
    if (!std::isfinite(event_time) || event_time < current_time_) {
      throw std::overflow_error("scheduled event time overflow");
    }
    events_.push({event_time, next_event_sequence_++, std::move(callback)});
    ++progress_revision_;
  }

  void start_flow(const std::string &id, const std::string &source,
                  const std::string &target, const std::uint64_t size_bytes,
                  Callback completion_callback, const bool is_control) {
    if (id.empty()) {
      throw std::invalid_argument("flow id must not be empty");
    }
    if (!flow_ids_.insert(id).second) {
      throw std::invalid_argument("duplicate flow id: " + id);
    }
    if (!completion_callback) {
      throw std::invalid_argument(
          "flow completion callback must not be empty");
    }

    // First-stage adapter policy: deterministic shortest-latency route.
    auto route = topology_.shortest_latency_route(source, target);
    const auto start_time = event_ceil_time_ns(current_time_);
    FlowRequest request{id, source, target, size_bytes, start_time,
                        is_control};
    const auto index = flows_.size();
    flows_.push_back({std::move(request), std::move(route),
                      static_cast<long double>(size_bytes), 0.0L, 0.0L, 0.0L,
                      false, false,
                      std::move(completion_callback)});
    ++progress_revision_;

    if (size_bytes == 0 || flows_[index].route.link_indices.empty()) {
      flows_[index].transmission_end_time = current_time_;
      schedule(0.0L, [this, index]() { complete_delivery(index); });
      return;
    }

    flows_[index].transmitting = true;
    active_.push_back(index);
    rates_dirty_ = true;
  }

  void run() {
    if (running_) {
      throw std::logic_error("event simulator is already running");
    }
    running_ = true;
    try {
      std::size_t stagnant_iterations = 0;
      while (!stop_requested_) {
        const auto previous_time = current_time_;
        const auto previous_progress_revision = progress_revision_;
        recompute_rates_if_needed();

        long double next_time = std::numeric_limits<long double>::infinity();
        if (!events_.empty()) {
          next_time = events_.top().time;
        }
        for (const auto flow_index : active_) {
          const auto &flow = flows_[flow_index];
          if (flow.rate_bytes_per_ns <= kEventEpsilon) {
            throw std::runtime_error("active flow has zero bandwidth");
          }
          auto completion_time =
              current_time_ + flow.remaining_bytes / flow.rate_bytes_per_ns;
          if (completion_time <= current_time_) {
            completion_time = std::nextafter(
                current_time_, std::numeric_limits<long double>::infinity());
          }
          if (!std::isfinite(completion_time)) {
            throw std::overflow_error("flow completion time overflow");
          }
          next_time = std::min(next_time, completion_time);
        }

        if (!std::isfinite(next_time)) {
          break;
        }
        if (next_time + kEventEpsilon < current_time_) {
          throw std::logic_error("simulation time moved backwards");
        }

        advance_to(next_time);
        finish_transmissions();
        process_ready_events();
        if (current_time_ <= previous_time + kEventEpsilon &&
            progress_revision_ == previous_progress_revision) {
          ++stagnant_iterations;
          if (stagnant_iterations > 1000) {
            std::ostringstream message;
            message << "event simulation cannot make numerical progress"
                    << ": time=" << static_cast<double>(current_time_)
                    << ", next=" << static_cast<double>(next_time)
                    << ", active_flows=" << active_.size()
                    << ", pending_events=" << events_.size();
            if (!active_.empty()) {
              const auto &flow = flows_[active_.front()];
              message << ", first_remaining_bytes="
                      << static_cast<double>(flow.remaining_bytes)
                      << ", first_rate_bytes_per_ns="
                      << static_cast<double>(flow.rate_bytes_per_ns);
            }
            throw std::runtime_error(message.str());
          }
        } else {
          stagnant_iterations = 0;
        }
      }
    } catch (...) {
      running_ = false;
      throw;
    }
    running_ = false;
  }

  void stop() { stop_requested_ = true; }

  long double now_ns() const { return current_time_; }

  bool has_pending_work() const {
    return !active_.empty() || !events_.empty();
  }

  SimulationReport report(bool completed_only = false) const {
    for (const auto &flow : flows_) {
      if (!flow.completed && !flow.failed && !completed_only) {
        throw std::logic_error(
            "cannot create a report while a flow is unfinished");
      }
    }

    SimulationReport output{};
    output.end_time_ns = event_ceil_time_ns(current_time_);
    output.flows.reserve(flows_.size());
    for (const auto &flow : flows_) {
      if (!flow.completed) {
        continue;
      }
      const auto transmission_end =
          event_ceil_time_ns(flow.transmission_end_time);
      const auto completion = event_ceil_time_ns(flow.completion_time);
      const auto start = flow.request.start_time_ns;
      const auto fct = completion >= start ? completion - start : 0;
      const double throughput =
          fct == 0 ? 0.0
                   : static_cast<double>(flow.request.size_bytes) * 8.0 /
                         static_cast<double>(fct);
      output.flows.push_back(
          {flow.request.id, flow.request.source, flow.request.target,
           flow.request.size_bytes, start, transmission_end, completion, fct,
           throughput, flow.route.nodes, flow.request.is_control});
    }

    const auto &links = topology_.links();
    output.links.reserve(links.size());
    for (std::size_t index = 0; index < links.size(); ++index) {
      const auto &link = links[index];
      double utilization = 0.0;
      if (output.end_time_ns > 0) {
        utilization = static_cast<double>(
            static_cast<long double>(link_bytes_[index]) /
            (link.capacity_bytes_per_ns *
             static_cast<long double>(output.end_time_ns)));
        utilization = std::min(1.0, std::max(0.0, utilization));
      }
      output.links.push_back(
          {link.source, link.target, link.bandwidth_gbps, link.latency_ns,
           link_bytes_[index], event_ceil_time_ns(link_busy_time_[index]),
           link_peak_flows_[index], utilization});
    }
    return output;
  }

  void advance_until(const long double absolute_time_ns) {
    if (!std::isfinite(absolute_time_ns) || absolute_time_ns < current_time_) {
      throw std::invalid_argument("advance time must be finite and not in the past");
    }
    // Stop at an external clock boundary, preserving all in-flight state.
    schedule(absolute_time_ns - current_time_, [this]() { stop(); });
    stop_requested_ = false;
    run();
    stop_requested_ = false;
  }

  long double next_event_time_ns() {
    recompute_rates_if_needed();
    auto result = events_.empty() ? std::numeric_limits<long double>::infinity()
                                 : events_.top().time;
    for (const auto index : active_) {
      const auto &flow = flows_[index];
      result = std::min(result, current_time_ + flow.remaining_bytes /
                                        flow.rate_bytes_per_ns);
    }
    return result;
  }

  std::vector<FlowProgress> progress() {
    recompute_rates_if_needed();
    std::vector<FlowProgress> output;
    for (const auto &flow : flows_) {
      output.push_back(flow_progress(flow));
    }
    return output;
  }

  std::vector<FlowProgress> take_terminal_events() {
    std::vector<FlowProgress> output;
    output.reserve(terminal_events_.size());
    for (const auto index : terminal_events_) {
      auto progress = flow_progress(flows_[index]);
      if (progress.completed) {
        progress.completion_time_ns = event_ceil_time_ns(progress.completion_time_ns);
      }
      output.push_back(std::move(progress));
    }
    terminal_events_.clear();
    return output;
  }

  // Adapter extension: stop affected flows before allocating the next interval.
  // Even propagation-phase flows fail conservatively; no acknowledgement model.
  void set_link_enabled(const std::string &source, const std::string &target, bool enabled) {
    topology_.set_link_enabled(source, target, enabled);
    if (enabled) return;
    for (std::size_t flow_index = 0; flow_index < flows_.size(); ++flow_index) {
      auto &flow = flows_[flow_index];
      if (flow.completed || flow.failed) continue;
      for (const auto index : flow.route.link_indices) {
        const auto &link = topology_.links()[index];
        if (link.source != source || link.target != target) continue;
        flow.failed = true;
        flow.failure_time = current_time_;
        flow.failure_reason = "link_down";
        flow.transmitting = false;
        flow.rate_bytes_per_ns = 0;
        terminal_events_.push_back(flow_index);
        ++progress_revision_;
        break;
      }
    }
    active_.erase(std::remove_if(active_.begin(), active_.end(),
                                [&](auto index) { return flows_[index].failed; }), active_.end());
    rates_dirty_ = true;
  }

private:
  struct FlowRuntime {
    FlowRequest request;
    Route route;
    long double remaining_bytes;
    long double rate_bytes_per_ns;
    long double transmission_end_time;
    long double completion_time;
    bool transmitting;
    bool completed;
    Callback completion_callback;
    bool failed = false;
    long double failure_time = 0;
    std::string failure_reason;
  };

  static FlowProgress flow_progress(const FlowRuntime &flow) {
    return {flow.request.id, flow.remaining_bytes,
            flow.transmitting ? flow.rate_bytes_per_ns : 0.0L,
            flow.transmitting, flow.completed, flow.transmission_end_time,
            flow.completion_time, flow.failed, flow.failure_time,
            flow.failure_reason, flow.route.nodes};
  }

  struct ScheduledEvent {
    long double time;
    std::uint64_t sequence;
    Callback callback;
  };

  struct EventGreater {
    bool operator()(const ScheduledEvent &left,
                    const ScheduledEvent &right) const {
      if (left.time != right.time) {
        return left.time > right.time;
      }
      return left.sequence > right.sequence;
    }
  };

  void advance_to(const long double next_time) {
    const auto elapsed = std::max(0.0L, next_time - current_time_);
    for (const auto flow_index : active_) {
      auto &flow = flows_[flow_index];
      flow.remaining_bytes -= flow.rate_bytes_per_ns * elapsed;
      if (flow.remaining_bytes < kByteEpsilon) {
        flow.remaining_bytes = 0.0L;
      }
    }
    for (std::size_t index = 0; index < link_active_flows_.size(); ++index) {
      if (link_active_flows_[index] > 0) {
        link_busy_time_[index] += elapsed;
      }
    }
    current_time_ = next_time;
  }

  void finish_transmissions() {
    bool finished_any = false;
    for (const auto flow_index : active_) {
      auto &flow = flows_[flow_index];
      if (flow.remaining_bytes > kByteEpsilon) {
        continue;
      }
      flow.transmitting = false;
      flow.transmission_end_time = current_time_;
      ++progress_revision_;
      for (const auto link_index : flow.route.link_indices) {
        event_checked_add(link_bytes_[link_index], flow.request.size_bytes);
      }
      const auto latency =
          static_cast<long double>(flow.route.propagation_latency_ns);
      schedule(latency,
               [this, flow_index]() { complete_delivery(flow_index); });
      finished_any = true;
    }

    if (finished_any) {
      active_.erase(std::remove_if(active_.begin(), active_.end(),
                                   [&](const auto index) {
                                     return !flows_[index].transmitting;
                                   }),
                    active_.end());
      rates_dirty_ = true;
    }
  }

  void complete_delivery(const std::size_t flow_index) {
    auto &flow = flows_.at(flow_index);
    if (flow.completed || flow.failed) {
      return;
    }
    flow.completed = true;
    flow.completion_time = current_time_;
    terminal_events_.push_back(flow_index);
    ++progress_revision_;
    auto callback = std::move(flow.completion_callback);
    callback();
  }

  void process_ready_events() {
    while (!events_.empty() &&
           events_.top().time <= current_time_ + kEventEpsilon) {
      auto event = events_.top();
      events_.pop();
      ++progress_revision_;
      event.callback();
    }
  }

  void recompute_rates_if_needed() {
    if (!rates_dirty_) {
      return;
    }
    rates_dirty_ = false;
    std::fill(link_active_flows_.begin(), link_active_flows_.end(), 0);
    for (const auto flow_index : active_) {
      flows_[flow_index].rate_bytes_per_ns = 0.0L;
      for (const auto link_index : flows_[flow_index].route.link_indices) {
        ++link_active_flows_[link_index];
      }
    }
    for (std::size_t index = 0; index < link_active_flows_.size(); ++index) {
      link_peak_flows_[index] =
          std::max(link_peak_flows_[index], link_active_flows_[index]);
    }

    const auto &links = topology_.links();
    std::vector<long double> residual_capacity;
    residual_capacity.reserve(links.size());
    for (const auto &link : links) {
      residual_capacity.push_back(link.capacity_bytes_per_ns);
    }
    std::vector<std::size_t> unresolved = active_;

    while (!unresolved.empty()) {
      std::vector<std::size_t> unresolved_per_link(links.size(), 0);
      for (const auto flow_index : unresolved) {
        for (const auto link_index : flows_[flow_index].route.link_indices) {
          ++unresolved_per_link[link_index];
        }
      }

      long double increment = std::numeric_limits<long double>::infinity();
      std::size_t limiting_link = links.size();
      for (std::size_t index = 0; index < links.size(); ++index) {
        if (unresolved_per_link[index] == 0) {
          continue;
        }
        const auto candidate =
            residual_capacity[index] /
            static_cast<long double>(unresolved_per_link[index]);
        if (candidate < increment) {
          increment = candidate;
          limiting_link = index;
        }
      }
      if (!std::isfinite(increment) || increment <= kEventEpsilon ||
          limiting_link == links.size()) {
        throw std::runtime_error(
            "unable to allocate positive bandwidth to active flows");
      }

      for (const auto flow_index : unresolved) {
        flows_[flow_index].rate_bytes_per_ns += increment;
      }
      for (std::size_t index = 0; index < links.size(); ++index) {
        residual_capacity[index] -=
            increment *
            static_cast<long double>(unresolved_per_link[index]);
        if (residual_capacity[index] < kEventEpsilon) {
          residual_capacity[index] = 0.0L;
        }
      }

      std::vector<bool> saturated(links.size(), false);
      for (std::size_t index = 0; index < links.size(); ++index) {
        const auto tolerance = std::max(
            kEventEpsilon, links[index].capacity_bytes_per_ns * kEventEpsilon);
        saturated[index] = unresolved_per_link[index] > 0 &&
                           residual_capacity[index] <= tolerance;
      }
      saturated[limiting_link] = true;

      std::vector<std::size_t> next_unresolved;
      for (const auto flow_index : unresolved) {
        bool limited = false;
        for (const auto link_index : flows_[flow_index].route.link_indices) {
          if (saturated[link_index]) {
            limited = true;
            break;
          }
        }
        if (!limited) {
          next_unresolved.push_back(flow_index);
        }
      }
      if (next_unresolved.size() >= unresolved.size()) {
        throw std::logic_error(
            "max-min allocation did not resolve an active flow");
      }
      unresolved = std::move(next_unresolved);
    }
  }

  Topology topology_;
  std::set<std::string> flow_ids_;
  std::vector<FlowRuntime> flows_;
  std::vector<std::size_t> active_;
  std::vector<std::size_t> terminal_events_;
  std::priority_queue<ScheduledEvent, std::vector<ScheduledEvent>, EventGreater>
      events_;
  std::vector<std::uint64_t> link_bytes_;
  std::vector<long double> link_busy_time_;
  std::vector<std::size_t> link_peak_flows_;
  std::vector<std::size_t> link_active_flows_;
  long double current_time_ = 0.0L;
  std::uint64_t next_event_sequence_ = 0;
  std::uint64_t progress_revision_ = 0;
  bool rates_dirty_ = true;
  bool running_ = false;
  bool stop_requested_ = false;
};

EventSimulator::EventSimulator(Topology topology)
    : impl_(std::make_unique<Impl>(std::move(topology))) {}

EventSimulator::~EventSimulator() = default;
EventSimulator::EventSimulator(EventSimulator &&) noexcept = default;
EventSimulator &EventSimulator::operator=(EventSimulator &&) noexcept = default;

void EventSimulator::schedule(const long double delay_ns, Callback callback) {
  impl_->schedule(delay_ns, std::move(callback));
}

void EventSimulator::start_flow(const std::string &id,
                                const std::string &source,
                                const std::string &target,
                                const std::uint64_t size_bytes,
                                Callback completion_callback,
                                const bool is_control) {
  impl_->start_flow(id, source, target, size_bytes,
                    std::move(completion_callback), is_control);
}

void EventSimulator::run() { impl_->run(); }
void EventSimulator::stop() { impl_->stop(); }
long double EventSimulator::now_ns() const { return impl_->now_ns(); }
bool EventSimulator::has_pending_work() const {
  return impl_->has_pending_work();
}
SimulationReport EventSimulator::report() const { return impl_->report(); }
void EventSimulator::advance_until(long double time) { impl_->advance_until(time); }
long double EventSimulator::next_event_time_ns() { return impl_->next_event_time_ns(); }
std::vector<FlowProgress> EventSimulator::progress() { return impl_->progress(); }
std::vector<FlowProgress> EventSimulator::take_terminal_events() {
  return impl_->take_terminal_events();
}
SimulationReport EventSimulator::completed_report() const { return impl_->report(true); }
void EventSimulator::set_link_enabled(const std::string &source, const std::string &target, bool enabled) {
  impl_->set_link_enabled(source, target, enabled);
}

} // namespace flowsim
