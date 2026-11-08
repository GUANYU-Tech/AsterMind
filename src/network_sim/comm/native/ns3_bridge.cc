// ns-3 point-to-point packet execution; no replacement packet timing formula.
#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/map-scheduler.h>
#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

#ifndef SERVING_NS3_VERSION
#define SERVING_NS3_VERSION "unrecorded"
#endif

namespace serving {
class ObservableScheduler : public ns3::MapScheduler {
public:
  static ObservableScheduler* current;
  static ns3::TypeId GetTypeId() {
    static ns3::TypeId id = ns3::TypeId("serving::ObservableScheduler")
      .SetParent<ns3::MapScheduler>().AddConstructor<ObservableScheduler>();
    return id;
  }
  ObservableScheduler() { current = this; }
  ~ObservableScheduler() override { current = nullptr; }
};
ObservableScheduler* ObservableScheduler::current = nullptr;
NS_OBJECT_ENSURE_REGISTERED(ObservableScheduler);

struct Flow { uint64_t id, bytes, received = 0, at; std::vector<int> path; bool ended = false; };
struct PacketInfo { uint64_t flow; size_t hop; uint32_t bytes; };
struct Outcome { uint64_t id, at, remaining; bool failed; };
struct Edge { ns3::Ptr<ns3::NetDevice> tx, rx; uint64_t packets = 0; };
static bool active = false;

class Runtime {
public:
  uint32_t packetBytes;
  std::string error;
  std::vector<Edge> edges;
  std::map<uint64_t, Flow> flows;
  std::map<uint64_t, PacketInfo> packets;
  std::deque<Outcome> outcomes;
  explicit Runtime(uint32_t bytes): packetBytes(bytes) {
    if (active) throw std::runtime_error("ns-3 permits one live runtime per process");
    ns3::Time::SetResolution(ns3::Time::NS);
    ns3::ObjectFactory factory;
    factory.SetTypeId(ObservableScheduler::GetTypeId());
    ns3::Simulator::SetScheduler(factory);
    active = true;
  }
  ~Runtime() { ns3::Simulator::Destroy(); active = false; }
  uint64_t Now() { return ns3::Simulator::Now().GetNanoSeconds(); }
  uint64_t Next() {
    auto scheduler = ObservableScheduler::current;
    return !scheduler || scheduler->IsEmpty() ? UINT64_MAX : scheduler->PeekNext().key.m_ts;
  }
  void Boundary(uint64_t at) {
    // Drain every event at this timestamp, including zero-delay children,
    // before returning control to the common serving event loop.
    if (Next() <= at) ns3::Simulator::ScheduleNow(&Runtime::Boundary, this, at);
    else ns3::Simulator::Stop();
  }
  void Advance(uint64_t at) {
    if (at < Now()) throw std::invalid_argument("ns-3 clock cannot move backwards");
    ns3::Simulator::Schedule(ns3::NanoSeconds(at - Now()), &Runtime::Boundary, this, at);
    ns3::Simulator::Run();
  }
  int AddEdge(double gbps, uint64_t latency) {
    ns3::NodeContainer nodes; nodes.Create(2);
    ns3::PointToPointHelper helper;
    helper.SetDeviceAttribute("DataRate", ns3::DataRateValue(ns3::DataRate(uint64_t(gbps * 1e9))));
    helper.SetChannelAttribute("Delay", ns3::TimeValue(ns3::NanoSeconds(latency)));
    helper.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize", ns3::QueueSizeValue(ns3::QueueSize("1000000p")));
    auto devices = helper.Install(nodes);
    devices.Get(1)->SetReceiveCallback(ns3::MakeCallback(&Runtime::Receive, this));
    edges.push_back({devices.Get(0), devices.Get(1)});
    return int(edges.size() - 1);
  }
  void Finish(Flow& flow, bool failed) {
    if (flow.ended) return;
    flow.ended = true;
    outcomes.push_back({flow.id, Now(), flow.bytes - flow.received, failed});
  }
  void Send(ns3::Ptr<ns3::Packet> packet) {
    auto& info = packets.at(packet->GetUid());
    auto& flow = flows.at(info.flow);
    if (flow.ended) { packets.erase(packet->GetUid()); return; }
    auto& edge = edges.at(flow.path.at(info.hop++));
    edge.packets++;
    if (!edge.tx->Send(packet, edge.rx->GetAddress(), 0x0800)) {
      Finish(flow, true);
      packets.erase(packet->GetUid());
    }
  }
  bool Receive(ns3::Ptr<ns3::NetDevice>, ns3::Ptr<const ns3::Packet> packet, uint16_t, const ns3::Address&) {
    auto found = packets.find(packet->GetUid());
    if (found == packets.end()) return true;
    auto info = found->second;
    auto& flow = flows.at(info.flow);
    if (flow.ended) { packets.erase(found); return true; }
    if (info.hop == flow.path.size()) {
      flow.received += info.bytes;
      packets.erase(found);
      if (flow.received == flow.bytes) Finish(flow, false);
    } else Send(packet->Copy());
    return true;
  }
  void Start(uint64_t identifier) {
    auto& flow = flows.at(identifier);
    if (!flow.bytes) { Finish(flow, false); return; }
    for (uint64_t offset = 0; offset < flow.bytes; offset += packetBytes) {
      if (flow.ended) break;
      uint32_t bytes = uint32_t(std::min(uint64_t(packetBytes), flow.bytes - offset));
      auto packet = ns3::Create<ns3::Packet>(bytes);
      packets.emplace(packet->GetUid(), PacketInfo{identifier, 0, bytes});
      Send(packet);
    }
  }
  void Submit(uint64_t identifier, uint64_t bytes, uint64_t at, const int* path, int count) {
    if (at < Now() || count < 1 || flows.count(identifier)) throw std::invalid_argument("invalid ns-3 flow");
    for (int i = 0; i < count; ++i) if (path[i] < 0 || size_t(path[i]) >= edges.size()) throw std::invalid_argument("unknown ns-3 edge");
    flows.emplace(identifier, Flow{identifier, bytes, 0, at, std::vector<int>(path, path + count), false});
    ns3::Simulator::Schedule(ns3::NanoSeconds(at - Now()), &Runtime::Start, this, identifier);
  }
};
template<class F> int guarded(Runtime* runtime, F function) {
  try { runtime->error.clear(); function(); return 0; }
  catch (const std::exception& e) { runtime->error = e.what(); return -1; }
}
} // namespace serving

extern "C" {
const char* n3_version() { return SERVING_NS3_VERSION; }
void* n3_create(uint32_t bytes) { try { return new serving::Runtime(bytes); } catch (...) { return nullptr; } }
void n3_destroy(serving::Runtime* runtime) { delete runtime; }
const char* n3_error(serving::Runtime* runtime) { return runtime->error.c_str(); }
int n3_add_edge(serving::Runtime* runtime, double rate, uint64_t latency) {
  int result = -1; serving::guarded(runtime, [&] { result = runtime->AddEdge(rate, latency); }); return result;
}
int n3_submit(serving::Runtime* runtime, uint64_t id, uint64_t bytes, uint64_t at, const int* path, int count) {
  return serving::guarded(runtime, [&] { runtime->Submit(id, bytes, at, path, count); });
}
int n3_advance(serving::Runtime* runtime, uint64_t at) { return serving::guarded(runtime, [&] { runtime->Advance(at); }); }
uint64_t n3_next(serving::Runtime* runtime) { return runtime->Next(); }
int n3_poll(serving::Runtime* runtime, uint64_t* id, uint64_t* at, uint64_t* remaining, int* failed) {
  if (runtime->outcomes.empty()) return 0;
  auto value = runtime->outcomes.front(); runtime->outcomes.pop_front();
  *id = value.id; *at = value.at; *remaining = value.remaining; *failed = value.failed; return 1;
}
uint64_t n3_packets(serving::Runtime* runtime) { uint64_t result = 0; for (auto& edge: runtime->edges) result += edge.packets; return result; }
}
