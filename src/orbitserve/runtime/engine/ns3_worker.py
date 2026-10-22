"""Native ns-3 point-to-point serialization with causal RPC boundaries.

OrbitServe chooses hops and accounts for propagation. This adapter deliberately
models packet arbitration, not TCP congestion control or IP route discovery.
"""
import json
import sys
from importlib.metadata import version

NATIVE = r'''
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include <map>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
namespace orbitserve_ns3 {
using namespace ns3;
struct Flow {std::string id,key; double total=0, done=0; bool cancelled=false;};
struct Link {
    Ptr<PointToPointNetDevice> device;
    std::deque<std::string> ready;
    std::string current;
    double rate=0, start=0, duration=0, bytes=0;
    bool busy=false;
    std::vector<std::string> pools;
};
class Engine {
 public:
    Engine() {Time::SetResolution(Time::PS);}
    std::map<std::string,Flow> flows;
    std::map<std::string,std::unique_ptr<Link>> links;
    std::vector<std::string> completed;
    std::map<std::string,double> poolRates;
    std::map<std::string,bool> poolBusy;
    std::string lastServed;
    double now() {return Simulator::Now().GetSeconds();}
    Link& link(std::string key) {
        if (!links.count(key)) {
            auto row=std::make_unique<Link>();
            NodeContainer nodes; nodes.Create(2);
            PointToPointHelper helper;
            helper.SetDeviceAttribute("DataRate",DataRateValue(DataRate(1000000000ULL)));
            helper.SetChannelAttribute("Delay",TimeValue(Seconds(0)));
            auto devices=helper.Install(nodes);
            row->device=DynamicCast<PointToPointNetDevice>(devices.Get(0));
            row->device->TraceConnectWithoutContext("PhyTxEnd",MakeBoundCallback(&Engine::txEnd,this,key));
            links[key]=std::move(row);
        }
        return *links.at(key);
    }
    static void txEnd(Engine* self,std::string key,Ptr<const Packet> packet) {
        auto& row=self->link(key); auto& flow=self->flows.at(row.current);
        if (!flow.cancelled) {
            flow.done+=row.bytes;
            if (flow.done>=flow.total-1e-7) {flow.done=flow.total;self->completed.push_back(flow.id);Simulator::Stop();}
            else row.ready.push_back(flow.id);
        }
        row.busy=false;
        for(auto& pool:row.pools) self->poolBusy[pool]=false;
        // PointToPointNetDevice finishes its own state transition after trace callbacks.
        self->lastServed=key;
        Simulator::ScheduleNow(&Engine::pumpAll,self);
    }
    void addPool(std::string key,double rate){poolRates[key]=rate;poolBusy[key]=false;}
    void attachPool(std::string key,std::string pool){link(key).pools.push_back(pool);}
    void pumpAll() {
        // Rotate the first resource considered to avoid starving another GSL.
        for(auto it=links.upper_bound(lastServed);it!=links.end();++it)pump(it->first);
        for(auto it=links.begin();it!=links.end() && it->first<=lastServed;++it)pump(it->first);
    }
    void pump(std::string key) {
        auto& row=link(key);
        if(row.busy || row.rate<=0)return;
        for(auto& pool:row.pools)if(poolBusy[pool])return;
        while(!row.ready.empty()) {
            std::string id=row.ready.front(); row.ready.pop_front();
            auto& flow=flows.at(id);
            if(flow.cancelled || flow.done>=flow.total)continue;
            double logical=std::min(1400.0,flow.total-flow.done);
            uint32_t payload=static_cast<uint32_t>(std::ceil(logical));
            row.current=id; row.bytes=logical;row.start=now();
            double capacity=row.rate;
            for(auto& pool:row.pools){capacity=std::min(capacity,poolRates[pool]);poolBusy[pool]=true;}
            auto rate=DataRate(static_cast<uint64_t>(std::max(1.0,capacity)));
            row.device->SetDataRate(rate);
            row.duration=rate.CalculateBytesTxTime(payload+2).GetSeconds();
            row.busy=true;
            row.device->Send(Create<Packet>(payload),row.device->GetBroadcast(),0x0800);
            return;
        }
    }
    void configure(std::string key,double rate) {auto& row=link(key);row.rate=rate;pump(key);}
    void start(std::string id,std::string key,double bytes) {
        Flow flow; flow.id=id;flow.key=key;flow.total=bytes;
        flows[id]=flow;link(key).ready.push_back(id);pump(key);
    }
    double progress(std::string id) {
        auto& flow=flows.at(id);auto& row=link(flow.key);
        double amount=flow.done;
        if(row.busy && row.current==id)amount+=row.bytes*std::min(1.0,std::max(0.0,(now()-row.start)/std::max(1e-12,row.duration)));
        return std::min(flow.total,amount);
    }
    void cancel(std::string id){flows.at(id).cancelled=true;}
    void advance(double target) {
        auto ticks=static_cast<int64_t>(std::ceil(target*1e12));
        auto delay=std::max<int64_t>(0,ticks-Simulator::Now().GetTimeStep());
        if(target>now() && delay==0)delay=1;
        auto deadline=Simulator::Schedule(TimeStep(delay),[](){Simulator::Stop();});
        Simulator::Run();Simulator::Remove(deadline);
    }
};
}
'''


def main():
    from ns import ns
    import cppyy
    # Resolve module libraries before compiling the small adapter.
    ns.PointToPointHelper
    cppyy.cppdef(NATIVE)
    engine = cppyy.gbl.orbitserve_ns3.Engine()
    initial = json.loads(sys.stdin.readline())
    for key, rate in initial.get('pools', {}).items():
        engine.addPool(key, rate)
    for row in initial['links']:
        for pool in row.get('pools', ()):
            engine.attachPool(row['key'], pool)
    active = set()

    def emit(value):
        print(json.dumps({'time_s': engine.now(), **value}, allow_nan=False), flush=True)

    emit({'ready': True, 'version': version('ns3')})
    for line in sys.stdin:
        try:
            command = json.loads(line)
            op = command['op']
            if op == 'close':
                ns.Simulator.Destroy()
                emit({'closed': True})
                return
            if op == 'configure':
                for key, rate in command['capacities'].items():
                    engine.configure(key, rate)
                emit({'configured': True})
            elif op == 'start':
                engine.start(command['id'], command['key'], command['bytes'] / 1_000_000)
                active.add(command['id'])
                emit({'started': command['id']})
            elif op == 'cancel':
                for identity in command['ids']:
                    engine.cancel(identity)
                    active.discard(identity)
                emit({'cancelled': command['ids']})
            elif op == 'advance':
                start = engine.now()
                before = {i: engine.progress(i) for i in active}
                engine.advance(float(command['until_s']))
                done = [str(i) for i in engine.completed if str(i) in active]
                engine.completed.clear()
                progress = {i: max(0.0, engine.progress(i) - before[i]) for i in active}
                active.difference_update(done)
                emit({'completed': sorted(done), 'bytes_sent': progress, 'interval_start_s': start})
            else:
                raise ValueError('unknown ns-3 operation')
        except Exception as error:
            emit({'error': f'{type(error).__name__}: {error}'})


if __name__ == '__main__':
    main()
